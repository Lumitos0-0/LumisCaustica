// Computes direct in-scattered light at each Froxel cell using 3D Spatial Reservoir Exchange (ReSTIR) across adjacent froxels.
import froxel_common;
import world_common;
import sky;

[[vk::push_constant]] FroxelLightPush pc;

[[vk::binding(0, 0)]] public RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] public Texture3D<float4> froxelMedia;
[[vk::binding(2, 0)]] [format("rgba16f")] public RWTexture3D<float4> froxelLight;
[[vk::binding(3, 0)]] public Sampler2D skyViewLut;
[[vk::binding(4, 0)]] public Sampler2D transmittanceLut;

// Helper to decode R11G11B10 packed radiance from Light record
float unpackUnsignedFloat(uint bits, uint mantissaBits) {
    uint mantissaMask = (1u << mantissaBits) - 1u;
    uint mantissa = bits & mantissaMask;
    uint exponent = (bits >> mantissaBits) & 31u;
    if (exponent == 0u) {
        return float(mantissa) * exp2(float(1 - 15 - int(mantissaBits)));
    }
    return (1.0 + float(mantissa) / float(1u << mantissaBits))
            * exp2(float(int(exponent) - 15));
}

float3 decodeLightRadiance(uint packed) {
    return float3(unpackUnsignedFloat(packed & 0x7ffu, 6u),
                  unpackUnsignedFloat((packed >> 11u) & 0x7ffu, 6u),
                  unpackUnsignedFloat((packed >> 22u) & 0x3ffu, 5u));
}

float3 decodeLightHalfU(Light light) {
    return float3(unpackHalf2(light.halfUxy), unpackHalf2(light.halfUzVx).x);
}

float3 decodeLightHalfV(Light light) {
    return float3(unpackHalf2(light.halfUzVx).y, unpackHalf2(light.halfVyz));
}

float decodeLightArea(Light light) {
    return 4.0 * length(cross(decodeLightHalfU(light), decodeLightHalfV(light)));
}

float3 decodeLightNormal(Light light) {
    float3 n = normalize(cross(decodeLightHalfU(light), decodeLightHalfV(light)));
    return (light.section & 0x40000000u) != 0u ? -n : n;
}

int3 decodeLightSectionCoord(Light light) {
    uint packed = light.section;
    return int3(int(packed & 0x3ffu), int((packed >> 10u) & 0x3ffu), int((packed >> 20u) & 0x3ffu));
}

// Low-discrepancy pseudo-random number generator
float rndf(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) * (1.0 / 4294967296.0);
}

// Select one light from the global emitted-power distribution in O(1)
void selectGlobalLight(inout uint proposalSeed, out uint lightIndex) {
    float aliasSample = rndf(proposalSeed) * float(pc.lightCount);
    uint column = min(uint(aliasSample), pc.lightCount - 1u);
    if (pc.lightAliasAddr != 0) {
        LightAlias a = ConstPtr<LightAlias>(pc.lightAliasAddr)[column];
        bool self = aliasSample - float(column) < a.accept;
        lightIndex = self ? column : a.aliasIndex;
    } else {
        lightIndex = column;
    }
}

// Select one light from a chunk section's local power alias table in O(1)
void selectSectionLight(uint firstLight, uint lightCount, inout uint proposalSeed, out uint lightIndex) {
    float aliasSample = rndf(proposalSeed) * float(lightCount);
    uint column = min(uint(aliasSample), lightCount - 1u);
    LightAlias alias = ConstPtr<LightAlias>(pc.lightLocalAliasAddr)[firstLight + column];
    bool self = aliasSample - float(column) < alias.accept;
    uint localIndex = self ? column : alias.aliasIndex;
    lightIndex = firstLight + localIndex;
}

// Select a light through a cell's section span alias table in O(1)
void selectLightGridSpanLight(LightGridCell cell, inout uint proposalSeed, out uint lightIndex) {
    ConstPtr<LightGridSpan> spans = ConstPtr<LightGridSpan>(pc.lightGridSpanAddr);
    float aliasSample = rndf(proposalSeed) * float(cell.spanCount);
    uint column = min(uint(aliasSample), cell.spanCount - 1u);
    LightGridSpan span = spans[cell.spanOffset + column];
    bool self = aliasSample - float(column) < span.accept;
    uint firstLight = self ? span.firstLight : span.aliasFirstLight;
    uint lightCount = self ? (span.packedLightCounts & 0xffffu) : (span.packedLightCounts >> 16u);
    selectSectionLight(firstLight, lightCount, proposalSeed, lightIndex);
}

// Sample the exact proposal mixture q = alpha*qCell + (1-alpha)*qGlobal
void selectLightGridLight(LightGridCell cell, bool useLocal, inout uint proposalSeed, out uint lightIndex) {
    if (useLocal) {
        selectLightGridSpanLight(cell, proposalSeed, lightIndex);
    } else {
        selectGlobalLight(proposalSeed, lightIndex);
    }
}

// Reconstruct proposal probability density function
float proposalPdf(Light light, float3 le, LightGridCell cell, int3 cellCoord, float localProbability) {
    float power = max(0.0, decodeLightArea(light) * dot(le, float3(0.2126, 0.7152, 0.0722)));
    float globalPdf = pc.lightAliasAddr != 0 ? power * pc.lightRebase.w : 1.0 / float(pc.lightCount);
    float localPdf = 0.0;
    if (localProbability > 0.0 && power > 0.0) {
        int3 delta = decodeLightSectionCoord(light) - cellCoord;
        if (all(abs(delta) <= int3(2, 2, 2))) {
            float3 deltaF = float3(delta);
            localPdf = power * cell.invWeightSum / max(1.0, dot(deltaF, deltaF));
        }
    }
    return localProbability * localPdf + (1.0 - localProbability) * globalPdf;
}

// Volumetric ReSTIR Reservoir
struct VolumetricReservoir {
    uint lightIndex;
    float3 samplePos;
    float3 sampleRadiance;
    float M;
    float W;
    float wSum;
    float phat;
};

VolumetricReservoir resEmpty() {
    VolumetricReservoir r;
    r.lightIndex = 0u;
    r.samplePos = float3(0.0, 0.0, 0.0);
    r.sampleRadiance = float3(0.0, 0.0, 0.0);
    r.M = 0.0;
    r.W = 0.0;
    r.wSum = 0.0;
    r.phat = 0.0;
    return r;
}

// LDS Groupshared memory for 3D spatial reservoir exchange across adjacent froxels
groupshared VolumetricReservoir s_sharedReservoirs[8][8][4];

[shader("compute")]
[numthreads(8, 8, 4)]
void main(uint3 dispatchId : SV_DispatchThreadID, uint3 groupThreadId : SV_GroupThreadID) {
    uint3 gridDims = pc.gridDims.xyz;
    bool isValidVoxel = all(dispatchId < gridDims);

    float4 media = isValidVoxel ? froxelMedia[dispatchId] : float4(0.0);
    float3 scattering = media.rgb;
    float extinction = media.a;

    float zNear = pc.params0.x;
    float zFar = pc.params0.y;
    float maxDist = pc.params0.z;
    float preExposure = max(pc.params0.w, 1.0e-6);

    float g1 = pc.params1.x; // Forward Mie scattering anisotropy for celestial beams (e.g. 0.65)
    float g2 = pc.params1.y; // Backscattering anisotropy (e.g. -0.25)
    float lobeWeight = pc.params1.z; // Weight between forward and back lobes (e.g. 0.85)
    float sunShaftMultiplier = pc.params1.w > 0.0 ? pc.params1.w : 1.0;

    // Stable voxel center depth to eliminate distance fluctuation noise on point emissives
    float linearZ = sliceToLinearDepth(float(dispatchId.z) + 0.5, zNear, zFar, float(gridDims.z));

    float3 rayDir;
    froxelRay(ndc, pc.invViewProj, rayDir);

    float3 camRelPos = rayDir * linearZ;
    float dist = length(camRelPos);
    float3 worldPos = camRelPos + pc.camOffset;

    bool hasMedia = isValidVoxel && (extinction > 1.0e-7) && (dot(scattering, scattering) > 1.0e-7) && (dist <= maxDist);

    // ============================================================================================
    // Step 1: Initial RIS Candidate Proposal
    // ============================================================================================
    VolumetricReservoir r_init = resEmpty();
    uint seed = dispatchId.x * 1664525u + dispatchId.y * 1013904223u + dispatchId.z * 22695477u + pc.gridDims.w * 2654435761u;
    uint proposalSeed = seed ^ 0x9e3779b9u;

    bool enableBlockLights = hasMedia && (pc.lightGridDims.w != 0) && (pc.lightBufAddr != 0) && (pc.lightCount > 0u);
    ConstPtr<Light> lights = ConstPtr<Light>(pc.lightBufAddr);

    if (enableBlockLights) {
        LightGridCell gridCell;
        gridCell.spanOffset = 0u;
        gridCell.spanCount = 0u;
        gridCell.invWeightSum = 0.0;
        int3 gridCellCoord = int3(0, 0, 0);

        int3 gridDimsL = pc.lightGridDims.xyz;
        bool hasGridCell = false;

        if (all(gridDimsL > int3(0)) && pc.lightGridCellAddr != 0 && pc.lightGridSpanAddr != 0) {
            float3 gridLookup = worldPos;
            int3 coord = int3(floor((gridLookup - pc.lightGridOrigin.xyz) / pc.lightGridOrigin.w));

            if (all(coord >= int3(0)) && all(coord < gridDimsL)) {
                uint linearCell = (uint(coord.z) * uint(gridDimsL.y) + uint(coord.y)) * uint(gridDimsL.x) + uint(coord.x);
                gridCell = ConstPtr<LightGridCell>(pc.lightGridCellAddr)[linearCell];
                gridCellCoord = coord;
                hasGridCell = (gridCell.spanCount > 0u);
            }
        }

        uint candidateCount = max(pc.risCandidates, 8u);
        uint globalCandidateCount = hasGridCell ? max(1u, (candidateCount + 2u) / 4u) : candidateCount;
        uint localCandidateCount = candidateCount - globalCandidateCount;
        float localProbability = float(localCandidateCount) / float(candidateCount);

        r_init.M = float(candidateCount);

        for (uint c = 0u; c < candidateCount; c++) {
            uint li;
            uint globalsBefore = (c * globalCandidateCount) / candidateCount;
            uint globalsAfter = ((c + 1u) * globalCandidateCount) / candidateCount;
            bool useLocal = hasGridCell && (globalsAfter == globalsBefore);

            selectLightGridLight(gridCell, useLocal, proposalSeed, li);
            if (li >= pc.lightCount) continue;

            Light lg = lights[li];

            float s = rndf(seed) * 2.0 - 1.0;
            float t = rndf(seed) * 2.0 - 1.0;
            float3 sp = lg.pos + pc.lightRebase.xyz + s * decodeLightHalfU(lg) + t * decodeLightHalfV(lg);

            float3 toL = sp - worldPos;
            float dist2 = dot(toL, toL);

            if (dist2 > 0.001 && dist2 < 1024.0) {
                float lightDist = sqrt(dist2);
                float3 wi = toL / lightDist;

                float3 lightNormal = decodeLightNormal(lg);
                float cosL = dot(lightNormal, -wi);
                float emissionFactor = max(cosL, 0.25);

                float area = decodeLightArea(lg);
                float3 le = decodeLightRadiance(lg.le);

                // Smooth regularized form factor (eliminates singularity fireflies)
                float formFactor = (emissionFactor * area) / (dist2 + 0.35);

                // Continuous C^1 quadratic distance fade reaching 0 at 32m
                float distFade = max(1.0 - (lightDist / 32.0), 0.0);
                float smoothDistFade = distFade * distFade;

                // Volumetric phase function
                float cosTheta = dot(rayDir, wi);
                float phase = dualLobePhase(cosTheta, g1 * 0.5, g2 * 0.5, 0.5);

                float3 unshadowedContrib = le * (formFactor * smoothDistFade * phase * scattering * 5.0);
                float phat = dot(unshadowedContrib, float3(0.2126, 0.7152, 0.0722));

                if (phat > 0.0) {
                    float sourcePdf = proposalPdf(lg, le, gridCell, gridCellCoord, localProbability);
                    float w = phat / max(sourcePdf, 1.0e-20);
                    r_init.wSum += w;

                    if (rndf(seed) * r_init.wSum < w) {
                        r_init.lightIndex = li;
                        r_init.samplePos = sp;
                        r_init.sampleRadiance = unshadowedContrib;
                        r_init.phat = phat;
                    }
                }
            }
        }

        r_init.W = r_init.phat > 0.0 ? (r_init.wSum / (r_init.M * r_init.phat)) : 0.0;
    }

    // Write initial reservoir to LDS shared memory for 3D spatial exchange
    s_sharedReservoirs[groupThreadId.x][groupThreadId.y][groupThreadId.z] = r_init;

    // Synchronize workgroup to ensure all 256 local reservoirs are visible
    GroupMemoryBarrierWithGroupSync();

    // ============================================================================================
    // Step 2: 3D Spatial Reservoir Exchange Across Adjacent Froxels
    // ============================================================================================
    VolumetricReservoir r_spatial = r_init;

    if (enableBlockLights) {
        // Gather 6 orthogonal 3D spatial neighbors within the shared workgroup
        const int3 neighborOffsets[6] = {
            int3(-1,  0,  0),
            int3( 1,  0,  0),
            int3( 0, -1,  0),
            int3( 0,  1,  0),
            int3( 0,  0, -1),
            int3( 0,  0,  1)
        };

        for (int i = 0; i < 6; i++) {
            int3 nCoord = int3(groupThreadId) + neighborOffsets[i];
            if (all(nCoord >= int3(0)) && all(nCoord < int3(8, 8, 4))) {
                VolumetricReservoir neighbor = s_sharedReservoirs[nCoord.x][nCoord.y][nCoord.z];

                if (neighbor.phat > 0.0 && neighbor.W > 0.0 && neighbor.M > 0.0 && neighbor.lightIndex < pc.lightCount) {
                    // Re-evaluate neighbor's candidate at current voxel's position
                    float3 toL = neighbor.samplePos - worldPos;
                    float dist2 = dot(toL, toL);

                    if (dist2 > 0.001 && dist2 < 1024.0) {
                        float lightDist = sqrt(dist2);
                        float3 wi = toL / lightDist;

                        Light lg = lights[neighbor.lightIndex];
                        float3 lightNormal = decodeLightNormal(lg);
                        float cosL = dot(lightNormal, -wi);
                        float emissionFactor = max(cosL, 0.25);

                        float area = decodeLightArea(lg);
                        float3 le = decodeLightRadiance(lg.le);

                        float formFactor = (emissionFactor * area) / (dist2 + 0.35);
                        float distFade = max(1.0 - (lightDist / 32.0), 0.0);
                        float smoothDistFade = distFade * distFade;

                        float cosTheta = dot(rayDir, wi);
                        float phase = dualLobePhase(cosTheta, g1 * 0.5, g2 * 0.5, 0.5);

                        float3 currContrib = le * (formFactor * smoothDistFade * phase * scattering * 5.0);
                        float currPhat = dot(currContrib, float3(0.2126, 0.7152, 0.0722));

                        if (currPhat > 0.0) {
                            // Spatial ReSTIR merge weight
                            float w_n = currPhat * neighbor.W * neighbor.M;
                            r_spatial.wSum += w_n;
                            r_spatial.M += neighbor.M;

                            if (rndf(seed) * r_spatial.wSum < w_n) {
                                r_spatial.lightIndex = neighbor.lightIndex;
                                r_spatial.samplePos = neighbor.samplePos;
                                r_spatial.sampleRadiance = currContrib;
                                r_spatial.phat = currPhat;
                            }
                        }
                    }
                }
            }
        }

        // Clamp spatial M to avoid over-accumulation
        r_spatial.M = min(r_spatial.M, 64.0);
        r_spatial.W = r_spatial.phat > 0.0 ? (r_spatial.wSum / (r_spatial.M * r_spatial.phat)) : 0.0;
    }

    if (!isValidVoxel) {
        return;
    }

    if (!hasMedia) {
        froxelLight[dispatchId] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float3 totalInscatter = float3(0.0, 0.0, 0.0);

    // ============================================================================================
    // Step 3: Direct Celestial (Sun / Moon) Light Shafts
    // ============================================================================================
    float3 sunDir = pc.sunDirIllum.xyz;
    float sunIlluminance = pc.sunDirIllum.w * sunShaftMultiplier; // lux at top of atmosphere
    float celestialShadowVis = 0.0;

    if (sunIlluminance > 0.0 && pc.enableShadows != 0u) {
        float cosTheta = dot(rayDir, sunDir);
        float phase = dualLobePhase(cosTheta, g1, g2, lobeWeight);

        float viewerAltitudeKm = clamp((pc.camPos.y + camRelPos.y - 64.0) / 100.0, 0.0, 99.0);
        float viewerRadiusKm = ATMOS_BOTTOM_KM + max(viewerAltitudeKm, MIN_VIEWER_ALTITUDE_KM);
        float3 sunTransmittance = transmittanceToSpace(transmittanceLut, viewerRadiusKm, sunDir, pc.sunAngularRadius);

        celestialShadowVis = 1.0;
        RayDesc ray;
        ray.Origin = worldPos + sunDir * 0.1;
        ray.Direction = sunDir;
        ray.TMin = 0.05;
        ray.TMax = 500.0;

        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> q;
        q.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, CULL_ALL, ray);
        q.Proceed();
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            celestialShadowVis = 0.0;
        }

        float3 directSunLight = bt709ToAcesCg(sunTransmittance * sunIlluminance * celestialShadowVis);
        totalInscatter += directSunLight * (phase * scattering);

        if (celestialShadowVis > 0.0) {
            float2 skyUv = float2(0.5, 0.5);
            float3 ambientSkyLuminance = bt709ToAcesCg(skyViewLut.SampleLevel(skyUv, 0.0).rgb);
            totalInscatter += ambientSkyLuminance * (VOLUMETRICS_INV_4PI * scattering * 0.05);
        }
    }

    // ============================================================================================
    // Step 4: Shading & Exactly ONE Hardware RayQuery Shadow Ray for Spatial Survivor
    // ============================================================================================
    if (enableBlockLights && (r_spatial.phat > 0.0) && (r_spatial.W > 0.0)) {
        float voxelSize = computeSliceThickness(dispatchId.z, zNear, zFar, float(gridDims.z));
        float shadowVis = 1.0;

        if (pc.enableShadows != 0u) {
            float3 toSurvivor = r_spatial.samplePos - worldPos;
            float survivorDist = length(toSurvivor);

            RayDesc ray;
            ray.Origin = worldPos;
            ray.Direction = toSurvivor / max(survivorDist, 1.0e-4);
            ray.TMin = 0.10;
            ray.TMax = max(survivorDist - 0.10, 0.01);

            RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> q;
            q.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, CULL_ALL, ray);
            q.Proceed();

            if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                shadowVis = 0.0;
            }
        }

        float3 survivorRadiance = r_spatial.sampleRadiance * (shadowVis * r_spatial.W);
        totalInscatter += clampVoxelMaxRadiance(survivorRadiance, voxelSize);
    }

    // Apply pre-exposure so float16 volume stores stable HDR ranges
    float3 storedRadiance = totalInscatter * preExposure;
    froxelLight[dispatchId] = float4(clamp(storedRadiance, float3(0.0), float3(HALF_MAX)), extinction);
}
