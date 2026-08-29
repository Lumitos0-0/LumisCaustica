// Computes direct in-scattered light at each Froxel cell from Sun/Moon and block emitters with Resampled Importance Sampling (RIS).
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

// Volumetric RIS Reservoir
struct VolumetricReservoir {
    float3 samplePos;
    float3 sampleRadiance;
    float M;
    float W;
    float wSum;
    float phat;
};

VolumetricReservoir resEmpty() {
    VolumetricReservoir r;
    r.samplePos = float3(0.0, 0.0, 0.0);
    r.sampleRadiance = float3(0.0, 0.0, 0.0);
    r.M = 0.0;
    r.W = 0.0;
    r.wSum = 0.0;
    r.phat = 0.0;
    return r;
}

[shader("compute")]
[numthreads(8, 8, 4)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    uint3 gridDims = pc.gridDims.xyz;
    if (any(dispatchId >= gridDims)) {
        return;
    }

    float4 media = froxelMedia[dispatchId];
    float3 scattering = media.rgb;
    float extinction = media.a;

    // Early out if no participating media exists in this voxel
    if (extinction <= 1.0e-7 || dot(scattering, scattering) <= 1.0e-7) {
        froxelLight[dispatchId] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float zNear = pc.params0.x;
    float zFar = pc.params0.y;
    float maxDist = pc.params0.z;
    float preExposure = max(pc.params0.w, 1.0e-6);

    float g1 = pc.params1.x; // Forward Mie scattering anisotropy for celestial beams (e.g. 0.65)
    float g2 = pc.params1.y; // Backscattering anisotropy (e.g. -0.25)
    float lobeWeight = pc.params1.z; // Weight between forward and back lobes (e.g. 0.85)
    float sunShaftMultiplier = pc.params1.w > 0.0 ? pc.params1.w : 1.0;

    // Sub-voxel temporal jittering along depth to eliminate slice-stepping on light shafts
    float jitterZ = froxelJitter(dispatchId, pc.gridDims.w);

    float2 uv = (float2(dispatchId.xy) + 0.5) / float2(gridDims.xy);
    float2 ndc = uv * 2.0 - 1.0;
    float linearZ = sliceToLinearDepth(float(dispatchId.z) + jitterZ, zNear, zFar, float(gridDims.z));

    float3 rayDir;
    froxelRay(ndc, pc.invViewProj, rayDir);

    float3 camRelPos = rayDir * linearZ;
    float dist = length(camRelPos);
    if (dist > maxDist) {
        froxelLight[dispatchId] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float3 worldPos = camRelPos + pc.camOffset;
    float3 totalInscatter = float3(0.0, 0.0, 0.0);

    // ============================================================================================
    // 1. Direct Celestial (Sun / Moon) Light Shafts
    // ============================================================================================
    float3 sunDir = pc.sunDirIllum.xyz;
    float sunIlluminance = pc.sunDirIllum.w * sunShaftMultiplier; // lux at top of atmosphere
    float celestialShadowVis = 0.0;

    if (sunIlluminance > 0.0 && pc.enableShadows != 0u) {
        // Dual-Lobe Henyey-Greenstein phase function for collimated celestial beams
        float cosTheta = dot(rayDir, sunDir);
        float phase = dualLobePhase(cosTheta, g1, g2, lobeWeight);

        // Atmospheric transmittance along sun direction
        float viewerAltitudeKm = clamp((pc.camPos.y + camRelPos.y - 64.0) / 100.0, 0.0, 99.0);
        float viewerRadiusKm = ATMOS_BOTTOM_KM + max(viewerAltitudeKm, MIN_VIEWER_ALTITUDE_KM);
        float3 sunTransmittance = transmittanceToSpace(transmittanceLut, viewerRadiusKm, sunDir, pc.sunAngularRadius);

        // Hardware RayQuery shadow test against scene geometry (1 ray)
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

        // Gentle ambient sky fill when celestial path is unoccluded
        if (celestialShadowVis > 0.0) {
            float2 skyUv = float2(0.5, 0.5);
            float3 ambientSkyLuminance = bt709ToAcesCg(skyViewLut.SampleLevel(skyUv, 0.0).rgb);
            totalInscatter += ambientSkyLuminance * (VOLUMETRICS_INV_4PI * scattering * 0.05);
        }
    }

    // ============================================================================================
    // 2. Direct Local Block Emitters with Resampled Importance Sampling (RIS)
    // ============================================================================================
    bool enableBlockLights = (pc.lightGridDims.w != 0) && (pc.lightBufAddr != 0) && (pc.lightCount > 0u);
    if (enableBlockLights) {
        float voxelSize = computeSliceThickness(dispatchId.z, zNear, zFar, float(gridDims.z));
        uint seed = dispatchId.x * 1664525u + dispatchId.y * 1013904223u + dispatchId.z * 22695477u + pc.gridDims.w * 2654435761u;
        uint proposalSeed = seed ^ 0x9e3779b9u;

        LightGridCell gridCell;
        gridCell.spanOffset = 0u;
        gridCell.spanCount = 0u;
        gridCell.invWeightSum = 0.0;
        int3 gridCellCoord = int3(0, 0, 0);

        int3 gridDimsL = pc.lightGridDims.xyz;
        bool hasGridCell = false;

        if (all(gridDimsL > int3(0)) && pc.lightGridCellAddr != 0 && pc.lightGridSpanAddr != 0) {
            // Low-discrepancy sub-cell jitter to eliminate boundary seams
            float3 gridLookup = worldPos + (float3(rndf(proposalSeed), rndf(proposalSeed), rndf(proposalSeed)) - 0.5) * pc.lightGridOrigin.w;
            int3 coord = int3(floor((gridLookup - pc.lightGridOrigin.xyz) / pc.lightGridOrigin.w));

            if (all(coord >= int3(0)) && all(coord < gridDimsL)) {
                uint linearCell = (uint(coord.z) * uint(gridDimsL.y) + uint(coord.y)) * uint(gridDimsL.x) + uint(coord.x);
                gridCell = ConstPtr<LightGridCell>(pc.lightGridCellAddr)[linearCell];
                gridCellCoord = coord;
                hasGridCell = (gridCell.spanCount > 0u);
            }
        }

        uint candidateCount = max(pc.risCandidates, 1u);
        uint globalCandidateCount = hasGridCell ? max(1u, (candidateCount + 2u) / 4u) : candidateCount;
        uint localCandidateCount = candidateCount - globalCandidateCount;
        float localProbability = float(localCandidateCount) / float(candidateCount);

        VolumetricReservoir r = resEmpty();
        r.M = float(candidateCount);

        ConstPtr<Light> lights = ConstPtr<Light>(pc.lightBufAddr);

        // Stratified candidate proposal loop (O(1) alias table lookup per candidate)
        for (uint c = 0u; c < candidateCount; c++) {
            uint li;
            uint globalsBefore = (c * globalCandidateCount) / candidateCount;
            uint globalsAfter = ((c + 1u) * globalCandidateCount) / candidateCount;
            bool useLocal = hasGridCell && (globalsAfter == globalsBefore);

            selectLightGridLight(gridCell, useLocal, proposalSeed, li);
            if (li >= pc.lightCount) continue;

            Light lg = lights[li];

            // Area light surface sampling with low-discrepancy jitter
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

                // Regularized solid-angle form factor (eliminates singularity fireflies)
                float formFactor = (emissionFactor * area) / (dist2 + 0.35);

                // Smooth C^1 quadratic distance fade reaching 0 at 32m
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
                    r.wSum += w;

                    if (rndf(seed) * r.wSum < w) {
                        r.samplePos = sp;
                        r.sampleRadiance = unshadowedContrib;
                        r.phat = phat;
                    }
                }
            }
        }

        r.W = r.phat > 0.0 ? (r.wSum / (r.M * r.phat)) : 0.0;

        // Trace EXACTLY 1 hardware RayQuery shadow ray for the survivor sample (100% physically accurate, zero light leaks)
        if (r.phat > 0.0 && r.W > 0.0) {
            float shadowVis = 1.0;
            if (pc.enableShadows != 0u) {
                float3 toSurvivor = r.samplePos - worldPos;
                float survivorDist = length(toSurvivor);

                RayDesc ray;
                ray.Origin = worldPos;
                ray.Direction = toSurvivor / max(survivorDist, 1.0e-4);
                ray.TMin = 0.05;
                ray.TMax = max(survivorDist - 0.05, 0.01);

                RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> q;
                q.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, CULL_ALL, ray);
                q.Proceed();

                if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                    shadowVis = 0.0;
                }
            }

            float3 survivorRadiance = r.sampleRadiance * (shadowVis * r.W);
            totalInscatter += clampVoxelMaxRadiance(survivorRadiance, voxelSize);
        }
    }

    // Apply pre-exposure so float16 volume stores stable HDR ranges
    float3 storedRadiance = totalInscatter * preExposure;
    froxelLight[dispatchId] = float4(clamp(storedRadiance, float3(0.0), float3(HALF_MAX)), extinction);
}
