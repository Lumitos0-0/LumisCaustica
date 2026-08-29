// Computes direct in-scattered light at each Froxel cell from Sun/Moon and block emitters with fast hardware RayQuery shadows.
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

    float g1 = pc.params1.x; // Forward Mie scattering anisotropy (e.g. 0.65)
    float g2 = pc.params1.y; // Backscattering anisotropy (e.g. -0.25)
    float lobeWeight = pc.params1.z; // Weight between forward and back lobes (e.g. 0.85)
    float sunShaftMultiplier = pc.params1.w > 0.0 ? pc.params1.w : 1.0;

    // Sub-voxel temporal jittering along depth to eliminate slice-stepping & pixelation on light shafts
    float jitterZ = froxelJitter(dispatchId, pc.gridDims.w);

    float2 uv = (float2(dispatchId.xy) + 0.5) / float2(gridDims.xy);
    float2 ndc = uv * 2.0 - 1.0;
    float linearZ = sliceToLinearDepth(float(dispatchId.z) + jitterZ, zNear, zFar, float(gridDims.z));

    float3 camOrigin, rayDir;
    froxelRay(ndc, pc.invViewProj, camOrigin, rayDir);

    float3 camRelPos = camOrigin + rayDir * linearZ;
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

    if (sunIlluminance > 0.0) {
        // Dual-Lobe Henyey-Greenstein phase function:
        // cosTheta = dot(rayDir, sunDir) where rayDir points away from camera, sunDir points to sun
        float cosTheta = dot(rayDir, sunDir);
        float phase = dualLobePhase(cosTheta, g1, g2, lobeWeight);

        // Sample atmospheric transmittance along sun direction
        float viewerAltitudeKm = clamp((pc.camPos.y + camRelPos.y - 64.0) / 100.0, 0.0, 99.0);
        float viewerRadiusKm = ATMOS_BOTTOM_KM + max(viewerAltitudeKm, MIN_VIEWER_ALTITUDE_KM);
        float3 sunTransmittance = transmittanceToSpace(transmittanceLut, viewerRadiusKm, sunDir, pc.sunAngularRadius);

        // Shadow ray against Top-Level Acceleration Structure using fast hardware RayQuery
        float shadowVis = 1.0;
        if (pc.enableShadows != 0u) {
            RayDesc ray;
            ray.Origin = worldPos + sunDir * 0.05;
            ray.Direction = sunDir;
            ray.TMin = 0.01;
            ray.TMax = 10000.0;

            RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> q;
            q.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, CULL_SECONDARY, ray);
            q.Proceed();
            if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                shadowVis = 0.0;
            }
        }

        // Convert BT.709 sky lux to ACEScg
        float3 directSunLight = bt709ToAcesCg(sunTransmittance * sunIlluminance * shadowVis);
        totalInscatter += directSunLight * (phase * scattering);
    }

    // ============================================================================================
    // 2. Direct Local Block Emitters (Torches, Lanterns, Glowstone, Lava, Jack-o'-Lanterns)
    // ============================================================================================
    bool enableBlockLights = (pc.lightGridDims.w != 0) && (pc.lightBufAddr != 0) && (pc.lightCount > 0u);
    if (enableBlockLights) {
        float3 blockInscatter = float3(0.0, 0.0, 0.0);
        float voxelSize = computeSliceThickness(dispatchId.z, zNear, zFar, float(gridDims.z));

        // Locate light grid cell containing this voxel
        if (pc.lightGridCellAddr != 0 && pc.lightGridSpanAddr != 0 && all(pc.lightGridDims.xyz > int3(0))) {
            int3 cellCoord = int3(floor((worldPos - pc.lightGridOrigin.xyz) / pc.lightGridOrigin.w));
            
            // Continuous evaluation across grid borders: clamp cell lookup within valid bounds so boundary voxels don't hard-clip
            if (all(cellCoord >= int3(-2)) && all(cellCoord <= pc.lightGridDims.xyz + int3(1))) {
                int3 clampedCoord = clamp(cellCoord, int3(0), pc.lightGridDims.xyz - int3(1));
                uint linearCell = (uint(clampedCoord.z) * uint(pc.lightGridDims.y) + uint(clampedCoord.y))
                                  * uint(pc.lightGridDims.x) + uint(clampedCoord.x);
                LightGridCell cell = ConstPtr<LightGridCell>(pc.lightGridCellAddr)[linearCell];
                
                if (cell.spanCount > 0u) {
                    ConstPtr<LightGridSpan> spans = ConstPtr<LightGridSpan>(pc.lightGridSpanAddr);
                    
                    // Iterate through candidate spans in the cell (up to 16 spans)
                    uint maxSpans = min(cell.spanCount, 16u);
                    float spanWeight = float(cell.spanCount) / float(maxSpans);

                    for (uint s = 0u; s < maxSpans; s++) {
                        LightGridSpan span = spans[cell.spanOffset + s];
                        uint firstLight = span.firstLight;
                        uint count = span.packedLightCounts & 0xffffu;
                        if (count == 0u) continue;

                        uint lightsToTest = min(count, 4u);
                        float lightWeight = float(count) / float(lightsToTest);
                        float totalWeight = spanWeight * lightWeight;

                        for (uint li = 0u; li < lightsToTest; li++) {
                            // Deterministic stride across lights: tests all lights in small blocks (torches, pumpkins)
                            uint strideOffset = (count <= 4u) ? li : (li * count) / lightsToTest;
                            uint lightIndex = min(firstLight + strideOffset, pc.lightCount - 1u);
                            Light lg = ConstPtr<Light>(pc.lightBufAddr)[lightIndex];

                            float3 lightPos = lg.pos + pc.lightRebase.xyz;
                            float3 toL = lightPos - worldPos;
                            float dist2 = dot(toL, toL);

                            if (dist2 > 0.001 && dist2 < 576.0) { // expansive 24 meters reach
                                float lightDist = sqrt(dist2);
                                float3 wi = toL / lightDist;

                                float3 lightNormal = decodeLightNormal(lg);
                                float cosL = dot(lightNormal, -wi);
                                float emissionFactor = max(cosL, 0.35);

                                float area = decodeLightArea(lg);
                                float3 le = decodeLightRadiance(lg.le);

                                // Atmospheric diffusion falloff: smooth room-filling glow fading out to 24m
                                float distFade = max(1.0 - (lightDist / 24.0), 0.0);
                                float smoothDistFade = distFade * distFade;
                                float G = (emissionFactor * area * smoothDistFade) / (lightDist * 0.75 + 0.6);

                                float cosThetaL = dot(rayDir, wi);
                                float phaseL = dualLobePhase(cosThetaL, g1, g2, lobeWeight);

                                float3 sampleContrib = le * (G * phaseL * scattering * totalWeight * 0.7);

                                // Early-out if contribution is imperceptible
                                if (dot(sampleContrib, sampleContrib) > 1.0e-7) {
                                    float shadowVis = 1.0;
                                    if (pc.enableShadows != 0u) {
                                        RayDesc ray;
                                        ray.Origin = worldPos;
                                        ray.Direction = wi;
                                        ray.TMin = 0.05;
                                        ray.TMax = max(lightDist - 0.05, 0.01);

                                        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> q;
                                        q.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, CULL_SECONDARY, ray);
                                        q.Proceed();

                                        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                                            shadowVis = 0.0;
                                        }
                                    }

                                    if (shadowVis > 0.0) {
                                        blockInscatter += clampVoxelMaxRadiance(sampleContrib, voxelSize);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        totalInscatter += blockInscatter;
    }

    // ============================================================================================
    // 3. Ambient Sky Illumination
    // ============================================================================================
    // Gentle diffuse ambient sky light from sky-view LUT for soft overhead fill
    float skyOcclusion = 1.0;
    if (pc.enableShadows != 0u) {
        RayDesc skyRay;
        skyRay.Origin = worldPos + float3(0.0, 0.05, 0.0);
        skyRay.Direction = float3(0.0, 1.0, 0.0);
        skyRay.TMin = 0.01;
        skyRay.TMax = 128.0;

        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> qSky;
        qSky.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, CULL_SECONDARY, skyRay);
        qSky.Proceed();
        if (qSky.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            skyOcclusion = 0.0;
        }
    }

    if (skyOcclusion > 0.0) {
        float2 skyUv = float2(0.5, 0.5);
        float3 ambientSkyLuminance = bt709ToAcesCg(skyViewLut.SampleLevel(skyUv, 0.0).rgb);
        totalInscatter += ambientSkyLuminance * (VOLUMETRICS_INV_4PI * scattering * 0.04 * skyOcclusion);
    }

    // Apply pre-exposure so float16 volume stores stable HDR ranges
    float3 storedRadiance = totalInscatter * preExposure;
    froxelLight[dispatchId] = float4(clamp(storedRadiance, float3(0.0), float3(HALF_MAX)), extinction);
}
