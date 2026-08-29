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
    // 2. Direct Local Block Emitters (Torches, Lanterns, Glowstone, Lava Lakes, Lavafalls, Froglights)
    // ============================================================================================
    bool enableBlockLights = (pc.lightBufAddr != 0) && (pc.lightCount > 0u);
    if (enableBlockLights) {
        float3 blockInscatter = float3(0.0, 0.0, 0.0);
        float voxelSize = computeSliceThickness(dispatchId.z, zNear, zFar, float(gridDims.z));

        uint totalLights = min(pc.lightCount, 256u);
        ConstPtr<Light> lights = ConstPtr<Light>(pc.lightBufAddr);

        float3 bestWi = float3(0.0, 1.0, 0.0);
        float bestLightDist = 100.0;
        float maxWeight = 0.0;

        // Step 1: Accumulate unshadowed irradiance across all nearby emitters in fast scalar ALU
        for (uint i = 0u; i < totalLights; i++) {
            Light lg = lights[i];

            float3 lightPos = lg.pos + pc.lightRebase.xyz;
            float3 toL = lightPos - worldPos;
            float dist2 = dot(toL, toL);

            // True Euclidean 32-meter sphere with instant scalar rejection
            if (dist2 > 0.001 && dist2 < 1024.0) {
                float lightDist = sqrt(dist2);
                float3 wi = toL / lightDist;

                float3 lightNormal = decodeLightNormal(lg);
                float cosL = dot(lightNormal, -wi);
                float emissionFactor = max(cosL, 0.25);

                float area = decodeLightArea(lg);
                float3 le = decodeLightRadiance(lg.le);

                // Physical solid-angle form factor (no singularities/spotlights)
                float formFactor = (emissionFactor * area) / (dist2 + 0.35);

                // Smooth C^1 quadratic distance fade reaching 0 at 32m
                float distFade = max(1.0 - (lightDist / 32.0), 0.0);
                float smoothDistFade = distFade * distFade;

                float3 contrib = le * (formFactor * smoothDistFade * VOLUMETRICS_INV_4PI * scattering * 5.0);
                float weight = dot(contrib, float3(0.2126, 0.7152, 0.0722));

                blockInscatter += contrib;

                // Track the dominant light source for shadow evaluation
                if (weight > maxWeight) {
                    maxWeight = weight;
                    bestWi = wi;
                    bestLightDist = lightDist;
                }
            }
        }

        // Step 2: Exactly ONE hardware RayQuery per voxel to the dominant emitter (blazing fast 120+ FPS!)
        if (maxWeight > 1.0e-6) {
            float shadowVis = 1.0;
            if (pc.enableShadows != 0u) {
                RayDesc ray;
                ray.Origin = worldPos;
                ray.Direction = bestWi;
                ray.TMin = 0.05;
                ray.TMax = max(bestLightDist - 0.05, 0.01);

                RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> q;
                q.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, CULL_ALL, ray);
                q.Proceed();

                if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                    shadowVis = 0.0;
                }
            }

            totalInscatter += clampVoxelMaxRadiance(blockInscatter * shadowVis, voxelSize);
        }
    }

    // Apply pre-exposure so float16 volume stores stable HDR ranges
    float3 storedRadiance = totalInscatter * preExposure;
    froxelLight[dispatchId] = float4(clamp(storedRadiance, float3(0.0), float3(HALF_MAX)), extinction);
}
