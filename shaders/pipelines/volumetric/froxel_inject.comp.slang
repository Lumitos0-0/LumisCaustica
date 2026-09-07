import bindings;
import world_common;
import world_core;
import volumetric_common;
import sky;

static const uint CULL_SECONDARY = 0x01u;

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

float3 lightRadiance(Light light) {
    uint packed = light.le;
    return float3(unpackUnsignedFloat(packed & 0x7ffu, 6u),
            unpackUnsignedFloat((packed >> 11u) & 0x7ffu, 6u),
            unpackUnsignedFloat((packed >> 22u) & 0x3ffu, 5u));
}

float3 lightHalfU(Light light) {
    return float3(unpackHalf2(light.halfUxy), unpackHalf2(light.halfUzVx).x);
}

float3 lightHalfV(Light light) {
    return float3(unpackHalf2(light.halfUzVx).y, unpackHalf2(light.halfVyz));
}

float3 lightGeometricNormal(Light light) {
    float3 normal = normalize(cross(lightHalfU(light), lightHalfV(light)));
    return (light.section & 0x40000000u) != 0u ? -normal : normal;
}

float lightArea(Light light) {
    return 4.0 * length(cross(lightHalfU(light), lightHalfV(light)));
}

// Generate uniform spherical probe directions for indirect GI ambient field sampling
float3 sphericalProbeDir(float2 pixel, uint frame, uint sampleIdx) {
    float u1 = frac(interleavedGradientNoise(pixel + float(sampleIdx) * 17.13, frame));
    float u2 = frac(interleavedGradientNoise(pixel + float(sampleIdx) * 31.37 + 0.5, frame));
    float z = 1.0 - 2.0 * u1;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * 3.14159265359 * u2;
    return float3(r * cos(phi), z, r * sin(phi));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    if (dispatchId.x >= uint(FROXEL_DIMS.x) || dispatchId.y >= uint(FROXEL_DIMS.y) || dispatchId.z >= uint(FROXEL_DIMS.z)) {
        return;
    }

    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];

    // Sub-voxel temporal blue-noise jittering along view ray to eliminate slicing bands
    float jitter = interleavedGradientNoise(float2(dispatchId.xy), pc.frameIndex) - 0.5;
    float continuousSlice = float(dispatchId.z) + jitter;

    // UV in [0, 1] for current froxel column
    float2 uv = (float2(dispatchId.xy) + 0.5) / float2(FROXEL_DIMS.xy);
    float2 ndc = uv * 2.0 - 1.0;

    // Unproject near and far plane points in camera-relative space (Reversed-Z: 1=near, 0=far)
    float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    float3 rayDir = normalize(farP - nearP);

    // Froxel slice depth with continuous jitter
    float linearDepth = froxelSliceToDepth(continuousSlice, FROXEL_DIMS.z);
    float3 hitCamRel = rayDir * linearDepth;
    float3 hitPos = hitCamRel + worldPush.camOffset;

    // Medium density with exponential height falloff
    float altitude = hitPos.y;
    float heightDiff = max(altitude - pc.fogHeightBaseline, 0.0);
    float heightAtten = exp(-heightDiff / max(pc.fogHeightFalloff, 1.0));
    float baseDensity = max(pc.fogDensity, 0.0) * heightAtten;

    // Single-scattering albedo ~0.9, absorption ~0.1
    float scatteringCoeff = baseDensity * 0.9;
    float absorptionCoeff = baseDensity * 0.1;
    float extinction = scatteringCoeff + absorptionCoeff;

    // =========================================================================
    // PATH 1: DIRECT SUN/MOON SHAFTS (MCRTX High-Frequency Direct Inscattering)
    // =========================================================================
    // Uses hardware RayQuery with RAY_FLAG_FORCE_OPAQUE (no geometry leaks).
    // Evaluates Henyey-Greenstein anisotropic phase (g = 0.80) for forward god rays.
    SkyState sky = skyState(worldPush);
    CelestialLight celLight = dominantCelestialLight(sky, transmittanceLut);
    float3 sunIllumAces = bt709ToAcesCg(celLight.illuminance);
    float3 lightDir = celLight.dir;

    float3 directShaftRadiance = float3(0.0);
    if (scatteringCoeff > 1.0e-7 && dot(sunIllumAces, float3(1.0)) > 1.0e-6) {
        RayDesc ray;
        ray.Origin = hitPos;
        ray.Direction = lightDir;
        ray.TMin = 0.05;
        ray.TMax = 250.0;

        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
        q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, CULL_SECONDARY, ray);
        while (q.Proceed()) {
        }
        if (q.CommittedStatus() == COMMITTED_NOTHING) {
            float cosTheta = dot(rayDir, lightDir);
            float phase = henyeyGreenstein(cosTheta, clamp(pc.anisotropy, -0.9, 0.95));
            directShaftRadiance = sunIllumAces * (phase * scatteringCoeff * max(pc.directStrength, 0.0));
        }
    }

    // =========================================================================
    // PATH 2: TRUE VOLUMETRIC INDIRECT GI (MCRTX Low-Frequency Ambient Volume)
    // =========================================================================
    // Represents multi-scattered diffuse light from the entire surrounding environment.
    // Isotropic phase function: p = 1 / (4 * PI)
    const float isotropicPhase = 1.0 / (4.0 * 3.14159265359);
    float3 volumetricGiRadiance = float3(0.0);

    if (scatteringCoeff > 1.0e-7) {
        // A. Diffuse Sky Dome Irradiance (gated by vertical sky occlusion)
        float skyOcclusion = 1.0;
        {
            RayDesc skyOcclusionRay;
            skyOcclusionRay.Origin = hitPos;
            skyOcclusionRay.Direction = float3(0.0, 1.0, 0.0);
            skyOcclusionRay.TMin = 0.05;
            skyOcclusionRay.TMax = 96.0;

            RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> skyQ;
            skyQ.TraceRayInline(topLevelAS, RAY_FLAG_NONE, CULL_SECONDARY, skyOcclusionRay);
            while (skyQ.Proceed()) {}
            if (skyQ.CommittedStatus() != COMMITTED_NOTHING) {
                skyOcclusion = 0.0; // Fully occluded: inside cave, tunnel, or building
            }
        }

        float2 sunZenithUv = skyViewLutUv(sky.viewerRadiusKm, 0.99, lightViewCosine(float3(0.0, 1.0, 0.0), sky.sunDir), SKY_VIEW_BODY_SUN);
        float2 moonZenithUv = skyViewLutUv(sky.viewerRadiusKm, 0.99, lightViewCosine(float3(0.0, 1.0, 0.0), sky.moonDir), SKY_VIEW_BODY_MOON);
        float3 zenithSkyLuminance709 = skyViewLut.SampleLevel(sunZenithUv, 0.0).rgb + skyViewLut.SampleLevel(moonZenithUv, 0.0).rgb;
        float3 skyLuminanceAces = bt709ToAcesCg(zenithSkyLuminance709);
        float3 skyAmbientGI = skyLuminanceAces * (skyOcclusion * isotropicPhase * scatteringCoeff);

        // B. Spherical Environment / Enclosure Gather (Multi-directional room bounce)
        float3 environmentGI = float3(0.0);
        {
            float3 probeDir = sphericalProbeDir(float2(dispatchId.xy), pc.frameIndex, 0u);
            RayDesc probeRay;
            probeRay.Origin = hitPos;
            probeRay.Direction = probeDir;
            probeRay.TMin = 0.05;
            probeRay.TMax = 16.0;

            RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> envQ;
            envQ.TraceRayInline(topLevelAS, RAY_FLAG_NONE, CULL_SECONDARY, probeRay);
            while (envQ.Proceed()) {}
            if (envQ.CommittedStatus() == COMMITTED_NOTHING) {
                // Ray escaped into open sky: sample sky luminance in probe direction
                if (probeDir.y > 0.0) {
                    float cosView = lightViewCosine(probeDir, sky.sunDir);
                    float2 probeSkyUv = skyViewLutUv(sky.viewerRadiusKm, probeDir.y, cosView, SKY_VIEW_BODY_SUN);
                    float3 probeSkyAces = bt709ToAcesCg(skyViewLut.SampleLevel(probeSkyUv, 0.0).rgb);
                    environmentGI = probeSkyAces * (isotropicPhase * scatteringCoeff * 0.5);
                }
            }
        }

        // C. Continuous Diffuse Radiation Field from Local Emitters (Torches, Lava, Lanterns)
        // Integrates the smooth ambient flux of nearby emitters into the diffuse GI pool
        float3 emitterAmbientGI = float3(0.0);
        if (pc.lightGridCellAddr != 0 && pc.lightBufAddr != 0 && pc.lightGridSpanAddr != 0
                && worldPush.lightCount > 0 && pc.emissiveStrength > 0.0) {
            float3 cellLocal = (hitPos - worldPush.lightGridOrigin.xyz) / max(worldPush.lightGridOrigin.w, 1.0);
            int3 cellCoord = int3(floor(cellLocal));
            if (cellCoord.x >= 0 && cellCoord.y >= 0 && cellCoord.z >= 0
                    && cellCoord.x < worldPush.lightGridDims.x
                    && cellCoord.y < worldPush.lightGridDims.y
                    && cellCoord.z < worldPush.lightGridDims.z) {
                uint linearCell = (uint(cellCoord.z) * uint(worldPush.lightGridDims.y) + uint(cellCoord.y))
                        * uint(worldPush.lightGridDims.x) + uint(cellCoord.x);
                LightGridCell cell = ConstPtr<LightGridCell>(pc.lightGridCellAddr)[linearCell];
                if (cell.spanCount > 0u) {
                    ConstPtr<LightGridSpan> spans = ConstPtr<LightGridSpan>(pc.lightGridSpanAddr);
                    ConstPtr<Light> lights = ConstPtr<Light>(pc.lightBufAddr);
                    uint sampleSpans = min(cell.spanCount, 4u);
                    for (uint s = 0; s < sampleSpans; s++) {
                        LightGridSpan span = spans[cell.spanOffset + s];
                        uint firstLight = span.firstLight;
                        uint count = span.packedLightCounts & 0xffffu;
                        uint maxFetch = min(count, 4u);
                        for (uint l = 0; l < maxFetch; l++) {
                            Light light = lights[firstLight + l];
                            float3 lightPos = light.pos;
                            float3 toLight = lightPos - hitPos;
                            float dist2 = dot(toLight, toLight);
                            const float maxRadius = 24.0;
                            const float maxRadius2 = maxRadius * maxRadius;
                            if (dist2 > 0.04 && dist2 < maxRadius2) {
                                float dist = sqrt(dist2);
                                float3 emitterDir = toLight / dist;
                                float3 emitterNrm = lightGeometricNormal(light);
                                float cosL = max(0.0, dot(emitterNrm, -emitterDir));
                                if (cosL > 0.0) {
                                    float area = lightArea(light);
                                    float3 le = lightRadiance(light);
                                    // Smooth cubic distance attenuation window for diffuse ambient irradiance
                                    float normDist = dist / maxRadius;
                                    float window = max(0.0, 1.0 - normDist * normDist);
                                    float diffuseFlux = (area * cosL * window * window) / (4.0 * 3.14159265359 * (dist2 + 1.0));

                                    // Hardware visibility test against walls and solid blocks
                                    RayDesc localRay;
                                    localRay.Origin = hitPos;
                                    localRay.Direction = emitterDir;
                                    localRay.TMin = 0.05;
                                    localRay.TMax = max(dist - 0.1, 0.05);

                                    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> lq;
                                    lq.TraceRayInline(topLevelAS, RAY_FLAG_NONE, CULL_SECONDARY, localRay);
                                    while (lq.Proceed()) {}
                                    if (lq.CommittedStatus() == COMMITTED_NOTHING) {
                                        // Isotropic diffuse scattered inscattering
                                        emitterAmbientGI += le * (diffuseFlux * isotropicPhase * scatteringCoeff * pc.emissiveStrength);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        volumetricGiRadiance = (skyAmbientGI + environmentGI + emitterAmbientGI) * max(pc.giStrength, 0.0);
    }

    // =========================================================================
    // COMBINE & PRE-EXPOSE (Photometric Domain)
    // =========================================================================
    float3 currentRadiance = (directShaftRadiance + volumetricGiRadiance) * worldPush.preExposure;

    // =========================================================================
    // TEMPORAL REPROJECTION (MCRTX Exponential History Smoothing)
    // =========================================================================
    float3 prevCamRel = hitCamRel + worldPush.camDelta;
    float4 prevClip = mul(worldPush.prevViewProj, float4(prevCamRel, 1.0));
    float3 historyRadiance = currentRadiance;
    bool historyValid = false;

    if (prevClip.w > 0.0) {
        float2 prevNdc = prevClip.xy / prevClip.w;
        float2 prevUv = prevNdc * 0.5 + 0.5;
        float prevLinearDepth = length(prevCamRel);
        float prevSlice = depthToFroxelSlice(prevLinearDepth, FROXEL_DIMS.z);
        float3 prevUVW = float3(prevUv, (prevSlice + 0.5) / float(FROXEL_DIMS.z));

        if (all(prevUVW >= float3(0.0)) && all(prevUVW <= float3(1.0))) {
            historyRadiance = prevFroxelRadiance.SampleLevel(prevUVW, 0.0).rgb;
            historyValid = true;
        }
    }

    float alpha = historyValid ? clamp(1.0 - pc.temporalWeight, 0.01, 1.0) : 1.0;
    float3 filteredRadiance = lerp(historyRadiance, currentRadiance, alpha);

    outFroxelRadiance[dispatchId] = float4(filteredRadiance, 1.0);
    outFroxelExtinction[dispatchId] = float4(float3(extinction), 1.0);
}
