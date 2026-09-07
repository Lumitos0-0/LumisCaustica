import bindings;
import world_common;
import world_core;
import volumetric_common;
import sky;

static const uint CULL_SECONDARY = 0x01u;

// Low-discrepancy spherical probe sampling (Fibonacci / Blue Noise) for stochastic Volumetric GI
float3 sphericalProbeDir(float2 pixel, uint frame, uint sampleIdx, uint totalSamples) {
    float offset = interleavedGradientNoise(pixel, frame);
    float i = float(sampleIdx) + offset;
    float phi = 2.0 * 3.14159265359 * frac(i * 0.61803398875);
    float cosTheta = 1.0 - (2.0 * (float(sampleIdx) + 0.5)) / float(totalSamples);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    return float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
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
    // Multi-scattered diffuse light from the entire surrounding environment.
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

        // B. Multi-Directional Spherical Gather (MCRTX Pure Stochastic Radiative Transfer)
        // Number of probe rays scales with quality: High=1, Very High=2, Ultra=4
        uint totalProbes = pc.qualityMode == 2u ? 4u : (pc.qualityMode == 1u ? 2u : 1u);
        float3 probeRadianceSum = float3(0.0);

        for (uint s = 0; s < totalProbes; s++) {
            float3 probeDir = sphericalProbeDir(float2(dispatchId.xy), pc.frameIndex, s, totalProbes);
            RayDesc probeRay;
            probeRay.Origin = hitPos;
            probeRay.Direction = probeDir;
            probeRay.TMin = 0.05;
            probeRay.TMax = 24.0;

            RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> probeQ;
            probeQ.TraceRayInline(topLevelAS, RAY_FLAG_NONE, CULL_SECONDARY, probeRay);
            while (probeQ.Proceed()) {}

            if (probeQ.CommittedStatus() == COMMITTED_NOTHING) {
                // Ray escaped into open sky: sample sky luminance in probe direction
                if (probeDir.y > 0.0) {
                    float cosView = lightViewCosine(probeDir, sky.sunDir);
                    float2 probeSkyUv = skyViewLutUv(sky.viewerRadiusKm, probeDir.y, cosView, SKY_VIEW_BODY_SUN);
                    float3 probeSkyAces = bt709ToAcesCg(skyViewLut.SampleLevel(probeSkyUv, 0.0).rgb);
                    probeRadianceSum += probeSkyAces * (isotropicPhase * scatteringCoeff);
                }
            } else {
                // Surface hit nearby: ambient room bounce based on surrounding enclosure
                float hitDist = probeQ.RayTCurrent();
                float proximityWeight = clamp(1.0 - (hitDist / 24.0), 0.0, 1.0);
                // Subtle ambient bounce matching the surrounding environment luminance
                probeRadianceSum += skyLuminanceAces * (skyOcclusion * 0.25 * proximityWeight * isotropicPhase * scatteringCoeff);
            }
        }

        float3 environmentGI = probeRadianceSum / float(totalProbes);
        volumetricGiRadiance = (skyAmbientGI + environmentGI) * max(pc.giStrength, 0.0);
    }

    // =========================================================================
    // COMBINE & PRE-EXPOSE (Photometric Domain)
    // =========================================================================
    float3 currentRadiance = (directShaftRadiance + volumetricGiRadiance) * worldPush.preExposure;

    // =========================================================================
    // TEMPORAL REPROJECTION & NEIGHBORHOOD CLAMP (Eliminates Reprojection Lag)
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

    // Dynamic color box clamping to eliminate ghosting trails, camera smearing, and lag
    float3 minHistBound = currentRadiance * 0.2;
    float3 maxHistBound = currentRadiance * 4.0 + 0.05;
    float3 clampedHistory = clamp(historyRadiance, minHistBound, maxHistBound);

    // Responsive alpha blending (0.82 default weight = 18% new, 82% history)
    float alpha = historyValid ? clamp(1.0 - pc.temporalWeight, 0.08, 1.0) : 1.0;
    float3 filteredRadiance = lerp(clampedHistory, currentRadiance, alpha);

    outFroxelRadiance[dispatchId] = float4(filteredRadiance, 1.0);
    outFroxelExtinction[dispatchId] = float4(float3(extinction), 1.0);
}
