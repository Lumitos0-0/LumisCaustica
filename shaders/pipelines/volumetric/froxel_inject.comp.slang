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

    // Stable froxel center in screen/NDC space (stable world-locked coordinates prevent swimming)
    float2 uv = (float2(dispatchId.xy) + 0.5) / float2(FROXEL_DIMS.xy);
    float2 ndc = uv * 2.0 - 1.0;

    // Unproject near and far plane points in camera-relative space (Reversed-Z: 1=near, 0=far)
    float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    float3 rayDir = normalize(farP - nearP);

    // Exact slice depth for froxel center
    float sliceDepth = froxelSliceToDepth(float(dispatchId.z) + 0.5, FROXEL_DIMS.z);
    float3 hitCamRel = rayDir * sliceDepth;
    float3 hitPos = hitCamRel + worldPush.camOffset;

    // Medium density with smooth exponential height falloff
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
        while (q.Proceed()) {}
        if (q.CommittedStatus() == COMMITTED_NOTHING) {
            float cosTheta = dot(rayDir, lightDir);
            float phase = henyeyGreenstein(cosTheta, clamp(pc.anisotropy, -0.9, 0.95));
            directShaftRadiance = sunIllumAces * (phase * scatteringCoeff * max(pc.directStrength, 0.0));
        }
    }

    // =========================================================================
    // PATH 2: TRUE VOLUMETRIC INDIRECT GI (MCRTX Multi-Directional Ambient Volume)
    // =========================================================================
    const float isotropicPhase = 1.0 / (4.0 * 3.14159265359);
    float3 volumetricGiRadiance = float3(0.0);

    if (scatteringCoeff > 1.0e-7) {
        // Multi-directional spherical probe gather (smooth ambient transition without harsh binary step)
        uint totalProbes = pc.qualityMode == 2u ? 4u : (pc.qualityMode == 1u ? 2u : 1u);
        float3 probeRadianceSum = float3(0.0);

        for (uint s = 0; s < totalProbes; s++) {
            float3 probeDir = sphericalProbeDir(float2(dispatchId.xy), pc.frameIndex, s, totalProbes);
            RayDesc probeRay;
            probeRay.Origin = hitPos;
            probeRay.Direction = probeDir;
            probeRay.TMin = 0.05;
            probeRay.TMax = 32.0;

            RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> probeQ;
            probeQ.TraceRayInline(topLevelAS, RAY_FLAG_NONE, CULL_SECONDARY, probeRay);
            while (probeQ.Proceed()) {}

            if (probeQ.CommittedStatus() == COMMITTED_NOTHING) {
                // Ray escaped to sky: sample sky luminance in probe direction
                float cosView = lightViewCosine(probeDir, sky.sunDir);
                float2 probeSkyUv = skyViewLutUv(sky.viewerRadiusKm, max(probeDir.y, 0.01), cosView, SKY_VIEW_BODY_SUN);
                float3 probeSkyAces = bt709ToAcesCg(skyViewLut.SampleLevel(probeSkyUv, 0.0).rgb);
                probeRadianceSum += probeSkyAces * (isotropicPhase * scatteringCoeff);
            } else {
                // Surface hit: soft ambient bounce reflecting surrounding room
                float hitDist = probeQ.CommittedRayT();
                float proximityWeight = clamp(hitDist / 32.0, 0.05, 1.0);
                float2 zenithUv = skyViewLutUv(sky.viewerRadiusKm, 0.99, lightViewCosine(float3(0.0, 1.0, 0.0), sky.sunDir), SKY_VIEW_BODY_SUN);
                float3 ambientSkyAces = bt709ToAcesCg(skyViewLut.SampleLevel(zenithUv, 0.0).rgb);
                probeRadianceSum += ambientSkyAces * (proximityWeight * 0.2 * isotropicPhase * scatteringCoeff);
            }
        }

        volumetricGiRadiance = (probeRadianceSum / float(totalProbes)) * max(pc.giStrength, 0.0);
    }

    // =========================================================================
    // COMBINE & PRE-EXPOSE (Photometric Domain)
    // =========================================================================
    float3 currentRadiance = (directShaftRadiance + volumetricGiRadiance) * worldPush.preExposure;

    // =========================================================================
    // WORLD-STABLE TEMPORAL REPROJECTION & NEIGHBORHOOD CLAMP
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

    // Soft color box clamping to eliminate ghosting trails while preserving smooth gradients
    float3 minHistBound = currentRadiance * 0.2;
    float3 maxHistBound = currentRadiance * 4.0 + 0.05;
    float3 clampedHistory = clamp(historyRadiance, minHistBound, maxHistBound);

    // Responsive alpha blending (0.82 default weight = 18% new, 82% history)
    float alpha = historyValid ? clamp(1.0 - pc.temporalWeight, 0.08, 1.0) : 1.0;
    float3 filteredRadiance = lerp(clampedHistory, currentRadiance, alpha);

    outFroxelRadiance[dispatchId] = float4(filteredRadiance, 1.0);
    outFroxelExtinction[dispatchId] = float4(float3(extinction), 1.0);
}
