import bindings;
import world_common;
import world_core;
import volumetric_common;
import sky;

static const uint CULL_SECONDARY = 0x01u;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    if (dispatchId.x >= uint(FROXEL_DIMS.x) || dispatchId.y >= uint(FROXEL_DIMS.y) || dispatchId.z >= uint(FROXEL_DIMS.z)) {
        return;
    }

    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];

    // UV in [0, 1] for current froxel column
    float2 uv = (float2(dispatchId.xy) + 0.5) / float2(FROXEL_DIMS.xy);
    float2 ndc = uv * 2.0 - 1.0;

    // Unproject near and far plane points in camera-relative space (Reversed-Z: 1=near, 0=far)
    float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    float3 rayDir = normalize(farP - nearP);

    // Froxel slice depth
    float linearDepth = froxelSliceToDepth(float(dispatchId.z), FROXEL_DIMS.z);
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

    // Direct sun/moon directional light and shadow ray query
    SkyState sky = skyState(worldPush);
    CelestialLight celLight = dominantCelestialLight(sky, transmittanceLut);
    float3 sunIllumAces = bt709ToAcesCg(celLight.illuminance);
    float3 lightDir = celLight.dir;

    float3 sunRadiance = float3(0.0);
    if (scatteringCoeff > 1.0e-7 && dot(sunIllumAces, float3(1.0)) > 1.0e-6) {
        RayDesc ray;
        ray.Origin = hitPos;
        ray.Direction = lightDir;
        ray.TMin = 0.05;
        ray.TMax = 250.0;

        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> q;
        q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, CULL_SECONDARY, ray);
        while (q.Proceed()) {
        }
        if (q.CommittedStatus() == COMMITTED_NOTHING) {
            float cosTheta = dot(rayDir, lightDir);
            float phase = henyeyGreenstein(cosTheta, clamp(pc.anisotropy, -0.9, 0.95));
            sunRadiance = sunIllumAces * (phase * scatteringCoeff * max(pc.directStrength, 0.0));
        }
    }

    // Sky dome / indirect volumetric inscatter sampled from Hillaire Sky-View LUT
    float2 sunSkyUv = skyViewLutUv(sky.viewerRadiusKm, rayDir.y, lightViewCosine(rayDir, sky.sunDir), SKY_VIEW_BODY_SUN);
    float2 moonSkyUv = skyViewLutUv(sky.viewerRadiusKm, rayDir.y, lightViewCosine(rayDir, sky.moonDir), SKY_VIEW_BODY_MOON);
    float3 skyLuminance709 = skyViewLut.SampleLevel(sunSkyUv, 0.0).rgb + skyViewLut.SampleLevel(moonSkyUv, 0.0).rgb;
    float3 skyLuminanceAces = bt709ToAcesCg(skyLuminance709);
    float isotropicPhase = 1.0 / (4.0 * 3.14159265359);
    float3 skyInscatter = skyLuminanceAces * (isotropicPhase * scatteringCoeff * max(pc.giStrength, 0.0));

    // Pre-expose current sample radiance into stored photometric domain
    float3 currentRadiance = (sunRadiance + skyInscatter) * worldPush.preExposure;

    // Temporal Reprojection across camera movement
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

    float alpha = historyValid ? clamp(1.0 - pc.temporalWeight, 0.02, 1.0) : 1.0;
    float3 filteredRadiance = lerp(historyRadiance, currentRadiance, alpha);

    outFroxelRadiance[dispatchId] = float4(filteredRadiance, 1.0);
    outFroxelExtinction[dispatchId] = float4(float3(extinction), 1.0);
}
