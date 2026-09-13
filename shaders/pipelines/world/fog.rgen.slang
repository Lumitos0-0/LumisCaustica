// Froxel-bake pass of the volumetric fog: one thread per froxel column marches its depth slices
// front-to-back, accumulating single-scattered sun/moonlight, and stores the integrated
// scattering + transmittance of every slice. pipelines/fog/apply.comp samples the volume at each pixel's
// scene depth and blends it over the path-traced image.
//
// Occlusion comes from TLAS shadow rays (visibility(), the same query surface NEE uses), so fog shadows
// match surface shadows exactly — including stained-glass tinting and cutout alpha — with no shadow maps.
// The bake runs unjittered at 1/8 render resolution; the per-column slice dither is a static hash, keeping
// the volume temporally stable with or without DLSS-RR.
import world_common;
import world_core;
import bindings;
import math;
import sky;
import trace;
import fog;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 dispatchIndex = DispatchRaysIndex().xy;
    uint fogW;
    uint fogH;
    uint fogD;
    fogVolume.GetDimensions(fogW, fogH, fogD);
    if (dispatchIndex.x >= fogW || dispatchIndex.y >= fogH || fogD == 0u) {
        return;
    }

    float density = worldPush.fogParams0.x;
    float maxDist = max(worldPush.fogParams0.z, FOG_NEAR + 1.0);
    float sliceMode = worldPush.fogParams3.x;
    float sliceExp = max(worldPush.fogParams3.y, 1.0);
    float waterAniso = worldPush.fogParams3.z;
    float falloff = max(worldPush.fogParams0.w, 1.0e-3);
    float heightBase = worldPush.fogParams1.x;
    float time = worldPush.fogParams1.y;
    float noiseAmt = clamp(worldPush.fogParams1.w, 0.0, 1.0);
    float3 albedo = clamp(worldPush.fogParams2.xyz, 0.0, 1.0);
    bool submerged = (worldPush.flags & 1u) != 0u;
    if (submerged) {
        density *= 6.0;
        albedo = clamp(worldPush.waterParams.xyz, 0.0, 1.0);
    }
    // Per-medium phase: air and water haze scatter differently, so each has its own anisotropy.
    float aniso = clamp(submerged ? waterAniso : worldPush.fogParams0.y, -0.9, 0.9);

    SkyState sky = skyState(worldPush);
    float viewerAltBlocks = worldPush.skyLook2.z * 100.0;

    // Spatiotemporal jitter (see RtStbn): the column's sub-froxel XY offset and in-slice sample
    // position vary every frame; the apply pass integrates the sequence over time while single
    // frames keep blue-noise spatial character. W is reserved for phase-2 ray jitter.
    uint stbnW;
    uint stbnH;
    uint stbnD;
    stbnTex.GetDimensions(stbnW, stbnH, stbnD);
    float2 stbnXY = float2(float(stbnW), float(stbnH));
    float stbnZ = float(max(stbnD, 1u));
    float4 stbn = stbnTex.SampleLevel(
            float3((float2(dispatchIndex) + 0.5) / stbnXY, (float(worldPush.frameIndex % max(stbnD, 1u)) + 0.5) / stbnZ),
            0.0);
    int2 pix = int2(dispatchIndex);
    float2 uv = (float2(pix) + 0.5 + (stbn.xy - 0.5)) / float2(float(fogW), float(fogH));
    float2 ndc = uv * 2.0 - 1.0;
    float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    float3 ro = nearP + worldPush.camOffset;
    float3 rd = normalize(farP - nearP);

    // Slices are spaced in view depth (what the apply pass recovers from the depth buffer), while
    // extinction integrates over Euclidean length; the ray's cosine against the view axis converts.
    float4 axisNearH = mul(worldPush.invViewProj, float4(0.0, 0.0, 1.0, 1.0));
    float4 axisFarH = mul(worldPush.invViewProj, float4(0.0, 0.0, 0.0, 1.0));
    float3 axisNear = axisNearH.xyz / axisNearH.w;
    float3 axisFar = axisFarH.xyz / axisFarH.w;
    float3 fwd = normalize(axisFar - axisNear);
    float cosF = max(dot(rd, fwd), 1.0e-3);

    // Both bodies' post-atmosphere illuminance, from the same transmittance LUT that tints the visible
    // discs and drives surface NEE — fog, sky and terrain can never disagree about the light. A body more
    // than ~50x dimmer than the other (or below the horizon) contributes nothing visible, so its shadow
    // ray is skipped for the whole column.
    float3 sunE = sky.sunIlluminance
            * transmittanceToSpace(transmittanceLut, sky.viewerRadiusKm, sky.sunDir, sky.sunAngularRadius);
    float3 moonE = sky.moonIlluminance
            * transmittanceToSpace(transmittanceLut, sky.viewerRadiusKm, sky.moonDir, sky.moonAngularRadius);
    float sunLum = dot(sunE, float3(0.2126, 0.7152, 0.0722));
    float moonLum = dot(moonE, float3(0.2126, 0.7152, 0.0722));
    float lightMax = max(max(sunLum, moonLum), 1.0e-6);
    bool useSun = sunLum >= lightMax * 0.02;
    bool useMoon = moonLum >= lightMax * 0.02;
    float3 sunAces = bt709ToAcesCg(sunE);
    float3 moonAces = bt709ToAcesCg(moonE);
    float phaseSun = hg(clamp(dot(rd, sky.sunDir), -1.0, 1.0), aniso);
    float phaseMoon = hg(clamp(dot(rd, sky.moonDir), -1.0, 1.0), aniso);

    float sliceCount = float(fogD);
    float3 scatter = float3(0.0, 0.0, 0.0);
    float transmit = 1.0;
    float preExposure = worldPush.preExposure;
    for (uint iz = 0u; iz < fogD; ++iz) {
        float s0 = float(iz);
        float z0 = fogSliceViewDepth(s0, sliceCount, maxDist, sliceMode, sliceExp);
        float z1 = fogSliceViewDepth(s0 + 1.0, sliceCount, maxDist, sliceMode, sliceExp);
        float zm = fogSliceViewDepth(s0 + 0.5 + (stbn.z - 0.5) * 0.9, sliceCount, maxDist, sliceMode, sliceExp);
        float3 pos = ro + rd * (zm / cosF);
        float dt = (z1 - z0) / cosF;

        float aboveBase = (pos.y - worldPush.camOffset.y) + viewerAltBlocks - heightBase;
        float heightAtten = exp(-max(aboveBase, 0.0) / falloff);
        // Detail noise fades in past the near field so nearby air stays smooth and only distant fog
        // carries visible structure (which is also where the coarse volume hides it best).
        float detail = lerp(1.0, fogDensityNoise(pos, time), noiseAmt * smoothstep(4.0, 12.0, zm));
        float sigmaT = density * (0.3 + 0.7 * heightAtten) * detail;
        if (sigmaT * dt > 1.0e-6 && transmit > 0.004) {
            // No ambient or airglow fill: unlit fog adds no light (it only attenuates), so caves and
            // night air go fully dark and occlusion reads exactly.
            float3 inscatter = float3(0.0, 0.0, 0.0);
            if (useSun) {
                inscatter += sunAces * visibility(pos, sky.sunDir, 10000.0).transmittance * phaseSun;
            }
            if (useMoon) {
                inscatter += moonAces * visibility(pos, sky.moonDir, 10000.0).transmittance * phaseMoon;
            }
            scatter += transmit * (sigmaT * albedo) * inscatter * dt;
            transmit *= exp(-sigmaT * dt);
        }
        fogVolume[int3(pix.x, pix.y, int(iz))] = float4(scatter * preExposure, transmit);
    }
}
