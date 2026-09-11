// Froxel-bake pass of the volumetric fog: one thread per froxel column marches its depth slices
// front-to-back, accumulating single-scattered sun/moonlight plus sky ambient, and stores the integrated
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

// Sky ambient for the in-scatter integral: the baked dome sampled straight up (both bodies' slices), in
// linear BT.709 like every other sky.slang output; the caller converts to the ACEScg transport basis.
float3 fogSkyAmbient(SkyState state) {
    float3 up = float3(0.0, 1.0, 0.0);
    float2 sunUv = skyViewLutUv(state.viewerRadiusKm, 1.0,
            lightViewCosine(up, state.sunDir), SKY_VIEW_BODY_SUN);
    float2 moonUv = skyViewLutUv(state.viewerRadiusKm, 1.0,
            lightViewCosine(up, state.moonDir), SKY_VIEW_BODY_MOON);
    return skyViewLut.SampleLevel(sunUv, 0.0).rgb + skyViewLut.SampleLevel(moonUv, 0.0).rgb;
}

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
    float aniso = clamp(worldPush.fogParams0.y, -0.9, 0.9);
    float maxDist = max(worldPush.fogParams0.z, FOG_NEAR + 1.0);
    float falloff = max(worldPush.fogParams0.w, 1.0e-3);
    float heightBase = worldPush.fogParams1.x;
    float time = worldPush.fogParams1.y;
    float ambientScale = max(worldPush.fogParams1.z, 0.0);
    float noiseAmt = clamp(worldPush.fogParams1.w, 0.0, 1.0);
    float3 albedo = clamp(worldPush.fogParams2.xyz, 0.0, 1.0);
    bool submerged = (worldPush.flags & 1u) != 0u;
    if (submerged) {
        density *= 6.0;
        albedo = clamp(worldPush.waterParams.xyz, 0.0, 1.0);
    }

    SkyState sky = skyState(worldPush);
    float viewerAltBlocks = worldPush.skyLook2.z * 100.0;

    int2 pix = int2(dispatchIndex);
    float2 uv = (float2(pix) + 0.5) / float2(float(fogW), float(fogH));
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
    float3 ambient = bt709ToAcesCg(fogSkyAmbient(sky)) * ambientScale
            + float3(sky.airglowLuminance);

    float sliceDither = fogIgn(dispatchIndex);
    float sliceCount = float(fogD);
    float3 scatter = float3(0.0, 0.0, 0.0);
    float transmit = 1.0;
    float preExposure = worldPush.preExposure;
    for (uint iz = 0u; iz < fogD; ++iz) {
        float s0 = float(iz);
        float z0 = fogSliceViewDepth(s0, sliceCount, maxDist);
        float z1 = fogSliceViewDepth(s0 + 1.0, sliceCount, maxDist);
        float zm = fogSliceViewDepth(s0 + 0.5 + (sliceDither - 0.5) * 0.8, sliceCount, maxDist);
        float3 pos = ro + rd * (zm / cosF);
        float dt = (z1 - z0) / cosF;

        float aboveBase = (pos.y - worldPush.camOffset.y) + viewerAltBlocks - heightBase;
        float heightAtten = exp(-max(aboveBase, 0.0) / falloff);
        float detail = lerp(1.0, fogDensityNoise(pos, time), noiseAmt);
        float sigmaT = density * (0.3 + 0.7 * heightAtten) * detail;
        if (sigmaT * dt > 1.0e-6 && transmit > 0.004) {
            float3 inscatter = ambient;
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
