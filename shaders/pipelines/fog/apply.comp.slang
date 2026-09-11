// Fog apply pass: samples the froxel volume (baked by pipelines/world/fog.rgen) at each pixel's scene
// depth and blends it over the render-res path-traced image, in place, before DLSS-RR — so
// reconstruction, exposure metering and bloom all see the finished (scene + fog) signal.
//
// The volume stores pre-exposed scene-linear scattering, matching outputImage exactly: scattering is
// additive and transmittance is multiplicative, so the display pass divides one residual exposure out of
// their sum and both stay correct.
//
// Upsampling is a joint-bilateral 2x2: each tap fetches the volume at its own pixel's depth and is
// weighted by view-depth similarity to the center pixel, so fog never bleeds across silhouettes.
import world_common;
import bindings;
import fog;

[[vk::push_constant]] PushAddr pcAddr;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    WorldPush push = ConstPtr<WorldPush>(pcAddr.worldPushAddr)[0];
    int2 pix = int2(dispatchId.xy);
    uint w;
    uint h;
    outputImage.GetDimensions(w, h);
    if (pix.x >= int(w) || pix.y >= int(h)) {
        return;
    }
    uint fogW;
    uint fogH;
    uint fogD;
    fogVolume.GetDimensions(fogW, fogH, fogD);
    if (fogD == 0u) {
        return;
    }
    float maxDist = max(push.fogParams0.z, FOG_NEAR + 1.0);
    // Reversed-Z hardware depth (near = 1, far = 0) -> view depth through the projection's third row:
    // depth = (A * viewZ + B) / -viewZ, so -viewZ = B / (depth + A).
    float projA = push.curViewProj[2][2];
    float projB = push.curViewProj[2][3];
    float centerDepth = projB / max(depthImage[pix] + projA, 1.0e-6);
    // Tap the four volume texels surrounding the pixel; each uses its own pixel's depth, so taps on the
    // far side of a silhouette land in a different slice and get down-weighted to ~nothing.
    float2 pixF = float2(pix) + 0.5;
    float2 fogTexel = float2(float(w) / float(max(fogW, 1u)), float(h) / float(max(fogH, 1u)));
    float2 fullRes = float2(float(w), float(h));
    float4 fogAccum = float4(0.0, 0.0, 0.0, 0.0);
    float wSum = 0.0;
    for (int i = 0; i < 4; ++i) {
        float2 tapPixF = clamp(pixF + (float2(float(i & 1), float(i >> 1)) - 0.5) * fogTexel,
                float2(0.5, 0.5), fullRes - 0.5);
        float tapDepth = projB / max(depthImage[int2(tapPixF)] + projA, 1.0e-6);
        float tapSlice = clamp(fogViewDepthToSlice(tapDepth, 1.0, maxDist), 0.0, 1.0);
        float4 tapFog = fogVolume.SampleLevel(float3(tapPixF / fullRes, tapSlice), 0.0);
        // Symmetric tolerance in log view-depth: ~2% off keeps ~full weight, ~10% off keeps ~20%.
        float dr = log(tapDepth / max(centerDepth, 1.0e-3));
        float weight = exp(-dr * dr * 150.0);
        fogAccum += tapFog * weight;
        wSum += weight;
    }
    // Fallback center tap: sub-pixel geometry (distant leaves, wires) can depth-reject all four
    // taps, collapsing the blend to black; the epsilon-weighted center keeps those pixels' own fog.
    float centerSlice = clamp(fogViewDepthToSlice(centerDepth, 1.0, maxDist), 0.0, 1.0);
    float4 centerFog = fogVolume.SampleLevel(float3(pixF / fullRes, centerSlice), 0.0);
    float4 fogSample = (fogAccum + centerFog * 0.01) / (wSum + 0.01);
    float4 scene = outputImage[pix];
    outputImage[pix] = float4(scene.rgb * fogSample.a + fogSample.rgb, scene.a);
}
