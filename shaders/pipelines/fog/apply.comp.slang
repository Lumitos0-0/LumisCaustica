// Fog apply pass: spatially upsamples the accumulation pass's RESOLVED volume at each pixel's
// scene depth and blends it over the render-res path-traced image, in place, before DLSS-RR —
// so reconstruction, exposure metering and bloom all see the finished signal.
//
// All temporal work happens in accumulate.comp (3D world-space reprojection into the A/B volume
// pair); this pass is purely spatial, so it can never smear, ghost, or drag history. The slot to
// read is selected by frame parity, mirroring the accumulate write side (even frames resolved
// into A, odd into B). The pass also snapshots the hardware depth into prevDepth for next
// frame's accumulate occlusion check.
//
// The volume stores pre-exposed scene-linear scattering, matching outputImage exactly: scattering is
// additive and transmittance is multiplicative, so the display pass divides one residual exposure out of
// their sum and both stay correct.
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
    volARw.GetDimensions(fogW, fogH, fogD);
    if (fogD == 0u) {
        return;
    }
    // Read slot mirrors the accumulate write side: even frames resolved into A, odd into B.
    bool evenFrame = (push.frameIndex & 1u) == 0u;
    float maxDist = max(push.fogParams0.z, FOG_NEAR + 1.0);
    // Reversed-Z hardware depth (near = 1, far = 0) -> view depth through the projection's third row:
    // depth = (A * viewZ + B) / -viewZ, so -viewZ = B / (depth + A).
    float projA = push.curViewProj[2][2];
    float projB = push.curViewProj[2][3];
    float centerDepth = projB / max(depthImage[pix] + projA, 1.0e-6);
    float2 pixF = float2(pix) + 0.5;
    float2 fogTexel = float2(float(w) / float(max(fogW, 1u)), float(h) / float(max(fogH, 1u)));
    float2 fullRes = float2(float(w), float(h));
    // Joint-bilateral 2x2: each tap fetches the resolved volume at its own pixel's depth and is
    // weighted by view-depth similarity to the center pixel, so fog never bleeds across silhouettes.
    float centerSlice = clamp(fogViewDepthToSlice(centerDepth, 1.0, maxDist), 0.0, 1.0);
    float3 centerUvz = float3(pixF / fullRes, centerSlice);
    float4 centerFog = evenFrame ? volA.SampleLevel(centerUvz, 0.0) : volB.SampleLevel(centerUvz, 0.0);
    float4 fogAccum = float4(0.0, 0.0, 0.0, 0.0);
    float wSum = 0.0;
    for (int i = 0; i < 4; ++i) {
        float2 tapPixF = clamp(pixF + (float2(float(i & 1), float(i >> 1)) - 0.5) * fogTexel,
                float2(0.5, 0.5), fullRes - 0.5);
        float tapDepth = projB / max(depthImage[int2(tapPixF)] + projA, 1.0e-6);
        float tapSlice = clamp(fogViewDepthToSlice(tapDepth, 1.0, maxDist), 0.0, 1.0);
        float3 tapUvz = float3(tapPixF / fullRes, tapSlice);
        float4 tapFog = evenFrame ? volA.SampleLevel(tapUvz, 0.0) : volB.SampleLevel(tapUvz, 0.0);
        // Symmetric tolerance in log view-depth: ~2% off keeps ~full weight, ~10% off keeps ~20%.
        float dr = log(tapDepth / max(centerDepth, 1.0e-3));
        float weight = exp(-dr * dr * 150.0);
        fogAccum += tapFog * weight;
        wSum += weight;
    }
    // Epsilon-weighted center keeps fully-rejected pixels' own fog instead of collapsing to black.
    float4 current = (fogAccum + centerFog * 0.01) / (wSum + 0.01);

    // Snapshot the hardware depth for next frame's accumulate occlusion check, then composite.
    prevDepthRw[pix] = depthImage[pix];
    float4 scene = outputImage[pix];
    outputImage[pix] = float4(scene.rgb * current.a + current.rgb, scene.a);
}
