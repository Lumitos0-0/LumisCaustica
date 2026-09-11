// Fog apply pass: temporally accumulates the froxel volume (baked by pipelines/world/fog.rgen) at
// each pixel's scene depth and blends it over the render-res path-traced image, in place, before
// DLSS-RR — so reconstruction, exposure metering and bloom all see the finished signal.
//
// The bake jitters sub-froxel XY + slice position every frame (spatiotemporal blue noise), so each
// frame's volume is noisy; this pass reprojects the previous frame's result through the guide motion
// vectors and blends it with the live bilateral upsample. History is clamped to the live taps' range
// and rejected on disocclusion, so ghosts die in a few frames and new pixels fall back to the
// single-frame result. Ping-ponged history pair, selected by frame parity (uniform branch): the read
// image is never the write image, so no barrier is needed within the dispatch; across frames the
// pre-apply full barrier (same queue, in-order submission) covers the read-after-write.
//
// The volume stores pre-exposed scene-linear scattering, matching outputImage exactly: scattering is
// additive and transmittance is multiplicative, so the display pass divides one residual exposure out of
// their sum and both stay correct.
import world_common;
import bindings;
import fog;

[[vk::push_constant]] PushAddr pcAddr;

// History blend weight per frame, and the disocclusion tolerance in log view-depth (~10%): tighter
// rejects more (safer, noisier in motion), looser keeps history longer.
static const float FOG_FEEDBACK = 0.85;
static const float FOG_REJECT_DEPTH = 0.1;

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
    float2 pixF = float2(pix) + 0.5;
    float2 fogTexel = float2(float(w) / float(max(fogW, 1u)), float(h) / float(max(fogH, 1u)));
    float2 fullRes = float2(float(w), float(h));
    // Joint-bilateral 2x2: each tap fetches the volume at its own pixel's depth and is weighted by
    // view-depth similarity to the center pixel, so fog never bleeds across silhouettes. The taps
    // also bound the history clamp below, so track their component-wise range (unweighted).
    float centerSlice = clamp(fogViewDepthToSlice(centerDepth, 1.0, maxDist), 0.0, 1.0);
    float4 centerFog = fogVolume.SampleLevel(float3(pixF / fullRes, centerSlice), 0.0);
    float4 fogAccum = float4(0.0, 0.0, 0.0, 0.0);
    float wSum = 0.0;
    float4 tapMin = centerFog;
    float4 tapMax = centerFog;
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
        tapMin = min(tapMin, tapFog);
        tapMax = max(tapMax, tapFog);
    }
    // Epsilon-weighted center keeps fully-rejected pixels' own fog instead of collapsing to black.
    float4 current = (fogAccum + centerFog * 0.01) / (wSum + 0.01);

    // Temporal blend: reproject through the jitter-free guide motion (render pixels, current -> previous).
    float2 prevPixF = pixF + motionImage[pix];
    bool inBounds = all(prevPixF >= 0.5) && all(prevPixF <= fullRes - 0.5);
    int2 prevPix = int2(clamp(prevPixF, float2(0.5, 0.5), fullRes - 0.5));
    float prevDepth = projB / max(depthImage[prevPix] + projA, 1.0e-6);
    bool depthOk = abs(log(prevDepth / max(centerDepth, 1.0e-3))) < FOG_REJECT_DEPTH;
    bool useHistory = inBounds && depthOk && push.fogParams2.w > 0.5;
    // Even frames write A (reading B), odd frames write B (reading A); frame 0's garbage is never
    // read because the host clears the history-valid flag until the first write lands.
    bool evenFrame = (push.frameIndex & 1u) == 0u;
    float4 histRaw = evenFrame ? fogHistoryB[prevPix] : fogHistoryA[prevPix];
    float4 histClamped = clamp(histRaw, tapMin, tapMax);
    float4 fogSample = useHistory ? lerp(histClamped, current, 1.0 - FOG_FEEDBACK) : current;
    if (evenFrame) {
        fogHistoryA[pix] = fogSample;
    } else {
        fogHistoryB[pix] = fogSample;
    }
    float4 scene = outputImage[pix];
    outputImage[pix] = float4(scene.rgb * fogSample.a + fogSample.rgb, scene.a);
}
