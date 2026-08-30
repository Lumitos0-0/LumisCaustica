// Temporal filtering of the injected froxel lighting. One raygen thread per fog cell.
//
// The injection is a single-sample estimate per lighting-probe cell (sun ray + emitter next-event +
// radiance probe), so it is noisy; this pass resamples the decimated probe grid back to the fog grid,
// reprojects the previous frame's filtered result into the current camera (same prevViewProj + camDelta
// convention as the reflection motion vectors) and blends, the standard technique for removing ghosting
// without letting noise accumulate. The pair of filtered volumes ping-pongs on frameIndex parity; bit 1
// of froxelGrid.w reports whether the history from the previous frame is valid.
//
// The blend itself happens in a gamma-encoded space (exponent 1/5, the production DDGI/RTXGI probe
// update curve): high-DR single samples — a probe whose ray hit a bright emitter or a sun-facing glint
// — are compressed before they enter the history, so what would be a per-cell firefly becomes a stable
// low-frequency term, and the steady-state value approximates the perceptually averaged light field
// instead of being dominated by the brightest frame. This is deliberate bias: a stable, slightly dimmer
// field is preferred over an unstable one with spikes (the same tradeoff NVIDIA's irradiance probes
// make). The variance clip below still runs in linear space against the same-frame neighborhood, where
// it guards the reprojection itself (camera swings, light changes).
//
// The stored quantity is {sigma_s * L per unit length, scalar extinction}, so color and density are
// filtered together with one weight, keeping the integrate pass's Beer-Lambert slice integral
// consistent with whatever blend was applied.
import world_common;
import world_core;
import bindings;
import math;
import froxel;

public static const float FROXEL_CLIP_SIGMA = 2.0; // variance-clip range in standard deviations
// Perceptual accumulation exponent (store x^(1/gamma), decode x^gamma). 5.0 is the production DDGI
// value: it keeps light changes perceptually linear in time while compressing firefly spikes hard.
public static const float FROXEL_ACCUM_GAMMA = 5.0;
public static const float FROXEL_ACCUM_INV_GAMMA = 1.0 / FROXEL_ACCUM_GAMMA;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    int3 dims = froxelDimensions();
    uint3 tid = DispatchRaysIndex();
    if (tid.x >= uint(dims.x) || tid.y >= uint(dims.y) || tid.z >= uint(dims.z)) {
        return;
    }

    uint writeIdx = worldPush.frameIndex & 1u;
    uint historyIdx = 1u - writeIdx;
    // Injection runs at the decimated probe resolution; resample it back to the full fog grid here. One
    // pass handles the upscale so the firefly-encoding blend, variance clip and integrate all see the
    // same smoothed field.
    uint injectWidth;
    uint injectHeight;
    uint injectDepth;
    froxelScattering.GetDimensions(injectWidth, injectHeight, injectDepth);
    float2 probeScale = float2(dims.xy) / float2(injectWidth, injectHeight);
    float2 probeCoord = (float2(tid.xy) + 0.5) / probeScale;
    float probeZ = (float(tid.z) + 0.5) * float(injectDepth) / float(dims.z);
    int3 probeDims = int3(int(injectWidth), int(injectHeight), int(injectDepth));
    float3 probeCoord3D = float3(probeCoord, probeZ);
    float4 current = froxelTrilinear(froxelScattering, probeDims, probeCoord3D);
    float4 result = current;

    if (froxelEnabled() && froxelHistoryValid()) {
        // Reconstruct this cell's world position and reproject it into the previous frame's camera.
        float3 origin;
        float3 dir;
        froxelRay(float2(tid.xy) + 0.5, origin, dir);
        float z0 = froxelBoundaryDistance(float(tid.z));
        float z1 = froxelBoundaryDistance(float(tid.z + 1));
        float dist = 0.5 * (z0 + z1);
        float3 pos = origin + dir * dist;
        float3 prevCam = worldPush.camOffset - worldPush.camDelta;
        float4 prevClip = mul(worldPush.prevViewProj, float4(pos - prevCam, 1.0));
        bool valid = prevClip.w > 0.0;
        if (valid) {
            float2 prevNdc = prevClip.xy / prevClip.w;
            float3 prevCoord;
            prevCoord.xy = (prevNdc * 0.5 + 0.5) * float2(dims.xy);
            float prevDist = length(pos - prevCam);
            prevCoord.z = froxelSliceForDistance(prevDist);
            // The two filtered lanes are separate bindings (not a descriptor array), so the parity
            // branch is the only way the compiler lets us choose one — which is exactly what we want.
            float4 history;
            if (historyIdx == 0u) {
                history = froxelTrilinear(froxelFilteredA, dims, prevCoord);
            } else {
                history = froxelTrilinear(froxelFilteredB, dims, prevCoord);
            }

            // Variance clipping: bound the history against a local mean ± k·sigma of the current
            // injection, so stale/incorrect history (camera swings, light changes) cannot ghost. The
            // neighborhood is the containing 3x3 probe block, read directly from the decimated
            // scattering so this stays cheap.
            float3 mean = float3(0.0, 0.0, 0.0);
            float3 meanSq = float3(0.0, 0.0, 0.0);
            float count = 0.0;
            int2 probeCell = clamp(int2(floor(probeCoord)), int2(0, 0),
                    int2(int(injectWidth), int(injectHeight)) - 1);
            int probeSlice = clamp(int(floor(probeZ)), 0, int(injectDepth) - 1);
            int3 probeMax = int3(int(injectWidth) - 1, int(injectHeight) - 1, int(injectDepth) - 1);
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    int3 n = clamp(int3(probeCell.x + dx, probeCell.y + dy, probeSlice),
                            int3(0, 0, 0), probeMax);
                    float4 c = froxelScattering[n];
                    mean += c.rgb;
                    meanSq += c.rgb * c.rgb;
                    count += 1.0;
                }
            }
            mean /= count;
            float3 variance = abs(meanSq / count - mean * mean);
            float3 sigma = sqrt(variance);
            float3 lo = mean - FROXEL_CLIP_SIGMA * sigma;
            float3 hi = mean + FROXEL_CLIP_SIGMA * sigma;
            float3 clippedRgb = clamp(history.rgb, lo, hi);
            // Extinction is non-negative and should not bounce around after a settings change, but a
            // stale fog-free history must not make a just-lit region transparent either: clamp against
            // a small multiple of the current value.
            float clippedExtinction = clamp(history.w, 0.0,
                    max(current.w * 2.0, current.w + 0.01));
            float temporalWeight = clamp(worldPush.fogShape.w, 0.0, 0.99);
            // Gamma-encoded blend (see file header): HDR single samples cannot spike the output.
            float3 currentEnc = pow(max(current.rgb, float3(0.0)), float3(FROXEL_ACCUM_INV_GAMMA));
            float3 historyEnc = pow(max(clippedRgb, float3(0.0)), float3(FROXEL_ACCUM_INV_GAMMA));
            float3 blendedEnc = lerp(currentEnc, historyEnc, temporalWeight);
            result = float4(pow(blendedEnc, float3(FROXEL_ACCUM_GAMMA)),
                    lerp(current.w, clippedExtinction, temporalWeight));
        }
    }
    if (writeIdx == 0u) {
        froxelFilteredA[int3(tid)] = result;
    } else {
        froxelFilteredB[int3(tid)] = result;
    }
}
