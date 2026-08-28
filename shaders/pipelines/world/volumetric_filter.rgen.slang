// Spatiotemporal resolve of the froxel field.
//
// The inject pass writes one raw, noisy sample per froxel. This pass gathers the current frame's 3x3
// neighbourhood at the froxel's own slice, reprojects the previous frame's accumulated volume into it,
// clips that history to the neighbourhood's own mean and spread, and blends.
//
// The clip is the point of the pass, and it replaces two things at once:
//   - the old fixed ceiling (`current * 4 + 8`), which clipped only from above and was tuned for
//     neither bright emitter regions nor dim ones, and
//   - the disocclusion test, which would otherwise need a previous-frame depth buffer. When the
//     reprojected history describes a different part of the field -- the camera slid onto newly
//     revealed cells, or a light changed -- it lands outside the current neighbourhood's spread and is
//     pulled back to it instead of smearing. Where the field genuinely is noisy the spread is wide and
//     the history passes through untouched, so this costs no responsiveness.
//
// A 3x3 XY support at the froxel's own slice is the correct comparison support here: the reprojected
// history is a point sample landing on that slice, not a slab of it.
//
// This is the A-SVGF structure (Schied et al. 2017, "Spatiotemporal Variance-Guided Filtering") adapted
// to a froxel volume, minus the adaptive history length: that needs per-froxel state, and therefore
// another volume and binding.

import world_common;
import world_core;
import bindings;
import froxel;

// Clip half-width, in neighbourhood standard deviations. 1.5 keeps nearly all of a Gaussian history
// while rejecting a stale sample one and a half spreads out. The nine-tap variance estimate is itself
// noisy, so going tighter starts clipping real detail.
public static const float HISTORY_CLIP_SIGMAS = 1.5;

// Lower bound on the clip band, as a fraction of the neighbourhood mean. Stops the band collapsing to
// nothing when the nine taps happen to agree, which would otherwise throw away a valid history.
public static const float MIN_SPREAD_FRACTION = 0.05;

// Centre weight of the 3x3 kernel against the eight neighbours at 1. Keeps denoising while preserving
// the higher-frequency visibility transitions that define light-shaft edges. Previously this fell with
// camera speed to compensate for a motion-adaptive temporal weight; the variance clip below now keeps
// stale history in check on its own, and a speed-dependent blur cost more in smeared depth than it
// returned in noise.
public static const float KERNEL_CENTER_WEIGHT = 4.0;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint3 index = DispatchRaysIndex();
    uint3 dispatchDimensions = DispatchRaysDimensions();
    if (any(index >= dispatchDimensions)) {
        return;
    }

    int3 dimensions = worldPush.froxelGrid.xyz;
    int3 center = int3(index);

    float4 sum = float4(0.0);
    float4 sumSquares = float4(0.0);
    float weightSum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        float wy = y == 0 ? KERNEL_CENTER_WEIGHT : 1.0;
        for (int x = -1; x <= 1; ++x) {
            float wx = x == 0 ? KERNEL_CENTER_WEIGHT : 1.0;
            int3 samplePosition = clamp(center + int3(x, y, 0), int3(0), dimensions - 1);
            float4 value = max(froxelScattering[samplePosition], float4(0.0));
            float weight = wx * wy;
            sum += value * weight;
            sumSquares += value * value * weight;
            weightSum += weight;
        }
    }
    float4 mean = sum / weightSum;
    // Second moment about the origin minus the squared mean, floored: the subtraction can go slightly
    // negative across a flat neighbourhood in fp32, and the source was rounded through fp16.
    float4 deviation = sqrt(max(sumSquares / weightSum - mean * mean, float4(0.0)));
    // Nine taps estimate the spread noisily, so a neighbourhood that happens to come out uniform would
    // otherwise discard an entirely valid history. The floor is a fraction of the mean and is small
    // enough that a genuine change still falls outside it; it only stops the band collapsing to zero.
    float4 spread = max(deviation * HISTORY_CLIP_SIGMAS, mean * MIN_SPREAD_FRACTION);

    float4 resolved = mean;
    if (froxelHistoryValid()) {
        // Reproject the froxel centre. The inject pass jitters its own sample position to supersample
        // the cell over time; this pass resolves the cell as a whole, so it uses the centre.
        float3 rayOrigin;
        float3 viewRay;
        froxelRay(float2(index.xy) + float2(0.5), rayOrigin, viewRay);
        float3 samplePosition = rayOrigin + viewRay * froxelBoundaryDistance(float(index.z) + 0.5);
        float4 history;
        if (froxelReprojectHistory(samplePosition, history)) {
            float4 clipped = clamp(history, mean - spread, mean + spread);
            resolved = lerp(mean, clipped, worldPush.fogShape.w);
        }
    }
    froxelFiltered[center] = min(max(resolved, float4(0.0)), float4(HALF_MAX));
}
