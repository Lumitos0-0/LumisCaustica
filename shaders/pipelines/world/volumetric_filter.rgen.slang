// Low-pass the temporally accumulated scattering field in screen-froxel space before Z integration.
// This filter operates on the continuous volume itself, not scene geometry. Final composition samples the
// 3D field at each pixel's exact depth. A centre-weighted kernel keeps denoising while preserving the
// higher-frequency visibility transitions that define light-shaft edges.

import world_common;
import world_core;
import bindings;

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
    float4 filtered = float4(0.0);
    float weightSum = 0.0;
    // The inject pass sheds temporal weight as the camera moves. Take that slack up here rather than
    // letting it reach the image: the centre weight falls from 4 (parked, sharp) to 1 (full box average
    // at a block per frame), worth ~2.25x more variance reduction from the same nine taps. The ramp is
    // linear in camera travel and much steeper than froxelMotionLoss(), because nine taps are a weak
    // substitute for a ~20-frame history and have to give up sharpness quickly to keep up. Tap count is
    // fixed, so moving never costs bandwidth.
    static const float FULL_BLUR_TRAVEL = 1.0; // blocks per frame
    float centerWeight = lerp(4.0, 1.0, saturate(length(worldPush.camDelta) / FULL_BLUR_TRAVEL));
    for (int y = -1; y <= 1; ++y) {
        float wy = y == 0 ? centerWeight : 1.0;
        for (int x = -1; x <= 1; ++x) {
            float wx = x == 0 ? centerWeight : 1.0;
            int3 samplePosition = clamp(center + int3(x, y, 0), int3(0), dimensions - 1);
            float weight = wx * wy;
            filtered += froxelScattering[samplePosition] * weight;
            weightSum += weight;
        }
    }
    froxelFiltered[center] = filtered / weightSum;
}
