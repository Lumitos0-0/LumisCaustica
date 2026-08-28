// Low-pass the temporally accumulated scattering field in screen-froxel space before Z integration.
// This filter operates on the continuous volume itself, not scene geometry. Final composition samples the
// 3D field at each pixel's exact depth. A centre-weighted kernel keeps denoising while preserving the
// higher-frequency visibility transitions that define light-shaft edges.

import world_common;
import world_core;
import bindings;
import froxel;

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
    // letting it reach the image: the centre weight falls from 4 (parked, sharp) to 1 (moving, full box
    // average), which is worth about 2.25x more variance reduction from the same nine taps. The tap
    // count is fixed, so moving never costs bandwidth.
    float centerWeight = lerp(4.0, 1.0, froxelMotionLoss());
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
