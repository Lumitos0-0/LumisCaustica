// Low-pass the injected scattering field in screen-froxel space before stochastic raymarching.
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
    for (int y = -1; y <= 1; ++y) {
        float wy = y == 0 ? 4.0 : 1.0;
        for (int x = -1; x <= 1; ++x) {
            float wx = x == 0 ? 4.0 : 1.0;
            int3 samplePosition = clamp(center + int3(x, y, 0), int3(0), dimensions - 1);
            float weight = wx * wy;
            filtered += froxelScattering[samplePosition] * weight;
            weightSum += weight;
        }
    }
    froxelFiltered[center] = filtered / weightSum;
}
