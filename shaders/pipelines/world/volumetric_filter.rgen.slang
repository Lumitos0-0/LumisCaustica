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

    // 3D spatial filter: 3x3 binomial kernel in XY plus cross-slice smoothing in Z.
    // Denoises local-emitter variance across froxel cells and softens inter-slice transitions
    // without degrading directional light-shaft boundaries.
    for (int y = -1; y <= 1; ++y) {
        float wy = y == 0 ? 4.0 : 1.0;
        for (int x = -1; x <= 1; ++x) {
            float wx = x == 0 ? 4.0 : 1.0;
            int3 samplePosition = clamp(center + int3(x, y, 0), int3(0), dimensions - 1);
            float weight = wx * wy * 4.0;
            filtered += froxelScattering[samplePosition] * weight;
            weightSum += weight;
        }
    }

    // Cross-slice taps in Z
    if (center.z > 0) {
        int3 samplePrevZ = center + int3(0, 0, -1);
        filtered += froxelScattering[samplePrevZ] * 1.5;
        weightSum += 1.5;
    }
    if (center.z + 1 < dimensions.z) {
        int3 sampleNextZ = center + int3(0, 0, 1);
        filtered += froxelScattering[sampleNextZ] * 1.5;
        weightSum += 1.5;
    }

    froxelFiltered[center] = filtered / weightSum;
}
