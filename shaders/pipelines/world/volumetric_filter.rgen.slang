// Low-pass the temporally accumulated scattering field in screen-froxel space before Z integration.
// A compact depth-aware 3x3 Gaussian removes single-cell Monte Carlo fireflies and sub-cell shadow noise
// without carrying lighting across opaque silhouettes or from an open/deep column into a blocked one.

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
    float centerDepth = froxelOpaqueDistance(float2(center.xy) + 0.5);
    float sliceNear = froxelBoundaryDistance(float(center.z));
    float sliceFar = froxelBoundaryDistance(float(center.z + 1));
    float depthSigma = max((sliceFar - sliceNear) * 2.0, 1.0);
    float4 filtered = float4(0.0);
    float weightSum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        float wy = y == 0 ? 2.0 : 1.0;
        for (int x = -1; x <= 1; ++x) {
            float wx = x == 0 ? 2.0 : 1.0;
            int3 samplePosition = clamp(center + int3(x, y, 0), int3(0), dimensions - 1);
            float sampleDepth = froxelOpaqueDistance(float2(samplePosition.xy) + 0.5);
            // Bilateral depth weighting prevents a nearby wall/ground column from borrowing a lit froxel
            // from an adjacent hole whose first surface is many blocks farther away. Occupancy itself is
            // carried by the injected sample; do not hard-cut the filtered column at this point depth,
            // because silhouette cells intentionally retain their far/background layer.
            float depthWeight = exp(-abs(sampleDepth - centerDepth) / depthSigma);
            float weight = wx * wy * depthWeight;
            filtered += froxelScattering[samplePosition] * weight;
            weightSum += weight;
        }
    }
    froxelFiltered[center] = weightSum > 1.0e-6 ? filtered / weightSum : float4(0.0);
}
