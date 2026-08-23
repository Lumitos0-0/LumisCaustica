// Low-pass the current scattering field in screen-froxel space before Z integration. This pass carries
// the denoising burden at depth edges, where temporal history is intentionally rejected to avoid trails.
// A depth-aware 5x5 binomial kernel removes one-ray celestial shadows and local-light RIS variation
// without carrying lighting from an open/deep column into a blocked one.

import world_common;
import world_core;
import bindings;
import froxel;

public float froxelSpatialWeight(int offset) {
    int distance = abs(offset);
    return distance == 0 ? 6.0 : (distance == 1 ? 4.0 : 1.0);
}

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
    for (int y = -2; y <= 2; ++y) {
        float wy = froxelSpatialWeight(y);
        for (int x = -2; x <= 2; ++x) {
            float wx = froxelSpatialWeight(x);
            int3 samplePosition = clamp(center + int3(x, y, 0), int3(0), dimensions - 1);
            float sampleDepth = froxelOpaqueDistance(float2(samplePosition.xy) + 0.5);
            // Occupancy is carried by injection; depth only controls cross-column filtering. Do not
            // hard-cut here because silhouette cells intentionally retain their far/background layer.
            float depthWeight = exp(-abs(sampleDepth - centerDepth) / depthSigma);
            float weight = wx * wy * depthWeight;
            filtered += froxelScattering[samplePosition] * weight;
            weightSum += weight;
        }
    }
    froxelFiltered[center] = weightSum > 1.0e-6 ? filtered / weightSum : float4(0.0);
}
