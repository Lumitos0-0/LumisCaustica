// Depth-aware bilateral filter for the temporally accumulated scattering field.
// Uses geometry depth awareness from volumeDepth to preserve razor-sharp light shaft edges and
// prevent fog blobbing across geometric boundaries (windows, pillars, walls).

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
    float4 filtered = float4(0.0);
    float weightSum = 0.0;

    for (int y = -1; y <= 1; ++y) {
        float wy = y == 0 ? 4.0 : 1.0;
        for (int x = -1; x <= 1; ++x) {
            float wx = x == 0 ? 4.0 : 1.0;
            int3 samplePosition = clamp(center + int3(x, y, 0), int3(0), dimensions - 1);
            float sampleDepth = froxelOpaqueDistance(float2(samplePosition.xy) + 0.5);
            
            // Bilateral depth-stopping weight: prevents open-air fog/shafts from bleeding across geometry edges
            float depthDelta = abs(centerDepth - sampleDepth);
            float depthWeight = exp(-depthDelta * 0.35);

            float weight = wx * wy * depthWeight;
            filtered += froxelScattering[samplePosition] * weight;
            weightSum += weight;
        }
    }
    froxelFiltered[center] = filtered / max(weightSum, 1.0e-5);
}
