// Low-pass the temporally accumulated scattering field in screen-froxel space before Z integration.
// This filter operates on the continuous volume itself, not scene geometry. At rest a tight centre-
// weighted 3x3 kernel preserves shafts. While camera motion lowers temporal history, RGB switches to a
// wider 5x5 binomial kernel so one-frame local-light estimates do not appear as large bright/dark puffs.
// Extinction keeps the tight kernel in both modes, preserving density and transmittance detail.

import world_common;
import world_core;
import bindings;

public float motionSpatialWeight(int offset) {
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
    bool cameraMoving = length(worldPush.camDelta) > 0.002 || worldPush.fogLighting.w > 0.001;
    float3 filteredScattering = float3(0.0);
    float scatteringWeightSum = 0.0;
    float filteredExtinction = 0.0;
    float extinctionWeightSum = 0.0;
    for (int y = -2; y <= 2; ++y) {
        if (!cameraMoving && abs(y) > 1) continue;
        float wy = cameraMoving ? motionSpatialWeight(y) : (y == 0 ? 4.0 : 1.0);
        for (int x = -2; x <= 2; ++x) {
            if (!cameraMoving && abs(x) > 1) continue;
            float wx = cameraMoving ? motionSpatialWeight(x) : (x == 0 ? 4.0 : 1.0);
            int3 samplePosition = clamp(center + int3(x, y, 0), int3(0), dimensions - 1);
            float4 sampleValue = froxelScattering[samplePosition];
            float weight = wx * wy;
            filteredScattering += sampleValue.rgb * weight;
            scatteringWeightSum += weight;

            // Density is already coherent in world space. Never widen it just because lighting history is
            // unavailable during motion, or the fog itself would visibly breathe/lose structure.
            if (abs(x) <= 1 && abs(y) <= 1) {
                float densityWeight = (x == 0 ? 4.0 : 1.0) * (y == 0 ? 4.0 : 1.0);
                filteredExtinction += sampleValue.a * densityWeight;
                extinctionWeightSum += densityWeight;
            }
        }
    }
    froxelFiltered[center] = float4(filteredScattering / scatteringWeightSum,
            filteredExtinction / extinctionWeightSum);
}
