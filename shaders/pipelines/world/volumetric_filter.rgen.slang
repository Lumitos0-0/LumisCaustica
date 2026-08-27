// Stage 2: deterministic 3x3x3 spatial reconstruction guided by the explicit froxel depth volume.
// Extinction is no longer used as an edge classifier: doing so outlined Minecraft block boundaries when
// world-space fBm changed rapidly. First-interface depth instead prevents light crossing actual geometry,
// while a smooth tolerance avoids turning block silhouettes into hard filtering seams.

import world_common;
import world_core;
import bindings;
import froxel;

[shader("raygeneration")]
void main() {
    uint3 index = DispatchRaysIndex().xyz;
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    int3 dimensions = froxelDimensions();
    if (!froxelEnabled() || any(index >= uint3(dimensions))) {
        return;
    }

    int3 centerCoordinate = int3(index);
    float4 center = froxelScattering[centerCoordinate];
    if (center.a <= 1.0e-7) {
        froxelFiltered[centerCoordinate] = float4(0.0);
        return;
    }

    float centerDepth = froxelDepth[centerCoordinate];
    float depthTolerance = max(worldPush.fogLighting.w, 0.05);
    float4 sum = float4(0.0);
    float weightSum = 0.0;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int3 coordinate = centerCoordinate + int3(dx, dy, dz);
                if (any(coordinate < int3(0)) || any(coordinate >= dimensions)) {
                    continue;
                }
                float4 sampleValue = froxelScattering[coordinate];
                float sampleDepth = froxelDepth[coordinate];
                float sampleNear = froxelBoundaryDistance(float(coordinate.z));
                // Never pull a value from behind that neighboring camera ray's first interface.
                if (sampleValue.a <= 1.0e-7 || sampleNear >= sampleDepth) {
                    continue;
                }
                float depthDifference = abs(sampleDepth - centerDepth);
                float normalizedDifference = depthDifference / depthTolerance;
                float geometryWeight = 1.0 / (1.0 + normalizedDifference * normalizedDifference);
                float spatialWeight = exp(-0.5 * float(dx * dx + dy * dy + dz * dz));
                float weight = geometryWeight * spatialWeight;
                sum += sampleValue * weight;
                weightSum += weight;
            }
        }
    }
    froxelFiltered[centerCoordinate] = min(sum / max(weightSum, 1.0e-6), float4(HALF_MAX));
}
