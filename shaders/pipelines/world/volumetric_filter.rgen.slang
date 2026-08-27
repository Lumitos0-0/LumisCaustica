// Stage 2: one deterministic bilateral 3x3x3 pass over the current frame's source field. Density and
// first-interface rejection prevent emissive energy crossing terrain/fog boundaries; no temporal input,
// reprojection, or frame-varying kernel is involved.

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
                if (sampleValue.a <= 1.0e-7) {
                    continue;
                }
                float relativeExtinctionDifference = abs(sampleValue.a - center.a)
                        / max(max(sampleValue.a, center.a), 1.0e-5);
                float bilateralWeight = exp(-relativeExtinctionDifference * worldPush.fogLighting.w);
                float spatialWeight = exp(-0.5 * float(dx * dx + dy * dy + dz * dz));
                float weight = bilateralWeight * spatialWeight;
                sum += sampleValue * weight;
                weightSum += weight;
            }
        }
    }
    froxelFiltered[centerCoordinate] = min(sum / max(weightSum, 1.0e-6), float4(HALF_MAX));
}
