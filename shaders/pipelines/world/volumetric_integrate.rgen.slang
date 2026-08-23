// One invocation per XY column. Prefix-integrate scattering and extinction from the camera through all
// nonlinear depth slices using the analytic constant-medium Beer-Lambert segment solution.

import world_common;
import world_core;
import bindings;
import froxel;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 column = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    if (any(column >= dimensions)) {
        return;
    }

    float3 accumulatedScattering = float3(0.0);
    float accumulatedTransmittance = 1.0;
    float previousDistance = 0.0;
    for (int z = 0; z < worldPush.froxelGrid.z; ++z) {
        float farDistance = froxelBoundaryDistance(float(z + 1));
        float segmentLength = max(farDistance - previousDistance, 0.0);
        float4 cell = froxelScattering[int3(int2(column), z)];
        float extinction = max(cell.a, 0.0);
        float segmentTransmittance = exp(-extinction * segmentLength);
        float sourceIntegral = extinction > 1.0e-7
                ? (1.0 - segmentTransmittance) / extinction : segmentLength;
        accumulatedScattering += accumulatedTransmittance * max(cell.rgb, float3(0.0)) * sourceIntegral;
        accumulatedTransmittance *= segmentTransmittance;
        froxelIntegrated[int3(int2(column), z)] = float4(
                min(accumulatedScattering, float3(HALF_MAX)),
                clamp(accumulatedTransmittance, 0.0, 1.0));
        previousDistance = farDistance;
    }
}
