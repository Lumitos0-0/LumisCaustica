// One invocation per XY column. Prefix-integrate scattering and extinction from the camera through all
// nonlinear depth slices using the analytic constant-medium Beer-Lambert segment solution, and return
// the resolved field to the scattering volume that becomes the next frame's history.

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
        float4 cell = froxelFiltered[int3(int2(column), z)];
        float extinction = max(cell.a, 0.0);
        float segmentTransmittance = exp(-extinction * segmentLength);
        float sourceIntegral = extinction > 1.0e-7
                ? (1.0 - segmentTransmittance) / extinction : segmentLength;
        accumulatedScattering += accumulatedTransmittance * max(cell.rgb, float3(0.0)) * sourceIntegral;
        accumulatedTransmittance *= segmentTransmittance;
        froxelIntegrated[int3(int2(column), z)] = float4(
                min(accumulatedScattering, float3(HALF_MAX)),
                clamp(accumulatedTransmittance, 0.0, 1.0));
        // Hand the resolved value back into this frame's scattering volume, which is next frame's
        // history. The filter pass cannot do this itself: it gathers its neighbourhood from that very
        // image, so writing it there would race with its own reads. This pass is column-local and each
        // thread owns exactly one column, so the writes are disjoint and this is the safe place to
        // close the loop. Without it the ping-pong history would hold the inject pass's raw, single
        // sample instead of the resolved field.
        froxelScattering[int3(int2(column), z)] = cell;
        previousDistance = farDistance;
    }
}
