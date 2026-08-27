// Stage 3: deterministic front-to-back prefix integration. Each segment uses the closed-form solution
// for a constant medium, avoiding stochastic ray marching and preserving exact Beer-Lambert opacity.

import world_common;
import world_core;
import bindings;
import froxel;

[shader("raygeneration")]
void main() {
    uint3 dispatchIndex = DispatchRaysIndex().xyz;
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    int3 dimensions = froxelDimensions();
    if (!froxelEnabled() || dispatchIndex.x >= uint(dimensions.x)
            || dispatchIndex.y >= uint(dimensions.y)) {
        return;
    }

    int2 xy = int2(dispatchIndex.xy);
    float opaqueDistance = froxelOpaqueDistance(float2(xy) + 0.5);
    float3 accumulatedScattering = float3(0.0);
    float transmittance = 1.0;
    for (int z = 0; z < dimensions.z; ++z) {
        float segmentNear = froxelBoundaryDistance(float(z));
        float segmentFar = min(froxelBoundaryDistance(float(z + 1)), opaqueDistance);
        float segmentLength = max(0.0, segmentFar - segmentNear);
        float4 medium = froxelFiltered[int3(xy, z)];
        float extinction = max(0.0, medium.a);
        if (segmentLength > 0.0 && extinction > 1.0e-7) {
            float segmentTransmittance = exp(-extinction * segmentLength);
            // medium.rgb is the source per unit length. The exact integral over the segment is
            // source * (1-exp(-sigma_t*ds))/sigma_t.
            float3 segmentScattering = medium.rgb
                    * ((1.0 - segmentTransmittance) / extinction);
            accumulatedScattering += transmittance * segmentScattering;
            transmittance *= segmentTransmittance;
        }
        froxelIntegrated[int3(xy, z)] = float4(
                min(accumulatedScattering, float3(HALF_MAX)), saturate(transmittance));
    }
}
