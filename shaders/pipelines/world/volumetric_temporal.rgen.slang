// Stage 3: low-weight, depth-validated temporal stabilization. History is reprojected from the current
// froxel's world position, sampled trilinearly, constrained to the current 3x3x3 source neighborhood,
// and rejected at disocclusions using the previous froxel depth volume. The user weight is capped on the
// CPU at 0.35, so this pass reduces movement noise without the long trails of conventional fog history.

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

    int3 coordinate = int3(index);
    float4 current = froxelFiltered[coordinate];
    if (current.a <= 1.0e-7 || !froxelHistoryValid() || worldPush.fogTemporal.x <= 0.0) {
        froxelResolved[coordinate] = current;
        return;
    }

    float currentDepth = froxelDepth[coordinate];
    float nearDistance = froxelBoundaryDistance(float(index.z));
    float farDistance = froxelBoundaryDistance(float(index.z + 1u));
    float currentDistance = 0.5 * (nearDistance + min(farDistance, currentDepth));
    float3 rayOrigin;
    float3 viewRay;
    froxelRay(float2(index.xy) + 0.5, rayOrigin, viewRay);
    float3 samplePosition = rayOrigin + viewRay * currentDistance;

    // camDelta converts a current camera-relative point to the previous camera-relative frame, matching
    // the path tracer's motion-vector convention in guides.slang.
    float3 previousCameraRelative = samplePosition - worldPush.camOffset + worldPush.camDelta;
    float previousDistance = length(previousCameraRelative);
    float4 previousClip = mul(worldPush.prevViewProj, float4(previousCameraRelative, 1.0));
    if (previousClip.w <= 0.0 || previousDistance >= worldPush.fogDistance.x) {
        froxelResolved[coordinate] = current;
        return;
    }
    float2 previousUv = previousClip.xy / previousClip.w * 0.5 + 0.5;
    float previousSlice = froxelSliceForDistance(previousDistance);
    float3 previousCoordinate = float3(
            previousUv * float2(dimensions.xy) - 0.5, previousSlice - 0.5);
    if (!froxelCoordinateValid(previousCoordinate)) {
        froxelResolved[coordinate] = current;
        return;
    }

    float previousDepth = froxelTrilinearHistoryDepth(previousCoordinate);
    float depthTolerance = max(worldPush.fogLighting.w, 0.05);
    float previousClearance = previousDepth - previousDistance;
    if (previousClearance < -depthTolerance) {
        froxelResolved[coordinate] = current;
        return;
    }

    // Neighborhood clipping is spatial only and reads the completed filter image. It bounds changes in
    // colored emitter selection without using extinction as an edge detector.
    float4 neighborhoodMin = current;
    float4 neighborhoodMax = current;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int3 neighbor = clamp(coordinate + int3(dx, dy, dz), int3(0), dimensions - 1);
                float4 value = froxelFiltered[neighbor];
                neighborhoodMin = min(neighborhoodMin, value);
                neighborhoodMax = max(neighborhoodMax, value);
            }
        }
    }
    float4 history = clamp(froxelTrilinearHistory(previousCoordinate),
            neighborhoodMin, neighborhoodMax);
    float currentClearance = currentDepth - currentDistance;
    float surfaceConfidence = saturate(min(currentClearance, max(previousClearance, 0.0))
            / depthTolerance);
    float historyWeight = min(worldPush.fogTemporal.x, 0.35) * surfaceConfidence;
    froxelResolved[coordinate] = lerp(current, history, historyWeight);
}
