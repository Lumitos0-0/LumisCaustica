// Integrate the spatially filtered froxel volume along each PIXEL's view ray, clamped at the first
// opaque/water surface. This is the per-pixel raymarch that keeps the coarse grid from ever being
// visible: each pixel marches its own continuous ray through the volume with trilinear sampling, and a
// STATIC per-pixel sub-slice dither breaks up the finite-depth discretization spatially instead of
// forming camera-facing layers. Because the dither depends only on the pixel (never the frame) and the
// volume is deterministic, the fog is stable frame-to-frame with no temporal accumulation needed.

import world_common;
import world_core;
import bindings;
import froxel;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dispatchDimensions = DispatchRaysDimensions().xy;
    if (any(pixel >= dispatchDimensions)) {
        return;
    }

    if (!froxelEnabled()) {
        froxelIntegrated[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float rayEnd = clamp(volumeDepth[int2(pixel)], 0.0, worldPush.fogDistance.x);
    if (rayEnd <= 0.0) {
        froxelIntegrated[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    int3 grid = worldPush.froxelGrid.xyz;
    float2 uv = (float2(pixel) + 0.5 + worldPush.jitter) / float2(dispatchDimensions);

    // Static per-pixel sub-slice dither (screen-space, frame-stable). Interleaves the residual discrete
    // slice boundaries so bilinear sampling hides them, without causing temporal shimmer.
    float sliceJitter = frac(52.9829189 * frac(dot(float2(pixel), float2(0.06711056, 0.00583715)))) - 0.5;

    float3 accumulated = float3(0.0);
    float transmittance = 1.0;
    for (int slice = 0; slice < grid.z; ++slice) {
        float segmentStart = froxelBoundaryDistance(float(slice));
        if (segmentStart >= rayEnd) {
            break;
        }
        float segmentEnd = min(froxelBoundaryDistance(float(slice + 1)), rayEnd);
        float segmentLength = max(segmentEnd - segmentStart, 0.0);

        float z = clamp((float(slice) + 0.5 + sliceJitter * 0.65) / float(grid.z),
                0.5 / float(grid.z), 1.0 - 0.5 / float(grid.z));
        float3 coordinate = float3(
                uv * float2(grid.xy) - 0.5,
                z * float(grid.z) - 0.5);
        float4 medium = froxelTrilinearFiltered(coordinate);

        float extinction = max(medium.a, 1.0e-8);
        float segmentTransmittance = exp(-extinction * segmentLength);
        float sourceIntegral = extinction > 1.0e-7
                ? (1.0 - segmentTransmittance) / extinction : segmentLength;
        accumulated += transmittance * max(medium.rgb, float3(0.0)) * sourceIntegral;
        transmittance *= segmentTransmittance;
    }

    froxelIntegrated[pixel] = float4(
            min(accumulated, float3(HALF_MAX)),
            clamp(transmittance, 0.0, 1.0));
}
