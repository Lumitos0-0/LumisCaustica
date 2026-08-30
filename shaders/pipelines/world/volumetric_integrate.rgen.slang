// Camera-outward integration of the filtered froxel lighting. One raygen thread per XY column.
//
// Marching Z boundaries from the camera, each step applies the exact Beer–Lambert slice integral with
// constant extinction inside the slice:  T_step = exp(-sigma * step) and
// S_step = T * (sigma_s * L) * (1 - T_step) / sigma. The one-divide form is the analytic integral of
// the transmittance weighted light, so energy is conserved across the nonlinear Z spacing. The march
// stops at the column's first visible interface (volumeDepth), leaving fog behind walls unlit, and the
// remaining slices hold the final value so the fog composite can sample past the interface safely.
//
// Output per slice: {cumulative in-scattering.rgb, cumulative transmittance}; applyFroxelFog does
// L' = L * T + S in scene-linear space.
import world_common;
import world_core;
import bindings;
import froxel;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    int3 dims = froxelDimensions();
    uint3 tid = DispatchRaysIndex();
    if (tid.x >= uint(dims.x) || tid.y >= uint(dims.y) || tid.z != 0u) {
        return;
    }

    float clipDist = froxelOpaqueDistance(float2(tid.xy) + 0.5);
    if (clipDist < 0.0) {
        clipDist = worldPush.fogDistance.x;
    }
    clipDist = min(clipDist, worldPush.fogDistance.x);
    uint writeIdx = worldPush.frameIndex & 1u;
    float3 accumulated = float3(0.0, 0.0, 0.0);
    float transmittance = 1.0;

    for (int z = 0; z < dims.z; ++z) {
        float b0 = froxelBoundaryDistance(float(z));
        float b1 = froxelBoundaryDistance(float(z + 1));
        float end = min(b1, clipDist);
        float step = end - b0;
        // Lane chosen by parity branch (same convention as the filter pass).
        float4 cell;
        if (writeIdx == 0u) {
            cell = froxelFilteredA[int3(int(tid.x), int(tid.y), z)];
        } else {
            cell = froxelFilteredB[int3(int(tid.x), int(tid.y), z)];
        }
        if (step > 0.0) {
            float sigma = max(cell.w, 0.0);
            float tau = sigma * step;
            float transmitted = exp(-tau);
            // Integral of sigma_s*L*T over the slice for constant sigma:
            // (sigma_s*L/sigma) * (1 - T) == cell.rgb * (1 - T)/sigma. Falls back to the step-length
            // limit (cell.rgb * step) when the slice is optically thin.
            float3 slice = tau > 1.0e-4
                    ? cell.rgb * ((1.0 - transmitted) / max(sigma, 1.0e-6))
                    : cell.rgb * step;
            accumulated += transmittance * slice;
            transmittance *= transmitted;
        }
        // Every slice is written (also past the clip, with the terminal values) so a reprojected
        // history lookup can never read uninitialized data even though it stops at the interface.
        froxelIntegrated[int3(int(tid.x), int(tid.y), z)] = float4(accumulated, transmittance);
    }
}
