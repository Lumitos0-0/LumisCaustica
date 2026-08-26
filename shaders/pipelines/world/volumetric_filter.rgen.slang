// Low-pass the injected scattering field in screen-froxel space before stochastic raymarching.
// This filter operates on the continuous volume itself, not scene geometry. Final composition samples the
// 3D field at each pixel's exact depth. A centre-weighted kernel keeps denoising while preserving the
// higher-frequency visibility transitions that define light-shaft edges.

import world_common;
import world_core;
import bindings;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint3 index = DispatchRaysIndex();
    uint3 dispatchDimensions = DispatchRaysDimensions();
    if (any(index >= dispatchDimensions)) {
        return;
    }

    int3 center = int3(index);
    // Volumetric spatial filter bypassed for testing: passes raw injection directly to DLSS-RR
    froxelFiltered[center] = froxelScattering[center];
}
