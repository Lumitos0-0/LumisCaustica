// One invocation per froxel: evaluate the participating-medium material, atmosphere and emissive-block
// lighting, trace visibility, and temporally reproject the previous scattering field.
//
// RtVolumetrics.record skips this dispatch entirely when volumetrics are disabled, so the enable bit in
// froxelGrid.w is always set here and the pass never has to clear a stale volume.

import world_common;
import world_core;
import bindings;
import sky;
import froxel;
import volumetric_lighting;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint3 index = DispatchRaysIndex();
    uint3 dimensions = DispatchRaysDimensions();
    if (any(index >= dimensions)) {
        return;
    }
    celestialLight = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);
    froxelScattering[int3(index)] = injectFroxel(index);
}
