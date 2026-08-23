// One invocation per froxel: evaluate the participating-medium material, atmosphere and emissive-block
// lighting, trace visibility, and temporally reproject the previous scattering field.

import world_common;
import world_core;
import bindings;
import sky;
import volumetric_lighting;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint3 index = DispatchRaysIndex();
    uint3 dimensions = DispatchRaysDimensions();
    if (any(index >= dimensions)) {
        return;
    }
    if (!froxelEnabled()) {
        froxelScattering[int3(index)] = float4(0.0);
        return;
    }

    celestialLight = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);
    froxelScattering[int3(index)] = injectFroxel(index);
}
