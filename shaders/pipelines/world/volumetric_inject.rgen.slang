// Stage 1: deterministic, history-free source/extinction injection at canonical frustum-froxel centres.

import world_common;
import world_core;
import bindings;
import sky;
import froxel;
import volumetric_lighting;

[shader("raygeneration")]
void main() {
    uint3 index = DispatchRaysIndex().xyz;
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    if (!froxelEnabled() || any(index >= uint3(froxelDimensions()))) {
        return;
    }

    celestialLight = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);
    froxelScattering[int3(index)] = injectFroxel(index);
}
