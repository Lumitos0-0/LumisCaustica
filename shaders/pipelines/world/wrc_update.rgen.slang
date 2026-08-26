// One invocation per World Radiance Cache cell: evaluates direct celestial light, ambient sky,
// and local block emitter irradiance, writing the result to worldRadianceCache.

import world_common;
import world_core;
import bindings;
import sky;
import wrc;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint3 index = DispatchRaysIndex();
    uint3 dimensions = DispatchRaysDimensions();
    if (any(index >= dimensions)) {
        return;
    }
    if (!wrcEnabled()) {
        worldRadianceCache[int3(index)] = float4(0.0);
        return;
    }

    celestialLight = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);
    worldRadianceCache[int3(index)] = evaluateWrcCell(int3(index));
}
