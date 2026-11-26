import world_common;
import bindings;
import sky;
import trace;
import world_core;
import froxel_fog;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint idx = DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().x;
    if (idx >= fogLightCount(worldPush)) {
        return;
    }

    uint nearCount = fogCellCount(worldPush.fogNearGrid);
    bool nearGrid = idx < nearCount;
    uint cellIdx = nearGrid ? idx : idx - nearCount;
    int4 grid = nearGrid ? worldPush.fogNearGrid : worldPush.fogFarGrid;
    float4 origin = nearGrid ? worldPush.fogNearOrigin : worldPush.fogFarOrigin;
    uint width = uint(grid.x);
    uint plane = uint(grid.x * grid.y);
    uint3 cell = uint3(cellIdx % width, cellIdx / width % uint(grid.y), cellIdx / plane);

    SkyState sky = skyState(worldPush);
    CelestialLight light = dominantCelestialLight(sky, transmittanceLut);
    // World-aligned cells and the light's centre direction are independent of the frame's jitter.
    float3 position = origin.xyz + float3(cell) * origin.w;
    float3 filtered = float3(0.0);
    if (any(light.illuminance > 0.0001)) {
        filtered = visibility(position, light.dir, 10000.0).transmittance;
    }
    DevicePtr<float4> cache = DevicePtr<float4>(worldPush.fogBufferAddr);
    cache[idx] = float4(filtered, 0.0);
}
