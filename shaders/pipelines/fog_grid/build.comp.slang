// Fog-grid bake: builds the per-frame light-space terrain transmittance volume the fog march samples.
//
// The fog direct term needs T_terrain(p -> sun) for every march sample p. Firing a TLAS visibility ray
// per sample is what made the fog expensive; this pass instead pre-computes the answer for every point of
// a camera-centered box, once per frame, and the march then reads it with a handful of constant-time
// fetches. The volume is built by marching straight lines along the dominant light direction through the
// same occupancy tiles the terrain mesher produces (16^3 per section, 2x2x2 subcell masks + material
// class), so the data is the real scene geometry — not a rasterized shadow map — and can be re-integrated
// per frame at negligible cost.
//
// Layout: four levels, each 128^3 cells of a single packed uint (RGB = transmittance, 8 bits/channel).
// Level k has cellSize 2^k blocks and covers a cube of extent 64*2^k blocks around the camera. Cell
// (u,v,z) stores transmittance from that cell to the cube's +z face (z = light direction), so a fog
// sample's terrain transmittance is one fetch per level chained at the shared boundaries — the same
// telescoping ratio the atmosphere LUTs use, but in a cache that costs a few fetches instead of a ray.
//
// The section grid is indexed by 16-block-aligned world coordinates: the grid's origin is the player's
// section at the last recenter, and gridShift = terrain rebase base − grid origin (blocks) maps the
// rebased p the bake holds into that space. Sections publish/shrink through the CPU writing their tile's
// device address into the grid (0 = no terrain, so the march skips straight through unloaded space).
//
// Deterministic and world-anchored: the volume depends only on the current dominant light direction, the
// camera center and the published terrain, so DLSS-RR sees a static medium across frames.

import world_common;
import sky;

// The bake's state is one push block (world_common.FogGridPush) plus this one sampler: the dominant-light
// choice must match world.rgen exactly, so it samples the raygen's own transmittance LUT.
[[vk::push_constant]] FogGridPush pcFog;

[[vk::binding(0, 0)]] Sampler2D transmittanceLut;

public static const int FOG_GRID_CELLS_PER_AXIS = 64; // section grid: 1024-block radius / 16
public static const int FOG_GRID_HALF          = 32;
public static const int FOG_VOLUME_CELLS       = 128; // cells per axis per level
public static const int FOG_VOLUME_LEVELS      = 4;

// Per-1-block-sample transmittance from the section grid + occupancy tile. p is rebased world blocks.
float3 fogTileTransmittance(float3 p, float3 waterTint) {
    if (pcFog.fogGridAddr == 0u) {
        return float3(1.0, 1.0, 1.0); // no terrain published yet: clear, exactly like an escaping ray
    }
    float3 q = p + float3(float(pcFog.gridShiftX), float(pcFog.gridShiftY), float(pcFog.gridShiftZ));
    int sx = int(floor(q.x * (1.0 / 16.0)));
    int sy = int(floor(q.y * (1.0 / 16.0)));
    int sz = int(floor(q.z * (1.0 / 16.0)));
    if (sx < -FOG_GRID_HALF || sx >= FOG_GRID_HALF
            || sy < -FOG_GRID_HALF || sy >= FOG_GRID_HALF
            || sz < -FOG_GRID_HALF || sz >= FOG_GRID_HALF) {
        return float3(1.0, 1.0, 1.0);
    }
    uint index = uint((sz + FOG_GRID_HALF) * FOG_GRID_CELLS_PER_AXIS * FOG_GRID_CELLS_PER_AXIS
            + (sy + FOG_GRID_HALF) * FOG_GRID_CELLS_PER_AXIS + (sx + FOG_GRID_HALF));
    uint64_t tileAddr = ConstPtr<uint64_t>(pcFog.fogGridAddr)[index];
    if (tileAddr == 0u) {
        return float3(1.0, 1.0, 1.0); // unloaded/outside window: clear, exactly like an escaping ray
    }
    float3 local = q - float3(float(sx * 16), float(sy * 16), float(sz * 16));
    int bx = clamp(int(local.x), 0, 15);
    int by = clamp(int(local.y), 0, 15);
    int bz = clamp(int(local.z), 0, 15);
    // Tile layout: one uint per block, x-major (by*16 + bz)*16 + bx — kept in lock-step with
    // RtTerrainMesher's occupancy emission.
    uint tile = ConstPtr<uint32_t>(tileAddr)[uint((by * 16 + bz) * 16 + bx)];
    uint mask = tile & 0xFFu;
    uint cls = (tile >> 8u) & 7u;
    if (mask == 0u) {
        return float3(1.0, 1.0, 1.0);
    }
    float3 lf = local - float3(float(bx), float(by), float(bz));
    int sxx = min(int(lf.x * 2.0), 1);
    int syy = min(int(lf.y * 2.0), 1);
    int szz = min(int(lf.z * 2.0), 1);
    uint bit = 1u << uint(sxx + syy * 2 + szz * 4);
    if ((mask & bit) == 0u) {
        return float3(1.0, 1.0, 1.0);
    }
    // Solid: any presence blocks the whole block — the ray cannot leave through a culled face without
    // crossing one of the faces the mesher emitted (vanilla's face culling only hides faces against other
    // solid geometry), so the interior is opaque by construction. Cutout/translucent/water keep the
    // per-subcell behaviour: the ray passes through the open subcells exactly like the any-hit path does.
    if (cls == 1u) {
        return float3(0.0, 0.0, 0.0);
    }
    if (cls == 2u) {
        return float3(0.5, 0.5, 0.5); // foliage: fixed alpha, dappled by the volume's lateral jitter
    }
    if (cls == 3u) {
        return float3(0.9, 0.9, 0.9); // glass/ice: mostly transmissive
    }
    if (cls == 4u) {
        return waterTint; // one block of water, same tint the visibility any-hit uses
    }
    return float3(1.0, 1.0, 1.0);
}

uint packTransmittance(float3 t) {
    uint3 q = uint3(clamp(t, float3(0.0), float3(1.0)) * 255.0 + 0.5);
    return q.x | (q.y << 8u) | (q.z << 16u);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    int level = int(tid.z);
    int cx = int(tid.x);
    int cy = int(tid.y);
    if (cx >= FOG_VOLUME_CELLS || cy >= FOG_VOLUME_CELLS || level >= FOG_VOLUME_LEVELS) {
        return;
    }
    WorldPush push = ConstPtr<WorldPush>(pcFog.worldPushAddr)[0];
    SkyState sky = skyState(push);
    CelestialLight light = dominantCelestialLight(sky, transmittanceLut);
    // Same frame the raygen derives its NEE light with; the tint is computed in the same space the
    // visibility any-hit multiplies it in.
    float3 waterTint = push.waterParams.xyz;

    float3 right;
    float3 up;
    celestialSquareFrame(light.dir, right, up);
    float size = exp2(float(level));               // 1, 2, 4, 8 blocks
    float halfExtent = 64.0 * size;
    int substeps = 1 << level;                     // z-substeps keep 1-block water/leaf steps exact

    // Level stride in uint32 ELEMENTS, not bytes: DevicePtr<uint32_t> indexes storage in elements, so a
    // byte-count levelOffset would push every level past k*8 MiB beyond the 32 MiB volume (level 3 would
    // write at +96 MiB). The fog.slang fetch uses the same element stride — kept in lock-step.
    uint cellsPerLevel = FOG_VOLUME_CELLS * FOG_VOLUME_CELLS * FOG_VOLUME_CELLS;
    uint levelOffset = uint(level) * cellsPerLevel;

    float3 backT = float3(1.0, 1.0, 1.0);
    for (int z = FOG_VOLUME_CELLS - 1; z >= 0; z--) {
        float zw = (float(z) + 0.5 - float(FOG_VOLUME_CELLS / 2)) * size;
        float uw = (float(cx) + 0.5 - float(FOG_VOLUME_CELLS / 2)) * size;
        float vw = (float(cy) + 0.5 - float(FOG_VOLUME_CELLS / 2)) * size;
        float3 base = push.camOffset + light.dir * zw + right * uw + up * vw;

        float3 cellT = float3(1.0, 1.0, 1.0);
        for (int s = 0; s < substeps; s++) {
            float zoff = ((float(s) + 0.5) / float(substeps) - 0.5) * size;
            cellT *= fogTileTransmittance(base + light.dir * zoff, waterTint);
        }
        backT *= cellT;
        uint outIndex = levelOffset + (uint(z) * FOG_VOLUME_CELLS + uint(cy)) * FOG_VOLUME_CELLS + uint(cx);
        DevicePtr<uint32_t>(pcFog.fogVolumeAddr)[outIndex] = packTransmittance(backT);
    }
}
