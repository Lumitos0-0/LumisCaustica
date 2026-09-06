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
import tile;

// The bake's state is one push block (declared by tile.slang, shared with the openness bake) plus this one
// sampler: the dominant-light choice must match world.rgen exactly, so it samples the raygen's own
// transmittance LUT.
[[vk::binding(0, 0)]] Sampler2D transmittanceLut;

public static const int FOG_VOLUME_CELLS       = 128; // cells per axis per level
public static const int FOG_VOLUME_LEVELS      = 4;

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
    // check.comp.slang ran first in the same submission (barrier between): when nothing that affects the
    // volume changed since the last bake, it clears this flag and the volume is already up to date.
    if (ConstPtr<uint32_t>(pcFog.fogStateAddr)[2u] == 0u) {
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
    // World-pinned lattice anchor: snaps the camera to the light-space 8-block lattice (see sky.slang
    // fogVolumeAnchor), so the cell set never moves as the camera travels — only the window indexing
    // shifts, at whole-cell steps, every 8 blocks. This is what removes the per-frame cell swimming.
    float3 anchor = fogVolumeAnchor(push.camOffset, light.dir);
    float size = exp2(float(level));               // 1, 2, 4, 8 blocks
    float halfExtent = 64.0 * size;
    int substeps = 1 << level;                     // z-substeps keep 1-block water/leaf steps exact

    // Level stride in uint32 ELEMENTS, not bytes: DevicePtr<uint32_t> indexes storage in elements, so a
    // byte-count levelOffset would push every level past k*8 MiB beyond the 32 MiB volume (level 3 would
    // write at +96 MiB). The fog.slang fetch uses the same element stride — kept in lock-step.
    uint cellsPerLevel = FOG_VOLUME_CELLS * FOG_VOLUME_CELLS * FOG_VOLUME_CELLS;
    uint levelOffset = uint(level) * cellsPerLevel;

    float3 backT = float3(1.0, 1.0, 1.0);
    // Per-column section-grid cache (see fogTileTransmittance): 0x7fffffff ensures the first step of the
    // march fetches, and the addresses stay valid for the whole bake (the grid is CPU-patched only
    // between frames, and a grid change invalidates the volume, triggering a fresh bake).
    uint64_t cachedTileAddr = 0u;
    int cachedSx = 0x7fffffff;
    int cachedSy = 0x7fffffff;
    int cachedSz = 0x7fffffff;
    // Sticky per-column "crosses tinted cutout/translucent" flag: the march runs z from the sun side
    // toward the camera, so once a coloured material is crossed every nearer cell (which accumulates the
    // same transmittance) is flagged for the exact-colour ray too.
    bool columnColored = false;
    for (int z = FOG_VOLUME_CELLS - 1; z >= 0; z--) {
        float zw = (float(z) + 0.5 - float(FOG_VOLUME_CELLS / 2)) * size;
        float uw = (float(cx) + 0.5 - float(FOG_VOLUME_CELLS / 2)) * size;
        float vw = (float(cy) + 0.5 - float(FOG_VOLUME_CELLS / 2)) * size;
        float3 base = anchor + light.dir * zw + right * uw + up * vw;

        float3 cellT = float3(1.0, 1.0, 1.0);
        bool cellColored = false;
        for (int s = 0; s < substeps; s++) {
            float zoff = ((float(s) + 0.5) / float(substeps) - 0.5) * size;
            bool colored;
            cellT *= fogTileTransmittance(base + light.dir * zoff, waterTint,
                    cachedTileAddr, cachedSx, cachedSy, cachedSz, colored);
            cellColored = cellColored || colored;
        }
        columnColored = columnColored || cellColored;
        backT *= cellT;
        uint outIndex = levelOffset + (uint(z) * FOG_VOLUME_CELLS + uint(cy)) * FOG_VOLUME_CELLS + uint(cx);
        DevicePtr<uint32_t>(pcFog.fogVolumeAddr)[outIndex] = packTransmittance(backT)
                | (columnColored ? FOG_PACK_EXACT_COLOR : 0u);
    }
}
