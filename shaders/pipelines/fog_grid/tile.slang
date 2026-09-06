// Shared terrain-occupancy tile reader for the fog-grid family (build.comp, openness.comp). The section
// grid is indexed by 16-block-aligned world coordinates: the grid's origin is the player's section at the
// last recenter, and gridShift = terrain rebase base − grid origin (blocks) maps the rebased p the bake
// holds into that space. Sections publish/shrink through the CPU writing their tile's device address into
// the grid (0 = no terrain, so readers skip straight through unloaded space).
//
// Tile layout: one uint per block, x-major (by*16 + bz)*16 + bx — kept in lock-step with
// RtTerrainMesher's occupancy emission. Bits 0..7 = 2x2x2 subcell mask, bits 8..10 = material class
// (1 solid, 2 cutout foliage, 3 translucent glass/ice, 4 water; 0 = empty).
//
// Bit 24 of the fog volume's packed RGBA8 transmittance uint: set when the baked column crosses a tinted
// cutout/translucent material, signalling the march to use the per-cell tint probe cache.
import world_common;

[[vk::push_constant]] FogGridPush pcFog;

public static const int FOG_GRID_CELLS_PER_AXIS = 64; // section grid: 1024-block radius / 16
public static const int FOG_GRID_HALF          = 32;

public static const uint FOG_PACK_EXACT_COLOR = 1u << 24u;

/**
 * Reads the occupancy tile at p (rebased world blocks). Returns material class (0 = empty/unloaded, so the
 * caller treats it as clear) and fills the per-march cache so a marching column re-fetches the grid entry
 * only once per crossed 16-block tile — same addresses, ~16x fewer dependent grid loads.
 */
uint fogTileClass(float3 p, inout uint64_t cachedTileAddr, inout int cachedSx,
                  inout int cachedSy, inout int cachedSz, inout uint tile) {
    if (pcFog.fogGridAddr == 0u) {
        return 0u;
    }
    float3 q = p + float3(float(pcFog.gridShiftX), float(pcFog.gridShiftY), float(pcFog.gridShiftZ));
    int sx = int(floor(q.x * (1.0 / 16.0)));
    int sy = int(floor(q.y * (1.0 / 16.0)));
    int sz = int(floor(q.z * (1.0 / 16.0)));
    if (sx < -FOG_GRID_HALF || sx >= FOG_GRID_HALF
            || sy < -FOG_GRID_HALF || sy >= FOG_GRID_HALF
            || sz < -FOG_GRID_HALF || sz >= FOG_GRID_HALF) {
        return 0u;
    }
    if (sx != cachedSx || sy != cachedSy || sz != cachedSz) {
        uint index = uint((sz + FOG_GRID_HALF) * FOG_GRID_CELLS_PER_AXIS * FOG_GRID_CELLS_PER_AXIS
                + (sy + FOG_GRID_HALF) * FOG_GRID_CELLS_PER_AXIS + (sx + FOG_GRID_HALF));
        cachedTileAddr = ConstPtr<uint64_t>(pcFog.fogGridAddr)[index];
        cachedSx = sx;
        cachedSy = sy;
        cachedSz = sz;
    }
    if (cachedTileAddr == 0u) {
        return 0u;
    }
    float3 local = q - float3(float(sx * 16), float(sy * 16), float(sz * 16));
    int bx = clamp(int(local.x), 0, 15);
    int by = clamp(int(local.y), 0, 15);
    int bz = clamp(int(local.z), 0, 15);
    tile = ConstPtr<uint32_t>(cachedTileAddr)[uint((by * 16 + bz) * 16 + bx)];
    return (tile >> 8u) & 7u;
}

/**
 * Per-1-block-sample transmittance from the section grid + occupancy tile: solid blocks are opaque,
 * cutout/translucent keep the per-subcell behaviour (the ray passes through the open subcells exactly like
 * the any-hit path does) with the cache's grey approximation, water keeps the camera-biome tint.
 * `colored` reports a tinted cutout/translucent material (the march's exact probe territory).
 */
float3 fogTileTransmittance(float3 p, float3 waterTint,
                            inout uint64_t cachedTileAddr, inout int cachedSx,
                            inout int cachedSy, inout int cachedSz,
                            out bool colored) {
    colored = false;
    uint tile = 0u;
    uint cls = fogTileClass(p, cachedTileAddr, cachedSx, cachedSy, cachedSz, tile);
    if (cls == 0u || (tile & 0xFFu) == 0u) {
        return float3(1.0, 1.0, 1.0);
    }
    float3 q = p + float3(float(pcFog.gridShiftX), float(pcFog.gridShiftY), float(pcFog.gridShiftZ));
    int sx = int(floor(q.x * (1.0 / 16.0)));
    int sy = int(floor(q.y * (1.0 / 16.0)));
    int sz = int(floor(q.z * (1.0 / 16.0)));
    float3 local = q - float3(float(sx * 16), float(sy * 16), float(sz * 16));
    int bx = clamp(int(local.x), 0, 15);
    int by = clamp(int(local.y), 0, 15);
    int bz = clamp(int(local.z), 0, 15);
    float3 lf = local - float3(float(bx), float(by), float(bz));
    int sxx = min(int(lf.x * 2.0), 1);
    int syy = min(int(lf.y * 2.0), 1);
    int szz = min(int(lf.z * 2.0), 1);
    uint bit = 1u << uint(sxx + syy * 2 + szz * 4);
    if ((tile & bit) == 0u) {
        return float3(1.0, 1.0, 1.0);
    }
    // Solid: any presence blocks the whole block — the ray cannot leave through a culled face without
    // crossing one of the faces the mesher emitted (vanilla's face culling only hides faces against other
    // solid geometry), so the interior is opaque by construction.
    if (cls == 1u) {
        return float3(0.0, 0.0, 0.0);
    }
    if (cls == 2u) {
        colored = true; // foliage: grey approximation only; the tint probe restores per-leaf dapple
        return float3(0.5, 0.5, 0.5);
    }
    if (cls == 3u) {
        colored = true; // glass/ice: grey approximation only; the tint probe restores per-channel tint
        return float3(0.9, 0.9, 0.9);
    }
    if (cls == 4u) {
        return waterTint; // one block of water; camera-biome tint is the same approximation the any-hit
                          // would average per body, and bodies are everywhere near water (not flagged)
    }
    return float3(1.0, 1.0, 1.0);
}

/** Sky openness factor (1 = clear, 0 = blocked) for one sample: the sky-ambient term's occlusion. */
float fogTileOpenness(float3 p, inout uint64_t cachedTileAddr, inout int cachedSx,
                      inout int cachedSy, inout int cachedSz) {
    uint tile = 0u;
    uint cls = fogTileClass(p, cachedTileAddr, cachedSx, cachedSy, cachedSz, tile);
    if (cls == 0u || (tile & 0xFFu) == 0u) {
        return 1.0;
    }
    // Openness does not need the subcell test: ambient light is soft and coarse, and treating the whole
    // block by its class keeps the ambient stable frame-to-frame.
    if (cls == 1u) {
        return 0.0;
    }
    if (cls == 2u) {
        return 0.7;
    }
    if (cls == 3u) {
        return 0.92;
    }
    if (cls == 4u) {
        return 0.6;
    }
    return 1.0;
}
