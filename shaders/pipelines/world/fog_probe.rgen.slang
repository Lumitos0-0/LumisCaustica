// Fog tint probe: a THIRD raygen of the world RT pipeline (raygen index 2, dispatched right after the fog
// grid bake). The transmittance cache is monochrome by construction (solid 0, cutout 0.5, translucent 0.9
// grey), so the march cannot restore per-channel stained-glass tint or leaf texture-alpha dapple from it.
// This pass instead fires ONE real terrain-only visibility ray per FLAGGED cell of a 64^3 RGBA8 probe
// cache (1-block cells, ±32 blocks, same world-pinned light-space lattice as the volume), and any march
// within that window replaces its cache value with the probed per-channel tint. Colors are per CELL, not
// per march, so the exact rays cost is amortized over every pixel that sees the cell.
//
// Deterministic and state-gated like the bake: cells trace along the DOMINANT body direction (stable,
// RR-friendly), and the pass early-outs on the same change-check decision as the bake, so a static
// world/sun keeps the previous probe values (they are world-pinned, not camera-pinned).
//
// The probe is terrain-only (CULL_FOG_PROBE): entities use the fog march's separate entity-only ray, and
// the first-person body must never blank it. The any-hit path is the real one — cutout alpha-tests,
// translucent and water tint and pass through — so the stored value IS the old per-step ray's answer at
// the cell center.

import world_common;
import world_core;
import sky;
import trace;
import math;
import bindings;
import tile;

public static const int FOG_PROBE_CELLS = 64;
public static const uint FOG_PROBE_CLEAR = 0u; // sentinel: cell has no tint value (or is unprobed)

[shader("raygeneration")]
void main() {
    if (pc.fogProbeAddr == 0u || pc.fogStateAddr == 0u) {
        return;
    }
    // Same change-check gate as the bake: decision 0 means "nothing that affects the fog volumes changed",
    // so the cached probe values from the last bake are still exactly right.
    if (ConstPtr<uint32_t>(pc.fogStateAddr)[2u] == 0u) {
        return;
    }

    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    SkyState sky = skyState(worldPush);
    CelestialLight light = dominantCelestialLight(sky, transmittanceLut);
    float3 right;
    float3 up;
    celestialSquareFrame(light.dir, right, up);
    float3 anchor = fogVolumeAnchor(worldPush.camOffset, light.dir);

    uint3 dims = DispatchRaysDimensions();
    uint total = max(dims.x * dims.y, 1u);
    uint flat0 = DispatchRaysIndex().y * dims.x + DispatchRaysIndex().x;
    uint cells = uint(FOG_PROBE_CELLS);
    uint cellCount = cells * cells * cells;
    for (uint flat = flat0; flat < cellCount; flat += total) {
        uint u = flat % cells;
        uint v = (flat / cells) % cells;
        uint z = flat / (cells * cells);
        float zw = float(z) + 0.5 - float(FOG_PROBE_CELLS / 2);
        float uw = float(u) + 0.5 - float(FOG_PROBE_CELLS / 2);
        float vw = float(v) + 0.5 - float(FOG_PROBE_CELLS / 2);
        float3 center = anchor + light.dir * zw + right * uw + up * vw;

        // The volume's level-0 lattice is the same 1-block lattice shifted by 32 cells (its centre offset
        // is 64, ours 32), so the column flag is one fetch away.
        uint half = uint(FOG_PROBE_CELLS / 2);
        uint volumeIndex = ((z + half) * 128u + (v + half)) * 128u + (u + half);
        uint packed = ConstPtr<uint32_t>(pc.fogVolumeAddr)[volumeIndex];
        if ((packed & tile.FOG_PACK_EXACT_COLOR) == 0u) {
            DevicePtr<uint32_t>(pc.fogProbeAddr)[flat] = FOG_PROBE_CLEAR;
            continue;
        }
        VisibilityResult vis = visibilityFogProbe(center, light.dir, 512.0);
        float3 t = vis.transmittance;
        // Alpha 255 marks a PROBED cell: the sentinel is the all-zero word, and even a probe of black
        // glass stays non-zero (RGB may be 0 but alpha is 255).
        uint q = uint(clamp(t.x, 0.0, 1.0) * 255.0 + 0.5)
                | (uint(clamp(t.y, 0.0, 1.0) * 255.0 + 0.5) << 8u)
                | (uint(clamp(t.z, 0.0, 1.0) * 255.0 + 0.5) << 16u)
                | 0xFF000000u;
        DevicePtr<uint32_t>(pc.fogProbeAddr)[flat] = q;
    }
}
