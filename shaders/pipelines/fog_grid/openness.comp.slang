// Fog sky-openness bake: a coarse per-cell "how much sky is visible from here" volume for the fog march's
// sky-ambient in-scatter term. Without this term fog is lit ONLY by the direct sun beam, so it goes black
// inside sun-shadow; with it, shadowed fog scatters skylight (dimmer, sky-tinted) while shafts stay bright.
//
// Layout: 32^3 RGBA8 cells of 8 blocks in the SAME world-pinned light-space lattice as the transmittance
// volume (sky.fogVolumeAnchor), covering ±128 blocks. Each channel is one sky lobe: the four directions
// ±lightRight/±lightUp each biased 0.5 toward the world zenith (every lobe is ~90° from the dominant body,
// so the sun/moon DISC never lands on a lobe). The value is the terrain only: each block of the walk
// contributes its per-block openness factor (solid→0, cutout→0.9564, translucent→0.9896, water→0.9381,
// empty→1.0 — see tile.slang), multiplied over 512 ONE-BLOCK steps outward (512-block reach, bounded by
// the section grid ±512 blocks). Sampling every block means a wall thinner than a cell can never fall
// between samples and be "ignored": any ≥1-block occluder along a lobe is hit.
//
// The lobe directions MUST stay in lock-step with fog.slang's fogSkyLobes(): the march evaluates the same
// four directions, so the openness the bake stored matches the lobes the march weights.
//
// Gated by the same change-check decision as the transmittance bake (runs right after it in the same
// submission, so an unchanged world/sun/anchor skips both).

import world_common;
import sky;
import tile;

[[vk::binding(0, 0)]] Sampler2D transmittanceLut;

public static const int FOG_OPENNESS_CELLS = 32;
public static const int FOG_OPENNESS_SIZE  = 8;    // blocks per cell
public static const int FOG_OPENNESS_STEPS = 512;  // one-block steps = 512-block reach

void fogSkyLobeDirs(float3 right, float3 up, out float3 d0, out float3 d1,
                    out float3 d2, out float3 d3) {
    float3 upW = float3(0.0, 1.0, 0.0);
    d0 = normalize(right + upW * 0.5);
    d1 = normalize(-right + upW * 0.5);
    d2 = normalize(up + upW * 0.5);
    d3 = normalize(-up + upW * 0.5);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= FOG_OPENNESS_CELLS || tid.y >= FOG_OPENNESS_CELLS || tid.z >= FOG_OPENNESS_CELLS) {
        return;
    }
    if (ConstPtr<uint32_t>(pcFog.fogStateAddr)[2u] == 0u) {
        return;
    }
    WorldPush push = ConstPtr<WorldPush>(pcFog.worldPushAddr)[0];
    SkyState sky = skyState(push);
    CelestialLight light = dominantCelestialLight(sky, transmittanceLut);
    float3 right;
    float3 up;
    celestialSquareFrame(light.dir, right, up);
    float3 anchor = fogVolumeAnchor(push.camOffset, light.dir);

    float3 d0;
    float3 d1;
    float3 d2;
    float3 d3;
    fogSkyLobeDirs(right, up, d0, d1, d2, d3);

    float zw = (float(tid.z) + 0.5 - float(FOG_OPENNESS_CELLS / 2)) * FOG_OPENNESS_SIZE;
    float uw = (float(tid.x) + 0.5 - float(FOG_OPENNESS_CELLS / 2)) * FOG_OPENNESS_SIZE;
    float vw = (float(tid.y) + 0.5 - float(FOG_OPENNESS_CELLS / 2)) * FOG_OPENNESS_SIZE;
    float3 center = anchor + light.dir * zw + right * uw + up * vw;

    // One tile cache per lobe: consecutive one-block steps of a lobe stay inside a 16-block tile, so each
    // lobe keeps its own cache instead of the four lobes evicting each other's tile every step.
    uint64_t tile0 = 0u;
    uint64_t tile1 = 0u;
    uint64_t tile2 = 0u;
    uint64_t tile3 = 0u;
    int s0x = 0x7fffffff, s0y = 0x7fffffff, s0z = 0x7fffffff;
    int s1x = 0x7fffffff, s1y = 0x7fffffff, s1z = 0x7fffffff;
    int s2x = 0x7fffffff, s2y = 0x7fffffff, s2z = 0x7fffffff;
    int s3x = 0x7fffffff, s3y = 0x7fffffff, s3z = 0x7fffffff;

    float o0 = 1.0;
    float o1 = 1.0;
    float o2 = 1.0;
    float o3 = 1.0;
    for (int s = 0; s < FOG_OPENNESS_STEPS; s++) {
        float t = float(s) + 0.5; // one sample per block, centered in the block
        float3 p = center;
        o0 *= fogTileOpenness(p + d0 * t, tile0, s0x, s0y, s0z);
        o1 *= fogTileOpenness(p + d1 * t, tile1, s1x, s1y, s1z);
        o2 *= fogTileOpenness(p + d2 * t, tile2, s2x, s2y, s2z);
        o3 *= fogTileOpenness(p + d3 * t, tile3, s3x, s3y, s3z);
        if (o0 + o1 + o2 + o3 <= 1.0e-4) {
            break;
        }
    }

    uint pack = uint(clamp(o0, 0.0, 1.0) * 255.0 + 0.5)
            | (uint(clamp(o1, 0.0, 1.0) * 255.0 + 0.5) << 8u)
            | (uint(clamp(o2, 0.0, 1.0) * 255.0 + 0.5) << 16u)
            | (uint(clamp(o3, 0.0, 1.0) * 255.0 + 0.5) << 24u);
    uint index = (tid.z * FOG_OPENNESS_CELLS + tid.y) * FOG_OPENNESS_CELLS + tid.x;
    DevicePtr<uint32_t>(pcFog.fogOpennessAddr)[index] = pack;
}
