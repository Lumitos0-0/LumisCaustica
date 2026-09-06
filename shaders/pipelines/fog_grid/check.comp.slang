// Fog-grid change check. One tiny dispatch runs before the bake: it derives the same key the bake is a
// function of (world-pinned volume anchor, dominant light direction, water tint, section-grid address +
// shift + publish version) and compares it with the key the volume was last baked with, storing the
// out-of-date flag in a small device buffer. The bake reads that flag and early-outs when nothing
// changed, so a static world with a static sun costs two near-empty dispatches instead of a full 8M
// occupancy march every frame. The anchor/light/water values are derived exactly as the bake derives
// them, so a stale key can never diverge from the bake's inputs.
//
// State layout (uint32 words at fogStateAddr):
//   [0..1] last-baked key (2 words)
//   [2]    decision: 0 = volume already up to date, 1 = bake this frame

import world_common;
import sky;

[[vk::push_constant]] FogGridPush pcFog;
[[vk::binding(0, 0)]] Sampler2D transmittanceLut;

uint hashMix(uint h, uint v) {
    h ^= v;
    h *= 0x85ebca6bu;
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;
    return h ^ (h >> 16u);
}

uint hashFloat(float v, uint h) {
    return hashMix(h, asuint(v));
}

uint hashFloat3(float3 v, uint h) {
    h = hashFloat(v.x, h);
    h = hashFloat(v.y, h);
    return hashFloat(v.z, h);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    WorldPush push = ConstPtr<WorldPush>(pcFog.worldPushAddr)[0];
    SkyState sky = skyState(push);
    CelestialLight light = dominantCelestialLight(sky, transmittanceLut);

    uint key0 = 2166136261u;
    key0 = hashFloat3(light.dir, key0);
    key0 = hashFloat3(fogVolumeAnchor(push.camOffset, light.dir), key0);
    key0 = hashFloat3(push.waterParams.xyz, key0);
    key0 = hashMix(key0, hashFloat(light.halfAngle, 0u));
    key0 = hashMix(key0, uint(pcFog.fogGridAddr));
    key0 = hashMix(key0, uint(pcFog.fogGridAddr >> 32u));
    key0 = hashMix(key0, uint(pcFog.gridShiftX));
    key0 = hashMix(key0, uint(pcFog.gridShiftY));
    key0 = hashMix(key0, uint(pcFog.gridShiftZ));

    uint key1 = 0x9e3779b9u;
    key1 = hashFloat3(light.dir, key1);
    key1 = hashFloat3(fogVolumeAnchor(push.camOffset, light.dir), key1);
    key1 = hashFloat3(push.waterParams.xyz, key1);
    key1 = hashMix(key1, uint(pcFog.fogGridVersion));
    key1 = hashMix(key1, uint(pcFog.fogGridAddr));
    key1 = hashMix(key1, uint(pcFog.fogGridAddr >> 32u));
    key1 = hashMix(key1, uint(pcFog.gridShiftX) ^ uint(pcFog.gridShiftY) ^ uint(pcFog.gridShiftZ));

    DevicePtr<uint32_t> state = DevicePtr<uint32_t>(pcFog.fogStateAddr);
    bool changed = state[0] != key0 || state[1] != key1;
    if (changed) {
        state[0] = key0;
        state[1] = key1;
    }
    state[2] = changed ? 1u : 0u;
}
