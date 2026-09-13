// Froxel volumetric fog: slice mapping, density noise, and sampling dither shared by the bake raygen
// (pipelines/world/fog.rgen) and the apply pass (pipelines/fog/apply.comp). Pure functions only — no
// bindings — so both pipelines can import this regardless of their descriptor sets.

import world_common;

// View depth (blocks) of the first slice boundary. Fog between the eye and this plane is negligible;
// geometry closer than this samples slice 0, which holds ~nothing.
public static const float FOG_NEAR = 0.5;

// Slice-distribution modes (fogParams3.x). Exponential is the legacy Frostbite-style mapping: slice
// thickness proportional to distance, matching perspective foreshortening. Power is the RTX Remix
// mapping: view depth follows (s/N)^exponent, spreading slices toward the horizon so far voxels stay
// small — paid for out of the near range's generous slice budget (fixed-slice conservation).
public static const float FOG_SLICE_EXPONENTIAL = 0.0;
public static const float FOG_SLICE_POWER = 1.0;

// Slice boundary s in [0, sliceCount] -> view depth.
public float fogSliceViewDepth(float s, float sliceCount, float maxDist, float sliceMode, float sliceExp) {
    if (sliceMode < 0.5) {
        return FOG_NEAR * pow(maxDist / FOG_NEAR, s / sliceCount);
    }
    float t = clamp(s / sliceCount, 0.0, 1.0);
    return FOG_NEAR + (maxDist - FOG_NEAR) * pow(t, sliceExp);
}

// View depth -> slice coordinate in [0, sliceCount]. The apply pass calls this with sliceCount = 1 to
// get a normalized coordinate and samples the volume; clamping past the last slice keeps distant
// geometry fully fogged.
public float fogViewDepthToSlice(float viewDepth, float sliceCount, float maxDist, float sliceMode, float sliceExp) {
    if (sliceMode < 0.5) {
        return sliceCount * log(max(viewDepth, 1.0e-3) / FOG_NEAR) / log(maxDist / FOG_NEAR);
    }
    float t = clamp((viewDepth - FOG_NEAR) / max(maxDist - FOG_NEAR, 1.0e-3), 0.0, 1.0);
    return sliceCount * pow(t, 1.0 / max(sliceExp, 1.0e-3));
}

// Dave Hoskins' hash without the sine (same family as the starfield hash in sky.rmiss).
public float fogHash(float3 p) {
    p = frac(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return frac((p.x + p.y) * p.z);
}

public float fogValueNoise(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);
    return lerp(
        lerp(lerp(fogHash(i + float3(0.0, 0.0, 0.0)), fogHash(i + float3(1.0, 0.0, 0.0)), u.x),
             lerp(fogHash(i + float3(0.0, 1.0, 0.0)), fogHash(i + float3(1.0, 1.0, 0.0)), u.x), u.y),
        lerp(lerp(fogHash(i + float3(0.0, 0.0, 1.0)), fogHash(i + float3(1.0, 0.0, 1.0)), u.x),
             lerp(fogHash(i + float3(0.0, 1.0, 1.0)), fogHash(i + float3(1.0, 1.0, 1.0)), u.x), u.y), u.z);
}

// Gentle density variation around a mean of ~1: two value-noise octaves over 10-block cells,
// drifting slowly on the wind so shafts evolve instead of sitting still. Kept low-contrast — coarse
// near-field blobs read as smoke rather than air — and faded out near the camera by the bake.
public float fogDensityNoise(float3 worldPos, float time) {
    float3 p = worldPos * (1.0 / 10.0) + float3(time * 0.022, 0.0, time * 0.008);
    float n = fogValueNoise(p) * 0.65 + fogValueNoise(p * 2.7 + 13.7) * 0.35;
    return clamp(n * 0.9 + 0.4, 0.0, 2.0);
}
