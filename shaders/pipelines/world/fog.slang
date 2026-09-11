// Froxel volumetric fog: slice mapping, density noise, and sampling dither shared by the bake raygen
// (pipelines/world/fog.rgen) and the apply pass (pipelines/fog/apply.comp). Pure functions only — no
// bindings — so both pipelines can import this regardless of their descriptor sets.

import world_common;

// View depth (blocks) of the first slice boundary. Fog between the eye and this plane is negligible;
// geometry closer than this samples slice 0, which holds ~nothing.
public static const float FOG_NEAR = 0.5;

// Slice boundary s in [0, sliceCount] -> view depth. Exponential spacing (Frostbite-style froxels) keeps
// slice thickness proportional to distance, matching perspective foreshortening and depth precision.
public float fogSliceViewDepth(float s, float sliceCount, float maxDist) {
    return FOG_NEAR * pow(maxDist / FOG_NEAR, s / sliceCount);
}

// View depth -> slice coordinate in [0, sliceCount]. The apply pass normalizes this by sliceCount and
// samples the volume; clamping past the last slice keeps distant geometry fully fogged.
public float fogViewDepthToSlice(float viewDepth, float sliceCount, float maxDist) {
    return sliceCount * log(max(viewDepth, 1.0e-3) / FOG_NEAR) / log(maxDist / FOG_NEAR);
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

// Large-scale density variation around a mean of ~1: two value-noise octaves over 28-block cells,
// drifting slowly on the wind so shafts evolve instead of sitting still.
public float fogDensityNoise(float3 worldPos, float time) {
    float3 p = worldPos * (1.0 / 28.0) + float3(time * 0.008, 0.0, time * 0.003);
    float n = fogValueNoise(p) * 0.65 + fogValueNoise(p * 2.7 + 13.7) * 0.35;
    return clamp(n * 1.4 + 0.15, 0.0, 2.0);
}

// Interleaved gradient noise (Jimenez 2014): one dither value per froxel column for the in-slice sample
// position. Static — not animated per frame — so the volume is temporally stable without a TAA pass to
// integrate it; the apply pass's trilinear fetch diffuses the residual pattern.
public float fogIgn(uint2 pix) {
    return frac(52.9829189 * frac(dot(float2(pix) + 0.5, float2(0.06711056, 0.00583715))));
}
