// Frustum-aligned froxel participating media: grid geometry, world-stable procedural density, temporal
// history access and the camera-ray fog composite. Depends on core, bindings, sky and math.
//
// One grid per frame, aligned to the camera frustum: xy cells cover square blocks of render pixels and
// Z follows a power distribution so nearby air gets more samples than distant haze. Each cell stores
//   froxelScattering  {sigma_s * L per unit length, scalar extinction}
//   froxelFilteredA/B the same quantity after temporal reprojection+variance clipping (ping-pong)
//   froxelIntegrated  {cumulative in-scattering.rgb, cumulative transmittance} along each XY column
// The injected "light" term is a per-unit-length radiance (sigma_s * L), so the integrate pass can apply
// the analytic Beer-Lambert slice integral and conserve energy across Z slices; nothing here is
// pre-multiplied by step length.
import world_common;
import world_core;
import bindings;
import sky;
import math;

// Flags packed into worldPush.froxelGrid.w.
public static const uint FROXEL_ENABLED = 1u;
public static const uint FROXEL_HISTORY_VALID = 2u;
public static const uint FROXEL_UNDERWATER = 4u;
// Bits 8..15 carry the emitter RIS candidate count, bits 16..19 the emitter samples per cell.

public bool froxelEnabled() {
    return (uint(worldPush.froxelGrid.w) & FROXEL_ENABLED) != 0u;
}

public bool froxelHistoryValid() {
    return (uint(worldPush.froxelGrid.w) & FROXEL_HISTORY_VALID) != 0u;
}

public bool froxelUnderwater() {
    return (uint(worldPush.froxelGrid.w) & FROXEL_UNDERWATER) != 0u;
}

public uint froxelLocalLightCandidates() {
    return (uint(worldPush.froxelGrid.w) >> 8u) & 0xffu;
}

public uint froxelEmitterSamples() {
    return max(1u, (uint(worldPush.froxelGrid.w) >> 16u) & 0x0fu);
}

public int3 froxelDimensions() {
    return worldPush.froxelGrid.xyz;
}

/**
 * XY resolution of the lighting probe grid. The inject pass shades one cell per 2x2 fog cells and the
 * filter/integrate passes run at full grid; the temporal filter is what resolves the difference, so the
 * light field can be far cheaper without the fog itself losing resolution. Z is un-decimated because
 * extinction changes per slice (density, height falloff), while the light field is smooth across slices.
 */
public int3 froxelInjectDimensions() {
    int3 dims = froxelDimensions();
    return int3((dims.x + 1) >> 1, (dims.y + 1) >> 1, dims.z);
}

/** Camera-ray distance at one nonlinear Z boundary (boundary 0 = camera, depth = max distance). */
public float froxelBoundaryDistance(float boundary) {
    float normalized = saturate(boundary / float(worldPush.froxelGrid.z));
    return pow(normalized, worldPush.fogDistance.y) * worldPush.fogDistance.x;
}

/** Continuous boundary coordinate for a camera-ray distance, in the same power-distributed space. */
public float froxelSliceForDistance(float distance) {
    float normalized = saturate(distance / worldPush.fogDistance.x);
    return pow(normalized, 1.0 / worldPush.fogDistance.y) * float(worldPush.froxelGrid.z);
}

/**
 * Reconstruct one unjittered frustum ray from a floating XY coordinate in cell-center space (the center
 * of cell i is i + 0.5). Same unprojection as the primary pass: near/far plane points from the inverse
 * view-projection, then shifted into the terrain's rebased coordinates by camOffset.
 */
public void froxelRay(float2 coordinate, out float3 origin, out float3 direction) {
    float2 uv = coordinate / float2(worldPush.froxelGrid.xy);
    float2 ndc = uv * 2.0 - 1.0;
    float4 nearH = mul(worldPush.invViewProj, float4(ndc, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(ndc, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    origin = nearP + worldPush.camOffset;
    direction = normalize(farP - nearP);
}

/**
 * First visible interface distance for a camera-ray column, or -1 when the primary ray missed geometry.
 * Pass A writes float depth into volumeDepth at render resolution; the grid cell is mapped to the pixel
 * at its center so a whole column shares the interface that stops fog accumulation.
 */
public float froxelOpaqueDistance(float2 containingCellCenter) {
    uint depthWidth;
    uint depthHeight;
    volumeDepth.GetDimensions(depthWidth, depthHeight);
    float2 cellSize = float2(depthWidth, depthHeight) / float2(worldPush.froxelGrid.xy);
    float2 pixel = containingCellCenter * cellSize - 0.5;
    int2 ip = int2(clamp(pixel, float2(0.0), float2(depthWidth, depthHeight) - 1.0));
    return volumeDepth[ip];
}

// ---- Procedural density ---------------------------------------------------

public float froxelHash(float3 p) {
    p = frac(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return frac((p.x + p.y) * p.z);
}

public float froxelValueNoise(float3 p) {
    float3 cell = floor(p);
    float3 f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);
    float n000 = froxelHash(cell + float3(0.0, 0.0, 0.0));
    float n100 = froxelHash(cell + float3(1.0, 0.0, 0.0));
    float n010 = froxelHash(cell + float3(0.0, 1.0, 0.0));
    float n110 = froxelHash(cell + float3(1.0, 1.0, 0.0));
    float n001 = froxelHash(cell + float3(0.0, 0.0, 1.0));
    float n101 = froxelHash(cell + float3(1.0, 0.0, 1.0));
    float n011 = froxelHash(cell + float3(0.0, 1.0, 1.0));
    float n111 = froxelHash(cell + float3(1.0, 1.0, 1.0));
    float nx00 = lerp(n000, n100, u.x);
    float nx10 = lerp(n010, n110, u.x);
    float nx01 = lerp(n001, n101, u.x);
    float nx11 = lerp(n011, n111, u.x);
    return lerp(lerp(nx00, nx10, u.y), lerp(nx01, nx11, u.y), u.z);
}

/** Three-octave band-limited FBM, the density's spatial variation. */
public float froxelFbm(float3 p) {
    float result = 0.0;
    float amplitude = 0.5714286;
    for (uint octave = 0u; octave < 3u; ++octave) {
        result += amplitude * froxelValueNoise(p);
        p = p * 2.03 + float3(17.1, 9.2, 13.7);
        amplitude *= 0.5;
    }
    return result;
}

/**
 * Scalar extinction at a rebased world position. The base height, falloff, noise amount/frequency and
 * base extinction come from worldPush (Java already selected underwater parameters when submerged) and
 * the noise field is anchored to world coordinates via fogWorld.xyz with slow wind advection from
 * fogWorld.w, so the fog does not swim when the terrain rebase origin shifts.
 */
public float froxelExtinction(float3 position) {
    float heightMultiplier = exp(clamp((worldPush.fogShape.x - position.y)
            * worldPush.fogDistance.w, -3.0, 3.0));
    float3 worldPosition = position + worldPush.fogWorld.xyz;
    float3 wind = float3(worldPush.fogWorld.w * 0.018, 0.0, worldPush.fogWorld.w * 0.009);
    float noise = froxelFbm(worldPosition * worldPush.fogShape.z + wind);
    float noiseMultiplier = lerp(1.0 - worldPush.fogShape.y,
            1.0 + worldPush.fogShape.y, noise);
    return worldPush.fogDistance.z * clamp(heightMultiplier * noiseMultiplier, 0.05, 12.0);
}

/** Deterministic per-frame sub-cell jitter so injection decorrelates across frames without drifting. */
public float2 froxelFrameJitter() {
    uint seed = worldPush.frameIndex * 2246822519u ^ 3266489917u;
    seed = pcg(seed);
    return float2(float(seed & 0xffffu) / 65535.0,
            float((seed >> 16u) & 0xffffu) / 65535.0) - 0.5;
}

// ---- Volume sampling ------------------------------------------------------

/**
 * Trilinear sample of one 3D volume at a continuous coordinate in cell units where cell i's center is
 * i + 0.5 (so coord == dimensions means just past the last center; clamped before the taps). {@code dims}
 * must be the volume's own dimensions — the lighting-probe volume is half XY of the fog grid, so it is
 * NOT always {@code froxelDimensions()}. RWTexture resources are passed through by handle, so the same
 * lookup serves history and integrated volumes.
 */
public float4 froxelTrilinear(RWTexture3D<float4> volume, int3 dims, float3 coord) {
    float3 p = clamp(coord - 0.5, float3(0.0), float3(dims) - 1.0);
    int3 lo = int3(floor(p));
    int3 hi = min(lo + 1, dims - 1);
    float3 f = p - floor(p);
    float4 c000 = volume[int3(lo.x, lo.y, lo.z)];
    float4 c100 = volume[int3(hi.x, lo.y, lo.z)];
    float4 c010 = volume[int3(lo.x, hi.y, lo.z)];
    float4 c110 = volume[int3(hi.x, hi.y, lo.z)];
    float4 c001 = volume[int3(lo.x, lo.y, hi.z)];
    float4 c101 = volume[int3(hi.x, lo.y, hi.z)];
    float4 c011 = volume[int3(lo.x, hi.y, hi.z)];
    float4 c111 = volume[int3(hi.x, hi.y, hi.z)];
    float4 c00 = lerp(c000, c100, f.x);
    float4 c10 = lerp(c010, c110, f.x);
    float4 c01 = lerp(c001, c101, f.x);
    float4 c11 = lerp(c011, c111, f.x);
    float4 c0 = lerp(c00, c10, f.y);
    float4 c1 = lerp(c01, c11, f.y);
    return lerp(c0, c1, f.z);
}

/**
 * Composite the accumulated volume onto a frame's path-traced radiance. fog.xyz is the in-scattering
 * already attenuated along the camera ray; fog.w is its transmittance, so this is exactly
 * L' = L * T + S — scene-linear, so it composes with DLSS-RR, exposure and the display transform.
 * Ray miss keeps fog up to max distance; a first interface stops the fog there (volumeDepth).
 */
public float3 applyFroxelFog(uint2 pixel, uint2 dimensions, float3 radiance) {
    if (!froxelEnabled()) {
        return radiance;
    }
    uint depthWidth;
    uint depthHeight;
    volumeDepth.GetDimensions(depthWidth, depthHeight);
    int2 ip = int2(min(pixel, uint2(depthWidth, depthHeight) - 1u));
    float3 coord;
    coord.xy = (float2(pixel) + 0.5)
            / (float2(dimensions) / float2(worldPush.froxelGrid.xy));
    float hit = volumeDepth[ip];
    float sampleDist = hit < 0.0 ? worldPush.fogDistance.x
            : clamp(hit, 1.0e-3, worldPush.fogDistance.x);
    coord.z = froxelSliceForDistance(sampleDist);
    float4 fog = froxelTrilinear(froxelIntegrated, froxelDimensions(), coord);
    return radiance * fog.w + fog.rgb;
}
