// Frustum-froxel geometry, procedural density, temporal reprojection, and final volume sampling.
//
// The volume stores two different quantities:
//   scattering/history = {in-scattered radiance per block, scalar extinction}
//   integrated         = {camera-to-boundary in-scattering, scalar transmittance}
// Keeping extinction linear until the integration pass follows the Frostbite V-buffer model and lets
// the Beer-Lambert segment integral conserve energy even when a Z slice spans many blocks.

import world_common;
import world_core;
import bindings;
import sky;

public bool froxelEnabled() {
    return (uint(worldPush.froxelGrid.w) & 1u) != 0u;
}

public bool froxelHistoryValid() {
    return (uint(worldPush.froxelGrid.w) & 2u) != 0u;
}

public bool froxelUnderwater() {
    return (uint(worldPush.froxelGrid.w) & 4u) != 0u;
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

/** Camera-ray distance at one nonlinear Z boundary. */
public float froxelBoundaryDistance(float boundary) {
    float normalized = saturate(boundary / float(worldPush.froxelGrid.z));
    return pow(normalized, worldPush.fogDistance.y) * worldPush.fogDistance.x;
}

/** Continuous boundary coordinate for a camera-ray distance. */
public float froxelSliceForDistance(float distance) {
    float normalized = saturate(distance / worldPush.fogDistance.x);
    return pow(normalized, 1.0 / worldPush.fogDistance.y) * float(worldPush.froxelGrid.z);
}

/** Reconstruct one unjittered frustum ray from a floating froxel XY coordinate. */
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

/** First camera interface for a floating XY froxel coordinate, sampled from Pass A's current depth. */
public float froxelOpaqueDistance(float2 coordinate) {
    uint depthWidth;
    uint depthHeight;
    volumeDepth.GetDimensions(depthWidth, depthHeight);
    float2 uv = clamp(coordinate / float2(worldPush.froxelGrid.xy), 0.0, 1.0);
    int2 pixel = min(int2(uv * float2(depthWidth, depthHeight)),
            int2(int(depthWidth) - 1, int(depthHeight) - 1));
    return clamp(volumeDepth[pixel], 0.0, worldPush.fogDistance.x);
}

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

/** World-stable height fog with a slowly advected, band-limited density field. */
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

/** Atmosphere-LUT radiance without sun/moon sprites, in the path tracer's ACEScg transport basis. */
public float3 froxelSkyRadiance(float3 direction, SkyState state) {
    float2 sunUv = skyViewLutUv(state.viewerRadiusKm, direction.y,
            lightViewCosine(direction, state.sunDir), SKY_VIEW_BODY_SUN);
    float2 moonUv = skyViewLutUv(state.viewerRadiusKm, direction.y,
            lightViewCosine(direction, state.moonDir), SKY_VIEW_BODY_MOON);
    float3 bt709 = skyViewLut.SampleLevel(sunUv, 0.0).rgb
            + skyViewLut.SampleLevel(moonUv, 0.0).rgb
            + state.airglowLuminance;
    return bt709ToAcesCg(max(bt709, float3(0.0)));
}

public float4 froxelHistoryLoad(int3 p) {
    return froxelHistory[p];
}

public float4 froxelFilteredLoad(int3 p) {
    return froxelFiltered[p];
}

public float4 froxelTrilinearHistory(float3 coordinate) {
    int3 dimensions = froxelDimensions();
    float3 bounded = clamp(coordinate, float3(0.0), float3(dimensions - 1));
    int3 lo = int3(floor(bounded));
    int3 hi = min(lo + int3(1), dimensions - 1);
    float3 f = frac(bounded);
    float4 c000 = froxelHistoryLoad(int3(lo.x, lo.y, lo.z));
    float4 c100 = froxelHistoryLoad(int3(hi.x, lo.y, lo.z));
    float4 c010 = froxelHistoryLoad(int3(lo.x, hi.y, lo.z));
    float4 c110 = froxelHistoryLoad(int3(hi.x, hi.y, lo.z));
    float4 c001 = froxelHistoryLoad(int3(lo.x, lo.y, hi.z));
    float4 c101 = froxelHistoryLoad(int3(hi.x, lo.y, hi.z));
    float4 c011 = froxelHistoryLoad(int3(lo.x, hi.y, hi.z));
    float4 c111 = froxelHistoryLoad(int3(hi.x, hi.y, hi.z));
    return lerp(lerp(lerp(c000, c100, f.x), lerp(c010, c110, f.x), f.y),
            lerp(lerp(c001, c101, f.x), lerp(c011, c111, f.x), f.y), f.z);
}

/** Trilinear sample of the spatially filtered scattering volume (interpolated in the per-pixel raymarch). */
public float4 froxelTrilinearFiltered(float3 coordinate) {
    int3 dimensions = froxelDimensions();
    float3 bounded = clamp(coordinate, float3(0.0), float3(dimensions - 1));
    int3 lo = int3(floor(bounded));
    int3 hi = min(lo + int3(1), dimensions - 1);
    float3 f = frac(bounded);
    float4 c000 = froxelFilteredLoad(int3(lo.x, lo.y, lo.z));
    float4 c100 = froxelFilteredLoad(int3(hi.x, lo.y, lo.z));
    float4 c010 = froxelFilteredLoad(int3(lo.x, hi.y, lo.z));
    float4 c110 = froxelFilteredLoad(int3(hi.x, hi.y, lo.z));
    float4 c001 = froxelFilteredLoad(int3(lo.x, lo.y, hi.z));
    float4 c101 = froxelFilteredLoad(int3(hi.x, lo.y, hi.z));
    float4 c011 = froxelFilteredLoad(int3(lo.x, hi.y, hi.z));
    float4 c111 = froxelFilteredLoad(int3(hi.x, hi.y, hi.z));
    return lerp(lerp(lerp(c000, c100, f.x), lerp(c010, c110, f.x), f.y),
            lerp(lerp(c001, c101, f.x), lerp(c011, c111, f.x), f.y), f.z);
}

/** Reproject a translated-world froxel sample into the previous frame's scattering volume.
 *  Improved coordinate handling: uses explicit half-pixel offset consistency and a small
 *  safety margin on slice bounds to reduce boundary drift during fast camera motion. */
public bool froxelReprojectHistory(float3 position, out float4 history) {
    history = float4(0.0);
    float3 previousCameraRelative = position - worldPush.camOffset + worldPush.camDelta;
    float4 previousClip = mul(worldPush.prevViewProj, float4(previousCameraRelative, 1.0));
    if (previousClip.w <= 0.0) {
        return false;
    }
    float2 uv = previousClip.xy / previousClip.w * 0.5 + 0.5;
    float previousDistance = length(previousCameraRelative);
    float slice = froxelSliceForDistance(previousDistance);

    // Safety margins prevent sampling exactly on the edge of the history volume
    const float2 uvMargin = float2(0.0015, 0.0015);
    const float sliceMargin = 0.75;
    if (any(uv < uvMargin) || any(uv > (1.0 - uvMargin)) ||
            slice < sliceMargin || slice > float(worldPush.froxelGrid.z) - sliceMargin) {
        return false;
    }

    // Consistent half-pixel offset with the rest of the pipeline
    float3 coordinate = float3(uv * float2(worldPush.froxelGrid.xy) - 0.5, slice - 0.5);
    history = froxelTrilinearHistory(coordinate);
    return true;
}

/** Per-pixel fog {in-scattering.rgb, transmittance} produced by the integrate pass's clamped raymarch. */
public float4 froxelAtPixel(uint2 pixel, uint2 imageSize) {
    if (any(pixel >= imageSize)) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    return froxelIntegrated[pixel];
}

public float3 applyFroxelFog(uint2 pixel, uint2 imageSize, float3 sceneRadiance) {
    if (!froxelEnabled()) {
        return sceneRadiance;
    }
    float4 fog = froxelAtPixel(pixel, imageSize);
    return sceneRadiance * clamp(fog.a, 0.0, 1.0) + max(fog.rgb, float3(0.0));
}
