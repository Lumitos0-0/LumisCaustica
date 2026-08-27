// Frustum-froxel geometry, static world-space density, and continuous volume reconstruction.
//
// froxelScattering/froxelFiltered store {single-scattering source radiance per block, extinction}.
// froxelIntegrated stores cumulative {camera-to-boundary in-scattering, transmittance}. Every frame is
// rebuilt from stable cell centres, spatially filtered using explicit geometry depth, stabilized with a
// low-weight depth-validated history sample, and analytically integrated before post-RR reconstruction.

import world_common;
import world_core;
import bindings;
import sky;

public bool froxelEnabled() {
    return (uint(worldPush.froxelGrid.w) & 1u) != 0u;
}

public bool froxelHistoryValid() {
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

/** First camera interface for a floating XY froxel coordinate, sampled from Pass A's depth. */
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

/** Three-octave, normalized fBm. Inputs are world-stable and deliberately have no time term. */
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

/** Extinction at the configured reference height with static world-space fBm modulation. */
public float froxelExtinction(float3 position) {
    float heightMultiplier = exp(clamp((worldPush.fogShape.x - position.y)
            * worldPush.fogDistance.w, -6.0, 6.0));
    float3 worldPosition = position + worldPush.fogWorld.xyz;
    float noise = froxelFbm(worldPosition * worldPush.fogShape.z);
    float noiseMultiplier = lerp(1.0 - worldPush.fogShape.y,
            1.0 + worldPush.fogShape.y, noise);
    return worldPush.fogDistance.z * clamp(heightMultiplier * noiseMultiplier, 0.0, 32.0);
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

public bool froxelCoordinateValid(float3 coordinate) {
    return all(coordinate >= float3(-0.5))
            && all(coordinate <= float3(froxelDimensions()) - 0.5);
}

/** Trilinear history fetch used after world-position reprojection. */
public float4 froxelTrilinearHistory(float3 coordinate) {
    int3 dimensions = froxelDimensions();
    float3 bounded = clamp(coordinate, float3(0.0), float3(dimensions - 1));
    int3 lo = int3(floor(bounded));
    int3 hi = min(lo + int3(1), dimensions - 1);
    float3 f = frac(bounded);
    float4 c000 = froxelHistory[int3(lo.x, lo.y, lo.z)];
    float4 c100 = froxelHistory[int3(hi.x, lo.y, lo.z)];
    float4 c010 = froxelHistory[int3(lo.x, hi.y, lo.z)];
    float4 c110 = froxelHistory[int3(hi.x, hi.y, lo.z)];
    float4 c001 = froxelHistory[int3(lo.x, lo.y, hi.z)];
    float4 c101 = froxelHistory[int3(hi.x, lo.y, hi.z)];
    float4 c011 = froxelHistory[int3(lo.x, hi.y, hi.z)];
    float4 c111 = froxelHistory[int3(hi.x, hi.y, hi.z)];
    return lerp(lerp(lerp(c000, c100, f.x), lerp(c010, c110, f.x), f.y),
            lerp(lerp(c001, c101, f.x), lerp(c011, c111, f.x), f.y), f.z);
}

public float froxelTrilinearHistoryDepth(float3 coordinate) {
    int3 dimensions = froxelDimensions();
    float3 bounded = clamp(coordinate, float3(0.0), float3(dimensions - 1));
    int3 lo = int3(floor(bounded));
    int3 hi = min(lo + int3(1), dimensions - 1);
    float3 f = frac(bounded);
    float c000 = froxelHistoryDepth[int3(lo.x, lo.y, lo.z)];
    float c100 = froxelHistoryDepth[int3(hi.x, lo.y, lo.z)];
    float c010 = froxelHistoryDepth[int3(lo.x, hi.y, lo.z)];
    float c110 = froxelHistoryDepth[int3(hi.x, hi.y, lo.z)];
    float c001 = froxelHistoryDepth[int3(lo.x, lo.y, hi.z)];
    float c101 = froxelHistoryDepth[int3(hi.x, lo.y, hi.z)];
    float c011 = froxelHistoryDepth[int3(lo.x, hi.y, hi.z)];
    float c111 = froxelHistoryDepth[int3(hi.x, hi.y, hi.z)];
    return lerp(lerp(lerp(c000, c100, f.x), lerp(c010, c110, f.x), f.y),
            lerp(lerp(c001, c101, f.x), lerp(c011, c111, f.x), f.y), f.z);
}

public float4 froxelTrilinearIntegrated(float3 coordinate) {
    int3 dimensions = froxelDimensions();
    float3 bounded = clamp(coordinate, float3(0.0), float3(dimensions - 1));
    int3 lo = int3(floor(bounded));
    int3 hi = min(lo + int3(1), dimensions - 1);
    float3 f = frac(bounded);
    float4 c000 = froxelIntegrated[int3(lo.x, lo.y, lo.z)];
    float4 c100 = froxelIntegrated[int3(hi.x, lo.y, lo.z)];
    float4 c010 = froxelIntegrated[int3(lo.x, hi.y, lo.z)];
    float4 c110 = froxelIntegrated[int3(hi.x, hi.y, lo.z)];
    float4 c001 = froxelIntegrated[int3(lo.x, lo.y, hi.z)];
    float4 c101 = froxelIntegrated[int3(hi.x, lo.y, hi.z)];
    float4 c011 = froxelIntegrated[int3(lo.x, hi.y, hi.z)];
    float4 c111 = froxelIntegrated[int3(hi.x, hi.y, hi.z)];
    return lerp(lerp(lerp(c000, c100, f.x), lerp(c010, c110, f.x), f.y),
            lerp(lerp(c001, c101, f.x), lerp(c011, c111, f.x), f.y), f.z);
}

/** Cumulative fog at a display-space UV, reconstructed trilinearly from the integrated volume. */
public float4 froxelAtUv(float2 uv) {
    float2 gridCoordinate = uv * float2(worldPush.froxelGrid.xy);
    float distance = froxelOpaqueDistance(gridCoordinate);
    if (distance <= 0.0) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float slice = froxelSliceForDistance(distance);
    float3 coordinate = float3(gridCoordinate - 0.5, slice - 1.0);
    float4 integrated = froxelTrilinearIntegrated(coordinate);
    if (slice < 1.0) {
        // Texel zero contains the complete first segment. Recover the exact constant-medium fraction so
        // fog begins continuously at the camera rather than popping at the first slice boundary.
        float fraction = saturate(slice);
        float fullTransmittance = clamp(integrated.a, 1.0e-6, 1.0);
        float partialTransmittance = pow(fullTransmittance, fraction);
        float fullOpacity = max(1.0 - fullTransmittance, 1.0e-6);
        integrated.rgb *= (1.0 - partialTransmittance) / fullOpacity;
        integrated.a = partialTransmittance;
    }
    return integrated;
}
