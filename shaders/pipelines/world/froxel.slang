// Frustum-froxel geometry, procedural density, and stochastic raymarching integration.
//
// Scattering and scalar extinction are injected into the camera-aligned volume and spatially filtered.
// The integration model uses per-pixel stochastic raymarching through the filtered volume up to the first
// surface interface, evaluating the analytic constant-medium Beer-Lambert segment solution.
// Sub-slice stochastic jitter converts residual discretization into high-frequency Monte Carlo noise
// that DLSS Ray Reconstruction reconstructs with temporal stability and sharp geometry boundaries.

import world_common;
import world_core;
import bindings;
import sky;

public bool froxelEnabled() {
    return (uint(worldPush.froxelGrid.w) & 1u) != 0u;
}

public bool froxelHistoryValid() {
    return false;
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

public float4 froxelFilteredLoad(int3 p) {
    return froxelFiltered[p];
}

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

public float4 froxelHistoryLoad(int3 p) {
    return froxelFilteredLoad(p);
}

public float4 froxelIntegratedLoad(int3 p) {
    return froxelFilteredLoad(p);
}

public float4 froxelTrilinearHistory(float3 coordinate) {
    return froxelTrilinearFiltered(coordinate);
}

public float4 froxelTrilinearIntegrated(float3 coordinate) {
    return froxelTrilinearFiltered(coordinate);
}

public bool froxelReprojectHistory(float3 position, out float4 history) {
    history = float4(0.0);
    return false;
}

/**
 * Stochastic raymarching through the spatially filtered participating media volume.
 * Marches the view ray up to the first camera interface, evaluating the Beer-Lambert segment integral.
 * Per-frame stochastic sub-slice jitter enables DLSS Ray Reconstruction to reconstruct clean, sharp
 * light shafts and volumetric shadows without temporal accumulation lag.
 */
public void froxelRaymarch(uint2 pixel, uint2 imageSize, float rayEnd,
                           out float3 accumulatedScattering, out float accumulatedTransmittance) {
    accumulatedScattering = float3(0.0);
    accumulatedTransmittance = 1.0;

    if (rayEnd <= 0.0) {
        return;
    }

    int depthSlices = worldPush.froxelGrid.z;
    if (depthSlices <= 0) {
        return;
    }

    float maxDistance = worldPush.fogDistance.x;
    float clampedEnd = min(rayEnd, maxDistance);

    // Decorrelated per-pixel spatial hash combined with a temporal golden-ratio sequence across frames.
    float spatialJitter = frac(52.9829189 * frac(dot(float2(pixel), float2(0.06711056, 0.00583715))));
    float temporalOffset = frac(float(worldPush.frameIndex & 1023u) * 0.61803398875);
    float sliceJitter = frac(spatialJitter + temporalOffset) - 0.5;

    float2 uv = (float2(pixel) + 0.5 + worldPush.jitter) / float2(imageSize);
    float2 gridXY = uv * float2(worldPush.froxelGrid.xy) - 0.5;

    int maxSlice = min(int(ceil(froxelSliceForDistance(clampedEnd))), depthSlices);
    float previousDistance = 0.0;

    for (int z = 0; z < maxSlice; ++z) {
        if (previousDistance >= clampedEnd) {
            break;
        }

        float nextBoundary = float(z + 1);
        float nextDistance = (z + 1 >= depthSlices)
                ? maxDistance
                : min(froxelBoundaryDistance(nextBoundary), maxDistance);

        float segmentStart = previousDistance;
        float segmentEnd = min(nextDistance, clampedEnd);
        float segmentLength = max(segmentEnd - segmentStart, 0.0);

        if (segmentLength > 1.0e-6) {
            float zCoord = clamp(float(z) + 0.5 + sliceJitter * 0.65, 0.0, float(depthSlices - 1));
            float3 sampleCoord = float3(gridXY, zCoord);
            float4 medium = froxelTrilinearFiltered(sampleCoord);

            float extinction = max(medium.a, 0.0);
            float segmentTransmittance = exp(-extinction * segmentLength);
            float sourceIntegral = (extinction > 1.0e-7)
                    ? ((1.0 - segmentTransmittance) / extinction)
                    : segmentLength;

            accumulatedScattering += accumulatedTransmittance * max(medium.rgb, float3(0.0)) * sourceIntegral;
            accumulatedTransmittance *= segmentTransmittance;

            if (accumulatedTransmittance <= 1.0e-4) {
                break;
            }
        }

        previousDistance = nextDistance;
    }
}

/** Cumulative fog at the first primary interface via stochastic raymarching. */
public float4 froxelAtPixel(uint2 pixel, uint2 imageSize) {
    float distance = clamp(volumeDepth[int2(pixel)], 0.0, worldPush.fogDistance.x);
    if (distance <= 0.0) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float3 scattering;
    float transmittance;
    froxelRaymarch(pixel, imageSize, distance, scattering, transmittance);
    return float4(min(scattering, float3(HALF_MAX)), clamp(transmittance, 0.0, 1.0));
}

public float3 applyFroxelFog(uint2 pixel, uint2 imageSize, float3 sceneRadiance) {
    if (!froxelEnabled()) {
        return sceneRadiance;
    }
    float distance = clamp(volumeDepth[int2(pixel)], 0.0, worldPush.fogDistance.x);
    if (distance <= 0.0) {
        return sceneRadiance;
    }
    float3 scattering;
    float transmittance;
    froxelRaymarch(pixel, imageSize, distance, scattering, transmittance);
    return sceneRadiance * clamp(transmittance, 0.0, 1.0) + min(scattering, float3(HALF_MAX));
}
