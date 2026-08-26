// Spatially Hashed World Radiance Cache (SHARC).
// Uses toroidal world-anchored spatial hashing to eliminate temporal flashing, index-shifting artifacts,
// and camera-motion flicker while providing O(1) multi-scattered ambient and emissive lookups.

import world_common;
import world_core;
import bindings;
import sky;
import math;
import trace;
import lighting;

public bool wrcEnabled() {
    return (uint(worldPush.wrcDims.w) & 1u) != 0u;
}

public bool wrcHistoryValid() {
    return (uint(worldPush.wrcDims.w) & 2u) != 0u;
}

public int3 wrcDimensions() {
    return worldPush.wrcDims.xyz;
}

public float wrcCellSize() {
    return max(worldPush.wrcOrigin.w, 0.25);
}

/** Toroidally wraps an unbounded 3D integer world cell into the 3D storage texture dimensions. */
public int3 wrcToroidalTexel(int3 worldCell, int3 dims) {
    int3 m = worldCell % dims;
    return (m + dims) % dims;
}

/** Reconstruct continuous world-space coordinate from rebased shader position. */
public float3 wrcToWorldPos(float3 rebasedPos) {
    return rebasedPos + worldPush.fogWorld.xyz;
}

/** Convert continuous world-space coordinate back to rebased shader position. */
public float3 wrcToRebasedPos(float3 worldPos) {
    return worldPos - worldPush.fogWorld.xyz;
}

/** Safe fetch of one toroidally addressed world cell. */
public float4 wrcLoadWorldCell(int3 worldCell) {
    int3 dims = wrcDimensions();
    int3 texel = wrcToroidalTexel(worldCell, dims);
    return worldRadianceCache[texel];
}

/**
 * Continuous trilinear lookup of cached irradiance in world space.
 * Because interpolation is anchored in absolute world coordinates, moving or rotating the camera
 * produces ZERO flickering, flashing, or popping.
 */
public float3 sampleWorldRadianceCache(float3 rebasedPos) {
    if (!wrcEnabled()) {
        return float3(0.0);
    }

    float3 worldPos = wrcToWorldPos(rebasedPos);
    float cellSize = wrcCellSize();
    int3 dims = wrcDimensions();

    // Check distance to player to smoothly fade out at the outer boundary of the active clip volume
    float3 camWorld = worldPush.camOffset + worldPush.fogWorld.xyz;
    float maxSpan = float(dims.x) * cellSize * 0.48;
    float distToCam = length(worldPos - camWorld);
    if (distToCam > maxSpan) {
        return float3(0.0);
    }

    float3 normalizedCoord = worldPos / cellSize - 0.5;
    int3 lo = int3(floor(normalizedCoord));
    int3 hi = lo + int3(1);
    float3 f = frac(normalizedCoord);

    float4 c000 = wrcLoadWorldCell(int3(lo.x, lo.y, lo.z));
    float4 c100 = wrcLoadWorldCell(int3(hi.x, lo.y, lo.z));
    float4 c010 = wrcLoadWorldCell(int3(lo.x, hi.y, lo.z));
    float4 c110 = wrcLoadWorldCell(int3(hi.x, hi.y, lo.z));
    float4 c001 = wrcLoadWorldCell(int3(lo.x, lo.y, hi.z));
    float4 c101 = wrcLoadWorldCell(int3(hi.x, lo.y, hi.z));
    float4 c011 = wrcLoadWorldCell(int3(lo.x, hi.y, hi.z));
    float4 c111 = wrcLoadWorldCell(int3(hi.x, hi.y, hi.z));

    float4 interpolated = lerp(
            lerp(lerp(c000, c100, f.x), lerp(c010, c110, f.x), f.y),
            lerp(lerp(c001, c101, f.x), lerp(c011, c111, f.x), f.y),
            f.z);

    // Smooth boundary falloff near maximum cache distance
    float edgeFade = saturate((maxSpan - distToCam) / max(cellSize * 2.0, 1.0));
    return max(interpolated.rgb, float3(0.0)) * edgeFade;
}

/**
 * Directional irradiance lookup for surface shading with normal-guided probe offset.
 */
public float3 sampleWorldRadianceCacheWithDirection(float3 p, float3 normal) {
    float3 samplePos = p + normal * (wrcCellSize() * 0.45);
    return sampleWorldRadianceCache(samplePos);
}

/**
 * Evaluates incoming celestial, sky, and emissive irradiance for one toroidally assigned world cell.
 */
public float4 evaluateWrcCell(int3 texelIndex) {
    int3 dims = wrcDimensions();
    float cellSize = wrcCellSize();

    // Map texelIndex to the unique world cell closest to the current camera position
    float3 camWorld = worldPush.camOffset + worldPush.fogWorld.xyz;
    int3 camWorldCell = int3(floor(camWorld / cellSize));
    int3 offset = ((texelIndex - (camWorldCell % dims) + dims + dims / 2) % dims) - dims / 2;
    int3 targetWorldCell = camWorldCell + offset;

    float3 probeWorldPos = (float3(targetWorldCell) + 0.5) * cellSize;
    float3 probeRebasedPos = wrcToRebasedPos(probeWorldPos);

    uint seed = (uint(targetWorldCell.x) * 73856093u ^ uint(targetWorldCell.y) * 19349663u ^ uint(targetWorldCell.z) * 83492791u)
            ^ (worldPush.frameIndex * 26699u);
    seed = pcg(seed);
    uint proposalSeed = pcg(seed);

    // 1. Direct celestial illumination (Sun / Moon)
    float3 celestialTerm = float3(0.0);
    if (max(celestialLight.illuminance.r,
            max(celestialLight.illuminance.g, celestialLight.illuminance.b)) > 1.0e-6) {
        float3 lightDirection = celestialLight.dir;
        VisibilityResult shadow = visibility(probeRebasedPos, lightDirection, 10000.0);
        celestialTerm = celestialLight.illuminance * shadow.transmittance * INV_PI;
    }

    // 2. Multi-scattered ambient sky environment radiance
    SkyState state = skyState(worldPush);
    float2 sunUv = skyViewLutUv(state.viewerRadiusKm, 0.5,
            lightViewCosine(float3(0.0, 1.0, 0.0), state.sunDir), SKY_VIEW_BODY_SUN);
    float3 skyBt709 = skyViewLut.SampleLevel(sunUv, 0.0).rgb + state.airglowLuminance;
    float3 skyRadiance = bt709ToAcesCg(max(skyBt709, float3(0.0))) * 0.15;

    // 3. Local block emitter lighting from the Light Grid
    float3 emissiveTerm = float3(0.0);
    if (worldPush.lightCount > 0u && pc.lightBufAddr != 0 && pc.lightLocalAliasAddr != 0) {
        LightGridCell gridCell;
        int3 gridCellCoord;
        if (findLightGridCell(probeRebasedPos, gridCell, gridCellCoord) && gridCell.invWeightSum > 0.0) {
            uint candidates = min(4u, worldPush.lightCount);
            for (uint c = 0u; c < candidates; ++c) {
                uint lightIndex;
                selectLightGridSpanLight(gridCell, proposalSeed, lightIndex);
                Light light = ConstPtr<Light>(pc.lightBufAddr)[lightIndex];
                float3 emission = lightRadiance(light);
                float area = lightArea(light);
                float3 lightPos = light.pos + worldPush.lightRebase.xyz;
                float3 toLight = lightPos - probeRebasedPos;
                float dist2 = dot(toLight, toLight);
                if (dist2 > 1.0e-4) {
                    float dist = sqrt(dist2);
                    float3 wi = toLight / dist;
                    float3 lnrm = lightGeometricNormal(light);
                    // Hemispherical emission factor so torch points radiate in 360 degrees
                    float cosL = max(0.35, dot(lnrm, -wi));
                    float3 vis = visibility(probeRebasedPos, wi, dist * 0.99).transmittance;
                    emissiveTerm += emission * (cosL * area * INV_PI / max(dist2, 0.25)) * vis;
                }
            }
            emissiveTerm /= float(candidates);
        }
    }

    float3 totalRadiance = celestialTerm + skyRadiance + emissiveTerm;
    float4 current = float4(min(totalRadiance, float3(HALF_MAX)), 1.0);

    // Spatially stable temporal accumulation in world space
    if (wrcHistoryValid()) {
        float4 previous = worldRadianceCache[texelIndex];
        current.rgb = lerp(current.rgb, previous.rgb, 0.85);
    }

    return current;
}
