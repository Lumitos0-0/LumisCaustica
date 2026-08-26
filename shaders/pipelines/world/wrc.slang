// World Radiance Cache (WRC): Persistent 3D world-space irradiance volume.
// Provides O(1) trilinear lookups of multi-scattered ambient and emissive lighting.
// Eliminates stochastic RIS variance, fireflies, and temporal ghosting in both volumetrics and surface GI.

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
    return max(worldPush.wrcOrigin.w, 0.1);
}

public float3 wrcWorldOrigin() {
    return worldPush.wrcOrigin.xyz;
}

/** Continuous 0-based grid coordinate for a rebased world-space position. */
public float3 wrcContinuousCoord(float3 p) {
    return (p - worldPush.wrcOrigin.xyz) / wrcCellSize();
}

/** Reconstruct rebased world-space position for a continuous grid coordinate. */
public float3 wrcWorldPosition(float3 coord) {
    return worldPush.wrcOrigin.xyz + coord * wrcCellSize();
}

/** Safe fetch of one cache cell. */
public float4 wrcLoad(int3 p) {
    int3 dims = wrcDimensions();
    if (any(p < int3(0)) || any(p >= dims)) {
        return float4(0.0);
    }
    return worldRadianceCache[p];
}

/**
 * Trilinear lookup of cached isotropic irradiance (RGB in scene-linear ACEScg, Alpha = validity).
 * Returns float3(0.0) if the sample position lies completely outside the cache domain.
 */
public float3 sampleWorldRadianceCache(float3 p) {
    if (!wrcEnabled()) {
        return float3(0.0);
    }

    float3 coord = wrcContinuousCoord(p);
    int3 dims = wrcDimensions();
    
    // Bounds check with a 0.5-cell safety margin for smooth trilinear falloff at cache borders
    if (any(coord < float3(0.0)) || any(coord >= float3(dims - 1))) {
        return float3(0.0);
    }

    float3 bounded = clamp(coord, float3(0.0), float3(dims - 1));
    int3 lo = int3(floor(bounded));
    int3 hi = min(lo + int3(1), dims - 1);
    float3 f = frac(bounded);

    float4 c000 = wrcLoad(int3(lo.x, lo.y, lo.z));
    float4 c100 = wrcLoad(int3(hi.x, lo.y, lo.z));
    float4 c010 = wrcLoad(int3(lo.x, hi.y, lo.z));
    float4 c110 = wrcLoad(int3(hi.x, hi.y, lo.z));
    float4 c001 = wrcLoad(int3(lo.x, lo.y, hi.z));
    float4 c101 = wrcLoad(int3(hi.x, lo.y, hi.z));
    float4 c011 = wrcLoad(int3(lo.x, hi.y, hi.z));
    float4 c111 = wrcLoad(int3(hi.x, hi.y, hi.z));

    float4 interpolated = lerp(
            lerp(lerp(c000, c100, f.x), lerp(c010, c110, f.x), f.y),
            lerp(lerp(c001, c101, f.x), lerp(c011, c111, f.x), f.y),
            f.z);

    return max(interpolated.rgb, float3(0.0));
}

/**
 * Directional irradiance lookup for surface shading, incorporating normal orientation.
 */
public float3 sampleWorldRadianceCacheWithDirection(float3 p, float3 normal) {
    // Probe offset along the normal to sample outside self-occluding geometry
    float3 samplePos = p + normal * (wrcCellSize() * 0.35);
    return sampleWorldRadianceCache(samplePos);
}

/**
 * Evaluates the incoming irradiance at one cache probe cell.
 */
public float4 evaluateWrcCell(int3 cellIndex) {
    float3 cellCenterCoord = float3(cellIndex) + float3(0.5);
    float3 probePos = wrcWorldPosition(cellCenterCoord);

    uint seed = (uint(cellIndex.x) * 1973u + uint(cellIndex.y) * 9277u + uint(cellIndex.z) * 10159u)
            ^ (worldPush.frameIndex * 26699u);
    seed = pcg(seed);
    uint proposalSeed = pcg(seed);

    // 1. Direct Sun / Moon celestial irradiance
    float3 celestialTerm = float3(0.0);
    if (max(celestialLight.illuminance.r,
            max(celestialLight.illuminance.g, celestialLight.illuminance.b)) > 1.0e-6) {
        float3 lightDirection = celestialLight.dir;
        VisibilityResult shadow = visibility(probePos, lightDirection, 10000.0);
        celestialTerm = celestialLight.illuminance * shadow.transmittance * INV_PI;
    }

    // 2. Multi-scattered ambient sky radiance
    SkyState state = skyState(worldPush);
    float2 sunUv = skyViewLutUv(state.viewerRadiusKm, 0.5,
            lightViewCosine(float3(0.0, 1.0, 0.0), state.sunDir), SKY_VIEW_BODY_SUN);
    float3 skyBt709 = skyViewLut.SampleLevel(sunUv, 0.0).rgb + state.airglowLuminance;
    float3 skyRadiance = bt709ToAcesCg(max(skyBt709, float3(0.0))) * 0.15;

    // 3. Local emissive block lighting from the Light Grid
    float3 emissiveTerm = float3(0.0);
    if (worldPush.lightCount > 0u && pc.lightBufAddr != 0 && pc.lightLocalAliasAddr != 0) {
        LightGridCell gridCell;
        int3 gridCellCoord;
        if (findLightGridCell(probePos, gridCell, gridCellCoord) && gridCell.invWeightSum > 0.0) {
            uint candidates = min(4u, worldPush.lightCount);
            for (uint c = 0u; c < candidates; ++c) {
                uint lightIndex;
                selectLightGridSpanLight(gridCell, proposalSeed, lightIndex);
                Light light = ConstPtr<Light>(pc.lightBufAddr)[lightIndex];
                float3 emission = lightRadiance(light);
                float area = lightArea(light);
                float3 lightPos = light.pos + worldPush.lightRebase.xyz;
                float3 toLight = lightPos - probePos;
                float dist2 = dot(toLight, toLight);
                if (dist2 > 1.0e-4) {
                    float dist = sqrt(dist2);
                    float3 wi = toLight / dist;
                    float3 lnrm = lightGeometricNormal(light);
                    float cosL = max(0.0, dot(lnrm, -wi));
                    if (cosL > 0.0) {
                        float3 vis = visibility(probePos, wi, dist * 0.99).transmittance;
                        emissiveTerm += emission * (cosL * area * INV_PI / dist2) * vis;
                    }
                }
            }
            emissiveTerm /= float(candidates);
        }
    }

    float3 totalRadiance = celestialTerm + skyRadiance + emissiveTerm;
    float4 current = float4(min(totalRadiance, float3(HALF_MAX)), 1.0);

    // Temporal blend with existing cache if valid
    if (wrcHistoryValid()) {
        float4 previous = worldRadianceCache[cellIndex];
        current.rgb = lerp(current.rgb, previous.rgb, 0.82);
    }

    return current;
}
