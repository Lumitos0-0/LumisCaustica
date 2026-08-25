// Single-scattering light injection for the froxel grid. Directional illumination uses the same
// atmosphere-derived sun/moon light as surface NEE and traces through the same TLAS. Emissive blocks are
// selected from the existing power-weighted light hierarchy, so volumetrics add no duplicate light list.

import world_common;
import world_core;
import bindings;
import sky;
import math;
import medium;
import trace;
import lighting;
import water;
import froxel;

public struct VolumeReservoir {
    public float3 samplePosition;
    public float3 contribution;
    public float weightSum;
    public float target;
    public float weight;
    public float M;        // number of candidates considered (for consistency with surface Reservoir)
};

public VolumeReservoir emptyVolumeReservoir() {
    VolumeReservoir reservoir;
    reservoir.samplePosition = float3(0.0);
    reservoir.contribution = float3(0.0);
    reservoir.weightSum = 0.0;
    reservoir.target = 0.0;
    reservoir.weight = 0.0;
    reservoir.M = 0.0;
    return reservoir;
}

/** Unoccluded incoming radiance from one uniformly sampled emitter rectangle. */
public float3 evaluateVolumeLight(float3 samplePosition, float3 lightPosition, float3 lightNormal,
                                  float3 emission, float area, float3 viewRay) {
    float3 toLight = lightPosition - samplePosition;
    float distance2 = dot(toLight, toLight);
    if (distance2 <= 1.0e-5) {
        return float3(0.0);
    }
    float3 lightDirection = toLight * rsqrt(distance2);
    float emitterCosine = max(0.0, dot(lightNormal, -lightDirection));
    float phase = hg(dot(lightDirection, viewRay), worldPush.fogOptics.w);
    return emission * (emitterCosine * area * phase / distance2);
}

/**
 * RIS estimate of nearby block-emitter radiance scattered toward the camera, with one visibility ray.
 * Volumetric emitters deliberately use only the light-grid neighborhood: a distant global proposal has
 * vanishing contribution but a very large inverse-PDF weight, which appears as isolated outdoor fireflies
 * at one sample per froxel. The five-section local span covers every emitter that can visibly illuminate
 * Minecraft-scale fog while giving the estimator a bounded, stable domain.
 *
 * Now stores M (candidate count) in the reservoir for future extensibility / debugging.
 */
public float3 froxelEmitterLighting(float3 samplePosition, float3 viewRay,
                                    inout uint sampleSeed, inout uint proposalSeed) {
    uint candidateCount = min(froxelLocalLightCandidates(), worldPush.lightCount);
    if (candidateCount == 0u || pc.lightBufAddr == 0 || pc.lightLocalAliasAddr == 0) {
        return float3(0.0);
    }

    LightGridCell gridCell;
    int3 gridCellCoord;
    if (!findLightGridCell(samplePosition, gridCell, gridCellCoord)
            || gridCell.invWeightSum <= 0.0) {
        return float3(0.0);
    }
    VolumeReservoir reservoir = emptyVolumeReservoir();
    reservoir.M = float(candidateCount);

    for (uint candidate = 0u; candidate < candidateCount; ++candidate) {
        uint lightIndex;
        selectLightGridSpanLight(gridCell, proposalSeed, lightIndex);
        Light light = ConstPtr<Light>(pc.lightBufAddr)[lightIndex];
        float3 emission = lightRadiance(light);
        float area = lightArea(light);
        float3 rectanglePoint = light.pos + worldPush.lightRebase.xyz
                + (rndf(sampleSeed) * 2.0 - 1.0) * lightHalfU(light)
                + (rndf(sampleSeed) * 2.0 - 1.0) * lightHalfV(light);
        float3 contribution = evaluateVolumeLight(samplePosition, rectanglePoint,
                lightGeometricNormal(light), emission, area, viewRay);
        float target = luminance(contribution);
        if (target <= 0.0) {
            continue;
        }
        // localProbability=1 reconstructs the exact weighted neighborhood proposal PDF.
        float sourcePdf = proposalPdf(light, emission, gridCell, gridCellCoord, 1.0);
        float candidateWeight = target / max(sourcePdf, 1.0e-20);
        reservoir.weightSum += candidateWeight;
        if (rndf(sampleSeed) * reservoir.weightSum < candidateWeight) {
            reservoir.samplePosition = rectanglePoint;
            reservoir.contribution = contribution;
            reservoir.target = target;
        }
    }

    if (reservoir.target <= 0.0) {
        return float3(0.0);
    }
    // Improved weight: consistent with surface RIS W = wSum / (M * p-hat)
    reservoir.weight = reservoir.weightSum / max(reservoir.M * reservoir.target, 1.0e-20);
    float3 toLight = reservoir.samplePosition - samplePosition;
    float distance = length(toLight);
    if (distance <= RAY_TMIN) {
        return float3(0.0);
    }
    float3 visibilityToLight = visibility(samplePosition, toLight / distance, distance * 0.999).transmittance;
    return reservoir.contribution * reservoir.weight * visibilityToLight;
}

public float3 froxelDirectionalLighting(float3 samplePosition, float3 viewRay, uint2 cellIndex,
                                         out float3 skyVisibility) {
    skyVisibility = float3(0.0);
    if (max(celestialLight.illuminance.r,
            max(celestialLight.illuminance.g, celestialLight.illuminance.b)) <= 1.0e-6) {
        return float3(0.0);
    }
    // Two stratified visibility samples per froxel instead of a single binary point test. The old build's
    // single ray produced hard lit/unlit cells at silhouettes -> intense outline shimmer. Averaging two
    // decorrelated offsets integrates visibility over the cell footprint. The offsets are a stable
    // low-discrepancy pattern from the grid index (world position would alias at large block coords) and
    // do not depend on frameIndex, so the volume stays deterministic.
    float3 lightDirection = celestialLight.dir;
    float2 cellHash = frac(float2(float(cellIndex.x) * 0.7548776662, float(cellIndex.y) * 0.5698402909));
    float2 vJitter = (cellHash * 2.0 - 1.0) * 0.5;
    float3 shadowOffset = float3(vJitter.x, vJitter.y, 0.0) * 0.5;

    VisibilityResult shadow1 = visibility(samplePosition + shadowOffset, lightDirection, 10000.0);
    VisibilityResult shadow2 = visibility(samplePosition - shadowOffset, lightDirection, 10000.0);

    float3 transmittance = (shadow1.transmittance + shadow2.transmittance) * 0.5;
    skyVisibility = transmittance;

    float causticFocus = 1.0;
    if (froxelUnderwater() && (shadow1.waterHitT > 0.0 || shadow2.waterHitT > 0.0)) {
        float avgWaterT = (shadow1.waterHitT + shadow2.waterHitT) * 0.5;
        // The camera path already accounts for water absorption. This is the independent light-to-sample
        // path from the surface, attenuated through water before scattering toward the viewer.
        skyVisibility *= exp(-waterExtinction(worldPush.waterParams.xyz) * avgWaterT);
        if ((worldPush.flags & 16u) != 0u && worldPush.fogLighting.z > 0.0) {
            float3 surfacePosition = samplePosition + lightDirection * avgWaterT;
            float pattern = waterCaustic(surfacePosition, lightDirection, avgWaterT);
            causticFocus = max(0.0, lerp(1.0, pattern, worldPush.fogLighting.z));
        }
    }
    float3 visibleIlluminance = celestialLight.illuminance * skyVisibility * causticFocus;
    return visibleIlluminance * hg(dot(lightDirection, viewRay), worldPush.fogLighting.y);
}

public float3 froxelScatteringColor() {
    if (!froxelUnderwater()) {
        return worldPush.fogOptics.rgb;
    }
    float3 biomeTint = max(worldPush.waterParams.xyz, float3(1.0e-4));
    biomeTint /= max(biomeTint.r, max(biomeTint.g, biomeTint.b));
    // Preserve biome identity while maintaining the cyan-blue particulate scattering expected underwater.
    return lerp(float3(0.25, 0.65, 1.0), biomeTint, 0.35);
}

public float3 clampFroxelSource(float3 source) {
    // Keeps an isolated emitter sample below the range that becomes a visible dot after temporal reuse;
    // direct forward-scattered noon sunlight remains well below this ceiling.
    static const float MAX_SOURCE_LUMINANCE = 250.0;
    float sourceLuminance = luminance(source);
    return sourceLuminance > MAX_SOURCE_LUMINANCE
            ? source * (MAX_SOURCE_LUMINANCE / sourceLuminance) : source;
}

public float4 injectFroxel(uint3 index) {
    // Deterministic, frame-independent seed: the froxel volume must not change from frame to frame for a
    // static scene, otherwise any stochastic term re-rolls every frame and (in the old temporal world)
    // produces boil. No frameIndex, no frame-varying jitter.
    uint seed = index.x * 1973u + index.y * 9277u + index.z * 10159u;
    seed = pcg(seed);
    uint proposalSeed = pcg(seed);

    // Sample at the deterministic froxel CENTRE (no per-frame jitter). This is what made the prior clean
    // build boil-free: the volume is a pure function of world position, so it is identical every frame
    // and needs no temporal accumulation. Depth is used only to gate local emitters behind the first
    // camera surface (leak prevention).
    float2 cellCoordinate = float2(index.xy) + 0.5;
    float distance = froxelBoundaryDistance(float(index.z) + 0.5);
    float3 rayOrigin;
    float3 viewRay;
    froxelRay(cellCoordinate, rayOrigin, viewRay);
    float3 samplePosition = rayOrigin + viewRay * distance;
    bool cameraVisibleSample = distance <= froxelOpaqueDistance(cellCoordinate);

    float extinction = froxelExtinction(samplePosition);
    if (extinction <= 1.0e-7) {
        return float4(0.0);
    }
    SkyState state = skyState(worldPush);
    // A cheap low-frequency approximation to the hemispherical sky integral. It deliberately has no
    // visibility ray: terrain occlusion is supplied by the directional and local terms, while this term
    // represents the broad multiple-scattered environment already baked into the sky LUT.
    float3 ambient = 0.12 * (0.5 * froxelSkyRadiance(viewRay, state)
            + 0.5 * froxelSkyRadiance(float3(0.0, 1.0, 0.0), state));
    float3 skyVisibility;
    float3 directional = worldPush.fogLighting.x
            * froxelDirectionalLighting(samplePosition, viewRay, index.xy, skyVisibility);

    // Leak-prevention policy (improved):
    // - Local emitters: gated behind first surface (prevents fireflies from hidden lava/torches).
    // - Directional: always evaluated (its visibility ray already respects geometry).
    // - Ambient: always evaluated (low-frequency sky contribution).
    // This keeps the volume continuously populated for filtering while eliminating the original
    // "sample behind surface" artifact for stochastic local lights.
    float3 emitter = float3(0.0);
    if (cameraVisibleSample) {
        // Deterministic multi-reservoir estimator, seeded from the froxel index (no frameIndex), so the
        // emissive light choice is stable in space across frames. Gated behind the first surface so hidden
        // lava/torch emitters don't leak into visible cells.
        uint sampleCount = froxelEmitterSamples();
        for (uint lightSample = 0u; lightSample < sampleCount; ++lightSample) {
            uint emitterSeed = seed ^ (lightSample + 1u) * 0x85ebca6bu;
            uint emitterProposalSeed = proposalSeed ^ (lightSample + 1u) * 0xc2b2ae35u;
            emitterSeed = pcg(emitterSeed);
            emitterProposalSeed = pcg(emitterProposalSeed);
            emitter += froxelEmitterLighting(
                    samplePosition, viewRay, emitterSeed, emitterProposalSeed);
        }
        emitter /= float(sampleCount);
    }
    float3 incident = ambient * skyVisibility + directional + emitter;
    float3 source = clampFroxelSource(extinction * froxelScatteringColor() * incident);
    // Deterministic {in-scattering, extinction}. There is deliberately NO temporal accumulation: this
    // volume is recomputed identically each frame for a static scene, so it needs no history/reprojection
    // and cannot boil/ghost/lag. The integrate pass raymarches it per-pixel with a stable spatial dither.
    return min(float4(max(source, float3(0.0)), extinction), float4(HALF_MAX));
}
