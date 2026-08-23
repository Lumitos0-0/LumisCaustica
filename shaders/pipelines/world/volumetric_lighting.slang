// Single-scattering light injection for the froxel grid. Directional illumination uses the same
// atmosphere-derived sun/moon light as surface NEE and traces through the same TLAS. Emissive blocks are
// selected from the existing power-weighted light hierarchy, so volumetrics add no duplicate light list.

import world_common;
import world_core;
import bindings;
import sky;
import math;
import trace;
import lighting;
import froxel;

public struct VolumeReservoir {
    public float3 samplePosition;
    public float3 contribution;
    public float weightSum;
    public float target;
    public float weight;
};

public VolumeReservoir emptyVolumeReservoir() {
    VolumeReservoir reservoir;
    reservoir.samplePosition = float3(0.0);
    reservoir.contribution = float3(0.0);
    reservoir.weightSum = 0.0;
    reservoir.target = 0.0;
    reservoir.weight = 0.0;
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
    reservoir.weight = reservoir.weightSum / (float(candidateCount) * reservoir.target);
    float3 toLight = reservoir.samplePosition - samplePosition;
    float distance = length(toLight);
    if (distance <= RAY_TMIN) {
        return float3(0.0);
    }
    float3 visibilityToLight = visibility(samplePosition, toLight / distance, distance * 0.999).transmittance;
    return reservoir.contribution * reservoir.weight * visibilityToLight;
}

public float3 froxelDirectionalLighting(float3 samplePosition, float3 viewRay,
                                         out float3 skyVisibility) {
    skyVisibility = float3(0.0);
    if (max(celestialLight.illuminance.r,
            max(celestialLight.illuminance.g, celestialLight.illuminance.b)) <= 1.0e-6) {
        return float3(0.0);
    }
    // Surface paths sample the finite sun/moon square stochastically. A one-ray froxel cannot do that
    // without turning every visibility boundary into a bright/dark Monte Carlo cell. Trace the body centre
    // here and let the depth-aware spatial kernel provide the low-frequency volumetric penumbra.
    float3 lightDirection = celestialLight.dir;
    skyVisibility = visibility(samplePosition, lightDirection, 10000.0).transmittance;
    float3 visibleIlluminance = celestialLight.illuminance * skyVisibility;
    return visibleIlluminance * hg(dot(lightDirection, viewRay), worldPush.fogOptics.w);
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
    uint seed = index.x * 1973u + index.y * 9277u + index.z * 10159u
            ^ worldPush.frameIndex * 26699u;
    seed = pcg(seed);

    float2 cellCoordinate;
    float minColumnDepth;
    float maxColumnDepth;
    froxelBackgroundDepth(int2(index.xy), cellCoordinate, minColumnDepth, maxColumnDepth);
    float depthEdgeThreshold = max(1.0, minColumnDepth * 0.08);
    bool depthEdge = maxColumnDepth - minColumnDepth > depthEdgeThreshold;
    float nearDistance = froxelBoundaryDistance(float(index.z));
    float farDistance = froxelBoundaryDistance(float(index.z + 1u));
    // A large nonlinear slice can straddle the first opaque surface. Sampling its unconstrained midpoint
    // may land underground, cache a lava emitter there, and then apply that sample to the visible part of
    // the slice in front of the ground. The farthest of five stable tile probes preserves background fog
    // at silhouettes while broad opaque ground still clips every underground slice.
    float visibleFarDistance = min(farDistance, maxColumnDepth);
    if (visibleFarDistance <= nearDistance) {
        return float4(0.0);
    }
    // Edge cells have no temporal history by design, so a frame-varying Z sample would remain visible as
    // raw noise. Use their segment midpoint; smooth cells keep jitter and temporal supersampling.
    float distanceFraction = depthEdge ? 0.5 : rndf(seed);
    float distance = lerp(nearDistance, visibleFarDistance, distanceFraction);
    float3 rayOrigin;
    float3 viewRay;
    froxelRay(cellCoordinate, rayOrigin, viewRay);
    float3 samplePosition = rayOrigin + viewRay * distance;

    // Block emitters use a world-quantized, frame-independent stream. Their selected light and rectangle
    // point therefore do not reshuffle every frame; the spatial filter handles the remaining estimator
    // variation without requiring long history that would trail during camera motion.
    float stableDistance = 0.5 * (nearDistance + visibleFarDistance);
    float3 emitterPosition = rayOrigin + viewRay * stableDistance;
    int3 emitterCell = int3(floor(emitterPosition + worldPush.fogWorld.xyz));
    uint emitterSeed = uint(emitterCell.x) * 73856093u
            ^ uint(emitterCell.y) * 19349663u ^ uint(emitterCell.z) * 83492791u;
    uint emitterProposalSeed = emitterSeed ^ 0x9e3779b9u;
    emitterSeed = pcg(emitterSeed);
    emitterProposalSeed = pcg(emitterProposalSeed);

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
    float3 directional = froxelDirectionalLighting(samplePosition, viewRay, skyVisibility);
    float3 incident = ambient * skyVisibility
            + directional
            + froxelEmitterLighting(emitterPosition, viewRay, emitterSeed, emitterProposalSeed);
    float3 source = clampFroxelSource(extinction * worldPush.fogOptics.rgb * incident);
    float4 current = float4(max(source, float3(0.0)), extinction);

    if (froxelHistoryValid()) {
        float4 history;
        if (froxelReprojectHistory(samplePosition, history)) {
            // Bound stale moving-light/shadow history while retaining enough range for stochastic emitter
            // samples. Extinction gets a tighter multiplicative clamp because its world field is smooth.
            history.rgb = min(max(history.rgb, float3(0.0)), current.rgb * 4.0 + float3(8.0));
            history.a = clamp(history.a, current.a * 0.5, current.a * 2.0 + 1.0e-6);
            // A depth-edge cell can switch background representatives during camera motion. Reject its
            // history outright: even a small blend leaves a visible outline trail, while block-emitter
            // sampling is now frame-stable and the spatial pass supplies edge denoising without history.
            // Smooth cells retain history, but translation lowers confidence continuously instead of
            // dragging the full stationary-camera weight through motion.
            float cameraHistoryConfidence = exp(-length(worldPush.camDelta) * 2.0);
            float historyWeight = depthEdge
                    ? 0.0 : worldPush.fogShape.w * cameraHistoryConfidence;
            current = lerp(current, history, historyWeight);
        }
    }
    return min(current, float4(HALF_MAX));
}
