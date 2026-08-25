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
    float phase = dualLobeHg(dot(lightDirection, viewRay), worldPush.fogOptics.w, -0.15, 0.80);
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
    // A single randomized angular visibility ray turns foliage edges into temporal salt-and-pepper noise.
    // The deterministic body-centre ray is coherent; the existing Gaussian/temporal passes soften it at
    // the volume's intended frequency without changing surface-light penumbrae.
    float3 lightDirection = celestialLight.dir;
    VisibilityResult shadow = visibility(samplePosition, lightDirection, 10000.0);
    skyVisibility = shadow.transmittance;
    float3 causticFocus = float3(1.0);
    if (froxelUnderwater() && shadow.waterHitT > 0.0) {
        // The camera path already accounts for water absorption. This is the independent light-to-sample
        // path from the surface, attenuated through water before scattering toward the viewer.
        skyVisibility *= exp(-waterExtinction(worldPush.waterParams.xyz) * shadow.waterHitT);
        if ((worldPush.flags & 16u) != 0u && worldPush.fogLighting.z > 0.0) {
            float3 surfacePosition = samplePosition + lightDirection * shadow.waterHitT;
            causticFocus = waterCausticRgb(surfacePosition, lightDirection, shadow.waterHitT, worldPush.fogLighting.z);
        }
    }
    float3 visibleIlluminance = celestialLight.illuminance * skyVisibility * causticFocus;
    float cosTheta = dot(lightDirection, viewRay);
    float phase = dualLobeHg(cosTheta, worldPush.fogLighting.y, -0.22, 0.75);
    return visibleIlluminance * phase;
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
    uint seed = index.x * 1973u + index.y * 9277u + index.z * 10159u
            ^ worldPush.frameIndex * 26699u;
    seed = pcg(seed);
    uint proposalSeed = pcg(seed);

    // Jitter inside the complete froxel so temporal accumulation supersamples one continuous 3D field.
    // Scene depth must not truncate the field itself: doing so creates empty rectangular columns that make
    // the frustum grid visible. Depth is used only below to suppress local emitters sampled behind the
    // first camera surface.
    float2 cellCoordinate = float2(index.xy) + float2(rndf(seed), rndf(seed));
    float nearDistance = froxelBoundaryDistance(float(index.z));
    float farDistance = froxelBoundaryDistance(float(index.z + 1u));
    float distance = lerp(nearDistance, farDistance, rndf(seed));
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
            * froxelDirectionalLighting(samplePosition, viewRay, skyVisibility);
    // The original leak came from a partially occupied Z slice choosing its one light sample behind the
    // first surface (for example inside a lava cave) and applying that result to the visible part of the
    // slice. Suppress only that local-emitter term; keep the volume populated and continuously filterable.
    float3 emitter = float3(0.0);
    if (cameraVisibleSample) {
        // The same multi-reservoir estimator runs in moving and stationary views. Temporal accumulation
        // still refines it, but does not switch the renderer to a different base lighting algorithm.
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
    float4 current = float4(max(source, float3(0.0)), extinction);
    return min(current, float4(HALF_MAX));
}
