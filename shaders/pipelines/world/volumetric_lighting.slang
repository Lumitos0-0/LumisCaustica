// Deterministic single-scattering injection for the frustum-froxel grid. Sun/moon visibility is traced
// through the frame TLAS. Colored block emissives reuse Caustica's existing area-light hierarchy; local
// proposals are restricted to the five-section neighborhood and clamped before storage so a low-PDF or
// near-emitter sample cannot become a firefly. Injection has no frame-varying random sequence; a later
// pass may apply only a low, depth-validated temporal weight.

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
    public float M;
};

public VolumeReservoir emptyVolumeReservoir() {
    VolumeReservoir reservoir;
    reservoir.samplePosition = float3(0.0);
    reservoir.contribution = float3(0.0);
    reservoir.weightSum = 0.0;
    reservoir.target = 0.0;
    reservoir.M = 0.0;
    return reservoir;
}

public float3 clampLuminance(float3 value, float maximum) {
    float valueLuminance = luminance(max(value, float3(0.0)));
    return valueLuminance > maximum && maximum > 0.0
            ? value * (maximum / valueLuminance) : value;
}

/** Unoccluded incoming radiance from one uniformly sampled emitter rectangle. */
public float3 evaluateVolumeLight(float3 samplePosition, float3 lightPosition, float3 lightNormal,
                                  float3 emission, float area, float3 viewRay) {
    float3 toLight = lightPosition - samplePosition;
    float distance2 = dot(toLight, toLight);
    if (distance2 <= 1.0e-5 || area <= 0.0) {
        return float3(0.0);
    }
    float3 lightDirection = toLight * rsqrt(distance2);
    float emitterCosine = max(0.0, dot(lightNormal, -lightDirection));
    // A point approximation to a finite rectangle otherwise diverges as 1/r² close to its surface.
    // Irradiance from a one-sided diffuse emitter is bounded by pi*Le, so cap its solid-angle factor at pi.
    float solidAngle = min(emitterCosine * area / distance2, PI);
    float phase = hg(dot(lightDirection, viewRay), worldPush.fogOptics.w);
    return emission * (solidAngle * phase);
}

/**
 * Stable RIS estimate of nearby block-emitter radiance scattered toward the camera. The proposal stream
 * is derived from quantized world position rather than frame index, and global proposals are excluded:
 * distant lights have negligible fog contribution but can carry extreme inverse-PDF weights.
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
        // The sequence is deterministic for a world-space sample. Sampling the finite rectangle avoids
        // hard point-light shadow edges while the solid-angle cap above removes the near-light singularity.
        float3 rectanglePoint = light.pos + worldPush.lightRebase.xyz
                + (rndf(sampleSeed) * 2.0 - 1.0) * lightHalfU(light)
                + (rndf(sampleSeed) * 2.0 - 1.0) * lightHalfV(light);
        float3 contribution = evaluateVolumeLight(samplePosition, rectanglePoint,
                lightGeometricNormal(light), emission, area, viewRay);
        float target = luminance(contribution);
        if (target <= 0.0) {
            continue;
        }
        float sourcePdf = proposalPdf(light, emission, gridCell, gridCellCoord, 1.0);
        // The estimator is deliberately bounded rather than mathematically heroic: a degenerate low-PDF
        // record must not poison an fp16 volume or turn one froxel into a persistent bright voxel.
        float candidateWeight = min(target / max(sourcePdf, 1.0e-8), 1.0e8);
        reservoir.weightSum = min(reservoir.weightSum + candidateWeight, 1.0e9);
        if (rndf(sampleSeed) * reservoir.weightSum < candidateWeight) {
            reservoir.samplePosition = rectanglePoint;
            reservoir.contribution = contribution;
            reservoir.target = target;
        }
    }

    if (reservoir.target <= 0.0 || reservoir.M <= 0.0) {
        return float3(0.0);
    }
    float weight = reservoir.weightSum / (reservoir.M * reservoir.target);
    float3 toLight = reservoir.samplePosition - samplePosition;
    float distance = length(toLight);
    if (distance <= RAY_TMIN) {
        return float3(0.0);
    }
    float3 visibilityToLight = visibility(samplePosition, toLight / distance,
            max(distance * 0.999, RAY_TMIN)).transmittance;
    return max(reservoir.contribution * weight * visibilityToLight, float3(0.0));
}

public float3 froxelDirectionalLighting(float3 samplePosition, float3 viewRay,
                                         out float3 skyVisibility) {
    float3 lightDirection = celestialLight.dir;
    VisibilityResult shadow = visibility(samplePosition, lightDirection, 10000.0);
    skyVisibility = shadow.transmittance;
    float3 visibleIlluminance = celestialLight.illuminance * skyVisibility;
    return visibleIlluminance * hg(dot(lightDirection, viewRay), worldPush.fogLighting.y);
}

/** Hash a quantized world position into a repeatable proposal stream. */
public uint froxelWorldSeed(float3 samplePosition, uint salt) {
    int3 q = int3(floor((samplePosition + worldPush.fogWorld.xyz) * 2.0));
    uint seed = uint(q.x) * 1973u ^ uint(q.y) * 9277u ^ uint(q.z) * 10159u ^ salt;
    return pcg(seed);
}

public float4 injectFroxel(uint3 index) {
    // Stable cell centres avoid view-dependent sub-cell noise. The density itself remains world anchored,
    // while trilinear reconstruction and the 3x3 bilateral pass hide the camera-frustum cell boundaries.
    float2 cellCoordinate = float2(index.xy) + 0.5;
    float nearDistance = froxelBoundaryDistance(float(index.z));
    float farDistance = froxelBoundaryDistance(float(index.z + 1u));
    float opaqueDistance = froxelOpaqueDistance(cellCoordinate);
    // Never evaluate behind the first primary interface. In the terminal intersected cell, move the
    // representative point to the centre of only its visible portion instead of discarding that portion.
    if (nearDistance >= opaqueDistance) {
        return float4(0.0);
    }
    float distance = 0.5 * (nearDistance + min(farDistance, opaqueDistance));
    float3 rayOrigin;
    float3 viewRay;
    froxelRay(cellCoordinate, rayOrigin, viewRay);
    float3 samplePosition = rayOrigin + viewRay * distance;

    float extinction = froxelExtinction(samplePosition);
    if (extinction <= 1.0e-7) {
        return float4(0.0);
    }

    SkyState state = skyState(worldPush);
    float3 skyVisibility;
    float3 directional = worldPush.fogLighting.x
            * froxelDirectionalLighting(samplePosition, viewRay, skyVisibility);
    // The broad atmosphere term is low-frequency and shares the directional visibility as a conservative
    // cave mask. It is intentionally configurable independently from density and direct shafts.
    float3 ambient = worldPush.fogShape.w * skyVisibility
            * (0.5 * froxelSkyRadiance(viewRay, state)
                    + 0.5 * froxelSkyRadiance(float3(0.0, 1.0, 0.0), state));

    float3 emitter = float3(0.0);
    if (worldPush.fogWorld.w > 0.0) {
        uint sampleCount = froxelEmitterSamples();
        uint baseSeed = froxelWorldSeed(samplePosition, 0x85ebca6bu);
        uint baseProposalSeed = froxelWorldSeed(samplePosition, 0xc2b2ae35u);
        for (uint lightSample = 0u; lightSample < sampleCount; ++lightSample) {
            uint sampleSeed = baseSeed ^ (lightSample + 1u) * 0x9e3779b9u;
            uint proposalSeed = baseProposalSeed ^ (lightSample + 1u) * 0x7f4a7c15u;
            sampleSeed = pcg(sampleSeed);
            proposalSeed = pcg(proposalSeed);
            emitter += froxelEmitterLighting(samplePosition, viewRay, sampleSeed, proposalSeed);
        }
        emitter *= worldPush.fogWorld.w / float(sampleCount);
    }

    float3 scatteringCoefficient = extinction * clamp(worldPush.fogOptics.rgb, 0.0, 1.0);
    float3 localSource = scatteringCoefficient * emitter;
    localSource = clampLuminance(localSource, worldPush.fogLighting.z);
    float3 source = scatteringCoefficient * (ambient + directional) + localSource;
    return min(float4(max(source, float3(0.0)), extinction), float4(HALF_MAX));
}
