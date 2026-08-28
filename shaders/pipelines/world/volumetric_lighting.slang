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

// R3 low-discrepancy additive recurrence (Roberts): the three-dimensional generalized golden-ratio
// vector, built from the real root of x^4 = x + 1. Advancing a point by this vector each frame fills a
// cell far more evenly than re-drawing an independent sample.
public static const float3 R3_ALPHA = float3(0.8191725134, 0.6710436067, 0.5497004779);
// Long enough that a wrap never falls inside one temporal history window, short enough to keep the
// recurrence's integer part exactly representable in float32.
public static const uint JITTER_CYCLE = 4096u;



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
 * Single-sample NEE estimate of nearby block-emitter radiance scattered toward the camera, with one
 * visibility ray. Volumetric emitters deliberately use only the light-grid neighborhood: a distant global
 * proposal has vanishing contribution but a very large inverse-PDF weight, which appears as isolated
 * outdoor fireflies at one sample per froxel. The five-section local span covers every emitter that can
 * visibly illuminate Minecraft-scale fog while giving the estimator a bounded, stable domain.
 *
 * <p>One proposal, not resampled. Every candidate costs a light fetch plus an alias-table lookup on a
 * grid that runs to millions of cells, and selecting among them by luminance buys little here: the rare
 * bright pick that would suppress variance is also the one that lands as a firefly until the temporal
 * window averages it out. Plain NEE against the grid's own proposal PDF is the estimator without that.
 */
public float3 froxelEmitterLighting(float3 samplePosition, float3 viewRay,
                                    inout uint sampleSeed, inout uint proposalSeed) {
    if (worldPush.lightCount == 0u || pc.lightBufAddr == 0 || pc.lightLocalAliasAddr == 0) {
        return float3(0.0);
    }

    LightGridCell gridCell;
    int3 gridCellCoord;
    if (!findLightGridCell(samplePosition, gridCell, gridCellCoord)
            || gridCell.invWeightSum <= 0.0) {
        return float3(0.0);
    }

    uint lightIndex;
    selectLightGridSpanLight(gridCell, proposalSeed, lightIndex);
    Light light = ConstPtr<Light>(pc.lightBufAddr)[lightIndex];
    float3 emission = lightRadiance(light);
    float3 rectanglePoint = light.pos + worldPush.lightRebase.xyz
            + (rndf(sampleSeed) * 2.0 - 1.0) * lightHalfU(light)
            + (rndf(sampleSeed) * 2.0 - 1.0) * lightHalfV(light);
    float3 contribution = evaluateVolumeLight(samplePosition, rectanglePoint,
            lightGeometricNormal(light), emission, lightArea(light), viewRay);
    if (luminance(contribution) <= 0.0) {
        return float3(0.0);
    }
    // localProbability=1 reconstructs the exact weighted neighborhood proposal PDF. With a single sample
    // the RIS weight collapses to the ordinary Monte Carlo estimator, contribution / pdf.
    float sourcePdf = proposalPdf(light, emission, gridCell, gridCellCoord, 1.0);
    float3 toLight = rectanglePoint - samplePosition;
    float distance = length(toLight);
    if (distance <= RAY_TMIN) {
        return float3(0.0);
    }
    return contribution / max(sourcePdf, 1.0e-20)
            * visibility(samplePosition, toLight / distance, distance * 0.999).transmittance;
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
    float causticFocus = 1.0;
    if (froxelUnderwater() && shadow.waterHitT > 0.0) {
        // The camera path already accounts for water absorption. This is the independent light-to-sample
        // path from the surface, attenuated through water before scattering toward the viewer.
        skyVisibility *= exp(-waterExtinction(worldPush.waterParams.xyz) * shadow.waterHitT);
        if ((worldPush.flags & 16u) != 0u && worldPush.fogLighting.z > 0.0) {
            float3 surfacePosition = samplePosition + lightDirection * shadow.waterHitT;
            float pattern = waterCaustic(surfacePosition, lightDirection, shadow.waterHitT);
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

// How much of the celestial shadow ray the ambient sky term inherits. `skyVisibility` is the sun/moon
// shadow, so it doubles as a crude enclosure test, but at 1.0 it blacks out fog that is merely in the
// sun's shadow under open sky: the shaded side of a hill, under a tree, the north face of a building.
// 0.75 keeps enclosed volumes falling off without losing ambient entirely.
public static const float AMBIENT_OCCLUSION = 0.75;

// Bounds the STOCHASTIC emitter term only, in scene radiance. A single light sample can draw a very
// large inverse-PDF weight, and one such sample survives temporal reuse as a visible dot.
//
// The celestial term must not share this ceiling. It is deterministic (one fixed shadow ray), so it
// produces no outliers, and clamping the combined `extinction * albedo * incident` instead made the
// integrated radiance approach `ceiling / extinction`: sun shafts got DIMMER as fog density rose, and
// directional-strength stopped working because it is applied before the clamp. Clamping incident
// radiance keeps the result monotone in extinction and leaves sun glare to exposure, bloom, and the
// display transform, which is what they are there for.
//
// 70000 keeps the previous emitter appearance at the default extinction: the old ceiling was 250 on
// `extinction * albedo * incident`, i.e. ~70000 on incident at extinction 0.0035, albedo 0.92.
public float3 clampFroxelEmitter(float3 emitter) {
    static const float MAX_EMITTER_LUMINANCE = 70000.0;
    float emitterLuminance = luminance(emitter);
    return emitterLuminance > MAX_EMITTER_LUMINANCE
            ? emitter * (MAX_EMITTER_LUMINANCE / emitterLuminance) : emitter;
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
    //
    // Each froxel walks its own low-discrepancy path through its cell rather than re-drawing an
    // independent point every frame: an R3 additive recurrence offset per froxel by a Cranley-Patterson
    // rotation. Over the ~20 frames the exponential history actually keeps, that covers the cell far more
    // evenly than a hash, so the blend converges on the froxel's true average instead of random-walking
    // around it. The per-froxel offset keeps neighbouring cells from swimming in lockstep, which the
    // spatial filter depends on. The emitter seeds above stay per-frame random on purpose: they select
    // from a discrete light list, and temporal averaging is what refines that choice.
    uint jitterSeed = index.x * 1973u + index.y * 9277u + index.z * 10159u;
    jitterSeed = pcg(jitterSeed);
    float3 jitterOffset; // one statement per component: rndf advances the seed, and argument
    jitterOffset.x = rndf(jitterSeed); // evaluation order is not specified within one expression
    jitterOffset.y = rndf(jitterSeed);
    jitterOffset.z = rndf(jitterSeed);
    float3 jitter = frac(jitterOffset + float(worldPush.frameIndex % JITTER_CYCLE) * R3_ALPHA);

    float2 cellCoordinate = float2(index.xy) + jitter.xy;
    float nearDistance = froxelBoundaryDistance(float(index.z));
    float farDistance = froxelBoundaryDistance(float(index.z + 1u));
    float distance = lerp(nearDistance, farDistance, jitter.z);
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
    // A cheap low-frequency approximation to the hemispherical sky integral. It has no visibility ray of
    // its own; this term represents the broad multiple-scattered environment already baked into the sky
    // LUT, and it borrows the celestial shadow ray below as a partial enclosure term.
    float3 ambient = 0.12 * (0.5 * froxelSkyRadiance(viewRay, state)
            + 0.5 * froxelSkyRadiance(float3(0.0, 1.0, 0.0), state));
    // Occluded samples get no celestial light, for the same reason the emitter term below is gated: a
    // shadow ray born inside or behind geometry culls back faces, so it can escape to the sky and report
    // full visibility, and that radiance reaches visible pixels through the XY filter and through the
    // trilinear Z reconstruction, which deliberately reaches one cell past the first surface. Gating
    // also stops every froxel behind the world from tracing a full-length shadow ray. The cell stays
    // populated with extinction and residual ambient so the field remains continuous and filterable.
    float3 skyVisibility = float3(0.0);
    float3 directional = float3(0.0);
    if (cameraVisibleSample) {
        directional = worldPush.fogLighting.x
                * froxelDirectionalLighting(samplePosition, viewRay, skyVisibility);
    }
    // The original leak came from a partially occupied Z slice choosing its one light sample behind the
    // first surface (for example inside a lava cave) and applying that result to the visible part of the
    // slice. Suppress only that local-emitter term; keep the volume populated and continuously filterable.
    float3 emitter = float3(0.0);
    if (cameraVisibleSample) {
        emitter = froxelEmitterLighting(samplePosition, viewRay, seed, proposalSeed);
    }
    float3 ambientTerm = ambient * lerp(float3(1.0), skyVisibility, AMBIENT_OCCLUSION);
    float3 incident = ambientTerm + directional + clampFroxelEmitter(emitter);
    float3 source = extinction * froxelScatteringColor() * incident;
    // One raw stochastic sample of this froxel's slice. The resolve that owns the spatial gather, the
    // reprojection, the history clip and the blend lives in volumetric_filter.rgen.slang, which can see
    // the neighbouring froxels this pass cannot.
    return min(float4(max(source, float3(0.0)), extinction), float4(HALF_MAX));
}
