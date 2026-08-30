// Froxel light + extinction injection. One raygen thread per lighting-probe cell (the probe grid is
// decimated relative to the fog grid — see froxelInjectDimensions; DispatchRaysDimensions is that 3D
// grid). Each cell derives its camera-ray position, evaluates the world-stable density field, then
// lightens the single-scattering term with three sources:
//
//   directional — sun/moon through the same weather-independent path-traced visibility() the surface
//                 NEE uses, so tinted glass/water/translucent geometry colors the in-scattered light
//                 and soft shadows break up the shafts;
//   emitters    — next-event estimation over the existing power-weighted block-emitter light grid:
//                 ONE alias-sampled local emitter per probe, one path-traced visibility ray, weighted
//                 by its exact sampling pdf. This is the only place block-emitter light enters the fog
//                 (probe paths never gather direct hits — an HDR hit per probe cell is the fog firefly
//                 source), and the same low-variance idea as the surface NEE for the sun;
//   GI          — the path tracer itself. The in-scatter integral over the remaining radiance is
//                 importance-sampled by the HG phase about the view ray (exact pdf, so the estimator is
//                 just the sampled radiance), and each sample is traced with tracePath() under the
//                 restricted probe path (no surface RIS, no emission gather, no SSS, diffuse-only, no
//                 stars/disc): terrain/canopy GI and skylight reach the fog with the same local shading
//                 model as the image, and the smooth atmosphere LUT enters through the miss shader.
//
// All light-attenuation paths go through visibility(), whose any-hit shaders already accumulate
// per-channel Beer-Lambert transmission (tinted glass, water, translucent blocks, entity dielectrics),
// so fog lit through a red window pane is red — this is the path-traced advantage over shadow maps.
// Probe paths additionally traverse real dielectric volumes (glass, ice) with the same medium stack as
// the image, so their radiance is attenuated by the surfaces they pass.
//
// The temporal filter accumulates this volume in a gamma-encoded space (see volumetric_filter): single
// high-DR samples are compressed before they enter the history, which is what turns the remaining
// probe-sampling noise into a stable low-frequency field instead of bright per-cell dots.
import world_common;
import world_core;
import bindings;
import sky;
import math;
import trace;
import water;
import froxel;
import medium;
import segment;
import path_trace;
import lighting;

public static const float FROXEL_INJECTION_MAX = 30000.0; // per-unit-length in-scatter ceiling (fp16 safe)

/** Unoccluded in-scattered radiance from one uniformly sampled emitter rectangle (phase ∝ HG). */
public float3 evaluateVolumeEmitter(float3 samplePosition, float3 lightPosition, float3 lightNormal,
                             float3 emission, float area, float3 viewRay, float g) {
    float3 toLight = lightPosition - samplePosition;
    float dist2 = dot(toLight, toLight);
    if (dist2 <= 1.0e-5) {
        return float3(0.0);
    }
    float dist = sqrt(dist2);
    float3 lightDir = toLight / dist;
    float emitterCosine = max(0.0, dot(lightNormal, -lightDir));
    float phase = hg(dot(lightDir, viewRay), g);
    return emission * (emitterCosine * area * phase / dist2);
}

/**
 * Next-event estimate of nearby block-emitter in-scatter with ONE visibility ray. Samples the local
 * light-grid span by its power-weighted alias table (the exact proposal pdf is recoverable, so the
 * estimator divides by it), which turns torch/lava glow into a bounded low-variance term that the
 * temporal accumulation converges — a stochastic path that occasionally HITS an emitter reads as
 * fireflies instead, since the probability of hitting a small HDR surface from a probe cell is tiny.
 */
public float3 froxelEmitterLighting(float3 samplePosition, float3 viewRay, float g,
                             inout uint sampleSeed, inout uint proposalSeed) {
    if (worldPush.lightCount == 0u || pc.lightBufAddr == 0 || pc.lightLocalAliasAddr == 0) {
        return float3(0.0, 0.0, 0.0);
    }
    LightGridCell gridCell;
    int3 gridCellCoord;
    if (!findLightGridCell(samplePosition, gridCell, gridCellCoord)
            || gridCell.invWeightSum <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    uint lightIndex;
    selectLightGridSpanLight(gridCell, proposalSeed, lightIndex);
    Light light = ConstPtr<Light>(pc.lightBufAddr)[lightIndex];
    float3 emission = lightRadiance(light);
    float area = lightArea(light);
    float3 rectanglePoint = light.pos + worldPush.lightRebase.xyz
            + (rndf(sampleSeed) * 2.0 - 1.0) * lightHalfU(light)
            + (rndf(sampleSeed) * 2.0 - 1.0) * lightHalfV(light);
    float3 contribution = evaluateVolumeEmitter(samplePosition, rectanglePoint,
            lightGeometricNormal(light), emission, area, viewRay, g);
    // localProbability=1 reconstructs the exact weighted neighborhood proposal PDF.
    float sourcePdf = proposalPdf(light, emission, gridCell, gridCellCoord, 1.0);
    if (sourcePdf <= 1.0e-20) {
        return float3(0.0, 0.0, 0.0);
    }
    float3 toLight = rectanglePoint - samplePosition;
    float dist = length(toLight);
    if (dist <= RAY_TMIN) {
        return float3(0.0, 0.0, 0.0);
    }
    float3 visible = visibility(samplePosition, toLight / dist, dist * 0.999).transmittance;
    return contribution * visible / sourcePdf;
}

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    // Same directional light derivation as the indirect pass: one LUT-driven value for terrain NEE and
    // for the fog's sun/moon in-scattering, so the shafts cannot disagree with surface lighting.
    celestialLight = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);

    // The probe grid is decimated in XY and Z (see froxelInjectDimensions), one shade per probe block;
    // the filter pass trilinearly resamples it back to the full grid. Rays are the dominant cost of this
    // pass, so the decimation is the largest single lever available.
    int3 dims = froxelInjectDimensions();
    uint3 tid = DispatchRaysIndex();
    if (tid.x >= uint(dims.x) || tid.y >= uint(dims.y) || tid.z >= uint(dims.z)) {
        return;
    }
    float4 outCell = float4(0.0, 0.0, 0.0, 0.0);
    // Center of the covered probe block in full-grid cell units (cell i's center is i + 0.5). Sub-cell
    // jitter decorrelates the sampled position across frames without moving the block; the temporal
    // filter then averages the noise away instead of it sitting as a per-cell bias.
    float2 probeScale = float2(froxelDimensions().xy) / float2(dims.xy);
    float2 probeCenter = (float2(tid.xy) + 0.5) * probeScale;
    float2 cellCenter = probeCenter + froxelFrameJitter() * probeScale;
    float dist0 = froxelBoundaryDistance(float(tid.z));
    float dist1 = froxelBoundaryDistance(float(tid.z + 1));
    float dist = 0.5 * (dist0 + dist1);
    float3 origin;
    float3 dir;
    froxelRay(cellCenter, origin, dir);
    float3 pos = origin + dir * dist;

    // Do not light cells behind the first visible interface: fog stops at the wall/glass/water surface
    // the camera actually sees (matching the integrate pass clip and the surface NEE's own occlusion).
    int3 itid = int3(tid);
    float opaque = froxelOpaqueDistance(probeCenter);
    if (opaque >= 0.0 && dist > opaque) {
        froxelScattering[itid] = outCell;
        return;
    }

    float sigmaT = froxelExtinction(pos);
    if (sigmaT <= 0.0) {
        froxelScattering[itid] = outCell;
        return;
    }
    float3 albedo = clamp(worldPush.fogOptics.rgb, 0.0, 1.0);
    float g = clamp(worldPush.fogOptics.w, -0.95, 0.95);
    float3 sigmaS = sigmaT * albedo;
    uint seed = (tid.x * 1973u + tid.y * 9277u + tid.z * 5939u + 26699u)
            ^ (worldPush.frameIndex * 2654435761u);
    seed = pcg(seed);
    uint proposalSeed = (tid.x * 12289u + tid.y * 41521u + tid.z * 25457u)
            ^ (worldPush.frameIndex * 40503u);
    proposalSeed = pcg(proposalSeed);

    float3 inScatter = float3(0.0, 0.0, 0.0);

    // 1) Directional sun/moon with path-traced per-channel visibility. The sampled light direction also
    // drives the phase function so jitter softens shafts spatially like the surface NEE softens shadows.
    float directional = worldPush.fogLighting.x;
    if (directional > 0.0 && max(celestialLight.illuminance.r,
            max(celestialLight.illuminance.g, celestialLight.illuminance.b)) > 1.0e-6) {
        float3 lightDir = celestialLight.dir;
        if (celestialLight.halfAngle > 0.0) {
            lightDir = sampleSquare(celestialLight.dir, celestialLight.halfAngle, seed);
        }
        VisibilityResult shadow = visibility(pos, lightDir, 10000.0);
        float3 transmittance = shadow.transmittance;
        // Underwater: same wave-refraction caustic weighting as the surface NEE, at the water exit.
        // Blended by the configured strength so 0 leaves the pure path-traced transmittance intact.
        if (froxelUnderwater() && shadow.waterHitT > 0.0) {
            float caustic = waterCaustic(pos + lightDir * shadow.waterHitT,
                    lightDir, shadow.waterHitT);
            transmittance *= 1.0 + (caustic - 1.0) * worldPush.fogLighting.z;
        }
        float phase = hg(dot(dir, lightDir), g);
        inScatter += sigmaS * celestialLight.illuminance * phase * transmittance * directional;
    }

    // 2) Block-emitter next-event estimation, one light per probe per frame, pdf-correct, gated by the
    // same GI strength as the radiance probe so one setting controls the whole non-directional fog
    // light field. The temporal filter accumulates the contributions, so over a few frames every probe
    // covers the whole nearby light set.
    float giStrength = worldPush.fogLighting.y;
    if (giStrength > 0.0) {
        inScatter += sigmaS * froxelEmitterLighting(pos, dir, g, seed, proposalSeed) * giStrength;
    }

    // 3) Path-traced radiance probe (terrain GI + skylight). Directions are sampled from the HG phase
    // about the view ray — the exact pdf, so the average sample radiance IS the phase-weighted
    // in-scatter integral with no extra weight. Each sample path starts at the cell, shades its first
    // vertex under the restricted probe path and continues up to the configured GI depth; the temporal
    // filter resolves the resulting noise over frames. The celestial disc and the discrete starfield
    // stay hidden on probe rays: the directional term above already represents the sun/moon with its
    // own phase weight, and a single per-cell star sample would read as speckle instead of glow. The
    // atmosphere LUT and airglow still enter through the miss shader, so the sky stays smooth.
    uint giSamples = froxelGiSamples();
    if (giSamples > 0u && giStrength > 0.0) {
        float3 radiance = float3(0.0, 0.0, 0.0);
        MediumStack probeMedium = makeMediumStack(froxelUnderwater()
                ? makeDielectricMedium(true, WATER_IOR, worldPush.waterParams.xyz, 1.0)
                : airMedium());
        int probeBounce = max(1, int(worldPush.maxBounces) - int(froxelGiBounces()));
        for (uint s = 0u; s < giSamples; ++s) {
            // u points from the cell toward the sampled light (like the directional term's lightDir);
            // the incoming photon direction is -u and the collected radiance is exactly L from -u.
            float3 u = samplePhase(dir, g, seed);
            PathSegment probe = makePathSegment(pos, u, float3(1.0, 1.0, 1.0), probeMedium,
                    0.0, 0.0, seed, probeBounce, false, true);
            radiance += tracePath(probe, tid.xy, s);
        }
        inScatter += sigmaS * radiance * giStrength * (1.0 / float(giSamples));
    }

    froxelScattering[itid] = float4(clamp(inScatter, float3(0.0), float3(FROXEL_INJECTION_MAX)),
            clamp(sigmaT, 0.0, 64.0));
}
