// Froxel light + extinction injection. One raygen thread per lighting-probe cell (the probe grid is
// half-resolution in XY — see froxelInjectDimensions; DispatchRaysDimensions is that 3D grid). Each cell
// derives its camera-ray position, evaluates the world-stable density field, then lightens the
// single-scattering term with three sources:
//
//   directional — sun/moon through the same weather-independent path-traced visibility() the surface
//                 NEE uses, so tinted glass/water/translucent geometry colors the in-scattered light;
//   ambient     — analytic vertical transmittance (the exponential density field has a closed-form
//                 optical depth) times a hemisphere average of the sky LUT, with the LUT's light-view
//                 cosine clamped so the sun's forward-scatter lobe is not double counted with the
//                 directional term;
//   emitters    — a small RIS reservoir over the existing power-weighted light grid, one shadow ray per
//                 survivor. Per-cell decorrelated proposals avoid tile-shaped noise.
//
// All light-attenuation paths go through visibility(), whose any-hit shaders already accumulate
// per-channel Beer-Lambert transmission (tinted glass, water, translucent blocks, entity dielectrics),
// so fog lit through a red window pane is red — this is the path-traced advantage over shadow maps.
import world_common;
import world_core;
import bindings;
import sky;
import math;
import trace;
import lighting;
import water;
import froxel;

public static const float FROXEL_INJECTION_MAX = 30000.0; // per-unit-length in-scatter ceiling (fp16 safe)
// ~20 degrees; beyond the sun's forward-scatter core, so the ambient average excludes the lobe that the
// directional term already represents through the HG phase.
public static const float FROXEL_AMBIENT_LIGHT_COS = 0.94;

public struct FroxelReservoir {
    public float3 samplePosition;
    public float3 contribution;
    public float weightSum;
    public float target;
}

public FroxelReservoir emptyFroxelReservoir() {
    FroxelReservoir r;
    r.samplePosition = float3(0.0, 0.0, 0.0);
    r.contribution = float3(0.0, 0.0, 0.0);
    r.weightSum = 0.0;
    r.target = 0.0;
    return r;
}

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
 * RIS estimate of nearby block-emitter radiance scattered toward the camera, with one visibility ray.
 * Uses only the local light-grid span: a distant global proposal has vanishing contribution but a very
 * large inverse-PDF weight, which reads as isolated fireflies at one sample per froxel.
 */
public float3 froxelEmitterLighting(float3 samplePosition, float3 viewRay, float g,
                             inout uint sampleSeed, inout uint proposalSeed) {
    uint candidateCount = min(froxelLocalLightCandidates(), worldPush.lightCount);
    if (candidateCount == 0u || pc.lightBufAddr == 0 || pc.lightLocalAliasAddr == 0) {
        return float3(0.0, 0.0, 0.0);
    }
    LightGridCell gridCell;
    int3 gridCellCoord;
    if (!findLightGridCell(samplePosition, gridCell, gridCellCoord)
            || gridCell.invWeightSum <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    FroxelReservoir reservoir = emptyFroxelReservoir();
    for (uint candidate = 0u; candidate < candidateCount; ++candidate) {
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
        return float3(0.0, 0.0, 0.0);
    }
    reservoir.weightSum *= 1.0 / max(float(candidateCount), 1.0);
    float weight = reservoir.weightSum / max(reservoir.target, 1.0e-20);
    float3 toLight = reservoir.samplePosition - samplePosition;
    float dist = length(toLight);
    if (dist <= RAY_TMIN) {
        return float3(0.0, 0.0, 0.0);
    }
    float3 visible = visibility(samplePosition, toLight / dist, dist * 0.999).transmittance;
    return reservoir.contribution * weight * visible;
}

/**
 * Analytical hemisphere-average sky radiance (BT.709 → ACEScg). The sky LUT is parameterized by
 * view-direction zenith cosine and light-view cosine; five fixed directions (up + four at 30°
 * elevation) averaged give a stable ambient anchor with no extra rays. The light-view cosine is clamped:
 * the LUT's near-sun rows contain the forward-scatter peak of the atmosphere, which the directional term
 * already adds through the HG phase — including it here would double-count the sun and wash the fog out.
 */
public float3 froxelAmbientSkyRadiance(SkyState state) {
    float3 sum = float3(0.0);
    float count = 0.0;
    for (int i = 0; i < 4; ++i) {
        float az = float(i) * 1.57079632679;
        float3 dir = normalize(float3(cos(az) * 0.866, 0.5, sin(az) * 0.866));
        float2 sunUv = skyViewLutUv(state.viewerRadiusKm, dir.y,
                min(lightViewCosine(dir, state.sunDir), FROXEL_AMBIENT_LIGHT_COS), SKY_VIEW_BODY_SUN);
        float2 moonUv = skyViewLutUv(state.viewerRadiusKm, dir.y,
                min(lightViewCosine(dir, state.moonDir), FROXEL_AMBIENT_LIGHT_COS), SKY_VIEW_BODY_MOON);
        sum += skyViewLut.SampleLevel(sunUv, 0.0).rgb + skyViewLut.SampleLevel(moonUv, 0.0).rgb;
        count += 1.0;
    }
    float2 sunUv = skyViewLutUv(state.viewerRadiusKm, 1.0,
            min(lightViewCosine(float3(0.0, 1.0, 0.0), state.sunDir), FROXEL_AMBIENT_LIGHT_COS),
            SKY_VIEW_BODY_SUN);
    float2 moonUv = skyViewLutUv(state.viewerRadiusKm, 1.0,
            min(lightViewCosine(float3(0.0, 1.0, 0.0), state.moonDir), FROXEL_AMBIENT_LIGHT_COS),
            SKY_VIEW_BODY_MOON);
    sum += skyViewLut.SampleLevel(sunUv, 0.0).rgb + skyViewLut.SampleLevel(moonUv, 0.0).rgb;
    count += 1.0;
    return bt709ToAcesCg(max(sum / count + state.airglowLuminance, float3(0.0)));
}

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    // Same directional light derivation as the indirect pass: one LUT-driven value for terrain NEE and
    // for the fog's sun/moon in-scattering, so the shafts cannot disagree with surface lighting.
    celestialLight = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);

    // The probe grid is half-resolution in XY (see froxelInjectDimensions): one shade per 2x2 fog cells,
    // with the filter pass trilinearly resampling it back to the full grid. Rays are the dominant cost of
    // this pass, so a 4x thread reduction is the largest single lever available.
    int3 dims = froxelInjectDimensions();
    uint3 tid = DispatchRaysIndex();
    if (tid.x >= uint(dims.x) || tid.y >= uint(dims.y) || tid.z >= uint(dims.z)) {
        return;
    }
    float4 outCell = float4(0.0, 0.0, 0.0, 0.0);
    // Center of the covered 2x2 block in full-grid cell units (cell i's center is i + 0.5). Sub-cell
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
    uint proposalSeed = (tid.x * 12289u + tid.y * 41521u + tid.z * 25457u) ^ (worldPush.frameIndex * 40503u);
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

    // 2) Ambient sky: analytic vertical transmittance against the hemisphere-average sky radiance. The
    // density field is exponential in height, so the optical depth from a point to space is exactly
    // sigma / falloff — no ray needed. This is the cost/quality trade behind the 2x2 probe decimation:
    // canopy/cliff occlusion is dropped from the ambient term, but the sun/moon path below still carries
    // the scene's real geometric occlusion and therefore the shafts.
    float ambient = worldPush.fogLighting.y;
    if (ambient > 0.0) {
        float scaleHeight = worldPush.fogDistance.w > 1.0e-3
                ? 1.0 / worldPush.fogDistance.w : 16.0; // 16 blocks when falloff is disabled (underwater)
        float skyVis = exp(-min(sigmaT * scaleHeight, 8.0));
        // Isotropic phase for the diffuse ambient term: sigma_s * L * p(1/4pi) * transmittance.
        inScatter += sigmaS * froxelAmbientSkyRadiance(skyState(worldPush)) * skyVis * ambient
                * (1.0 / (4.0 * SKY_PI));
    }

    // 3) Local emitter RIS, one independent reservoir per configured sample, averaged. The survivor's
    // shadow ray is also path-traced, so a red window in front of a light colors the shaft.
    uint emitterSamples = froxelEmitterSamples();
    if (emitterSamples > 0u && froxelLocalLightCandidates() > 0u) {
        float3 emitter = float3(0.0, 0.0, 0.0);
        for (uint s = 0u; s < emitterSamples; ++s) {
            emitter += froxelEmitterLighting(pos, dir, g, seed, proposalSeed);
        }
        inScatter += emitter * (1.0 / float(emitterSamples));
    }

    froxelScattering[itid] = float4(clamp(inScatter, float3(0.0), float3(FROXEL_INJECTION_MAX)),
            clamp(sigmaT, 0.0, 64.0));
}
