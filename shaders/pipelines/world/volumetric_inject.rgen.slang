// Froxel light + extinction injection. One raygen thread per lighting-probe cell (the probe grid is
// half-resolution in XY — see froxelInjectDimensions; DispatchRaysDimensions is that 3D grid). Each cell
// derives its camera-ray position, evaluates the world-stable density field, then lightens the
// single-scattering term with two sources:
//
//   directional — sun/moon through the same weather-independent path-traced visibility() the surface
//                 NEE uses, so tinted glass/water/translucent geometry colors the in-scattered light
//                 and soft shadows break up the shafts;
//   GI          — the path tracer itself. The in-scatter integral over incoming directions is
//                 importance-sampled by the HG phase about the view ray (exact pdf, so the estimator is
//                 just the sampled radiance), and each sample is traced with tracePath(): the first
//                 hit shades exactly like a surface pixel (emission + sun NEE + emitter RIS + SSS) and
//                 continues into the scene, so block-emitter light, terrain/canopy GI and skylight all
//                 reach the fog with the same local lighting model as the image. A probe ray may not
//                 see the celestial disc — the directional term represents the sun/moon directly, so a
//                 disc sample would be a firefly on top of it — but the atmosphere LUT glow still
//                 enters through the miss shader, giving the orange halo around the sun direction.
//
// All light-attenuation paths go through visibility(), whose any-hit shaders already accumulate
// per-channel Beer-Lambert transmission (tinted glass, water, translucent blocks, entity dielectrics),
// so fog lit through a red window pane is red — this is the path-traced advantage over shadow maps.
// The probe path additionally traverses real dielectric volumes (glass, ice) with the same medium
// stack as the image, so its radiance is attenuated by the surfaces it passes.
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

public static const float FROXEL_INJECTION_MAX = 30000.0; // per-unit-length in-scatter ceiling (fp16 safe)

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

    // 2) Path-traced radiance probe (GI + emissives + skylight). Directions are sampled from the HG
    // phase about the view ray — the exact pdf, so the average sample radiance IS the phase-weighted
    // in-scatter integral with no extra weight. Each sample path starts at the cell and shades its first
    // vertex like a surface pixel, then continues up to the configured GI depth; the temporal filter
    // resolves the resulting noise over frames. The celestial disc stays hidden on probe rays (the
    // directional term above is that light, with its own phase weight) and the first vertex emitter
    // radiance is gathered even on hidden-disc rays, because an emissive surface seen directly by the
    // fog is the light being sampled — no previous RIS along this path accounted for it.
    float giStrength = worldPush.fogLighting.y;
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
