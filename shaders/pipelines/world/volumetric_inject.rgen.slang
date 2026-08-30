// Froxel light + extinction injection. One raygen thread per sun-shaft light cell (the grid is
// decimated relative to the fog grid — see froxelInjectDimensions; DispatchRaysDimensions is that 3D
// grid). Each cell derives its camera-ray position, evaluates the world-stable density field, then
// lightens the single-scattering term with two sources:
//
//   directional — sun/moon through the same weather-independent path-traced visibility() the surface
//                 NEE uses, so tinted glass/water/translucent geometry colors the in-scattered light
//                 and soft shadows break up the shafts. This is the only per-cell ray in the whole
//                 volumetric pipeline, and it is a single deterministic NEE (low variance);
//   GI/emitters — the world-anchored light probe volume (volumetric_light): one trilinear lookup.
//                 The volume is progressively accumulated at fixed world positions, so the fog's
//                 skylight, terrain GI and torch/lava glow are stable and smooth by construction —
//                 no per-cell sampling noise, no camera-wobble. The lookup returns the average
//                 incident radiance, so the term is exactly sigma_s * C (phase was folded in when the
//                 probes were sampled).
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
import water;
import froxel;

public static const float FROXEL_INJECTION_MAX = 30000.0; // per-unit-length in-scatter ceiling (fp16 safe)

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    // Same directional light derivation as the indirect pass: one LUT-driven value for terrain NEE and
    // for the fog's sun/moon in-scattering, so the shafts cannot disagree with surface lighting.
    celestialLight = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);

    // The sun-shaft grid is decimated in XY and Z (see froxelInjectDimensions); the filter pass
    // resamples it back to the full grid. Rays are the dominant cost of this pass, so the decimation is
    // the largest single lever available.
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

    // 2) World-anchored radiance probe volume: averaged incident radiance from skylight, terrain GI
    // and block emitters, accumulated over up to 64 frames per probe (see volumetric_light). The
    // lookup is stable under camera motion by construction, phase was folded into the average when the
    // probes were sampled, and the field already carries the configured strength (the update pass
    // scales it), so the fog term needs no extra factor here.
    if (worldPush.fogLighting.y > 0.0) {
        inScatter += sigmaS * froxelLightSample(pos);
    }

    froxelScattering[itid] = float4(clamp(inScatter, float3(0.0), float3(FROXEL_INJECTION_MAX)),
            clamp(sigmaT, 0.0, 64.0));
}
