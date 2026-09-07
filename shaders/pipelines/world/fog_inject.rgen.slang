// Froxel fog injection: one ray-traced, colour-carrying shadow query per froxel.
//
// This is a raygen in the WORLD ray-tracing pipeline rather than a standalone compute pass, and that is
// the load-bearing decision of the whole fog system. It means injection reuses the exact same SBT, the
// same any-hit records and the same visibility() helper as surface next-event estimation, so a shaft
// passing through stained glass is tinted by the same code that tints the shadow the pane casts on the
// floor. The two cannot drift. An inline-rayQuery compute pass would have had to reimplement the atlas
// alpha test and the translucent tint accumulation, which is precisely the duplication that would make
// coloured transmission a second, divergent implementation.
//
// Dispatch is 3D: one thread per froxel. Output is RAW (un-accumulated) in-scatter per unit length in
// absolute ACEScg; the temporal blend and the depth integration both belong to the volumetric compute
// passes, which own the history and therefore own the denoising.
//
// Ordinary TraceRay, deliberately: visibility() is already a terminate-on-first-hit query with no
// closest-hit shader, and froxel neighbours are highly coherent, so there is nothing for an SER reorder
// barrier to recover.
import world_common;
import world_core;
import bindings;
import math;
import trace;
import sky;
import fog;

// Shadow rays are cast to the full trace range: an occluder for a shaft may be far outside the froxel
// grid's own depth range (a distant mountain shadowing near-camera air is the common case).
static const float FOG_SHADOW_TMAX = 10000.0;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    FogGrid grid = fogGrid(worldPush);
    int3 index = int3(DispatchRaysIndex());
    if (!grid.enabled || any(index >= grid.dims)) {
        return;
    }

    // The dominant celestial body is derived exactly as indirect.rgen derives it, from the same
    // transmittance LUT, so the light scattering in the air and the light landing on terrain are one
    // number and cannot disagree about the sun's colour, direction or whether the moon has taken over.
    celestialLight = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);

    // The whole grid is offset by the camera's own sub-pixel jitter this frame, so the fog signal moves
    // in lockstep with the geometry signal DLSS-RR reconstructs. Both values are grid-wide and derived
    // from the push block, so the integration pass reproduces this froxel's position exactly.
    FogRay ray = fogColumnRay(worldPush, grid, index.xy, fogGridJitterUv(worldPush));
    float distance = fogSliceDistance(grid, float(index.z) + 0.5 + fogSliceJitter(worldPush));
    float3 position = ray.origin + ray.dir * distance;

    float extinction = fogDensity(worldPush, position);
    if (extinction < FOG_DENSITY_EPSILON) {
        fogScatter[index] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Sample a point on the body's disc rather than its centre direction. This is what gives a shaft
    // edge a physically-sized penumbra; softening it later with a blur would be both slower and wrong,
    // and the "no spatial blur" rule depends on getting the softness from here instead.
    //
    // Driven by the R2 low-discrepancy sequence rather than a PRNG: this single sample is temporally
    // accumulated, so successive frames should probe the disc in a stratified pattern rather than at
    // independent random points. White noise leaves low-frequency error that survives both the bilinear
    // fetch and the temporal average, which is what showed up as clumpy speckle.
    float3 lightDir = celestialLight.dir;
    if (celestialLight.halfAngle > 0.0) {
        lightDir = sampleSquareAt(lightDir, celestialLight.halfAngle,
                fogSampleR2(index, worldPush.frameIndex));
    }
    // visibility() returns per-channel transmittance: the shadow SBT's any-hit records alpha-test
    // cutouts and MULTIPLY translucent/water tint into it. That float3 is the entire colour-carrying
    // mechanism of this system, and it only works because the ray starts at the froxel's OWN position
    // and therefore crosses whatever glass, leaves or water lie between it and the sun.
    //
    // There is deliberately no depth cull here. Skipping froxels behind the g-buffer surface looks like
    // free performance, but the composite reads this volume through a TRILINEAR sampler, so a froxel is
    // also fetched by the XY taps of neighbouring columns whose surfaces are at completely different
    // depths. Anything fabricated for a culled froxel — zero, or an unshadowed value, or a query moved
    // in front of the occluder — bleeds sideways across silhouettes and, for the moved-origin variant,
    // skips the very pane that was supposed to tint the shaft. Bedrock RTX dispatches its full
    // 256x128x64 grid unconditionally for the same reason.
    float3 transmittance = visibility(position, lightDir, FOG_SHADOW_TMAX).transmittance;

    // Store the PHASELESS in-scatter: scattering coefficient x light x visibility, with no view
    // dependence. The phase term is applied later, at integration time, from the current frame's ray
    // direction.
    //
    // This matters because the stored value is temporally accumulated. Baking the phase in makes the
    // history describe the sun-view angle of the frame it was captured in, so every camera rotation
    // leaves a tail of stale phase values, and the strong forward lobe (g = 0.7) magnifies the error
    // enormously right where the shaft is brightest. That is what smeared the beams. Visibility and
    // extinction genuinely are view-independent and are the only things worth averaging over time;
    // the phase is exact and free to evaluate per frame.
    float scattering = extinction * worldPush.fogParams1.w;
    float3 direct = scattering * celestialLight.illuminance * transmittance;

    // RGB: the phaseless in-scatter the integration pass multiplies by this frame's phase.
    // A: this froxel's EXTINCTION.
    //
    // Carrying extinction here rather than recomputing it in the integration pass is what makes the two
    // passes agree by construction. They must evaluate the density field at exactly the same jittered
    // position — a divergence would mean the radiance and the medium describe different points — and
    // sharing the value makes that impossible to get wrong. It also halves the cost of the density
    // field, which is the most expensive per-froxel term now that the shadow ray is the only other one.
    fogScatter[index] = float4(clamp(direct, float3(0.0), float3(HALF_MAX)),
            clamp(extinction, 0.0, HALF_MAX));
}
