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

/**
 * Farthest scene distance across the froxel column's screen footprint, in the same radial metric the
 * grid uses. Reversed-Z, so the SMALLEST depth value is the farthest surface; taking the farthest over
 * the footprint is the conservative choice, because culling a froxel that is still visible through any
 * pixel of the column would punch a hole in the shaft.
 */
float fogColumnSceneDistance(FogGrid grid, int2 columnIndex, float3 rayOrigin) {
    uint2 depthDims;
    gDepth.GetDimensions(depthDims.x, depthDims.y);
    float2 scale = float2(depthDims) / float2(grid.dims.xy);
    float2 base = float2(columnIndex) * scale;
    int2 lo = int2(clamp(base, float2(0.0), float2(depthDims) - 1.0));
    int2 hi = int2(clamp(base + scale - 1.0, float2(0.0), float2(depthDims) - 1.0));

    // A froxel column covers a whole tile of screen pixels, and this cull must be conservative across
    // ALL of them: culling a froxel that any one pixel of the tile can still see through punches a hole
    // in the fog for that pixel. Sampling only the four corners is not enough — a tile straddling a
    // silhouette can have every corner on the near object while its interior sees far background, and
    // the background pixels then lose their fog entirely. That reads exactly as "fog disappears behind
    // edges". Walk the tile on a fixed grid instead, and treat any sky tap as "cull nothing".
    const int TAPS = 4;
    float minDepth = 1.0;
    for (int y = 0; y < TAPS; ++y) {
        for (int x = 0; x < TAPS; ++x) {
            float2 t = float2(float(x), float(y)) / float(TAPS - 1);
            int2 texel = int2(round(lerp(float2(lo), float2(hi), t)));
            float d = gDepth[clamp(texel, int2(0, 0), int2(depthDims) - 1)];
            if (d <= 0.0) {
                return 1.0e30; // sky anywhere in the tile: nothing bounds the column
            }
            minDepth = min(minDepth, d);
        }
    }
    // Reconstruct the far-most hit point from the reversed-Z guide depth. The column's own NDC is not
    // needed to a sub-froxel accuracy here, so the unjittered column centre is used.
    float2 uv = (float2(columnIndex) + 0.5) / float2(grid.dims.xy);
    float2 ndc = uv * 2.0 - 1.0;
    float4 h = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, minDepth, 1.0));
    if (h.w <= 0.0) {
        return 1.0e30;
    }
    float3 hit = h.xyz / h.w + worldPush.camOffset;
    return length(hit - rayOrigin);
}

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

    // Shared with the integration pass so both agree on exactly where this froxel was sampled.
    float3 jitter = fogJitter(worldPush, index.xy, index.z);
    FogRay ray = fogColumnRay(worldPush, grid, index.xy, jitter.xy);
    float distance = fogSliceDistance(grid, float(index.z) + 0.5 + jitter.z);
    float3 position = ray.origin + ray.dir * distance;

    float extinction = fogDensity(worldPush, position);
    if (extinction < FOG_DENSITY_EPSILON) {
        fogScatter[index] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Froxels far behind the g-buffer surface skip their SHADOW RAY, which is the expensive part, but
    // they must still be filled with a plausible value rather than zero.
    //
    // The composite fetches this volume through a TRILINEAR sampler, so a froxel is read by the XY taps
    // of neighbouring columns too. At a silhouette the neighbouring column can be at a completely
    // different depth — background at 100 blocks beside a pillar at 5 — and no amount of Z margin helps,
    // because the background pixel's XY tap lands on the pillar's column at the FAR slice, which the
    // cull had zeroed. Writing zero there is what produced the dark halo hugging every edge: the cull,
    // an optimisation, was corrupting data the composite genuinely reads.
    //
    // Writing the unshadowed in-scatter instead keeps the field continuous and costs nothing (the ray is
    // still skipped). It is "wrong" only in being unoccluded, and nothing samples it directly — it is
    // reachable solely by interpolation from neighbours, where plausible fog is far closer to correct
    // than a hole.
    bool skipShadowRay = false;
    if (fogDepthCullEnabled(worldPush)) {
        float sceneDistance = fogColumnSceneDistance(grid, index.xy, ray.origin);
        float margin = fogSliceDistance(grid, float(index.z) + 0.5 + FOG_DEPTH_CULL_MARGIN_SLICES)
                - distance;
        skipShadowRay = distance > sceneDistance + margin;
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
    // mechanism of this system.
    float3 transmittance = skipShadowRay
            ? float3(1.0, 1.0, 1.0)
            : visibility(position, lightDir, FOG_SHADOW_TMAX).transmittance;

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

    // Flat isotropic fill, folded in here rather than kept separate: being isotropic its phase is the
    // constant 1/4pi, so it needs no view direction and can ride in the same channels. Defaults to
    // zero, since it is unoccluded and would otherwise glow inside caves.
    float ambient = worldPush.fogParams2.z;
    float3 ambientTerm = ambient > 0.0
            ? scattering * FOG_INV_4PI * ambient * celestialLight.illuminance
            : float3(0.0, 0.0, 0.0);

    // RGB carries the phaseless DIRECT term; the integration pass multiplies it by this frame's phase.
    // Alpha carries the already-phased ambient as a luminance, reconstructed against the light colour
    // at integration time — the ambient is a small achromatic fill, so a scalar is ample and it avoids
    // a second RGB volume purely to keep two terms apart.
    float ambientLuma = fogLuminance(ambientTerm);
    fogScatter[index] = float4(clamp(direct, float3(0.0), float3(HALF_MAX)),
            clamp(ambientLuma, 0.0, HALF_MAX));
}
