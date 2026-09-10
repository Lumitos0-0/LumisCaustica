// Aerial-perspective fog: an analytic exponential-height participating medium, integrated along each
// traced path segment rather than composited in screen space.
//
// The medium model is MCRTX's (Bedrock RTX), taken from its pipeline contract rather than its code:
// extinction split into scattering and absorption, a Henyey-Greenstein phase, and a vertical density
// profile described by a scale height and a reference altitude. Two things are deliberately not copied.
// Its result is a froxel volume composited after the trace against a depth guide, because a deferred
// renderer has no path to accumulate into; here the same physics is integrated per segment inside the
// transport, so the volume dims and lights reflections, refractions and indirect bounces exactly as it
// dims primary visibility, and a volume never needs reprojecting. And its fog is lit through a per-light
// incident-radiance meter amortized across the volume, and the same split is kept here: the meter's sun and
// sky taps are gathered per voxel, while the medium's own attenuation toward the light stays a closed-form
// integral, so a march tap costs eight texel reads and a handful of exponentials and no rays at all.
//
// Light is looked up where it lands rather than where the eye ends up: a frustum-aligned volume, one voxel
// per screen column and log-depth slice, filled by sun_froxels.rgen with a few fixed rays per voxel and
// sampled by every tap of every segment. Testing the surface behind the air instead reports the ground's
// lighting, which collapses a shaft onto the patch of floor it lands on and flips between a leaf and the
// gap beside it as the frame's jitter moves. A volume can do neither, and costs a fraction of the rays.
//
// Its numbers are vanilla's too, as far as they can be read: the per-biome `volumetric` fog settings in
// Bedrock's resource pack author density as a 0..1 factor over per-channel scattering and absorption, and the
// temperate entry is 0.05 x 0.04 scattering with absorption 0 at Henyey-Greenstein g = 0.6 -- which is what
// the shipped defaults are. Jungles and swamps run to 0.06 at g = 0.4, clear dry air to g = 0.75; the
// height pair is degenerate in every vanilla biome (both 320), so the falloff here is ours.
//
// Density is a function of ABSOLUTE world altitude: path positions are rebased to the terrain origin,
// which moves as the camera travels, so every altitude goes through `rebaseY` (that origin's block Y,
// pushed per frame). An exponential would otherwise slide with the player. The profile is flat at and
// below `refY` rather than growing without bound downward, so a cave's air is no denser than a
// mountaintop's; what makes caves and interiors dark is the volume above, not the profile.

import world_common;
import world_core;
import bindings;
import math;
import medium;
import trace;
import sky;

public static const uint FOG_ENABLED_BIT = 32u; // WorldPush.flags bit 5
public static const float FOG_INV_4PI = 1.0 / (4.0 * 3.14159265359);

// Backward-lobe anisotropy and weight of the two-lobe phase, and the ceiling on either integral's
// exponent so a deep downward column saturates to zero instead of overflowing.
public static const float FOG_BACK_G = -0.25;
public static const float FOG_FORWARD_WEIGHT = 0.65;
public static const float FOG_MAX_TAU = 40.0;

// One float4 pair on WorldPush carries the whole medium; these names exist so the shading sites read as
// intent rather than as lane indexing.
public struct FogMedium {
    public float density;     // extinction at and below the reference altitude, 1/block
    public float scaleHeight; // e-folding distance above it, blocks
    public float refY;        // absolute altitude the profile flattens at, blocks
    public float g;           // Henyey-Greenstein anisotropy of the forward lobe
    public float albedo;      // fraction of extinction that scatters rather than absorbs
    public float reach;       // distance the medium is integrated over: sun column, and the near field
    public float rebaseY;     // absolute altitude of rebased y = 0
    public int maxSteps;      // taps resolving a segment's near field
    public bool active;
};

/**
 * The frame's fog, or an inactive FogMedium when the volume should cost nothing.
 *
 * `active` is the pushed bit AND a non-zero density, so no caller has to guard the profile itself and
 * disabling the volume lands on exactly the plain Beer-Lambert path below: "off" and "zero density" are
 * then one behaviour, not two that happen to look alike.
 */
public FogMedium fogMedium(WorldPush push) {
    FogMedium f;
    f.density = push.fogParams.x;
    f.scaleHeight = max(push.fogParams.y, 1.0e-3);
    f.refY = push.fogParams.z;
    f.g = clamp(push.fogParams.w, -0.9, 0.9);
    f.albedo = clamp(push.fogTuning.x, 0.0, 1.0);
    f.reach = max(push.fogTuning.y, 1.0);
    f.rebaseY = push.fogTuning.z;
    f.maxSteps = max((int)push.fogTuning.w, 1);
    f.active = (push.flags & FOG_ENABLED_BIT) != 0u && f.density > 0.0;
    return f;
}

/**
 * Extinction (1/block) at a rebased position; `albedo` is what splits it into scattering and absorption.
 */
public float fogDensityAt(FogMedium f, float3 rebasedPos) {
    float above = max(rebasedPos.y + f.rebaseY - f.refY, 0.0);
    return f.density * exp(-above / f.scaleHeight);
}

/**
 * Two-lobe Henyey-Greenstein, 4pi-normalised so it integrates to 1 over the sphere. The forward lobe is
 * what makes a sun-aligned volume read as beams; at the anisotropy that looks right (g ~ 0.6) it alone
 * peaks near eleven times isotropic, which turns "look toward the sun" into a frame-filling white-out.
 * Blending in a weak backward lobe spends about a third of that peak, keeps the forward bias the beams
 * need, and lifts the back-scattered haze on the far side of the volume. Separate from math.slang's
 * single-lobe `hg()` because thin-surface transmission genuinely wants a single forward lobe.
 */
public float fogPhase(FogMedium f, float cosTheta) {
    float gf = f.g;
    float gf2 = gf * gf;
    float forward = (1.0 - gf2) / pow(max(0.0, 1.0 + gf2 - 2.0 * gf * cosTheta), 1.5);
    float gb = FOG_BACK_G;
    float gb2 = gb * gb;
    float backward = (1.0 - gb2) / pow(max(0.0, 1.0 + gb2 - 2.0 * gb * cosTheta), 1.5);
    return (FOG_FORWARD_WEIGHT * forward + (1.0 - FOG_FORWARD_WEIGHT) * backward) * FOG_INV_4PI;
}

/**
 * Transmittance of the medium itself from a point toward the light: the fog's own aerial-perspective
 * self-shadowing. Without it every sunlit point out to the horizon contributes at full strength and the
 * volume integrates into a uniform veil with dark wedges cut into it, rather than beams standing out
 * against fog the light has already been spent crossing.
 *
 * Closed form, not marched: for the exponential part of the profile the column has an exact antiderivative
 * along a climbing ray, and taking it also yields the light's reddening with altitude for free. A ray
 * heading down or level has no such bound — density only grows — so those integrate over the reach
 * and let exp(-tau) saturate. Below `refY` the real profile flattens where this keeps growing, which can
 * only ever darken in-scatter slightly; modelling the kink would cost a branch and a second closed form.
 *
 * `sigma` is the caller's already-evaluated density at the tap: the closed form below needs nothing else
 * about the point, so recomputing the profile here would be a second exponential per tap for nothing.
 */
public float fogColumnToLight(FogMedium f, float sigma, float3 lightDir) {
    float dy = lightDir.y;
    float tau;
    if (dy > 1.0e-4) {
        float climb = min(f.reach * dy / f.scaleHeight, FOG_MAX_TAU);
        tau = sigma * f.scaleHeight / dy * (1.0 - exp(-climb));
    } else {
        tau = sigma * f.reach;
    }
    return exp(-min(tau, FOG_MAX_TAU));
}

public static const float FOG_NEGLIGIBLE_LIGHT = 1.0e-3;

/**
 * Sun and sky light reaching the aerial medium, gathered into a frustum-aligned volume once per frame.
 *
 * <p>The march needs to know what light arrives at a particular bit of air, and only a sample taken there
 * can say so. Testing the surface behind the air instead — which is what a per-pixel shadow map does —
 * reports the ground's lighting and reports it for every tap in front of it:
 * a shaft then collapses onto the patch of floor it lands on and the air above that patch stays flat. It is
 * also why a beam measured on screen cannot be stable: which leaf a jittered primary ray strikes changes
 * with the frame's jitter, and the answer differs by a whole unit of light between the two possibilities.
 *
 * <p>So the volume holds one float4 per (screen column, log-depth slice): rgb is the transmittance to the
 * sun, and alpha is the sky's own contribution -- the mean of (transmittance along a direction) times (this
 * frame's sky-view LUT in that direction) over the whole sphere, which is the isotropic term of the same
 * integral the body's lobe is normalised against. Both are estimated from a few FIXED directions, which is
 * what lets the march sample a coarse structure without noise or crawl, and the whole frame pays
 * the pushed sun and sky ray counts of rays per voxel instead of one per tap per bounce per sample.
 *
 * <p>Screen-aligned columns, world-correct contents: a voxel is defined by the view ray through its column
 * at a given depth, and the pixels-per-voxel divisor comes in through the push (fogVolume.x) rather than
 * being mirrored here and in Java, so the setting that trades grid size for cost cannot desync the two. The
 * grid thins out with distance the way the eye needs it to (a near voxel is
 * centimetres wide, a far one is many blocks) and no reprojection is involved — the grid is rebuilt every
 * frame from the current camera, so nothing has to be revalidated when the player turns.
 */
// Rays per voxel, toward the sun and over the sky, are pushed (`caustica.rt.fog.sunRays` and `skyRays`) and
// taken exactly: zero is a meaningful answer and not an unset lane, since "no direct sun in the fog at
// all" is how you see what the shafts are made of. These are the ceilings a hand-edited config cannot pass.
public static const int FOG_MAX_SUN_RAYS = 16;
public static const int FOG_MAX_SKY_RAYS = 32;    // directions over the whole sphere, not the dome
// How much of the filtered value replaces the gathered one, per channel, times the pushed filter strength:
// the ambient is a veil and takes the filter fully, the sun's transmittance keeps a quarter of its edge so a
// shaft stays a shaft. That asymmetry is MCRTX's -- `BlurGIInscatter` exists for the GI volume and nothing
// blurs the sun's in-scatter -- and it is the reason a wide beam survives the filter at all. Slice neighbours
// count for less than same-slice ones, since one slice step is a long way at distance.
public static const float FOG_FILTER_AMBIENT_WEIGHT = 1.0;
public static const float FOG_FILTER_SUN_WEIGHT = 0.75;
public static const float FOG_FILTER_SLICE_WEIGHT = 0.35;
// A wide kernel is what MCRTX's own comment asks for -- "separable 2D box blur with kernel size depending on
// history length" -- so the pushed strength buys radius past 1.0 rather than only more blending: a long
// history has already averaged the noise in time and can afford to spend the spatial taps on a wider detail.
public static const int FOG_FILTER_MAX_RADIUS = 2;

/** The gather's per-voxel ray budget, from the push. */
public int fogSunRays() {
    return clamp((int)round(worldPush.fogGather.x), 0, FOG_MAX_SUN_RAYS);
}

public int fogSkyRays() {
    return clamp((int)round(worldPush.fogGather.y), 0, FOG_MAX_SKY_RAYS);
}
public static const float FOG_FROXEL_Z_NEAR = 1.0;   // blocks: the first slice's inner edge
// Shortest distance a march tap covers. Light is now looked up per tap, so a step sized as a share of
// `reach` (512/8 = 64 blocks) gave every segment inside a room a single sample: the medium resolved as flat
// panels whose edges followed the volume's grid. This is the floor under that; long views still fall back on
// `reach / steps` so the tap budget stays bounded by `steps`.
public static const float FOG_TAP_MIN = 2.0;
// BT.709 weights, the same ones the sky module uses to decide which body is brighter.
public static const float3 FOG_LUMA = float3(0.2126, 0.7152, 0.0722);
// Only the *mean* of the dome's radiance survives in one fp16 channel, so the veil is tinted by the sky's
// average colour rather than by whatever each direction actually brought. The body's own path keeps its full
// per-direction tint, because that is the term with a lobe and the one shafts are made of.
public static const float3 FOG_SKY_TINT = float3(0.62, 0.75, 1.0);

/**
 * The volume's extent in voxels, from the pushed divisor rather than a constant, because Java sizes the image
 * from the same number and a setting that only one side obeyed would index the wrong grid silently. The floor
 * is load-bearing beyond tidiness: the gather looks up this voxel's cell in the depth guides at
 * `idx * divisor`, and a grid that ever came out wider than the frame divided by the divisor would read
 * outside them, so the minimum is one voxel rather than a nicer-looking 32.
 */
public float2 fogFroxelDims(float2 renderSize, float divisor) {
    return max(float2(1.0, 1.0), floor(renderSize / max(divisor, 1.0)));
}

/** The divisor this frame's volume was built with, from the push. */
public float fogFroxelDivisor(WorldPush push) {
    return max(push.fogVolume.x, 1.0);
}

/**
 * How many slices the volume has, from the push for the same reason as the divisor. Read through the module
 * static rather than threaded through the three lookup helpers, because the image's own depth is the only
 * other authority and the launch takes it from Java.
 */
public int fogFroxelSlices() {
    return (int)max(worldPush.fogVolume.y, 1.0);
}

/**
 * The depth a slice covers: geometric, because the near field needs centimetre-scale detail and the far
 * field only needs to not band, and a linear split of `reach` would spend almost all slices on the horizon.
 */
public float fogFroxelSliceDepth(int slice, int slices, float reach) {
    float ratio = log(max(reach, FOG_FROXEL_Z_NEAR * 2.0)) - log(FOG_FROXEL_Z_NEAR);
    return FOG_FROXEL_Z_NEAR * exp(ratio * (float(slice) + 0.5) / float(max(slices, 1)));
}

/**
 * Fill one voxel. `idx` is its position in the volume and `dims` the volume's own extent, so the launch is
 * the whole gather: one ray set per thread, no loop over slices.
 */
/**
 * Temporal accumulation, done in the gather's own write: the texel blends this frame's sample toward the
 * value it held last frame, so a voxel that flips between lit and shadowed every frame because four taps can
 * only ever answer four times settles into the gradient the eye actually expects. It is nearly free and needs
 * no second image because the grid is screen-locked rather than world-locked: each thread reads the one texel
 * it is about to write, so the volume IS its own history and there is nothing to reproject. Rotation is
 * therefore exact -- texels turn with the camera -- and translation is the only thing that slides new world
 * content into a texel, which is what the pushed weight in `fogVolume.z` is for: Java fades it out as the
 * camera's per-frame motion approaches the size of a voxel, and pushes 0 on the frame after the volume is
 * recreated, so an undefined first read is never trusted. MCRTX does the same in a separate pass
 * (`AccumulateInscatter` plus `previousVolumetricsAreValid`); folding it into the gather is the one place a
 * blend can be applied without a ping-pong pair, since the two must agree on the texel either way.
 */
static float4 fogAccumulateHistory(uint3 t, float4 fresh) {
    float w = clamp(worldPush.fogVolume.z, 0.0, 0.98);
    if (w <= 0.0) {
        return fresh;
    }
    return lerp(fresh, sunFroxels[t], float4(w));
}

public void fogGatherSunFroxels(uint3 idx, uint3 dims) {
    FogMedium f = fogMedium(worldPush);
    if (!f.active || idx.z >= dims.z || dims.x == 0u || dims.y == 0u) {
        return;
    }
    // The ray is the one the primary pass walks, minus the jitter: this grid is the frustum itself,
    // not the sample positions inside it, and mixing jitter in here would make every voxel's contents crawl.
    float2 uv = (float2(idx.xy) + 0.5) / float2(dims.xy);
    float2 ndc = uv * 2.0 - 1.0;
    float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float wn = nearH.w, wf = farH.w;
    if (wn == 0.0 || wf == 0.0) {
        // A degenerate ray for this column. Written as nothing through the history blend rather than left
        // alone: the volume is now persistent, so an early return here would pin whatever this texel held
        // last frame into every frame after it, on a condition that is true every frame.
        sunFroxels[uint3(idx.xy, idx.z)] = fogAccumulateHistory(uint3(idx.xy, idx.z),
                float4(0.0, 0.0, 0.0, 0.0));
        return;
    }
    float3 nearP = nearH.xyz / wn;
    float3 farP = farH.xyz / wf;
    float3 rayDir = farP - nearP;
    // The parameter this grid slices on is clip w, which is linear along the ray and is the same scalar the
    // read side recovers by projecting a tap — so a voxel and the taps inside it agree on depth without
    // either inverting a matrix, and the interpolation between slices measures distance rather than a proxy.
    float span = max(abs(wf - wn), 1.0e-4);
    float depth = fogFroxelSliceDepth(int(idx.z), int(dims.z), f.reach);
    float t = (depth - wn) / span;
    float3 voxelRel = nearP + rayDir * t;
    float3 voxel = voxelRel + worldPush.camOffset;

    // Froxel z-bounds: a voxel the primary ray sails past is inside geometry, and a shadow ray started inside
    // geometry has its first face on the *back* of the block that contains it, which is exactly the face the
    // secondary cull mode discards. So the ray leaves, and the voxel reports sun and open sky from the middle
    // of a wall. Nothing downstream can recover from that -- the march's taps sit in air, and the air is what
    // reads those neighbours -- so a shell of falsely-lit voxels around every surface in view was smearing
    // into the medium as light that seemed to come from no object at all. Behind the surface there is no air
    // to scatter, which is also the honest answer, so the voxel is written as nothing rather than left stale.
    float divisor = fogFroxelDivisor(worldPush);
    // How much of this voxel's light survives the depth bound: 1 in air, fading to 0 past a surface. Hoisted
    // out of the block below because both of the ray loops scale by it, and a voxel whose cell cannot be
    // looked up (an absurdly tiny render target) simply keeps 1.
    float bound = 1.0;
    if (dims.x > 1u && dims.y > 1u) {
        uint2 lo = idx.xy * uint(divisor);
        uint2 hi = lo + uint2(float2(divisor)) - 1u;
        float4 cornerZ = float4(gDepth[lo], gDepth[uint2(hi.x, lo.y)],
                gDepth[uint2(lo.x, hi.y)], gDepth[hi]);
        // The farthest surface in the cell, not the nearest: a cell that straddles an edge must not reject
        // the air in front of that edge, and a whole cell of error is already the grid's own resolution.
        float deviceZ = min(min(cornerZ.x, cornerZ.y), min(cornerZ.z, cornerZ.w));
        // Compared in the guides' own space: hardware depth, reversed, so 1 is near and 0 is sky. Pulling
        // that onto this voxel's ray needs no near/far constants for the two sides to agree on, and a sky
        // cell unprojects to the far plane, which rejects nothing.
        float4 surfH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, deviceZ, 1.0));
        if (surfH.w != 0.0) {
            float3 surfRel = surfH.xyz / surfH.w;
            float3 along = normalize(rayDir);
            // Allowed to sit this far past the surface: a quarter of the distance, which is the slice law's
            // own half-gap (32 geometric slices over `reach` is a factor of 1.2 per step), plus the cell's
            // footprint, which grows with depth at the same rate. Below a block it stops mattering, so the
            // floor keeps the near field from clipping against the geometry it is attached to.
            float slop = max(1.0, 0.25 * max(dot(surfRel - nearP, along), 0.0));
            float past = dot(voxelRel - surfRel, along) - slop;
            // Faded rather than cut: a per-cell hard zero would put the grid's own edges back into the
            // medium, which is the artefact this bound exists to remove. Over one slop's worth of distance
            // past the surface the voxel goes from full light to none, so a boundary between two cells is a
            // ramp the march's own interpolation can follow.
            bound = 1.0 - saturate(past / slop);
            if (bound <= 0.0) {
                // Past the fade, and behind a surface there is no air to scatter: written as nothing rather
                // than left holding whatever the last frame put here. Through the history blend, which is why
                // a shadow entering a voxel ramps in over a few frames instead of punching a hard hole in the
                // medium at whatever depth the surface happened to cross this frame.
                sunFroxels[uint3(idx.xy, idx.z)] = fogAccumulateHistory(uint3(idx.xy, idx.z),
                        float4(0.0, 0.0, 0.0, 0.0));
                return;
            }
        }
    }

    // The frame's dominant body, resolved the same way the indirect trace resolves it. Recomputed per voxel
    // rather than read from the module static the raygens publish, because this pass runs before the one that
    // fills it and a shadow field lagging a frame would show up as light that does not follow the sun.
    CelestialLight sun = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    // Fixed per voxel, never the frame's: a pattern that moves is a pattern the denoiser has to guess at.
    uint s = idx.x * 1973u + idx.y * 9277u + idx.z * 26699u;
    float sigma = fogDensityAt(f, voxel);
    float3 sunVis = float3(0.0, 0.0, 0.0);
    // The cheap-out is about not tracing at all, since these are the frame's expensive rays: air whose own
    // column says no direct light arrives already has the answer. Occlusion by geometry is the case that
    // needs the rays, and it gets them.
    int sunRays = fogSunRays();
    if (sunRays > 0 && fogColumnToLight(f, sigma, sun.dir) >= FOG_NEGLIGIBLE_LIGHT) {
        float3 sum = float3(0.0, 0.0, 0.0);
        for (int i = 0; i < sunRays; i++) {
            float3 dir = sampleSquare(sun.dir, sun.halfAngle, s);
            sum += visibility(voxel, dir, f.reach).transmittance;
        }
        sunVis = sum * (bound / float(sunRays));
    }
    // What the sky itself delivers here, as an in-scattering source: the mean radiance over the whole sphere
    // of (transmittance along the direction) x (the frame's sky-view LUT in that direction). That mean IS
    // the isotropic term of the phase integral -- 1/4pi of the incoming radiance, which is the normalisation
    // fogPhase already carries for the body -- so the two terms need no relative fudge factor, and the LUT
    // already holds the atmosphere's own forward scattering, which is why haze brightens toward the sun
    // without the sun's disc being in it. The disc is not in the LUT at all, so there is nothing to
    // double-count against the direct term above.
    SkyState sky = skyState(worldPush);
    float3 ambient = float3(0.0, 0.0, 0.0);
    int skyRays = fogSkyRays();
    for (int i = 0; i < skyRays; i++) {
        float cosTheta = 1.0 - 2.0 * (float(i) + 0.5) / float(skyRays);
        float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
        float azimuth = 6.28318530718 * (float(i) * 0.618034);
        float3 dir = float3(cos(azimuth) * sinTheta, cosTheta, sin(azimuth) * sinTheta);
        float2 sunUv;
        float2 moonUv;
        skyDomeUv(dir, sky, sunUv, moonUv);
        float3 inSky = skyViewLut.SampleLevel(sunUv, 0.0).rgb + skyViewLut.SampleLevel(moonUv, 0.0).rgb;
        ambient += visibility(voxel, dir, f.reach).transmittance * inSky;
    }
    sunFroxels[uint3(idx.xy, idx.z)] = fogAccumulateHistory(uint3(idx.xy, idx.z), float4(sunVis,
            skyRays > 0 ? bound * dot(ambient, FOG_LUMA) / float(skyRays) : 0.0));
}

/**
 * Spatial filter, one thread per voxel, no rays: a 3x3x3 box over the gathered volume, mixed into the
 * gathered value channel by channel. It exists because the grid is coarse relative to what it now carries --
 * on a canopy the sun test returns 0 or 1 per voxel, and 261k independent coin flips read as puffs of light
 * hanging between the leaves rather than as shade. A filter is also what MCRTX does, and its asymmetry is
 * worth copying exactly: the pass list has BlurGIInscatter and NO blur for the sun's in-scatter, because the
 * ambient is a veil that can be smoothed freely while the transmittance IS the shaft. So the ambient is
 * averaged outright and the sun's own value mostly, which keeps a wide beam intact: averaging is the identity
 * on anything wider than a voxel, and the only structures that soften are the ones the grid cannot carry
 * anyway -- which is why this runs on the volume (once per voxel per frame) instead of in the march (once per
 * tap per segment per bounce per sample).
 *
 * It is a plain weighted box, empties included: a blur that ignores the voxels holding nothing would lift
 * shadow toward light instead of averaging it, which is a contrast loss dressed as a denoiser.
 */
public void fogFilterSunFroxels(uint3 idx, uint3 dims) {
    if (idx.z >= dims.z || dims.x == 0u || dims.y == 0u) {
        return;
    }
    float4 src = sunFroxels[uint3(idx.xy, idx.z)];
    float strength = clamp(worldPush.fogGather.z, 0.0, 2.0);
    if (strength <= 0.0) {
        sunFroxelsFiltered[uint3(idx.xy, idx.z)] = src;
        return;
    }
    int radius = strength > 1.0 ? FOG_FILTER_MAX_RADIUS : 1;
    float4 sum = float4(0.0, 0.0, 0.0, 0.0);
    float weight = 0.0;
    int3 centre = int3(idx.xy, int(idx.z));
    for (int dz = -radius; dz <= radius; dz++) {
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int3 c = centre + int3(dx, dy, dz);
                // Clamped on both axes: at the volume's own edge a voxel averages the edge rather than
                // reaching outside the launch, which is also what keeps the screen border from filtering
                // toward black. Depth is weighted down because adjacent slices are far apart in world units
                // out there, and a shaft's own length already smooths along the view ray.
                c.xy = clamp(c.xy, int2(0, 0), int2(dims.xy) - 1);
                c.z = clamp(c.z, 0, int(dims.z) - 1);
                float s = (dz == 0) ? 1.0 : FOG_FILTER_SLICE_WEIGHT;
                sum += sunFroxels[uint3(c.xy, uint(c.z))] * s;
                weight += s;
            }
        }
    }
    // Every voxel in the box contributes, empties included. Skipping the empty ones -- which is what an
    // "adaptive" filter tempts you into -- is not a blur but a max-pool toward light: a shadowed voxel next
    // to one lit neighbour would take that neighbour's value at full weight, so a shaft's dark edge filled in
    // and the lit voxels around it spread, which is precisely the halo this filter was written to remove.
    float4 avg = sum / max(weight, 1.0e-4);
    float sunW = FOG_FILTER_SUN_WEIGHT * strength;
    float skyW = FOG_FILTER_AMBIENT_WEIGHT * min(strength, 1.0);
    sunFroxelsFiltered[uint3(idx.xy, idx.z)] = lerp(src, avg,
            float4(sunW, sunW, sunW, skyW));
}

/**
 * Fog diagnostics for the current thread, live only while the debug view is on: the module-static pattern the
 * indirect trace already uses for its dominant light, because the two things that need them (the march, and
 * the raygen that stores the frame) are four functions apart and widening `fogIntegrateSegment`'s signature a
 * second time for an inspection feature would be the worse trade.
 */
public static float3 fogDbgScatter;
public static float fogDbgExtinct;
public static float fogDbgSlice;
public static float fogDbgScale;

/** Start a path's fog diagnostics. `invSamples` keeps the reported values independent of the sample count. */
public void fogDebugBegin(float invSamples) {
    fogDbgScatter = float3(0.0, 0.0, 0.0);
    fogDbgExtinct = 0.0;
    fogDbgSlice = 0.0;
    fogDbgScale = invSamples;
}

/**
 * The fog's own contribution to a path, as the debug view wants it: in-scatter (red), the fraction of the
 * ray's light the medium removed (green), and the volume's slice the last sample came from (blue).
 */
public float3 fogDebugColour() {
    // Log-mapped: the scatter term is scene radiance and spans many orders of magnitude, and a linear map
    // would show only the pixels next to the sun.
    float scatter = dot(fogDbgScatter, FOG_LUMA) * max(fogDbgScale, 0.0);
    return float3(log2(1.0 + max(scatter, 0.0)) / 8.0, saturate(fogDbgExtinct * max(fogDbgScale, 0.0)),
            fogDbgSlice);
}

static float4 fogFroxelTexel(int2 c, int slice, float2 dims) {
    // Clamped on both axes rather than only where the weight is nonzero: a zero-weight out-of-range index is
    // still out of range, and the edge texels replicating is what keeps a tap near the screen border from
    // reading whatever the allocator left there.
    int2 cc = clamp(c, int2(0, 0), int2(dims) - 1);
    int s = clamp(slice, 0, fogFroxelSlices() - 1);
    return sunFroxelsFiltered[uint3(cc, uint(s))];
}

/**
 * What light reaches a point in the medium: the volume, trilineared. Beyond the last slice the answer holds,
 * because so does the medium's own `reach` — the far field is one long attenuated stretch either way.
 */
// One slice, bilineared across the column. Split out because a slice is a 2x2 footprint and the read wants
// two of them, and writing both inline reads like arithmetic rather than like a lookup.
static float4 fogFroxelLayer(int2 b, float2 w, int slice, float2 dims) {
    return lerp(lerp(fogFroxelTexel(b, slice, dims), fogFroxelTexel(b + int2(1, 0), slice, dims), w.x),
            lerp(fogFroxelTexel(b + int2(0, 1), slice, dims),
                    fogFroxelTexel(b + int2(1, 1), slice, dims), w.x), w.y);
}

public float4 fogSunFroxels(float3 p, float2 renderSize, float reach) {
    float4 clip = mul(worldPush.curViewProj, float4(p - worldPush.camOffset, 1.0));
    if (clip.w <= 0.0) {
        // Behind the camera, which reflection segments reach constantly: the nearest edge of the grid is the
        // least wrong answer, and unlike a zero it does not punch a dark hole in a mirror.
        clip.w = 1.0e-4;
    }
    float2 ndc = clip.xy / clip.w;
    float2 dims = fogFroxelDims(renderSize, fogFroxelDivisor(worldPush));
    float2 g = clamp((ndc * 0.5 + 0.5) * dims - 0.5, float2(0.0, 0.0), dims - 1.0);
    float2 bf = floor(g);
    float2 w = g - bf;
    int2 b = int2(bf);
    float slices = float(fogFroxelSlices());
    float ratio = log(max(reach, FOG_FROXEL_Z_NEAR * 2.0)) - log(FOG_FROXEL_Z_NEAR);
    float zf = clamp(log(max(clip.w, FOG_FROXEL_Z_NEAR)) - log(FOG_FROXEL_Z_NEAR), 0.0, ratio)
            / ratio * slices - 0.5;
    float zfrac = zf - floor(zf);
    int z0 = int(floor(zf));
    // Reported by the debug view so a wrong VALUE and a wrong MAPPING are distinguishable on screen: this is
    // the slice the tap landed in, normalised over the volume's depth; its steps are the slice boundaries.
    fogDbgSlice = (float(z0) + 0.5) / max(float(fogFroxelSlices()), 1.0);
    // Eight texels: bilinear in the column, linear across slices. That interpolation is what lets a coarse
    // grid carry a soft boundary — the march asks for a point, not for a texel, so a shaft edge lands where
    // it lands rather than on whichever slice it fell into.
    return lerp(fogFroxelLayer(b, w, z0, dims), fogFroxelLayer(b, w, z0 + 1, dims), zfrac);
}

/**
 * Light reaching a fog tap, in `sun.illuminance`'s units: the analytic column times the volume's sun
 * transmittance, plus the sky term that volume's alpha carries.
 *
 * <p>The column and the field answer different questions and both are needed. The column integrates the
 * medium's own profile toward the body, which is what reddens light through a long horizontal path and what
 * tints it through a pane a tap happens to be behind; a single sample at the end of a segment could never
 * know either, because they vary along the way. The field knows about *geometry*, which a profile cannot
 * see: a roof, a wall, sixty blocks of hillside.
 */
public float3 fogScatterAt(FogMedium f, CelestialLight sun, float sigma, float phase, float3 sunVis,
        float domeRadiance) {
    // The sun's own path through the medium is the analytic column; the volume says what geometry leaves of
    // it here. The two are multiplied rather than merged because they answer for different things, and a
    // reader that saw only one of them would read the other as a bug: the column is what tints a tap that
    // happens to sit behind a pane, the volume is what keeps a sealed room's air from glowing.
    float3 direct = phase * sun.illuminance * fogColumnToLight(f, sigma, sun.dir) * sunVis;
    // `domeRadiance` is the gather's mean of (transmittance x sky) over the whole sphere, which is already
    // 1/4pi of the incoming radiance -- the same normalisation fogPhase carries -- so the two terms need no
    // factor between them: this is first-order in-scatter from the sky, not a guess at its brightness. It is
    // attenuated by the medium above the tap, which for this profile is exactly sigma times the scale height:
    // the vertical integral of an exponential is itself. Without that, thick fog would keep receiving its
    // full skylight from a sky it can no longer see and would glow at the density settings where the sun's
    // own term has correctly gone to nothing.
    float3 ambient = FOG_SKY_TINT * domeRadiance * exp(-sigma * f.scaleHeight);
    return f.albedo * (direct + ambient);
}

/**
 * Transport through the medium along one segment of a path. This is the renderer's single seam for what
 * happens to radiance in transit: it REPLACES the plain Beer-Lambert multiply, so a dielectric body's own
 * absorption and the fog's in-scatter are decided by one function and cannot disagree about which medium
 * the segment lay in or double-charge for it.
 *
 * `T` multiplies the running throughput; `S` is radiance to add, already weighted by the transmittance from
 * the segment start to each tap, so a caller must not scale it again.
 *
 * The segment is split into a resolved near field plus one remainder tap. `maxSteps` taps of
 * `reach / maxSteps` cover the region where density and the sun column actually vary; the rest is
 * integrated once at its midpoint. Spreading the same tap count over a whole escaped ray instead would put
 * most of the atmosphere inside a single tap and band the horizon, while the error left in the far field
 * sits where transmittance has already gone to nothing.
 */
/** Where tap `i` ends, on a square law over the resolved part of a segment: near taps are dense, far ones
 * are sparse, and one tap alone still covers the whole stretch exactly as an even split would. */
static float fogTapEdge(float resolved, int steps, int i) {
    float u = clamp(float(i) / float(max(steps, 1)), 0.0, 1.0);
    return resolved * u * u;
}

public void fogIntegrateSegment(MediumStack medium, FogMedium f, CelestialLight sun,
                         float3 ro, float3 rd, float t, float2 size,
                         out float3 T, out float3 S) {
    T = float3(1.0, 1.0, 1.0);
    S = float3(0.0, 0.0, 0.0);
    if (t <= 0.0) {
        return;
    }

    // Inside a volume dielectric, that body owns the transport: its extinction is per-body, calibrated per
    // block of depth, and it is not in-scattering here.
    if (medium.current.water || any(medium.current.extinction > 0.0)) {
        T = exp(-medium.current.extinction * t);
        return;
    }
    if (!f.active) {
        return;
    }

    // The phase depends only on the segment's direction, so it is evaluated once for the whole march; the
    // profile and the sun column are what genuinely vary from tap to tap.
    float phase = fogPhase(f, dot(-rd, sun.dir));
    float tapLen = clamp(t / float(f.maxSteps), FOG_TAP_MIN, max(f.reach / float(f.maxSteps), FOG_TAP_MIN));
    float resolved = min(t, tapLen * float(f.maxSteps));
    int steps = min(max((int)ceil(resolved / tapLen), 1), f.maxSteps);
    // Taps are placed on a quadratic in the segment, not evenly: density and the sun column both change
    // fastest where the ray starts, and a march that spends its budget where the light varies needs no random
    // offset to look continuous -- EscheridiaRT's march jitters its steps because it accumulates the whole
    // image for hundreds of frames, and a deterministic bias is the half of that which costs no noise here.
    // `fogTapEdge(i)` is the far edge of tap i, so tap i covers [edge(i-1), edge(i)] and samples its middle.

    // Light is looked up per tap rather than per segment, which is the entire point of the volume: a
    // stretch of air that starts in a sunbeam and ends in a building's shade has to scatter accordingly at
    // both ends, and one value for the segment would have to pick a side. The lookup is eight texel reads,
    // so this is not the expensive part of the march -- the rays that filled them were, and they were paid
    // once per voxel for the frame instead of once per tap per bounce per sample.

    for (int i = 0; i < steps; i++) {
        float t0 = fogTapEdge(resolved, steps, i);
        float t1 = fogTapEdge(resolved, steps, i + 1);
        float dt = max(t1 - t0, 1.0e-3);
        float3 tapMid = ro + rd * (0.5 * (t0 + t1));
        float sigma = fogDensityAt(f, tapMid);
        // exp(-sigma*dt) and (1 - exp(-sigma*dt)) are the constant-medium transmittance and the same
        // integral divided by sigma, so a thin tap costs the same expression as a thick one and neither
        // divides by a small number.
        float segmentT = exp(-sigma * dt);
        float4 lit = fogSunFroxels(tapMid, size, f.reach);
        float3 litAt = fogScatterAt(f, sun, sigma, phase, lit.rgb, lit.a) * (1.0 - segmentT);
        S += T * litAt;
        // Both diagnostics are what the *ray* lost or gained at this segment, so they use the same T the
        // real terms do and stay in the same units the pixel is composed in: scatter in radiance, the
        // extinct term as a fraction of the light that reached this far.
        fogDbgScatter += T * litAt;
        fogDbgExtinct += dot(T, FOG_LUMA) * (1.0 - segmentT);
        T *= segmentT;
    }

    if (t > resolved) {
        float far = t - resolved;
        float3 farMid = ro + rd * (resolved + 0.5 * far);
        float sigma = fogDensityAt(f, farMid);
        float segmentT = exp(-sigma * far);
        float4 lit = fogSunFroxels(farMid, size, f.reach);
        float3 litAt = fogScatterAt(f, sun, sigma, phase, lit.rgb, lit.a) * (1.0 - segmentT);
        S += T * litAt;
        // Both diagnostics are what the *ray* lost or gained at this segment, so they use the same T the
        // real terms do and stay in the same units the pixel is composed in: scatter in radiance, the
        // extinct term as a fraction of the light that reached this far.
        fogDbgScatter += T * litAt;
        fogDbgExtinct += dot(T, FOG_LUMA) * (1.0 - segmentT);
        T *= segmentT;
    }
}

/**
 * The camera-to-first-interface segment, re-derived by the indirect pass for the one record pass A shaded
 * away instead of recording.
 *
 * A primary hit on an opaque block leaves a record that still starts at the eye, so the radiance trace
 * walks that stretch itself and needs nothing here. A dielectric hit is different: pass A consumes the
 * interface, records a continuation that starts at the surface, and keeps that segment's absorption in the
 * record's throughput — but throughput is a multiplier, so the light the air scattered into that stretch
 * has nowhere to live. Reconstructing the ray is cheaper than widening the record ABI to carry it, and
 * doing it here keeps the budget honest: the prefix is added once per sample, where the record it belongs
 * to is read by every SPP resample and by both halves of a split, and it pays for light the same way any
 * other segment does. The unprojection below mirrors primary.rgen
 * except for the far plane it never needs, jitter included, because the prefix has to end on the surface
 * that ray hit. `hitPos` is the biased one the record carries, so the length is off by at most SURF_BIAS
 * against a scale height of blocks.
 */
public void fogPrimarySegment(WorldPush push, FogMedium f, CelestialLight sun, MediumStack medium,
                       float2 pix, float2 size, float3 hitPos,
                       out float3 T, out float3 S) {
    T = float3(1.0, 1.0, 1.0);
    S = float3(0.0, 0.0, 0.0);
    // Deliberately NOT the full segment transport here. Pass A already multiplied this stretch's own body
    // absorption into the record's throughput, so re-charging a medium would attenuate submerged water (or
    // a first glass pane) twice, and the aerial medium does not apply inside another body in the first
    // place. The prefix therefore exists purely for what a multiplier cannot carry: in-scattered light.
    if (!f.active || medium.current.water || any(medium.current.extinction > 0.0)) {
        return;
    }
    float2 uv = (pix + 0.5) / size;
    float2 jndc = (uv + push.jitter / size) * 2.0 - 1.0;
    float4 nearH = mul(push.invViewProj, float4(jndc.x, jndc.y, 1.0, 1.0));
    float3 eye = nearH.xyz / nearH.w + push.camOffset;
    float3 toHit = hitPos - eye;
    float hitDistance = length(toHit);
    if (hitDistance <= 1.0e-4) {
        return;
    }
    fogIntegrateSegment(medium, f, sun, eye, toHit / hitDistance, hitDistance, size, T, S);
}
