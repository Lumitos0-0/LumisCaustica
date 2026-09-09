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
 * sun, alpha is how much of the sky the voxel can see. Both are estimated from a few FIXED directions,
 * which is what lets the march sample a coarse structure without noise or crawl, and the whole frame pays
 * `FOG_FROXEL_SUN_TAPS + FOG_FROXEL_SKY_TAPS` rays per voxel instead of one per tap per bounce per sample.
 *
 * <p>Screen-aligned columns, world-correct contents: a voxel is defined by the view ray through its column
 * at a given depth, so the grid thins out with distance the way the eye needs it to (a near voxel is
 * centimetres wide, a far one is many blocks) and no reprojection is involved — the grid is rebuilt every
 * frame from the current camera, so nothing has to be revalidated when the player turns.
 */
public static const int FOG_FROXEL_SLICES = 32;      // log-depth slices; Java's volume depth
public static const int FOG_FROXEL_DIVISOR = 16;     // screen pixels per voxel edge, both axes
public static const int FOG_FROXEL_SUN_TAPS = 4;
public static const int FOG_FROXEL_SKY_TAPS = 4;
public static const float FOG_FROXEL_Z_NEAR = 1.0;   // blocks: the first slice's inner edge
// The sky's share of in-scattering, for the light that arrives from every direction except the body's.
// Fog ambient is a veil rather than a light source, so it is estimated from the sky's openness alone: a
// cool tint at a fixed fraction of the dominant body's illuminance, which follows the body from noon to
// moonlight without a second colour term, and which needs no occlusion test to be right about a sealed room
// because the openness estimate already is that test.
public static const float FOG_SKY_LUMINANCE = 0.06;
// BT.709 weights, the same ones the sky module uses to decide which body is brighter.
public static const float3 FOG_LUMA = float3(0.2126, 0.7152, 0.0722);
public static const float3 FOG_SKY_TINT = float3(0.62, 0.75, 1.0);

/** The volume's extent in voxels. Both sides derive it from the render size, so it needs no push field. */
public float2 fogFroxelDims(float2 renderSize) {
    return max(float2(32.0, 20.0), floor(renderSize / float(FOG_FROXEL_DIVISOR)));
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
public void fogGatherSunFroxels(uint3 idx, uint3 dims) {
    FogMedium f = fogMedium(worldPush);
    if (!f.active || idx.z >= uint(FOG_FROXEL_SLICES) || dims.x == 0u || dims.y == 0u) {
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
    float3 voxel = nearP + rayDir * t + worldPush.camOffset;

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
    if (fogColumnToLight(f, sigma, sun.dir) >= FOG_NEGLIGIBLE_LIGHT) {
        float3 sum = float3(0.0, 0.0, 0.0);
        for (int i = 0; i < FOG_FROXEL_SUN_TAPS; i++) {
            float3 dir = sampleSquare(sun.dir, sun.halfAngle, s);
            sum += visibility(voxel, dir, f.reach).transmittance;
        }
        sunVis = sum / float(FOG_FROXEL_SUN_TAPS);
    }
    // Sky openness, cosine weighted so a voxel under a canopy reports the light it can actually gather
    // rather than the fraction of the dome it can see. Uniform hemisphere directions, golden-angle spread.
    float open = 0.0;
    for (int i = 0; i < FOG_FROXEL_SKY_TAPS; i++) {
        float cosTheta = sqrt((float(i) + 0.5) / float(FOG_FROXEL_SKY_TAPS));
        float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
        float azimuth = 6.28318530718 * (float(i) * 0.618034 + 0.5);
        float3 dir = float3(cos(azimuth) * sinTheta, cosTheta, sin(azimuth) * sinTheta);
        // Reduced to one number, unlike the sun's transmittance: the ambient's colour is this pass's tint
        // rather than whatever glass happens to be overhead, because four directions through a dome is not a
        // spectrum worth keeping and the veil it produces would not show one.
        open += dot(visibility(voxel, dir, f.reach).transmittance, FOG_LUMA) * cosTheta;
    }
    sunFroxels[uint3(idx.xy, idx.z)] = float4(sunVis, open / float(FOG_FROXEL_SKY_TAPS));
}

static float4 fogFroxelTexel(int2 c, int slice, float2 dims) {
    // Clamped on both axes rather than only where the weight is nonzero: a zero-weight out-of-range index is
    // still out of range, and the edge texels replicating is what keeps a tap near the screen border from
    // reading whatever the allocator left there.
    int2 cc = clamp(c, int2(0, 0), int2(dims) - 1);
    int s = clamp(slice, 0, FOG_FROXEL_SLICES - 1);
    return sunFroxels[uint3(cc, uint(s))];
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
    float2 dims = fogFroxelDims(renderSize);
    float2 g = clamp((ndc * 0.5 + 0.5) * dims - 0.5, float2(0.0, 0.0), dims - 1.0);
    float2 bf = floor(g);
    float2 w = g - bf;
    int2 b = int2(bf);
    float slices = float(FOG_FROXEL_SLICES);
    float ratio = log(max(reach, FOG_FROXEL_Z_NEAR * 2.0)) - log(FOG_FROXEL_Z_NEAR);
    float zf = clamp(log(max(clip.w, FOG_FROXEL_Z_NEAR)) - log(FOG_FROXEL_Z_NEAR), 0.0, ratio)
            / ratio * slices - 0.5;
    float zfrac = zf - floor(zf);
    int z0 = int(floor(zf));
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
        float skyOpen) {
    // The sun's own path through the medium is the analytic column; the volume says what geometry leaves of
    // it here. The two are multiplied rather than merged because they answer for different things, and a
    // reader that saw only one of them would read the other as a bug: the column is what tints a tap that
    // happens to sit behind a pane, the volume is what keeps a sealed room's air from glowing.
    float3 direct = phase * sun.illuminance * fogColumnToLight(f, sigma, sun.dir) * sunVis;
    float3 ambient = FOG_SKY_TINT * (FOG_SKY_LUMINANCE * sun.illuminance) * skyOpen;
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
    float tapLen = max(f.reach / float(f.maxSteps), 1.0e-3);
    float resolved = min(t, tapLen * float(f.maxSteps));
    int steps = min(max((int)ceil(resolved / tapLen), 1), f.maxSteps);
    float dt = resolved / float(steps);

    // Light is looked up per tap rather than per segment, which is the entire point of the volume: a
    // stretch of air that starts in a sunbeam and ends in a building's shade has to scatter accordingly at
    // both ends, and one value for the segment would have to pick a side. The lookup is eight texel reads,
    // so this is not the expensive part of the march -- the rays that filled them were, and they were paid
    // once per voxel for the frame instead of once per tap per bounce per sample.

    for (int i = 0; i < steps; i++) {
        float3 tapMid = ro + rd * (dt * (float(i) + 0.5));
        float sigma = fogDensityAt(f, tapMid);
        // exp(-sigma*dt) and (1 - exp(-sigma*dt)) are the constant-medium transmittance and the same
        // integral divided by sigma, so a thin tap costs the same expression as a thick one and neither
        // divides by a small number.
        float segmentT = exp(-sigma * dt);
        float4 lit = fogSunFroxels(tapMid, size, f.reach);
        S += T * fogScatterAt(f, sun, sigma, phase, lit.rgb, lit.a) * (1.0 - segmentT);
        T *= segmentT;
    }

    if (t > resolved) {
        float far = t - resolved;
        float3 farMid = ro + rd * (resolved + 0.5 * far);
        float sigma = fogDensityAt(f, farMid);
        float segmentT = exp(-sigma * far);
        float4 lit = fogSunFroxels(farMid, size, f.reach);
        S += T * fogScatterAt(f, sun, sigma, phase, lit.rgb, lit.a) * (1.0 - segmentT);
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
