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
// incident-radiance meter amortized across the volume. The height profile and the medium's own extinction
// toward the light stay closed form here, so one segment of one path costs two shadow rays no meter could
// have saved; the volume is still where per-tap light resolution -- every light, every bounce, at a
// fraction of the rays -- has to come from.
//
// Geometry gets a vote: each region of a segment asks one shadow ray whether the tap region sees the
// directional body at all. Without it a sealed room in-scatters at full daylight and the volume reads as a
// uniform veil, because the analytic height profile below cannot tell a roof from clear air. One ray per
// region (not per tap) is the deliberate trade: per-tap visibility is what the froxel incident-light cache
// amortizes, at a fraction of the rays, and it is the reason a light meter earns its keep here at all.
//
// Density is a function of ABSOLUTE world altitude: path positions are rebased to the terrain origin,
// which moves as the camera travels, so every altitude goes through `rebaseY` (that origin's block Y,
// pushed per frame). An exponential would otherwise slide with the player. The profile is flat at and
// below `refY` rather than growing without bound downward, so a cave's air is no denser than a
// mountaintop's; what makes caves and interiors dark is the shadow ray above, not the profile.

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
 * heading down or level has no such bound — density only grows — so those integrate over the pushed reach
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
// Sun-disc samples per shadow-field texel. Four is enough to turn a hard edge into a gradient at this
// resolution, and the field is written once per pixel of the *frame*, so it is not multiplied by bounces
// or samples per pixel the way a per-tap test is.
public static const int FOG_SUN_SHADOW_TAPS = 4;

/**
 * The sun's shadow as a per-pixel field: pass A writes it from the surface it just resolved, and every fog
 * tap of every bounce of that pixel reads it back as a fetch.
 *
 * <p>This is the shape MCRTX uses. Its `SunShadowRayGenInline` writes `outputBufferSunLightShadow` at
 * render size, `ShadowDenoising` (separable, ping-ponged) and the checkerboard field blend filter that
 * against `previousSunLightShadowBuffer`, and the in-scatter step *samples* the result rather than testing
 * the sun per sample. The filtering is why their shafts are clean, so Caustica needs the shared field more
 * than it needs a denoiser: it already averages the field across bounces, SPP resamples and neighbouring
 * pixels, and it has nothing to filter out, because the taps are fixed rather than random.
 *
 * <p>Before this there was a jittered `visibility()` ray per fog segment, which is exactly as quiet as the
 * sample budget makes it -- sharp, correct, and grainy along beam edges at one sample per pixel. The field
 * is cheaper at any budget above one bounce and does not need samples to converge.
 *
 * <p>Determinism is the point of that last part. A jittered ray per sample is only as quiet as the sample
 * budget can make it, and a path that runs out of budget shows up as grain along beam edges -- the worst
 * place for it, because the eye reads flicker in a high-contrast edge far more readily than in the flat
 * haze between. Fixed offsets per texel give a field that is bit-identical frame to frame: beam edges that
 * are soft because they average several rays, not because ten frames averaged.
 *
 * <p>The field is half resolution in each axis, which is what buys the softness rather than costing it:
 * four pixels share a texel and the read interpolates, so a penumbra about a pixel wide still lands as a
 * gradient. One quarter of the texels at four taps each is one ray per pixel per frame for the whole
 * shadowing problem, whatever the sample budget.
 */
public void fogWriteSunShadow(int2 pix, float2 size, float3 surface, bool hasSurface, CelestialLight sun) {
    FogMedium f = fogMedium(worldPush);
    if (!f.active) {
        return;
    }
    int2 fieldSize = int2((size + 1.0) * 0.5);
    int2 texel = int2((float2(pix) + 0.5) * 0.5);
    // One thread per texel, the top-left pixel of its block. All four pixels of the block would otherwise
    // race to write it, and a race for last write wins is precisely the frame-to-frame instability that this
    // whole design exists to avoid.
    if (fieldSize.x <= 0 || fieldSize.y <= 0 || any(texel * 2 != pix)) {
        return;
    }
    float3 value = float3(0.0, 0.0, 0.0);
    float sigma = fogDensityAt(f, surface);
    // The same cheap-out the march used to pay for itself with: where the medium's own column carries no
    // light at this altitude, the answer is already zero and the four rays that would prove it are the
    // expensive part of the frame. Caves and sealed rooms therefore cost a store, not traces.
    if (hasSurface && fogColumnToLight(f, sigma, sun.dir) >= FOG_NEGLIGIBLE_LIGHT) {
        // Fixed per texel, deliberately not the path's seed: mixing the frame index in here is what turns
        // a shadow field into noise the denoiser then has to guess at.
        uint s = uint(pix.x) * 1973u + uint(pix.y) * 9277u + 26699u;
        float3 sum = float3(0.0, 0.0, 0.0);
        for (int i = 0; i < FOG_SUN_SHADOW_TAPS; i++) {
            // A point on the body's square disc, offset off the surface along it so the ray leaves the face
            // it started on. With halfAngle at zero this is the exact direction, which is what a moonless
            // pinhole look would ask for.
            float3 dir = sampleSquare(sun.dir, sun.halfAngle, s);
            sum += visibility(surface + dir * SURF_BIAS, dir, f.reach).transmittance;
        }
        value = sum / float(FOG_SUN_SHADOW_TAPS);
    }
    // The alpha is not spare precision, it is the anchor. A primary ray that escaped to the sky resolved no
    // surface, and borrowing the eye's occlusion for the horizon haze would dim it whenever the camera
    // stands in shade. alpha=0 says "nothing here yet", and the reader answers with the analytic column
    // alone -- unshadowed, which is what the distant sky's fog used to get, and still does.
    gSunShadow[texel] = float4(value, hasSurface ? 1.0 : 0.0);
}

/**
 * Read the field for a pixel: bilinear across the half-resolution texels, by hand because a storage image
 * has no sampler of its own. What this replaces is a `visibility()` ray per fog segment: same four
 * questions answered, four loads instead of four traces, and one lookup per path instead of per tap.
 */
public float3 fogSunShadow(float2 pix, float2 size) {
    int2 fieldSize = int2((size + 1.0) * 0.5);
    if (fieldSize.x <= 0 || fieldSize.y <= 0) {
        return float3(1.0, 1.0, 1.0);
    }
    float2 p = clamp((pix + 0.5) * 0.5 - 0.5, float2(0.0, 0.0), float2(fieldSize) - 1.0);
    int2 lo = int2(floor(p));
    float2 w = p - float2(lo);
    // All four corners are clamped, not just the far ones: at the last texel the +1 neighbour exists only as
    // a zero-weight term, and a zero-weight out-of-range index is still out of range.
    int2 x1 = int2(min(lo.x + 1, fieldSize.x - 1), lo.y);
    int2 y1 = int2(lo.x, min(lo.y + 1, fieldSize.y - 1));
    int2 c11 = min(lo + int2(1, 1), fieldSize - 1);
    float4 a = lerp(lerp(gSunShadow[lo], gSunShadow[x1], w.x),
            lerp(gSunShadow[y1], gSunShadow[c11], w.x), w.y);
    // The anchor interpolates with the value, so a texel that resolved no surface pulls its share of the
    // blend toward "unshadowed" instead of toward zero. No divide: alpha is a weight, not a count.
    return lerp(float3(1.0, 1.0, 1.0), a.xyz, a.w);
}

/**
 * Sun light reaching a fog tap, in `sun.illuminance`'s units: the analytic column times the shadow field.
 *
 * <p>The column and the field answer different questions and both are needed. The column integrates the
 * medium's own profile toward the body, which is what reddens light through a long horizontal path and what
 * tints it through a pane a tap happens to be behind; a single sample at the end of a segment could never
 * know either, because they vary along the way. The field knows about *geometry*, which a profile cannot
 * see: a roof, a wall, sixty blocks of hillside.
 */
public float3 fogScatterAt(FogMedium f, CelestialLight sun, float sigma, float phase) {
    return f.albedo * phase * sun.illuminance * fogColumnToLight(f, sigma, sun.dir);
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
                         float3 ro, float3 rd, float t, float3 segmentLight,
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

    // The pixel's sun-shadow value arrives from the caller, already resolved: one lookup per traced path,
    // shared by every segment of it. Both regions below multiply by the same value, so a shadow cannot
    // apply to the near half of a stretch and not the far half -- a beam is either there for this pixel or
    // it is not, and anything else reads as a seam sliding across the image as the camera moves.

    for (int i = 0; i < steps; i++) {
        float3 tapMid = ro + rd * (dt * (float(i) + 0.5));
        float sigma = fogDensityAt(f, tapMid);
        // exp(-sigma*dt) and (1 - exp(-sigma*dt)) are the constant-medium transmittance and the same
        // integral divided by sigma, so a thin tap costs the same expression as a thick one and neither
        // divides by a small number.
        float segmentT = exp(-sigma * dt);
        S += T * fogScatterAt(f, sun, sigma, phase) * segmentLight * (1.0 - segmentT);
        T *= segmentT;
    }

    if (t > resolved) {
        float far = t - resolved;
        float3 farMid = ro + rd * (resolved + 0.5 * far);
        float sigma = fogDensityAt(f, farMid);
        float segmentT = exp(-sigma * far);
        S += T * fogScatterAt(f, sun, sigma, phase) * segmentLight * (1.0 - segmentT);
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
 * to is read by every SPP resample and by both halves of a split. It pays the same single light ray as any
 * other segment, so a frame over water costs one more. The unprojection below mirrors primary.rgen
 * except for the far plane it never needs, jitter included, because the prefix has to end on the surface
 * that ray hit. `hitPos` is the biased one the record carries, so the length is off by at most SURF_BIAS
 * against a scale height of blocks.
 */
public void fogPrimarySegment(WorldPush push, FogMedium f, CelestialLight sun, MediumStack medium,
                       float2 pix, float2 size, float3 hitPos, float3 segmentLight,
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
    fogIntegrateSegment(medium, f, sun, eye, toHit / hitDistance, hitDistance, segmentLight, T, S);
}
