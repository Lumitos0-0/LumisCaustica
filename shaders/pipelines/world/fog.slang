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
// incident-radiance meter amortized across the volume, which pays off only when the volume is sampled
// millions of times; a per-segment sun term is one closed-form column, so nothing needs amortizing yet.
//
// Density is a function of ABSOLUTE world altitude: path positions are rebased to the terrain origin,
// which moves as the camera travels, so every altitude goes through `rebaseY` (that origin's block Y,
// pushed per frame). An exponential would otherwise slide with the player. The profile is flat at and
// below `refY` rather than growing without bound downward, so a deep cave stays visible.

import world_common;
import world_core;
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

/**
 * What one tap scatters toward the eye, in the units this march wants: the tap's in-scattered radiance is
 * this value times (1 - exp(-sigma*dt)), because integrating a constant medium over a length divides a
 * sigma out and the sigma_s/sigma ratio of a uniform-albedo profile is just the albedo.
 *
 * Only the directional body lights the volume today, and its illuminance is the one terrain already pays
 * for — the column integral is the entire extra cost. Emitter light is the missing half: a torch currently
 * dims the fog in front of it rather than glowing in it. That needs incident light sampled per tap, and
 * per tap is exactly where a meter would plug in: its term is a lookup by the tap's position, the one
 * argument this signature still lacks.
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
                         float3 ro, float3 rd, float t,
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

    for (int i = 0; i < steps; i++) {
        float3 mid = ro + rd * (dt * (float(i) + 0.5));
        float sigma = fogDensityAt(f, mid);
        // exp(-sigma*dt) and (1 - exp(-sigma*dt)) are the constant-medium transmittance and the same
        // integral divided by sigma, so a thin tap costs the same expression as a thick one and neither
        // divides by a small number.
        float segmentT = exp(-sigma * dt);
        S += T * fogScatterAt(f, sun, sigma, phase) * (1.0 - segmentT);
        T *= segmentT;
    }

    if (t > resolved) {
        float far = t - resolved;
        float3 mid = ro + rd * (resolved + 0.5 * far);
        float sigma = fogDensityAt(f, mid);
        float segmentT = exp(-sigma * far);
        S += T * fogScatterAt(f, sun, sigma, phase) * (1.0 - segmentT);
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
 * doing it here keeps the budget honest: the prefix is added once per sample, where the record it belongs to is
 * read by every SPP resample and by both halves of a split. The unprojection below mirrors primary.rgen
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
    fogIntegrateSegment(medium, f, sun, eye, toHit / hitDistance, hitDistance, T, S);
}
