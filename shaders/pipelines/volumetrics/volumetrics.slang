// Volumetric fog: the geometry, the medium integrals and the sampling every volumetric pass shares.
//
// Two conventions run through all of it.
//
// SPACE. Everything is done in CAMERA-RELATIVE world units (blocks), the space `WorldPush.invViewProj`
// unprojects into. Rays handed to the TLAS add `WorldPush.camOffset` to reach the terrain's rebased
// space; nothing else does. Working camera-relative is not cosmetic: Minecraft coordinates reach the
// tens of millions, and a froxel centre differenced against a camera at that magnitude loses so many
// mantissa bits that the grid visibly swims as the player walks.
//
// OPTICAL DEPTH. The medium's colour is constant and only its DENSITY varies (exponentially with
// height), so the whole medium reduces to one scalar field. Distances along a ray are therefore carried
// as `tau`, the integral of that density in block-densities, and per-channel transmittance is recovered
// as exp(-extinction * tau). The payoff is that tau(t) is analytically invertible (see
// `distanceForOpticalDepth`), which is what lets the shaft march importance-sample the scattering
// distance exactly in one sample instead of ray-marching a fixed number of steps.
module volumetrics;

import world_common;
import sky;

public static const float VOLUME_PI = 3.14159265359;

// Froxel slices are distributed as pow(w, exponent) so texel density follows where the fog actually has
// contrast. 2.0 puts roughly half the slices in the nearest quarter of the range, which is where the
// view ray sweeps the most world space per pixel. Kept a constant rather than a setting: it changes the
// meaning of every stored slice, so it can only be varied together with a full history reset, and no
// value other than 2 has proven better in the reference implementations this follows.
public static const float VOLUME_DEPTH_EXPONENT = 2.0;

// Smallest density the medium is allowed to reach. Purely numerical: the optical-depth inverse divides
// by the density at the ray origin, and a camera far above the fog layer drives that to zero.
public static const float VOLUME_MIN_DENSITY = 1.0e-6;

// Ceiling on any radiance stored in a volumetric buffer, in cd/m^2.
//
// These buffers are rgba16f and hold radiance BEFORE the pre-exposure scale, so the numbers in them are
// absolute: at noon a shaft's in-scatter is tens of thousands, and the phase function's forward peak
// multiplies that by another order of magnitude as anisotropy approaches 1. Half-float saturates to
// infinity at 65504, and an infinity here does not stay local — see volumeSanitize. The ceiling is set
// far enough below the format's limit that no product of a clamped value can reach it either.
public static const float VOLUME_MAX_RADIANCE = 16384.0;

// Ceiling for the composite's write into the display image. That image is also rgba16f, but it carries
// PRE-EXPOSED values whose scale is chosen so the scene sits near the exposure key, so a fog-specific
// ceiling would clip the scene itself. This bounds only the format.
public static const float VOLUME_MAX_STORED = 60000.0;

// Ceiling on a stored optical depth. A downward ray through an exponential layer accumulates
// density faster than distance, so this is far above any transmittance that resolves to a visible
// value; it exists to keep the exponential's argument finite, not to shape the fog.
public static const float VOLUME_MAX_OPTICAL_DEPTH = 1.0e4;

// How far a froxel's sky-visibility ray looks before giving up and calling the sky unoccluded. Sized
// for Minecraft interiors: it has to clear a tall cave or a deep shaft, but every extra block is paid
// on every froxel. Past this the fog under a very large overhang reads slightly too bright.
public static const float VOLUME_SKY_OCCLUSION_DISTANCE = 96.0;

// Geometry buckets inside a terrain section's BLAS, matching RtAccel.BUCKET_*. Only these two let light
// through, and the visibility queries below skip them so a window or a pond does not cast a hard shadow
// into the fog.
public static const uint VOLUME_BUCKET_TRANSLUCENT = 2u;
public static const uint VOLUME_BUCKET_WATER = 3u;

// Instance cull mask. Matches trace.slang's CULL_SECONDARY: the same set the path tracer's shadow rays
// see, which excludes the first-person camera owner's own billboards.
public static const uint VOLUME_CULL_MASK = 0x01u;

public static const uint VOLUME_DEBUG_OFF = 0u;
public static const uint VOLUME_DEBUG_AMBIENT = 1u;
public static const uint VOLUME_DEBUG_SHAFT_RAW = 2u;
public static const uint VOLUME_DEBUG_SHAFT_FILTERED = 3u;
public static const uint VOLUME_DEBUG_TRANSMITTANCE = 4u;
public static const uint VOLUME_DEBUG_HISTORY = 5u;

// ---- Numerical guards ---------------------------------------------------------------------------
//
// Every value that enters or leaves a temporal buffer passes through one of these.
//
// A single infinity or NaN in a volumetric buffer is not a local defect the next frame overwrites. The
// temporal passes blend history forward unconditionally, so a poisoned texel survives for as long as
// its history lives; the reprojected fetch is filtered and the neighbourhood statistics in the shaft
// resolve read a 3x3 window, so each frame hands the value to every neighbour. The corruption therefore
// grows across the screen instead of decaying. It also escapes the system: the composite writes into
// the image auto-exposure meters, so one bad pixel drags the exposure of the entire frame.
//
// Infinities arise here for an ordinary reason rather than a pathological one. These buffers are
// rgba16f and hold absolute radiance, and a forward-scattering phase function aimed at the sun produces
// values within one order of magnitude of half-float's limit at default settings. Clamping is what
// keeps a legitimate bright sample from becoming a permanent one.

/** Clamps a radiance into `ceiling`, mapping every non-finite component to zero. */
public float3 volumeSanitizeTo(float3 v, float ceiling) {
    float3 result;
    [unroll]
    for (int c = 0; c < 3; c++) {
        float x = v[c];
        result[c] = isfinite(x) ? clamp(x, 0.0, ceiling) : 0.0;
    }
    return result;
}

/** Clamps a radiance destined for one of the volumetric buffers. */
public float3 volumeSanitize(float3 v) {
    return volumeSanitizeTo(v, VOLUME_MAX_RADIANCE);
}

/** Scalar form, for the distance and frame-count channels the colour buffers carry alongside rgb. */
public float volumeSanitizeScalar(float v, float ceiling) {
    return isfinite(v) ? clamp(v, 0.0, ceiling) : 0.0;
}

// ---- Hashing ------------------------------------------------------------------------------------
//
// Self-contained rather than reusing math.slang's pcg, which reaches the world pipeline's descriptor
// set through world_core and would drag every trace binding into these compute passes.

public uint volumeHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

public float volumeRandom(inout uint seed) {
    seed = volumeHash(seed);
    return float(seed) * (1.0 / 4294967296.0);
}

public float2 volumeRandom2(inout uint seed) {
    return float2(volumeRandom(seed), volumeRandom(seed));
}

/**
 * Per-pixel decorrelated seed. The frame index is mixed in through its own hash rather than added, so
 * consecutive frames do not produce neighbouring sequences (which shows up as a slow-crawling pattern
 * rather than as noise the temporal filter can average away).
 */
public uint volumeSeed(uint2 pixel, uint frameIndex, uint salt) {
    return volumeHash(pixel.x * 1973u + pixel.y * 9277u + salt * 26699u)
            ^ volumeHash(frameIndex * 2654435761u + salt);
}

// ---- Phase function -----------------------------------------------------------------------------

/** Henyey-Greenstein, normalised over the sphere so it integrates to 1. */
public float volumeHgPhase(float g, float cosTheta) {
    float gg = clamp(g, -0.95, 0.95);
    float denom = 1.0 + gg * gg - 2.0 * gg * cosTheta;
    return (1.0 - gg * gg) / (4.0 * VOLUME_PI * max(denom, 1.0e-4) * sqrt(max(denom, 1.0e-4)));
}

/** Orthonormal basis around `n` (Duff et al., branchless). */
public void volumeBasis(float3 n, out float3 t, out float3 b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

/**
 * Exact inverse-CDF sample of Henyey-Greenstein around `forward`: returns a direction whose density is
 * {@code volumeHgPhase(g, dot(result, forward))}. Because the sample is drawn from the phase function
 * itself, phase/pdf is identically 1 and the estimator that uses it never needs the density back — see
 * the froxel ambient term.
 *
 * <p>The signs match {@link volumeHgPhase}'s {@code 1 + g^2 - 2*g*cos} denominator, so g > 0 peaks
 * toward `forward`. Sampling routines written against the opposite convention (PBRT's, whose phase
 * denominator is {@code 1 + g^2 + 2*g*cos}) produce the mirrored lobe and quietly invert forward and
 * back scattering.
 */
public float3 volumeSampleHg(float g, float3 forward, float2 xi) {
    float gg = clamp(g, -0.95, 0.95);
    float cosTheta;
    if (abs(gg) < 1.0e-3) {
        cosTheta = 1.0 - 2.0 * xi.x;
    } else {
        float sqrTerm = (1.0 - gg * gg) / (1.0 - gg + 2.0 * gg * xi.x);
        cosTheta = (1.0 + gg * gg - sqrTerm * sqrTerm) / (2.0 * gg);
    }
    cosTheta = clamp(cosTheta, -1.0, 1.0);
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    float phi = 2.0 * VOLUME_PI * xi.y;
    float3 t;
    float3 b;
    volumeBasis(forward, t, b);
    return normalize(sinTheta * cos(phi) * t + sinTheta * sin(phi) * b + cosTheta * forward);
}

/**
 * Uniform direction inside the cone of half-angle `halfAngle` around `axis`.
 *
 * <p>This is the whole reason sun shafts come out looking right. Aiming the visibility ray at the
 * sun's centre gives a hard binary edge; sampling the disc gives a penumbra of angular width
 * {@code 2*halfAngle*(1 - occluderDistance/sampleDistance)} — zero where the shaft meets its occluder
 * and about 0.53 degrees far behind it. That is the real, distance-dependent softness of a sunbeam, and
 * it costs one extra sine and cosine.
 */
public float3 volumeSampleCone(float3 axis, float halfAngle, float2 xi) {
    float cosMax = cos(halfAngle);
    float cosTheta = 1.0 - xi.x * (1.0 - cosMax);
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    float phi = 2.0 * VOLUME_PI * xi.y;
    float3 t;
    float3 b;
    volumeBasis(axis, t, b);
    return normalize(sinTheta * cos(phi) * t + sinTheta * sin(phi) * b + cosTheta * axis);
}

// ---- Medium -------------------------------------------------------------------------------------

/** Dimensionless density multiplier at a camera-relative height. */
public float volumeDensity(VolumetricMedium m, float cameraRelativeY, float camY) {
    if (m.heightFalloff <= 0.0) {
        return 1.0;
    }
    return max(exp(-(cameraRelativeY + camY - m.heightBase) * m.heightFalloff), VOLUME_MIN_DENSITY);
}

/**
 * Optical depth from the ray origin out to `t`, in block-densities.
 *
 * <p>With density(s) = a*exp(-b*s) along the ray (a the density at the origin, b the height falloff
 * projected onto the ray direction) the integral is closed-form. The b -> 0 branch is not just a
 * division guard: a horizontal ray is the common case, and it has to reduce exactly to a*t or the fog
 * visibly steps as the view pitches through the horizon.
 */
public float volumeOpticalDepth(VolumetricMedium m, float originDensity, float dirY, float t) {
    float b = dirY * m.heightFalloff;
    if (abs(b * t) < 1.0e-4) {
        return originDensity * t;
    }
    return originDensity * (1.0 - exp(-b * t)) / b;
}

/** Largest optical depth reachable along the ray. Finite when the ray climbs out of the fog layer. */
public float volumeMaxOpticalDepth(VolumetricMedium m, float originDensity, float dirY) {
    float b = dirY * m.heightFalloff;
    return b > 0.0 ? originDensity / b : 1.0e30;
}

/** Inverse of {@link volumeOpticalDepth}. Undefined past `volumeMaxOpticalDepth`; callers clamp first. */
public float volumeDistanceForOpticalDepth(VolumetricMedium m, float originDensity, float dirY, float tau) {
    float b = dirY * m.heightFalloff;
    if (abs(b) < 1.0e-6 || originDensity <= 0.0) {
        return tau / max(originDensity, VOLUME_MIN_DENSITY);
    }
    float inner = 1.0 - tau * b / originDensity;
    if (inner <= 1.0e-6) {
        return 1.0e30;
    }
    return -log(inner) / b;
}

/** Per-channel transmittance for an optical depth. */
public float3 volumeTransmittance(VolumetricMedium m, float tau) {
    return exp(-m.extinction * tau);
}

/** Scattering coefficient per block at the reference height. */
public float3 volumeScattering(VolumetricMedium m) {
    return m.extinction * m.albedo;
}

/**
 * Scalar extinction the distance sampler is built around. Luminance-weighted so the channel carrying
 * most of the visible signal gets the tightest fit, which keeps the exp(-extinction*tau)/exp(-bar*tau)
 * ratio in the estimator close to 1 and the colour noise low.
 */
public float volumeMeanExtinction(VolumetricMedium m) {
    return max(dot(m.extinction, float3(0.2126, 0.7152, 0.0722)), 1.0e-7);
}

/**
 * Expected optical depth of a scattering event under the truncated-exponential distance sampler, i.e.
 * the centroid of where the in-scatter along this ray actually comes from.
 *
 * <p>Deterministic — it depends only on the medium and the segment length, never on the random sample.
 * That is the point: it is the anchor the shaft pass's temporal reprojection uses, and an anchor drawn
 * from the same 1-sample estimator it is meant to stabilise would be as noisy as the signal.
 */
public float volumeMeanOpticalDepth(float sigmaBar, float tauMax) {
    float x = sigmaBar * tauMax;
    if (x < 1.0e-3) {
        return tauMax * 0.5; // limit of the expression below; the exact form is all cancellation here
    }
    float u = exp(-x);
    return (1.0 - (x + 1.0) * u) / (sigmaBar * max(1.0 - u, 1.0e-8));
}

/**
 * Energy-conserving in-scatter across one depth slice (Hillaire, Frostbite 2015).
 *
 * <p>`scatter` is the in-scatter per unit length at the slice, `deltaTau` the slice's optical depth and
 * `length` its extent in blocks. Simply adding `scatter` and multiplying transmittance afterwards is
 * not energy conservative and shows up as fog that darkens as the slice count rises — which is exactly
 * the wrong direction, since more slices should converge, not change the answer.
 */
public float3 volumeSliceIntegral(VolumetricMedium m, float3 scatter, float deltaTau, float sliceLength) {
    float3 opticalDepth = m.extinction * deltaTau;
    float3 result;
    [unroll]
    for (int c = 0; c < 3; c++) {
        float od = opticalDepth[c];
        // The limit of (1 - exp(-od))/od as od -> 0 is 1; below this threshold the ratio is closer to 1
        // than fp32 can express and evaluating it directly is pure cancellation.
        float factor = od > 1.0e-5 ? (1.0 - exp(-od)) / od : 1.0;
        result[c] = scatter[c] * sliceLength * factor;
    }
    return result;
}

// ---- Froxel grid geometry -----------------------------------------------------------------------

/** Ray distance in blocks at normalised slice coordinate `w` in [0,1]. */
public float volumeSliceDistance(VolumetricMedium m, float w) {
    return pow(saturate(w), VOLUME_DEPTH_EXPONENT) * m.maxDistance;
}

/** Inverse of {@link volumeSliceDistance}; the coordinate the composite samples the volume at. */
public float volumeSliceCoord(VolumetricMedium m, float distance) {
    return pow(saturate(distance / max(m.maxDistance, 1.0e-4)), 1.0 / VOLUME_DEPTH_EXPONENT);
}

/**
 * Camera-relative ray through a screen UV, unjittered.
 *
 * <p>Unprojecting two points rather than assuming the eye sits at the origin is required, not tidy:
 * Minecraft bakes view bobbing into the projection matrix, so the effective eye position moves and only
 * the near-plane unprojection knows where it is.
 */
public void volumeCameraRay(float4x4 invViewProj, float2 uv, out float3 origin, out float3 dir) {
    float2 ndc = uv * 2.0 - 1.0;
    float4 nearH = mul(invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    origin = nearP;
    dir = normalize(farP - nearP);
}

/**
 * Distance along the primary ray to the surface a guide-buffer depth describes.
 *
 * <p>gDepth holds reversed-Z NDC depth (near = 1, far = 0), the value a rasteriser would write, because
 * that is what DLSS-RR wants. Recovering a distance therefore means unprojecting, not scaling. A depth
 * at or below zero is the sky: report the medium's own range so the fog stops where the atmosphere LUT
 * takes over instead of integrating to the far plane.
 */
public float volumeSurfaceDistance(float4x4 invViewProj, float2 uv, float depth, float fallback) {
    if (depth <= 0.0) {
        return fallback;
    }
    float2 ndc = uv * 2.0 - 1.0;
    float4 nearH = mul(invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 hitH = mul(invViewProj, float4(ndc.x, ndc.y, depth, 1.0));
    if (abs(hitH.w) < 1.0e-8) {
        return fallback;
    }
    return length(hitH.xyz / hitH.w - nearH.xyz / nearH.w);
}

/**
 * Screen UV a camera-relative point had in the previous frame, or a negative w flag if it had none.
 *
 * <p>`camDelta` converts current-camera-relative into previous-camera-relative, so this stays valid
 * across a terrain rebase (both are camera-relative, and the rebase only moves the world origin).
 * Returns w = 1 when the reprojection is usable.
 */
public float3 volumeReprojectUv(WorldPush push, float3 cameraRelative) {
    float4 clip = mul(push.prevViewProj, float4(cameraRelative + push.camDelta, 1.0));
    if (clip.w <= 1.0e-6) {
        return float3(0.0, 0.0, -1.0);
    }
    float2 uv = (clip.xy / clip.w) * 0.5 + 0.5;
    if (any(uv < float2(0.0, 0.0)) || any(uv > float2(1.0, 1.0))) {
        return float3(uv, -1.0);
    }
    return float3(uv, 1.0);
}

// ---- Sky lighting -------------------------------------------------------------------------------

/**
 * Sky radiance arriving at a froxel from `dir`, summed over both celestial bodies.
 *
 * <p>The sky-view LUT is baked at the camera's altitude and carries atmospheric in-scatter only — the
 * sun and moon discs are drawn separately by world.rmiss — so this cannot double-count the directional
 * term the shaft pass integrates. Using the camera's LUT for a froxel up to a couple of hundred blocks
 * away is an approximation, but at Caustica's 100-blocks-per-kilometre sky scale that is a two-kilometre
 * displacement against a hundred-kilometre shell.
 */
public float3 volumeSkyRadiance(Sampler2D skyViewLut, SkyState sky, float3 dir) {
    float viewZenithCos = dir.y;
    float3 total = float3(0.0, 0.0, 0.0);
    float2 sunUv = skyViewLutUv(sky.viewerRadiusKm, viewZenithCos,
            lightViewCosine(dir, sky.sunDir), SKY_VIEW_BODY_SUN);
    total += skyViewLut.SampleLevel(sunUv, 0.0).rgb;
    float2 moonUv = skyViewLutUv(sky.viewerRadiusKm, viewZenithCos,
            lightViewCosine(dir, sky.moonDir), SKY_VIEW_BODY_MOON);
    total += skyViewLut.SampleLevel(moonUv, 0.0).rgb;
    return max(total, float3(0.0, 0.0, 0.0));
}
