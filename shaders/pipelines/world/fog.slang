// Froxel volumetric fog: grid parameterisation, the density field, and the phase function.
//
// Shared verbatim by the injection raygen and the two volumetric compute passes, so the grid the rays
// are traced through and the grid the composite samples can never disagree. This module deliberately
// depends on world_common ONLY — no bindings, no globals, no rays — because its three consumers each
// declare a different descriptor set. Everything takes WorldPush as a parameter.
//
// Space: froxel positions are produced in the terrain's REBASED coordinates, the same space
// world.rgen traces in (camera-relative + WorldPush.camOffset). Absolute world coordinates are only
// needed for the height falloff and the noise anchor, and those add WorldPush.fogAnchor.
//
// Units: distances are BLOCKS, extinction is 1/blocks, and in-scatter is absolute ACEScg radiance per
// unit length. Nothing here is pre-exposed; the composite applies WorldPush.preExposure once.
module fog;

import world_common;

public static const float FOG_PI = 3.14159265359;
public static const float FOG_INV_4PI = 0.07957747155;

// Temporal-history bounds (see volumetric/integrate.comp.slang). The reference is the mean of
// 2*TAPS+1 of a column's slices in the current frame; the medium is smooth along Z, so that mean is a
// low-variance measurement of the region while a single slice is not.
//
// SIGMA is how many standard errors of that mean the history may drift before it is pulled back.
// MIN_ERROR floors the interval so a near-black region cannot make the test infinitely strict — that
// floor is what stops dim fog from having its history rejected every frame and never converging.
public static const int   FOG_STALENESS_TAPS        = 8;
public static const float FOG_STALENESS_SIGMA       = 6.0;
public static const float FOG_STALENESS_MIN_ERROR   = 0.02;

// Effective sample count retained when a history is clamped (drifted but probably still informative)
// and when a froxel is disoccluded and reseeded from its neighbours. Both keep partial credit rather
// than restarting at one sample, which would put a raw binary visibility estimate straight on screen.
public static const float FOG_STALENESS_RETAINED_FRAMES = 3.0;
public static const float FOG_DISOCCLUSION_FRAMES       = 2.0;

// Variance-adaptive blur along Z (see volumetric/integrate.comp.slang). Radius 2 while a froxel is
// still noisy, 1 while it settles, 0 once converged, keyed off the history length. Confined to the
// depth axis because the medium is smooth along a view ray while a shaft's structure lives in XY:
// blurring Z costs no shaft definition, whereas blurring XY smears the froxel lattice across
// silhouettes. Turning it off at convergence is what keeps the edge bit-identical to no filtering.
public static const float FOG_ZBLUR_NOISY_FRAMES     = 2.0;
public static const float FOG_ZBLUR_SETTLING_FRAMES  = 5.0;

// Toroidal shift applied to the blue-noise slice coordinate per unit of froxel screen Y, so adjacent
// columns read decorrelated regions of the mask. Coprime with the mask's 128 height so the shift cycles
// through all rows rather than a short repeating subset.
public static const int FOG_BLUE_NOISE_COLUMN_SHIFT = 37;

/**
 * Last slice of a column that lies entirely in front of a surface at `distance`.
 *
 * Every slice past this one integrates light through space the surface occludes, so any filter that
 * reads one drags exterior radiance into the visible column — sunlight painted on the inside of a wall.
 * That failure has recurred from three separate filters in this system, so the bound is defined once,
 * here, and both the integration and the composite are held to it.
 *
 * A slice is fully visible only if it STARTS before the surface: the slice containing the surface
 * straddles it and its sample position can sit beyond, so it does not qualify.
 */
public int fogVisibleSliceLimit(FogGrid g, float distance) {
    int slice = int(floor(fogDistanceSlice(g, distance)));
    slice = clamp(slice, 0, g.dims.z - 1);
    // Walk back while the slice's near edge is already past the surface.
    for (int i = 0; i < 3; ++i) {
        if (slice <= 0 || fogSliceDistance(g, float(slice)) <= distance) {
            break;
        }
        --slice;
    }
    return slice;
}

// Below this extinction a froxel contributes nothing a half can represent over a whole slice, so the
// injection pass skips its shadow ray entirely. With the default height falloff this culls most of the
// grid in clear weather, which is the single largest performance lever in the system.
public static const float FOG_DENSITY_EPSILON = 1.0e-5;

// Must match ACESCG_LUMA in world_core. Duplicated because world_core pulls in the world descriptor
// set, which the volumetric compute passes do not have.
public static const float3 FOG_ACESCG_LUMA = float3(0.2722287168, 0.6740817658, 0.0536895174);

public float fogLuminance(float3 c) {
    return dot(c, FOG_ACESCG_LUMA);
}

// ---- Grid parameterisation ----------------------------------------------------------------------
//
// Frustum-aligned, RADIAL slices: t is distance along the camera ray from the near-plane point, not a
// planar view-Z. Radial keeps t(s) and its exact inverse s(t) one-liners and needs no camera forward
// vector in the push block; the slight non-planarity at frustum corners is irrelevant for a medium
// this low-frequency.
//
// Slices are distributed by a power curve so near froxels are short (where shafts have structure) and
// far ones are long (where the medium is nearly homogeneous).

public struct FogGrid {
    public int3   dims;
    public float  near;
    public float  far;
    public float  depthExp;
    public int    samplesPerFroxel;
    public bool   enabled;
};

public FogGrid fogGrid(WorldPush push) {
    FogGrid g;
    g.dims = push.fogGridDims.xyz;
    g.near = push.fogParams0.x;
    g.far = push.fogParams0.y;
    g.depthExp = max(push.fogParams0.z, 1.0e-3);
    // w carries the per-froxel shadow sample count; zero disables the system outright, so the two
    // meanings cannot disagree the way a separate enable flag could.
    g.samplesPerFroxel = max(push.fogGridDims.w, 0);
    g.enabled = g.samplesPerFroxel > 0 && g.dims.x > 0 && g.dims.y > 0 && g.dims.z > 0;
    return g;
}

/** Distance along the ray at the (possibly fractional) slice index. */
public float fogSliceDistance(FogGrid g, float slice) {
    float n = clamp(slice / float(g.dims.z), 0.0, 1.0);
    return g.near + (g.far - g.near) * pow(n, g.depthExp);
}

/** Exact inverse of fogSliceDistance: the fractional slice index a distance lands on. */
public float fogDistanceSlice(FogGrid g, float distance) {
    float span = max(g.far - g.near, 1.0e-6);
    float n = clamp((distance - g.near) / span, 0.0, 1.0);
    return float(g.dims.z) * pow(n, 1.0 / g.depthExp);
}

/** Normalized W texture coordinate for a distance — what the composite feeds the trilinear fetch. */
public float fogDistanceW(FogGrid g, float distance) {
    return clamp(fogDistanceSlice(g, distance) / float(g.dims.z), 0.0, 1.0);
}

public struct FogRay {
    public float3 origin;   // rebased near-plane point for this froxel column
    public float3 dir;      // unit
};

/**
 * Camera ray through a froxel column, offset by the frame's grid jitter.
 *
 * `jitterUv` is in UV, matching how primary.rgen offsets the camera ray, so the froxel grid and the
 * camera sample the frustum through the same shifted lens each frame.
 *
 * Reconstructed from two unprojected points exactly like primary.rgen, because Minecraft bakes view
 * bobbing into the projection and the effective eye therefore comes from the matrix rather than from
 * the origin.
 */
public FogRay fogColumnRay(WorldPush push, FogGrid g, int2 columnIndex, float2 jitterUv) {
    float2 uv = (float2(columnIndex) + 0.5) / float2(g.dims.xy) + jitterUv;
    float2 ndc = uv * 2.0 - 1.0;
    float4 nearH = mul(push.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(push.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    FogRay r;
    r.origin = nearP + push.camOffset;
    r.dir = normalize(farP - nearP);
    return r;
}

// ---- Density field -------------------------------------------------------------------------------
//
// Minecraft authors no fog volumes, so the medium is synthesised: an exponential height falloff, a
// weather multiplier, and world-anchored animated value noise for shape. Without the noise a shaft
// reads as a clean geometric cone; a little low-frequency variance is what sells it as air.

public float fogHash13(float3 p) {
    p = frac(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return frac((p.x + p.y) * p.z);
}

/** Trilinearly-interpolated 3D value noise in [0,1], smoothstep-faded per axis. */
float fogValueNoise(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = fogHash13(i + float3(0.0, 0.0, 0.0));
    float n100 = fogHash13(i + float3(1.0, 0.0, 0.0));
    float n010 = fogHash13(i + float3(0.0, 1.0, 0.0));
    float n110 = fogHash13(i + float3(1.0, 1.0, 0.0));
    float n001 = fogHash13(i + float3(0.0, 0.0, 1.0));
    float n101 = fogHash13(i + float3(1.0, 0.0, 1.0));
    float n011 = fogHash13(i + float3(0.0, 1.0, 1.0));
    float n111 = fogHash13(i + float3(1.0, 1.0, 1.0));
    return lerp(lerp(lerp(n000, n100, f.x), lerp(n010, n110, f.x), f.y),
                lerp(lerp(n001, n101, f.x), lerp(n011, n111, f.x), f.y), f.z);
}

/**
 * Extinction (1/blocks) at a rebased world position.
 *
 * fogAnchor carries the terrain rebase origin reduced modulo 4096 (the same trick waterAnchor uses):
 * adding it reconstructs a world-pinned coordinate that stays small enough for fp32, so the noise
 * pattern is fixed in the world instead of swimming as the player moves and the rebase origin shifts.
 */
public float fogDensity(WorldPush push, float3 rebasedPos) {
    float baseDensity = push.fogParams0.w;
    if (baseDensity <= 0.0) {
        return 0.0;
    }
    float3 anchored = rebasedPos + push.fogAnchor.xyz;
    float heightFalloff = push.fogParams1.x;
    float baseY = push.fogParams1.y;
    float height = exp(-max(0.0, anchored.y - baseY) * heightFalloff);

    float noiseScale = push.fogParams2.x;
    float noiseStrength = push.fogParams2.y;
    float shape = 1.0;
    if (noiseStrength > 0.0 && noiseScale > 0.0) {
        float3 domain = anchored * noiseScale + push.fogAnchor.w;
        shape = 1.0 + noiseStrength * (fogValueNoise(domain) * 2.0 - 1.0);
    }
    float weather = push.fogParams2.w;
    return max(0.0, baseDensity * weather * height * shape);
}

// ---- Sampling offsets ----------------------------------------------------------------------------

/**
 * Sub-cell offset of the WHOLE froxel grid for this frame, in UV.
 *
 * This is the camera's own sub-pixel jitter, converted to UV, exactly as primary.rgen applies it
 * (`uv + worldPush.jitter / size`). Both the camera ray and the froxel column ray therefore receive the
 * same offset, which is what makes the scheme work at all.
 *
 * The previous approach offset each column by an independent hash. That is jitter without a resolve:
 * the injection sampled a position that moved every frame, while the composite always read the texel at
 * its NOMINAL centre, so the temporal filter averaged values describing different points and called it
 * convergence. It could never antialias the lattice either, because the lattice is defined by where the
 * volume is READ, and the read never moved.
 *
 * Driving the grid from the camera jitter fixes both. The fog signal now moves in lockstep with the
 * geometry signal DLSS-RR is already reconstructing, so RR resolves the two consistently instead of
 * fighting an offset it knows nothing about. It also leaves the pixel-to-froxel correspondence
 * jitter-invariant: column c sits at uv (c + 0.5)/dims + j and a pixel's ray passes through uv_pixel + j,
 * so the two match when (c + 0.5)/dims == uv_pixel — the same lookup the composite already performs,
 * with no jitter term needed at read time.
 */
public float2 fogGridJitterUv(WorldPush push) {
    return push.fogJitter.xy;
}

/** The previous frame's grid offset, needed to read the history in its own jittered frame. */
public float2 fogPrevGridJitterUv(WorldPush push) {
    return push.fogJitter.zw;
}

/**
 * Per-frame slice offset in slice units, in [-0.5, 0.5].
 *
 * Depth has no camera jitter to inherit — the camera's is purely lateral — so the slice axis keeps its
 * own low-discrepancy offset. It is derived from the frame index rather than passed in, which
 * guarantees the injection and the integration agree on it without a second push field to keep in sync.
 * A shared, grid-wide offset (not per-froxel) so the whole grid stays coherent, as above.
 */
public float fogSliceJitter(WorldPush push) {
    uint bits = push.frameIndex + 1u;
    float scale = 0.5;
    float result = 0.0;
    [unroll]
    for (int b = 0; b < 16; ++b) {
        if ((bits & 1u) != 0u) {
            result += scale;
        }
        bits >>= 1u;
        scale *= 0.5;
    }
    return result - 0.5;
}

/**
 * Sun-disc sample pair for a froxel, drawn from the spatiotemporal blue-noise mask.
 *
 * The mask is blue in both space and time (measured on the shipped asset: 22.6x spatial and 8.5x
 * temporal high-over-low power). That concentrates the sampling error in high frequencies, which is
 * only useful if a low-pass filter exists to remove it AND the mask is indexed along the axis that
 * filter operates on.
 *
 * Our only spatial filter runs along Z, inside a froxel column, so the mask's Y axis is mapped to the
 * SLICE rather than to the froxel's screen Y. Neighbouring slices then receive blue-noise decorrelated
 * samples and the Z blur is exactly the filter the mask is designed for. Indexing by screen (x, y)
 * instead — the obvious choice — gives every slice in a column the same mask value, so the Z blur never
 * crosses the axis the mask organises and the whole benefit disappears; measured, that indexing scored
 * 0.99x against a plain low-discrepancy sequence while this one scores 1.10x.
 *
 * The froxel's screen Y is folded in as a toroidal shift of the slice coordinate, which the mask
 * supports by construction, so columns stay decorrelated from one another.
 *
 * `sampleIndex` separates the multiple shadow rays a single froxel casts in one frame: they advance
 * along the mask's time axis, so they are blue-noise distributed with respect to each other as well as
 * across frames.
 */
public float2 fogBlueNoise(Sampler3D mask, int3 index, uint frameIndex, uint sampleIndex,
                           uint samplesPerFroxel) {
    uint3 dims;
    mask.GetDimensions(dims.x, dims.y, dims.z);
    float x = float(index.x % int(dims.x)) + 0.5;
    // Slice on Y, with the column's screen Y as a toroidal offset.
    int shifted = index.z + index.y * FOG_BLUE_NOISE_COLUMN_SHIFT;
    float y = float(((shifted % int(dims.y)) + int(dims.y)) % int(dims.y)) + 0.5;
    uint t = (frameIndex * max(samplesPerFroxel, 1u) + sampleIndex) % dims.z;
    float z = float(t) + 0.5;
    return mask.SampleLevel(float3(x / float(dims.x), y / float(dims.y), z / float(dims.z)), 0.0).rg;
}

// ---- Phase ---// ---- Phase ---------------------------------------------------------------------------------------

/** Henyey-Greenstein, 4pi-normalised. Local copy: math.slang pulls in the world descriptor set. */
public float fogHg(float cosTheta, float g) {
    float g2 = g * g;
    float denom = max(1.0 + g2 - 2.0 * g * cosTheta, 1.0e-4);
    return (1.0 - g2) * FOG_INV_4PI / (denom * sqrt(denom));
}

// Real atmospheric aerosols have a strong forward lobe plus a weak backscatter one. A single HG cannot
// be both, and the tight forward lobe is exactly what makes a shaft read as a shaft when looking toward
// the sun, so it is not something to compromise by lowering g.
public static const float FOG_BACK_LOBE_RATIO  = 0.5;
public static const float FOG_BACK_LOBE_WEIGHT = 0.15;

public float fogPhase(float cosTheta, float anisotropy) {
    float g = clamp(anisotropy, -0.95, 0.95);
    return lerp(fogHg(cosTheta, g), fogHg(cosTheta, -g * FOG_BACK_LOBE_RATIO), FOG_BACK_LOBE_WEIGHT);
}

// ---- Flag bits carried in WorldPush.fogFlags -----------------------------------------------------

public static const uint FOG_FLAG_HISTORY_VALID = 1u;

public bool fogHistoryValid(WorldPush push) {
    return (push.fogFlags & FOG_FLAG_HISTORY_VALID) != 0u;
}
