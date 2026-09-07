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

// Staleness test for the temporal history (see volumetric/integrate.comp.slang). The local reference is
// the mean of 2*TAPS+1 slices' current samples along Z; SIGMA is how many standard errors of that mean
// the history may differ by before it is treated as describing a froxel that no longer exists. Tuned by
// simulation: sigma 6 with 8 taps either side recovers from a visibility change about 4x faster than no
// test at all, with slightly LOWER steady-state noise, whereas tighter values over-trigger and bias the
// converged estimate.
public static const int   FOG_STALENESS_TAPS  = 8;
public static const float FOG_STALENESS_SIGMA = 6.0;

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
    public bool   enabled;
};

public FogGrid fogGrid(WorldPush push) {
    FogGrid g;
    g.dims = push.fogGridDims.xyz;
    g.near = push.fogParams0.x;
    g.far = push.fogParams0.y;
    g.depthExp = max(push.fogParams0.z, 1.0e-3);
    g.enabled = push.fogGridDims.w != 0 && g.dims.x > 0 && g.dims.y > 0 && g.dims.z > 0;
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
 * Spatiotemporal low-discrepancy sample pair for a froxel's sun-disc lookup.
 *
 * White noise (a raw PCG hash, which is what this used before) spreads its error evenly across all
 * frequencies, so the low-frequency part survives both the bilinear fetch and the temporal average and
 * is exactly what reads as clumpy speckle. This uses R2 — Roberts' generalised golden-ratio sequence —
 * offset per froxel, which keeps successive frames' samples maximally separated and pushes the residual
 * into high frequencies that the temporal average removes fastest. It is the cheap analytic stand-in
 * for the 256x256x128 blue-noise array MCRTX ships, needing no texture and no descriptor.
 */
public float2 fogSampleR2(int3 index, uint frameIndex) {
    // Plastic-constant reciprocals; the canonical R2 lattice generators.
    const float2 R2 = float2(0.7548776662466927, 0.5698402909980532);
    float offset = fogHash13(float3(index)) * 4096.0;
    return frac(R2 * (float(frameIndex) + offset));
}

// ---- Phase ---------------------------------------------------------------------------------------

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
