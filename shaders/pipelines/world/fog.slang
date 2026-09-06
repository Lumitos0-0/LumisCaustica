// Volumetric air fog: exponential height haze with physically based in-scatter, integrated into the
// path integral in world.rgen (indirect pass) and at the pass-A dielectric split. This is the same
// medium the atmosphere's sky-view LUT runs with — the same Rayleigh/Cornette-Shanks phase functions and
// the same transmittance LUT for light reaching a sample — but at block scale around the camera rather
// than planet scale.
//
// Two light terms, both physically required:
//
//   * Direct — sun/moon light scattered toward the viewer: sigmaS * phase(cos) * E_sample * T_terrain.
//     E_sample is the dominant body's viewer-space illuminance corrected by the transmittance-LUT ratio
//     between the sample's column and the viewer's. T_terrain comes from the per-frame light-space fog
//     volume (fog_grid.build.comp): a world-pinned transmittance cache built by marching light columns
//     through the terrain mesher's occupancy tiles. The fetch is the NEAREST cell on every cascade level
//     (no bilinear tap — shadow edges stay binary and cell-exact instead of leaking a penumbra across
//     geometry), with a crossfade across each ring boundary so the cascade switch never pops.
//   * Sky ambient — fog scatters skylight too; without a sky term it goes black inside sun-shadow no
//     matter how the direct ray is managed. Four sky lobes (90° from the dominant body, so the sun/moon
//     disc never lands on one) sample this frame's sky-view LUT once per march; each step weights them by
//     the phase + a per-cell sky-openness factor baked into a coarse 32^3 RGBA8 volume
//     (fog_grid/openness.comp.slang) so enclosed/indoor fog does not glow.
//
// Colour: the volume is monochrome (solid 0, cutout 0.5, translucent 0.9 grey). A per-frame tint probe
// (world/fog_probe.rgen.slang) fires ONE real terrain-only ray per FLAGGED cell of a 64^3 cache (1-block
// cells, ±32 blocks) through the real any-hit path — per-channel stained-glass tint + leaf texture-alpha
// dapple — and the march replaces its cache value with the probed colour for samples inside the window.
// The probe is per CELL (amortized over every pixel that sees it), not per march.
//
// Entities are the one dynamic occluder the cache excludes: one entity-only RT ray per march from the
// segment midpoint (stable — never derived from a tint/edge sample), so the march's RT cost is at most
// ONE ray regardless of how much tinted geometry is visible.
//
// The march integrates each constant-density step analytically (Hillaire's
// (L - L*stepT)/sigmaT form, identical to the sky module), so the count needed to avoid banding is
// small. Samples are spaced quadratically: the first steps are sub-block because that is where canopy
// beam edges live, and later steps pull apart where the distance fade and the visibility jitter make
// the estimate smooth anyway.
//
// Air only. Water and the other volume dielectrics carry their own absorption through MediumStack, and
// the two never overlap: this module is invoked only when the current medium's extinction is zero.
//
// Everything is deterministic per (path, step): the march offset and light jitter come from the path
// seed, and the density field is world-anchored and never animated, so temporal reconstruction sees a
// static medium rather than a flickering one. The density is also evaluated at a sample's own altitude
// relative to the viewer's pushed altitude (100:1 block-to-km scale), so no sea-level or world-origin
// value needs to reach the shader.

import world_common;
import world_core;
import sky;
import trace;
import math;
import bindings;

public static const int   FOG_STEPS = 12;
public static const float FOG_MIN_DENSITY = 1.0e-4; // per block; below this a sample fires no entity ray
// Tint probe cache (world/fog_probe.rgen.slang): 64^3 RGBA8 cells of 1 block, ±32 blocks around the
// light-space anchor. 0 = unprobed cell (use the cache's grey value); alpha 255 = probed.
public static const int   FOG_PROBE_CELLS = 64;
public static const int   FOG_PROBE_HALF  = 32;
// Sky openness volume (fog_grid/openness.comp.slang): 32^3 RGBA8 cells of 8 blocks, ±128 blocks. Each
// channel is one sky lobe; out of range = 1.0 (assumed open).
public static const int   FOG_OPENNESS_CELLS = 32;
public static const int   FOG_OPENNESS_HALF  = 16;
public static const int   FOG_OPENNESS_SIZE  = 8;
// Four lobes approximate the hemisphere integral: Σ ΔΩ·phase_i ≈ (π/2)·Σ phase_i (each lobe covers a
// quarter of the visible hemisphere), so a uniform sky of radiance U lights fog as U/2 — exactly the
// isotropic hemisphere in-scatter, per unit illuminance of the sky.
public static const float FOG_AMBIENT_WEIGHT = SKY_PI * 0.5;
// Light-space fog volume (see fog_grid.build.comp): 4 levels, 128^3 cells each, cellSize 2^k blocks and
// extent 64*2^k blocks around the camera. Kept in lock-step with that pass.
public static const int   FOG_VOLUME_LEVELS = 4;
public static const int   FOG_VOLUME_CELLS = 128;
// Bit 24 of the packed transmittance uint (bits 0..23 are RGB8), kept in lock-step with
// fog_grid/build.comp.slang: set by the bake for cells whose column crosses tinted cutout/translucent
// materials, signalling the march to look up the per-cell tint probe cache (fog_probe.rgen) for the real
// per-channel transmittance instead of the grey approximation.
public static const uint  FOG_PACK_EXACT_COLOR = 1u << 24u;

public struct FogState {
    public bool   enabled;         // bit 1 of flags — whole effect
    public float  density;         // per-block scattering at the fog base
    public float  mieFraction;     // 0 = Rayleigh phase, 1 = Cornette-Shanks
    public float  mieG;            // forward-scatter anisotropy of the Mie part
    public float  scaleHeight;     // blocks per e-fold of density above the base
    public float  farEnd;          // blocks; density fades to zero over the last third
    public float  scatterFraction; // sigmaS / sigmaT
    public float  intensity;       // global density multiplier
    public float  warmth;          // 0 = neutral beams, 1 = golden (weighted by low sun)
};

public FogState fogState(WorldPush push) {
    FogState f;
    f.enabled = (push.flags & 2u) != 0u;
    f.density = max(push.fogLook0.x, 0.0);
    f.mieFraction = clamp(push.fogLook0.y, 0.0, 1.0);
    f.scaleHeight = max(push.fogLook0.z, 1.0);
    // The light-space fog volume covers 512 blocks around the camera; beyond that the march has no
    // terrain-transmittance data, so the authored fade is capped to the cache's extent.
    f.farEnd = clamp(push.fogLook0.w, 1.0, 512.0);
    f.scatterFraction = clamp(push.fogLook1.x, 1.0e-3, 1.0);
    f.mieG = clamp(push.fogLook1.y, 0.0, 0.95);
    f.intensity = max(push.fogLook1.z, 0.0);
    f.warmth = clamp(push.fogLook1.w, 0.0, 1.0);
    return f;
}

public bool fogIsEnabled(WorldPush push) {
    return (push.flags & 2u) != 0u;
}

/** Sample altitude above sea level in BLOCKS, from the pushed viewer altitude (km) and the rebased delta. */
public float fogAltitudeBlocks(float3 p) {
    return (p.y - worldPush.camOffset.y) + 100.0f * worldPush.skyLook2.z;
}

/** Same altitude in km, in the sky module's planet-radius convention. */
public float fogAltitudeKm(float3 p) {
    return worldPush.skyLook2.z + (p.y - worldPush.camOffset.y) * 0.01f;
}

public float fogDensity(FogState fog, float3 p) {
    // Exponential above the base; below it (caves, ocean floors) the density saturates rather than
    // growing without bound, which reads as still air in enclosed space.
    float altitude = fogAltitudeBlocks(p);
    return fog.intensity * fog.density * exp(-max(altitude, 0.0) / fog.scaleHeight);
}

// Cornette-Shanks with an authored g (sky.miePhase is fixed to the atmosphere's 0.8; the fog needs its
// own so the beam tightness is a look-package choice). Same normalisation as the sky's copy.
public float fogMiePhase(float cosTheta, float g) {
    float g2 = g * g;
    float k = 3.0 / (8.0 * SKY_PI) * (1.0 - g2) / (2.0 + g2);
    return k * (1.0 + cosTheta * cosTheta) / pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1.0e-4), 1.5);
}

public struct FogIntegral {
    public float3 inscatter;     // ACEScg radiance added to the path (the accumulated L)
    public float3 transmittance; // Beer-Lambert extinction the segment's later contributions inherit
};

float3 decodeFogPacked(uint packed) {
    return float3(float(packed & 0xFFu), float((packed >> 8u) & 0xFFu),
            float((packed >> 16u) & 0xFFu)) * (1.0 / 255.0);
}

// Terrain transmittance from p along lightDir, from the per-frame light-space fog volume. The volume is
// a chain of four cubes: the level containing p covers to its own +z face, then each coarser level takes
// over at the shared boundary — the chain telescopes to T(p -> light) with at most five fetches.
// lightDir MUST be the same dominant, UNJITTERED direction the bake used (light.dir, not the per-sample
// jittered sampleSquare result): the volume is a single fixed light-space frame, and rotating it per
// sample would remap every fetch to different cells. right/up are the same frame, computed once per
// march segment rather than re-derived per step.
//
// Every level is fetched at the NEAREST cell — no bilinear tap. A tap averages an opaque and a clear cell
// across a blend zone (1 block near, up to 8 blocks at the far cascade), which is what leaked light past
// solid geometry and made distant edges soft/displaced. Nearest keeps shadow edges binary and cell-exact
// at every distance; the deterministic world-pinned lattice makes the resulting jaggies static, so
// DLSS-RR temporally smooths them instead of smearing a stochastic edge. Ring boundaries (64/128/256
// blocks) are crossfaded per level so the cascade switch never pops. The bake stores a CONSERVATIVE
// per-channel min over sub-columns (see build.comp.slang), so the coarse levels are never brighter than
// the fine levels at a shared boundary — the crossfade therefore blends agreement, not a leak. `tinted`
// reports whether the fetched column was baked with a tinted cutout/translucent material (stained glass,
// leaves, water): for those samples the march reads the per-cell tint probe cache instead of the grey
// approximation.
public float3 fogTerrainTransmittance(float3 p, float3 lightDir, float3 right, float3 up,
                                      float3 anchor, out bool tinted) {
    // rel is against the same world-pinned lattice anchor the bake used (sky.fogVolumeAnchor), so a world
    // point maps to the same cell regardless of camera position.
    float3 rel = p - anchor;
    float uw = dot(rel, right);
    float vw = dot(rel, up);
    float zw = dot(rel, lightDir);
    float axisMax = max(abs(uw), max(abs(vw), abs(zw)));

    int first = 0;
    for (int k = 0; k < FOG_VOLUME_LEVELS - 1; k++) {
        float extent = 64.0 * exp2(float(k));
        if (axisMax > extent) {
            first = k + 1;
        }
    }

    bool tintedCoarse;
    float3 transmittance = fogChain(uw, vw, zw, first, tintedCoarse);
    tinted = tintedCoarse;

    // Crossfade with the finer level across each ring boundary (64, 128, 256 blocks): the two chains
    // differ only by resolution, and blending over a +/-1-cell window removes the pop when `first`
    // switches levels as the camera moves.
    if (first > 0) {
        float ring = 64.0 * exp2(float(first - 1));
        float w = exp2(float(first - 1));
        if (axisMax > ring - w) {
            bool tintedFine;
            float3 fine = fogChain(uw, vw, zw, first - 1, tintedFine);
            float t = clamp((axisMax - (ring - w)) / (2.0 * w), 0.0, 1.0);
            transmittance = lerp(fine, transmittance, t);
            tinted = tinted || tintedFine;
        }
    }
    return transmittance;
}

// The telescoping chain for one starting level: level `first` at the sample's own z cell, then each
// coarser level at the shared boundary cell (z = 32.0 cells), all with the same narrow 2x2 lateral tap.
float3 fogChain(float uw, float vw, float zw, int first, out bool tinted) {
    tinted = false;
    float3 transmittance = float3(1.0, 1.0, 1.0);
    for (int k = first; k < FOG_VOLUME_LEVELS; k++) {
        float size = exp2(float(k));
        // Coordinates are CELLS per axis (the volume stores 128 cells of size 2^k per level), so the
        // fetch index is simply cellCoordinate + 64. For k == first this is the sample itself; for the
        // coarser levels it is the boundary the previous level ends at (64 * 2^(k-1) blocks = 32 cells
        // in level k), so the fetched cells tile the ray exactly once.
        float u = uw / size;
        float v = vw / size;
        float z = (k == first) ? (zw / size) : 32.0;
        int cz = clamp(int(floor(z + 64.0)), 0, FOG_VOLUME_CELLS - 1);
        uint levelBase = uint(k) * uint(FOG_VOLUME_CELLS * FOG_VOLUME_CELLS * FOG_VOLUME_CELLS);
        uint zBase = uint(cz) * uint(FOG_VOLUME_CELLS * FOG_VOLUME_CELLS);

        int cu = clamp(int(floor(u + 64.0)), 0, FOG_VOLUME_CELLS - 1);
        int cv = clamp(int(floor(v + 64.0)), 0, FOG_VOLUME_CELLS - 1);
        uint packed = ConstPtr<uint32_t>(pc.fogVolumeAddr)[levelBase + zBase
                + uint(cv) * uint(FOG_VOLUME_CELLS) + uint(cu)];
        tinted = tinted || (packed & FOG_PACK_EXACT_COLOR) != 0u;
        transmittance *= decodeFogPacked(packed);
    }
    return transmittance;
}

public FogIntegral fogNoScatter() {
    FogIntegral result;
    result.inscatter = float3(0.0, 0.0, 0.0);
    result.transmittance = float3(1.0, 1.0, 1.0);
    return result;
}

// Sky-ambient lobe directions. Lock-step with fog_grid/openness.comp.slang's fogSkyLobeDirs() — the
// openness volume stores per-channel openness for EXACTLY these directions, so a change here requires the
// same change there (and vice versa).
void fogSkyLobeDirs(float3 lightRight, float3 lightUp, out float3 d0, out float3 d1,
                    out float3 d2, out float3 d3) {
    float3 upW = float3(0.0, 1.0, 0.0);
    d0 = normalize(lightRight + upW * 0.5);
    d1 = normalize(-lightRight + upW * 0.5);
    d2 = normalize(lightUp + upW * 0.5);
    d3 = normalize(-lightUp + upW * 0.5);
}

// Sky radiance from the frame's sky-view LUT (the same source world.rmiss uses, so fog ambient and
// surface GI agree). The lobe set is ~90° from the dominant body, so the sun/moon DISC never lands on
// a lobe — the ambient is the sky gradient, not a false extra disc.
float3 fogSkyLobeRadiance(Sampler2D lut, SkyState sky, float3 dir) {
    float2 sunUv = skyViewLutUv(sky.viewerRadiusKm, dir.y, lightViewCosine(dir, sky.sunDir),
            SKY_VIEW_BODY_SUN);
    float2 moonUv = skyViewLutUv(sky.viewerRadiusKm, dir.y, lightViewCosine(dir, sky.moonDir),
            SKY_VIEW_BODY_MOON);
    return lut.SampleLevel(sunUv, 0.0).rgb + lut.SampleLevel(moonUv, 0.0).rgb;
}

float fogLobePhase(float3 viewDir, float3 lobeDir, FogState fog) {
    float cosTheta = dot(viewDir, lobeDir);
    return lerp(rayleighPhase(cosTheta), fogMiePhase(cosTheta, fog.mieG), fog.mieFraction);
}

// Tint probe cache lookup at light-space coordinates (blocks). Returns the raw word: 0 = unprobed or out
// of window (the caller keeps the cache's grey value), anything else is a probed cell whose RGBA8 words
// decode to the REAL terrain transmittance at the cell centre — per-channel tint + texture-alpha dapple,
// measured once per flagged cell by fog_probe.rgen. The alpha byte distinguishes a probed BLACK glass
// cell (RGB 0, alpha 255) from the all-zero sentinel.
uint fogProbeCell(float uw, float vw, float zw) {
    if (pc.fogProbeAddr == 0u
            || abs(uw) >= FOG_PROBE_HALF || abs(vw) >= FOG_PROBE_HALF || abs(zw) >= FOG_PROBE_HALF) {
        return 0u;
    }
    int u = clamp(int(floor(uw + FOG_PROBE_HALF)), 0, FOG_PROBE_CELLS - 1);
    int v = clamp(int(floor(vw + FOG_PROBE_HALF)), 0, FOG_PROBE_CELLS - 1);
    int z = clamp(int(floor(zw + FOG_PROBE_HALF)), 0, FOG_PROBE_CELLS - 1);
    return ConstPtr<uint32_t>(pc.fogProbeAddr)[uint((z * FOG_PROBE_CELLS + v) * FOG_PROBE_CELLS + u)];
}

// Sky openness for the four lobes at this cell (8-block cells, ±128 blocks). Out of range CLAMPS to the
// nearest boundary cell instead of assuming fully open: the bake already walks 512 blocks outward from a
// boundary cell, so the boundary value is the best far-field occlusion estimate, and "open" past the grid
// was exactly how distant fog glowed through terrain just outside the cache window.
float4 fogOpenness(float uw, float vw, float zw) {
    float4 open = float4(1.0, 1.0, 1.0, 1.0);
    if (pc.fogOpennessAddr == 0u) {
        return open;
    }
    float halfBlocks = float(FOG_OPENNESS_HALF * FOG_OPENNESS_SIZE);
    // floor before clamp: adding halfBlocks first keeps the division non-negative, and clamp handles the
    // truly out-of-range side, so the same expression serves both in-range and boundary cases.
    int u = clamp(int(floor((uw + halfBlocks) / float(FOG_OPENNESS_SIZE))), 0, FOG_OPENNESS_CELLS - 1);
    int v = clamp(int(floor((vw + halfBlocks) / float(FOG_OPENNESS_SIZE))), 0, FOG_OPENNESS_CELLS - 1);
    int z = clamp(int(floor((zw + halfBlocks) / float(FOG_OPENNESS_SIZE))), 0, FOG_OPENNESS_CELLS - 1);
    uint packed = ConstPtr<uint32_t>(pc.fogOpennessAddr)[uint((z * FOG_OPENNESS_CELLS + v)
            * FOG_OPENNESS_CELLS + u)];
    return float4(float(packed & 0xFFu), float((packed >> 8u) & 0xFFu),
            float((packed >> 16u) & 0xFFu), float((packed >> 24u) & 0xFFu)) * (1.0 / 255.0);
}

/**
 * March the fog along one segment and return its view-integrated in-scatter plus transmittance.
 *
 * {@code ro}/{@code rd} are the segment as the path tracer holds it; {@code light} is the dominant
 * celestial light in ACEScg illuminance and {@code sky} the per-frame sky state (its viewer radius and
 * body directions are what the atmosphere ratio needs). {@code skyViewLut} is the same sky-view LUT the
 * miss shader samples, used for the sky-ambient lobes. {@code tMax} is the geometry distance (miss case
 * uses 10000; the march caps itself at {@code fog.farEnd}).
 */
public FogIntegral integrateFog(FogState fog, float3 ro, float3 rd, float tMax, uint seed,
                                CelestialLight light, SkyState sky, Sampler2D skyViewLut) {
    FogIntegral result = fogNoScatter();
    if (!fog.enabled || fog.intensity <= 0.0 || tMax <= 0.0) {
        return result;
    }
    float tEnd = min(tMax, fog.farEnd);
    if (tEnd <= 0.0) {
        return result;
    }
    // Whole-segment cull: five scale heights above the base the density is already below the march
    // threshold, so a bounce high in the air (or a flying player) never pays the march or its rays.
    float fogTop = fog.scaleHeight * 5.0;
    if (fogAltitudeBlocks(ro) > fogTop && fogAltitudeBlocks(ro + rd * tEnd) > fogTop) {
        return result;
    }

    // Warmth weights the DIRECT term toward a sunrise gold as the sun drops toward the horizon; at noon
    // it is neutral (the atmosphere LUT's own colour dominates) and moonlight stays cool — the gold is
    // a solar look, not a night one.
    bool solar = dot(light.dir, sky.sunDir) > dot(light.dir, sky.moonDir);
    float warm = solar ? fog.warmth * (1.0 - smoothstep(0.0, 0.35, light.dir.y)) : 0.0;
    float3 beamTint = lerp(float3(1.0, 1.0, 1.0), float3(1.0, 0.86, 0.62), warm);

    float3 T = float3(1.0, 1.0, 1.0);
    uint marchSeed = seed;
    marchSeed = pcg(marchSeed);
    float jitter = rndf(marchSeed);
    // Fetch frame + anchor of the baked volumes: fixed for the whole segment (the bake used the dominant
    // direction and the same world-pinned lattice), so derive them once rather than once per fog step.
    float3 lightRight;
    float3 lightUp;
    celestialSquareFrame(light.dir, lightRight, lightUp);
    float3 fogAnchor = fogVolumeAnchor(worldPush.camOffset, light.dir);

    // Sky-ambient lobes, once per march (they are view/sample independent): four directions ~90° from the
    // dominant body so the sun/moon disc never lands on a lobe. Radiance from the SAME sky-view LUT the
    // miss shader uses, so the fog's shadow-area colour and the surface GI agree.
    float3 lobe0;
    float3 lobe1;
    float3 lobe2;
    float3 lobe3;
    fogSkyLobeDirs(lightRight, lightUp, lobe0, lobe1, lobe2, lobe3);
    float3 sky0 = fogSkyLobeRadiance(skyViewLut, sky, lobe0);
    float3 sky1 = fogSkyLobeRadiance(skyViewLut, sky, lobe1);
    float3 sky2 = fogSkyLobeRadiance(skyViewLut, sky, lobe2);
    float3 sky3 = fogSkyLobeRadiance(skyViewLut, sky, lobe3);

    // Entity occlusion: ONE entity-only RT ray per march, from the midpoint, stable by construction
    // (never derived from a tint/edge sample — that was the remaining "fog disappears" path). Lazy, so a
    // march fully culled by the height/density checks pays nothing.
    bool entityRayReady = false;
    float3 entityTransmittance = float3(1.0, 1.0, 1.0);

    for (int i = 0; i < FOG_STEPS; i++) {
        float u0 = (float(i) + jitter) / float(FOG_STEPS);
        float u1 = (float(i + 1) + jitter) / float(FOG_STEPS);
        float t0 = tEnd * u0 * u0;
        float t1 = min(tEnd * u1 * u1, tEnd);
        float dt = t1 - t0;
        if (dt <= 0.0) {
            continue;
        }
        float tMid = 0.5 * (t0 + t1);
        float3 p = ro + rd * tMid;
        // Distance fade: dissolve over the last third instead of cutting off at a hard fog wall.
        float fade = 1.0 - smoothstep(fog.farEnd * 0.7, fog.farEnd, tMid);
        float sigma = fogDensity(fog, p) * fade;
        if (sigma <= FOG_MIN_DENSITY) {
            continue; // no density means no extinction and no visibility ray
        }
        // Extinction is neutral; only the light that is SCATTERED carries the beam tint, so the
        // transmittance of the fog itself does not shift hue with the warmth setting.
        float sigmaScalar = sigma / fog.scatterFraction;
        float3 sigmaT = float3(sigmaScalar, sigmaScalar, sigmaScalar);
        float3 sigmaS = sigma * beamTint;
        float3 stepT = exp(-sigmaT * dt);

        // Direct: the terrain transmittance comes from the per-frame light-space volume (a handful of
        // constant-time fetches — the entire ray cost is gone). The volume is built in the frame of the
        // DOMINANT, unjittered body direction (see fog_grid.build.comp: celestialSquareFrame(light.dir)),
        // so the fetch MUST use the same axis or the per-sample direction jitter — which rotates the
        // frame by up to the body's half-angle — makes the same world point land in different cells per
        // sample and step, flashing through gaps. The fetch is crisp-nearest, so edges stay binary and
        // cell-exact; the jittered direction below is only for the entity ray (same sample direction the
        // surface NEE and the old per-ray visibility use).
        uint stepSeed = seed ^ (uint(i) * 0x9e3779b9u + 0x85ebca6bu);
        stepSeed = pcg(stepSeed);
        float3 lightDir = light.halfAngle > 0.0
                ? sampleSquare(light.dir, light.halfAngle, stepSeed) : light.dir;
        bool tinted;
        float3 terrainT = fogTerrainTransmittance(p, light.dir, lightRight, lightUp, fogAnchor, tinted);
        // Light-space cell of this sample in the volumes' shared world-pinned lattice.
        float3 rel = p - fogAnchor;
        float uw = dot(rel, lightRight);
        float vw = dot(rel, lightUp);
        float zw = dot(rel, light.dir);
        // Near-field per-cell tint: the probe cache holds the REAL terrain transmittance (per-channel
        // glass tint + leaf texture dapple) measured once per flagged cell by fog_probe.rgen — exact,
        // per-sample-ish colour with NO per-march ray. Beyond the probe window (or unprobed cells) the
        // cache's grey value is used, which is correct for tint-free columns anyway.
        if (tinted) {
            uint probe = fogProbeCell(uw, vw, zw);
            if (probe != 0u) {
                terrainT = decodeFogPacked(probe);
            }
        }
        if ((worldPush.flags & 8u) != 0u) {
            if (!entityRayReady) {
                // One entity-occlusion ray per segment, from the midpoint. Terrain occlusion already comes
                // from the cache (an entity-only ray never sees terrain, and re-testing it would apply it
                // twice); bounded to the fog volume's extent because the march never scatters beyond
                // farEnd (≤ 512). The direction is a single jittered sample of the body, shared by all
                // steps: same dynamic-occluder answer as the old per-step rays at far lower cost.
                uint segmentSeed = seed ^ 0x2c1b3c6du;
                segmentSeed = pcg(segmentSeed);
                float3 segmentLightDir = light.halfAngle > 0.0
                        ? sampleSquare(light.dir, light.halfAngle, segmentSeed) : light.dir;
                float3 mid = ro + rd * (tEnd * 0.5);
                entityTransmittance = visibilityEntities(mid, segmentLightDir, 512.0).transmittance;
                entityRayReady = true;
            }
            terrainT *= entityTransmittance;
        }
        // Sky ambient: fog scatters skylight, so sun-shadow never turns it black. Per-cell openness from
        // the baked coarse volume keeps it out of enclosed/indoor fog; lobes keep it directional.
        float4 open = fogOpenness(uw, vw, zw);
        float3 ambient = sigmaS * FOG_AMBIENT_WEIGHT * (
                sky0 * fogLobePhase(rd, lobe0, fog) * open.x
                + sky1 * fogLobePhase(rd, lobe1, fog) * open.y
                + sky2 * fogLobePhase(rd, lobe2, fog) * open.z
                + sky3 * fogLobePhase(rd, lobe3, fog) * open.w);
        float cosTheta = dot(rd, light.dir);
        float phase = lerp(rayleighPhase(cosTheta), fogMiePhase(cosTheta, fog.mieG), fog.mieFraction);
        float radiusKm = ATMOS_BOTTOM_KM + max(fogAltitudeKm(p), MIN_VIEWER_ALTITUDE_KM);
        // E at the sample = E at the viewer (already atmospheric, includes the space->viewer column)
        // times the CHANGE between the two columns. The sky transmittance LUT gives both, so the ratio
        // is exact and never multiplies the atmosphere twice — which would square the sunset extinction
        // and turn the most dramatic beam times black. Near the horizon both terms collapse together, so
        // below a luminance threshold the correction is disabled rather than dividing by zero.
        float3 viewerT = transmittanceToSpace(transmittanceLut, sky.viewerRadiusKm, lightDir, light.halfAngle);
        float3 sampleT = transmittanceToSpace(transmittanceLut, radiusKm, lightDir, light.halfAngle);
        float viewerLum = dot(viewerT, float3(0.2126, 0.7152, 0.0722));
        float3 sampleRelT = viewerLum > 1.0e-4 ? sampleT / max(viewerT, float3(1.0e-4)) : float3(1.0);
        float3 direct = sigmaS * (phase * (light.illuminance * bt709ToAcesCg(sampleRelT) * terrainT));

        // Analytic constant-medium segment integral, same form the sky's raymarch uses.
        float3 scattered = direct + ambient;
        float3 safeExtinction = max(sigmaT, float3(1.0e-6));
        result.inscatter += T * (scattered - scattered * stepT) / safeExtinction;
        T *= stepT;
    }
    result.transmittance = T;
    return result;
}
