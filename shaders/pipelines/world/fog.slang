// Volumetric air fog: exponential height haze with physically based in-scatter, integrated into the
// path integral in world.rgen (indirect pass) and at the pass-A dielectric split. This is the same
// medium the atmosphere's sky-view LUT runs with — the same Rayleigh/Cornette-Shanks phase functions and
// the same transmittance LUT for light reaching a sample — but at block scale around the camera rather
// than planet scale.
//
// The single term is the direct beam:
//
//   * Direct — sun/moon light scattered toward the viewer: sigmaS * phase(cos) * E_sample * T_terrain.
//     E_sample is the dominant body's viewer-space illuminance corrected by the transmittance-LUT ratio
//     between the sample's column and the viewer's (the ratio, not the raw LUT — the atmosphere must
//     never be applied twice). T_terrain comes from the per-frame light-space fog volume (see
//     fog_grid.build.comp): a camera-centered transmittance cache built by marching light columns through
//     the terrain mesher's occupancy tiles, so a march sample costs a handful of constant-time fetches
//     instead of a TLAS visibility ray. The volume is rebuilt from the real section geometry every frame
//     and reads at the same block-subcell resolution, so shafts keep their crisp canopy-sliced edges with
//     real penumbra — but without the ray cost. The cache excludes dynamic entities and is monochrome by
//     construction, so the march keeps AT MOST ONE RT visibility ray per segment: an entity-only occluder
//     ray when any entity exists (flags bit 3), or one exact-colour probe at a tinted cutout/translucent
//     column — used only for its own sample so per-sample shadowing (the cache) is never overwritten.
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
// A tinted column whose cache transmittance is below this is shadowed/occluded: the exact probe would
// only measure the occluder (and in shadowed areas must never fire — that was the recent "fog disappears"
// regression). Below this the cache's dark value is already the right answer.
public static const float FOG_PROBE_MIN_TRANSMITTANCE = 0.05;
// Light-space fog volume (see fog_grid.build.comp): 4 levels, 128^3 cells each, cellSize 2^k blocks and
// extent 64*2^k blocks around the camera. Kept in lock-step with that pass.
public static const int   FOG_VOLUME_LEVELS = 4;
public static const int   FOG_VOLUME_CELLS = 128;
// Bit 24 of the packed transmittance uint (bits 0..23 are RGB8), kept in lock-step with
// fog_grid/build.comp.slang: set by the bake for cells whose column crosses tinted cutout/translucent
// materials, signalling the march to replace the grey approximation with one exact terrain ray.
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
// Every level uses the same deterministic 2x2 tap, but with a HALF-cell blend zone (a quarter cell each
// side of the boundary) instead of the full cell. The old full-cell tap averaged an opaque and a clear
// cell across the whole cell width, letting light "leak" half a cell past solid geometry; the old coarse
// levels were nearest-fetched, so edges snapped and jumped as the camera moved through the 64/128/256-block
// ring boundaries. The narrow tap keeps a designed penumbra that is spatially stable. `tinted` reports
// whether the tapped cells belong to a column the bake flagged as crossing tinted cutout/translucent
// materials (stained glass, leaves, water): the march fires one exact-colour probe for those columns,
// because the cache's monochrome approximation cannot carry per-channel tint or texture-alpha dapple.
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

        // 2x2 tap in cell-centre coordinates (cell centre of index c is at u = c, so the sample position
        // is u + 64 - 0.5). The blend is compressed to a quarter cell around the boundary so a shadow
        // edge stays nearly binary while remaining stable frame-to-frame.
        float uf = u + 64.0 - 0.5;
        float vf = v + 64.0 - 0.5;
        int u0 = clamp(int(floor(uf)), 0, FOG_VOLUME_CELLS - 1);
        int v0 = clamp(int(floor(vf)), 0, FOG_VOLUME_CELLS - 1);
        int u1 = min(u0 + 1, FOG_VOLUME_CELLS - 1);
        int v1 = min(v0 + 1, FOG_VOLUME_CELLS - 1);
        float wu = smoothstep(0.25, 0.75, uf - floor(uf));
        float wv = smoothstep(0.25, 0.75, vf - floor(vf));

        uint p00 = ConstPtr<uint32_t>(pc.fogVolumeAddr)[levelBase + zBase
                + uint(v0) * uint(FOG_VOLUME_CELLS) + uint(u0)];
        uint p01 = ConstPtr<uint32_t>(pc.fogVolumeAddr)[levelBase + zBase
                + uint(v0) * uint(FOG_VOLUME_CELLS) + uint(u1)];
        uint p10 = ConstPtr<uint32_t>(pc.fogVolumeAddr)[levelBase + zBase
                + uint(v1) * uint(FOG_VOLUME_CELLS) + uint(u0)];
        uint p11 = ConstPtr<uint32_t>(pc.fogVolumeAddr)[levelBase + zBase
                + uint(v1) * uint(FOG_VOLUME_CELLS) + uint(u1)];

        tinted = tinted || ((p00 | p01 | p10 | p11) & FOG_PACK_EXACT_COLOR) != 0u;

        float3 t00 = decodeFogPacked(p00);
        float3 t01 = decodeFogPacked(p01);
        float3 t10 = decodeFogPacked(p10);
        float3 t11 = decodeFogPacked(p11);
        float3 levelT = lerp(lerp(t00, t01, wu), lerp(t10, t11, wu), wv);
        transmittance *= levelT;
    }
    return transmittance;
}

public FogIntegral fogNoScatter() {
    FogIntegral result;
    result.inscatter = float3(0.0, 0.0, 0.0);
    result.transmittance = float3(1.0, 1.0, 1.0);
    return result;
}

/**
 * March the fog along one segment and return its view-integrated in-scatter plus transmittance.
 *
 * {@code ro}/{@code rd} are the segment as the path tracer holds it; {@code light} is the dominant
 * celestial light in ACEScg illuminance and {@code sky} the per-frame sky state (its viewer radius and
 * body directions are what the atmosphere ratio needs). {@code tMax} is the geometry distance (miss case
 * uses 10000; the march caps itself at {@code fog.farEnd}).
 */
public FogIntegral integrateFog(FogState fog, float3 ro, float3 rd, float tMax, uint seed,
                                CelestialLight light, SkyState sky) {
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
    // Fetch frame + anchor of the baked volume: fixed for the whole segment (the bake used the dominant
    // direction and the same world-pinned lattice), so derive them once rather than once per fog step.
    float3 lightRight;
    float3 lightUp;
    celestialSquareFrame(light.dir, lightRight, lightUp);
    float3 fogAnchor = fogVolumeAnchor(worldPush.camOffset, light.dir);

    // Transmittance source for the whole march: either the exact-colour probe (below) or the entity-only
    // ray. ONE RT ray per march segment at most in both modes — the old per-step cost was 12 full-TLAS
    // rays per fog segment. Both are lazy, so fog entirely culled by the height/density checks or fully
    // shadowed (cache < FOG_PROBE_MIN_TRANSMITTANCE) pays nothing.
    //
    // EXACT-PROBE MODE. The cache cannot carry per-channel tint (it is monochrome by construction) and its
    // half-cell tap softens edges slightly, so for tinted cutout/translucent columns the march fires ONE
    // ray at the first still-LIT tinted sample. Critically, the measured value is used ONLY for its own
    // step: a probe fired inside a shadow measures ~0 at that sample, and reusing it for the rest of the
    // march was what made fog disappear in occluded areas. Other tinted steps keep the cache MAGNITUDE
    // (their own occlusion is correct per-cell) and take only the probe's hue; entity occlusion comes from
    // a tint-robust ratio of the same ray, so it never darkens a march that is merely tinted.
    bool exactProbe = false;
    int exactStep = -1;
    float3 exactT = float3(1.0, 1.0, 1.0);
    float3 exactTint = float3(1.0, 1.0, 1.0);
    float3 exactEntityT = float3(1.0, 1.0, 1.0);
    // Probe pass: identical cheap sample evaluation as the integration loop below (no RT unless a tinted
    // lit column is found) — keep the two in sync.
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
        float fade = 1.0 - smoothstep(fog.farEnd * 0.7, fog.farEnd, tMid);
        float sigma = fogDensity(fog, p) * fade;
        if (sigma <= FOG_MIN_DENSITY) {
            continue;
        }
        uint stepSeed = seed ^ (uint(i) * 0x9e3779b9u + 0x85ebca6bu);
        stepSeed = pcg(stepSeed);
        float3 lightDir = light.halfAngle > 0.0
                ? sampleSquare(light.dir, light.halfAngle, stepSeed) : light.dir;
        bool tinted;
        float3 terrainT = fogTerrainTransmittance(p, light.dir, lightRight, lightUp, fogAnchor, tinted);
        if (tinted && any(terrainT > FOG_PROBE_MIN_TRANSMITTANCE)) {
            exactT = visibilityFogExact(p, lightDir, 512.0).transmittance;
            // Hue normalised by the brightest channel (stable for strongly tinted glass/leaves).
            float tintMax = max(max(exactT.x, exactT.y), exactT.z);
            exactTint = tintMax > 1.0e-4 ? exactT / float3(tintMax) : float3(1.0, 1.0, 1.0);
            // Entity occlusion: entities attenuate every channel; tint attenuates channels unevenly, so
            // the MAX per-channel ratio is the closest colour-independent estimate. Clamped to [0,1] so
            // a tinted ray can never darken the march as if an occluder were there.
            float3 ratio = exactT / max(terrainT, float3(1.0e-4));
            exactEntityT = clamp(max(max(ratio.x, ratio.y), ratio.z), 0.0, 1.0);
            exactStep = i;
            exactProbe = true;
            break;
        }
    }

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
        // sample and step, flashing through gaps. The penumbra is the volume's deterministic narrow 2x2
        // tap; the jittered direction below is shared by the probe/entity fallback rays (same sample
        // direction the surface NEE and the old per-ray visibility use).
        uint stepSeed = seed ^ (uint(i) * 0x9e3779b9u + 0x85ebca6bu);
        stepSeed = pcg(stepSeed);
        float3 lightDir = light.halfAngle > 0.0
                ? sampleSquare(light.dir, light.halfAngle, stepSeed) : light.dir;
        bool tinted;
        float3 terrainT = fogTerrainTransmittance(p, light.dir, lightRight, lightUp, fogAnchor, tinted);
        if (exactProbe) {
            if (i == exactStep) {
                // The measured ground truth for THIS sample only: per-channel tint, alpha dapple, crisp
                // edge, and real entity occlusion in one ray.
                terrainT = exactT;
            } else {
                // Cache per-sample magnitude (each step's own shadow/occlusion stays correct), the
                // probe's hue for tinted columns, and the probe's entity estimate for every step.
                if (tinted) {
                    terrainT *= exactTint;
                }
                terrainT *= exactEntityT;
            }
        } else if ((worldPush.flags & 8u) != 0u) {
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
        float3 scattered = direct;
        float3 safeExtinction = max(sigmaT, float3(1.0e-6));
        result.inscatter += T * (scattered - scattered * stepT) / safeExtinction;
        T *= stepT;
    }
    result.transmittance = T;
    return result;
}
