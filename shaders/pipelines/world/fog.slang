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
//     real penumbra — but without the ray cost. Entities are the one dynamic occluder the cache excludes;
//     when any entity exists (flags bit 3) the march keeps one RT visibility ray per sample for exactly
//     the old behaviour.
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
// Light-space fog volume (see fog_grid.build.comp): 4 levels, 128^3 cells each, cellSize 2^k blocks and
// extent 64*2^k blocks around the camera. Kept in lock-step with that pass.
public static const int   FOG_VOLUME_LEVELS = 4;
public static const int   FOG_VOLUME_CELLS = 128;

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

// Terrain transmittance from p along lightDir, from the per-frame light-space fog volume. The volume is
// a chain of four cubes: the level containing p covers to its own +z face, then each coarser level takes
// over at the shared boundary — the chain telescopes to T(p -> light) with at most five fetches. Fetch
// coordinates are jittered by the path seed so foliage dapples like the per-ray alpha test used to.
public float3 fogTerrainTransmittance(float3 p, float3 lightDir, inout uint seed) {
    float3 right;
    float3 up;
    celestialSquareFrame(lightDir, right, up);
    float3 rel = p - worldPush.camOffset;
    float uw = dot(rel, right);
    float vw = dot(rel, up);
    float zw = dot(rel, lightDir);

    // Jitter is ±1 cell of the fetched level on the sample's own fetch only, so it crosses one cell
    // boundary at every level (level-0 cells are 1 block, so this is the subcell dapple the per-ray
    // alpha test used to produce) while the shared coarse boundary cells stay fixed per frame and the
    // far field cannot flicker.
    float jitter = (rndf(seed) - 0.5) * 2.0;

    int first = 0;
    for (int k = 0; k < FOG_VOLUME_LEVELS - 1; k++) {
        float extent = 64.0 * exp2(float(k));
        if (max(abs(uw), max(abs(vw), abs(zw))) > extent) {
            first = k + 1;
        }
    }
    float3 transmittance = float3(1.0, 1.0, 1.0);
    for (int k = first; k < FOG_VOLUME_LEVELS; k++) {
        float size = exp2(float(k));
        // Coordinates are CELLS per axis (the volume stores 128 cells of size 2^k per level), so the
        // fetch index is simply cellCoordinate + 64. For k == first this is the sample itself; for the
        // coarser levels it is the boundary the previous level ends at (64 * 2^(k-1) blocks = 32 cells
        // in level k), so the fetched cells tile the ray exactly once.
        float boundary = (k == first) ? 0.0 : 32.0;
        float u = (k == first) ? (uw / size + jitter) : (uw / size);
        float v = (k == first) ? (vw / size + jitter) : (vw / size);
        float z = (boundary > 0.0 ? boundary : zw / size);
        int cu = clamp(int(floor(u + 64.0)), 0, FOG_VOLUME_CELLS - 1);
        int cv = clamp(int(floor(v + 64.0)), 0, FOG_VOLUME_CELLS - 1);
        int cz = clamp(int(floor(z + 64.0)), 0, FOG_VOLUME_CELLS - 1);
        uint index = uint((cz * FOG_VOLUME_CELLS + cv) * FOG_VOLUME_CELLS + cu)
                + uint(k) * uint(FOG_VOLUME_CELLS * FOG_VOLUME_CELLS * FOG_VOLUME_CELLS);
        uint packed = ConstPtr<uint32_t>(pc.fogVolumeAddr)[index];
        float3 levelT = float3(float(packed & 0xFFu), float((packed >> 8u) & 0xFFu),
                float((packed >> 16u) & 0xFFu)) * (1.0 / 255.0);
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
        // constant-time fetches — the entire ray cost is gone). The light direction is jittered within the
        // body's half-angle for the penumbra the surface NEE already uses; when any entity is present
        // (flags bit 3) one RT visibility ray per sample adds the dynamic occluders the cache excludes,
        // restoring exactly the old entity behaviour.
        uint stepSeed = seed ^ (uint(i) * 0x9e3779b9u + 0x85ebca6bu);
        stepSeed = pcg(stepSeed);
        float3 lightDir = light.halfAngle > 0.0
                ? sampleSquare(light.dir, light.halfAngle, stepSeed) : light.dir;
        float3 terrainT = fogTerrainTransmittance(p, lightDir, stepSeed);
        if ((worldPush.flags & 8u) != 0u) {
            // Bounded to the fog volume's extent: the march never scatters beyond farEnd (≤ 512), so an
            // occluder farther away cannot affect this segment — shorter rays, same result.
            VisibilityResult vis = visibility(p, lightDir, 512.0);
            terrainT *= vis.transmittance;
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
