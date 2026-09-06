// Volumetric air fog: exponential height haze with physically based in-scatter, integrated into the
// path integral in world.rgen (indirect pass) and at the pass-A dielectric split. This is the same
// medium the atmosphere's sky-view LUT runs with — the same Rayleigh/Cornette-Shanks phase functions,
// the same transmittance LUT for light reaching a sample, the same multiple-scattering LUT for light
// already scattered by the air — but at block scale around the camera rather than planet scale.
//
// The fog RTE splits into two terms that share one march:
//
//   * Direct — sun/moon light scattered toward the viewer: sigmaS * phase(cos) * E_sample *
//     T_terrain. E_sample is the dominant body's viewer-space illuminance corrected by the
//     transmittance-LUT ratio between the sample's column and the viewer's (the ratio, not the raw LUT —
//     the atmosphere must never be applied twice). T_terrain is a hardware RT visibility ray per march
//     sample (the same TerminateOnFirstHit/SkipClosestHit path surface NEE uses — this renderer has no
//     shadow map), so canopy and buildings slice the beams at block resolution with real penumbra, not a
//     binary light test. The LUT colouring is the sky's own, so fog carries the same sunset hue.
//   * Ambient — the sky's multiple-scattering LUT, evaluated per body and multiplied by the fog's
//     scattering. It arrives from every direction and has no phase, exactly as
//     sky.integrateScatteredLuminance adds it. This is what keeps twilight and moonlit night fog
//     physical rather than black; terrain GI in-scatter is a separate transport term that is not part
//     of this module.
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
public static const float FOG_MIN_DENSITY = 1.0e-4; // per block; below this a sample fires no visibility ray

public struct FogState {
    public bool   enabled;         // bit 1 of flags — whole effect
    public bool   ambient;         // bit 2 of flags — sky-multiscatter ambient in-scatter
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
    f.ambient = (push.flags & 4u) != 0u;
    f.density = max(push.fogLook0.x, 0.0);
    f.mieFraction = clamp(push.fogLook0.y, 0.0, 1.0);
    f.scaleHeight = max(push.fogLook0.z, 1.0);
    f.farEnd = max(push.fogLook0.w, 1.0);
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
 * celestial light in ACEScg illuminance and {@code sky} the per-frame sky state (both bodies' ambient
 * illuminance/directions). {@code tMax} is the geometry distance (miss case uses 10000; the march caps
 * itself at {@code fog.farEnd}).
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
    // One sky-occlusion probe per march feeds the ambient term: the multiscatter LUT is light already
    // in the sky, so it must not light fog behind a roof. Single ray, shared by every step — coherent,
    // and it stays a rough heuristic rather than a second per-sample cost. Debug toggle: with the
    // ambient term off (flags bit 2) the probe is not fired at all, so the direct-beam-only test costs
    // the same as a plain fog-off frame minus the march.
    float skyVisible = 0.0;
    if (fog.ambient) {
        float3 mid = ro + rd * (tEnd * 0.5);
        VisibilityResult skyVis = visibility(mid, float3(0.0, 1.0, 0.0), 10000.0);
        skyVisible = max(skyVis.transmittance.x,
                max(skyVis.transmittance.y, skyVis.transmittance.z));
    }

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

        // Direct: one RT visibility ray toward the jittered body. Jitter within the body's half-angle
        // gives the penumbra the surface NEE already uses; the ray itself is the accurate terrain
        // transmittance (T_terrain), NOT a shadow-map binary.
        uint stepSeed = seed ^ (uint(i) * 0x9e3779b9u + 0x85ebca6bu);
        stepSeed = pcg(stepSeed);
        float3 lightDir = light.halfAngle > 0.0
                ? sampleSquare(light.dir, light.halfAngle, stepSeed) : light.dir;
        VisibilityResult vis = visibility(p, lightDir, 10000.0);
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
        float3 direct = sigmaS * (phase * (light.illuminance * bt709ToAcesCg(sampleRelT) * vis.transmittance));

        // Ambient: second-and-higher order atmosphere light per unit illuminance, both bodies, with one
        // coarse terrain-sky occlusion probe so enclosed fog does not glow. Skipped when the debug
        // toggle disables it, leaving the direct beam term as the only fog light.
        float3 ambient = float3(0.0, 0.0, 0.0);
        if (fog.ambient) {
            float3 ambient709 = sampleMultiScatter(multiScatterLut, radiusKm, sky.sunDir.y)
                    * sky.sunIlluminance
                    + sampleMultiScatter(multiScatterLut, radiusKm, sky.moonDir.y) * sky.moonIlluminance;
            ambient = sigmaS * bt709ToAcesCg(ambient709)
                    * float3(skyVisible, skyVisible, skyVisible);
        }

        // Analytic constant-medium segment integral, same form the sky's raymarch uses.
        float3 scattered = direct + ambient;
        float3 safeExtinction = max(sigmaT, float3(1.0e-6));
        result.inscatter += T * (scattered - scattered * stepT) / safeExtinction;
        T *= stepT;
    }
    result.transmittance = T;
    return result;
}
