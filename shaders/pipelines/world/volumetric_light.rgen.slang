// World-anchored radiance probe volume for the fog (RTXGI-style). One raygen thread per light-volume
// cell; the volume is a regular world-space grid snapped to FROXEL_LIGHT_CELL, so probes stay at fixed
// world positions while the camera moves — this is what removes the grid wobble: a frustum-aligned
// grid re-samples different world positions every frame and its history fights the camera. Each probe
// accumulates an estimate of the average incident radiance at its position:
//
//   * one uniform-sphere GI path per sample (restricted probe path: no surface RIS, no direct emitter
//     gather, no SSS/gloss continuation, no stars, no celestial disc) — terrain/canopy/sky radiance;
//   * pdf-correct next-event samples over the block-emitter light grid, phase-folded into the
//     isotropic average (torch/lava glow enters the fog with a bounded low-variance estimate, not an
//     accidental path hit);
//
// into a gamma-encoded running average (the production DDGI probe-update curve: HDR single samples are
// compressed before they enter history, so what would be a bright per-cell dot becomes a stable
// low-frequency term). The fog inject pass queries this volume trilinearly — probes are far fewer than
// fog light cells and each is averaged over up to 64 frames, so the fog's GI term carries no
// per-cell sampling noise by construction.
//
// The ping-pong pair lets history follow the camera-anchor shift: the update reads the previous lane at
// (index - shift), which is the position this probe occupied last frame (same world coordinates), so
// accumulated data survives camera motion instead of restarting.
import world_common;
import world_core;
import bindings;
import sky;
import math;
import trace;
import water;
import froxel;
import medium;
import segment;
import path_trace;
import lighting;

public static const float FROXEL_LIGHT_GAMMA = 5.0;
public static const float FROXEL_LIGHT_INV_GAMMA = 1.0 / FROXEL_LIGHT_GAMMA;
public static const float FROXEL_LIGHT_MAX_COUNT = 64.0;
public static const uint FROXEL_LIGHT_EMITTER_SAMPLES = 2u;

/**
 * Isotropic block-emitter contribution: next-event sample over the local light-grid span, divided by
 * the exact alias-table pdf (so the estimate is unbiased and bounded), with phase folded into the
 * spherical average. The stored field is the average incident radiance (1/4π times the sphere
 * integral), and the fog query approximates the emitter's phase by the isotropic 1/(4π) — torches read
 * as round glow instead of shafted spikes.
 */
public float3 froxelLightEmitter(float3 position, inout uint sampleSeed, inout uint proposalSeed) {
    if (worldPush.lightCount == 0u || pc.lightBufAddr == 0 || pc.lightLocalAliasAddr == 0) {
        return float3(0.0, 0.0, 0.0);
    }
    LightGridCell gridCell;
    int3 gridCellCoord;
    if (!findLightGridCell(position, gridCell, gridCellCoord)
            || gridCell.invWeightSum <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    float3 sum = float3(0.0, 0.0, 0.0);
    for (uint s = 0u; s < FROXEL_LIGHT_EMITTER_SAMPLES; ++s) {
        uint lightIndex;
        selectLightGridSpanLight(gridCell, proposalSeed, lightIndex);
        Light light = ConstPtr<Light>(pc.lightBufAddr)[lightIndex];
        float3 emission = lightRadiance(light);
        float area = lightArea(light);
        float3 rectanglePoint = light.pos + worldPush.lightRebase.xyz
                + (rndf(sampleSeed) * 2.0 - 1.0) * lightHalfU(light)
                + (rndf(sampleSeed) * 2.0 - 1.0) * lightHalfV(light);
        float3 toLight = rectanglePoint - position;
        float dist = length(toLight);
        if (dist <= RAY_TMIN) {
            continue;
        }
        float3 lightDir = toLight / dist;
        float emitterCosine = max(0.0, dot(lightGeometricNormal(light), -lightDir));
        if (emitterCosine <= 0.0) {
            continue;
        }
        // localProbability=1 reconstructs the exact weighted neighborhood proposal PDF.
        float sourcePdf = proposalPdf(light, emission, gridCell, gridCellCoord, 1.0);
        if (sourcePdf <= 1.0e-20) {
            continue;
        }
        float3 visible = visibility(position, lightDir, dist * 0.999).transmittance;
        // Area-light point sample: integrand * area / selection pdf. The 1/(4π) converts the emitter's
        // directional radiance into the same average-incident-radiance units as the GI samples above.
        sum += emission * (emitterCosine * area / (dist * dist)) * visible
                * (1.0 / sourcePdf) / (4.0 * PI);
    }
    return sum * (1.0 / float(FROXEL_LIGHT_EMITTER_SAMPLES));
}

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    // Same directional light derivation as every other raygen: the probe paths' surface NEE needs it.
    celestialLight = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);

    int3 dims = froxelLightDimensions();
    uint3 tid = DispatchRaysIndex();
    if (any(tid >= uint3(dims))) {
        return;
    }
    float3 cellSize = float3(FROXEL_LIGHT_CELL);
    float3 anchorCur = froxelLightAnchor(worldPush.camOffset);
    float3 anchorPrev = froxelLightAnchor(worldPush.camOffset - worldPush.camDelta);
    int3 shift = int3(round((anchorCur - anchorPrev) / cellSize));
    float3 pos = anchorCur + (float3(tid) + 0.5) * cellSize;

    // History from other probes carries over only if this probe occupied the same world position last
    // frame (in-bounds shifted read) and the scene's fog parameters are unchanged (historyValid). Probes
    // entering as the camera moves (out-of-bounds shifted read) seed from the nearest in-bounds probe
    // with halved confidence, so new cells interpolate the field instead of flashing a single raw
    // sample. A teleport shifts far and must restart, not inherit unrelated radiance.
    uint writeIdx = worldPush.frameIndex & 1u;
    uint historyIdx = 1u - writeIdx;
    float4 prev = float4(0.0, 0.0, 0.0, 0.0);
    bool prevValid = froxelHistoryValid() && all(abs(shift) <= int3(4, 4, 4));
    if (prevValid) {
        int3 prevIdx = int3(tid) - shift;
        int3 readIdx = clamp(prevIdx, int3(0, 0, 0), dims - 1);
        if (historyIdx == 0u) {
            prev = froxelLightA[readIdx];
        } else {
            prev = froxelLightB[readIdx];
        }
        if (any(prevIdx != readIdx)) {
            prev.w = min(prev.w, 2.0); // half-weight warm-up from the boundary neighbour
        }
    }

    // Sample this probe's average incident radiance. GI strength gates the whole non-directional
    // field, so the value pushed to the volume is already scaled and setting changes reset history.
    float giStrength = worldPush.fogLighting.y;
    uint giSamples = froxelGiSamples();
    float3 radiance = float3(0.0, 0.0, 0.0);
    if (giStrength > 0.0 && giSamples > 0u) {
        uint seed = (tid.x * 1973u + tid.y * 9277u + tid.z * 5939u + 26699u)
                ^ (worldPush.frameIndex * 2654435761u);
        seed = pcg(seed);
        uint proposalSeed = (tid.x * 12289u + tid.y * 41521u + tid.z * 25457u)
                ^ (worldPush.frameIndex * 40503u);
        proposalSeed = pcg(proposalSeed);
        MediumStack probeMedium = makeMediumStack(froxelUnderwater()
                ? makeDielectricMedium(true, WATER_IOR, worldPush.waterParams.xyz, 1.0)
                : airMedium());
        // One shaded surface per path is enough: the volume field is low-frequency and the fog is a
        // blurry medium — deeper chains cost rays for detail that cannot show.
        int probeBounce = max(1, int(worldPush.maxBounces) - int(froxelGiBounces()));
        for (uint s = 0u; s < giSamples; ++s) {
            // Uniform sphere direction (phase folded into the isotropic average).
            float3 dir = samplePhase(float3(0.0, 1.0, 0.0), 0.0, seed);
            PathSegment probe = makePathSegment(pos, dir, float3(1.0, 1.0, 1.0), probeMedium,
                    0.0, 0.0, seed, probeBounce, false, true);
            radiance += tracePath(probe, tid.xy, s);
        }
        radiance *= 1.0 / float(giSamples);
        radiance += froxelLightEmitter(pos, seed, proposalSeed);
        radiance *= giStrength;
    }

    // Progressive gamma-encoded accumulation: each probe converges to a stable field in ~64 frames
    // (or instantly on first discovery), and a single HDR sample can never dominate the history.
    float3 encoded = pow(max(radiance, float3(0.0)), float3(FROXEL_LIGHT_INV_GAMMA));
    float4 result;
    if (prevValid && prev.w > 0.0) {
        float count = min(prev.w + 1.0, FROXEL_LIGHT_MAX_COUNT);
        float weight = 1.0 / count;
        result = float4(lerp(prev.rgb, encoded, weight), count);
    } else {
        result = float4(encoded, 1.0);
    }
    if (writeIdx == 0u) {
        froxelLightA[int3(tid)] = result;
    } else {
        froxelLightB[int3(tid)] = result;
    }
}
