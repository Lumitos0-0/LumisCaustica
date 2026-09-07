// Froxel fog: temporal accumulation + depth integration, one thread per froxel COLUMN.
//
// Both jobs live in one pass because the integration must march Z in order anyway, so the blended
// value is produced and consumed inside a single traversal — no intermediate volume, one read and one
// write of the accumulation image per froxel.
//
// This pass owns the denoising. There is deliberately NO spatial blur anywhere in the direct fog path:
// blurring a froxel volume is what turns shafts into glowing mush, and it is the failure this whole
// design is arranged to avoid. Cleanliness comes from three things instead — the per-frame grid jitter
// the injection pass sampled with (which makes trilinear interpolation behave as temporal
// supersampling rather than permanent blur), the change-adaptive blend below, and the sun-disc area
// sampling that already gave the shaft edge a physical penumbra.
import world_common;
import sky;
import fog;
import bindings;

[[vk::push_constant]] PushAddr pc;

static const uint GROUP_SIZE = 8;

[shader("compute")]
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID) {
    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    FogGrid grid = fogGrid(worldPush);
    int2 column = int2(dispatchThreadId.xy);
    if (!grid.enabled || any(column >= grid.dims.xy)) {
        return;
    }

    // Column-constant part of the sampling offset; the Z component varies per slice below. Must match
    // fog_inject exactly or density would be evaluated at a different point than the ray was traced.
    float3 columnJitter = fogJitter(worldPush, column, 0);
    FogRay ray = fogColumnRay(worldPush, grid, column, columnJitter.xy);
    // The reprojection must describe a fixed world point across frames, so it needs the column ray with
    // the jitter removed. Reusing the jittered ray would move the "stable" position by up to a froxel
    // every frame, so the history lookup would chase its own sampling offset and the accumulated result
    // would never converge — which defeats the entire purpose of jittering.
    FogRay stableRay = fogColumnRay(worldPush, grid, column, float2(0.0, 0.0));
    bool historyValid = fogHistoryValid(worldPush);

    // Blend weight for this frame's sample against the accumulated history. Fixed, not adaptive --
    // see the note at the blend below. fogTemporal.yz are unused by the current filter.
    float alphaFloor = worldPush.fogTemporal.x;

    // The phase is applied HERE, per frame, from the current ray direction — the injection stores a
    // phaseless value precisely so that camera rotation cannot leave a tail of stale phase in the
    // history. Only the direction is needed, and skyState is a pure function of the push block, so no
    // transmittance LUT (which this pass does not bind) is required.
    SkyState sky = skyState(worldPush);
    float3 lightDir = sky.sunIlluminance >= sky.moonIlluminance ? sky.sunDir : sky.moonDir;
    float anisotropy = worldPush.fogParams1.z;

    float3 integrated = float3(0.0, 0.0, 0.0);
    float transmittance = 1.0;
    float previousDistance = grid.near;

    for (int slice = 0; slice < grid.dims.z; ++slice) {
        int3 index = int3(column, slice);
        float4 scatterSample = fogScatter[index];
        float3 raw = scatterSample.rgb;
        float rawAmbient = scatterSample.a;

        float sliceDistance = fogSliceDistance(grid, float(slice) + 0.5 + fogJitter(worldPush, column, slice).z);
        float3 position = ray.origin + ray.dir * sliceDistance;
        float extinction = fogDensity(worldPush, position);

        // ---- Temporal accumulation.
        //
        // Reprojection uses the UNJITTERED froxel centre on the UNJITTERED ray: the jitter is a
        // sampling offset, not motion, and carrying it into the reprojection would chase the history
        // around by a froxel each frame and undo the supersampling it exists to provide.
        float3 stablePosition = stableRay.origin
                + stableRay.dir * fogSliceDistance(grid, float(slice) + 0.5);
        // Fallback when reprojection fails: this frame's sample, with a history length of one.
        float4 history = float4(raw, 1.0);
        bool reprojected = false;
        if (historyValid) {
            // Camera-relative -> previous clip. camDelta moves the point into the previous frame's
            // camera-relative space, exactly as the guide motion vectors do.
            float3 cameraRelative = stablePosition - worldPush.camOffset;
            float4 previousClip = mul(worldPush.prevViewProj,
                    float4(cameraRelative + worldPush.camDelta, 1.0));
            if (previousClip.w > 0.0) {
                float3 previousNdc = previousClip.xyz / previousClip.w;
                float2 previousUv = previousNdc.xy * 0.5 + 0.5;
                // Distance from the PREVIOUS eye, so the slice lookup matches how that frame's grid was
                // built rather than reusing this frame's radial distance.
                float previousDistanceAlongRay = length(cameraRelative + worldPush.camDelta);
                float previousW = fogDistanceW(grid, previousDistanceAlongRay);
                if (all(previousUv >= 0.0) && all(previousUv <= 1.0)
                        && previousDistanceAlongRay >= grid.near
                        && previousDistanceAlongRay <= grid.far) {
                    history = fogHistoryPrev.SampleLevel(float3(previousUv, previousW), 0.0);
                    reprojected = true;
                }
            }
        }

        // Fixed-rate exponential moving average, as Frostbite and UE use for exactly this buffer.
        //
        // An alpha driven by |raw - history| cannot work here and was the cause of the speckle: this
        // froxel is a ONE-SAMPLE Monte Carlo estimate, so consecutive raw values differ enormously by
        // design wherever the sun disc is partly occluded — the shaft edges and cave mouths where the
        // artefact was worst. Reacting to that difference pins alpha near 1, which is the same as
        // having no temporal filter at all, and the estimator never converges. The relative form made
        // it worse still: dividing by a near-zero mean makes dark froxels the most reactive of all,
        // which is why an unlit cave was the noisiest thing on screen.
        //
        // A neighbourhood colour clamp — the usual TAA companion to a fixed EMA — is deliberately NOT
        // applied. It assumes neighbours are a better estimate of the truth than the history, which
        // holds for a resolved TAA colour buffer but not here: every neighbour is itself a one-sample
        // estimate drawn from the same binary visibility distribution, so the clamp bound is the full
        // sample range and clamping into it just re-injects the noise the average removed. Measured, it
        // widened the steady-state spread from 3.0 to 10.0 on a half-occluded froxel.
        //
        // Genuine lighting changes are instead handled by the EMA's own time constant. At alpha 0.04
        // that is a ~25 frame tail, which for a low-frequency medium reads as fog settling rather than
        // as smearing.
        //
        // The alpha channel carries the froxel's history LENGTH in frames rather than a second moment
        // (which only the discarded adaptive rule consumed). It lets a young froxel average as 1/n --
        // the unbiased mean of the samples it has actually seen -- instead of crawling from black at a
        // fixed 4% per frame. Without it, every froxel that fails reprojection (screen edges, anything
        // newly disoccluded, the first frame after a rebase) spends ~25 frames visibly filling in.
        float historyLength = reprojected ? min(history.a + 1.0, 1.0 / max(alphaFloor, 1.0e-3)) : 1.0;
        float alpha = max(1.0 / historyLength, alphaFloor);

        float3 blended = lerp(history.rgb, raw, alpha);
        blended = clamp(blended, float3(0.0), float3(HALF_MAX));
        fogHistory[index] = float4(blended, historyLength);
        // Apply this frame's phase to the accumulated, view-INDEPENDENT scattering.
        //
        // The ambient fill is isotropic and unshadowed, so it carries no Monte Carlo noise and needs no
        // temporal accumulation at all — this frame's value is already exact. It is taken straight from
        // the raw sample (where it was stored pre-scaled by 1/4pi) rather than blended, which also keeps
        // it out of the history and leaves the alpha channel free for the history length.
        float phase = fogPhase(dot(ray.dir, lightDir), anisotropy);
        float3 lit = blended * phase + rawAmbient;

        // ---- Depth integration.
        //
        // Analytic integration of the constant-medium step, the same closed form
        // sky.slang::integrateScatteredLuminance uses: exact over the step rather than a midpoint
        // sample, which is what lets a modest slice count run without visible banding.
        //
        // The step is split at the slice CENTRE and the prefix is stored there, because that is where a
        // linear sampler reads it: texel s covers w in [s/dims.z, (s+1)/dims.z] and its centre — the
        // coordinate at which its value is returned unblended — is (s+0.5)/dims.z. Storing the integral
        // to the slice's far edge instead would bias every fetch half a slice too far, which is
        // invisible near the camera and grows to metres out at the far plane where slices are longest.
        float sliceCentre = fogSliceDistance(grid, float(slice) + 0.5);
        float sliceFar = fogSliceDistance(grid, float(slice) + 1.0);
        if (extinction > FOG_DENSITY_EPSILON) {
            float toCentre = max(sliceCentre - previousDistance, 0.0);
            if (toCentre > 0.0) {
                float stepTransmittance = exp(-extinction * toCentre);
                integrated += transmittance * (lit - lit * stepTransmittance) / extinction;
                transmittance *= stepTransmittance;
            }
        }
        fogIntegrated[index] = float4(clamp(integrated, float3(0.0), float3(HALF_MAX)), transmittance);
        if (extinction > FOG_DENSITY_EPSILON) {
            float toFar = max(sliceFar - sliceCentre, 0.0);
            if (toFar > 0.0) {
                float stepTransmittance = exp(-extinction * toFar);
                integrated += transmittance * (lit - lit * stepTransmittance) / extinction;
                transmittance *= stepTransmittance;
            }
        }
        previousDistance = sliceFar;
    }
}
