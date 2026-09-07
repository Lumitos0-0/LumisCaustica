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

    // Same grid offset the injection used, so this pass reconstructs the exact position each froxel's
    // ray was traced from. Both come from the push block, so they cannot drift apart.
    float2 gridJitter = fogGridJitterUv(worldPush);
    float sliceJitter = fogSliceJitter(worldPush);
    FogRay ray = fogColumnRay(worldPush, grid, column, gridJitter);
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
    // The isotropic ambient fill shares the direct term's shadowing and colour, differing only in its
    // phase (the constant 1/4pi). It is therefore exactly `direct * ambient/4pi`, which folds into the
    // phase as an additive constant rather than needing its own channel or its own volume.
    //
    // Note this makes the fill OCCLUDED, unlike the earlier unshadowed version: it no longer glows
    // inside caves, so it is safe to raise above zero.
    float ambientPhase = FOG_INV_4PI * worldPush.fogParams2.z;

    float3 integrated = float3(0.0, 0.0, 0.0);
    float transmittance = 1.0;
    float previousDistance = grid.near;

    for (int slice = 0; slice < grid.dims.z; ++slice) {
        int3 index = int3(column, slice);
        float4 scatterSample = fogScatter[index];
        float3 raw = scatterSample.rgb;
        // Extinction comes from the injection pass rather than being recomputed here, so both passes
        // are guaranteed to describe the same point of the density field, and the field is evaluated
        // once per froxel per frame instead of twice.
        float extinction = scatterSample.a;
        float sliceDistance = fogSliceDistance(grid, float(slice) + 0.5 + sliceJitter);

        // ---- Temporal accumulation.
        //
        // Reproject the froxel's ACTUAL sample position — the jittered one this frame's ray was traced
        // from. The grid offset is now shared by every column rather than per-column noise, so the whole
        // lattice translates rigidly and reprojecting the true position is both correct and stable: the
        // history lookup below undoes the previous frame's offset explicitly.
        float3 samplePosition = ray.origin + ray.dir * sliceDistance;
        // Fallback when reprojection fails: this frame's sample, with a history length of one.
        float4 history = float4(raw, 1.0);
        bool reprojected = false;
        if (historyValid) {
            // Camera-relative -> previous clip. camDelta moves the point into the previous frame's
            // camera-relative space, exactly as the guide motion vectors do.
            float3 cameraRelative = samplePosition - worldPush.camOffset;
            float4 previousClip = mul(worldPush.prevViewProj,
                    float4(cameraRelative + worldPush.camDelta, 1.0));
            if (previousClip.w > 0.0) {
                float3 previousNdc = previousClip.xyz / previousClip.w;
                // Subtract the PREVIOUS frame's grid offset: the history volume was written through a
                // lens shifted by that amount, so a world point's texel there sits at
                // (screen uv - previous offset). Skipping this would misread the history by up to a
                // froxel every frame, which is exactly the jitter-without-a-resolve error this change
                // exists to remove.
                float2 previousUv = previousNdc.xy * 0.5 + 0.5 - fogPrevGridJitterUv(worldPush);
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

        // Fixed-rate exponential moving average with a 1/n warmup and a staleness test.
        //
        // An alpha driven by |raw - history| cannot work here: each froxel is a ONE-SAMPLE Monte Carlo
        // estimate, so consecutive raw values differ enormously by design wherever the sun disc is
        // partly occluded, and reacting to that pins alpha near 1 -- no temporal filter at all.
        //
        // But a pure fixed rate has the opposite failure: reprojection only checks that the froxel is on
        // screen and in range, never that the history still DESCRIBES this froxel. When the camera moves
        // so a froxel emerges from behind an occluder, the stale history keeps its full weight and drags
        // ~1/alpha frames of old radiance -- the reprojection lag.
        //
        // The test below distinguishes the two. A single slice is far too noisy to judge staleness
        // against, but the mean of several slices' CURRENT samples is a low-variance estimate of what
        // this region looks like now, and the medium is smooth along Z so neighbouring slices are a fair
        // reference. If the converged history disagrees with that estimate by more than several standard
        // errors, it is stale rather than merely noisy, and the froxel restarts its accumulation.
        float alpha;
        {
            float historyLength = reprojected ? history.a : 0.0;
            if (reprojected && historyLength > 2.0) {
                float localSum = 0.0;
                int taps = 0;
                for (int d = -FOG_STALENESS_TAPS; d <= FOG_STALENESS_TAPS; ++d) {
                    int neighbour = clamp(slice + d, 0, grid.dims.z - 1);
                    localSum += fogLuminance(fogScatter[int3(column, neighbour)].rgb);
                    ++taps;
                }
                float local = localSum / float(taps);
                // Standard error of the local mean, floored so a near-zero region cannot make the test
                // infinitely strict.
                float standardError = max(sqrt(max(local, 0.0) / float(taps)), local * 0.05 + 1.0e-6);
                if (abs(fogLuminance(history.rgb) - local) > FOG_STALENESS_SIGMA * standardError) {
                    historyLength = 0.0;
                }
            }
            historyLength = min(historyLength + 1.0, 1.0 / max(alphaFloor, 1.0e-3));
            alpha = max(1.0 / historyLength, alphaFloor);
            history.a = historyLength;
        }

        float3 blended = lerp(history.rgb, raw, alpha);
        blended = clamp(blended, float3(0.0), float3(HALF_MAX));
        fogHistory[index] = float4(blended, history.a);
        // Apply this frame's phase to the accumulated, view-INDEPENDENT scattering.
        float phase = fogPhase(dot(ray.dir, lightDir), anisotropy) + ambientPhase;
        float3 lit = blended * phase;

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
