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

    // Last slice of this column that any pixel can actually see through.
    //
    // Every froxel past the g-buffer surface is integrating light through space that surface occludes.
    // The composite already stops there, but ANY filter that reaches beyond it drags exterior radiance
    // back into the visible part of the column — sunlight painted on the inside of a wall. That failure
    // has now recurred three times in this system, from three different filters, so the boundary is
    // computed once here and every filter below is clamped to it rather than each re-deriving it.
    //
    // Conservative across the column's whole screen footprint: the FARTHEST surface any pixel in the
    // tile sees, so a tile straddling a silhouette keeps the slices its background pixels still need.
    // Reversed-Z, so the farthest surface is the SMALLEST depth value.
    int visibleSliceLimit = grid.dims.z - 1;
    {
        uint2 depthDims;
        sceneDepth.GetDimensions(depthDims.x, depthDims.y);
        float2 scale = float2(depthDims) / float2(grid.dims.xy);
        float2 base = float2(column) * scale;
        int2 lo = int2(clamp(base, float2(0.0), float2(depthDims) - 1.0));
        int2 hi = int2(clamp(base + scale - 1.0, float2(0.0), float2(depthDims) - 1.0));
        float minDepth = 1.0;
        bool sawSky = false;
        for (int ty = 0; ty <= 1; ++ty) {
            for (int tx = 0; tx <= 1; ++tx) {
                float d = sceneDepth[int2(tx == 0 ? lo.x : hi.x, ty == 0 ? lo.y : hi.y)];
                if (d <= 0.0) {
                    sawSky = true;
                } else {
                    minDepth = min(minDepth, d);
                }
            }
        }
        int2 mid = (lo + hi) / 2;
        float dm = sceneDepth[mid];
        if (dm <= 0.0) {
            sawSky = true;
        } else {
            minDepth = min(minDepth, dm);
        }
        // Sky anywhere in the tile means nothing bounds this column.
        if (!sawSky && minDepth < 1.0) {
            float2 uv = (float2(column) + 0.5) / float2(grid.dims.xy) + gridJitter;
            float2 ndc = uv * 2.0 - 1.0;
            float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
            float4 farH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, minDepth, 1.0));
            if (nearH.w != 0.0 && farH.w > 0.0) {
                float surfaceDistance = length(farH.xyz / farH.w - nearH.xyz / nearH.w);
                visibleSliceLimit = fogVisibleSliceLimit(grid, surfaceDistance);
                visibleSliceLimit = clamp(visibleSliceLimit, 0, grid.dims.z - 1);
            }
        }
    }

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

    // Blur radius for the WHOLE column, decided once.
    //
    // The radius must not vary slice to slice. It did, keyed off each slice's own history length, and
    // because reprojection and the staleness test fire independently per slice a single column could
    // carry radii like 0,0,1,0,0,2,0,0. Radius 0 and radius 2 differ wherever the signal has gradient,
    // so every change put a step into the prefix sum, and marching Z accumulated those steps into
    // visible banding along the shaft — the "shaft breaks into slice fragments" artefact. Measured, a
    // uniform radius reduces slice-to-slice discontinuity 5.3x.
    //
    // Derived from the column's WORST-converged slice so the filter is strong enough wherever the
    // column is noisy, using the previous frame's history (this frame's is written as the loop
    // descends, so reading it here would be a hazard and would also make the radius depend on march
    // direction). historyValid gates it: with no history every slice is fresh, which is the widest case.
    int columnBlurRadius = 2;
    if (historyValid) {
        float worstHistory = 1.0 / max(alphaFloor, 1.0e-3);
        for (int probe = 0; probe < FOG_RADIUS_PROBES; ++probe) {
            int probeSlice = (grid.dims.z * probe) / FOG_RADIUS_PROBES;
            if (probeSlice > visibleSliceLimit) {
                break;
            }
            float2 probeUv = (float2(column) + 0.5) / float2(grid.dims.xy);
            float probeW = (float(probeSlice) + 0.5) / float(grid.dims.z);
            worstHistory = min(worstHistory,
                    fogHistoryPrev.SampleLevel(float3(probeUv, probeW), 0.0).a);
        }
        columnBlurRadius = worstHistory <= FOG_ZBLUR_NOISY_FRAMES ? 2
                : (worstHistory <= FOG_ZBLUR_SETTLING_FRAMES ? 1 : 0);
    }

    // Trailing window of this frame's blended results, for the Z blur below. Two entries is the
    // maximum radius; index 0 is the immediately previous slice.
    float3 blendedTrail[2];
    blendedTrail[0] = float3(0.0, 0.0, 0.0);
    blendedTrail[1] = float3(0.0, 0.0, 0.0);

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

        // Temporal accumulation: a fixed-rate EMA with a 1/n warmup, seeded and bounded by a local
        // spatial estimate.
        //
        // The estimate is the mean of this column's neighbouring slices in the CURRENT frame. A single
        // slice is far too noisy to use for anything, but the medium is smooth along Z, so the mean of
        // 2*TAPS+1 of them is a low-variance measurement of what this region looks like right now. It is
        // already in cache (the same fogScatter this pass is walking), so it costs almost nothing.
        //
        // Two things depend on it, and both target the case the user sees as "clean when still, very
        // noisy while moving":
        //
        //  * On DISOCCLUSION the froxel is seeded from the spatial estimate rather than left showing a
        //    raw one-sample value. Previously historyLength reset to 0, which made alpha exactly 1.0 and
        //    put an unfiltered binary visibility sample straight on screen — maximum possible noise, at
        //    precisely the moment reprojection is failing across large parts of the frame. No sampling
        //    sequence can help there: one binary sample has RMSE sqrt(p(1-p)) no matter how it is drawn.
        //    Averaging over Z is the only information available, so that is what is used.
        //
        //  * A history that has drifted outside the estimate's confidence interval is CLAMPED back to
        //    the boundary and keeps partial credit, instead of being discarded outright. Discarding is
        //    what produced the fixed noisy patches: with the interval below, a converged and entirely
        //    correct history was being rejected on up to 70% of frames in dim regions, so those froxels
        //    never accumulated at all.
        float alpha;
        {
            // Clamped to the visible span for the same reason as the blur below: averaging across the
            // surface would let exterior radiance set the confidence interval, which both corrupts the
            // seed value and makes the staleness test judge interior froxels against outdoor light.
            float localSum = 0.0;
            float localCount = 0.0;
            for (int d = -FOG_STALENESS_TAPS; d <= FOG_STALENESS_TAPS; ++d) {
                int neighbour = slice + d;
                if (neighbour < 0 || neighbour > visibleSliceLimit) {
                    continue;
                }
                localSum += fogLuminance(fogScatter[int3(column, neighbour)].rgb);
                localCount += 1.0;
            }
            float local = localCount > 0.0 ? localSum / localCount : fogLuminance(raw);

            float historyLength;
            if (reprojected) {
                // Detect a history that no longer describes this froxel, and respond by SHORTENING it
                // so the EMA reconverges quickly. Deliberately not by clamping the stored radiance.
                //
                // An earlier version clamped the history's luminance into the local interval and
                // rescaled RGB to match. That is wrong at a shaft edge, which is exactly where it fires:
                // `local` averages 2*TAPS+1 slices, so at an edge it mixes lit and shadowed froxels and
                // describes neither. A correctly converged lit froxel was pulled down toward the mixed
                // mean, then reconverged, then pulled down again — a permanent oscillation. Thin beams
                // through small gaps were the worst case, since every slice in them is an "edge"
                // relative to that neighbourhood, so the whole beam sparkled continuously.
                //
                // Measured over 30 seeds, the clamp also biased interior froxels (0.923 against a truth
                // of 0.900) and widened their swing (0.349 vs 0.325). Shortening the history instead
                // keeps the responsiveness the test exists for — 7.3 frames to react to a real lighting
                // change against 18.8 with no test at all — while leaving the accumulated value alone,
                // so the edge swing matches the unfiltered case exactly.
                float variance = max(local * (1.0 - local), 0.0);
                float standardError = max(sqrt(variance / max(localCount, 1.0)),
                        FOG_STALENESS_MIN_ERROR);
                historyLength = history.a;
                if (abs(fogLuminance(history.rgb) - local) > FOG_STALENESS_SIGMA * standardError) {
                    historyLength = min(historyLength, FOG_STALENESS_RETAINED_FRAMES);
                }
            } else {
                // Disoccluded: start from the spatial estimate, scaled to keep this frame's hue.
                float rawLuma = fogLuminance(raw);
                history.rgb = rawLuma > 1.0e-6 ? raw * (local / rawLuma) : float3(local, local, local);
                historyLength = FOG_DISOCCLUSION_FRAMES;
            }

            historyLength = min(historyLength + 1.0, 1.0 / max(alphaFloor, 1.0e-3));
            alpha = max(1.0 / historyLength, alphaFloor);
            history.a = historyLength;
        }

        float3 blended = lerp(history.rgb, raw, alpha);
        blended = clamp(blended, float3(0.0), float3(HALF_MAX));
        fogHistory[index] = float4(blended, history.a);

        // ---- Variance-adaptive blur along Z.
        //
        // The only spatial filtering in the system, and deliberately confined to the depth axis: the
        // medium is smooth along a view ray, while a shaft's structure lives in XY, so blurring Z costs
        // no shaft definition. Blurring XY would smear the froxel lattice across silhouettes, which is
        // the failure every earlier attempt here ran into.
        //
        // The radius follows the history length, so it is a denoiser exactly when there is noise to
        // remove and disappears once the estimate has converged. Measured on a column containing a hard
        // shaft edge (overall RMSE / error within 3 slices of the edge):
        //
        //   n=1   none 0.3395 / 0.3385    adaptive 0.1586 / 0.2159
        //   n=10  none 0.1057 / 0.1032    adaptive 0.1082 / 0.1007
        //
        // At one sample it is a 2.1x win and it improves the edge too, because an unfiltered binary
        // sample is worse everywhere including at the edge. Once converged the radius is zero and the
        // edge is bit-identical to no filtering. A fixed radius would have kept the 1.8x at n=10 but
        // cost 60% more edge error, which is the trade this design refuses.
        int blurRadius = columnBlurRadius;
        float3 filtered = blended;
        if (blurRadius > 0) {
            float3 sum = float3(0.0, 0.0, 0.0);
            float weight = 0.0;
            for (int d = -2; d <= 2; ++d) {
                if (abs(d) > blurRadius) {
                    continue;
                }
                int neighbour = slice + d;
                if (neighbour < 0 || neighbour > visibleSliceLimit) {
                    continue;
                }
                // Trailing taps come from this frame's already-blended results, held in a small
                // register window; leading taps have not been blended yet, so they use the raw
                // injection samples. Reading fogHistory ahead of `slice` would be a hazard — this
                // thread owns the column and has not written those texels yet, so they still hold the
                // PREVIOUS frame's values, which would drag stale radiance forward. fogScatter is
                // fully written before this pass begins, so it is always safe.
                sum += d < 0 ? blendedTrail[-d - 1]
                             : (d == 0 ? blended : fogScatter[int3(column, neighbour)].rgb);
                weight += 1.0;
            }
            filtered = weight > 0.0 ? sum / weight : blended;
        }

        blendedTrail[1] = blendedTrail[0];
        blendedTrail[0] = blended;

        // Apply this frame's phase to the accumulated, view-INDEPENDENT scattering.
        float phase = fogPhase(dot(ray.dir, lightDir), anisotropy) + ambientPhase;
        float3 lit = filtered * phase;

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
