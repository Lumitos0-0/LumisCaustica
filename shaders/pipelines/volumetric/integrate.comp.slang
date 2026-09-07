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

    float3 jitter = worldPush.fogJitter.xyz;
    FogRay ray = fogColumnRay(worldPush, grid, column, jitter.xy);
    bool historyValid = fogHistoryValid(worldPush);

    float alphaFloor = worldPush.fogTemporal.x;
    float changeSensitivity = worldPush.fogTemporal.y;
    float varianceWeight = worldPush.fogTemporal.z;

    float3 integrated = float3(0.0, 0.0, 0.0);
    float transmittance = 1.0;
    float previousDistance = grid.near;

    for (int slice = 0; slice < grid.dims.z; ++slice) {
        int3 index = int3(column, slice);
        float3 raw = fogScatter[index].rgb;

        float sliceDistance = fogSliceDistance(grid, float(slice) + 0.5 + jitter.z);
        float3 position = ray.origin + ray.dir * sliceDistance;
        float extinction = fogDensity(worldPush, position);

        // ---- Temporal accumulation.
        //
        // Reprojection uses the UNJITTERED froxel centre: the jitter is a sampling offset, not motion,
        // and carrying it into the reprojection would chase the history around by a froxel each frame
        // and undo the supersampling it exists to provide.
        float3 stablePosition = ray.origin + ray.dir * fogSliceDistance(grid, float(slice) + 0.5);
        float rawLuminance = fogLuminance(raw);
        float4 history = float4(raw, rawLuminance * rawLuminance);
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

        float historyMean = fogLuminance(history.rgb);
        float historySecondMoment = history.a;
        float variance = max(0.0, historySecondMoment - historyMean * historyMean);

        // Converged froxels keep a long history and go clean; froxels whose radiance is actually
        // changing snap toward the new value instead of smearing, which is what preserves shaft shape
        // when a door opens or the sun moves. A fixed alpha can only pick one of those behaviours.
        float alpha = 1.0;
        if (reprojected) {
            float change = abs(rawLuminance - historyMean) / (historyMean + 1.0e-4);
            alpha = clamp(alphaFloor + change * changeSensitivity + sqrt(variance) * varianceWeight,
                    alphaFloor, 1.0);
        }

        float3 blended = lerp(history.rgb, raw, alpha);
        float blendedSecondMoment = lerp(historySecondMoment, rawLuminance * rawLuminance, alpha);
        blended = clamp(blended, float3(0.0), float3(HALF_MAX));
        fogHistory[index] = float4(blended, min(blendedSecondMoment, HALF_MAX));

        // ---- Depth integration.
        //
        // Analytic integration of the constant-medium step, the same closed form
        // sky.slang::integrateScatteredLuminance uses: exact over the step rather than a midpoint
        // sample, which is what lets a modest slice count run without visible banding.
        float sliceFar = fogSliceDistance(grid, float(slice) + 1.0);
        float stepLength = max(sliceFar - previousDistance, 0.0);
        previousDistance = sliceFar;
        if (extinction > FOG_DENSITY_EPSILON && stepLength > 0.0) {
            float stepTransmittance = exp(-extinction * stepLength);
            integrated += transmittance * (blended - blended * stepTransmittance) / extinction;
            transmittance *= stepTransmittance;
        }
        fogIntegrated[index] = float4(clamp(integrated, float3(0.0), float3(HALF_MAX)), transmittance);
    }
}
