// Fog temporal accumulation: FRESH + HISTORY -> RESOLVED, in 3D world space.
//
// Why 3D: each voxel reprojects its own world position into the previous frame. A camera
// rotation moves every voxel's history fetch coherently, so the volume stays converged
// through rotation, translation, and zoom - there is no screen-space smear to reject and
// no per-pixel reset. Slices follow the configured distribution (see fog.slang), so the fixed
// 256x128 XY grid resolves near detail and distant silhouettes alike.
//
// Slots (see bindings.slang): FRESH holds this frame's jittered bake. A/B ping-pong by
// frame parity: even frames blend FRESH + B -> A, odd frames blend FRESH + A -> B.
// Ages count accumulated frames per voxel (0..31) and drive the running-average weight.
// prevDepth (written by the apply pass) rejects voxels whose reprojection lands behind
// different occluders than last frame.
//
// The box clamp comes from FRESH neighbors, not history: history must never veto new
// signal (a torch turning on would be suppressed for seconds). Instead the 7-tap fresh
// cross bounds the history fetch, so stale voxels get pulled toward the new signal
// while noise averages out over frames.

import world_common;
import bindings;
import fog;

[[vk::push_constant]] PushAddr pcAddr;

// Running-average window: 31 frames of history (~0.5s at 60fps) cuts STBN noise to
// invisibility on smooth gradients; the cap bounds ghosting after lighting steps.
static const float ACCUM_MAX_AGE = 31.0;
// Occlusion validation: a voxel's reprojected depth is rejected when it moved behind
// the previous surface by more than 10% + 5cm. Fresh noise (cm-scale slice jitter on
// near voxels) never trips the absolute term; real disocclusions (meters) always do.
static const float ACCUM_OCCLUSION_REL = 0.10;
static const float ACCUM_OCCLUSION_ABS = 0.05;

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    WorldPush push = ConstPtr<WorldPush>(pcAddr.worldPushAddr)[0];
    uint3 dims;
    volARw.GetDimensions(dims.x, dims.y, dims.z);
    if (any(dispatchId >= dims)) {
        return;
    }
    const bool evenFrame = (push.frameIndex & 1u) == 0u;

    // Fresh sample + 7-tap cross (center plus 6 face neighbors): exact texels, no
    // filtering. The cross bounds the history fetch below (change detection).
    float3 dimsF = float3(dims);
    int3 hi = int3(dims - uint3(1, 1, 1));
    float3 centerUv = (float3(dispatchId) + 0.5) / dimsF;
    float4 fresh = freshVol.SampleLevel(centerUv, 0.0);
    float4 boxLo = fresh;
    float4 boxHi = fresh;
    for (int axis = 0; axis < 3; axis++) {
        for (int dir = -1; dir <= 1; dir += 2) {
            int3 off = int3(0, 0, 0);
            off[axis] = dir;
            float3 uvz = (float3(clamp(int3(dispatchId) + off, int3(0, 0, 0), hi)) + 0.5) / dimsF;
            float4 tap = freshVol.SampleLevel(uvz, 0.0);
            boxLo = min(boxLo, tap);
            boxHi = max(boxHi, tap);
        }
    }

    // Voxel ray: NDC of this column, slice range mapped to view depth (fog.slang).
    // Slices have no UV flip: the bake wrote fogUv unflipped, accumulate reads it back.
    float2 fogUv = (float2(dispatchId.xy) + 0.5) / dimsF.xy;
    float2 ndc = float2(fogUv.x * 2.0 - 1.0, fogUv.y * 2.0 - 1.0);
    float sliceCount = dimsF.z;
    float maxDist = max(push.fogParams0.z, FOG_NEAR + 1.0);
    float sliceMode = push.fogParams3.x;
    float sliceExp = max(push.fogParams3.y, 1.0);
    float viewDepth = fogSliceViewDepth(float(dispatchId.z) + 0.5, sliceCount, maxDist, sliceMode, sliceExp);

    // World position of the voxel center: clip plane at view depth, unprojected.
    float projA = push.curViewProj[2][2];
    float projB = push.curViewProj[2][3];
    float4 clipH = float4(ndc, clamp(projB / viewDepth - projA, 0.0, 1.0), 1.0);
    float4 worldH = mul(push.invViewProj, clipH);
    float3 worldPos = worldH.xyz / worldH.w;

    // Reproject into the previous frame: prev clip position, previous view depth,
    // previous fog UVZ. History volume indexing must mirror the bake exactly
    // (linear XY, slice Z), so both converge to the same voxel grid.
    float4 prevClip = mul(push.prevViewProj, float4(worldPos, 1.0));
    float4 prevViewH = mul(push.prevInvViewProj, prevClip);
    float prevViewDepth = -prevViewH.z / prevViewH.w;
    float3 prevNdc = prevClip.xyz / prevClip.w;
    float2 prevFogUv = float2(prevNdc.x * 0.5 + 0.5, prevNdc.y * 0.5 + 0.5);
    float3 prevUvz = float3(prevFogUv, fogViewDepthToSlice(prevViewDepth, sliceCount, maxDist, sliceMode, sliceExp) / sliceCount);
    bool inRange = prevClip.w > 1.0e-6 && prevViewH.w > 1.0e-6
        && all(prevUvz >= 0.0) && all(prevUvz <= 1.0);

    // Occlusion validation: compare the voxel's reprojected depth against the
    // previous frame's surface depth at the reprojected pixel. Behind the old
    // surface -> the voxel was occluded last frame, its history is garbage.
    // In front (negative delta) is always kept: fog legitimately sits in front.
    float prevSurfDepth = prevDepthRo.SampleLevel(prevFogUv, 0.0).x;
    float prevSurfView = projB / max(prevSurfDepth + projA, 1.0e-6);
    bool occluded = prevViewDepth > prevSurfView * (1.0 + ACCUM_OCCLUSION_REL) + ACCUM_OCCLUSION_ABS;

    bool useHistory = inRange && !occluded && push.fogParams2.w > 0.5;
    float4 hist = evenFrame ? volB.SampleLevel(prevUvz, 0.0) : volA.SampleLevel(prevUvz, 0.0);
    float histAge = evenFrame ? ageBRo.SampleLevel(prevUvz, 0.0).x : ageARo.SampleLevel(prevUvz, 0.0).x;

    // Running average: the box lets fresh signal through instantly while history
    // converges; the age cap (31) bounds the blend window. Invalid history resets
    // the voxel to fresh with age 0.
    float4 clampedHist = clamp(hist, boxLo, boxHi);
    float newAge = useHistory ? min(histAge + 1.0, ACCUM_MAX_AGE) : 0.0;
    float count = newAge + 1.0;
    float4 resolved = useHistory ? fresh * (1.0 / count) + clampedHist * (newAge / count) : fresh;

    if (evenFrame) {
        volARw[dispatchId] = resolved;
        ageARw[dispatchId] = newAge;
    } else {
        volBRw[dispatchId] = resolved;
        ageBRw[dispatchId] = newAge;
    }
}
