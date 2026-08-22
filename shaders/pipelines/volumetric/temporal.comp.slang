// Pass 3 — Temporal reprojection + neighborhood clamping + confidence.
//
// For each froxel:
//   * Take the stable world center and reproject it into the previous frame's
//     froxel grid using prevViewProj / prevCamOffset.
//   * Sample the history volume with nearest-neighbor (baseline; trilinear deferred).
//   * Compute neighborhood min/max of current (3x3x3) and clamp history to that box
//     to suppress fireflies/ghosts.
//   * Compute confidence based on edge distance, history validity, motion magnitude,
//     depth compatibility, and density compatibility.
//   * Do NOT reject history on the basis of stochastic shadow bits — only on
//     reprojected position / grid validity / density mismatch.
//   * Lerp with historyBaseWeight * confidence and update fogHistoryConfidence.
//
// READS:  fogVolume (post-spatial, current frame), fogHistory, fogHistoryConfidence, depthImage.
// WRITES: fogCurrent (temporal output; Java issues a fogCurrent->fogVolume copy after dispatch
//         so integrate sees the stabilized froxel grid). This out-of-place write avoids the
//         classic in-place feedback hazard where neighboring invocations in the same workgroup
//         could observe each other's partially-written results during neighborhood clamping.

import bindings;
import common;

[[vk::push_constant]] VolumetricPush pc;

void ComputeNeighborhoodMinMax(uint3 id, uint3 dims, out float3 minRgb, out float3 maxRgb)
{
    minRgb = float3( 1e30f,  1e30f,  1e30f);
    maxRgb = float3(-1e30f, -1e30f, -1e30f);
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
    {
        int3 n = int3(id) + int3(dx, dy, dz);
        if (n.x < 0 || n.y < 0 || n.z < 0
            || n.x >= (int)dims.x || n.y >= (int)dims.y || n.z >= (int)dims.z) continue;
        float4 v = fogVolume[uint3(n)];
        minRgb = min(minRgb, v.xyz);
        maxRgb = max(maxRgb, v.xyz);
    }
    float3 range = max(maxRgb - minRgb, float3(1.0e-4f, 1.0e-4f, 1.0e-4f));
    // Widen slightly to avoid clipping genuine variation.
    minRgb -= range * 0.25f;
    maxRgb += range * 0.25f;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= FroxelSize(pc))) return;
    uint3 dims = FroxelSize(pc);

    float4 current = fogVolume[id];

    // Sample scene depth at this froxel's XY location (scale froxel XY -> render-res px).
    uint2 depthDims;
    depthImage.GetDimensions(depthDims.x, depthDims.y);
    uint2 depthPx = (uint2)FroxelXyToDepthPx(id.xy, dims.xy, depthDims);
    float curNdcZ = depthImage.Load(int3((int2)depthPx, 0)).x;
    float curDu = ((float(depthPx.x) + 0.5f) / float(depthDims.x)) * 2.0f - 1.0f;
    float curDv = 1.0f - ((float(depthPx.y) + 0.5f) / float(depthDims.y)) * 2.0f;
    float curSceneDepth = LinearizeSample(pc, curNdcZ, curDu, curDv);
    float curFroxelDepth = FroxelCenterViewDepth(pc, id.z);
    bool  curFroxelInFront = (curSceneDepth <= pc.nearPlane) || (curFroxelDepth <= curSceneDepth * 1.05f + 0.5f);

    float3 worldPos = FroxelCenterWorldSpace(pc, id);
    float3 prevUvw = WorldToFroxelPrevious(pc, worldPos);

    float4 history = float4(0.0f, 0.0f, 0.0f, current.w);
    float  histConf = 0.0f;
    bool   histValid = false;

    if (pc.historyValid > 0.5f
        && prevUvw.x >= 0.0f && prevUvw.x <= 1.0f
        && prevUvw.y >= 0.0f && prevUvw.y <= 1.0f
        && prevUvw.z >= 0.0f && prevUvw.z <= 1.0f)
    {
        // Nearest tap into history volume. (Trilinear would require a Sampler3D binding;
        // for baseline, nearest + spatial filter + neighborhood clamping is sufficient.)
        float3 prevCoord = prevUvw * float3(dims) - 0.5f;
        int3 pi = int3(floor(prevCoord + 0.5f));
        if (all(pi >= int3(0,0,0)) && all(pi < int3(dims)))
        {
            history = fogHistory[uint3(pi)];
            histConf = fogHistoryConfidence[uint3(pi)];
            histValid = true;
        }
    }

    // ---- Confidence factors ----
    float edgeConf = 1.0f;
    {
        float2 edge = min(prevUvw.xy, 1.0f - prevUvw.xy);
        edgeConf = smoothstep(0.0f, 0.05f, min(edge.x, edge.y));
        float zEdge = min(prevUvw.z, 1.0f - prevUvw.z);
        edgeConf *= smoothstep(0.0f, 0.05f, zEdge);
    }

    // Motion magnitude (camera-space translation between frames); large motion
    // reduces confidence but doesn't zero it out.
    float3 motion = pc.camOffset - pc.prevCamOffset;
    float motionLen = length(motion);
    float motionConf = exp(-motionLen * 0.05f);

    // Current-froxel depth check: if the froxel is clearly behind opaque geometry,
    // its value is untrustworthy this frame (and historically too).
    float curDepthConf = curFroxelInFront ? 1.0f : 0.0f;

    // Density compatibility: large density jump between current and history implies
    // a disocclusion or content change.
    float densityCompat = 1.0f - smoothstep(0.05f, 0.3f,
        abs(history.w - current.w) / max(current.w, 0.01f));

    // Neighborhood clamp on history RGB to prevent ghost trails.
    float3 minRgb, maxRgb;
    ComputeNeighborhoodMinMax(id, dims, minRgb, maxRgb);
    float3 histRgb = clamp(history.xyz, minRgb, maxRgb);
    float  histA   = history.w;

    float baseWeight = clamp(pc.historyBaseWeight, 0.0f, 0.99f);
    float effectiveWeight = histValid
        ? baseWeight * histConf * edgeConf * motionConf * curDepthConf * densityCompat
        : 0.0f;
    effectiveWeight = clamp(effectiveWeight, 0.0f, 0.99f);

    float3 outRgb = lerp(current.xyz, histRgb, effectiveWeight);
    // Extinction (alpha) should respond faster than radiance — use half the weight.
    float  outA   = lerp(current.w,   histA,   effectiveWeight * 0.5f);

    // Update confidence: grow toward 1 on valid history, decay on invalid.
    float newConf;
    if (histValid && curDepthConf > 0.5f)
        newConf = min(histConf + (1.0f - histConf) * 0.1f, 1.0f);
    else
        newConf = max(histConf - 0.3f, 0.0f);

    // Write to fogCurrent (out-of-place relative to fogVolume reads). Java copies
    // fogCurrent -> fogVolume after this dispatch, and fogVolume -> fogHistory after
    // integrate, so ping-pong ordering is well-defined.
    fogCurrent[id] = float4(outRgb, outA);
    fogHistoryConfidence[id] = newConf;
}
