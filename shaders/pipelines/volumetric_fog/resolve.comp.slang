// Froxel volumetric fog lighting resolve.
// Filters the raw per-froxel directional-illumination field in volume space and blends in a reprojected
// history sample. Doing this once per froxel is dramatically cheaper than asking the full-resolution
// view-ray integration pass to repair coarse lighting with many screen-space taps at every march step.

import world_common;
import volumetric_fog_common;

[[vk::push_constant]] FogIntegrationPush pc;

// Bindings - set 0
[[vk::binding(0, 0)]] Sampler3D currentVolume;
[[vk::binding(1, 0)]] Sampler3D historyVolume;
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture3D<float4> resolvedVolume;

static const float CURRENT_EXTINCTION_SIGMA = 12.0;
static const float CURRENT_LIGHTING_SIGMA = 3.0;
static const float HISTORY_EXTINCTION_SIGMA = 10.0;
static const float HISTORY_LIGHTING_SIGMA = 4.0;

float fogLuma(float3 v) {
    return dot(max(v, float3(0.0)), ACESCG_LUMA);
}

float relativeLightingDelta(float3 a, float3 b) {
    float la = fogLuma(a);
    float lb = fogLuma(b);
    return abs(la - lb) / max(max(la, lb), 1.0e-3);
}

float3 primaryRayDirFromPush(float2 ndc, WorldPush worldPush) {
    float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    return normalize(farP - nearP);
}

float3 cellUv(int3 cell, int3 dim) {
    return (float3(cell) + float3(0.5, 0.5, 0.5))
            / float3(float(dim.x), float(dim.y), float(dim.z));
}

void accumulateCurrentTap(int3 cell, int3 dim, float4 center, float baseWeight,
                          inout float4 accum, inout float weightSum,
                          inout float4 neighborhoodMin, inout float4 neighborhoodMax) {
    int3 clamped = clamp(cell, int3(0, 0, 0), dim - int3(1, 1, 1));
    float4 tap = currentVolume.SampleLevel(cellUv(clamped, dim), 0.0);
    neighborhoodMin = min(neighborhoodMin, tap);
    neighborhoodMax = max(neighborhoodMax, tap);

    float extinctionWeight = exp(-abs(tap.a - center.a) * CURRENT_EXTINCTION_SIGMA);
    float lightingWeight = exp(-relativeLightingDelta(tap.rgb, center.rgb) * CURRENT_LIGHTING_SIGMA);
    float weight = baseWeight * extinctionWeight * lightingWeight;
    accum += tap * weight;
    weightSum += weight;
}

bool reprojectHistory(float2 curUv, float curDepth, int3 dim, WorldPush worldPush,
                      float nearP, float farP, out float2 prevUv,
                      out float prevLinear01, out float reprojPixels) {
    float2 ndc = curUv * 2.0 - 1.0;
    float3 currentCamRel = primaryRayDirFromPush(ndc, worldPush) * curDepth;
    float3 prevCamRel = currentCamRel + worldPush.camDelta;
    float prevDepth = length(prevCamRel);
    if (!(prevDepth > nearP && prevDepth <= farP)) {
        prevUv = float2(0.0, 0.0);
        prevLinear01 = 0.0;
        reprojPixels = 0.0;
        return false;
    }

    float4 prevClip = mul(worldPush.prevViewProj, float4(prevCamRel, 1.0));
    if (prevClip.w <= 0.0) {
        prevUv = float2(0.0, 0.0);
        prevLinear01 = 0.0;
        reprojPixels = 0.0;
        return false;
    }

    float2 prevNdc = prevClip.xy / prevClip.w;
    prevUv = prevNdc * 0.5 + 0.5;
    float2 uvMin = float2(0.5, 0.5) / float2(float(dim.x), float(dim.y));
    float2 uvMax = float2(1.0, 1.0) - uvMin;
    if (prevUv.x < uvMin.x || prevUv.y < uvMin.y || prevUv.x > uvMax.x || prevUv.y > uvMax.y) {
        prevLinear01 = 0.0;
        reprojPixels = 0.0;
        return false;
    }

    prevLinear01 = froxelUnitFromDepth(prevDepth, nearP, farP);
    reprojPixels = length((prevUv - curUv) * float2(float(dim.x), float(dim.y)));
    return true;
}

[shader("compute")]
[numthreads(8, 8, 4)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    int3 dim = int3(pc.froxelDim.x, pc.froxelDim.y, pc.froxelDepth);
    if (int(dispatchId.x) >= dim.x || int(dispatchId.y) >= dim.y || int(dispatchId.z) >= dim.z) return;

    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];

    int3 cell = int3(dispatchId.xyz);
    float3 uvw = cellUv(cell, dim);
    float2 uv = uvw.xy;

    float nearP = max(pc.nearPlane, 0.1);
    float farP = min(max(pc.farPlane, nearP + 1.0), pc.maxDistance);
    float depth = froxelDepthFromUnit(uvw.z, nearP, farP);

    float4 center = currentVolume.SampleLevel(uvw, 0.0);
    float4 accum = float4(0.0, 0.0, 0.0, 0.0);
    float weightSum = 0.0;
    float4 neighborhoodMin = center;
    float4 neighborhoodMax = center;

    // Smooth the lighting field mainly across XY where the camera-frustum grid is most visible, with only
    // a mild Z influence so distant slabs blend without washing out near silhouettes. Include diagonals so
    // shafts through small openings do not keep a plus-shaped reconstruction footprint.
    accumulateCurrentTap(cell, dim, center, 0.36, accum, weightSum, neighborhoodMin, neighborhoodMax);
    accumulateCurrentTap(cell + int3(-1, 0, 0), dim, center, 0.09, accum, weightSum, neighborhoodMin, neighborhoodMax);
    accumulateCurrentTap(cell + int3(1, 0, 0), dim, center, 0.09, accum, weightSum, neighborhoodMin, neighborhoodMax);
    accumulateCurrentTap(cell + int3(0, -1, 0), dim, center, 0.09, accum, weightSum, neighborhoodMin, neighborhoodMax);
    accumulateCurrentTap(cell + int3(0, 1, 0), dim, center, 0.09, accum, weightSum, neighborhoodMin, neighborhoodMax);
    accumulateCurrentTap(cell + int3(-1, -1, 0), dim, center, 0.055, accum, weightSum, neighborhoodMin, neighborhoodMax);
    accumulateCurrentTap(cell + int3(-1, 1, 0), dim, center, 0.055, accum, weightSum, neighborhoodMin, neighborhoodMax);
    accumulateCurrentTap(cell + int3(1, -1, 0), dim, center, 0.055, accum, weightSum, neighborhoodMin, neighborhoodMax);
    accumulateCurrentTap(cell + int3(1, 1, 0), dim, center, 0.055, accum, weightSum, neighborhoodMin, neighborhoodMax);
    accumulateCurrentTap(cell + int3(0, 0, -1), dim, center, 0.04, accum, weightSum, neighborhoodMin, neighborhoodMax);
    accumulateCurrentTap(cell + int3(0, 0, 1), dim, center, 0.04, accum, weightSum, neighborhoodMin, neighborhoodMax);

    float4 resolved = accum / max(weightSum, 1.0e-5);
    // The visible artifact is mainly the coarse directional-light field, not the base density/extinction.
    // Keep extinction anchored to the unfiltered current froxel so silhouettes and fog-stop depth remain
    // crisp, then let history smooth it only lightly.
    resolved.a = center.a;

    if (pc.temporalEnabled != 0) {
        float2 prevUv;
        float prevLinear01;
        float reprojPixels;
        if (reprojectHistory(uv, depth, dim, worldPush, nearP, farP, prevUv, prevLinear01, reprojPixels)) {
            float4 history = historyVolume.SampleLevel(float3(prevUv, prevLinear01), 0.0);
            history = clamp(history, neighborhoodMin, neighborhoodMax);

            float extinctionTrust = exp(-abs(history.a - center.a) * HISTORY_EXTINCTION_SIGMA);
            float lightingTrust = exp(-relativeLightingDelta(history.rgb, resolved.rgb) * HISTORY_LIGHTING_SIGMA);
            float motionTrust = 1.0 / (1.0 + reprojPixels * 0.25);
            float distanceHistory = lerp(0.62, 0.90, saturate(depth * (1.0 / 32.0)));
            float historyWeight = distanceHistory * extinctionTrust * lightingTrust * motionTrust;
            resolved.rgb = lerp(resolved.rgb, history.rgb, historyWeight);
            resolved.a = lerp(center.a, history.a, min(historyWeight, 0.35));
        }
    }

    resolvedVolume[dispatchId] = resolved;
}
