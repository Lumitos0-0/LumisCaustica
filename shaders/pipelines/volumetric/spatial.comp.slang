// Pass 2 — Edge-aware 3x3 XY spatial filter.
//
// Reads fogCurrent (post-inject), applies a 3x3 bilateral weighted blur in XY only.
// We do NOT blur across Z in the baseline (per spec). Edge-stopping weights are
// driven by (1) world-space distance between froxel centers, (2) extinction density
// difference, (3) whether neighbors straddle a depth discontinuity relative to scene
// geometry at that XY.
//
// Output is written into fogVolume for the temporal pass to accumulate.

import bindings;
import common;

[[vk::push_constant]] VolumetricPush pc;

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= FroxelSize(pc))) return;

    uint3 dims = FroxelSize(pc);
    uint2 depthDims;
    depthImage.GetDimensions(depthDims.x, depthDims.y);

    // gDepth is at render resolution; froxel XY is renderRes/divisor, so scale to depth UV.
    int2 depthPx = FroxelXyToDepthPx(id.xy, dims.xy, depthDims);
    float ndcZ = depthImage[depthPx];
    float du = ((float(depthPx.x) + 0.5f) / float(depthDims.x)) * 2.0f - 1.0f;
    float dv = 1.0f - ((float(depthPx.y) + 0.5f) / float(depthDims.y)) * 2.0f;
    float sceneDepth = LinearizeSample(pc, ndcZ, du, dv);

    float4 center = fogCurrent[id];
    float3 centerWorld = FroxelCenterWorldSpace(pc, id);
    float  centerDepth = FroxelCenterViewDepth(pc, id.z);
    bool   centerBehind = (sceneDepth > pc.nearPlane) && (centerDepth > sceneDepth);

    float3 sumRgb = center.xyz;
    float  sumA   = center.w;
    float  sumW   = 1.0f;

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue;
            int3 nId = int3(id) + int3(dx, dy, 0);
            if (nId.x < 0 || nId.y < 0 || nId.z < 0
                || nId.x >= (int)dims.x || nId.y >= (int)dims.y || nId.z >= (int)dims.z) continue;

            float4 n = fogCurrent[uint3(nId)];
            float3 nWorld = FroxelCenterWorldSpace(pc, uint3(nId));
            float  nDepth = FroxelCenterViewDepth(pc, (uint)nId.z);

            // (1) Spatial falloff based on world-space distance.
            float dist = distance(nWorld, centerWorld);
            float spatialW = exp(-dist * dist * 4.0f);

            // (2) Density similarity (don't blur across hard density edges).
            float densityDiff = abs(n.w - center.w);
            float densityW = exp(-densityDiff * densityDiff * 25.0f);

            // (3) Depth discontinuity: if one neighbor is behind scene geometry and
            // the other isn't, kill the weight. Prevents bleeding fog onto geometry.
            bool nBehind = (sceneDepth > pc.nearPlane) && (nDepth > sceneDepth);
            float depthW = (centerBehind == nBehind) ? 1.0f : 0.0f;

            float w = spatialW * densityW * depthW;
            sumRgb += n.xyz * w;
            sumA   += n.w   * w;
            sumW   += w;
        }
    }

    float4 result = float4(sumRgb / max(sumW, kEpsilon), sumA / max(sumW, kEpsilon));
    fogVolume[id] = result;
}
