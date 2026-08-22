// Pass 2 — Edge-aware 3x3 XY spatial filter.
//
// Reads the raw injected froxel data (fogCurrent) and applies a 3x3 bilateral
// blur in XY only. No Z blur (trilinear reconstruction in integrate handles
// Z continuity). Edge stopping is driven by:
//   * world-space distance between froxel centers,
//   * density (sigma_t) difference,
//   * per-neighbor scene-depth validity: a neighbor froxel that lies
//     substantially behind the opaque geometry at its own XY is rejected
//     rather than blurring in invalid/empty-space values.
//
// Output is written into fogVolume for temporal accumulation.
// We sample depthImage at EACH neighbor's XY (not just the center) so the
// filter does not assume one depth value applies across the whole kernel.

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

    // --- Center sample.
    float4 center = fogCurrent[id];
    float3 centerWorld = FroxelCenterWorldSpace(pc, id);
    float  centerDepth = FroxelCenterViewDepth(pc, id.z);

    // Scene depth at the CENTER pixel.
    int2  cPx; float cDu, cDv;
    FroxelDepthSampleLoc(id.xy, dims.xy, depthDims, cPx, cDu, cDv);
    float cNdcZ = depthImage[cPx];
    float cSceneDepth = LinearizeSample(pc, cNdcZ, cDu, cDv);
    bool  centerBehind = (cSceneDepth > pc.nearPlane) && (centerDepth > cSceneDepth * 1.02f + 0.3f);

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

            // Early out: a zero-sigma neighbor (no medium) contributes nothing.
            if (n.w < 1.0e-6f) continue;

            float3 nWorld = FroxelCenterWorldSpace(pc, uint3(nId));
            float  nDepth = FroxelCenterViewDepth(pc, (uint)nId.z);

            // Per-neighbor scene depth sample (sampled at the neighbor's XY).
            int2  nPx; float nDu, nDv;
            FroxelDepthSampleLoc(uint2(nId.x, nId.y), dims.xy, depthDims, nPx, nDu, nDv);
            float nNdcZ = depthImage[nPx];
            float nSceneDepth = LinearizeSample(pc, nNdcZ, nDu, nDv);
            bool  nBehind = (nSceneDepth > pc.nearPlane) && (nDepth > nSceneDepth * 1.02f + 0.3f);

            // --- Weight terms.
            // (1) Spatial falloff by world distance (smooth, not harsh).
            float dist = distance(nWorld, centerWorld);
            float spatialW = exp(-dist * dist * 4.0f);

            // (2) Density similarity.
            float dd = abs(n.w - center.w);
            float densityW = exp(-dd * dd * 36.0f);

            // (3) Depth-validity: reject samples that straddle an occluder. If both
            // the center and neighbor are in front of their respective geometry
            // OR both are behind, allow the blur; if one is valid and the other
            // isn't, suppress weight to prevent bleeding light through walls.
            float depthW = (centerBehind == nBehind) ? 1.0f : 0.05f;

            // (4) Radiance similarity: reject large firefly differences softly.
            float lc = dot(center.xyz, float3(0.25f, 0.65f, 0.10f));
            float ln = dot(n.xyz,      float3(0.25f, 0.65f, 0.10f));
            float rdiff = abs(ln - lc) / max(max(lc, ln) + 0.5f, 0.01f);
            float radW = exp(-rdiff * rdiff * 4.0f);

            float w = spatialW * densityW * depthW * radW;
            sumRgb += n.xyz * w;
            sumA   += n.w   * w;
            sumW   += w;
        }
    }

    float4 result = float4(sumRgb / max(sumW, kEpsilon), sumA / max(sumW, kEpsilon));
    fogVolume[id] = result;
}
