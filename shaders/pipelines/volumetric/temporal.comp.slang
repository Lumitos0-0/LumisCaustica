// Pass 3 — Temporal reprojection + neighborhood clamping + confidence.
//
// For each CURRENT-frame froxel (stable center P_rebasedWorld in the current terrain rebase frame):
//   * Reproject that physical point into the PREVIOUS froxel grid via WorldToFroxelPrevious.
//   * Sample the history volume with TRILINEAR filtering at that continuous
//     UVW coordinate (the reprojected location is not an integer cell).
//   * Validate the history sample. History use requires the full conjunction
//       historyValid = previousFrameExists        (host latch)
//                    && projectionIsCompatible     (host latch)
//                    && cameraStateIsCompatible    (host latch)
//                    && gridIsStable               (host latch)
//                    && prevUvwInsideGrid          (per-froxel, evaluated here)
//     The host folds the first four into pc.historyValid (see RtVolumetricFog);
//     prevUvw inside [0,1]^3 is tested below against an UNCLAMPED w so depths
//     the previous grid never covered reject instead of sampling a clamped
//     boundary slice. Within a valid sample, depth/medium/motion/edge and physical
//     reprojection-error confidence terms then modulate (never gate) the blend weight.
//     History is NEVER rejected just because the binary shadow sample
//     changed -- that noise is what temporal integration is meant to converge.
//   * Compute a 3x3x3 neighborhood of CURRENT-frame (spatially-filtered) RGB
//     min/max and clamp history into that box (AABB clamping, no cross/YCbCr).
//   * Blend with historyBaseWeight * effectiveWeight; update per-froxel
//     confidence (grows toward 1 on valid frames, decays on invalid).
//
// Read/write discipline: fogHistory, fogHistoryConfidence and fogVolume are
// READ-ONLY in this dispatch. Outputs go to fogCurrent (color/extinction) and
// fogConfidenceCurrent (updated confidence), out-of-place vs. every read, so no
// invocation ever observes another invocation's partially-written results.
// After this dispatch, Java copies fogCurrent -> fogVolume so integrate.comp
// reads the temporally-stabilized grid. After integrate reads fogVolume, Java
// copies fogVolume -> fogHistory and fogConfidenceCurrent -> fogHistoryConfidence
// for next frame.

import bindings;
import common;

[[vk::push_constant]] VolumetricPush pc;

// Manual trilinear tap into a RWTexture3D. coord is in continuous [0, size) space.
float4 SampleTrilinear(RWTexture3D<float4> tex, float3 coord, uint3 size)
{
    float3 c = coord - 0.5f;
    int3   i0 = int3(floor(c));
    float3 f  = frac(c);
    int3   i1 = i0 + 1;

    // Clamp to grid (history out of bounds is handled by the caller via weight=0,
    // but we still need to not read outside the image).
    int3 i0c = clamp(i0, int3(0,0,0), (int3)size - int3(1,1,1));
    int3 i1c = clamp(i1, int3(0,0,0), (int3)size - int3(1,1,1));

    float4 s000 = tex[uint3(i0c.x, i0c.y, i0c.z)];
    float4 s100 = tex[uint3(i1c.x, i0c.y, i0c.z)];
    float4 s010 = tex[uint3(i0c.x, i1c.y, i0c.z)];
    float4 s110 = tex[uint3(i1c.x, i1c.y, i0c.z)];
    float4 s001 = tex[uint3(i0c.x, i0c.y, i1c.z)];
    float4 s101 = tex[uint3(i1c.x, i0c.y, i1c.z)];
    float4 s011 = tex[uint3(i0c.x, i1c.y, i1c.z)];
    float4 s111 = tex[uint3(i1c.x, i1c.y, i1c.z)];

    float4 s00 = lerp(s000, s100, f.x);
    float4 s10 = lerp(s010, s110, f.x);
    float4 s01 = lerp(s001, s101, f.x);
    float4 s11 = lerp(s011, s111, f.x);
    float4 s0  = lerp(s00,  s10,  f.y);
    float4 s1  = lerp(s01,  s11,  f.y);
    return lerp(s0, s1, f.z);
}

float SampleConfidenceTrilinear(RWTexture3D<float> tex, float3 coord, uint3 size)
{
    float3 c = coord - 0.5f;
    int3   i0 = int3(floor(c));
    float3 f  = frac(c);
    int3   i1 = i0 + 1;
    int3 i0c = clamp(i0, int3(0,0,0), (int3)size - int3(1,1,1));
    int3 i1c = clamp(i1, int3(0,0,0), (int3)size - int3(1,1,1));

    float s000 = tex[uint3(i0c.x, i0c.y, i0c.z)];
    float s100 = tex[uint3(i1c.x, i0c.y, i0c.z)];
    float s010 = tex[uint3(i0c.x, i1c.y, i0c.z)];
    float s110 = tex[uint3(i1c.x, i1c.y, i0c.z)];
    float s001 = tex[uint3(i0c.x, i0c.y, i1c.z)];
    float s101 = tex[uint3(i1c.x, i0c.y, i1c.z)];
    float s011 = tex[uint3(i0c.x, i1c.y, i1c.z)];
    float s111 = tex[uint3(i1c.x, i1c.y, i1c.z)];

    float s00 = lerp(s000, s100, f.x);
    float s10 = lerp(s010, s110, f.x);
    float s01 = lerp(s001, s101, f.x);
    float s11 = lerp(s011, s111, f.x);
    float s0  = lerp(s00,  s10,  f.y);
    float s1  = lerp(s01,  s11,  f.y);
    return lerp(s0, s1, f.z);
}

// 3x3x3 neighborhood min/max of current-frame (spatially-filtered) lighting
// for AABB history clamping.
void NeighborMinMax(uint3 id, uint3 dims, out float3 minRgb, out float3 maxRgb, out float minA, out float maxA)
{
    minRgb = float3( 1e30f,  1e30f,  1e30f);
    maxRgb = float3(-1e30f, -1e30f, -1e30f);
    minA   =  1e30f;
    maxA   = -1e30f;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
    {
        int3 n = int3(id) + int3(dx, dy, dz);
        n = clamp(n, int3(0,0,0), (int3)dims - int3(1,1,1));
        float4 v = fogVolume[uint3(n)];
        minRgb = min(minRgb, v.xyz);
        maxRgb = max(maxRgb, v.xyz);
        minA   = min(minA,   v.w);
        maxA   = max(maxA,   v.w);
    }
    // Widen the box slightly so we don't clip legitimate small variation.
    float3 range = max(maxRgb - minRgb, float3(1.0e-4f, 1.0e-4f, 1.0e-4f));
    minRgb -= range * 0.25f;
    maxRgb += range * 0.25f;
    float arange = max(maxA - minA, 1.0e-4f);
    minA -= arange * 0.25f;
    maxA += arange * 0.25f;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= FroxelSize(pc))) return;
    uint3 dims = FroxelSize(pc);

    float4 current = fogVolume[id];

    // Debug/current-frame isolation: lets ray origin + light direction + TLAS visibility be inspected
    // without any temporal history contribution.
    if (pc.debugView == 13u)
    {
        fogCurrent[id] = current;
        fogConfidenceCurrent[id] = 0.0f;
        return;
    }

    float3 worldPos = FroxelCenterRebasedWorldSpace(pc, id);

    // Continuous history froxel coordinate in UVW (all axes unclamped; the bounds
    // test below decides whether the point was covered by the previous grid at all),
    // plus the same coordinate in texel space for the trilinear taps.
    float3 prevUvw = WorldToFroxelPrevious(pc, worldPos);
    float3 prevCoord = prevUvw * float3(dims);   // [0, dims]

    // History validity: the host-side latches (previousFrameExists &&
    // projectionIsCompatible && cameraStateIsCompatible && gridIsStable) arrive
    // folded into pc.historyValid; prevUvwInsideGrid is the per-froxel half.
    bool prevUvwInsideGrid = all(prevUvw >= float3(0.0f, 0.0f, 0.0f))
                          && all(prevUvw <= float3(1.0f, 1.0f, 1.0f));
    bool histValid = (pc.historyValid > 0.5f) && prevUvwInsideGrid;

    float4 history = float4(current.xyz, current.w);
    float  histConf = 0.0f;

    if (histValid)
    {
        // Sample history trilinearly (softens sub-cell motion).
        history    = SampleTrilinear(fogHistory, prevCoord, dims);
        histConf   = SampleConfidenceTrilinear(fogHistoryConfidence, prevCoord, dims);

        // Depth compatibility: reprojected history froxel's view depth vs.
        // current-frame scene depth at this froxel's XY. If the current
        // froxel is clearly behind opaque geometry but history says there
        // was bright sunlit fog, treat as disocclusion.
        uint2 depthDims;
        depthImage.GetDimensions(depthDims.x, depthDims.y);
        int2  dPx; float dDu, dDv;
        FroxelDepthSampleLoc(id.xy, dims.xy, depthDims, dPx, dDu, dDv);
        float curNdcZ = depthImage[dPx];
        float curSceneDepth = LinearizeSample(pc, curNdcZ, dDu, dDv);
        float curFroxelDepth = FroxelCenterViewDepth(pc, id.z);

        bool curFroxelInFront = (curSceneDepth <= pc.nearPlane)
                             || (curFroxelDepth <= curSceneDepth * 1.05f + 0.5f);

        // Medium (density) compatibility: if extinction has changed sharply
        // (e.g. we've crossed a fog boundary this frame), reduce weight.
        float densityCompat = 1.0f - smoothstep(0.1f, 0.5f,
            abs(history.w - current.w) / max(current.w, 0.01f));

        // Screen-edge distance fade (avoid pulling history from off-screen
        // via the trilinear edge taps).
        float2 edge = min(prevUvw.xy, float2(1.0f, 1.0f) - prevUvw.xy);
        float zEdge = min(prevUvw.z, 1.0f - prevUvw.z);
        float edgeConf = smoothstep(0.0f, 0.03f, min(min(edge.x, edge.y), zEdge));

        // Physical reprojection validation: the nearest previous froxel center should still be close
        // to the current physical region. This catches stale camera-relative origins, incompatible
        // previous matrices, and depth-mapping drift before history can smear them.
        float reprojectionError = FroxelHistoryReprojectionError(pc, worldPos, prevUvw);
        float allowedError = max(0.75f, curFroxelDepth * 0.035f);
        float reprojectionConf = 1.0f - smoothstep(allowedError, allowedError * 4.0f,
                                                   reprojectionError);

        // Motion magnitude (camera translation between frames).
        float3 motion = pc.camOffset - pc.prevCamOffset;
        float motionLen = length(motion);
        float motionConf = exp(-motionLen * 0.05f);

        // Neighborhood clamping on history RGB/A into CURRENT-frame local
        // min/max box -- suppresses ghost trails from old lighting.
        float3 minRgb, maxRgb; float minA, maxA;
        NeighborMinMax(id, dims, minRgb, maxRgb, minA, maxA);
        float3 histRgbClamped = clamp(history.xyz, minRgb, maxRgb);
        float  histAClamped   = clamp(history.w,   minA,   maxA);
        // Reduce confidence when clamping actually changed the value (sign of a firefly/reprojected ghost).
        float clampDiff = length(history.xyz - histRgbClamped) / max(length(histRgbClamped) + 0.1f, 0.001f);
        float clampConf = 1.0f - smoothstep(0.1f, 0.6f, clampDiff);

        // Depth-confidence: if the current froxel is behind scene geometry,
        // don't trust history either (we're in a disoccluded region).
        float depthConf = curFroxelInFront ? 1.0f : 0.0f;

        float effective = histConf * edgeConf * motionConf * depthConf * densityCompat
                        * reprojectionConf * clampConf;

        float baseW = clamp(pc.historyBaseWeight, 0.0f, 0.99f);
        float w = clamp(baseW * effective, 0.0f, 0.97f);

        float3 outRgb = lerp(current.xyz, histRgbClamped, w);
        // Alpha (extinction) converges faster than lighting -- halve weight.
        float  outA   = lerp(current.w,   histAClamped,   w * 0.5f);

        fogCurrent[id] = float4(outRgb, outA);

        // Update confidence: grow toward 1 on valid frames, decay on invalid.
        float newConf;
        if (curFroxelInFront)
            newConf = min(histConf + (1.0f - histConf) * 0.15f, 1.0f);
        else
            newConf = max(histConf - 0.35f, 0.0f);
        fogConfidenceCurrent[id] = newConf;
    }
    else
    {
        fogCurrent[id] = float4(current.xyz, current.w);
        fogConfidenceCurrent[id] = 0.0f;
    }
}
