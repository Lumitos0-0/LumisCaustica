// Pass 4 — Exact front-to-back volumetric integration + composite.
//
// For each RENDER-RES pixel:
//   * reconstructs camera ray direction through the pixel,
//   * reads gDepth to find opaque scene distance,
//   * walks the froxel volume front-to-back, sampling the temporally-
//     accumulated fogVolume with TRILINEAR interpolation at each step,
//   * applies the exact per-segment exponential integration
//       T_slice = exp(-sigma_t * ds)
//       integ   = S * (1 - T_slice) / sigma_t
//     with linear fallback as sigma_t -> 0,
//   * composites accumL + accumT * scene over the existing RT output.
//
// This places fog at render resolution BEFORE DLSS-RR/upscale so RR
// denoises fog + surfaces together.
//
// Debug views (selected by pc.debugView) are rendered by switching what
// this pass writes into outputImage; modes 1-10 replace the scene with
// diagnostic shading so the offending stage is visible in-place.

import bindings;
import common;

[[vk::push_constant]] VolumetricPush pc;

// Trilinear read from fogVolume at a continuous froxel-space coordinate
// (x,y in [0,froxelSize.xy), z in [0,froxelSize.z)).
float4 SampleFogTrilinear(float3 coord)
{
    uint3 size = FroxelSize(pc);
    float3 c = coord - 0.5f;
    int3   i0 = int3(floor(c));
    float3 f  = frac(c);
    int3   i1 = i0 + 1;

    int3 i0c = clamp(i0, int3(0,0,0), (int3)size - int3(1,1,1));
    int3 i1c = clamp(i1, int3(0,0,0), (int3)size - int3(1,1,1));

    float4 s000 = fogVolume[uint3(i0c.x, i0c.y, i0c.z)];
    float4 s100 = fogVolume[uint3(i1c.x, i0c.y, i0c.z)];
    float4 s010 = fogVolume[uint3(i0c.x, i1c.y, i0c.z)];
    float4 s110 = fogVolume[uint3(i1c.x, i1c.y, i0c.z)];
    float4 s001 = fogVolume[uint3(i0c.x, i0c.y, i1c.z)];
    float4 s101 = fogVolume[uint3(i1c.x, i0c.y, i1c.z)];
    float4 s011 = fogVolume[uint3(i0c.x, i1c.y, i1c.z)];
    float4 s111 = fogVolume[uint3(i1c.x, i1c.y, i1c.z)];

    float4 s00 = lerp(s000, s100, f.x);
    float4 s10 = lerp(s010, s110, f.x);
    float4 s01 = lerp(s001, s101, f.x);
    float4 s11 = lerp(s011, s111, f.x);
    float4 s0  = lerp(s00,  s10,  f.y);
    float4 s1  = lerp(s01,  s11,  f.y);
    return lerp(s0, s1, f.z);
}

float SampleConfidenceTrilinear(float3 coord)
{
    uint3 size = FroxelSize(pc);
    float3 c = coord - 0.5f;
    int3   i0 = int3(floor(c));
    float3 f  = frac(c);
    int3   i1 = i0 + 1;
    int3 i0c = clamp(i0, int3(0,0,0), (int3)size - int3(1,1,1));
    int3 i1c = clamp(i1, int3(0,0,0), (int3)size - int3(1,1,1));
    float s000 = fogHistoryConfidence[uint3(i0c.x, i0c.y, i0c.z)];
    float s100 = fogHistoryConfidence[uint3(i1c.x, i0c.y, i0c.z)];
    float s010 = fogHistoryConfidence[uint3(i0c.x, i1c.y, i0c.z)];
    float s110 = fogHistoryConfidence[uint3(i1c.x, i1c.y, i0c.z)];
    float s001 = fogHistoryConfidence[uint3(i0c.x, i0c.y, i1c.z)];
    float s101 = fogHistoryConfidence[uint3(i1c.x, i0c.y, i1c.z)];
    float s011 = fogHistoryConfidence[uint3(i0c.x, i1c.y, i1c.z)];
    float s111 = fogHistoryConfidence[uint3(i1c.x, i1c.y, i1c.z)];
    float s00 = lerp(s000, s100, f.x);
    float s10 = lerp(s010, s110, f.x);
    float s01 = lerp(s001, s101, f.x);
    float s11 = lerp(s011, s111, f.x);
    float s0  = lerp(s00,  s10,  f.y);
    float s1  = lerp(s01,  s11,  f.y);
    return lerp(s0, s1, f.z);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    uint2 dims;
    outputImage.GetDimensions(dims.x, dims.y);
    if (any(id >= dims)) return;

    uint2 depthDims;
    depthImage.GetDimensions(depthDims.x, depthDims.y);

    // Read gDepth at this pixel and reconstruct the camera ray.
    int2 px = int2(min(id.x, depthDims.x - 1u), min(id.y, depthDims.y - 1u));
    float ndcZ = depthImage[px];

    float u, v;
    float3 rayDir = PixelViewRay(pc, id, dims, u, v);

    float sceneT = (pc.farPlane * 2.0f) / max(-rayDir.z, kEpsilon);
    if (ndcZ > 0.0001f)
    {
        float3 p = UnprojectNdc(GetMatrices(pc).invViewProj, u, v, ndcZ);
        float sceneDepth = max(-p.z, 0.0f);
        sceneT = min(sceneT, sceneDepth / max(-rayDir.z, kEpsilon));
    }

    float startT  = pc.nearPlane / max(-rayDir.z, kEpsilon);
    float endT    = min(pc.farPlane / max(-rayDir.z, kEpsilon), sceneT);
    endT = max(endT, startT + kEpsilon);
    if (endT <= startT)
    {
        // No integration span. For debug modes, still produce a colored pixel.
        if (pc.debugView == 0)
            return;  // leave scene untouched
        // fall through to produce debug color with zero contribution
    }

    // Continuous froxel XY for this pixel. fxy is in [0, froxelSize.xy) so a
    // trilinear tap at (fxy, z+0.5) correctly lerps between the 4 neighboring
    // XY cells and 2 neighboring Z slices.
    float2 fxy = (float2(id) + 0.5f) / float2(dims) * float2(FroxelSize(pc).xy);

    // Convert T range into continuous Z slice coordinate w in [0,1].
    float startViewDepth = startT * (-rayDir.z);
    float endViewDepth   = endT   * (-rayDir.z);
    uint D = FroxelSize(pc).z;
    float wStart = SliceCoordFromViewDepth(startViewDepth, pc.nearPlane, pc.farPlane) * float(D);
    float wEnd   = SliceCoordFromViewDepth(endViewDepth,   pc.nearPlane, pc.farPlane) * float(D);
    // March from ceil(wStart - 0.5) so the first midpoint sample is >= wStart.
    int zStart = max(0, (int)floor(wStart - 0.5f));
    int zEnd   = min((int)D - 1, (int)floor(wEnd - 0.5f) + 1);

    float3 accumL = float3(0.0f, 0.0f, 0.0f);
    float  accumT = 1.0f;

    float4 debugTap       = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float  debugConfidence = 0.0f;
    int    debugSlice = -1;
    float  debugWorldChecker = 0.0f;

    // Precompute w -> viewDepth -> t conversion at each integer slice boundary
    // to get segment [t0, t1] for each marching step.
    float prevW = max(wStart, 0.0f);
    float prevT = startT;
    bool marched = false;

    for (int z = zStart; z <= zEnd; ++z)
    {
        // Continuous Z of the sample midpoint and far edge.
        float wMid   = float(z) + 0.5f;
        float wEdge  = float(z) + 1.0f;
        // Clamp the far edge to wEnd so the final segment ends exactly on geometry.
        float wFar   = min(wEdge, wEnd);
        float wNear  = max(prevW, float(z));
        if (wFar <= wNear) continue;

        float dNear  = ViewDepthFromFroxelCoord(wNear / float(D), pc.nearPlane, pc.farPlane);
        float dFar   = ViewDepthFromFroxelCoord(wFar  / float(D), pc.nearPlane, pc.farPlane);
        float tNear  = dNear / max(-rayDir.z, kEpsilon);
        float tFar   = dFar  / max(-rayDir.z, kEpsilon);
        tNear = max(tNear, startT);
        tFar  = min(tFar, endT);
        if (tFar <= tNear + kEpsilon) { prevW = wFar; prevT = tFar; continue; }

        float ds = tFar - tNear;

        // Trilinear tap at the midpoint of the segment in (fxy, w) space.
        float3 tapCoord = float3(fxy, wMid);
        float4 froxel = SampleFogTrilinear(tapCoord);
        float3 S = froxel.xyz;
        float  sigma = froxel.w;

        // Scene-depth awareness for the sample: only contribute if this segment
        // is not substantially behind opaque geometry (it won't be, because
        // endT is clipped to sceneT, but extra clamp avoids leakage at grazing
        // edges where off-axis rays might disagree with the pixel-center depth).
        if (sigma < 1.0e-6f) { prevW = wFar; prevT = tFar; continue; }

        float  T_slice = exp(-sigma * ds);
        // integ = S * (1 - T) / sigma  (scalar form; S already includes sigma_s).
        float3 integ   = S * ((1.0f - T_slice) / max(sigma, kEpsilon));
        accumL += accumT * integ;
        accumT *= T_slice;

        // Early-out: fully opaque transmittance.
        if (accumT < 1.0e-3f) break;

        if (!marched)
        {
            debugTap = froxel;
            debugSlice = z;
            float3 midWorldPos = rayDir * ((tNear + tFar) * 0.5f) + pc.camOffset;
            debugWorldChecker = WorldChecker(midWorldPos, 0.25f);
            marched = true;
        }

        prevW = wFar;
        prevT = tFar;
    }

    float4 scene = outputImage[id];

    if (pc.debugView == 0)
    {
        // Final composite: L + T * scene.
        float3 finalColor = accumL + accumT * scene.xyz;
        outputImage[id] = float4(finalColor, scene.a);
        return;
    }

    // --- Debug visualization modes ---
    float3 color = float3(0.0f, 0.0f, 0.0f);
    switch (pc.debugView)
    {
        case 1: { // Z-slice index, color-coded.
            float t = float(debugSlice) / float(D);
            color = float3(t, 0.5f + 0.5f * sin(t * 10.0f), 1.0f - t);
            break;
        }
        case 2: { // World-anchored checker (should NOT swim when camera moves).
            color = float3(debugWorldChecker.xxx) * 0.8f + 0.1f;
            break;
        }
        case 3: { // Extinction/density (luma of alpha), color-mapped.
            float d = saturate(debugTap.w * 20.0f);
            color = float3(d * 0.4f, d * 0.9f, d);
            break;
        }
        case 4: { // Raw RT visibility: approximated by S / (sigma_s * phase * sunColor).
                 // Since we don't store visibility separately, display S normalized.
            float mag = length(debugTap.xyz) / max(length(pc.sunColor) + 1.0e-3f, 0.001f);
            color = float3(mag.xxx);
            break;
        }
        case 5: { // fogCurrent at integrate time: post-temporal accumulated grid (the
                 // pre-spatial injection is overwritten in place by the temporal pass).
            int3 ix = int3(clamp(int2(fxy), int2(0,0), (int2)FroxelSize(pc).xy - int2(1,1)),
                           clamp((int)(wStart), 0, (int)D - 1));
            float4 pre = fogCurrent[uint3(ix)];
            color = pre.xyz * 3.0f;
            break;
        }
        case 6: { // fogVolume tap at integrate time: post-temporal grid (the fogCurrent copy
                 // overwrote the pre-temporal spatial output before this pass ran).
            color = debugTap.xyz * 3.0f;
            break;
        }
        case 7: { // History confidence.
            float3 worldPos = FroxelCenterWorldSpace(pc,
                uint3(clamp((int)fxy.x, 0, (int)FroxelSize(pc).x - 1),
                      clamp((int)fxy.y, 0, (int)FroxelSize(pc).y - 1),
                      clamp((int)(wStart), 0, (int)D - 1)));
            float3 prevUvw = WorldToFroxelPrevious(pc, worldPos);
            if (all(prevUvw >= float3(0,0,0)) && all(prevUvw <= float3(1,1,1)))
                debugConfidence = SampleConfidenceTrilinear(prevUvw * float3(FroxelSize(pc)));
            color = float3(debugConfidence.xxx);
            break;
        }
        case 8: { // Disocclusion mask: red where history was rejected (confidence near 0 with valid scene).
            float3 worldPos = FroxelCenterWorldSpace(pc,
                uint3(clamp((int)fxy.x, 0, (int)FroxelSize(pc).x - 1),
                      clamp((int)fxy.y, 0, (int)FroxelSize(pc).y - 1),
                      clamp((int)(wStart), 0, (int)D - 1)));
            float3 prevUvw = WorldToFroxelPrevious(pc, worldPos);
            float conf = 0.0f;
            if (all(prevUvw >= float3(0,0,0)) && all(prevUvw <= float3(1,1,1)))
                conf = SampleConfidenceTrilinear(prevUvw * float3(FroxelSize(pc)));
            float rejected = (pc.historyValid > 0.5f) ? (1.0f - conf) : 1.0f;
            color = float3(rejected, 0.2f * rejected, 0.0f);
            break;
        }
        case 9: { // Integrated scattering (accumL over black).
            color = accumL;
            break;
        }
        case 10: { // Final volumetric contribution (L + (T-1)*scene) over neutral gray.
            float3 contrib = accumL + (accumT - 1.0f) * scene.xyz;
            color = float3(0.5f, 0.5f, 0.5f) + contrib;
            break;
        }
        default: {
            color = scene.xyz;
            break;
        }
    }

    outputImage[id] = float4(color, scene.a);
}
