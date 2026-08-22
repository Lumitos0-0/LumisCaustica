// Pass 4 — Exact front-to-back volumetric integration + composite.
//
// For each screen pixel at RENDER resolution:
//   * reconstruct the camera-relative view ray through this pixel,
//   * read scene depth (gDepth) and clip integration to it,
//   * walk froxel Z slices front-to-back, sampling fogVolume with a manual
//     2x2 XY bilinear tap at each slice (fogVolume is at froxel resolution,
//     render/divisor), nearest in Z,
//   * apply the exact exponential transmittance/in-scattering integration
//     (T = exp(-sigma*ds), integ = S*(1-T)/sigma),
//   * read the pre-fog ray-traced scene colour from outputImage,
//   * write final = accumL + accumT * scene back into outputImage.
//
// This places fog at render resolution BEFORE DLSS-RR/upscale so RR denoises
// fog and surfaces together.

import bindings;
import common;

[[vk::push_constant]] VolumetricPush pc;

// Load a froxel with nearest-integer clamping.
float4 LoadFroxelNearest(uint3 coord)
{
    uint3 c = clamp(coord, uint3(0,0,0), FroxelSize(pc) - uint3(1,1,1));
    return fogVolume[c];
}

// Bilinear XY tap at a continuous froxel XY position, nearest in Z.
// fxy is the continuous froxel XY coordinate (0..froxelSize.xy), in froxel space
// (not UV space). z is an integer slice index.
float4 SampleFroxelBilinear(float2 fxy, uint z)
{
    float2 base = fxy - 0.5f;
    int2 i00 = int2(floor(base));
    float2 t = base - float2(i00);
    uint2 u00 = uint2(max(i00, int2(0,0)));
    uint2 u11 = uint2(min(i00 + int2(1,1), int2(FroxelSize(pc).xy) - int2(1,1)));
    uint2 u01 = uint2(u00.x, u11.y);
    uint2 u10 = uint2(u11.x, u00.y);

    float4 s00 = fogVolume[uint3(u00, z)];
    float4 s10 = fogVolume[uint3(u10, z)];
    float4 s01 = fogVolume[uint3(u01, z)];
    float4 s11 = fogVolume[uint3(u11, z)];

    float4 s0 = lerp(s00, s10, t.x);
    float4 s1 = lerp(s01, s11, t.x);
    return lerp(s0, s1, t.y);
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

    // Read gDepth at this render-res pixel (depth buffer is at same render resolution).
    int2 depthPx = int2(min(id.x, depthDims.x - 1u), min(id.y, depthDims.y - 1u));
    float ndcZ = depthImage.Load(int3(depthPx, 0)).x;

    // Reconstruct the camera-relative ray direction through this pixel AND
    // get the NDC (u,v) we need for depth linearization (avoids recomputing).
    float u, v;
    float3 rayDir = PixelViewRay(pc, id, dims, u, v);

    float sceneDepth = pc.farPlane * 2.0f;
    FroxelMatrices m = GetMatrices(pc);
    if (ndcZ > 0.0001f)
    {
        float3 p = UnprojectNdc(m.invViewProj, u, v, ndcZ);
        sceneDepth = max(-p.z, 0.0f);
    }

    // Convert view-depth range to ray-parameter range along rayDir.
    // segStart_t / segEnd_t are in units of blocks ALONG THE RAY, measured from
    // the camera (not along -Z). ratio = 1 / max(-rayDir.z, eps).
    float zRatio = 1.0f / max(-rayDir.z, kEpsilon);
    float tStart = pc.nearPlane * zRatio;
    float tEnd   = min(pc.farPlane * zRatio, sceneDepth * zRatio);
    tEnd = max(tEnd, tStart + kEpsilon);
    if (tEnd <= tStart)
    {
        // No integration span (e.g. geometry closer than near plane); leave pixel untouched.
        return;
    }

    // Continuous froxel XY position for this render-res pixel. Since froxel XY is a uniform
    // downscale of render res by divisor, the fraction uv = (id+0.5)/dims maps directly to
    // fxy = uv * froxelSize.xy.
    float2 fxy = (float2(id) + 0.5f) / float2(dims) * float2(FroxelSize(pc).xy);

    // Determine the Z-slice range that covers [tStart*(-dir.z), tEnd*(-dir.z)] in view-depth.
    float startViewDepth = tStart * (-rayDir.z);
    float endViewDepth   = tEnd   * (-rayDir.z);
    float startCoord = SliceCoordFromViewDepth(startViewDepth, pc.nearPlane, pc.farPlane, FroxelSize(pc).z)
            * float(FroxelSize(pc).z);
    float endCoord   = SliceCoordFromViewDepth(endViewDepth,   pc.nearPlane, pc.farPlane, FroxelSize(pc).z)
            * float(FroxelSize(pc).z);
    uint zStart = uint(max(0, (int)floor(startCoord)));
    uint zEnd   = uint(min((int)FroxelSize(pc).z - 1, (int)ceil(endCoord)));

    float3 accumL = float3(0.0f, 0.0f, 0.0f);
    float3 accumT = float3(1.0f, 1.0f, 1.0f);

    for (uint z = zStart; z <= zEnd; ++z)
    {
        float zNear, zFar;
        FroxelSliceDepthRange(z, pc.nearPlane, pc.farPlane, FroxelSize(pc).z, zNear, zFar);
        // Convert slice Z (view-depth) range to ray-parameter range.
        float sliceTStart = zNear * zRatio;
        float sliceTEnd   = zFar  * zRatio;

        float segStart = max(tStart, sliceTStart);
        float segEnd   = min(tEnd,   sliceTEnd);
        if (segEnd <= segStart) continue;
        // ds is geometric path length in blocks; rayDir is normalized so t is block distance.
        float ds = segEnd - segStart;

        float4 froxel = SampleFroxelBilinear(fxy, z);
        float3 S = froxel.xyz;
        float  sigmaLuma = froxel.w;

        // Skip zero-density slices quickly.
        if (sigmaLuma < 1.0e-6f) continue;

        float3 sigma_t = SigmaTFromLuma(sigmaLuma);
        float3 T_slice = Transmittance(sigma_t, ds);
        float3 integ = IntegratedSource(S, sigma_t, ds);
        accumL += accumT * integ;
        accumT *= T_slice;

        // Early termination when fully occluded.
        if (max(max(accumT.x, accumT.y), accumT.z) < 1.0e-3f) break;
    }

    float4 scene = outputImage[id];
    float3 finalColor = accumL + accumT * scene.xyz;
    outputImage[id] = float4(finalColor, scene.a);
}
