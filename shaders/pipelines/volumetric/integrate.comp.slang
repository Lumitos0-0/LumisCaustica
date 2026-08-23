// Pass 4 — Exact front-to-back volumetric integration + composite.
//
// For each RENDER-RES pixel:
//   * reconstructs camera ray direction through the pixel,
//   * reads gDepth to find opaque scene distance,
//   * walks the froxel volume front-to-back, sampling temporally accumulated source radiance
//     and deterministic sigma_s/sigma_a coefficients with TRILINEAR interpolation,
//   * applies the exact per-segment Beer-Lambert integration per RGB channel
//       T_slice = exp(-(sigma_s + sigma_a) * ds)
//       integ   = S * (1 - T_slice) / (sigma_s + sigma_a)
//     with linear fallback as sigma_t -> 0,
//   * composites volumeRadiance + surfaceRadiance * volumeTransmittance over the existing RT output.
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

// Trilinear read from a froxel 3D image at a continuous froxel-space coordinate
// (x,y in [0,froxelSize.xy), z in [0,froxelSize.z)).
float4 SampleTextureTrilinear(RWTexture3D<float4> tex, float3 coord)
{
    uint3 size = FroxelSize(pc);
    float3 c = coord - 0.5f;
    int3   i0 = int3(floor(c));
    float3 f  = frac(c);
    int3   i1 = i0 + 1;

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

float4 SampleFogTrilinear(float3 coord)
{
    return SampleTextureTrilinear(fogVolume, coord);
}

float4 SampleMediumScatteringTrilinear(float3 coord)
{
    return SampleTextureTrilinear(fogMediumScattering, coord);
}

float4 SampleMediumAbsorptionTrilinear(float3 coord)
{
    return SampleTextureTrilinear(fogMediumAbsorption, coord);
}

float3 SegmentFactor(float3 sigmaT, float ds)
{
    float3 T = exp(-max(sigmaT, float3(0.0f, 0.0f, 0.0f)) * ds);
    return float3(
        sigmaT.x > 1.0e-6f ? (1.0f - T.x) / sigmaT.x : ds,
        sigmaT.y > 1.0e-6f ? (1.0f - T.y) / sigmaT.y : ds,
        sigmaT.z > 1.0e-6f ? (1.0f - T.z) / sigmaT.z : ds
    );
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
    float s000 = fogConfidenceCurrent[uint3(i0c.x, i0c.y, i0c.z)];
    float s100 = fogConfidenceCurrent[uint3(i1c.x, i0c.y, i0c.z)];
    float s010 = fogConfidenceCurrent[uint3(i0c.x, i1c.y, i0c.z)];
    float s110 = fogConfidenceCurrent[uint3(i1c.x, i1c.y, i0c.z)];
    float s001 = fogConfidenceCurrent[uint3(i0c.x, i0c.y, i1c.z)];
    float s101 = fogConfidenceCurrent[uint3(i1c.x, i0c.y, i1c.z)];
    float s011 = fogConfidenceCurrent[uint3(i0c.x, i1c.y, i1c.z)];
    float s111 = fogConfidenceCurrent[uint3(i1c.x, i1c.y, i1c.z)];
    float s00 = lerp(s000, s100, f.x);
    float s10 = lerp(s010, s110, f.x);
    float s01 = lerp(s001, s101, f.x);
    float s11 = lerp(s011, s111, f.x);
    float s0  = lerp(s00,  s10,  f.y);
    float s1  = lerp(s01,  s11,  f.y);
    return lerp(s0, s1, f.z);
}

// Shared reprojection probe for the diagnostic views: the previous-frame froxel UVW of the
// froxel under this pixel, going through the exact production mapping
// (FroxelCenterRebasedWorldSpace -> WorldToFroxelPrevious) so the visualization reflects the
// temporal pass's math. inside = the coordinate lies within the previous grid's [0,1]^3.
float3 DebugPrevUvw(float2 fxy, float wStart, uint D, out bool inside, out float3 worldPos)
{
    worldPos = FroxelCenterRebasedWorldSpace(pc,
        uint3(clamp((int)fxy.x, 0, (int)FroxelSize(pc).x - 1),
              clamp((int)fxy.y, 0, (int)FroxelSize(pc).y - 1),
              clamp((int)(wStart), 0, (int)D - 1)));
    float3 prevUvw = WorldToFroxelPrevious(pc, worldPos);
    inside = all(prevUvw >= float3(0,0,0)) && all(prevUvw <= float3(1,1,1));
    return prevUvw;
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
    float3 rayOriginView;
    float3 rayDir = PixelViewRayWithOrigin(pc, id, dims, rayOriginView, u, v);

    // t <-> view-depth scale for this ray: view depth is measured along the CAMERA forward
    // axis while rayDir is in rebased-world axes, so the factor is the angle cosine
    // dot(rayDir, cameraForward). Ray parameter t starts at the actual unprojected near-plane
    // point, matching the primary renderer and the froxel reconstruction helpers.
    float3 camFwd = CameraForwardWorld(GetMatrices(pc).invViewProj);
    float viewScale = max(dot(rayDir, camFwd), kEpsilon);

    float sceneDepthLimit = pc.farPlane * 2.0f;
    if (ndcZ > 0.0001f)
    {
        float3 p = UnprojectNdc(GetMatrices(pc).invViewProj, u, v, ndcZ);
        sceneDepthLimit = min(sceneDepthLimit, max(dot(p, camFwd), 0.0f));
    }

    float startViewDepth = pc.nearPlane;
    float endViewDepth   = min(pc.farPlane, sceneDepthLimit);
    float startT = RayTFromViewDepthOnRay(startViewDepth, rayOriginView, rayDir, camFwd);
    float endT   = RayTFromViewDepthOnRay(endViewDepth,   rayOriginView, rayDir, camFwd);
    if (endT <= startT + kEpsilon)
    {
        // No integration span. For debug modes, still produce a colored pixel.
        if (pc.debugView == 0)
            return;  // leave scene untouched
        endT = startT + kEpsilon;
        endViewDepth = startViewDepth;
    }

    // Continuous froxel XY for this pixel. fxy is in [0, froxelSize.xy) so a
    // trilinear tap at (fxy, z+0.5) correctly lerps between the 4 neighboring
    // XY cells and 2 neighboring Z slices.
    float2 fxy = (float2(id) + 0.5f) / float2(dims) * float2(FroxelSize(pc).xy);

    // Convert the view-depth range into continuous Z slice coordinate w in [0,1].
    uint D = FroxelSize(pc).z;
    float wStart = SliceCoordFromViewDepth(startViewDepth, pc.nearPlane, pc.farPlane) * float(D);
    float wEnd   = SliceCoordFromViewDepth(endViewDepth,   pc.nearPlane, pc.farPlane) * float(D);
    // March from ceil(wStart - 0.5) so the first midpoint sample is >= wStart.
    int zStart = max(0, (int)floor(wStart - 0.5f));
    int zEnd   = min((int)D - 1, (int)floor(wEnd - 0.5f) + 1);

    float3 accumL = float3(0.0f, 0.0f, 0.0f);
    // Transmittance used for in-scattered radiance between a sample and the camera.
    float3 accumT = float3(1.0f, 1.0f, 1.0f);
    // Transmittance applied to the already-rendered surface. When the camera is underwater, the primary
    // path already applied water absorption to the surface ray; airParams.w gates that absorption out
    // of surfaceT so water is not double-applied while water scattering still attenuates surfaces.
    float3 surfaceT = float3(1.0f, 1.0f, 1.0f);

    float4 debugTap       = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 debugScattering = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 debugAbsorption = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float  debugConfidence = 0.0f;
    int    debugSlice = -1;
    float  debugWorldChecker = 0.0f;

    // Precompute w -> viewDepth -> t conversion at each integer slice boundary
    // to get segment [t0, t1] for each marching step.
    float prevW = max(wStart, 0.0f);
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
        float tNear  = RayTFromViewDepthOnRay(dNear, rayOriginView, rayDir, camFwd);
        float tFar   = RayTFromViewDepthOnRay(dFar,  rayOriginView, rayDir, camFwd);
        tNear = max(tNear, startT);
        tFar  = min(tFar, endT);
        if (tFar <= tNear + kEpsilon) { prevW = wFar; continue; }

        float ds = tFar - tNear;

        // Trilinear taps at the midpoint of the segment in (fxy, w) space.
        float3 tapCoord = float3(fxy, wMid);
        float4 froxel = SampleFogTrilinear(tapCoord);
        float4 scatterTap = SampleMediumScatteringTrilinear(tapCoord);
        float4 absorbTap  = SampleMediumAbsorptionTrilinear(tapCoord);
        float3 S = max(froxel.xyz, float3(0.0f, 0.0f, 0.0f));
        float3 sigmaS = max(scatterTap.xyz, float3(0.0f, 0.0f, 0.0f));
        float3 sigmaA = max(absorbTap.xyz,  float3(0.0f, 0.0f, 0.0f));
        float3 sigmaT = sigmaS + sigmaA;

        // Skip truly empty segments. Source can be zero in shadow, but extinction still attenuates the
        // surface behind the volume, so only skip if both extinction and source are negligible.
        if (max(max(sigmaT.x, sigmaT.y), sigmaT.z) < 1.0e-7f
            && max(max(S.x, S.y), S.z) < 1.0e-7f)
        {
            prevW = wFar; continue;
        }

        float3 T_slice = exp(-sigmaT * ds);
        float3 integ = S * SegmentFactor(sigmaT, ds);
        accumL += accumT * integ;
        accumT *= T_slice;

        FroxelMatrices frame = GetMatrices(pc);
        float3 surfaceSigmaT = sigmaS + sigmaA * clamp(frame.airParams.w, 0.0f, 1.0f);
        surfaceT *= exp(-surfaceSigmaT * ds);

        // Early-out: fully opaque transmittance for both volume radiance and surface contribution.
        if (max(max(accumT.x, accumT.y), accumT.z) < 1.0e-3f
            && max(max(surfaceT.x, surfaceT.y), surfaceT.z) < 1.0e-3f) break;

        if (!marched)
        {
            debugTap = froxel;
            debugScattering = scatterTap;
            debugAbsorption = absorbTap;
            debugSlice = z;
            float3 midWorldPos = rayOriginView + rayDir * ((tNear + tFar) * 0.5f) + pc.camOffset;
            debugWorldChecker = WorldChecker(midWorldPos + GetMatrices(pc).noiseOrigin.xyz, 0.25f);
            marched = true;
        }

        prevW = wFar;
    }

    float4 scene = outputImage[id];

    if (pc.debugView == 0 || pc.debugView == 13u)
    {
        // Final composite: volumeRadiance + surfaceRadiance * volumeTransmittance.
        // Mode 13 reaches this same composite after temporal.comp intentionally bypassed history.
        float3 finalColor = accumL + surfaceT * scene.xyz;
        outputImage[id] = float4(finalColor, scene.a);
        return;
    }

    // --- Debug visualization modes ---
    float3 color = float3(0.0f, 0.0f, 0.0f);
    switch (pc.debugView)
    {
        case 1: { // DEBUG_VOLUME_DENSITY
            float d = saturate(debugScattering.w * 20.0f);
            color = float3(d * 0.4f, d * 0.9f, d);
            break;
        }
        case 2: { // DEBUG_VOLUME_SCATTERING
            color = saturate(debugScattering.xyz * 30.0f);
            break;
        }
        case 3: { // DEBUG_VOLUME_ABSORPTION
            color = saturate(debugAbsorption.xyz * 30.0f);
            break;
        }
        case 4: { // DEBUG_VOLUME_TRANSMITTANCE
            color = saturate(surfaceT);
            break;
        }
        case 5: { // DEBUG_VOLUME_LIGHT_VISIBILITY
            color = float3(saturate(debugAbsorption.w).xxx);
            break;
        }
        case 6: { // DEBUG_VOLUME_HISTORY: temporally reconstructed source radiance.
            color = debugTap.xyz * 3.0f;
            break;
        }
        case 7: { // DEBUG_VOLUME_HISTORY_WEIGHT: current confidence after temporal resolve.
            debugConfidence = SampleConfidenceTrilinear(float3(fxy, clamp(wStart + 0.5f, 0.0f, float(D) - 1.0f)));
            color = float3(debugConfidence.xxx);
            break;
        }
        case 8: { // DEBUG_VOLUME_FROXEL_SLICE
            float t = float(max(debugSlice, 0)) / max(float(D - 1u), 1.0f);
            color = float3(t, 0.5f + 0.5f * sin(t * 10.0f), 1.0f - t);
            break;
        }
        case 9: { // DEBUG_VOLUME_REPROJECTION_ERROR: physical error in blocks.
            bool inside;
            float3 debugWorldPos;
            float3 prevUvw = DebugPrevUvw(fxy, wStart, D, inside, debugWorldPos);
            float err = inside ? FroxelHistoryReprojectionError(pc, debugWorldPos, prevUvw) : 1.0e30f;
            float vis = saturate(err * 4.0f);
            color = inside ? float3(vis, saturate(1.0f - vis), 0.0f)
                           : float3(1.0f, 0.0f, 1.0f);
            break;
        }
        case 10: { // World-anchored checker (should NOT swim when camera moves).
            color = float3(debugWorldChecker.xxx) * 0.8f + 0.1f;
            break;
        }
        case 11: { // Integrated in-scattered radiance over black.
            color = accumL;
            break;
        }
        case 12: { // Final volumetric contribution (L + (T-1)*scene) over neutral gray.
            float3 contrib = accumL + (surfaceT - float3(1.0f, 1.0f, 1.0f)) * scene.xyz;
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
