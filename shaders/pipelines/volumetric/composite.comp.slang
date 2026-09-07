// Froxel fog: apply the integrated volume to the traced scene image.
//
// Runs at RENDER resolution, BEFORE DLSS-RR.
//
// This placement is the whole reason the fog lines up with geometry. The volume lookup needs the
// distance to the surface each pixel sees, and the only depth that exists is gDepth, written at render
// resolution by the primary raygen. Compositing at display resolution after reconstruction means
// sampling that render-res depth for display-res pixels: at a silhouette the two disagree, so a display
// pixel showing background can read foreground depth, its distance collapses, and its fog integral
// collapses with it. The result is a fringe of missing fog along every geometry edge, staircased at
// render-res granularity — the classic "blocks eating the fog" artefact. No upsampling filter can
// repair it, because the depth being sampled simply does not describe the pixel being shaded.
//
// Here, depth and colour are the same resolution, the same pixels, produced by the same pass, so they
// cannot disagree. DLSS-RR then upscales colour and fog together and they stay locked.
//
// Feeding fog to RR is safe because the volume arrives already denoised by its own temporal filter (see
// integrate.comp), so RR is resampling a converged image rather than denoising a noisy one. The sky
// takes the same route: it has no real albedo either and passes through RR on a synthetic guide.
//
// Reads and writes sceneImage in place. Safe: each thread touches exactly one texel and reads no
// neighbour's colour.
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
    int2 pixel = int2(dispatchThreadId.xy);
    uint2 sceneDims;
    sceneImage.GetDimensions(sceneDims.x, sceneDims.y);
    if (!grid.enabled || any(uint2(pixel) >= sceneDims)) {
        return;
    }

    // One depth texel per pixel, exactly. sceneDepth and sceneImage are the same size here, so this is
    // this pixel's own surface — no footprint, no neighbourhood, no filtering decision to get wrong.
    float depth = sceneDepth[pixel];

    // The pixel's own ray direction uses the camera's jittered NDC, exactly as primary.rgen traced it,
    // so the distance below describes the surface this pixel actually sees.
    float2 uv = (float2(pixel) + 0.5) / float2(sceneDims);
    float2 jitterUv = worldPush.jitter / float2(sceneDims);
    float2 ndc = (uv + jitterUv) * 2.0 - 1.0;

    // Radial distance to the surface, in the same metric the grid slices by. Sky pixels carry far depth
    // (~0 under reversed-Z) and land past grid.far, so they receive the fully integrated column — the
    // correct haze attenuation of the sky and the celestial discs, with no special case.
    float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float distance = grid.far;
    if (depth > 0.0) {
        float4 surfaceH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, depth, 1.0));
        if (surfaceH.w > 0.0) {
            distance = length(surfaceH.xyz / surfaceH.w - nearP);
        }
    }

    // Pull the sample back half a slice in W.
    //
    // Each slice stores its prefix at its CENTRE, so a fetch at the surface's exact w would interpolate
    // between the slice in front of the surface and the one behind it — and the slice behind is
    // integrated through space the surface occludes, which leaks exterior light through walls. Biasing
    // back half a slice keeps both taps in front of the surface.
    float surfaceSlice = fogDistanceSlice(grid, distance);
    float w = clamp((surfaceSlice - 0.5) / float(grid.dims.z), 0.0, 1.0);

    // Sample the volume at the UNJITTERED uv — deliberately, and the reasoning is worth stating because
    // the sign is easy to get backwards.
    //
    // The correct rule is to match RAY DIRECTIONS, not coordinates. Froxel column c was traced along the
    // ray through screen uv (c + 0.5)/dims + j, and this pixel's own primary ray went through uv + j
    // with the SAME j, since the grid now inherits the camera's jitter. Setting the two equal gives
    // (c + 0.5)/dims = uv: the offsets cancel, so the sampler coordinate is simply uv.
    //
    // (The reprojection in integrate.comp is the asymmetric case: it starts from a world point projected
    // to a previous SCREEN uv, with no matching ray to cancel against, so it must subtract the previous
    // frame's offset explicitly.)
    //
    // What this buys: the fog lattice now shifts each frame in the same direction the camera jitter
    // shifts geometry, and both land in the image DLSS-RR reconstructs. RR already knows that offset, so
    // it resolves fog and geometry together and the froxel grid is antialiased by the same mechanism
    // that antialiases edges — which a private per-froxel offset could never achieve, because the
    // composite read on a fixed lattice and so never saw it.
    //
    // Plain trilinear on top. Correct because depth is exact per pixel at render resolution; a
    // depth-weighted variant would be actively harmful, since rejecting neighbours at a silhouette
    // degenerates the fetch to a point sample of one column and stamps hard froxel-sized blocks along
    // exactly the edges it was meant to protect.
    float4 fog = fogIntegratedSampler.SampleLevel(float3(uv, w), 0.0);

    float4 scene = sceneImage[pixel];
    // PRE-EXPOSURE, load-bearing: sceneImage carries worldPush.preExposure (the display pass divides it
    // back out), while fogIntegrated is absolute scene radiance. Without this multiply the two stop
    // cancelling and every fog contribution is wrong by the frame's exposure scalar — silently, because
    // the error tracks brightness and so looks like a tuning problem rather than a bug.
    float3 inscatter = fog.rgb * worldPush.preExposure;
    float3 result = scene.rgb * fog.a + inscatter;
    sceneImage[pixel] = float4(clamp(result, float3(0.0), float3(HALF_MAX)), scene.a);
}
