// Froxel fog: apply the integrated volume to the reconstructed scene image.
//
// Runs at DISPLAY resolution, after DLSS-RR and before the exposure histogram.
//
//   * After RR because the volume carries its own temporal history and RR has no valid guide for
//     radiance that lives between the camera and the g-buffer surface — feeding fog in before
//     reconstruction is what makes it ghost and over-blur.
//   * Before exposure because the histogram meters this image; fog outside the meter makes
//     auto-exposure drift as the player walks into it.
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

    float2 uv = (float2(pixel) + 0.5) / float2(sceneDims);

    // This pixel's OWN depth, point sampled.
    //
    // It is tempting to take the nearest depth over a 2x2 footprint the way a bilateral upsample of a
    // half-resolution effect does, but that is wrong here and produces a dark halo along every
    // silhouette: a background pixel whose footprint clips a foreground object would look the volume up
    // at the FOREGROUND's short distance and so lose all the in-scatter that belongs between it and the
    // distant surface it actually sees. Fog depth is a per-pixel quantity; there is no neighbourhood to
    // choose from.
    uint2 depthDims;
    sceneDepth.GetDimensions(depthDims.x, depthDims.y);
    int2 depthTexel = clamp(int2(uv * float2(depthDims)), int2(0, 0), int2(depthDims) - 1);
    float depth = sceneDepth[depthTexel];

    // Reconstruct the view distance this pixel's surface sits at. Sky pixels have far depth (~0 under
    // reversed-Z), which lands past grid.far and therefore receives the full integrated column — the
    // correct haze attenuation of the sky and the celestial discs, with no special case.
    float distance;
    {
        float2 ndc = uv * 2.0 - 1.0;
        float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
        float3 nearP = nearH.xyz / nearH.w;
        if (depth > 0.0) {
            float4 surfaceH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, depth, 1.0));
            distance = surfaceH.w > 0.0 ? length(surfaceH.xyz / surfaceH.w - nearP) : grid.far;
        } else {
            distance = grid.far;
        }
    }

    // Pull the sample back half a slice before fetching.
    //
    // The volume stores each slice's prefix at its CENTRE, so a trilinear fetch at the surface's exact
    // w blends the slice in front of the surface with the one behind it. The slice behind is integrated
    // through space the surface occludes, so that blend leaks exterior fog through walls — standing in
    // a dark room with sun outside, up to a third of the fetched value came from beyond the wall. It is
    // most obvious in spectator/noclip, where the camera sits close to geometry and the near slices are
    // fractions of a block thick.
    //
    // Biasing by half a slice makes the fetch interpolate between the two slices IN FRONT of the
    // surface, so every froxel contributing to a pixel is one the pixel can actually see through.
    float surfaceSlice = fogDistanceSlice(grid, distance);
    float w = clamp((surfaceSlice - 0.5) / float(grid.dims.z), 0.0, 1.0);
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
