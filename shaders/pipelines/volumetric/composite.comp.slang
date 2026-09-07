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

    // Nearest-depth sample over the 2x2 render-resolution footprint. gDepth is render-res under RR and
    // reversed-Z, so the NEAREST surface is the LARGEST value; picking it makes the foreground win at a
    // silhouette, which stops fog computed for the distant background from bleeding over a near object.
    // That bleed reads as softness even when the volume itself is sharp, so this is part of the
    // anti-oversoftening story, not just a correctness detail.
    uint2 depthDims;
    sceneDepth.GetDimensions(depthDims.x, depthDims.y);
    float2 depthCoord = uv * float2(depthDims) - 0.5;
    int2 depthBase = int2(floor(depthCoord));
    float depth = 0.0;
    for (int tap = 0; tap < 4; ++tap) {
        int2 texel = clamp(depthBase + int2(tap & 1, (tap >> 1) & 1),
                int2(0, 0), int2(depthDims) - 1);
        depth = max(depth, sceneDepth[texel]);
    }

    // Reconstruct the view distance this pixel's surface sits at. Sky pixels have far depth (~0 under
    // reversed-Z), which lands past grid.far and therefore receives the full integrated column — the
    // correct haze attenuation of the sky and the celestial discs, with no special case.
    float3 origin;
    float distance;
    {
        float2 ndc = uv * 2.0 - 1.0;
        float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
        float3 nearP = nearH.xyz / nearH.w;
        origin = nearP;
        if (depth > 0.0) {
            float4 surfaceH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, depth, 1.0));
            distance = surfaceH.w > 0.0 ? length(surfaceH.xyz / surfaceH.w - nearP) : grid.far;
        } else {
            distance = grid.far;
        }
    }

    float w = fogDistanceW(grid, distance);
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
