// Post-reconstruction volumetric composition at display resolution.
//
// Fog is deliberately NOT part of the path-traced radiance DLSS-RR consumes. RR reprojects pixels with
// first-surface guides, but near-camera in-scattering has almost no parallax: composited pre-RR, it gets
// dragged with the background surface's motion vector (ghost shafts on camera moves) and its stochastic
// estimate is eaten by neighborhood clamping (pulsing). This raygen instead applies the froxel grid's
// integrated {in-scattering, transmittance} on top of the denoised display image, in the same
// pre-exposed space the display mapper divides back out:
//     composed = scene * T + inScatter * preExposure          (identical to scene-linear (L*T + I))
//
// The first-interface distance comes from Pass A's render-resolution volumeDepth, nearest-upsampled:
// the fog endpoint is a low-frequency optical quantity, so render-res quantization of its slice
// position is the residual budgeted artifact — far below the per-cell froxel granularity and far
// cheaper than tracing a display-res pass just for depth.
//
// The lookup is jitter-free: the compose input is the reconstructed (jitter-resolved) image, and the
// froxel field itself is built unjittered.

import world_common;
import world_core;
import bindings;
import froxel;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    int2 pixel = int2(DispatchRaysIndex().xy);
    uint2 dimensions = DispatchRaysDimensions().xy;

    // The composed image is the single scene source for every downstream stage, so a disabled medium
    // must still copy through rather than leaving foggedScene stale.
    float4 scene = rtSceneInput[pixel];
    if (!froxelEnabled()) {
        foggedScene[pixel] = scene;
        return;
    }

    uint depthWidth;
    uint depthHeight;
    volumeDepth.GetDimensions(depthWidth, depthHeight);
    float2 uv = (float2(pixel) + 0.5) / float2(dimensions);
    int2 renderPixel = min(int2(uv * float2(depthWidth, depthHeight)),
            int2(int(depthWidth) - 1, int(depthHeight) - 1));
    float distance = clamp(volumeDepth[renderPixel], 0.0, worldPush.fogDistance.x);

    float3 composed = scene.rgb;
    if (distance > 0.0) {
        float4 fog = froxelIntegratedAtUv(uv, distance);
        float transmittance = clamp(fog.a, 0.0, 1.0);
        composed = scene.rgb * transmittance
                + max(fog.rgb, float3(0.0)) * worldPush.preExposure;
    }
    foggedScene[pixel] = float4(min(composed, float3(HALF_MAX)), scene.a);
}
