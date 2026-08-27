// Stage 4: display-resolution post-reconstruction composition. Java binds the DLSS-RR output (or the
// fallback upscaled image) as froxelCompositeOutput, so this pass runs before exposure/bloom while never
// feeding temporally reconstructed surface guides with volumetric radiance.

import world_common;
import world_core;
import bindings;
import froxel;

[shader("raygeneration")]
void main() {
    uint2 pixel = DispatchRaysIndex().xy;
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    if (!froxelEnabled()) {
        return;
    }

    uint width;
    uint height;
    froxelCompositeOutput.GetDimensions(width, height);
    if (pixel.x >= width || pixel.y >= height) {
        return;
    }
    float2 uv = (float2(pixel) + 0.5) / float2(width, height);
    float4 fog = froxelAtUv(uv);
    float4 scene = froxelCompositeOutput[int2(pixel)];
    froxelCompositeOutput[int2(pixel)] = float4(
            min(max(scene.rgb, float3(0.0)) * fog.a
                    + fog.rgb * worldPush.preExposure, float3(HALF_MAX)), scene.a);
}
