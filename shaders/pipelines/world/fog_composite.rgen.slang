import world_common;
import bindings;
import froxel_fog;

[shader("raygeneration")]
void main() {
    WorldPush frame = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 size = DispatchRaysDimensions().xy;
    ConstPtr<float> depth = ConstPtr<float>(fogDepthAddress(frame));
    float firstHit = depth[pixel.y * size.x + pixel.x];
    // Evaluate the unjittered froxel grid along the jittered surface sample from Pass A.
    float2 uv = (float2(pixel) + 0.5 + frame.jitter) / float2(size);
    float4 fog = fogAtPixel(frame, uv, firstHit);
    float4 scene = outImage[int2(pixel)];
    outImage[int2(pixel)] = float4(scene.rgb * exp(-fog.a) + fog.rgb * frame.preExposure, scene.a);
}
