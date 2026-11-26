import world_common;
import bindings;
import froxel_fog;
import fog_transport;

[shader("raygeneration")]
void main() {
    WorldPush frame = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 pixel = DispatchRaysIndex().xy;
    float2 uv = (float2(pixel) + 0.5) / float2(DispatchRaysDimensions().xy);
    uint2 size;
    outImage.GetDimensions(size.x, size.y);
    uint2 depthPixel = fogDepthPixel(uv, size, frame.jitter);
    uint2 grid = uint2(frame.fogGrid.xy);
    ConstPtr<float> depths = ConstPtr<float>(fogDepthAddress(frame));
    ConstPtr<float4> columns = ConstPtr<float4>(frame.fogBufferAddr);
    float range = frame.fogOptions.y;
    float depth = min(range, depths[depthPixel.y * size.x + depthPixel.x]);
    float2 p = uv * float2(grid) - 0.5;
    int2 base = int2(floor(p));
    float2 w = frac(p);
    float4 sum = float4(0.0);
    float total = 0.0;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            uint2 col = uint2(clamp(base + int2(x,y), int2(0), int2(grid) - 1));
            uint2 src = fogDepthPixel((float2(col) + 0.5) / float2(grid), size, frame.jitter);
            float z = min(range, depths[src.y * size.x + src.x]);
            float error = abs(z - depth) / max(0.125, 0.02 * depth);
            float weight = (x == 0 ? 1.0 - w.x : w.x) * (y == 0 ? 1.0 - w.y : w.y)
                         * exp2(-error * error);
            sum += weight * columns[col.y * grid.x + col.x];
            total += weight;
        }
    }
    float4 fog = sum / max(total, 1e-6);
    // Thin geometry can cover a pixel while all four nearby column rays miss it.
    if (total < 0.25) {
        float4 exact = integrateFogRay(frame, uv, depth);
        fog = lerp(exact, fog, smoothstep(0.0, 0.25, total));
    }
    float4 scene = fogTarget[int2(pixel)];
    fogTarget[int2(pixel)] = float4(scene.rgb * exp(-fog.a) + fog.rgb * frame.preExposure, scene.a);
}
