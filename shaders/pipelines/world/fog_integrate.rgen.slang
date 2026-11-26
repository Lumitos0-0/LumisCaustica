import world_common;
import bindings;
import froxel_fog;
import fog_transport;

[shader("raygeneration")]
void main() {
    WorldPush frame = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 column = DispatchRaysIndex().xy;
    uint2 grid = uint2(frame.fogGrid.xy);
    uint2 size;
    outImage.GetDimensions(size.x, size.y);
    float2 uv = (float2(column) + 0.5) / float2(grid);
    uint2 pixel = fogDepthPixel(uv, size, frame.jitter);
    ConstPtr<float> depths = ConstPtr<float>(fogDepthAddress(frame));
    DevicePtr<float4> columns = DevicePtr<float4>(frame.fogBufferAddr);
    columns[column.y * grid.x + column.x] = integrateFogRay(frame, uv, depths[pixel.y * size.x + pixel.x]);
}
