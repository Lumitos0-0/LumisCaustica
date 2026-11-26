import world_common;
import volume_grid;
import bindings;

[shader("raygeneration")]
void main() {
    WorldPush f = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 col = DispatchRaysIndex().xy;
    uint2 size;
    outImage.GetDimensions(size.x, size.y);
    float2 tile = float2(size) / float2(f.fogGrid.xy);
    float2 center = float2(col) + 0.5;
    // The full support of the resolve tent includes neighbouring columns and the
    // two jittered depth pixels on either side of a display sample.
    int2 lo = max(int2(0), int2(floor((center - 1.0) * tile - f.jitter - 1.5)));
    int2 hi = min(int2(size) - 1, int2(ceil((center + 1.0) * tile - f.jitter + 0.5)));
    ConstPtr<float> depths = ConstPtr<float>(fogDepthAddress(f));
    float limit = 0.0;
    for (int y = lo.y; y <= hi.y && limit < f.fogOptions.y; ++y) {
        for (int x = lo.x; x <= hi.x; ++x) limit = max(limit, depths[uint(y) * size.x + uint(x)]);
    }
    DevicePtr<float> bounds = DevicePtr<float>(boundsAddress(f));
    bounds[col.y * uint(f.fogGrid.x) + col.x] = limit;
}
