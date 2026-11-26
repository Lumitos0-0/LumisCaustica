import world_common;
import bindings;
import froxel_fog;
import fog_transport;

[shader("raygeneration")]
void main() {
    WorldPush frame = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 column = DispatchRaysIndex().xy;
    float2 grid = float2(frame.fogGrid.xy);
    uint2 size;
    outImage.GetDimensions(size.x, size.y);
    // Cover the tent kernel's entire footprint plus both jittered depth taps at either edge.
    float2 center = float2(column) + 0.5;
    int2 first = clamp(int2(floor((center - 1.0) * float2(size) / grid - frame.jitter - 0.5)) - 1,
                       int2(0), int2(size) - 1);
    int2 last = clamp(int2(ceil((center + 1.0) * float2(size) / grid - frame.jitter - 0.5)) + 1,
                      int2(0), int2(size) - 1);
    ConstPtr<float> depth = ConstPtr<float>(fogDepthAddress(frame));
    float farthest = 0.0;
    for (int y = first.y; y <= last.y && farthest < frame.fogOptions.y; ++y) {
        for (int x = first.x; x <= last.x; ++x) {
            farthest = max(farthest, depth[uint(y) * size.x + uint(x)]);
        }
    }
    integrateFogColumn(frame, column, min(farthest, frame.fogOptions.y));
}
