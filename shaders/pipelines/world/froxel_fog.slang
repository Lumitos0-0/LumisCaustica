module froxel_fog;

import world_common;

public uint64_t fogDepthAddress(WorldPush frame) {
    return frame.fogBufferAddr + uint64_t(frame.fogGrid.x) * uint64_t(frame.fogGrid.y) * 16u;
}

// Primary depths include subpixel jitter. The display and fog columns use stable film coordinates.
public uint2 fogDepthPixel(float2 uv, uint2 size, float2 jitter) {
    return uint2(clamp(floor(uv * float2(size) - jitter), float2(0.0), float2(size - 1u)));
}
