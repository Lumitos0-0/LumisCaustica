module froxel_fog;

import world_common;

public uint64_t fogDepthAddress(WorldPush frame) {
    return frame.fogBufferAddr + uint64_t(frame.fogGrid.x) * uint64_t(frame.fogGrid.y)
            * uint64_t(frame.fogGrid.w) * 16u;
}

public float fogBoundary(float u, float range) {
    return range * u * (0.25 + 0.75 * u);
}

// Inverse of the quadratic slice distribution, rationalized to preserve near-camera precision.
public float fogLayer(float depth, float range, uint layers) {
    float x = saturate(depth / range);
    return float(layers) * (2.0 * x / (0.25 + sqrt(0.0625 + 3.0 * x)));
}
