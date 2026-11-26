module volume_grid;

import world_common;

// Both visibility banks store RGB transmittance for the sun and the moon. Prefixes
// store absolute scene radiance and particulate optical depth at each slice exit.
public uint columnCount(WorldPush f) { return uint(f.fogGrid.x * f.fogGrid.y); }
public uint cellCount(WorldPush f) { return columnCount(f) * uint(f.fogGrid.z); }
public uint64_t bankAddress(WorldPush f, uint bank) {
    return f.fogBufferAddr + uint64_t(bank * cellCount(f)) * 8u;
}
public uint64_t prefixAddress(WorldPush f) { return f.fogBufferAddr + uint64_t(cellCount(f)) * 16u; }
public uint64_t boundsAddress(WorldPush f) { return f.fogBufferAddr + uint64_t(cellCount(f)) * 32u; }
public uint64_t fogDepthAddress(WorldPush f) { return boundsAddress(f) + uint64_t(columnCount(f)) * 4u; }

public float sliceDepth(float u, float range) { return range * u * (0.25 + 0.75 * u); }
public float sliceIndex(float depth, float range, int slices) {
    float x = saturate(depth / range);
    return float(slices) * (2.0 * x / (0.25 + sqrt(0.0625 + 3.0 * x)));
}

// Solve inverseProjection * clip at homogeneous world w=1, with clip.w fixed
// to the requested view depth. Works with off-axis projections and camera bob.
public float3 volumePosition(WorldPush f, float2 uv, float depth) {
    float4 base = mul(f.invViewProj, float4((uv * 2.0 - 1.0) * depth, 0.0, depth));
    float4 dz = mul(f.invViewProj, float4(0.0, 0.0, 1.0, 0.0));
    return f.camOffset + base.xyz + dz.xyz * ((1.0 - base.w) / dz.w);
}

public uint packVisibility(float3 v) {
    uint3 q = uint3(round(sqrt(saturate(v * 0.25)) * 1023.0));
    return 0x80000000u | q.x | (q.y << 10u) | (q.z << 20u);
}
public float3 unpackVisibility(uint word) {
    float3 q = float3(uint3(word, word >> 10u, word >> 20u) & 1023u) / 1023.0;
    return 4.0 * q * q;
}

public uint visibleSlices(WorldPush f, uint column) {
    float depth = ConstPtr<float>(boundsAddress(f))[column];
    return min(uint(f.fogGrid.z), uint(ceil(sliceIndex(depth, f.fogOptions.y, f.fogGrid.z))) + 1u);
}

// Integral of exp(-sigma*s) over an interval, divided by interval length.
// Its series avoids cancellation in nearly transparent channels.
public float pathAverage(float tau) {
    return tau < 0.01 ? 1.0 - tau * (0.5 - tau / 6.0) : (1.0 - exp(-tau)) / tau;
}
public float3 pathAverage3(float3 tau) {
    return float3(pathAverage(tau.x), pathAverage(tau.y), pathAverage(tau.z));
}
