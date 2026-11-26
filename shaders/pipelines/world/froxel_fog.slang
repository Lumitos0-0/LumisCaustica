module froxel_fog;

import world_common;

public uint fogCellCount(int4 grid) {
    return uint(grid.x * grid.y * grid.z);
}

public uint fogCellIndex(uint3 cell, uint3 grid) {
    return (cell.z * grid.y + cell.y) * grid.x + cell.x;
}

public uint fogLightCount(WorldPush frame) {
    return fogCellCount(frame.fogNearGrid) + fogCellCount(frame.fogFarGrid);
}

public uint64_t fogFroxelAddress(WorldPush frame) {
    return frame.fogBufferAddr + uint64_t(fogLightCount(frame)) * 16u;
}

public uint64_t fogDepthAddress(WorldPush frame) {
    return fogFroxelAddress(frame) + uint64_t(fogCellCount(frame.fogGrid)) * 16u;
}

public float3 fogViewDirection(WorldPush frame, float2 uv) {
    float4 p = mul(frame.invViewProj, float4(uv * 2.0 - 1.0, 1.0, 1.0));
    return normalize(p.xyz / p.w);
}

// A froxel spans two radial distances on the same perspective ray.
public float fogSliceDistance(float boundary, uint slices, float range) {
    float t = boundary / float(slices);
    return (0.25 * t + 0.75 * t * t) * range;
}

public uint fogDistanceSlice(float distance, uint slices, float range) {
    float t = (2.0 * distance / range) /
            (0.25 + sqrt(0.0625 + 3.0 * distance / range));
    return min(uint(t * float(slices)), slices - 1u);
}

public float fogMeanTransmittance(float opticalDepth) {
    return opticalDepth < 0.001
            ? 1.0 - opticalDepth * 0.5
            : (1.0 - exp(-opticalDepth)) / opticalDepth;
}

public float fogPhase(float cosTheta, float g) {
    float d = 1.0 + g * g - 2.0 * g * cosTheta;
    return (1.0 - g * g) / (12.5663706 * d * sqrt(d));
}

float3 lookupLight(ConstPtr<float4> cache, uint offset, float3 local, int4 grid) {
    float3 p = clamp(local, float3(0.0), float3(grid.xyz - 1));
    uint3 a = uint3(floor(p));
    uint3 b = min(a + 1u, uint3(grid.xyz - 1));
    float3 w = frac(p);
    uint3 dims = uint3(grid.xyz);
    float3 z0y0 = lerp(
        cache[offset + fogCellIndex(a, dims)].rgb,
        cache[offset + fogCellIndex(uint3(b.x, a.y, a.z), dims)].rgb, w.x);
    float3 z0y1 = lerp(
        cache[offset + fogCellIndex(uint3(a.x, b.y, a.z), dims)].rgb,
        cache[offset + fogCellIndex(uint3(b.x, b.y, a.z), dims)].rgb, w.x);
    float3 z1y0 = lerp(
        cache[offset + fogCellIndex(uint3(a.x, a.y, b.z), dims)].rgb,
        cache[offset + fogCellIndex(uint3(b.x, a.y, b.z), dims)].rgb, w.x);
    float3 z1y1 = lerp(
        cache[offset + fogCellIndex(uint3(a.x, b.y, b.z), dims)].rgb,
        cache[offset + fogCellIndex(b, dims)].rgb, w.x);
    return lerp(lerp(z0y0, z0y1, w.y), lerp(z1y0, z1y1, w.y), w.z);
}

public float3 fogLightVisibility(WorldPush frame, float3 position) {
    ConstPtr<float4> cache = ConstPtr<float4>(frame.fogBufferAddr);
    float3 nearCell = (position - frame.fogNearOrigin.xyz) / frame.fogNearOrigin.w;
    float3 nearEdge = min(nearCell, float3(frame.fogNearGrid.xyz - 1) - nearCell);
    float weight = smoothstep(0.0, 5.0, min(min(nearEdge.x, nearEdge.y), nearEdge.z));
    if (weight >= 1.0) {
        return lookupLight(cache, 0u, nearCell, frame.fogNearGrid);
    }
    float3 farCell = (position - frame.fogFarOrigin.xyz) / frame.fogFarOrigin.w;
    float3 farLight = lookupLight(cache, fogCellCount(frame.fogNearGrid), farCell, frame.fogFarGrid);
    if (weight <= 0.0) {
        return farLight;
    }
    return lerp(farLight, lookupLight(cache, 0u, nearCell, frame.fogNearGrid), weight);
}

float4 fogColumnAtDepth(ConstPtr<float4> froxels, uint2 column, uint3 grid,
                        uint slice, float fraction) {
    uint idx = fogCellIndex(uint3(column, slice), grid);
    float4 back = froxels[idx];
    float4 front = float4(0.0);
    if (slice != 0u) {
        front = froxels[idx - grid.x * grid.y];
    }
    // Interpolate optical depth linearly and inscattering by the integrated fraction of the segment.
    float tau = max(0.0, back.a - front.a);
    float weight = fraction * fogMeanTransmittance(tau * fraction) / fogMeanTransmittance(tau);
    return float4(lerp(front.rgb, back.rgb, weight), front.a + tau * fraction);
}

public float4 fogAtPixel(WorldPush frame, float2 uv, float distance) {
    uint3 grid = uint3(frame.fogGrid.xyz);
    float range = frame.fogOptions.y;
    float end = clamp(distance, 0.0, range);
    uint slice = fogDistanceSlice(end, grid.z, range);
    float begin = fogSliceDistance(float(slice), grid.z, range);
    float finish = fogSliceDistance(float(slice + 1u), grid.z, range);
    float fraction = saturate((end - begin) / (finish - begin));

    float2 p = clamp(uv * float2(grid.xy) - 0.5, float2(0.0), float2(grid.xy - 1u));
    uint2 a = uint2(floor(p));
    uint2 b = min(a + 1u, grid.xy - 1u);
    float2 w = frac(p);
    ConstPtr<float4> froxels = ConstPtr<float4>(fogFroxelAddress(frame));
    // All four columns are evaluated at the exact same first-hit distance of this pixel.
    float4 top = lerp(fogColumnAtDepth(froxels, a, grid, slice, fraction),
                     fogColumnAtDepth(froxels, uint2(b.x, a.y), grid, slice, fraction), w.x);
    float4 bottom = lerp(fogColumnAtDepth(froxels, uint2(a.x, b.y), grid, slice, fraction),
                        fogColumnAtDepth(froxels, b, grid, slice, fraction), w.x);
    return lerp(top, bottom, w.y);
}
