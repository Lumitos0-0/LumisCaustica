import world_common;
import bindings;
import froxel_fog;

float4 fogLayerValue(ConstPtr<float4> data, uint start, uint4 columns, float4 weights) {
    return data[start + columns.x] * weights.x + data[start + columns.y] * weights.y
         + data[start + columns.z] * weights.z + data[start + columns.w] * weights.w;
}

// Returns in-scattered radiance and transmittance so subpixel coverage can mix both linearly.
float4 fogAtDepth(WorldPush frame, uint4 columns, float4 weights, float depth) {
    uint layers = uint(frame.fogGrid.w);
    float range = frame.fogOptions.y;
    float z = clamp(depth, 0.0, range);
    float u = fogLayer(z, range, layers);
    uint back = min(uint(ceil(u)), layers);
    if (back == 0u) return float4(0.0, 0.0, 0.0, 1.0);
    uint stride = uint(frame.fogGrid.x * frame.fogGrid.y);
    ConstPtr<float4> data = ConstPtr<float4>(frame.fogBufferAddr);
    float4 lo = float4(0.0);
    if (back > 1u) lo = fogLayerValue(data, (back - 2u) * stride, columns, weights);
    float4 hi = fogLayerValue(data, (back - 1u) * stride, columns, weights);
    float a = fogBoundary(float(back - 1u) / float(layers), range);
    float b = fogBoundary(float(back) / float(layers), range);
    float fraction = saturate((z - a) / (b - a));
    float tau = max(0.0, hi.a - lo.a);
    float blend = tau < 0.001 ? fraction : (1.0 - exp(-fraction * tau)) / (1.0 - exp(-tau));
    return float4(lerp(lo.rgb, hi.rgb, blend), exp(-lerp(lo.a, hi.a, fraction)));
}

float4 tentWeights(float2 f) {
    return float4((1.0-f.x)*(1.0-f.y), f.x*(1.0-f.y), (1.0-f.x)*f.y, f.x*f.y);
}

[shader("raygeneration")]
void main() {
    WorldPush frame = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 pixel = DispatchRaysIndex().xy;
    float2 uv = (float2(pixel) + 0.5) / float2(DispatchRaysDimensions().xy);
    uint2 size;
    outImage.GetDimensions(size.x, size.y);
    uint2 grid = uint2(frame.fogGrid.xy);
    float2 p = clamp(uv * float2(grid) - 0.5, float2(0.0), float2(grid - 1u));
    uint2 a = uint2(floor(p)), b = min(a + 1u, grid - 1u);
    uint4 cols = uint4(a.y * grid.x + a.x, a.y * grid.x + b.x,
                       b.y * grid.x + a.x, b.y * grid.x + b.x);
    float4 weights = tentWeights(frac(p));
    float2 q = clamp(uv * float2(size) - frame.jitter - 0.5, float2(0.0), float2(size - 1u));
    uint2 s = uint2(floor(q)), t = min(s + 1u, size - 1u);
    ConstPtr<float> depth = ConstPtr<float>(fogDepthAddress(frame));
    float4 z = min(frame.fogOptions.y, float4(depth[s.y*size.x+s.x], depth[s.y*size.x+t.x],
                  depth[t.y*size.x+s.x], depth[t.y*size.x+t.x]));
    float4 coverage = tentWeights(frac(q));
    float nearZ = min(min(z.x,z.y), min(z.z,z.w));
    float farZ = max(max(z.x,z.y), max(z.z,z.w));
    float4 fog;
    if (farZ - nearZ <= max(0.08, 0.015 * nearZ)) {
        fog = fogAtDepth(frame, cols, weights, dot(z, coverage));
    } else {
        // Integrate each visible surface separately at silhouettes, avoiding fictitious depths in air.
        fog = float4(0.0);
        for (int tap = 0; tap < 4; ++tap) {
            if (coverage[tap] > 0.0) fog += coverage[tap] * fogAtDepth(frame, cols, weights, z[tap]);
        }
    }
    float4 scene = fogTarget[int2(pixel)];
    fogTarget[int2(pixel)] = float4(scene.rgb * fog.a + fog.rgb * frame.preExposure, scene.a);
}
