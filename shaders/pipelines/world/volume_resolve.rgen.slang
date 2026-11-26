import world_common;
import volume_grid;
import bindings;
import medium;

float4 tent(float2 f) {
    return float4((1.0-f.x)*(1.0-f.y), f.x*(1.0-f.y), (1.0-f.x)*f.y, f.x*f.y);
}

float4 layerValue(ConstPtr<float4> prefixes, uint base, uint4 c, float4 w) {
    return w.x * prefixes[base+c.x] + w.y * prefixes[base+c.y]
         + w.z * prefixes[base+c.z] + w.w * prefixes[base+c.w];
}

float4 resolveSegment(WorldPush f, uint4 columns, float4 weights, float depth,
                      float nearDepth, float3 absorptionPerDepth) {
    uint n = uint(f.fogGrid.z);
    if (depth <= nearDepth) return float4(0.0, 0.0, 0.0, 1.0);
    float range = f.fogOptions.y;
    uint i = min(uint(floor(sliceIndex(depth, range, f.fogGrid.z))), n-1u);
    float a = max(nearDepth, sliceDepth(float(i) / float(n), range));
    float b = sliceDepth(float(i+1u) / float(n), range);
    float part = saturate((depth - a) / max(b-a, 0.00001));
    uint stride = columnCount(f);
    ConstPtr<float4> prefixes = ConstPtr<float4>(prefixAddress(f));
    float4 lo = float4(0.0);
    if (i > 0u) lo = layerValue(prefixes, (i-1u)*stride, columns, weights);
    float4 hi = layerValue(prefixes, i*stride, columns, weights);
    float3 tau = max(0.0, hi.a - lo.a) + absorptionPerDepth * max(0.0, b-a);
    float3 blend = part * pathAverage3(part * tau) / pathAverage3(tau);
    return float4(lerp(lo.rgb, hi.rgb, blend), exp(-lerp(lo.a, hi.a, part)));
}

[shader("raygeneration")]
void main() {
    WorldPush f = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 pixel = DispatchRaysIndex().xy;
    float2 uv = (float2(pixel) + 0.5) / float2(DispatchRaysDimensions().xy);
    uint2 grid = uint2(f.fogGrid.xy), size;
    outImage.GetDimensions(size.x, size.y);
    float2 p = clamp(uv * float2(grid) - 0.5, float2(0.0), float2(grid - 1u));
    uint2 a = uint2(floor(p)), b = min(a+1u, grid-1u);
    uint4 columns = uint4(a.y*grid.x+a.x, a.y*grid.x+b.x, b.y*grid.x+a.x, b.y*grid.x+b.x);
    float4 weights = tent(frac(p));
    float2 q = clamp(uv * float2(size) - f.jitter - 0.5, float2(0.0), float2(size-1u));
    uint2 s = uint2(floor(q)), t = min(s+1u, size-1u);
    ConstPtr<float> depths = ConstPtr<float>(fogDepthAddress(f));
    float4 z = float4(depths[s.y*size.x+s.x], depths[s.y*size.x+t.x],
                      depths[t.y*size.x+s.x], depths[t.y*size.x+t.x]);
    float4 coverage = tent(frac(q));
    float3 origin, dir;
    worldCameraRay(f, uv * 2.0 - 1.0, origin, dir);
    float nearDepth = mul(f.curViewProj, float4(origin - f.camOffset, 1.0)).w;
    float3 absorption = float3(0.0);
    if ((f.flags & 1u) != 0u) {
        absorption = waterExtinction(f.waterParams.xyz) / max(0.0001, mul(f.curViewProj, float4(dir, 0.0)).w);
    }
    float nearest = min(min(z.x,z.y), min(z.z,z.w));
    float farthest = max(max(z.x,z.y), max(z.z,z.w));
    float4 fog = float4(0.0);
    if (farthest-nearest <= max(0.04, 0.01*nearest)) {
        fog = resolveSegment(f, columns, weights, dot(z,coverage), nearDepth, absorption);
    } else {
        // At a silhouette, mix the separately terminated rays by coverage.
        for (int j=0; j<4; ++j) {
            if (coverage[j] > 0.0) fog += coverage[j] * resolveSegment(f, columns, weights, z[j], nearDepth, absorption);
        }
    }
    float4 scene = fogTarget[int2(pixel)];
    fogTarget[int2(pixel)] = float4(min(float3(HALF_MAX), scene.rgb * fog.a + fog.rgb * f.preExposure), scene.a);
}
