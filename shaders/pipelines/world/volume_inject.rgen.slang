import world_common;
import world_core;
import bindings;
import trace;
import sky;
import water;
import medium;
import volume_grid;

uint mixBits(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    return x ^ (x >> 16u);
}

float randomShift(uint x) { return float(mixBits(x) >> 8u) * (1.0 / 16777216.0); }

float3 directionOnDisc(float3 axis, float angle, float2 u) {
    float3 side = normalize(cross(abs(axis.y) < 0.95 ? float3(0,1,0) : float3(1,0,0), axis));
    float azimuth = 6.2831853 * u.y;
    return normalize(axis + tan(angle) * sqrt(u.x)
            * (cos(azimuth) * side + sin(azimuth) * cross(axis, side)));
}

bool readHistory(WorldPush f, float3 center, uint3 cell, out float3 sun, out float3 moon, out float shift) {
    sun = float3(0.0);
    moon = float3(0.0);
    shift = 0.0;
    if (f.fogTemporal.x <= 0.0) return false;
    float4 clip = mul(f.prevViewProj, float4(center - f.camOffset + f.camDelta, 1.0));
    if (clip.w <= 0.0 || clip.w >= f.fogOptions.y) return false;
    float3 p = float3((0.5 + 0.5 * clip.xy / clip.w) * float2(f.fogGrid.xy),
                      sliceIndex(clip.w, f.fogOptions.y, f.fogGrid.z)) - 0.5;
    if (any(p < -0.001) || any(p > float3(f.fogGrid.xyz - 1) + 0.001)) return false;
    shift = length(p - float3(cell));
    p = clamp(p, float3(0.0), float3(f.fogGrid.xyz - 1));
    ConstPtr<uint2> history = ConstPtr<uint2>(bankAddress(f, 1u - uint(f.fogGrid.w)));
    uint3 base = uint3(floor(p));
    float3 fraction = frac(p);
    float valid = 0.0;
    [unroll] for (uint z = 0u; z < 2u; ++z) {
        [unroll] for (uint y = 0u; y < 2u; ++y) {
            [unroll] for (uint x = 0u; x < 2u; ++x) {
                uint3 off = uint3(x,y,z);
                float3 k = lerp(1.0 - fraction, fraction, float3(off));
                float weight = k.x * k.y * k.z;
                if (weight <= 0.0001) continue;
                uint3 tap = min(base + off, uint3(f.fogGrid.xyz - 1));
                uint2 v = history[(tap.z * uint(f.fogGrid.y) + tap.y) * uint(f.fogGrid.x) + tap.x];
                if ((v.x & 0x80000000u) == 0u) continue;
                sun += weight * unpackVisibility(v.x);
                moon += weight * unpackVisibility(v.y);
                valid += weight;
            }
        }
    }
    if (valid < 0.98) return false;
    sun /= valid;
    moon /= valid;
    return true;
}

float3 sampleLight(WorldPush f, float3 pos, float3 light, float footprint, float angularRadius) {
    if ((f.flags & 1u) == 0u) return visibility(pos, light, 10000.0).transmittance;
    if (light.y <= 0.0) return float3(0.0);
    // Follow the submerged leg to the actual fluid surface. The second leg is in air.
    float3 throughWater = -refract(-light, float3(0.0, 1.0, 0.0), 1.0 / WATER_IOR);
    Payload hit = makeRadiancePayload(PAYLOAD_WATER_INTERFACE_QUERY, 0u);
    TraceRay(topLevelAS, RAY_FLAG_NONE, CULL_SECONDARY, SBT_RADIANCE, SBT_STRIDE_BUCKET,
            MISS_GUIDE, makeRay(pos, RAY_TMIN, throughWater, 512.0), hit);
    if (hit.hitT < 0.0 || hit.flags != MATERIAL_WATER) return float3(0.0);
    float3 n = unpackNormal(hit);
    float cosI = saturate(dot(light, n));
    if (cosI <= 0.0) return float3(0.0);
    float3 exitPos = pos + hit.hitT * throughWater;
    float3 skyVisibility = visibility(exitPos + n * SURF_BIAS, light, 10000.0).transmittance;
    if (!any(skyVisibility > 0.0)) return float3(0.0);
    float focus = 1.0;
    if ((f.flags & 16u) != 0u && f.fogTemporal.y > 0.0 && n.y > 0.5) {
        float h = hit.hitT * throughWater.y;
        float waveFootprint = max(footprint, hit.hitT * tan(angularRadius));
        focus = lerp(1.0, waveFocus(exitPos.xz + f.waterAnchor.xy, f.waterParams.w,
                                   light, h, waveFootprint), f.fogTemporal.y);
    }
    float f0 = ((WATER_IOR - 1.0) / (WATER_IOR + 1.0));
    float transmission = (1.0 - f0 * f0) * (1.0 - pow(1.0 - cosI, 5.0));
    float3 sigma = waterExtinction(f.waterParams.xyz) + f.fogOptions.x;
    return min(float3(4.0), skyVisibility * exp(-sigma * hit.hitT) * (transmission * focus));
}

[shader("raygeneration")]
void main() {
    WorldPush f = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 launch = DispatchRaysIndex().xy;
    uint rows = uint(f.fogGrid.y);
    uint3 cell = uint3(launch.x, launch.y % rows, launch.y / rows);
    uint col = cell.y * uint(f.fogGrid.x) + cell.x;
    uint index = cell.z * columnCount(f) + col;
    DevicePtr<uint2> output = DevicePtr<uint2>(bankAddress(f, uint(f.fogGrid.w)));
    if (cell.z >= visibleSlices(f, col)) {
        output[index] = uint2(0u);
        return;
    }

    float3 dims = float3(f.fogGrid.xyz);
    float3 centerCoord = (float3(cell) + 0.5) / dims;
    float depth = sliceDepth(centerCoord.z, f.fogOptions.y);
    float3 center = volumePosition(f, centerCoord.xy, depth);
    float3 oldSun, oldMoon;
    float shift;
    bool previous = readHistory(f, center, cell, oldSun, oldMoon, shift);
    bool waterMedium = (f.flags & 1u) != 0u;
    uint seed = mixBits(index + 1u);
    // Alternate the traced half by a cell hash. Newly revealed cells always trace.
    if (previous && f.fogTemporal.w > 0.0 && !waterMedium && shift < 0.25 && ((seed ^ f.frameIndex) & 1u) != 0u) {
        output[index] = uint2(packVisibility(oldSun), packVisibility(oldMoon));
        return;
    }

    float t = float(f.frameIndex & 4095u);
    float3 u = frac(float3(randomShift(seed), randomShift(seed+1u), randomShift(seed+2u))
                    + t * float3(0.81917251, 0.67104361, 0.54970048));
    float3 coord = (float3(cell) + u) / dims;
    float z = sliceDepth(coord.z, f.fogOptions.y);
    float3 pos = volumePosition(f, coord.xy, z);
    float footprint = 0.0;
    if (waterMedium) {
        float3 next = volumePosition(f, coord.xy + float2(1.0 / dims.x, 0.0), z);
        footprint = max(0.08, 0.5 * length(next - pos));
    }
    SkyState sky = skyState(f);
    float2 disc = frac(float2(randomShift(seed+3u), randomShift(seed+4u)) + t * float2(0.75487767, 0.56984029));
    float3 sun = float3(0.0), moon = float3(0.0);
    if (f.fogSun.w > 0.0 && sky.sunIlluminance > 0.0 && sky.sunDir.y >= horizonZenithCos(sky.viewerRadiusKm) - sin(sky.sunAngularRadius)) {
        sun = sampleLight(f, pos, directionOnDisc(sky.sunDir, sky.sunAngularRadius, disc), footprint, sky.sunAngularRadius);
    }
    if (f.fogMoon.w > 0.0 && sky.moonIlluminance > 0.0 && sky.moonDir.y >= horizonZenithCos(sky.viewerRadiusKm) - sin(sky.moonAngularRadius)) {
        moon = sampleLight(f, pos, directionOnDisc(sky.moonDir, sky.moonAngularRadius, frac(disc + 0.5)), footprint, sky.moonAngularRadius);
    }
    if (previous) {
        float weight = f.fogTemporal.x / (1.0 + min(shift, 4.0));
        sun = lerp(sun, oldSun, weight);
        moon = lerp(moon, oldMoon, weight);
    }
    output[index] = uint2(packVisibility(sun), packVisibility(moon));
}
