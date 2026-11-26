module fog_transport;

import world_common;
import bindings;
import sky;
import trace;
import froxel_fog;

float3 fogShadow(float3 position, float3 lightDir) {
    return visibility(position, lightDir, 10000.0).transmittance;
}

void integrateSegment(inout float4 volume, WorldPush frame, float3 origin, float3 dir,
                      float begin, float end, float3 incoming) {
    float mid = 0.5 * (begin + end);
    float altitude = origin.y + mid * dir.y + frame.fogAltitudeOffset;
    float4 fog = frame.fogOptions;
    float tau = (end - begin) * fog.x * exp(-max(0.0, altitude) * fog.w)
              * (1.0 - smoothstep(0.85 * fog.y, fog.y, mid));
    float scattered = tau < 0.001 ? tau * (1.0 - 0.5 * tau) : 1.0 - exp(-tau);
    volume.rgb += exp(-volume.a) * scattered * incoming;
    volume.a += tau;
}

public float4 integrateFogRay(WorldPush frame, float2 uv, float depth) {
    float3 origin, dir;
    worldCameraRay(frame, uv * 2.0 - 1.0, origin, dir);
    CelestialLight light = dominantCelestialLight(skyState(frame), transmittanceLut);
    float g = frame.fogOptions.z;
    float d = 1.0 + g * g - 2.0 * g * clamp(dot(dir, light.dir), -1.0, 1.0);
    float3 incoming = bt709ToAcesCg(light.illuminance) * ((1.0 - g*g) / (12.5663706 * d * sqrt(d)));
    bool lit = any(incoming > 0.0);
    float range = frame.fogOptions.y;
    float limit = clamp(depth, 0.0, range);
    uint slices = uint(frame.fogGrid.z);
    float4 volume = float4(0.0);
    float begin = 0.0;
    float3 left = lit ? fogShadow(origin, light.dir) : float3(0.0);
    for (uint i = 1u; i <= slices && begin < limit; ++i) {
        float u = float(i) / float(slices);
        float end = min(limit, range * u * (0.25 + 0.75 * u));
        float mid = 0.5 * (begin + end);
        float3 center = lit ? fogShadow(origin + mid * dir, light.dir) : float3(0.0);
        // Stay inside the visible interval at the terminal opaque surface.
        float inside = end - min(0.002, (end - begin) * 0.01);
        float3 right = lit ? fogShadow(origin + inside * dir, light.dir) : float3(0.0);
        if (any(max(abs(left - center), abs(right - center)) > 0.06)) {
            float3 front = fogShadow(origin + 0.5 * (begin + mid) * dir, light.dir);
            float3 back = fogShadow(origin + 0.5 * (mid + end) * dir, light.dir);
            integrateSegment(volume, frame, origin, dir, begin, mid, incoming * front);
            integrateSegment(volume, frame, origin, dir, mid, end, incoming * back);
        } else {
            integrateSegment(volume, frame, origin, dir, begin, end, incoming * center);
        }
        left = right;
        begin = end;
    }
    return volume;
}
