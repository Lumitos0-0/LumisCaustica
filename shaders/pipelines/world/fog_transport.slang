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

public void integrateFogColumn(WorldPush frame, uint2 column, float limit) {
    uint2 grid = uint2(frame.fogGrid.xy);
    float2 uv = (float2(column) + 0.5) / float2(grid);
    float3 origin, dir;
    worldCameraRay(frame, uv * 2.0 - 1.0, origin, dir);
    CelestialLight light = dominantCelestialLight(skyState(frame), transmittanceLut);
    float g = frame.fogOptions.z;
    float d = 1.0 + g * g - 2.0 * g * clamp(dot(dir, light.dir), -1.0, 1.0);
    float3 incoming = bt709ToAcesCg(light.illuminance) * ((1.0 - g*g) / (12.5663706 * d * sqrt(d)));
    bool lit = any(incoming > 0.0);
    float range = frame.fogOptions.y;
    uint slices = uint(frame.fogGrid.z);
    uint stride = grid.x * grid.y;
    uint index = column.y * grid.x + column.x;
    DevicePtr<float4> volumeBuffer = DevicePtr<float4>(frame.fogBufferAddr);
    float4 volume = float4(0.0);
    float begin = 0.0;
    // Deterministic stratification distributes shadow samples in depth across adjacent columns.
    uint phase = (column.x & 1u) + 2u * (column.y & 1u);
    for (uint i = 0u; i < slices; ++i) {
        float end = fogBoundary(float(i + 1u) / float(slices), range);
        float offset = (float((phase + 3u * i) & 3u) + 0.5) * 0.25;
        float3 position = origin + lerp(begin, end, offset) * dir;
        float3 shadow = lit ? fogShadow(position, light.dir) : float3(0.0);
        integrateSegment(volume, frame, origin, dir, begin, end, incoming * shadow);
        begin = end;
        if ((i & 3u) == 3u) {
            volumeBuffer[index] = volume;
            index += stride;
            if (end > limit + 0.0001 * range) break;
        }
    }
}
