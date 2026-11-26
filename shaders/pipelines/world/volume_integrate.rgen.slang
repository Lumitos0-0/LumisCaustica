import world_common;
import bindings;
import volume_grid;
import medium;
import world_core;
import sky;

float phaseHG(float mu, float g) {
    float d = max(0.0001, 1.0 + g*g - 2.0*g*clamp(mu, -1.0, 1.0));
    return (1.0 - g*g) / (12.5663706 * d * sqrt(d));
}

float3 incident(WorldPush f, SkyState sky, float3 viewRay, float3 axis, float angle,
                float illuminance, float4 artist) {
    float3 spectrum = artist.rgb * artist.w * illuminance
            * transmittanceToSpace(transmittanceLut, sky.viewerRadiusKm, axis, angle);
    if ((f.flags & 1u) != 0u) axis = -refract(-axis, float3(0,1,0), 1.0 / WATER_IOR);
    return bt709ToAcesCg(spectrum) * phaseHG(dot(viewRay, axis), f.fogOptions.z);
}

[shader("raygeneration")]
void main() {
    WorldPush f = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 cell = DispatchRaysIndex().xy;
    uint col = cell.y * uint(f.fogGrid.x) + cell.x;
    uint stride = columnCount(f);
    uint count = visibleSlices(f, col);
    float2 uv = (float2(cell) + 0.5) / float2(f.fogGrid.xy);
    float3 origin, dir;
    worldCameraRay(f, uv * 2.0 - 1.0, origin, dir);
    float nearDepth = mul(f.curViewProj, float4(origin - f.camOffset, 1.0)).w;
    float metresPerDepth = 1.0 / max(0.0001, mul(f.curViewProj, float4(dir, 0.0)).w);
    SkyState sky = skyState(f);
    float3 sun = incident(f, sky, dir, sky.sunDir, sky.sunAngularRadius, sky.sunIlluminance, f.fogSun);
    float3 moon = incident(f, sky, dir, sky.moonDir, sky.moonAngularRadius, sky.moonIlluminance, f.fogMoon);
    bool submerged = (f.flags & 1u) != 0u;
    float3 absorption = submerged ? waterExtinction(f.waterParams.xyz) : float3(0.0);
    ConstPtr<uint2> lighting = ConstPtr<uint2>(bankAddress(f, uint(f.fogGrid.w)));
    DevicePtr<float4> prefixes = DevicePtr<float4>(prefixAddress(f));
    float3 scatter = float3(0.0), eye = float3(1.0);
    float opticalDepth = 0.0, start = 0.0;
    for (uint slice = 0u; slice < count; ++slice) {
        uint index = slice * stride + col;
        float axial = sliceDepth(float(slice + 1u) / float(f.fogGrid.z), f.fogOptions.y);
        float end = max(0.0, (axial - nearDepth) * metresPerDepth);
        float mid = 0.5 * (start + end);
        float altitude = max(0.0, origin.y + mid * dir.y + f.fogTemporal.z);
        float density = f.fogOptions.x * (1.0 - smoothstep(0.85 * f.fogOptions.y, f.fogOptions.y, mid));
        if (!submerged) density *= exp(-altitude * f.fogOptions.w);
        float ds = end - start;
        float particleTau = ds * density;
        float3 tau = ds * absorption + particleTau;
        uint2 vis = lighting[index];
        float3 source = sun * unpackVisibility(vis.x) + moon * unpackVisibility(vis.y);
        scatter += eye * (particleTau * pathAverage3(tau)) * source;
        eye *= exp(-tau);
        // Native water already attenuates the underlying path-traced surface. Only
        // particulate extinction belongs in the multiplier applied to that surface.
        opticalDepth += particleTau;
        prefixes[index] = float4(scatter, opticalDepth);
        start = end;
    }
}
