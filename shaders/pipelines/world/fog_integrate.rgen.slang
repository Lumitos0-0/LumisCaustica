import world_common;
import bindings;
import sky;
import froxel_fog;

[shader("raygeneration")]
void main() {
    WorldPush frame = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 column = DispatchRaysIndex().xy;
    uint3 grid = uint3(frame.fogGrid.xyz);
    float2 uv = (float2(column) + 0.5) / float2(grid.xy);
    float3 dir = fogViewDirection(frame, uv);
    SkyState sky = skyState(frame);
    CelestialLight light = dominantCelestialLight(sky, transmittanceLut);
    float4 fog = frame.fogOptions;
    float phase = fogPhase(clamp(dot(dir, light.dir), -1.0, 1.0), fog.z);
    float3 illuminance = bt709ToAcesCg(light.illuminance) * phase;
    DevicePtr<float4> froxels = DevicePtr<float4>(fogFroxelAddress(frame));
    float3 scatter = float3(0.0);
    float tau = 0.0;
    float begin = 0.0;
    for (uint slice = 0u; slice < grid.z; ++slice) {
        float end = fogSliceDistance(float(slice + 1u), grid.z, fog.y);
        float mid = (begin + end) * 0.5;
        float3 position = frame.camOffset + dir * mid;
        float altitude = position.y + frame.fogAltitudeOffset;
        float density = fog.x * exp(-max(0.0, altitude) * fog.w)
                * (1.0 - smoothstep(fog.y * 0.85, fog.y, mid));
        float sliceTau = density * (end - begin);
        float3 visibilityRgb = fogLightVisibility(frame, position);
        scatter += exp(-tau) * (sliceTau * fogMeanTransmittance(sliceTau)) * illuminance * visibilityRgb;
        tau += sliceTau;
        froxels[fogCellIndex(uint3(column, slice), grid)] = float4(scatter, tau);
        begin = end;
    }
}
