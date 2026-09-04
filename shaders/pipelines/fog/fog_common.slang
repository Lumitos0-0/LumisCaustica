// Shared push-constant records for the primary-view froxel fog passes.
// The lighting pass receives the world-data address plus the published light-hierarchy addresses. It does
// not import the surface descriptor ABI, so the camera volume remains independent from the world SBT.
module fog_common;

import world_common;

// Froxel fields use fp16 for bandwidth. Keep their payload in a shared linear scale so a forward sun
// phase peak cannot overflow 65504, then restore absolute radiance at the integration seam.
public static const float FOG_RADIANCE_SCALE = 128.0;

public struct FogLightingPush {
    public uint64_t worldPushAddr;
    public uint64_t lightBufAddr;
    public uint64_t lightAliasAddr;
    public uint64_t lightLocalAliasAddr;
    public uint64_t lightGridCellAddr;
    public uint64_t lightGridSpanAddr;
    // xyz = froxel atlas dimensions (x, y, z), w = logarithmic far distance in world blocks.
    public float4 grid;
    // x = phase anisotropy, y = diagnostic local-emitter enable, z = direct enable, w = local samples.
    public float4 settings;
    // x = GI samples (zero disables GI), y = GI history radius, z = direct samples (zero disables direct),
    // w = cache cell size (zero disables cache/reuse). Cache reuse uses one world-space keyed proposal per cell.
    public float4 giSettings;
    // x = GI history blend, y = direct history blend, z = local history blend, w = cache history blend.
    public float4 temporalSettings;
    // x = GI history radius, y = direct history radius, z = local history radius, w = cache history radius.
    public float4 spatialSettings;
};

public struct FogIntegratePush {
    public uint64_t worldPushAddr;
    // xyz = froxel atlas dimensions (x, y, z), w = logarithmic far distance in world blocks.
    public float4 grid;
    // x = base density in inverse blocks, y = vertical falloff, z = world/rebase-space base height,
    // w = near integration distance.
    public float4 medium;
    // x = direct field scale, y = GI field scale, z = local field scale, w = cache field scale.
    public float4 lighting;
    // RGB = linear ACEScg scattering albedo; w = extinction-to-scattering scale.
    public float4 scatteringColor;
};
