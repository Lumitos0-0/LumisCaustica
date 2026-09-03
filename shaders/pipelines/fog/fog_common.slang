// Shared push-constant records for the primary-view froxel fog passes.
// The fog lighting pass deliberately receives only the WorldPush address rather than importing the world
// descriptor ABI. That keeps this hybrid volume renderer independent from the surface path's SBT and
// descriptor layout while still sharing the exact camera, rebase, sky, and exposure state.
module fog_common;

import world_common;

// Froxel fields use fp16 for bandwidth. Keep their payload in a shared linear scale so a forward sun
// phase peak cannot overflow 65504, then restore absolute radiance at the integration seam.
public static const float FOG_RADIANCE_SCALE = 128.0;

public struct FogLightingPush {
    public uint64_t worldPushAddr;
    public uint64_t lightBufAddr;
    // xyz = froxel atlas dimensions (x, y, z), w = logarithmic far distance in world blocks.
    public float4 grid;
    // x = direct phase anisotropy, y = local-emissive strength, z = history blend, w = local samples.
    public float4 settings;
    // x = GI ray samples, y = spatial history radius in froxels, z = direct froxel samples, w reserved.
    public float4 giSettings;
    public uint lightCount;
    public uint frameIndex;
};

public struct FogIntegratePush {
    public uint64_t worldPushAddr;
    // xyz = froxel atlas dimensions (x, y, z), w = logarithmic far distance in world blocks.
    public float4 grid;
    // x = base density in inverse blocks, y = vertical falloff, z = world/rebase-space base height,
    // w = near integration distance.
    public float4 medium;
    // x = direct field scale, y = GI field scale, z = scattering scale, w reserved.
    public float4 lighting;
    // RGB = linear ACEScg scattering albedo of the medium; w reserved.
    public float4 scatteringColor;
};
