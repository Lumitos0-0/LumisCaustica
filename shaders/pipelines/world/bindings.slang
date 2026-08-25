// Complete descriptor and push-constant interface for the world ray-tracing pipeline.
// Every world stage imports this module instead of redeclaring its own subset.

import world_common;

[[vk::push_constant]] public WorldPushConstants pc;

[[vk::binding(0, 0)]] public RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture2D<float4> outImage;

[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture2D<float4> gNormal;
[[vk::binding(4, 0)]] [format("rgba16f")] public RWTexture2D<float4> gAlbedo;
[[vk::binding(5, 0)]] [format("r32f")] public RWTexture2D<float> gDepth;
[[vk::binding(6, 0)]] [format("rg16f")] public RWTexture2D<float2> gMotion;
[[vk::binding(7, 0)]] [format("rgba16f")] public RWTexture2D<float4> gSpecAlbedo;
[[vk::binding(8, 0)]] [format("rg16f")] public RWTexture2D<float2> gSpecMotion;

[[vk::binding(2, 0)]] public Sampler2D blockAlbedoAtlas;
[[vk::binding(9, 0)]] public Sampler2D celestialsAtlas;
[[vk::binding(10, 0)]] public Sampler2D skyViewLut;
[[vk::binding(11, 0)]] public Sampler2D transmittanceLut;

// Camera-frustum participating media. volumeDepth stores the first primary interface distance so fog
// stops at glass/water instead of leaking through it. The scattering pair no longer ping-pongs (this
// system is deterministic and has no temporal accumulation); scattering[1]/froxelHistory is retained
// only to keep the descriptor layout unchanged. froxelFiltered removes single-cell noise before
// froxelIntegrated stores the per-pixel fog {in-scattering.rgb, transmittance} along the view ray.
[[vk::binding(12, 0)]] [format("r32f")] public RWTexture2D<float> volumeDepth;
[[vk::binding(13, 0)]] [format("rgba16f")] public RWTexture3D<float4> froxelScattering;
[[vk::binding(14, 0)]] [format("rgba16f")] public RWTexture3D<float4> froxelHistory;
[[vk::binding(15, 0)]] [format("rgba16f")] public RWTexture3D<float4> froxelFiltered;
[[vk::binding(16, 0)]] [format("rgba16f")] public RWTexture2D<float4> froxelIntegrated;

[[vk::binding(0, 1)]] public Sampler2D entityAlbedoTex[];
[[vk::binding(1, 1)]] public Sampler2D materialSurface0Tex[];
[[vk::binding(2, 1)]] public Sampler2D materialNormalAoTex[];
[[vk::binding(3, 1)]] public Sampler2D materialSurface1Tex[];
