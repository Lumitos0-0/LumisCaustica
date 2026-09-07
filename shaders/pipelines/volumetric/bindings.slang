import world_common;
import volumetric_common;

[[vk::push_constant]] public VolumetricPush pc;

[[vk::binding(0, 0)]] public RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] public Sampler2D skyViewLut;
[[vk::binding(2, 0)]] public Sampler2D transmittanceLut;
[[vk::binding(3, 0)]] public Sampler3D prevFroxelRadiance;
[[vk::binding(4, 0)]] [format("rgba16f")] public RWTexture3D<float4> outFroxelRadiance;
[[vk::binding(5, 0)]] [format("rgba16f")] public RWTexture3D<float4> outFroxelExtinction;
