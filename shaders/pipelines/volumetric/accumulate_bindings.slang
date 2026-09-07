import volumetric_common;

[[vk::push_constant]] public VolumetricPush pc;

[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture3D<float4> inFroxelRadiance;
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture3D<float4> inFroxelExtinction;
[[vk::binding(2, 0)]] [format("rgba16f")] public RWTexture3D<float4> outAccumulatedFroxels;
