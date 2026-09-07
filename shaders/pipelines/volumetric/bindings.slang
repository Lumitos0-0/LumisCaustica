// Descriptor interface shared by the two volumetric compute passes. Both declare all five bindings and
// each uses the subset it needs, mirroring how the three sky-LUT bakes share one set.
//
// fogScatter is the raw in-scatter the injection raygen wrote this frame (same image, same binding
// index in the world set). fogHistory/fogHistoryPrev are the ping-pong accumulation pair, and
// fogIntegrated is the camera-to-slice prefix the composite samples.

[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture3D<float4> fogScatter;
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture3D<float4> fogHistory;
[[vk::binding(2, 0)]] public Sampler3D fogHistoryPrev;
[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture3D<float4> fogIntegrated;
[[vk::binding(4, 0)]] public Sampler3D fogIntegratedSampler;
[[vk::binding(5, 0)]] [format("rgba16f")] public RWTexture2D<float4> sceneImage;
[[vk::binding(6, 0)]] [format("r32f")] public RWTexture2D<float> sceneDepth;
