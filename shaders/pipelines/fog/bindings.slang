// Descriptor interface for the fog apply pass: blends the froxel volume (pipelines/world/fog.rgen's
// output) over the render-res path-traced image at each pixel's scene depth.
[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] [format("r32f")] public RWTexture2D<float> depthImage;
[[vk::binding(2, 0)]] public Sampler3D fogVolume;
