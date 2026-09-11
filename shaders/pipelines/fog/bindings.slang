// Descriptor interface for the fog apply pass: blends the froxel volume (pipelines/world/fog.rgen's
// output) over the render-res path-traced image at each pixel's scene depth.
[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] [format("r32f")] public RWTexture2D<float> depthImage;
[[vk::binding(2, 0)]] public Sampler3D fogVolume;
// Temporal history ping-pong pair (RGBA = accumulated scatter + transmittance), selected by frame
// parity in the shader: the read image is never the write image, so no in-dispatch barrier is needed.
[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture2D<float4> fogHistoryA;
[[vk::binding(4, 0)]] [format("rgba16f")] public RWTexture2D<float4> fogHistoryB;
// Guide motion vectors (render pixels, current -> previous, jitter-free) for reprojection.
[[vk::binding(5, 0)]] [format("rg16f")] public RWTexture2D<float2> motionImage;
