// Descriptor interface of the display-res fog march. Integrates the resolved froxel cache along each
// camera ray up to the scene depth and composites it over the denoised scene image:
// fogged = scene * transmittance + in-scatter. Runs after DLSS-RR because the cache is already
// temporally clean and needs no denoiser, and before exposure/bloom so fog participates in both.
//
// Both halves of the resolved cache's ping-pong pair are bound and the pushed parity picks the half the
// filter pass just wrote — same reasoning as the filter's own bindings: no per-frame descriptor update.

[[vk::binding(0, 0)]] public Sampler3D froxelA;
[[vk::binding(1, 0)]] public Sampler3D froxelB;
// The denoised scene and its reversed-Z depth guide, read as storage images (nearest, exact texel).
[[vk::binding(2, 0)]] [format("rgba16f")] public RWTexture2D<float4> sceneImage;
[[vk::binding(3, 0)]] [format("r32f")] public RWTexture2D<float> depthImage;
[[vk::binding(4, 0)]] [format("rgba16f")] public RWTexture2D<float4> foggedImage;
