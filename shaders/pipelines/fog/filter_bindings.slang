// Descriptor interface of the froxel temporal/spatial filter pass.
//
// rawFroxel is the per-froxel single-scatter source term the world pipeline's froxel_light.rgen writes
// this frame (rgb = in-scatter radiance per unit distance, a = extinction). It is bound as a storage
// image and read with clamped integer loads: the 3x3 tent is a handful of texels wide, so a sampler
// would add nothing but a descriptor.
//
// The resolved cache is a ping-pong pair held in two 3D images. Each image is bound twice — once as a
// sampled history input and once as a storage write target — and the pushed parity selects which of the
// two roles each image plays this frame. Binding both roles up front means the descriptor set is written
// once per (re)allocation rather than once per frame, so no set is ever updated while a previous frame's
// command buffer might still be reading it.

// The current frame's unfiltered froxel source term.
[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture3D<float4> rawFroxel;
// Resolved cache, sampled: cacheA is the history when parity == 0, cacheB when parity == 1.
[[vk::binding(1, 0)]] public Sampler3D cacheA;
[[vk::binding(2, 0)]] public Sampler3D cacheB;
// Resolved cache, storage: writeA targets the image sampled as cacheB and writeB the image sampled as
// cacheA, so the pass always writes the half it is not reading.
[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture3D<float4> writeA;
[[vk::binding(4, 0)]] [format("rgba16f")] public RWTexture3D<float4> writeB;
