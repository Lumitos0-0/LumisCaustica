// Fog descriptor set, shared by the accumulate + apply passes. Both import this file, so the set
// layout is identical by construction; the binding generator validates apply.comp's reflection
// against its spec (accumulate.comp is covered by the shared import).
//
// Volume slots: FRESH (fogVolume) holds this frame's jittered bake, written by world/fog.rgen.
// A/B hold the temporally accumulated volumes, ping-ponged by frame parity (even frames write
// A, odd frames write B). The read-only and read-write bindings of each volume share one view.
[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] [format("r32f")] public RWTexture2D<float> depthImage;
// Fresh jittered bake (this frame). Sampled nearest: accumulate needs exact texels.
[[vk::binding(2, 0)]] public Sampler3D freshVol;
// Accumulated volumes, ping-ponged by frame parity. Sampled linear: history reprojects to
// arbitrary sub-voxel positions, so filtering reconstructs instead of nearest-snapping.
[[vk::binding(3, 0)]] public Sampler3D volA;
[[vk::binding(4, 0)]] public Sampler3D volB;
[[vk::binding(5, 0)]] [format("rgba16f")] public RWTexture3D<float4> volARw;
[[vk::binding(6, 0)]] [format("rgba16f")] public RWTexture3D<float4> volBRw;
// Per-voxel history age (frames accumulated, 0..31), same ping-pong. Sampled nearest: ages
// are per-voxel counters, never interpolated. R16F holds small integers exactly.
[[vk::binding(7, 0)]] public Sampler3D ageARo;
[[vk::binding(8, 0)]] public Sampler3D ageBRo;
[[vk::binding(9, 0)]] [format("r16f")] public RWTexture3D<float> ageARw;
[[vk::binding(10, 0)]] [format("r16f")] public RWTexture3D<float> ageBRw;
// Previous-frame hardware depth, written by the apply pass. Lets accumulate reject voxels
// whose reprojection moved behind (or in front of) different occluders. Sampled nearest:
// the voxel's own texel, not a blend.
[[vk::binding(11, 0)]] public Sampler2D prevDepthRo;
[[vk::binding(12, 0)]] [format("r32f")] public RWTexture2D<float> prevDepthRw;
