// Volumetric fog descriptor layout. All volumetric passes share one descriptor set layout
// so that a single VkDescriptorSet is reused across inject/spatial/temporal/integrate dispatches.

// Device pointer to a FroxelMatrices struct (invViewProj, curViewProj, prevViewProj).
// Matrices live in a persistently-mapped storage buffer so push constants stay under the
// 128-byte minimum-max-guarantee on every Vulkan device.
[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;

// Froxel data, current frame (RGBA16F):
//   RGB = in-scattered source radiance (after spatial filter / temporal accumulation)
//   A   = extinction coefficient sigma_t (1/blocks), scalar luminance
[[vk::binding(1, 0)]] [format("rgba16f")] RWTexture3D<float4> fogVolume;

// Current-frame injected froxel data (RGBA16F) before filtering/history.
// Used as input for spatial filter, and for neighborhood clamping in temporal.
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture3D<float4> fogCurrent;

// History froxel volume (RGBA16F): same semantic layout as fogVolume, written by previous frame.
[[vk::binding(3, 0)]] [format("rgba16f")] RWTexture3D<float4> fogHistory;

// History confidence / age per froxel (R16F):
//   0 = no history, 1 = fully converged.
[[vk::binding(4, 0)]] [format("r16f")] RWTexture3D<float> fogHistoryConfidence;

// 2D scene depth (R32F, render-res): stores post-projection z/w, reversed-Z (1=near, 0=sky/far).
// Same buffer as gDepth in the world pipeline; accessed only via Load (no combined sampler needed).
[[vk::binding(5, 0)]] [format("r32f")] RWTexture2D<float> depthImage;

// Render-res RT output image (RGBA16F, render-res): integrate.comp reads the pre-fog ray-traced
// colour and writes the composited scene*T + L result (overwriting in place), so fog is blended
// into the render target before DLSS-RR / upscale.
[[vk::binding(6, 0)]] [format("rgba16f")] RWTexture2D<float4> outputImage;
