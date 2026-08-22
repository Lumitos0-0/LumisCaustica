// Volumetric fog descriptor layout. All volumetric passes share one descriptor set layout
// so that a single VkDescriptorSet is reused across inject/spatial/temporal/integrate dispatches.
// Camera matrices and per-frame medium parameters are NOT bound here — they live in a small
// persistently-mapped storage buffer accessed by raw device address pushed in
// VolumetricPush.matricesAddr, which keeps push constants under the 128-byte Vulkan minimum.

[[vk::binding(0, 0)]] public RaytracingAccelerationStructure tlas;

// Froxel radiance data, current frame (RGBA16F):
//   RGB = in-scattered source radiance S = sigma_s * incidentRadiance * phase
//         in renderer radiance units per block. This is temporally reconstructed.
//   A   = scalar extinction luminance sigma_t (1/blocks), used only for history validation
//         and debug. The physically integrated per-channel extinction lives in fogMedium*.
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture3D<float4> fogVolume;

// Current-frame injected radiance data (RGBA16F) before filtering/history.
// Used as input for spatial filter, and for neighborhood clamping in temporal.
[[vk::binding(2, 0)]] [format("rgba16f")] public RWTexture3D<float4> fogCurrent;

// History froxel radiance volume (RGBA16F): same semantic layout as fogVolume, written by previous frame.
[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture3D<float4> fogHistory;

// History confidence / age per froxel (R16F):
//   0 = no history, 1 = fully converged.
// READ-ONLY inside the temporal pass: updated confidence goes to fogConfidenceCurrent and Java
// copies fogConfidenceCurrent -> fogHistoryConfidence at frame end, so the temporal dispatch
// never reads and writes the same image (ping-pong, same discipline as fogCurrent vs. fogVolume).
[[vk::binding(4, 0)]] [format("r16f")] public RWTexture3D<float> fogHistoryConfidence;

// 2D scene depth (R32F, render-res): stores post-projection z/w, reversed-Z (1=near, 0=sky/far).
// Same buffer as gDepth in the world pipeline; accessed only via Load (no combined sampler needed).
[[vk::binding(5, 0)]] [format("r32f")] public RWTexture2D<float> depthImage;

// Render-res RT output image (RGBA16F, render-res): integrate.comp reads the pre-fog ray-traced
// colour and writes the composited scene*T + L result (overwriting in place), so fog is blended
// into the render target before DLSS-RR / upscale.
[[vk::binding(6, 0)]] [format("rgba16f")] public RWTexture2D<float4> outputImage;

// Current-frame confidence output (R16F), written by temporal.comp. Copied into
// fogHistoryConfidence by Java after integrate so it becomes next frame's history confidence.
[[vk::binding(7, 0)]] [format("r16f")] public RWTexture3D<float> fogConfidenceCurrent;

// Explicit current-frame participating-medium coefficients (RGBA16F), stable/world-anchored:
//   fogMediumScattering.rgb = sigma_s, scattering coefficient (1/blocks)
//   fogMediumScattering.a   = density, approximately 1/block before albedo splitting
//   fogMediumAbsorption.rgb = sigma_a, absorption coefficient (1/blocks)
//   fogMediumAbsorption.a   = direct-light visibility in [0,1] for debug only
// These are not temporally accumulated: medium coefficients are deterministic functions of world state.
[[vk::binding(8, 0)]] [format("rgba16f")] public RWTexture3D<float4> fogMediumScattering;
[[vk::binding(9, 0)]] [format("rgba16f")] public RWTexture3D<float4> fogMediumAbsorption;
