// One descriptor set serves all six volumetric passes: every pass declares the whole set and uses the
// subset it needs, the way the sky LUT bakes do. RtVolumetrics allocates TWO sets and alternates them
// per frame — the pairs (0,1), (6,7) and (8,9) are the same two images with their roles swapped, which
// is the ping-pong the temporal accumulation runs on.
//
// Nothing here is a plain sampled image without a sampler: the reprojected history fetches want
// hardware trilinear/bilinear, and everything else is read through the storage-image path with explicit
// indices, so no format needs to advertise SAMPLED_IMAGE_FILTER_LINEAR beyond rgba16f (which is
// mandatory).

// Froxel volume, this frame. rgb = temporally accumulated in-scatter per unit length, a = the froxel's
// density multiplier.
[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture3D<float4> volumeAccumImage;
// Same layout, previous frame, sampled trilinearly at the reprojected froxel centre.
[[vk::binding(1, 0)]] public Sampler3D volumeAccumPrev;
// Binding 0's image again, as a sampled view, so the depth-slice scan can read neighbouring froxels.
[[vk::binding(2, 0)]] public Sampler3D volumeAccumCur;
// Prefix-integrated volume. rgb = in-scatter accumulated from the camera to the far edge of the slice
// (already transmittance-weighted), a = accumulated optical depth in block-densities.
[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture3D<float4> volumeIntegratedImage;
// Binding 3's image as a sampled view; the composite's single trilinear fetch.
[[vk::binding(4, 0)]] public Sampler3D volumeIntegratedLut;

// Sun/moon shaft, render resolution. rgb = in-scatter, a = transmittance-weighted mean scattering
// distance in blocks (the temporal pass's reprojection anchor).
[[vk::binding(5, 0)]] [format("rgba16f")] public RWTexture2D<float4> shaftRawImage;
[[vk::binding(6, 0)]] [format("rgba16f")] public RWTexture2D<float4> shaftAccumImage;
[[vk::binding(7, 0)]] public Sampler2D shaftAccumPrev;
// Accumulated frame count, driving both the accumulation weight and the spatial filter's width.
[[vk::binding(8, 0)]] [format("r32f")] public RWTexture2D<float> shaftAgeImage;
[[vk::binding(9, 0)]] [format("r32f")] public RWTexture2D<float> shaftAgePrev;
[[vk::binding(10, 0)]] [format("rgba16f")] public RWTexture2D<float4> shaftFilteredImage;

// DLSS-RR's display-resolution output, composited in place.
[[vk::binding(11, 0)]] [format("rgba16f")] public RWTexture2D<float4> sceneImage;
// gDepth: reversed-Z NDC depth at render resolution, despite the linear-sounding name it is given at
// allocation. volumeSurfaceDistance unprojects it.
[[vk::binding(12, 0)]] [format("r32f")] public RWTexture2D<float> depthImage;

[[vk::binding(13, 0)]] public RaytracingAccelerationStructure tlas;
[[vk::binding(14, 0)]] public Sampler2D skyViewLut;
[[vk::binding(15, 0)]] public Sampler2D transmittanceLut;
