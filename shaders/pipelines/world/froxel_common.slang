// Shared constants for the froxel volumetric fog. FROXEL_DEPTH is the number of camera-space depth
// slices in the frustum-aligned grid (the Z extent of the froxel 3D image); it must stay in lock-step
// with RtComposite.FROXEL_DEPTH, which sizes that image. Both the froxel light raygen and the raymarch
// compute use it so their depth parameterisation can never disagree.
module froxel_common;

public static const uint FROXEL_DEPTH = 64u;
