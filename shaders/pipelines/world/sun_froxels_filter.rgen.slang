// Spatial filter for the aerial medium's light volume, launched right after sun_froxels.rgen and before the
// indirect trace. One thread per voxel, no rays: it reads the volume the gather wrote and writes the smoothed
// copy the march samples. A raygen rather than a compute pass for the same reason the gather is one -- this
// pipeline's stages are raygen/closest-hit/miss -- and the launch is again the volume itself, so a voxel is a
// thread with no index arithmetic.
//
// Its own record rather than a mode on the gather's push, because both launches share one push buffer per
// frame: the CPU cannot hand the same dispatch pair two different values before submission.
import world_common;
import world_core;
import bindings;
import fog;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    fogFilterSunFroxels(DispatchRaysIndex(), DispatchRaysDimensions());
}
