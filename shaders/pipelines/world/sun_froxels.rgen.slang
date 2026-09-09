// Sun and sky visibility for the aerial medium, gathered once per frame into a frustum-aligned volume.
//
// The fog march needs to know what light reaches a particular bit of air, and only a sample taken there can
// say so: a per-pixel test reports the lighting of the surface behind the air instead, which collapses a
// shaft onto the patch of ground it lands on. So this pass owns one texel per (screen column, log-depth
// slice) and traces the handful of rays that decide it, and the march reads the result per tap.
//
// It is a raygen record of the world pipeline rather than a compute pass because `TraceRay` is not legal in
// a compute stage under Vulkan, and because the pipeline's descriptor set, its push layout and the barrier
// that separates Pass A from Pass B already exist. The launch is three-dimensional, which is what makes a
// volume one dispatch rather than a loop: `DispatchRaysDimensions().z` is the slice count.
//
// The taps are fixed points, not random ones, and the volume is view-independent in world terms, so nothing
// here can flicker: the same air reports the same light whatever the sample budget or the frame index. A
// screen-anchored test cannot make that promise, because which surface a jittered primary ray strikes at a
// foliage edge changes every frame.
import world_common;
import world_core;
import bindings;
import fog;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    fogGatherSunFroxels(DispatchRaysIndex(), DispatchRaysDimensions());
}
