// Froxel temporal + spatial filter. Reads this frame's raw single-scatter source term, blurs it 3x3 in
// the lateral axes, reprojects the previous frame's resolved cache onto this froxel, blends the two by
// accumulated sample count, and writes the resolved cache (rgb = source term, a = sample count) that the
// display-res march integrates.
//
// Temporal accumulation is what makes the volume clean by construction: two traced rays per froxel per
// frame is far too little to converge spatially, but a froxel's lighting changes slowly, so counting
// samples across frames converges it without a denoiser. The count is clamped (maxHistory) so camera
// motion cannot trail indefinitely, and a reprojection that leaves the previous frustum simply restarts
// the count at one sample.
//
// History is a ping-pong pair selected by the pushed parity: this pass reads one image and writes the
// other, so no invocation ever samples the texel it is overwriting. Extinction is deliberately not
// cached — the medium is analytic, so the march evaluates it exactly at its own sample points instead of
// interpolating a coarse grid, which keeps the composite's optical depth free of froxel quantisation.
import world_common;
import display_common;
import fog_common;
import filter_bindings;

[[vk::push_constant]] FogFilterPush pc;

static const float3 FOG_LUMA = float3(0.27222872, 0.67408177, 0.05368952);

float3 clampLuminance(float3 color, float ceiling) {
    float lum = dot(color, FOG_LUMA);
    return lum > ceiling ? color * (ceiling / lum) : color;
}

[numthreads(4, 4, 4)]
void main(uint3 id : SV_DispatchThreadID) {
    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    int2 grid = int2(pc.grid);
    float slices = max(worldPush.fogGrid.z, 1.0);
    // The dispatch rounds each axis up to the 4x4x4 group, so the tail lanes of a grid that is not a
    // multiple of four must not write past the cache extent.
    if (any(int3(id) >= int3(grid, int(slices)))) {
        return;
    }
    float2 uv = (float2(id.xy) + 0.5) / float2(grid);
    float u = (float(id.z) + 0.5) / slices;

    // Narrow cross blur with clamped taps: half this froxel's own value, an eighth from each lateral
    // neighbour. The sun term is deterministic, so the only salt this has to remove is the emitter RIS
    // sample; a wider tent would spend the beam edges the visibility rays and the finer froxel grid
    // worked for, and beam crispness is the whole point of the pass.
    float3 blurred = rawFroxel[int3(int2(id.xy), int(id.z))].rgb * 0.5;
    for (int axis = 0; axis < 4; axis++) {
        int2 offset = axis == 0 ? int2(1, 0) : axis == 1 ? int2(-1, 0)
                : axis == 2 ? int2(0, 1) : int2(0, -1);
        int2 tap = clamp(int2(id.xy) + offset, int2(0, 0), grid - int2(1, 1));
        blurred += rawFroxel[int3(tap, int(id.z))].rgb * 0.125;
    }
    blurred = clampLuminance(max(blurred, float3(0.0, 0.0, 0.0)), pc.fireflyClamp);

    // Reproject this froxel's centre into the previous frame. camDelta converts the current rebased
    // position into previous-frame camera-relative space, where the previous eye sits at the origin, so
    // its length is the previous ray distance and inverts straight back to a slice coordinate.
    float3 history = float3(0.0, 0.0, 0.0);
    float historySamples = 0.0;
    if (fogEnabled(worldPush)) {
        FogRay ray = fogCameraRay(worldPush, uv);
        float3 p = ray.origin + ray.dir * fogSliceDistance(worldPush, u);
        float3 prevRel = p - worldPush.camOffset + worldPush.camDelta;
        float4 prevClip = mul(worldPush.prevViewProj, float4(prevRel, 1.0));
        if (prevClip.w > 0.0) {
            float2 prevNdc = prevClip.xy / prevClip.w;
            if (all(abs(prevNdc) <= float2(1.0, 1.0))) {
                float3 prevUvw = float3(prevNdc * 0.5 + 0.5,
                                        fogSliceCoord(worldPush, length(prevRel)));
                float4 h = pc.parity == 0u ? cacheA.SampleLevel(prevUvw, 0.0)
                                           : cacheB.SampleLevel(prevUvw, 0.0);
                history = h.rgb;
                historySamples = h.a;
            }
        }
    }

    // Translation needs no fade: reprojection lands exactly on the same world point, and the cache is
    // a function of world position. Rotation is different — the cached source term baked the previous
    // frame's view direction into its phase evaluation — so the history's weight is faded by how far
    // the camera turned this frame, which is what keeps the volume feeling attached to the camera.
    historySamples *= clamp(pc.historyFade, 0.0, 1.0);
    float total = historySamples + 1.0;
    float4 resolved = float4((history * historySamples + blurred) / total, min(total, pc.maxHistory));
    if (pc.parity == 0u) {
        writeA[int3(id)] = resolved;
    } else {
        writeB[int3(id)] = resolved;
    }
}
