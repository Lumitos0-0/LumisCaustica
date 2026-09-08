// Froxel grid: the low-frequency half of the fog. One thread per froxel, one sky sample per froxel per
// frame, temporally accumulated over up to `maxAccumFrames` frames.
//
// This pass owns everything about the fog that does NOT need to be sharp: the sky's contribution and,
// once it lands, light from block emitters. The sun and moon are deliberately absent — a grid this
// coarse would blur a sun shaft to several times the width of its real penumbra, which is exactly the
// soft, uniformly-smeared look this whole design exists to avoid. Sharpness lives in shaft.comp.
//
// The ambient estimator is one sample of the phase function per froxel. Drawing the direction from
// Henyey-Greenstein itself makes phase/pdf identically one, so the whole in-scatter integral collapses
// to sigma_s * skyRadiance(omega) * visibility(omega). One texture fetch and one ray, and the
// 128-frame accumulation below is what turns that single sample into a converged answer.
import world_common;
import sky;
import volumetrics;
import volumetrics_common;
import volume_trace;
import bindings;

[[vk::push_constant]] VolumePush pc;

// Ceiling on a single froxel's accumulated in-scatter, in cd/m^2. A froxel that happens to sample a
// direction straight at the horizon under a low sun can be orders of magnitude brighter than its
// neighbours; without this it survives the accumulation as a permanent bright cell.
static const float FIREFLY_LUMINANCE_CLAMP = 1000.0;

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint3 dims = uint3(pc.gridX, pc.gridY, pc.gridZ);
    if (any(tid >= dims)) {
        return;
    }
    WorldPush push = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    VolumetricMedium m = push.fog;
    SkyState sky = skyState(push);

    // Jitter uniformly inside the froxel cell, in all three axes. Jittering only along the view ray
    // (as the original froxel-fog papers do, having no better option with a shadow map) leaves the
    // lateral cell boundaries fixed, and those are exactly the edges that show as a grid once anything
    // spatially varying — an occluder, a wall — passes through the volume.
    uint seed = volumeSeed(tid.xy, push.frameIndex, tid.z + 1u);
    float3 jitter = float3(volumeRandom(seed), volumeRandom(seed), volumeRandom(seed));
    float3 uvw = (float3(tid) + jitter) / float3(dims);

    float3 origin;
    float3 dir;
    volumeCameraRay(push.invViewProj, uvw.xy, origin, dir);
    float t = volumeSliceDistance(m, uvw.z);
    float3 cameraRelative = origin + dir * t;
    float density = volumeDensity(m, cameraRelative.y, push.camOffset.y);

    // Phase-importance-sampled sky direction, then the sky radiance arriving from it.
    float3 omega = volumeSampleHg(m.anisotropy, dir, volumeRandom2(seed));
    float3 radiance = volumeSkyRadiance(skyViewLut, sky, omega);

    // Sky visibility. Without it the fog inside a cave is lit by the full open sky and the cave fills
    // with glowing soup; this single ray, accumulated, is the difference between fog and that.
    if (pc.skyOcclusion != 0u && any(radiance > float3(0.0, 0.0, 0.0))) {
        if (!volumeVisible(cameraRelative + push.camOffset, omega, VOLUME_SKY_OCCLUSION_DISTANCE)) {
            radiance = float3(0.0, 0.0, 0.0);
        }
    }

    float3 inScatter = volumeScattering(m) * density * radiance * m.ambientIntensity;
    float luminance = dot(inScatter, float3(0.2126, 0.7152, 0.0722));
    if (luminance > FIREFLY_LUMINANCE_CLAMP) {
        inScatter *= FIREFLY_LUMINANCE_CLAMP / luminance;
    }

    // Reproject this froxel's centre into the previous frame's volume. camDelta turns
    // current-camera-relative into previous-camera-relative, so this survives a terrain rebase (both
    // spaces are anchored on the camera, and a rebase only moves the world origin).
    //
    // The previous slice coordinate is derived from the distance to the camera-relative ORIGIN rather
    // than to the previous frame's near-plane point, which is not recoverable without the previous
    // inverse projection. The two differ by the near distance, a few centimetres, which is below one
    // slice everywhere except the first — and the first slice is a few centimetres deep.
    float3 accumulated = inScatter;
    float frames = 1.0;
    if (pc.historyValid != 0u) {
        float3 reprojected = volumeReprojectUv(push, cameraRelative);
        float prevW = volumeSliceCoord(m, length(cameraRelative + push.camDelta));
        if (reprojected.z > 0.0 && prevW > 0.0 && prevW < 1.0) {
            float4 history = volumeAccumPrev.SampleLevel(float3(reprojected.xy, prevW), 0.0);
            frames = min(history.a + 1.0, float(max(pc.maxAccumFrames, 1u)));
            accumulated = lerp(history.rgb, inScatter, 1.0 / max(frames, 1.0));
        }
    }

    volumeAccumImage[tid] = float4(accumulated, frames);
}
