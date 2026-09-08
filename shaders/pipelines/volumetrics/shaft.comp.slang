// Sun and moon shafts, at render resolution, one sample per pixel. This is the pass the whole design is
// arranged around, so the reasoning behind each choice matters more than usual.
//
// WHY NOT IN THE FROXEL GRID. The sun subtends about 0.53 degrees, so a shaft edge cast by an occluder
// at distance d_o and seen at distance d_s has an angular penumbra of 2*theta_sun*(1 - d_o/d_s) — zero
// where the beam meets its occluder, widening to roughly eight display pixels at 1080p far behind it. A
// froxel grid at renderRes/8 blurs everything by twelve display pixels UNIFORMLY, which is both wider
// than the real penumbra and wrong in shape: near-camera shafts, which should be crisp, get the same
// smear as distant ones. Screen space costs one ray per pixel and reproduces the real, distance-
// dependent softness for free.
//
// WHY ONE SAMPLE IS ENOUGH. Every factor in the scattering integral except visibility is smooth and
// analytic here: transmittance is importance-sampled exactly, the phase function is a scalar, and the
// light's colour comes from the transmittance LUT. The only variance is binary occlusion, and
// Minecraft's sun moves 0.005 degrees per frame at 60fps — a full day is twenty real minutes — so
// unlike a game with swinging flashlights this signal is effectively static and tolerates the long
// accumulation that resolve.comp applies.
//
// WHY THE DISC IS SAMPLED. Aiming at the sun's centre gives a binary edge that no amount of filtering
// turns into a believable beam. Sampling the disc is what produces the penumbra described above, and it
// costs one sine and one cosine.
//
// THE DISTANCE SAMPLER. The medium's density varies with height but its colour does not, so the optical
// depth along a ray is closed-form and invertible. Sampling uniformly in optical depth rather than in
// distance makes the density term cancel out of the estimator entirely: the whole integral becomes
// Tr(tau) * sigma_s * phase * L * V(tau) / pdf(tau). That is exact for this medium, with no ray-march
// step count to trade against.
import world_common;
import sky;
import volumetrics;
import volumetrics_common;
import volume_trace;
import bindings;

[[vk::push_constant]] VolumePush pc;

// How far a shaft's occlusion ray looks. Longer than the fog itself, because the terrain that shadows a
// beam is often well outside the fog's own range — a hill on the horizon still darkens the valley. Only
// fully unoccluded rays pay the whole traversal, and those leave the loaded terrain quickly.
static const float SHAFT_RAY_LENGTH = 512.0;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 pixel = tid.xy;
    if (pixel.x >= pc.renderW || pixel.y >= pc.renderH) {
        return;
    }
    WorldPush push = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    VolumetricMedium m = push.fog;
    SkyState sky = skyState(push);

    // Reconstruct exactly the ray the path tracer traced for this pixel, jitter included. Matching it
    // is what lets the composite treat this buffer and gDepth as describing the same sample, and the
    // jitter doubles as free spatial supersampling of the shaft across frames.
    float2 size = float2(float(pc.renderW), float(pc.renderH));
    float2 uv = (float2(pixel) + 0.5) / size + push.jitter / size;
    float3 origin;
    float3 dir;
    volumeCameraRay(push.invViewProj, uv, origin, dir);

    float surfaceT = volumeSurfaceDistance(push.invViewProj, uv, depthImage[pixel], m.maxDistance);
    float segment = volumeSanitizeScalar(min(surfaceT, m.maxDistance), m.maxDistance);

    float originDensity = volumeDensity(m, origin.y, push.camOffset.y);
    float sigmaBar = volumeMeanExtinction(m);
    // A ray climbing out of an exponential fog layer reaches a finite total optical depth; stay just
    // inside it so the inverse mapping below never divides through zero.
    float tauLimit = volumeMaxOpticalDepth(m, originDensity, dir.y) * 0.999;
    float tauMax = min(volumeOpticalDepth(m, originDensity, dir.y, segment), tauLimit);

    // Deterministic scattering centroid. Stored rather than the sampled distance because resolve.comp
    // reprojects on it, and an anchor drawn from the same one-sample estimator it exists to stabilise
    // would be as noisy as the signal itself.
    float meanDistance = volumeDistanceForOpticalDepth(m, originDensity, dir.y,
            volumeMeanOpticalDepth(sigmaBar, tauMax));
    meanDistance = volumeSanitizeScalar(min(meanDistance, segment), m.maxDistance);

    CelestialLight light = dominantCelestialLight(sky, transmittanceLut);
    if (tauMax <= 1.0e-7 || !any(light.illuminance > float3(0.0, 0.0, 0.0))) {
        shaftRawImage[pixel] = float4(0.0, 0.0, 0.0, meanDistance);
        return;
    }

    // Truncated-exponential sample in optical depth. The truncation matters: without it a short segment
    // (a nearby wall) would mostly draw samples past its own end and throw them away.
    uint seed = volumeSeed(pixel, push.frameIndex, 7u);
    float range = 1.0 - exp(-sigmaBar * tauMax);
    float tau = -log(max(1.0 - volumeRandom(seed) * range, 1.0e-8)) / sigmaBar;
    // The pdf floor bounds the estimator's weight. Far out along the segment exp(-sigmaBar*tau)
    // underflows, and the transmittance factor in the numerator that would otherwise cancel it is
    // evaluated per channel, so the two do not vanish together.
    float pdf = max(sigmaBar * exp(-sigmaBar * tau) / max(range, 1.0e-8), 1.0e-12);
    float t = min(volumeDistanceForOpticalDepth(m, originDensity, dir.y, tau), segment);
    float3 cameraRelative = origin + dir * t;

    // Uniform over the solar/lunar disc. Because the body's radiance is uniform across it, sampling the
    // cone uniformly leaves the estimator's weight unchanged — the illuminance already integrates the
    // solid angle — so this only decides WHERE visibility is evaluated. That is precisely the penumbra.
    float3 toLight = volumeSampleCone(light.dir, light.halfAngle, volumeRandom2(seed));
    float visibility = volumeVisible(cameraRelative + push.camOffset, toLight, SHAFT_RAY_LENGTH)
            ? 1.0 : 0.0;

    float3 inScatter = float3(0.0, 0.0, 0.0);
    if (visibility > 0.0) {
        // cos(theta) between the light's arrival direction and the view ray, in the convention
        // volumeHgPhase is written for: g > 0 brightens the halo when looking toward the body.
        float phase = volumeHgPhase(m.anisotropy, dot(toLight, dir));
        // Clamped before it is stored: this buffer is rgba16f and the value is absolute radiance, so
        // the phase function's forward peak under a strong anisotropy setting reaches half-float's
        // limit, and the temporal resolve downstream would keep the resulting infinity alive forever.
        inScatter = volumeSanitize(volumeScattering(m) * light.illuminance * phase
                * volumeTransmittance(m, tau) * m.shaftIntensity / pdf);
    }

    shaftRawImage[pixel] = float4(inScatter, volumeSanitizeScalar(meanDistance, m.maxDistance));
}
