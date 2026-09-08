// Composite: applies the fog to DLSS-RR's display-resolution output, in place.
//
// WHY HERE. DLSS-RR is guided by surface depth, normal, albedo and motion. In-scatter has no surface,
// so handing it to RR means having the beams reprojected by a vector field that does not describe them
// — they smear and ghost, which is the one outcome this design is not allowed to produce. The
// volumetrics therefore denoise themselves (resolve.comp, filter.comp) and are added afterwards.
//
// It runs BEFORE the exposure histogram and bloom, both of which read this same image, and that is
// deliberate: a foggy scene genuinely is brighter, so auto-exposure should see it, and a sun shaft
// should feed the bloom pyramid the way any other bright feature does.
//
// UNITS. The trace writes radiance already scaled by WorldPush.preExposure and the display pass divides
// it back out, so everything added here — computed in absolute cd/m^2 — has to be scaled the same way.
// Transmittance is a ratio and needs no such correction.
import world_common;
import volumetrics;
import volumetrics_common;
import bindings;

[[vk::push_constant]] VolumePush pc;

/** Distance to the surface behind a render-resolution pixel, in blocks. */
float tapDistance(WorldPush push, VolumetricMedium m, int2 renderPixel, float2 renderSize) {
    uint2 clamped = uint2(clamp(renderPixel, int2(0, 0), int2(renderSize) - 1));
    // The render pixel's ray carried the frame's jitter; undo it so this describes the same screen
    // position the display pixel does.
    float2 uv = (float2(clamped) + 0.5 + push.jitter) / renderSize;
    return min(volumeSurfaceDistance(push.invViewProj, uv, depthImage[clamped], m.maxDistance),
            m.maxDistance);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 pixel = tid.xy;
    if (pixel.x >= pc.displayW || pixel.y >= pc.displayH) {
        return;
    }
    WorldPush push = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    VolumetricMedium m = push.fog;

    float2 displaySize = float2(float(pc.displayW), float(pc.displayH));
    float2 renderSize = float2(float(pc.renderW), float(pc.renderH));
    float2 uv = (float2(pixel) + 0.5) / displaySize;

    float3 origin;
    float3 dir;
    volumeCameraRay(push.invViewProj, uv, origin, dir);

    // Render-resolution footprint of this display pixel. The shaft buffer was written on jittered rays,
    // so the value stored at render index p describes screen position (p + 0.5 + jitter) / renderSize;
    // inverting that is what keeps the beams from wobbling half a render pixel every frame.
    float2 renderCoord = uv * renderSize - push.jitter - 0.5;
    int2 base = int2(floor(renderCoord));
    float2 frac = renderCoord - float2(base);

    float distances[4];
    distances[0] = tapDistance(push, m, base + int2(0, 0), renderSize);
    distances[1] = tapDistance(push, m, base + int2(1, 0), renderSize);
    distances[2] = tapDistance(push, m, base + int2(0, 1), renderSize);
    distances[3] = tapDistance(push, m, base + int2(1, 1), renderSize);

    // Foreground bias. DLSS-RR resolves sub-pixel geometry that the render-resolution depth cannot
    // describe, so at a silhouette one of these four taps is always wrong. Taking the nearest surface
    // makes the residual one-pixel fringe show too LITTLE fog rather than too much, which is by far the
    // less visible of the two errors.
    float reference = min(min(distances[0], distances[1]), min(distances[2], distances[3]));
    float tolerance = max(0.05 * reference, 0.5);

    float bilinear[4];
    bilinear[0] = (1.0 - frac.x) * (1.0 - frac.y);
    bilinear[1] = frac.x * (1.0 - frac.y);
    bilinear[2] = (1.0 - frac.x) * frac.y;
    bilinear[3] = frac.x * frac.y;

    int2 offsets[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };
    float3 shaft = float3(0.0, 0.0, 0.0);
    float shaftWeight = 0.0;
    [unroll]
    for (int i = 0; i < 4; i++) {
        float weight = bilinear[i] * exp(-abs(distances[i] - reference) / tolerance);
        uint2 tap = uint2(clamp(base + offsets[i], int2(0, 0), int2(renderSize) - 1));
        shaft += volumeSanitize(shaftFilteredImage[tap].rgb) * weight;
        shaftWeight += weight;
    }
    shaft = shaftWeight > 0.0 ? shaft / shaftWeight : float3(0.0, 0.0, 0.0);

    // Ambient in-scatter: one trilinear fetch of the prefix-integrated froxel volume at this pixel's
    // screen UV and its surface's slice coordinate.
    //
    // The half-slice offset is not a fudge. Texel z of that volume holds the integral out to the FAR
    // edge of slice z, while a trilinear fetch centres texel z at (z + 0.5) / gridZ — so reading at the
    // raw slice coordinate returns the integral to a point half a slice too far. In the near slices that
    // is centimetres; in the outermost ones, where the w^2 distribution has stretched a slice to ten
    // blocks or more, it is a visible thickening of distant fog.
    float slice = saturate(volumeSliceCoord(m, reference) - 0.5 / float(pc.gridZ));
    float3 ambient = volumeSanitize(volumeIntegratedLut.SampleLevel(float3(uv, slice), 0.0).rgb);

    // Transmittance is evaluated analytically per display pixel rather than read from the volume: it is
    // exact, it costs less than the fetch, and it carries no grid-resolution blur into the one term
    // that multiplies the whole rendered image.
    float originDensity = volumeDensity(m, origin.y, push.camOffset.y);
    float tau = volumeSanitizeScalar(volumeOpticalDepth(m, originDensity, dir.y, reference),
            VOLUME_MAX_OPTICAL_DEPTH);
    float3 transmittance = volumeTransmittance(m, tau);

    // In-scatter is absolute cd/m^2 and the scene is pre-exposed, so the added term is scaled to match.
    // At night the pre-exposure that centres a dim scene is large, and it multiplies a shaft computed in
    // absolute units, so the product is bounded here as well as at the source.
    float4 scene = sceneImage[pixel];
    float3 inScatter = volumeSanitizeTo((ambient + shaft) * push.preExposure, VOLUME_MAX_STORED);
    float3 result = scene.rgb * transmittance + inScatter;

    switch (pc.debugView) {
        case VOLUME_DEBUG_AMBIENT:
            result = ambient * push.preExposure;
            break;
        case VOLUME_DEBUG_SHAFT_RAW:
            result = shaftRawImage[uint2(clamp(base, int2(0, 0), int2(renderSize) - 1))].rgb
                    * push.preExposure;
            break;
        case VOLUME_DEBUG_SHAFT_FILTERED:
            result = shaft * push.preExposure;
            break;
        case VOLUME_DEBUG_TRANSMITTANCE:
            result = transmittance;
            break;
        case VOLUME_DEBUG_HISTORY:
            // Age normalised against the configured ceiling: black is a fresh disocclusion, white is a
            // fully converged pixel.
            result = float3(shaftAgeImage[uint2(clamp(base, int2(0, 0), int2(renderSize) - 1))]
                    / float(max(pc.maxAccumFrames, 1u)));
            break;
        default:
            break;
    }

    // Last gate before the image leaves this pass. Auto-exposure meters this same target and bloom
    // downsamples it, so a single non-finite pixel here does not stay a single pixel: it poisons the
    // frame's exposure and smears through the bloom pyramid. The ceiling is the storage format's, not a
    // fog-specific one, so nothing the scene already carried can be clipped by it.
    sceneImage[pixel] = float4(volumeSanitizeTo(result, VOLUME_MAX_STORED), scene.a);
}
