// Display-res fog march and composite. Integrates the resolved froxel cache along each camera ray from
// the eye to the scene depth and composites it over the denoised scene image:
//
//     fogged = scene * transmittance + inScatter
//
// Each slice segment uses the analytic constant-medium integral (S/sigmaT) * (1 - exp(-sigmaT * dt))
// rather than a midpoint sample, which is the same trick the sky march uses to stay banding-free at low
// step counts. The slice boundaries carry a per-pixel jitter so the residual discretisation becomes
// high-frequency noise in the slice index instead of visible banding planes; the cache's temporal
// accumulation integrates it away over frames.
//
// Runs after DLSS-RR (the cache is already temporally clean and needs no denoiser, and RR's surface
// guides must stay uncontaminated by volume content) and before exposure/bloom (so fog participates in
// auto-exposure and glows through bloom like any other scene radiance).
import world_common;
import display_common;
import fog_common;
import march_bindings;

[[vk::push_constant]] FogMarchPush pc;

float sliceJitter(uint2 pix) {
    uint h = pix.x * 1973u ^ pix.y * 9277u ^ pc.frameIndex * 2654435761u;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return float(h) / 4294967296.0;
}

[numthreads(8, 8, 1)]
void main(uint2 pix : SV_DispatchThreadID) {
    if (any(pix >= pc.size)) {
        return;
    }
    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    float2 uv = (float2(pix) + 0.5) / float2(pc.size);
    float4 scene = sceneImage[int2(pix)];
    if (pc.enabled == 0u || !fogEnabled(worldPush)) {
        foggedImage[int2(pix)] = float4(scene.rgb, 1.0);
        return;
    }
    FogRay ray = fogCameraRay(worldPush, uv);

    // Scene distance from the reversed-Z depth guide, nearest-sampled: bilinear depth would leak a
    // foreground surface's distance into a background pixel and clip its shafts short at every edge.
    int2 depthPix = int2(float2(pix) * float2(pc.depthSize) / float2(pc.size));
    float tScene = min(fogDistanceFromDepth(worldPush, ray, depthImage[depthPix]),
                       worldPush.fogGrid.x);

    float transmittance = 1.0;
    float3 inScatter = float3(0.0, 0.0, 0.0);
    float slices = max(worldPush.fogGrid.z, 1.0);
    float jitter = sliceJitter(pix);
    for (int s = 0; s < int(slices); s++) {
        float t0 = fogSliceDistance(worldPush, (float(s) + jitter) / slices);
        if (t0 >= tScene) {
            break;
        }
        float t1 = min(fogSliceDistance(worldPush, (float(s) + 1.0) / slices), tScene);
        float segment = t1 - t0;
        if (segment <= 0.0) {
            continue;
        }
        // Extinction comes from the analytic medium at this segment's midpoint rather than from the
        // cache, so optical depth is exact along the ray while only the (slowly varying, temporally
        // accumulated) lighting is interpolated from the coarse froxel grid.
        float sigmaT = fogDensityAt(worldPush, ray.origin + ray.dir * (0.5 * (t0 + t1)));
        if (sigmaT <= 1.0e-6) {
            continue;
        }
        float3 cacheUvw = float3(uv, (float(s) + 0.5) / slices);
        float3 source = pc.parity == 0u ? froxelB.SampleLevel(cacheUvw, 0.0).rgb
                                        : froxelA.SampleLevel(cacheUvw, 0.0).rgb;
        float segmentTransmittance = exp(-sigmaT * segment);
        inScatter += transmittance * (source / sigmaT) * (1.0 - segmentTransmittance);
        transmittance *= segmentTransmittance;
    }
    foggedImage[int2(pix)] = float4(clamp(scene.rgb * transmittance + inScatter,
                                    float3(0.0, 0.0, 0.0), float3(HALF_MAX, HALF_MAX, HALF_MAX)),
                              transmittance);
}
