// Front-to-back integration of the camera-view fog fields. Direct shafts, local diagnostic emitters, GI,
// and world-space emitter reuse are filtered separately before arriving here; this pass applies the single
// extinction model and composites at the pre-DLSS-RR seam.
import world_common;
import fog_common;

[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] public Sampler2D depthImage;
[[vk::binding(2, 0)]] public Sampler2D directVolume;
[[vk::binding(3, 0)]] public Sampler2D localVolume;
[[vk::binding(4, 0)]] public Sampler2D giVolume;
[[vk::binding(5, 0)]] public Sampler2D cacheVolume;

[[vk::push_constant]] FogIntegratePush pc;

static const float FOG_NEAR_DISTANCE = 0.5;

float3 fogVolumeSample(Sampler2D volume, float2 uv, float z, float3 dimensions) {
    float gx = max(dimensions.x, 1.0);
    float gy = max(dimensions.y, 1.0);
    float gz = max(dimensions.z, 1.0);
    float zf = clamp(z, 0.0, 1.0) * (gz - 1.0);
    uint z0 = uint(floor(zf));
    uint z1 = min(z0 + 1u, uint(gz - 1.0));
    float zBlend = zf - float(z0);
    float2 atlasSize = float2(gx, gy * gz);
    // uv is normalized screen position. Clamp to the current froxel row before sampling: the atlas
    // packs depth slices vertically, so a linear lookup at the outer row edge would bleed the next
    // slice into this one and turn a depth boundary into apparent temporal noise.
    float2 localTexel = clamp(uv * float2(gx, gy), 0.5, float2(gx, gy) - 0.5);
    float2 uv0 = (localTexel + float2(0.0, float(z0) * gy)) / atlasSize;
    float2 uv1 = (localTexel + float2(0.0, float(z1) * gy)) / atlasSize;
    return lerp(volume.SampleLevel(uv0, 0.0).rgb,
            volume.SampleLevel(uv1, 0.0).rgb, zBlend);
}

float fogSliceDistance(float slice, float farDistance, float nearDistance) {
    return exp2(lerp(log2(nearDistance),
            log2(max(farDistance, nearDistance + 1.0)), slice));
}

float fogDensityAt(float3 position) {
    float exponent = clamp(-(position.y - pc.medium.z) * pc.medium.y, -4.0, 4.0);
    return max(pc.medium.x, 0.0) * exp(exponent);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    uint width;
    uint height;
    outputImage.GetDimensions(width, height);
    if (dispatchId.x >= width || dispatchId.y >= height) return;

    int2 pixel = int2(dispatchId.xy);
    float4 storedScene = outputImage[pixel];
    WorldPush push = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    float preExposure = max(push.preExposure, 1.0e-6);
    float3 scene = max(storedScene.rgb / preExposure, float3(0.0));
    float2 uv = (float2(dispatchId.xy) + 0.5) / float2(width, height);
    float depth = depthImage.SampleLevel(uv, 0.0).r;

    float2 ndc = uv * 2.0 - 1.0;
    float4 nearH = mul(push.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(push.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    float3 rayDirection = normalize(farP - nearP);
    float nearAlongRay = length(nearP);

    // Reconstruct the visible primary endpoint. A sky depth of zero keeps the far fog endpoint instead of
    // accidentally treating the sky as a surface at the near plane.
    float surfaceDistance = max(pc.grid.w, FOG_NEAR_DISTANCE);
    if (depth > 1.0e-6) {
        float4 surfaceH = mul(push.invViewProj, float4(ndc.x, ndc.y, depth, 1.0));
        if (surfaceH.w > 0.0) {
            float3 surfaceCameraRelative = surfaceH.xyz / surfaceH.w;
            surfaceDistance = max(nearAlongRay,
                    dot(surfaceCameraRelative - nearP, rayDirection) + nearAlongRay);
        }
    }
    surfaceDistance = clamp(surfaceDistance, FOG_NEAR_DISTANCE, max(pc.grid.w, FOG_NEAR_DISTANCE));

    float transmittance = 1.0;
    float3 inScatter = float3(0.0);
    float nearDistance = max(pc.medium.w, 0.01);
    uint slices = max(uint(pc.grid.z), 1u);
    for (uint slice = 0u; slice < slices; slice++) {
        float z0 = float(slice) / float(slices);
        float z1 = float(slice + 1u) / float(slices);
        float segmentStart = fogSliceDistance(z0, pc.grid.w, nearDistance);
        float segmentEnd = fogSliceDistance(z1, pc.grid.w, nearDistance);
        if (segmentStart >= surfaceDistance) break;
        float clippedEnd = min(segmentEnd, surfaceDistance);
        float segmentLength = clippedEnd - segmentStart;
        if (segmentLength <= 0.0) continue;

        float centerDistance = segmentStart + segmentLength * 0.5;
        float3 position = nearP + rayDirection * max(centerDistance - nearAlongRay, 0.0)
                + push.camOffset;
        float extinction = max(fogDensityAt(position), 0.0);
        float stepTransmittance = exp(-extinction * segmentLength);
        // For a constant-density froxel, the integral of sigma_s * exp(-sigma_t s) is exactly
        // scatterScale * (1 - exp(-sigma_t L)). This avoids the bright/dark bias of sigma_s * L for
        // the long logarithmic slices and remains well behaved as density approaches zero.
        float integratedScattering = max(pc.scatteringColor.w, 0.0)
                * (1.0 - stepTransmittance);
        float volumeZ = (log2(max(centerDistance, FOG_NEAR_DISTANCE)) - log2(FOG_NEAR_DISTANCE))
                / max(log2(max(pc.grid.w, FOG_NEAR_DISTANCE + 1.0)) - log2(FOG_NEAR_DISTANCE), 1.0e-5);
        float3 direct = fogVolumeSample(directVolume, uv, volumeZ, pc.grid.xyz)
                * FOG_RADIANCE_SCALE;
        float3 local = fogVolumeSample(localVolume, uv, volumeZ, pc.grid.xyz)
                * FOG_RADIANCE_SCALE;
        float3 indirect = fogVolumeSample(giVolume, uv, volumeZ, pc.grid.xyz)
                * FOG_RADIANCE_SCALE;
        float3 cache = fogVolumeSample(cacheVolume, uv, volumeZ, pc.grid.xyz)
                * FOG_RADIANCE_SCALE;
        float3 incident = direct * max(pc.lighting.x, 0.0)
                + indirect * max(pc.lighting.y, 0.0)
                + local * max(pc.lighting.z, 0.0)
                + cache * max(pc.lighting.w, 0.0);
        inScatter += transmittance * integratedScattering
                * pc.scatteringColor.rgb * incident;
        transmittance *= stepTransmittance;
        if (transmittance < 0.002) break;
    }

    float3 result = scene * transmittance + inScatter;
    outputImage[pixel] = float4(clamp(result * preExposure, float3(0.0), float3(HALF_MAX)), storedScene.a);
}
