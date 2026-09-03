// Front-to-back camera integration of the shared froxel medium.
//
// directVolume and giVolume are already view-oriented lighting fields. This pass owns the only extinction
// model used by either field and writes the scene at the exact pre-DLSS-RR seam:
//     scene * transmittance + inScatter
// Sky pixels integrate to the fog far plane as well: the sky_view_lut is the background radiance at the
// far boundary, while this local camera medium belongs in front of it. That layers fog over the existing
// atmosphere without re-marching or double-counting the atmospheric LUT. Geometry pixels stop at their
// reconstructed primary surface.
import world_common;
import fog_common;

// These bindings are local to this compute entry point. Avoid a generic imported `bindings` module: the
// repository has one per pipeline and basename resolution can select the wrong descriptor ABI during
// reflection generation.
[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] public Sampler2D depthImage;
[[vk::binding(2, 0)]] public Sampler2D directVolume;
[[vk::binding(3, 0)]] public Sampler2D giVolume;

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
    // uv is already a normalized screen position. The atlas coordinate must preserve that position; a
    // second half-texel offset would make every pixel read the next froxel and turn the shaft boundaries
    // into visible stair-stepping.
    float2 localTexel = float2(uv.x * gx, uv.y * gy);
    float2 uv0 = (localTexel + float2(0.0, float(z0) * gy)) / atlasSize;
    float2 uv1 = (localTexel + float2(0.0, float(z1) * gy)) / atlasSize;
    return lerp(volume.SampleLevel(uv0, 0.0).rgb,
            volume.SampleLevel(uv1, 0.0).rgb, zBlend);
}

float fogSliceDistance(float slice, float farDistance) {
    return exp2(lerp(log2(FOG_NEAR_DISTANCE),
            log2(max(farDistance, FOG_NEAR_DISTANCE + 1.0)), slice));
}

float fogDensityAt(float3 position) {
    // Exponential height fog remains stable under the terrain rebase because base height is supplied in the
    // same rebased coordinate system as the ray-traced world position. Clamp the exponent so a malformed
    // dimension or a very deep cave cannot overflow and poison the HDR target.
    float exponent = clamp(-(position.y - pc.medium.z) * pc.medium.y, -4.0, 4.0);
    return max(pc.medium.x, 0.0) * exp(exponent);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    uint width;
    uint height;
    outputImage.GetDimensions(width, height);
    if (dispatchId.x >= width || dispatchId.y >= height) {
        return;
    }

    int2 pixel = int2(dispatchId.xy);
    float4 storedScene = outputImage[pixel];
    float preExposure = max(ConstPtr<WorldPush>(pc.worldPushAddr)[0].preExposure, 1.0e-6);
    float3 scene = max(storedScene.rgb / preExposure, float3(0.0));
    float depth = depthImage.SampleLevel((float2(dispatchId.xy) + 0.5) / float2(width, height), 0.0).r;

    // gDepth is reversed-Z hardware depth. A zero depth is the sky miss written by the primary guide path;
    // it remains a valid background endpoint, so sky rays integrate the local medium all the way to the fog
    // far plane instead of silently losing shafts against the sky.
    WorldPush push = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    float2 uv = (float2(dispatchId.xy) + 0.5) / float2(width, height);
    float2 ndc = uv * 2.0 - 1.0;
    float4 nearH = mul(push.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(push.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    float3 rayDirection = normalize(farP - nearP);
    float nearAlongRay = length(nearP);

    // Reconstruct the primary surface point at the exact guide depth. This avoids fogging behind the
    // visible surface, which is both wasteful and the usual source of halos around silhouettes.
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
    uint slices = max(uint(pc.grid.z), 1u);
    for (uint slice = 0u; slice < slices; slice++) {
        float z0 = float(slice) / float(slices);
        float z1 = float(slice + 1u) / float(slices);
        float segmentStart = fogSliceDistance(z0, pc.grid.w);
        float segmentEnd = fogSliceDistance(z1, pc.grid.w);
        if (segmentStart >= surfaceDistance) {
            break;
        }
        float clippedEnd = min(segmentEnd, surfaceDistance);
        float segmentLength = clippedEnd - segmentStart;
        if (segmentLength <= 0.0) {
            continue;
        }

        float centerDistance = segmentStart + segmentLength * 0.5;
        float3 position = nearP + rayDirection * max(centerDistance - nearAlongRay, 0.0)
                + push.camOffset;
        float extinction = max(fogDensityAt(position), 0.0);
        float scattering = extinction * max(pc.lighting.z, 0.0);
        float stepTransmittance = exp(-extinction * segmentLength);
        float volumeZ = (log2(max(centerDistance, FOG_NEAR_DISTANCE)) - log2(FOG_NEAR_DISTANCE))
                / max(log2(max(pc.grid.w, FOG_NEAR_DISTANCE + 1.0)) - log2(FOG_NEAR_DISTANCE), 1.0e-5);
        float3 volumeUv = float3(uv, volumeZ);
        float3 direct = fogVolumeSample(directVolume, volumeUv.xy, volumeUv.z, pc.grid.xyz)
                * FOG_RADIANCE_SCALE;
        float3 indirect = fogVolumeSample(giVolume, volumeUv.xy, volumeUv.z, pc.grid.xyz)
                * FOG_RADIANCE_SCALE;
        float3 incident = direct * max(pc.lighting.x, 0.0)
                + indirect * max(pc.lighting.y, 0.0);
        inScatter += transmittance * scattering * pc.scatteringColor.rgb
                * incident * segmentLength;
        transmittance *= stepTransmittance;
        if (transmittance < 0.002) {
            break;
        }
    }

    float3 result = scene * transmittance + inScatter;
    outputImage[pixel] = float4(clamp(result * preExposure, float3(0.0), float3(HALF_MAX)), storedScene.a);
}
