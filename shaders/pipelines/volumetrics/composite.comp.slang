// Depth-aware composite of the 3D Froxel volume onto the scene color matching main's exact coordinate alignment and continuous sub-slice recovery.
import froxel_common;
import world_common;

[[vk::push_constant]] FroxelCompositePush pc;

[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture2D<float4> sceneImage;
[[vk::binding(1, 0)]] public Sampler3D froxelVolume;
[[vk::binding(2, 0)]] [format("r32f")] public RWTexture2D<float> depthImage;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    uint2 screenDims = pc.screenAndDepth.xy;
    uint2 depthDims = pc.screenAndDepth.zw;
    if (any(dispatchId.xy >= screenDims)) {
        return;
    }

    int2 pix = int2(dispatchId.xy);
    float2 uv = (float2(pix) + 0.5) / float2(screenDims);
    float2 ndc = uv * 2.0 - 1.0;

    float gridNz = float(pc.gridDims.z);
    float zNear = pc.params0.x;
    float zFar = pc.params0.y;
    float maxDistance = pc.params0.z;

    // Sample the exact primary surface depth for this pixel
    int2 depthPix = clamp(int2(uv * float2(depthDims)), int2(0), int2(depthDims) - 1);
    float hwDepth = depthImage[depthPix];

    float linearDepth = maxDistance;
    if (hwDepth > 1.0e-5) {
        float4 hitH = mul(pc.invViewProj, float4(ndc.x, ndc.y, hwDepth, 1.0));
        if (hitH.w > 0.0) {
            linearDepth = min(length(hitH.xyz / hitH.w), maxDistance);
        }
    }

    if (linearDepth <= 0.0) {
        return;
    }

    // Convert linear depth to continuous slice coordinate matching main
    float slice = linearDepthToSlice(linearDepth, zNear, zFar, gridNz);
    float3 volumeUVW = float3(uv, clamp(slice / gridNz, 0.0, 1.0));
    float4 integrated = froxelVolume.SampleLevel(volumeUVW, 0.0);

    // Exact sub-slice 0 constant-medium recovery from main
    if (slice < 1.0) {
        float fraction = saturate(slice);
        float fullTransmittance = clamp(integrated.a, 1.0e-6, 1.0);
        float partialTransmittance = pow(fullTransmittance, fraction);
        float fullOpacity = max(1.0 - fullTransmittance, 1.0e-6);
        integrated.rgb *= (1.0 - partialTransmittance) / fullOpacity;
        integrated.a = partialTransmittance;
    }

    float3 inscatter = max(integrated.rgb, float3(0.0));
    float transmittance = clamp(integrated.a, 0.0, 1.0);

    // Composite over pre-exposed scene radiance
    float4 scene = sceneImage[pix];
    float3 compositedRadiance = scene.rgb * transmittance + inscatter;

    sceneImage[pix] = float4(clamp(compositedRadiance, float3(0.0), float3(HALF_MAX)), scene.a);
}
