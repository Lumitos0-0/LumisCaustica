// Depth-aware cross-bilateral composite of the 3D Froxel volume onto the scene color with silhouette halo suppression.
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

    float sliceZ = linearDepthToSlice(linearDepth, zNear, zFar, gridNz);
    float normSlice = clamp((sliceZ - 0.5) / gridNz, 0.0, 1.0);

    // ============================================================================================
    // Depth-Aware Cross-Bilateral 2x2 Reconstruction (Eliminates Silhouette Halos)
    // ============================================================================================
    float2 froxelCoord = uv * float2(pc.gridDims.xy) - 0.5;
    int2 baseFroxel = int2(floor(froxelCoord));
    float2 fracFroxel = froxelCoord - float2(baseFroxel);

    float4 accumSample = float4(0.0);
    float totalWeight = 0.0;

    const int2 offsets[4] = {
        int2(0, 0),
        int2(1, 0),
        int2(0, 1),
        int2(1, 1)
    };

    for (int i = 0; i < 4; i++) {
        int2 tapCoord = clamp(baseFroxel + offsets[i], int2(0), int2(pc.gridDims.xy) - 1);
        float2 tapUV = (float2(tapCoord) + 0.5) / float2(pc.gridDims.xy);

        // Bilinear spatial weight
        float wBilinear = ((offsets[i].x == 0) ? (1.0 - fracFroxel.x) : fracFroxel.x)
                        * ((offsets[i].y == 0) ? (1.0 - fracFroxel.y) : fracFroxel.y);

        // Tap depth lookup to detect geometric silhouettes
        int2 tapDepthPix = clamp(int2(tapUV * float2(depthDims)), int2(0), int2(depthDims) - 1);
        float tapHwDepth = depthImage[tapDepthPix];

        float tapLinearDepth = maxDistance;
        if (tapHwDepth > 1.0e-5) {
            float2 tapNdc = tapUV * 2.0 - 1.0;
            float4 tapHitH = mul(pc.invViewProj, float4(tapNdc.x, tapNdc.y, tapHwDepth, 1.0));
            if (tapHitH.w > 0.0) {
                tapLinearDepth = min(length(tapHitH.xyz / tapHitH.w), maxDistance);
            }
        }

        // Cross-bilateral depth edge rejection
        float depthDiff = abs(linearDepth - tapLinearDepth) / max(min(linearDepth, tapLinearDepth) * 0.15, 0.25);
        float wDepth = exp(-depthDiff * depthDiff);
        float w = max(wBilinear * wDepth, 1.0e-4);

        float4 tapSample = froxelVolume.SampleLevel(float3(tapUV, normSlice), 0.0);
        accumSample += tapSample * w;
        totalWeight += w;
    }

    if (totalWeight > 1.0e-5) {
        accumSample /= totalWeight;
    } else {
        accumSample = froxelVolume.SampleLevel(float3(uv, normSlice), 0.0);
    }

    // ============================================================================================
    // Exact Exponential Beer-Lambert Sub-Slice 0 Transmittance Recovery (from main)
    // ============================================================================================
    if (sliceZ < 1.0) {
        float fraction = saturate(sliceZ);
        float fullTransmittance = clamp(accumSample.a, 1.0e-6, 1.0);
        float partialTransmittance = pow(fullTransmittance, fraction);
        float fullOpacity = max(1.0 - fullTransmittance, 1.0e-6);
        accumSample.rgb *= (1.0 - partialTransmittance) / fullOpacity;
        accumSample.a = partialTransmittance;
    }

    float3 inscatter = accumSample.rgb;
    float transmittance = clamp(accumSample.a, 0.0, 1.0);

    // Composite over pre-exposed scene radiance
    float4 scene = sceneImage[pix];
    float3 compositedRadiance = scene.rgb * transmittance + inscatter;

    sceneImage[pix] = float4(clamp(compositedRadiance, float3(0.0), float3(HALF_MAX)), scene.a);
}
