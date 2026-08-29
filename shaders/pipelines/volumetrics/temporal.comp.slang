// 3D Spatio-Temporal Filter with 3D YCoCg AABB Variance Clamping for anti-ghosting and boiling suppression.
import froxel_common;
import world_common;

[[vk::push_constant]] FroxelTemporalPush pc;

[[vk::binding(0, 0)]] public Texture3D<float4> froxelLight;
[[vk::binding(1, 0)]] public Sampler3D froxelHistoryIn;
[[vk::binding(2, 0)]] [format("rgba16f")] public RWTexture3D<float4> froxelFiltered;
[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture3D<float4> froxelHistoryOut;

[shader("compute")]
[numthreads(8, 8, 4)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    uint3 gridDims = pc.gridDims.xyz;
    if (any(dispatchId >= gridDims)) {
        return;
    }

    float4 currentSample = froxelLight[dispatchId];
    float3 currentRadiance = currentSample.rgb;
    float extinction = currentSample.a;

    bool resetHistory = (pc.gridDims.w != 0u);
    float temporalWeight = clamp(pc.params0.x, 0.0, 0.95);
    float varianceGamma = max(pc.params0.y, 0.5);
    float zNear = pc.params0.z;
    float zFar = pc.params0.w;

    // Reconstruct current camera-relative position (at center of voxel for filtering)
    float3 curCamRel = froxelToCamRelative(float3(dispatchId), float3(gridDims),
                                          pc.invViewProj, zNear, zFar, 0.5);

    // ============================================================================================
    // 1. Gather 3x3x3 Spatial Neighborhood for Variance Box Clamping
    // ============================================================================================
    float3 m1 = float3(0.0, 0.0, 0.0);
    float3 m2 = float3(0.0, 0.0, 0.0);
    float sampleCount = 0.0;

    int3 id = int3(dispatchId);
    int3 maxCoord = int3(gridDims) - 1;

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int3 neighborId = clamp(id + int3(dx, dy, dz), int3(0), maxCoord);
                float3 neighborRGB = froxelLight[neighborId].rgb;
                float3 neighborYCoCg = rgbToYCoCg(neighborRGB);

                m1 += neighborYCoCg;
                m2 += neighborYCoCg * neighborYCoCg;
                sampleCount += 1.0;
            }
        }
    }

    float3 mean = m1 / sampleCount;
    float3 variance = max(m2 / sampleCount - mean * mean, float3(0.0));
    float3 stdDev = sqrt(variance);

    float3 minBound = mean - varianceGamma * stdDev;
    float3 maxBound = mean + varianceGamma * stdDev;

    // ============================================================================================
    // 2. 3D Reprojection to Previous Frame
    // ============================================================================================
    float3 prevCamRel = curCamRel + pc.camDelta;
    float3 prevFroxelUVW = camRelativeToFroxel(prevCamRel, pc.prevViewProj,
                                              float3(gridDims), zNear, zFar);

    bool historyValid = !resetHistory && all(prevFroxelUVW >= float3(0.0)) && all(prevFroxelUVW <= float3(1.0));

    float3 blendedRadiance = currentRadiance;

    if (historyValid) {
        float3 historyRGB = froxelHistoryIn.SampleLevel(prevFroxelUVW, 0.0).rgb;
        float3 historyYCoCg = rgbToYCoCg(historyRGB);

        // Clamp history to current neighborhood AABB
        float3 clampedHistoryYCoCg = clamp(historyYCoCg, minBound, maxBound);
        float3 clampedHistoryRGB = ycocgToRgb(clampedHistoryYCoCg);

        // Exponential moving average blend
        blendedRadiance = lerp(currentRadiance, clampedHistoryRGB, temporalWeight);
    }

    // Write outputs
    froxelFiltered[dispatchId] = float4(blendedRadiance, extinction);
    froxelHistoryOut[dispatchId] = float4(blendedRadiance, extinction);
}
