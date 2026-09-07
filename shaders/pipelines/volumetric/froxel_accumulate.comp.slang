import accumulate_bindings;
import volumetric_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    if (dispatchId.x >= uint(FROXEL_DIMS.x) || dispatchId.y >= uint(FROXEL_DIMS.y)) {
        return;
    }

    float3 accumulatedInscatter = float3(0.0);
    float accumulatedTransmittance = 1.0;
    float prevDepth = 0.0;

    for (int z = 0; z < FROXEL_DIMS.z; z++) {
        int3 coord = int3(int(dispatchId.x), int(dispatchId.y), z);
        float sliceDepth = froxelSliceToDepth(float(z), FROXEL_DIMS.z);
        float stepDist = max(sliceDepth - prevDepth, 1.0e-4);
        prevDepth = sliceDepth;

        float3 radiance = inFroxelRadiance[coord].rgb;
        float extinction = max(inFroxelExtinction[coord].r, 0.0);

        float stepTransmittance = exp(-extinction * stepDist);
        float3 stepInscatter = extinction > 1.0e-6
                ? (radiance - radiance * stepTransmittance) / extinction
                : radiance * stepDist;

        accumulatedInscatter += accumulatedTransmittance * stepInscatter;
        accumulatedTransmittance *= stepTransmittance;

        outAccumulatedFroxels[coord] = float4(accumulatedInscatter, accumulatedTransmittance);
    }
}
