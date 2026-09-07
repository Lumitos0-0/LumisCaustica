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

        // 3D spatial filter across (x +- 1, y +- 1, z +- 1) to eliminate high-frequency voxel stepping
        float3 radiance = float3(0.0);
        float totalWeight = 0.0;

        for (int oz = -1; oz <= 1; oz++) {
            int sz = clamp(z + oz, 0, FROXEL_DIMS.z - 1);
            float wz = (oz == 0) ? 0.6 : 0.2;
            for (int ox = -1; ox <= 1; ox++) {
                for (int oy = -1; oy <= 1; oy++) {
                    int3 sampleCoord = int3(clamp(int(dispatchId.x) + ox, 0, FROXEL_DIMS.x - 1),
                                            clamp(int(dispatchId.y) + oy, 0, FROXEL_DIMS.y - 1),
                                            sz);
                    float wxy = (ox == 0 && oy == 0) ? 0.5 : ((ox == 0 || oy == 0) ? 0.09375 : 0.03125);
                    float weight = wz * wxy;
                    radiance += inFroxelRadiance[sampleCoord].rgb * weight;
                    totalWeight += weight;
                }
            }
        }
        radiance /= max(totalWeight, 1.0e-5);

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
