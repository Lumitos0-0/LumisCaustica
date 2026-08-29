// Pre-integrates in-scattered radiance and transmittance along Z-slices using exact exponential integration.
import froxel_common;
import world_common;

[[vk::push_constant]] FroxelIntegratePush pc;

[[vk::binding(0, 0)]] public Texture3D<float4> froxelFiltered;
[[vk::binding(1, 0)]] public Texture3D<float4> froxelMedia;
[[vk::binding(2, 0)]] [format("rgba16f")] public RWTexture3D<float4> froxelVolume;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    uint3 gridDims = pc.gridDims.xyz;
    if (dispatchId.x >= gridDims.x || dispatchId.y >= gridDims.y) {
        return;
    }

    float zNear = pc.params0.x;
    float zFar = pc.params0.y;

    float3 accumRadiance = float3(0.0, 0.0, 0.0);
    float accumTransmittance = 1.0;

    // Front-to-back integration along the view ray through all Z slices
    for (uint z = 0u; z < gridDims.z; z++) {
        uint3 coord = uint3(dispatchId.xy, z);

        float3 inscatter = froxelFiltered[coord].rgb;
        float extinction = froxelMedia[coord].a;

        float stepThickness = computeSliceThickness(z, zNear, zFar, float(gridDims.z));

        // Exact exponential integration of the Radiative Transfer Equation (RTE)
        float stepTransmittance = exp(-extinction * stepThickness);
        float3 stepInscatter;

        if (extinction > 1.0e-6) {
            stepInscatter = inscatter * ((1.0 - stepTransmittance) / extinction);
        } else {
            stepInscatter = inscatter * stepThickness;
        }

        accumRadiance += stepInscatter * accumTransmittance;
        accumTransmittance *= stepTransmittance;

        // Store pre-integrated volume: RGB = accumulated in-scatter, A = accumulated transmittance
        froxelVolume[coord] = float4(accumRadiance, accumTransmittance);
    }
}
