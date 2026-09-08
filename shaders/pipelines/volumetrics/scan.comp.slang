// Depth-slice scan: turns the per-froxel in-scatter the grid pass produced into the prefix integral the
// composite reads with one trilinear fetch. One thread per froxel COLUMN, walking outward in z.
//
// Two things here are load-bearing.
//
// The slice integral is the energy-conserving form (Hillaire, Frostbite 2015). Accumulating
// `scatter` directly and multiplying transmittance afterwards is off by the factor
// (1 - exp(-od))/od, which is not a small correction: with 64 slices distributed as w^2 out to 128
// blocks the outermost slice is over ten blocks deep, and the naive form makes distant fog darken as
// the slice count rises instead of converging.
//
// The optical depth is analytic rather than summed from per-froxel densities. The medium's colour is
// constant and only its density varies, so the density integral along the column's ray is closed-form
// (volumeOpticalDepth) and the per-slice delta is an exact difference of two evaluations. Summing
// midpoint densities instead would make transmittance depend on the slice count too.
import world_common;
import volumetrics;
import volumetrics_common;
import bindings;

[[vk::push_constant]] VolumePush pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.gridX || tid.y >= pc.gridY) {
        return;
    }
    WorldPush push = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    VolumetricMedium m = push.fog;

    // Column centre, unjittered: this is the reconstruction grid the composite interpolates on, and
    // the grid pass's per-frame jitter has already been averaged into each cell by the accumulation.
    float2 uv = (float2(tid.xy) + 0.5) / float2(float(pc.gridX), float(pc.gridY));
    float3 origin;
    float3 dir;
    volumeCameraRay(push.invViewProj, uv, origin, dir);
    float originDensity = volumeDensity(m, origin.y, push.camOffset.y);

    float3 accumulated = float3(0.0, 0.0, 0.0);
    float previousTau = 0.0;
    float previousT = 0.0;

    for (uint z = 0u; z < pc.gridZ; z++) {
        float farW = float(z + 1u) / float(pc.gridZ);
        float farT = volumeSliceDistance(m, farW);
        float tau = volumeOpticalDepth(m, originDensity, dir.y, farT);
        float deltaTau = max(tau - previousTau, 0.0);
        float sliceLength = max(farT - previousT, 0.0);

        float centreW = (float(z) + 0.5) / float(pc.gridZ);
        float3 scatter = volumeAccumCur.SampleLevel(float3(uv, centreW), 0.0).rgb;

        accumulated += volumeTransmittance(m, previousTau)
                * volumeSliceIntegral(m, scatter, deltaTau, sliceLength);

        volumeIntegratedImage[uint3(tid.xy, z)] = float4(accumulated, tau);

        previousTau = tau;
        previousT = farT;
    }
}
