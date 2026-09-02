// Froxel volumetric fog column integration.
// Converts the resolved local froxel field (directional illumination in rgb, extinction in a) into a
// second 3D volume storing cumulative scattering.rgb and transmittance.a from the near plane through the
// FAR boundary of each slice. The final composite can then reconstruct a display pixel with a handful of
// depth-aware lookups instead of re-marching the whole view ray.

import world_common;
import volumetric_fog_common;

[[vk::push_constant]] FogIntegrationPush pc;

// Bindings - set 0
[[vk::binding(0, 0)]] Sampler3D froxelVolume;
[[vk::binding(1, 0)]] [format("rgba16f")] RWTexture3D<float4> integratedVolume;

static const float PI = 3.14159265359;

float hgPhase(float cosTheta, float g) {
    g = clamp(g, 0.0, 0.8);
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1.0e-4), 1.5));
}

float3 primaryRayDirFromPush(float2 ndc, WorldPush worldPush) {
    float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    return normalize(farP - nearP);
}

float activePhase(float3 rayDir) {
    float3 sunIllum = pc.sunIlluminance * pc.sunIntensity;
    float3 moonIllum = pc.moonIlluminance * pc.moonIntensity;
    bool sunEnabled = max(sunIllum.r, max(sunIllum.g, sunIllum.b)) > 1.0e-6;
    bool moonEnabled = max(moonIllum.r, max(moonIllum.g, moonIllum.b)) > 1.0e-6;
    if (sunEnabled) {
        return hgPhase(dot(rayDir, normalize(pc.sunDir)), pc.anisotropy);
    }
    if (moonEnabled) {
        return hgPhase(dot(rayDir, normalize(pc.moonDir)), pc.anisotropy);
    }
    return hgPhase(0.0, pc.anisotropy);
}

float3 localLitScatterCoeff(float4 froxel, float phase) {
    float extinction = max(froxel.a, 0.0);
    float3 directionalIllum = max(froxel.rgb, float3(0.0, 0.0, 0.0));
    float scatterOverExtinction = pc.extinction > 1.0e-6 ? pc.scattering / pc.extinction : 0.0;
    return directionalIllum * (extinction * scatterOverExtinction) * phase;
}

void integrateHomogeneousSegment(float3 litScatterCoeff, float extinction, float segmentLength,
                                 out float3 segmentScattering, out float segmentTransmittance) {
    if (!(segmentLength > 0.0)) {
        segmentScattering = float3(0.0, 0.0, 0.0);
        segmentTransmittance = 1.0;
        return;
    }
    if (extinction > 1.0e-6) {
        segmentTransmittance = exp(-extinction * segmentLength);
        segmentScattering = litScatterCoeff * ((1.0 - segmentTransmittance) / extinction);
    } else {
        segmentTransmittance = 1.0;
        segmentScattering = litScatterCoeff * segmentLength;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    int3 dim = int3(pc.froxelDim.x, pc.froxelDim.y, pc.froxelDepth);
    if (int(dispatchId.x) >= dim.x || int(dispatchId.y) >= dim.y) return;

    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    if ((worldPush.flags & 1u) != 0u) {
        for (int z = 0; z < dim.z; ++z) {
            integratedVolume[uint3(dispatchId.xy, uint(z))] = float4(0.0, 0.0, 0.0, 1.0);
        }
        return;
    }

    float nearP = max(pc.nearPlane, 0.1);
    float farP = min(max(pc.farPlane, nearP + 1.0), pc.maxDistance);
    float2 uv = (float2(dispatchId.xy) + float2(0.5, 0.5)) / float2(dim.xy);
    float2 ndc = uv * 2.0 - 1.0;
    float3 rayDir = primaryRayDirFromPush(ndc, worldPush);
    float phase = activePhase(rayDir);

    float3 cumulativeScattering = float3(0.0, 0.0, 0.0);
    float cumulativeTransmittance = 1.0;

    for (int z = 0; z < dim.z; ++z) {
        float zNear = froxelBoundaryDepth(z, dim.z, nearP, farP);
        float zFar = froxelBoundaryDepth(z + 1, dim.z, nearP, farP);
        float segmentLength = max(zFar - zNear, 0.0);
        float3 uvw = float3(uv, (float(z) + 0.5) / float(dim.z));
        float4 froxel = froxelVolume.SampleLevel(uvw, 0.0);
        float extinction = max(froxel.a, 0.0);
        float3 litScatterCoeff = localLitScatterCoeff(froxel, phase);

        float3 segmentScattering;
        float segmentTransmittance;
        integrateHomogeneousSegment(litScatterCoeff, extinction, segmentLength,
                segmentScattering, segmentTransmittance);

        cumulativeScattering += segmentScattering * cumulativeTransmittance;
        cumulativeTransmittance *= segmentTransmittance;

        integratedVolume[uint3(dispatchId.xy, uint(z))] = float4(cumulativeScattering, cumulativeTransmittance);
    }
}
