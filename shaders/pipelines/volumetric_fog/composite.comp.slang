// Froxel volumetric fog final composite.
// Reconstructs fog from the column-integrated cumulative volume built by integration.comp.slang. The
// local resolved froxel field is still sampled for partial-slice recovery, while depth-aware XY
// reconstruction rejects cross-surface blending that would otherwise leak bright background shafts across
// foreground silhouettes.

import world_common;
import volumetric_fog_common;

[[vk::push_constant]] FogIntegrationPush pc;

// Bindings - set 0
[[vk::binding(0, 0)]] Sampler3D froxelVolume;
[[vk::binding(1, 0)]] Sampler3D integratedVolume;
[[vk::binding(2, 0)]] Sampler2D fogStopDepth;
[[vk::binding(3, 0)]] [format("rgba16f")] RWTexture2D<float4> sceneImage;

static const float PI = 3.14159265359;
static const float SKY_DEPTH_EPSILON = 1.0e-6;

float hgPhase(float cosTheta, float g) {
    // Fog anisotropy is defined as forward scattering in the UI: looking toward the light should brighten.
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

float3 reconstructRebasedPos(float2 ndc, float ndcDepth, WorldPush worldPush) {
    float4 clip = float4(ndc.x, ndc.y, ndcDepth, 1.0);
    float4 view = mul(worldPush.invViewProj, clip);
    view /= view.w;
    return view.xyz + worldPush.camOffset;
}

float linearDepthFromRebased(float3 rebasedPos, WorldPush worldPush) {
    float3 camRel = rebasedPos - worldPush.camOffset;
    return length(camRel);
}

float depthToMarchDistance(float2 ndc, float depth, WorldPush worldPush) {
    if (depth <= SKY_DEPTH_EPSILON) {
        return pc.maxDistance;
    }
    float3 rebasedHitPos = reconstructRebasedPos(ndc, depth, worldPush);
    float tMax = linearDepthFromRebased(rebasedHitPos, worldPush);
    if (!(tMax > 0.1 && tMax <= 10000.0)) {
        return pc.maxDistance;
    }
    return min(tMax, pc.maxDistance);
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

struct FogState {
    float3 scattering;
    float transmittance;
};

FogState emptyFog() {
    FogState fog;
    fog.scattering = float3(0.0, 0.0, 0.0);
    fog.transmittance = 1.0;
    return fog;
}

float3 localLitScatterCoeff(float4 froxel, float phase) {
    float extinction = max(froxel.a, 0.0);
    float3 directionalIllum = max(froxel.rgb, float3(0.0, 0.0, 0.0));
    float scatterOverExtinction = pc.extinction > 1.0e-6 ? pc.scattering / pc.extinction : 0.0;
    return directionalIllum * (extinction * scatterOverExtinction) * phase;
}

FogState integrateHomogeneousSegment(float3 litScatterCoeff, float extinction, float segmentLength) {
    FogState fog = emptyFog();
    if (!(segmentLength > 0.0)) {
        return fog;
    }
    if (extinction > 1.0e-6) {
        fog.transmittance = exp(-extinction * segmentLength);
        fog.scattering = litScatterCoeff * ((1.0 - fog.transmittance) / extinction);
    } else {
        fog.transmittance = 1.0;
        fog.scattering = litScatterCoeff * segmentLength;
    }
    return fog;
}

void sampleColumnStop(int2 cell, WorldPush worldPush, int3 dim, out float stopDistance, out bool isSky) {
    float2 cellUv = (float2(cell) + float2(0.5, 0.5)) / float2(dim.xy);
    float depth = fogStopDepth.SampleLevel(cellUv, 0.0).r;
    isSky = depth <= SKY_DEPTH_EPSILON;
    stopDistance = depthToMarchDistance(cellUv * 2.0 - 1.0, depth, worldPush);
}

float depthAwareColumnWeight(float currentStopDistance, bool currentSky,
                             float columnStopDistance, bool columnSky) {
    if (currentSky != columnSky) {
        return 0.0;
    }
    float depthDelta = abs(columnStopDistance - currentStopDistance);
    float relativeDelta = depthDelta / max(max(currentStopDistance, columnStopDistance), 1.0);
    float absoluteDelta = depthDelta / max(min(currentStopDistance, columnStopDistance), 2.0);
    float weight = exp(-relativeDelta * 18.0 - absoluteDelta * 1.25);
    if (!currentSky && relativeDelta > 0.25) {
        weight *= 0.1;
    }
    return weight;
}

FogState evaluateColumnFog(int2 cell, float targetDistance, WorldPush worldPush,
                           int3 dim, float nearP, float farP) {
    FogState fog = emptyFog();
    if (!(targetDistance > nearP)) {
        return fog;
    }

    float2 cellUv = (float2(cell) + float2(0.5, 0.5)) / float2(dim.xy);
    float slicePos = saturate(froxelUnitFromDepth(targetDistance, nearP, farP)) * float(dim.z);
    int sliceIndex = min(int(floor(slicePos)), dim.z - 1);
    float sliceStart = froxelBoundaryDepth(sliceIndex, dim.z, nearP, farP);
    float sliceEnd = froxelBoundaryDepth(sliceIndex + 1, dim.z, nearP, farP);
    float partialLength = clamp(targetDistance - sliceStart, 0.0, max(sliceEnd - sliceStart, 0.0));

    if (sliceIndex > 0) {
        float3 prevUvw = float3(cellUv, (float(sliceIndex - 1) + 0.5) / float(dim.z));
        float4 prev = integratedVolume.SampleLevel(prevUvw, 0.0);
        fog.scattering = max(prev.rgb, float3(0.0, 0.0, 0.0));
        fog.transmittance = clamp(prev.a, 0.0, 1.0);
    }

    float3 rayDir = primaryRayDirFromPush(cellUv * 2.0 - 1.0, worldPush);
    float phase = activePhase(rayDir);
    float3 localUvw = float3(cellUv, (float(sliceIndex) + 0.5) / float(dim.z));
    float4 localFroxel = froxelVolume.SampleLevel(localUvw, 0.0);
    float extinction = max(localFroxel.a, 0.0);
    FogState partial = integrateHomogeneousSegment(localLitScatterCoeff(localFroxel, phase), extinction, partialLength);
    fog.scattering += partial.scattering * fog.transmittance;
    fog.transmittance *= partial.transmittance;
    return fog;
}

void accumulateColumnCandidate(int2 cell, float bilinearWeight,
                               float currentStopDistance, bool currentSky,
                               WorldPush worldPush, int3 dim, float nearP, float farP,
                               inout float3 scatteringSum,
                               inout float transmittanceSum,
                               inout float weightSum) {
    int2 clampedCell = clamp(cell, int2(0, 0), dim.xy - int2(1, 1));
    float columnStopDistance;
    bool columnSky;
    sampleColumnStop(clampedCell, worldPush, dim, columnStopDistance, columnSky);

    float weight = bilinearWeight * depthAwareColumnWeight(currentStopDistance, currentSky,
            columnStopDistance, columnSky);
    if (weight <= 1.0e-5) {
        return;
    }

    FogState fog = evaluateColumnFog(clampedCell, min(currentStopDistance, columnStopDistance),
            worldPush, dim, nearP, farP);
    scatteringSum += fog.scattering * weight;
    transmittanceSum += fog.transmittance * weight;
    weightSum += weight;
}

FogState reconstructFog(float2 uv, float currentStopDistance, bool currentSky,
                        WorldPush worldPush, int3 dim, float nearP, float farP) {
    float2 froxelPos = uv * float2(dim.xy) - 0.5;
    int2 base = int2(floor(froxelPos));
    float2 fracPart = frac(froxelPos);

    float3 scatteringSum = float3(0.0, 0.0, 0.0);
    float transmittanceSum = 0.0;
    float weightSum = 0.0;

    accumulateColumnCandidate(base + int2(0, 0), (1.0 - fracPart.x) * (1.0 - fracPart.y),
            currentStopDistance, currentSky, worldPush, dim, nearP, farP,
            scatteringSum, transmittanceSum, weightSum);
    accumulateColumnCandidate(base + int2(1, 0), fracPart.x * (1.0 - fracPart.y),
            currentStopDistance, currentSky, worldPush, dim, nearP, farP,
            scatteringSum, transmittanceSum, weightSum);
    accumulateColumnCandidate(base + int2(0, 1), (1.0 - fracPart.x) * fracPart.y,
            currentStopDistance, currentSky, worldPush, dim, nearP, farP,
            scatteringSum, transmittanceSum, weightSum);
    accumulateColumnCandidate(base + int2(1, 1), fracPart.x * fracPart.y,
            currentStopDistance, currentSky, worldPush, dim, nearP, farP,
            scatteringSum, transmittanceSum, weightSum);

    FogState fog = emptyFog();
    if (weightSum > 1.0e-5) {
        fog.scattering = scatteringSum / weightSum;
        fog.transmittance = transmittanceSum / weightSum;
        return fog;
    }

    int2 fallbackCell = clamp(int2(floor(froxelPos + 0.5)), int2(0, 0), dim.xy - int2(1, 1));
    float fallbackStopDistance;
    bool fallbackSky;
    sampleColumnStop(fallbackCell, worldPush, dim, fallbackStopDistance, fallbackSky);
    return evaluateColumnFog(fallbackCell, min(currentStopDistance, fallbackStopDistance),
            worldPush, dim, nearP, farP);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    int2 pix = int2(dispatchId.xy);
    int2 displaySize = pc.displaySize;
    if (pix.x >= displaySize.x || pix.y >= displaySize.y) return;

    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];

    // This pass models aerial/air fog only. The validated air-only fix remains: underwater views should
    // not integrate the same air volume inside water.
    if ((worldPush.flags & 1u) != 0u) {
        return;
    }

    int3 dim = int3(pc.froxelDim.x, pc.froxelDim.y, pc.froxelDepth);
    float nearP = max(pc.nearPlane, 0.1);
    float farP = min(max(pc.farPlane, nearP + 1.0), pc.maxDistance);

    float2 uv = (float2(pix) + 0.5) * pc.invDisplaySize;
    float2 ndc = uv * 2.0 - 1.0;
    float sceneDepth = fogStopDepth.SampleLevel(uv, 0.0).r;
    bool currentSky = sceneDepth <= SKY_DEPTH_EPSILON;
    float currentStopDistance = depthToMarchDistance(ndc, sceneDepth, worldPush);

    FogState fog = reconstructFog(uv, currentStopDistance, currentSky, worldPush, dim, nearP, farP);

    float4 scene = sceneImage[pix];
    // The traced scene buffer is pre-exposed before its fp16 write, so fog scattering must be brought into
    // the same storage scale before compositing or exposure changes make the fog appear to brighten/dim
    // independently of the scene.
    sceneImage[pix] = float4(scene.rgb * fog.transmittance + fog.scattering * worldPush.preExposure, scene.a);
}
