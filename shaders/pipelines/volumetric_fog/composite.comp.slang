// Froxel volumetric fog final composite.
// Reconstructs fog from the column-integrated cumulative volume built by integration.comp.slang.
// Depth-aware XY reconstruction rejects cross-surface blending that would otherwise leak bright
// background shafts across foreground silhouettes, while linear Z reconstruction between cumulative
// slice boundaries smooths the remaining froxel pattern instead of re-exposing one raw local slice.

import world_common;
import volumetric_fog_common;

[[vk::push_constant]] FogIntegrationPush pc;

// Bindings - set 0
[[vk::binding(0, 0)]] Sampler3D froxelVolume;
[[vk::binding(1, 0)]] Sampler3D integratedVolume;
[[vk::binding(2, 0)]] Sampler2D fogStopDepth;
[[vk::binding(3, 0)]] [format("rgba16f")] RWTexture2D<float4> sceneImage;

static const float SKY_DEPTH_EPSILON = 1.0e-6;

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

void sampleColumnStop(int2 cell, WorldPush worldPush, int3 dim, out float stopDistance, out bool isSky) {
    float2 cellUv = (float2(cell) + float2(0.5, 0.5)) / float2(dim.xy);
    float depth = fogStopDepth.SampleLevel(cellUv, 0.0).r;
    isSky = depth <= SKY_DEPTH_EPSILON;
    stopDistance = depthToMarchDistance(cellUv * 2.0 - 1.0, depth, worldPush);
}

float columnTentWeight(float sampleCoord, int cellCoord) {
    float center = float(cellCoord) + 0.5;
    return max(1.5 - abs(center - sampleCoord), 0.0);
}

float depthAwareColumnWeight(float currentStopDistance, bool currentSky,
                             float columnStopDistance, bool columnSky) {
    if (currentSky != columnSky) {
        return 0.0;
    }
    float tolerance = max(0.75, currentStopDistance * 0.15);
    return exp(-abs(columnStopDistance - currentStopDistance) / tolerance);
}

float4 loadIntegratedColumn(int2 cell, int zIndex, int3 dim) {
    int2 clampedCell = clamp(cell, int2(0, 0), dim.xy - int2(1, 1));
    int clampedZ = clamp(zIndex, 0, dim.z - 1);
    float3 uvw = float3((float2(clampedCell) + float2(0.5, 0.5)) / float2(dim.xy),
            (float(clampedZ) + 0.5) / float(dim.z));
    return integratedVolume.SampleLevel(uvw, 0.0);
}

float4 sampleIntegratedColumn(int2 cell, float zCoord, int3 dim) {
    float boundedZ = clamp(zCoord, 0.0, float(dim.z - 1));
    int zLo = int(floor(boundedZ));
    int zHi = min(zLo + 1, dim.z - 1);
    float zf = frac(boundedZ);
    float4 nearValue = loadIntegratedColumn(cell, zLo, dim);
    float4 farValue = loadIntegratedColumn(cell, zHi, dim);
    return lerp(nearValue, farValue, zf);
}

float4 depthAwareIntegrated(float3 coordinate, float pixelDistance, bool currentSky,
                            WorldPush worldPush, int3 dim) {
    float3 bounded = clamp(coordinate, float3(0.0, 0.0, 0.0),
            float3(float(dim.x - 1), float(dim.y - 1), float(dim.z - 1)));
    int2 base = int2(floor(bounded.xy));
    float2 sampleCoord = bounded.xy + 0.5;

    float4 total = float4(0.0, 0.0, 0.0, 0.0);
    float weightSum = 0.0;

    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            int2 cell = clamp(base + int2(ox, oy), int2(0, 0), dim.xy - int2(1, 1));
            float spatialWeight = columnTentWeight(sampleCoord.x, cell.x)
                    * columnTentWeight(sampleCoord.y, cell.y);
            if (spatialWeight <= 1.0e-5) {
                continue;
            }

            float columnStopDistance;
            bool columnSky;
            sampleColumnStop(cell, worldPush, dim, columnStopDistance, columnSky);
            float depthWeight = depthAwareColumnWeight(pixelDistance, currentSky,
                    columnStopDistance, columnSky);
            float weight = spatialWeight * depthWeight;
            if (weight <= 1.0e-5) {
                continue;
            }

            total += sampleIntegratedColumn(cell, bounded.z, dim) * weight;
            weightSum += weight;
        }
    }

    if (weightSum <= 1.0e-6) {
        int2 nearest = clamp(int2(round(bounded.xy)), int2(0, 0), dim.xy - int2(1, 1));
        return sampleIntegratedColumn(nearest, bounded.z, dim);
    }
    return total / weightSum;
}

FogState reconstructFog(float2 uv, float currentStopDistance, bool currentSky,
                        WorldPush worldPush, int3 dim, float nearP, float farP) {
    FogState fog = emptyFog();
    if (!(currentStopDistance > 0.0)) {
        return fog;
    }

    float slice = saturate(froxelUnitFromDepth(currentStopDistance, nearP, farP)) * float(dim.z);
    float3 coordinate = float3(uv * float2(dim.xy) - 0.5, slice - 1.0);
    float4 integrated = depthAwareIntegrated(coordinate, currentStopDistance, currentSky, worldPush, dim);

    if (slice < 1.0) {
        // Texel zero stores fog integrated through the first boundary. Recover a partial constant-medium
        // segment so fog starts continuously at the camera instead of appearing only once the first slice
        // is fully entered.
        float fraction = saturate(slice);
        float fullTransmittance = clamp(integrated.a, 1.0e-6, 1.0);
        float partialTransmittance = pow(fullTransmittance, fraction);
        float fullOpacity = max(1.0 - fullTransmittance, 1.0e-6);
        integrated.rgb *= (1.0 - partialTransmittance) / fullOpacity;
        integrated.a = partialTransmittance;
    }

    fog.scattering = max(integrated.rgb, float3(0.0, 0.0, 0.0));
    fog.transmittance = clamp(integrated.a, 0.0, 1.0);
    return fog;
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
