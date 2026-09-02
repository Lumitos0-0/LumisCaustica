// Froxel volumetric fog integration.
// Directional-light shadow/transmission is precomputed per froxel in injection.comp.slang, then the raw
// lighting field is filtered and temporally resolved in resolve.comp.slang. This pass restores the
// high-quality view-ray march while sampling only that cheaper prelit volume.

import world_common;
import volumetric_fog_common;

[[vk::push_constant]] FogIntegrationPush pc;

// Bindings - set 0
[[vk::binding(0, 0)]] Sampler3D froxelVolume;
[[vk::binding(1, 0)]] Sampler2D fogStopDepth;
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture2D<float4> sceneImage;

static const float PI = 3.14159265359;

uint pcgHash(uint v) {
    v = v * 747796405u + 2891336453u;
    uint w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}
float rnd(inout uint s) {
    s = pcgHash(s);
    return float(s >> 8u) * (1.0 / 16777216.0);
}
float ign(float2 p) {
    float3 magic = float3(0.06711056, 0.00583715, 52.9829189);
    return frac(magic.z * frac(dot(p, magic.xy)));
}
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

static const float SKY_DEPTH_EPSILON = 1.0e-6;

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

struct FogIntegralResult {
    float3 scattering;
    float transmittance;
};

FogIntegralResult integrateFogRay(float3 rayDir, float tMax, int sampleCount,
                                  uint seedBase, float temporalJitter, float2 sampleUv) {
    FogIntegralResult result;
    result.scattering = float3(0.0, 0.0, 0.0);
    result.transmittance = 1.0;
    if (!(tMax > 0.0) || sampleCount <= 0) {
        return result;
    }

    float nearP = max(pc.nearPlane, 0.1);
    float farP = min(max(pc.farPlane, nearP + 1.0), pc.maxDistance);
    float phase = activePhase(rayDir);
    float stepSize = tMax / float(sampleCount);
    uint seed = seedBase;

    for (int i = 0; i < sampleCount; i++) {
        float jitter = pc.jitterStrength != 0.0 ? (rnd(seed) - 0.5) * pc.jitterStrength : 0.0;
        float t = (float(i) + temporalJitter + jitter) * stepSize;
        t = clamp(t, 0.0, tMax);

        float safeT = max(t, nearP);
        float linear01 = froxelUnitFromDepth(safeT, nearP, farP);

        float4 froxel = froxelVolume.SampleLevel(float3(sampleUv, linear01), 0.0);
        float extinction = max(froxel.a, 0.0);
        float3 litScatterCoeff = max(froxel.rgb, float3(0.0));
        if (extinction <= 1.0e-6 && max(litScatterCoeff.r, max(litScatterCoeff.g, litScatterCoeff.b)) <= 1.0e-6) {
            continue;
        }

        float3 inScattering = litScatterCoeff * phase * stepSize;
        float stepTrans = exp(-extinction * stepSize);
        result.scattering += inScattering * result.transmittance;
        result.transmittance *= stepTrans;
        if (result.transmittance < 0.01) {
            break;
        }
    }
    return result;
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

    float2 invSize = pc.invDisplaySize;
    float2 uv = (float2(pix) + 0.5) * invSize;
    // Post-RR fog integration is already in final display space, so following the trace's sub-pixel jitter
    // here only reintroduces frame-to-frame shimmer against an otherwise stabilized scene.
    float2 ndc = uv * 2.0 - 1.0;

    float3 rayDir = primaryRayDirFromPush(ndc, worldPush);
    float sceneDepth = fogStopDepth.SampleLevel(uv, 0.0).r;
    float tMax = depthToMarchDistance(ndc, sceneDepth, worldPush);

    int sampleCount = max(pc.sampleCount, 1);
    uint seed = pix.x * 1973u + pix.y * 9277u + pc.frameIndex * 2654435761u;
    float temporalJitter = pc.temporalEnabled != 0
            ? ign(float2(pix) + float(pc.frameIndex) * 0.5)
            : 0.5;

    FogIntegralResult fog = integrateFogRay(rayDir, tMax, sampleCount,
            seed, temporalJitter, uv);

    float4 scene = sceneImage[pix];
    // The traced scene buffer is pre-exposed before its fp16 write, so fog scattering must be brought into
    // the same storage scale before compositing or exposure changes make the fog appear to brighten/dim
    // independently of the scene.
    sceneImage[pix] = float4(scene.rgb * fog.transmittance + fog.scattering * worldPush.preExposure, scene.a);
}
