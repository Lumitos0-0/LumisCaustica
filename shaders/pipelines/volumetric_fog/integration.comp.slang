// Froxel volumetric fog integration with stratified jittered raymarching
// and hardware ray-traced shadows for razor-sharp crepuscular rays.
// Supports color transmission through translucent materials.

import world_common;
import volumetric_fog_common;

[[vk::push_constant]] FogIntegrationPush pc;

// Bindings - set 0
[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] Sampler3D froxelVolume;
[[vk::binding(2, 0)]] Sampler2D gDepth;
[[vk::binding(3, 0)]] [format("rgba16f")] RWTexture2D<float4> sceneImage;
[[vk::binding(4, 0)]] Sampler2D blockAlbedoAtlas;

static const float PI = 3.14159265359;
static const uint CULL_SECONDARY = 0x01u;
static const float RAY_TMIN = 0.001;
static const float ALPHA_CUTOFF = 0.5;
static const float TRANSLUCENT_NEUTRAL_EXTINCTION = 0.15;
static const float WATER_SHADOW_TINT = 0.5;

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
    // This fog path is tuned only for forward scattering. Clamp off the near-singular tail as g
    // approaches 1 and treat any stale negative config value as isotropic.
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

struct ShadowResult {
    float3 transmittance;
    bool hitOpaque;
};

ShadowResult traceShadowWithTransmission(float3 origin, float3 dir, float tMax) {
    ShadowResult result;
    result.transmittance = float3(1.0, 1.0, 1.0);
    result.hitOpaque = false;

    RayDesc ray;
    ray.Origin = origin;
    ray.TMin = RAY_TMIN;
    ray.Direction = dir;
    ray.TMax = tMax;

    // Force every triangle candidate through shader code so solids can actually occlude the sun/moon
    // while water/translucent materials still accumulate colored transmission. With plain RAY_FLAG_NONE,
    // opaque triangles are handled entirely by fixed-function traversal and this pass never sees them,
    // which lets sunlight leak through caves/terrain and shows up as a camera-following fog wall.
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(topLevelAS, RAY_FLAG_NONE, CULL_SECONDARY, ray);

    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) continue;

        uint instanceId = query.CandidateInstanceID();
        uint primId = query.CandidatePrimitiveIndex();
        uint geomId = query.CandidateGeometryIndex();

        // Terrain: instance bits 0x800000 and 0x400000 not set
        if ((instanceId & 0x800000u) == 0u && (instanceId & 0x400000u) == 0u) {
            uint bucket = geomId;
            Section sec = ConstPtr<Section>(pc.tableAddr)[instanceId];
            uint tri = primId + sec.triBase[bucket];

            if (bucket == 3u) { // water
                TerrainPrim pr = ConstPtr<TerrainPrim>(sec.primAddr)[tri];
                float3 tint = bt709ToAcesCg(srgbToLinear(pr.tint.rgb));
                if (pc.colorTransmissionEnabled != 0) {
                    result.transmittance *= lerp(float3(1.0), tint, WATER_SHADOW_TINT);
                } else {
                    result.transmittance *= 0.9;
                }
                query.CommitNonOpaqueTriangleHit();
                continue;
            }
            if (bucket == 2u) { // translucent
                TerrainPrim pr = ConstPtr<TerrainPrim>(sec.primAddr)[tri];
                MaterialHeader header = ConstPtr<MaterialHeader>(pc.materialTableAddr)[pr.materialId];
                float3 avgColor = max(bt709ToAcesCg(header.average.rgb * srgbToLinear(pr.tint.rgb)), float3(1.0e-3));
                float3 colorExtinction = max(-log(avgColor), float3(0.0));
                float3 trans = exp(-colorExtinction * header.average.a - TRANSLUCENT_NEUTRAL_EXTINCTION);
                if (pc.colorTransmissionEnabled != 0) {
                    result.transmittance *= trans;
                } else {
                    float lum = dot(trans, float3(0.2126, 0.7152, 0.0722));
                    result.transmittance *= lum;
                }
                query.CommitNonOpaqueTriangleHit();
                continue;
            }
            if (bucket == 1u) { // cutout - alpha test using atlas
                ConstPtr<float2> uvs = ConstPtr<float2>(sec.uvAddr);
                float2 bary = query.CandidateTriangleBarycentrics();
                float3 b = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
                float2 uv = uvs[3u * tri + 0u] * b.x + uvs[3u * tri + 1u] * b.y + uvs[3u * tri + 2u] * b.z;
                float alpha = blockAlbedoAtlas.SampleLevel(uv, 0.0).a;
                if (alpha < ALPHA_CUTOFF) {
                    // Fully transparent cutout texels should behave like no hit at all.
                    continue;
                }
                result.hitOpaque = true;
                result.transmittance = float3(0.0);
                break;
            }
            // solid
            result.hitOpaque = true;
            result.transmittance = float3(0.0);
            break;
        } else {
            // Entity: opaque unless dielectric (glass) -> allow colored transmission
            uint eidx = instanceId & 0x3FFFFFu;
            EntityGeom g = ConstPtr<EntityGeom>(pc.entityTableAddr)[eidx];
            uint pid = primId + g.triBase[geomId];
            Prim pr = ConstPtr<Prim>(g.primAddr)[pid];
            MaterialHeader header = ConstPtr<MaterialHeader>(pc.materialTableAddr)[pr.materialId];
            if (header.model == 3u) { // dielectric
                float3 tint = bt709ToAcesCg(srgbToLinear(pr.tint.rgb));
                float transFactor = clamp(header.params.w, 0.0, 1.0);
                if (pc.colorTransmissionEnabled != 0) {
                    result.transmittance *= lerp(float3(1.0), tint, 0.8) * transFactor;
                } else {
                    result.transmittance *= transFactor;
                }
                query.CommitNonOpaqueTriangleHit();
                continue;
            }
            result.hitOpaque = true;
            result.transmittance = float3(0.0);
            break;
        }
    }
    return result;
}

struct FogIntegralResult {
    float3 scattering;
    float transmittance;
};

FogIntegralResult integrateFogRay(float3 rayOrigin, float3 rayDir, float tMax, int sampleCount,
                                  uint seedBase, float temporalJitter, WorldPush worldPush,
                                  bool sunEnabled, bool moonEnabled,
                                  float3 sunDir, float3 moonDir,
                                  float3 sunIllum, float3 moonIllum) {
    FogIntegralResult result;
    result.scattering = float3(0.0, 0.0, 0.0);
    result.transmittance = 1.0;
    if (!(tMax > 0.0) || sampleCount <= 0) {
        return result;
    }

    float stepSize = tMax / float(sampleCount);
    uint seed = seedBase;
    for (int i = 0; i < sampleCount; i++) {
        float jitter = pc.jitterStrength != 0.0 ? (rnd(seed) - 0.5) * pc.jitterStrength : 0.0;
        float t = (float(i) + temporalJitter + jitter) * stepSize;
        t = clamp(t, 0.0, tMax);

        float3 sampleRebasedPos = rayOrigin + rayDir * t;
        float3 sampleWorldPos = sampleRebasedPos + pc.terrainOrigin;

        float4 clip = mul(worldPush.curViewProj, float4(sampleRebasedPos - worldPush.camOffset, 1.0));
        if (clip.w <= 0.0) {
            continue;
        }
        float2 sampleNdc = clip.xy / clip.w;
        float2 sampleUv = sampleNdc * 0.5 + 0.5;
        float nearP = max(pc.nearPlane, 0.1);
        float farP = max(pc.farPlane, nearP + 1.0);
        float safeT = max(t, nearP);
        float linear01 = log(safeT / nearP) / log(farP / nearP);
        linear01 = saturate(linear01);
        float3 froxelUv = float3(sampleUv, linear01);
        float4 froxel = froxelVolume.SampleLevel(froxelUv, 0.0);
        float extinction = froxel.a;
        float3 scatteringCoeff = froxel.rgb;

        float height = sampleWorldPos.y;
        float heightFactor = exp(-max(height - 64.0, 0.0) * pc.heightFalloff);
        extinction *= heightFactor;
        scatteringCoeff *= heightFactor;

        if (extinction <= 1.0e-6 && max(scatteringCoeff.r, max(scatteringCoeff.g, scatteringCoeff.b)) <= 1.0e-6) {
            continue;
        }

        float3 shadowOrigin = sampleRebasedPos + rayDir * 0.01;
        float3 lighting = float3(0.0, 0.0, 0.0);
        if (sunEnabled) {
            ShadowResult sunShadow = traceShadowWithTransmission(shadowOrigin, sunDir, 10000.0);
            float phaseSun = hgPhase(dot(rayDir, sunDir), pc.anisotropy);
            lighting += phaseSun * sunIllum * sunShadow.transmittance;
        }
        if (moonEnabled) {
            ShadowResult moonShadow = traceShadowWithTransmission(shadowOrigin, moonDir, 10000.0);
            float phaseMoon = hgPhase(dot(rayDir, moonDir), pc.anisotropy);
            lighting += phaseMoon * moonIllum * moonShadow.transmittance;
        }

        float3 inScattering = scatteringCoeff * lighting * stepSize;
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

    float2 invSize = pc.invDisplaySize;
    float2 uv = (float2(pix) + 0.5) * invSize;
    float2 jndc = (uv + pc.jitterOffset * invSize) * 2.0 - 1.0;

    float3 rayDir = primaryRayDirFromPush(jndc, worldPush);
    float3 rayOrigin = worldPush.camOffset;

    // March fog against the SAME per-pixel render-resolution guide depth the tracer and DLSS-RR already
    // use, then composite directly into the render-resolution scene before reconstruction. This removes
    // the low-res/post-upscale resolve stage that was still producing silhouette stair-steps even after
    // the earlier two-layer fixes.
    float2 gDepthUv = uv;
    float sceneDepth = gDepth.SampleLevel(gDepthUv, 0.0).r;
    float tMax = depthToMarchDistance(jndc, sceneDepth, worldPush);

    int sampleCount = max(pc.sampleCount, 1);
    uint seed = pix.x * 1973u + pix.y * 9277u + pc.frameIndex * 2654435761u;

    float ignVal = ign(float2(pix) + float(pc.frameIndex) * 0.5);
    float temporalJitter = pc.temporalEnabled != 0 ? ignVal : 0.5;

    float3 sunDir = normalize(pc.sunDir);
    float3 moonDir = normalize(pc.moonDir);
    float3 sunIllum = pc.sunIlluminance * pc.sunIntensity;
    float3 moonIllum = pc.moonIlluminance * pc.moonIntensity;
    bool sunEnabled = max(sunIllum.r, max(sunIllum.g, sunIllum.b)) > 1.0e-6;
    bool moonEnabled = max(moonIllum.r, max(moonIllum.g, moonIllum.b)) > 1.0e-6;

    FogIntegralResult fog = integrateFogRay(rayOrigin, rayDir, tMax, sampleCount,
            seed, temporalJitter, worldPush,
            sunEnabled, moonEnabled, sunDir, moonDir, sunIllum, moonIllum);

    float4 scene = sceneImage[pix];
    sceneImage[pix] = float4(scene.rgb * fog.transmittance + fog.scattering, scene.a);
}
