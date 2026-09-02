// Froxel fog material + directional-light injection pass.
// Fills a 3D frustum-aligned froxel volume with pre-shadowed directional illumination in rgb and
// extinction in a. Directional-light shadow/transmission is evaluated once per froxel here so the final
// screen-space integration pass no longer fires shadow rays per pixel sample.

import world_common;
import volumetric_fog_common;

// Reuse the richer integration push block so the host can widen injection inputs without depending on a
// second generated push-record ABI update path.
[[vk::push_constant]] FogIntegrationPush pc;

// Bindings - set 0
[[vk::binding(0, 0)]] [format("rgba16f")] RWTexture3D<float4> froxelVolume;
[[vk::binding(1, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(2, 0)]] Sampler2D blockAlbedoAtlas;

static const uint CULL_SECONDARY = 0x01u;
static const float RAY_TMIN = 0.001;
static const float ALPHA_CUTOFF = 0.5;
static const float TRANSLUCENT_NEUTRAL_EXTINCTION = 0.15;
static const float TRANSMITTANCE_CUTOFF = 0.01;
static const uint MAX_TRANSMISSIVE_CROSSINGS = 4u;

float hash31(float3 p) {
    return frac(sin(dot(p, float3(12.9898, 78.233, 45.164))) * 43758.5453);
}

float smoothValueNoise(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);

    float n000 = hash31(i + float3(0.0, 0.0, 0.0));
    float n100 = hash31(i + float3(1.0, 0.0, 0.0));
    float n010 = hash31(i + float3(0.0, 1.0, 0.0));
    float n110 = hash31(i + float3(1.0, 1.0, 0.0));
    float n001 = hash31(i + float3(0.0, 0.0, 1.0));
    float n101 = hash31(i + float3(1.0, 0.0, 1.0));
    float n011 = hash31(i + float3(0.0, 1.0, 1.0));
    float n111 = hash31(i + float3(1.0, 1.0, 1.0));

    float nx00 = lerp(n000, n100, u.x);
    float nx10 = lerp(n010, n110, u.x);
    float nx01 = lerp(n001, n101, u.x);
    float nx11 = lerp(n011, n111, u.x);
    float nxy0 = lerp(nx00, nx10, u.y);
    float nxy1 = lerp(nx01, nx11, u.y);
    return lerp(nxy0, nxy1, u.z);
}

float3 primaryRayDirFroxel(float2 ndc, WorldPush worldPush) {
    float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    return normalize(farP - nearP);
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

    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(topLevelAS, RAY_FLAG_NONE, CULL_SECONDARY, ray);

    uint transmissiveCrossings = 0u;
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) continue;

        uint instanceId = query.CandidateInstanceID();
        uint primId = query.CandidatePrimitiveIndex();
        uint geomId = query.CandidateGeometryIndex();

        if ((instanceId & 0x800000u) == 0u && (instanceId & 0x400000u) == 0u) {
            uint bucket = geomId;
            Section sec = ConstPtr<Section>(pc.tableAddr)[instanceId];
            uint tri = primId + sec.triBase[bucket];

            if (bucket == BUCKET_WATER) { // water
                result.hitOpaque = true;
                result.transmittance = float3(0.0);
                break;
            }
            if (bucket == BUCKET_TRANSLUCENT) { // translucent terrain
                TerrainPrim pr = ConstPtr<TerrainPrim>(sec.primAddr)[tri];
                MaterialHeader header = ConstPtr<MaterialHeader>(pc.materialTableAddr)[pr.materialId];
                float3 avgColor = max(bt709ToAcesCg(header.average.rgb * srgbToLinear(pr.tint.rgb)), float3(1.0e-3));
                float3 colorExtinction = max(-log(avgColor), float3(0.0));
                float transmissionFactor = clamp(header.params.w, 0.0, 1.0);
                float3 trans = exp(-colorExtinction * header.average.a - TRANSLUCENT_NEUTRAL_EXTINCTION)
                        * transmissionFactor;
                if (pc.colorTransmissionEnabled != 0) {
                    result.transmittance *= trans;
                } else {
                    float lum = dot(trans, float3(0.2126, 0.7152, 0.0722));
                    result.transmittance *= lum;
                }
                transmissiveCrossings++;
                if (max(result.transmittance.r, max(result.transmittance.g, result.transmittance.b))
                        < TRANSMITTANCE_CUTOFF) {
                    result.hitOpaque = true;
                    result.transmittance = float3(0.0);
                    break;
                }
                if (transmissiveCrossings >= MAX_TRANSMISSIVE_CROSSINGS) {
                    break;
                }
                continue;
            }
            if (bucket == BUCKET_CUTOUT) { // cutout
                ConstPtr<float2> uvs = ConstPtr<float2>(sec.uvAddr);
                float2 bary = query.CandidateTriangleBarycentrics();
                float3 b = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
                float2 uv = uvs[3u * tri + 0u] * b.x + uvs[3u * tri + 1u] * b.y + uvs[3u * tri + 2u] * b.z;
                float alpha = blockAlbedoAtlas.SampleLevel(uv, 0.0).a;
                if (alpha < ALPHA_CUTOFF) {
                    continue;
                }
                result.hitOpaque = true;
                result.transmittance = float3(0.0);
                break;
            }
            result.hitOpaque = true;
            result.transmittance = float3(0.0);
            break;
        } else {
            uint eidx = instanceId & 0x3FFFFFu;
            EntityGeom g = ConstPtr<EntityGeom>(pc.entityTableAddr)[eidx];
            uint pid = primId + g.triBase[geomId];
            Prim pr = ConstPtr<Prim>(g.primAddr)[pid];
            MaterialHeader header = ConstPtr<MaterialHeader>(pc.materialTableAddr)[pr.materialId];
            if (header.model == MATERIAL_DIELECTRIC) {
                float3 tint = bt709ToAcesCg(srgbToLinear(pr.tint.rgb));
                float transFactor = clamp(header.params.w, 0.0, 1.0);
                if (pc.colorTransmissionEnabled != 0) {
                    result.transmittance *= tint * transFactor;
                } else {
                    result.transmittance *= transFactor;
                }
                transmissiveCrossings++;
                if (max(result.transmittance.r, max(result.transmittance.g, result.transmittance.b))
                        < TRANSMITTANCE_CUTOFF) {
                    result.hitOpaque = true;
                    result.transmittance = float3(0.0);
                    break;
                }
                if (transmissiveCrossings >= MAX_TRANSMISSIVE_CROSSINGS) {
                    break;
                }
                continue;
            }
            if (header.model == MATERIAL_WATER) {
                result.hitOpaque = true;
                result.transmittance = float3(0.0);
                break;
            }
            result.hitOpaque = true;
            result.transmittance = float3(0.0);
            break;
        }
    }
    return result;
}

float3 sampleLightingTap(float2 uv, float depth, WorldPush worldPush,
                         bool sunEnabled, float3 sunDir, float3 sunIllum,
                         bool moonEnabled, float3 moonDir, float3 moonIllum) {
    float2 clampedUv = clamp(uv, float2(1.0e-4, 1.0e-4), float2(0.9999, 0.9999));
    float2 ndc = clampedUv * 2.0 - 1.0;
    float3 tapRayDir = primaryRayDirFroxel(ndc, worldPush);
    float3 tapRebasedPos = worldPush.camOffset + tapRayDir * depth;

    float3 lighting = float3(0.0, 0.0, 0.0);
    if (sunEnabled) {
        ShadowResult sunShadow = traceShadowWithTransmission(tapRebasedPos + sunDir * 0.01, sunDir, 10000.0);
        lighting += sunIllum * sunShadow.transmittance;
    }
    if (moonEnabled) {
        ShadowResult moonShadow = traceShadowWithTransmission(tapRebasedPos + moonDir * 0.01, moonDir, 10000.0);
        lighting += moonIllum * moonShadow.transmittance;
    }
    return lighting;
}

float lightingLuma(float3 lighting) {
    return dot(max(lighting, float3(0.0)), ACESCG_LUMA);
}

void accumulateLightingSample(float3 sampleLighting,
                              inout float3 lighting, inout float tapCount,
                              inout float minLuma, inout float maxLuma) {
    float luma = lightingLuma(sampleLighting);
    lighting += sampleLighting;
    tapCount += 1.0;
    minLuma = min(minLuma, luma);
    maxLuma = max(maxLuma, luma);
}

void accumulateMirroredLightingPair(uint3 cell, uint pairSalt, float2 uv, float2 froxelTexel,
                                    float depth, WorldPush worldPush,
                                    bool sunEnabled, float3 sunDir, float3 sunIllum,
                                    bool moonEnabled, float3 moonDir, float3 moonIllum,
                                    inout float3 lighting, inout float tapCount,
                                    inout float minLuma, inout float maxLuma) {
    float2 offset = fogBlueNoiseOffset2D(cell, pc.frameIndex, pairSalt) * (froxelTexel * 0.85);
    float3 plusLighting = sampleLightingTap(uv + offset, depth, worldPush,
            sunEnabled, sunDir, sunIllum, moonEnabled, moonDir, moonIllum);
    float3 minusLighting = sampleLightingTap(uv - offset, depth, worldPush,
            sunEnabled, sunDir, sunIllum, moonEnabled, moonDir, moonIllum);
    accumulateLightingSample(plusLighting, lighting, tapCount, minLuma, maxLuma);
    accumulateLightingSample(minusLighting, lighting, tapCount, minLuma, maxLuma);
}

[shader("compute")]
[numthreads(8, 8, 4)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    int3 dim = int3(pc.froxelDim.x, pc.froxelDim.y, pc.froxelDepth);
    if (int(dispatchId.x) >= dim.x || int(dispatchId.y) >= dim.y || int(dispatchId.z) >= dim.z) return;

    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];

    // The resolved froxel pipeline now has real temporal history, so injection no longer needs to freeze
    // every sub-froxel decision forever. Use a spatiotemporal blue-noise-style pattern to vary the depth
    // slice position and froxel-footprint lighting taps over time; the resolve pass accumulates that into a
    // smoother volume instead of letting one static sample pattern imprint itself permanently on shafts.
    float jitterZ = (fogBlueNoiseScalar3D(dispatchId.xyz, pc.frameIndex, 29u) - 0.5) * pc.jitterStrength;

    float2 uv = (float2(dispatchId.xy) + float2(0.5, 0.5)) / float2(dim.xy);
    float2 ndc = uv * 2.0 - 1.0;
    float3 rayDir = primaryRayDirFroxel(ndc, worldPush);

    float linear01 = (float(dispatchId.z) + 0.5 + jitterZ) / float(dim.z);
    float nearP = max(pc.nearPlane, 0.1);
    float farP = min(max(pc.farPlane, nearP + 1.0), pc.maxDistance);
    float depth = froxelDepthFromUnit(linear01, nearP, farP);

    float3 rebasedPos = worldPush.camOffset + rayDir * depth;
    float3 worldPos = rebasedPos + pc.terrainOrigin;

    float height = worldPos.y;
    float heightFactor = exp(-max(height - 64.0, 0.0) * pc.heightFalloff);
    float highAltitudeFade = saturate(1.0 - max(height - 128.0, 0.0) * 0.01);
    float distanceFade = 1.0 / (1.0 + depth * 0.001);

    float baseDensity = pc.density * heightFactor * highAltitudeFade * distanceFade;
    // The old single-sample froxel path made this density modulation read back as visible static grain.
    // Keep only a much smoother, lower-amplitude world-space variation so global fog stays natural
    // without injecting a hash-shaped breakup into every shaft.
    float3 noisePos = worldPos * 0.01;
    float n = smoothValueNoise(noisePos);
    n = lerp(0.98, 1.02, n);
    baseDensity *= n;
    baseDensity = max(baseDensity, 0.0);

    float extinction = baseDensity * pc.extinction;
    float scatteringDensity = baseDensity * pc.scattering;
    if (extinction <= 1.0e-6 && scatteringDensity <= 1.0e-6) {
        froxelVolume[dispatchId] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float3 lighting = float3(0.0, 0.0, 0.0);
    float tapCount = 0.0;
    float minLuma = 1.0e30;
    float maxLuma = 0.0;
    float3 sunIllum = pc.sunIlluminance * pc.sunIntensity;
    float3 moonIllum = pc.moonIlluminance * pc.moonIntensity;
    bool sunEnabled = max(sunIllum.r, max(sunIllum.g, sunIllum.b)) > 1.0e-6;
    bool moonEnabled = max(moonIllum.r, max(moonIllum.g, moonIllum.b)) > 1.0e-6;
    float3 sunDir = sunEnabled ? normalize(pc.sunDir) : float3(0.0, 1.0, 0.0);
    float3 moonDir = moonEnabled ? normalize(pc.moonDir) : float3(0.0, 1.0, 0.0);

    // One shadowed lighting sample per froxel kept the new architecture fast, but it also exposed the
    // froxel lattice directly in shafts cast through thin openings and distant geometry. Now that the
    // volume has its own temporal resolve, spend taps where they matter: estimate the cell-average
    // directional lighting, and adaptively refine only when the first samples disagree strongly.
    float2 froxelTexel = 1.0 / float2(float(dim.x), float(dim.y));
    bool mediumQuality = dim.x >= 224 || dim.z >= 32;
    bool highQuality = dim.x >= 336 || dim.z >= 48;

    float3 centerLighting = sampleLightingTap(uv, depth, worldPush,
            sunEnabled, sunDir, sunIllum, moonEnabled, moonDir, moonIllum);
    accumulateLightingSample(centerLighting, lighting, tapCount, minLuma, maxLuma);

    if (mediumQuality) {
        accumulateMirroredLightingPair(dispatchId.xyz, 0u, uv, froxelTexel, depth, worldPush,
                sunEnabled, sunDir, sunIllum, moonEnabled, moonDir, moonIllum,
                lighting, tapCount, minLuma, maxLuma);

        float contrast = (maxLuma - minLuma) / max(maxLuma, 1.0e-3);
        bool refineMedium = contrast > 0.10;
        bool refineHigh = contrast > 0.28;

        if (refineMedium) {
            accumulateMirroredLightingPair(dispatchId.xyz, 19u, uv, froxelTexel, depth, worldPush,
                    sunEnabled, sunDir, sunIllum, moonEnabled, moonDir, moonIllum,
                    lighting, tapCount, minLuma, maxLuma);
        }
        if (highQuality && refineHigh) {
            accumulateMirroredLightingPair(dispatchId.xyz, 37u, uv, froxelTexel, depth, worldPush,
                    sunEnabled, sunDir, sunIllum, moonEnabled, moonDir, moonIllum,
                    lighting, tapCount, minLuma, maxLuma);
            accumulateMirroredLightingPair(dispatchId.xyz, 59u, uv, froxelTexel, depth, worldPush,
                    sunEnabled, sunDir, sunIllum, moonEnabled, moonDir, moonIllum,
                    lighting, tapCount, minLuma, maxLuma);
        }
    }
    lighting /= max(tapCount, 1.0);

    froxelVolume[dispatchId] = float4(lighting, extinction);
}
