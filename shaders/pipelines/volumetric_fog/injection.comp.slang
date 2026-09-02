// Froxel fog material + directional-light injection pass.
// Fills a 3D frustum-aligned froxel volume with pre-lit single-scattering coefficients in rgb and
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

uint pcgHash(uint v) {
    v = v * 747796405u + 2891336453u;
    uint w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}
float rnd(inout uint s) {
    s = pcgHash(s);
    return float(s >> 8u) * (1.0 / 16777216.0);
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

[shader("compute")]
[numthreads(8, 8, 4)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    int3 dim = int3(pc.froxelDim.x, pc.froxelDim.y, pc.froxelDepth);
    if (int(dispatchId.x) >= dim.x || int(dispatchId.y) >= dim.y || int(dispatchId.z) >= dim.z) return;

    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];

    uint seed = dispatchId.x * 1973u + dispatchId.y * 9277u + dispatchId.z * 26699u
            + pc.frameIndex * 2654435761u;
    float jitterX = (rnd(seed) - 0.5) * pc.jitterStrength;
    float jitterY = (rnd(seed) - 0.5) * pc.jitterStrength;
    float jitterZ = (rnd(seed) - 0.5) * pc.jitterStrength;

    float2 uv = (float2(dispatchId.xy) + float2(0.5, 0.5) + float2(jitterX, jitterY) + pc.jitterOffset)
            / float2(dim.xy);
    float2 ndc = uv * 2.0 - 1.0;
    float3 rayDir = primaryRayDirFroxel(ndc, worldPush);

    float linear01 = (float(dispatchId.z) + 0.5 + jitterZ) / float(dim.z);
    float nearP = max(pc.nearPlane, 0.1);
    float farP = max(pc.farPlane, nearP + 1.0);
    float depth = nearP * pow(farP / nearP, linear01);
    depth = clamp(depth, nearP, farP);
    depth = min(depth, pc.maxDistance);

    float3 rebasedPos = worldPush.camOffset + rayDir * depth;
    float3 worldPos = rebasedPos + pc.terrainOrigin;

    float height = worldPos.y;
    float heightFactor = exp(-max(height - 64.0, 0.0) * pc.heightFalloff);
    float highAltitudeFade = saturate(1.0 - max(height - 128.0, 0.0) * 0.01);
    float distanceFade = 1.0 / (1.0 + depth * 0.001);

    float baseDensity = pc.density * heightFactor * highAltitudeFade * distanceFade;
    float3 noisePos = worldPos * 0.02;
    float n = frac(sin(dot(noisePos, float3(12.9898, 78.233, 45.164))) * 43758.5453);
    n = lerp(0.8, 1.2, n);
    baseDensity *= n;
    baseDensity = max(baseDensity, 0.0);

    float extinction = baseDensity * pc.extinction;
    float scatteringCoeff = baseDensity * pc.scattering;
    if (extinction <= 1.0e-6 && scatteringCoeff <= 1.0e-6) {
        froxelVolume[dispatchId] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float3 lighting = float3(0.0, 0.0, 0.0);
    float3 sunIllum = pc.sunIlluminance * pc.sunIntensity;
    float3 moonIllum = pc.moonIlluminance * pc.moonIntensity;
    bool sunEnabled = max(sunIllum.r, max(sunIllum.g, sunIllum.b)) > 1.0e-6;
    bool moonEnabled = max(moonIllum.r, max(moonIllum.g, moonIllum.b)) > 1.0e-6;
    float3 sunDir = sunEnabled ? normalize(pc.sunDir) : float3(0.0, 1.0, 0.0);
    float3 moonDir = moonEnabled ? normalize(pc.moonDir) : float3(0.0, 1.0, 0.0);

    // The host keeps only the dominant celestial active for fog, so one froxel rgb triplet is enough to
    // store the pre-shadowed directional illumination consumed later by the screen integration pass.
    if (sunEnabled) {
        ShadowResult sunShadow = traceShadowWithTransmission(rebasedPos + sunDir * 0.01, sunDir, 10000.0);
        lighting += sunIllum * sunShadow.transmittance;
    }
    if (moonEnabled) {
        ShadowResult moonShadow = traceShadowWithTransmission(rebasedPos + moonDir * 0.01, moonDir, 10000.0);
        lighting += moonIllum * moonShadow.transmittance;
    }

    froxelVolume[dispatchId] = float4(float3(scatteringCoeff) * lighting, extinction);
}
