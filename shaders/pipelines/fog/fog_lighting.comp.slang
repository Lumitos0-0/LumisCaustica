// Primary-view froxel lighting. The fields are deliberately independent: directional shafts, diagnostic
// local emitters, surface/sky GI, and the world-space emitter-reuse proposal can be inspected or disabled
// without changing the others. Extinction is applied once by fog_integrate.
import world_common;
import sky;
import fog_common;

// Keep the descriptors local to this entry point. This prevents basename collisions with the other
// pipeline-local bindings modules during reflection generation.
[[vk::binding(0, 0)]] public RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture2D<float4> directVolume;
[[vk::binding(2, 0)]] [format("rgba16f")] public RWTexture2D<float4> localVolume;
[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture2D<float4> giVolume;
[[vk::binding(4, 0)]] [format("rgba16f")] public RWTexture2D<float4> cacheVolume;
[[vk::binding(5, 0)]] [format("rgba16f")] public RWTexture2D<float4> giAuxVolume;
[[vk::binding(6, 0)]] public Sampler2D directHistory;
[[vk::binding(7, 0)]] public Sampler2D localHistory;
[[vk::binding(8, 0)]] public Sampler2D giHistory;
[[vk::binding(9, 0)]] public Sampler2D giAuxHistory;
[[vk::binding(10, 0)]] public Sampler2D cacheHistory;
[[vk::binding(11, 0)]] public Sampler2D sceneImage;
[[vk::binding(12, 0)]] public Sampler2D depthImage;
[[vk::binding(13, 0)]] public Sampler2D skyViewLut;
[[vk::binding(14, 0)]] public Sampler2D transmittanceLut;

[[vk::push_constant]] FogLightingPush pc;

static const float FOG_PI = 3.14159265359;
static const float FOG_NEAR_DISTANCE = 0.5;
static const float FOG_LOCAL_MAX_DISTANCE = 192.0;
static const uint FOG_SECONDARY_MASK = 0x01u;
static const float FOG_RAY_TMIN = 0.01;
static const float3 FOG_LUMA = float3(0.27222872, 0.67408177, 0.05368952);

uint fogPcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint word = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (word >> 22u) ^ word;
}

float fogRandom(inout uint state) {
    state = fogPcg(state);
    return float(state) / 4294967296.0;
}

// A tiny Owen-free Halton pair with a per-world-cell Cranley-Patterson shift. The frame index advances the
// low-discrepancy sequence, while the cell shift prevents neighboring froxels from tracing identical rays.
float fogRadicalInverse2(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    return float(bits) * 2.3283064365386963e-10;
}

float fogRadicalInverse3(uint value) {
    float result = 0.0;
    float inverse = 1.0 / 3.0;
    for (uint i = 0u; i < 12u; i++) {
        uint digit = value - (value / 3u) * 3u;
        result += float(digit) * inverse;
        value /= 3u;
        inverse /= 3.0;
    }
    return result;
}

float2 fogHalton2(uint index) {
    return float2(fogRadicalInverse2(index), fogRadicalInverse3(index));
}

float2 fogSequence2(uint index, uint cellHash) {
    uint state = fogPcg(cellHash ^ 0x9e3779b9u);
    float2 shift = float2(fogRandom(state), fogRandom(state));
    return frac(fogHalton2(index + 1u) + shift);
}

float fogHg(float cosTheta, float g) {
    float gg = clamp(g, -0.95, 0.95);
    float denom = max(1.0 + gg * gg - 2.0 * gg * cosTheta, 1.0e-4);
    return (1.0 - gg * gg) / (4.0 * FOG_PI * pow(denom, 1.5));
}

float3 fogPhaseDirection(float3 viewDirection, float g, float2 xi, out float pdf) {
    float gg = clamp(g, -0.95, 0.95);
    float cosTheta;
    if (abs(gg) < 1.0e-3) {
        cosTheta = 1.0 - 2.0 * xi.x;
    } else {
        float k = (1.0 - gg * gg) / max(1.0 - gg + 2.0 * gg * xi.x, 1.0e-5);
        cosTheta = (1.0 + gg * gg - k * k) / (2.0 * gg);
        cosTheta = clamp(cosTheta, -1.0, 1.0);
    }
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    float phi = 2.0 * FOG_PI * xi.y;
    float3 axis = normalize(-viewDirection);
    float3 tangent = abs(axis.y) < 0.9
            ? normalize(cross(axis, float3(0.0, 1.0, 0.0)))
            : normalize(cross(axis, float3(1.0, 0.0, 0.0)));
    float3 bitangent = cross(axis, tangent);
    float3 direction = normalize(axis * cosTheta
            + tangent * (sinTheta * cos(phi))
            + bitangent * (sinTheta * sin(phi)));
    pdf = fogHg(cosTheta, gg);
    return direction;
}

float fogUnpackUnsignedFloat(uint bits, uint mantissaBits) {
    uint mantissaMask = (1u << mantissaBits) - 1u;
    uint mantissa = bits & mantissaMask;
    uint exponent = (bits >> mantissaBits) & 31u;
    if (exponent == 0u) {
        return float(mantissa) * exp2(float(1 - 15 - int(mantissaBits)));
    }
    return (1.0 + float(mantissa) / float(1u << mantissaBits))
            * exp2(float(int(exponent) - 15));
}

float3 fogLightRadiance(Light light) {
    uint packed = light.le;
    float r = fogUnpackUnsignedFloat(packed & 0x7ffu, 6u);
    float g = fogUnpackUnsignedFloat((packed >> 11u) & 0x7ffu, 6u);
    float b = fogUnpackUnsignedFloat((packed >> 22u) & 0x3ffu, 5u);
    return float3(r, g, b);
}

float3 fogLightHalfU(Light light) {
    float2 a = unpackHalf2(light.halfUxy);
    float2 b = unpackHalf2(light.halfUzVx);
    return float3(a.x, a.y, b.x);
}

float3 fogLightHalfV(Light light) {
    float2 a = unpackHalf2(light.halfUzVx);
    float2 b = unpackHalf2(light.halfVyz);
    return float3(a.y, b.x, b.y);
}

float3 fogLightNormal(Light light, float3 crossUv) {
    float3 n = normalize(crossUv);
    return (light.section & 0x40000000u) != 0u ? -n : n;
}

bool fogVisible(float3 origin, float3 direction, float maxDistance) {
    RayDesc ray;
    ray.Origin = origin;
    ray.TMin = FOG_RAY_TMIN;
    ray.Direction = direction;
    ray.TMax = maxDistance;

    // The volume uses opaque first-hit visibility. The surface path remains responsible for alpha-tested
    // shadow semantics; keeping this query bounded prevents local diagnostic work from dominating the pass.
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(topLevelAS, RAY_FLAG_NONE, FOG_SECONDARY_MASK, ray);
    while (query.Proceed()) {
    }
    return query.CommittedStatus() == COMMITTED_NOTHING;
}

float3 fogSkyRadiance(SkyState sky, float3 direction) {
    float3 result709 = float3(0.0);
    float2 sunUv = skyViewLutUv(sky.viewerRadiusKm, direction.y,
            lightViewCosine(direction, sky.sunDir), SKY_VIEW_BODY_SUN);
    float2 moonUv = skyViewLutUv(sky.viewerRadiusKm, direction.y,
            lightViewCosine(direction, sky.moonDir), SKY_VIEW_BODY_MOON);
    result709 += skyViewLut.SampleLevel(sunUv, 0.0).rgb;
    result709 += skyViewLut.SampleLevel(moonUv, 0.0).rgb;
    return bt709ToAcesCg(max(result709, float3(0.0)));
}

float3 fogCelestialRadiance(SkyState sky, bool sun) {
    float3 sourceDirection = sun ? sky.sunDir : sky.moonDir;
    float sourceIntensity = sun ? sky.sunIlluminance : sky.moonIlluminance;
    float angularRadius = sun ? sky.sunAngularRadius : sky.moonAngularRadius;
    float3 transmittance = transmittanceToSpace(
            transmittanceLut, sky.viewerRadiusKm, sourceDirection, angularRadius);
    return bt709ToAcesCg(float3(sourceIntensity) * transmittance);
}

bool fogFindLightCell(WorldPush push, float3 position, out LightGridCell cell, out int3 cellCoord) {
    cell.spanOffset = 0u;
    cell.spanCount = 0u;
    cell.invWeightSum = 0.0;
    cellCoord = int3(0, 0, 0);
    if (pc.lightGridCellAddr == 0 || pc.lightGridSpanAddr == 0) return false;
    int3 coord = int3(floor((position - push.lightGridOrigin.xyz) / push.lightGridOrigin.w));
    if (coord.x < 0 || coord.y < 0 || coord.z < 0
            || coord.x >= push.lightGridDims.x || coord.y >= push.lightGridDims.y
            || coord.z >= push.lightGridDims.z) return false;
    uint linear = (uint(coord.z) * uint(push.lightGridDims.y) + uint(coord.y))
            * uint(push.lightGridDims.x) + uint(coord.x);
    cell = ConstPtr<LightGridCell>(pc.lightGridCellAddr)[linear];
    cellCoord = coord;
    return cell.spanCount > 0u;
}

void fogSelectGlobal(WorldPush push, inout uint seed, out uint lightIndex) {
    uint count = max(push.lightCount, 1u);
    float aliasSample = fogRandom(seed) * float(count);
    uint column = min(uint(aliasSample), count - 1u);
    if (pc.lightAliasAddr != 0) {
        LightAlias alias = ConstPtr<LightAlias>(pc.lightAliasAddr)[column];
        lightIndex = aliasSample - float(column) < alias.accept ? column : alias.aliasIndex;
    } else {
        lightIndex = column;
    }
}

void fogSelectSection(uint firstLight, uint lightCount, inout uint seed, out uint lightIndex) {
    uint count = max(lightCount, 1u);
    float aliasSample = fogRandom(seed) * float(count);
    uint column = min(uint(aliasSample), count - 1u);
    if (pc.lightLocalAliasAddr != 0) {
        LightAlias alias = ConstPtr<LightAlias>(pc.lightLocalAliasAddr)[firstLight + column];
        uint selected = aliasSample - float(column) < alias.accept ? column : alias.aliasIndex;
        lightIndex = firstLight + selected;
    } else {
        lightIndex = firstLight + column;
    }
}

void fogSelectCell(LightGridCell cell, inout uint seed, out uint lightIndex) {
    ConstPtr<LightGridSpan> spans = ConstPtr<LightGridSpan>(pc.lightGridSpanAddr);
    float aliasSample = fogRandom(seed) * float(cell.spanCount);
    uint column = min(uint(aliasSample), cell.spanCount - 1u);
    LightGridSpan span = spans[cell.spanOffset + column];
    bool self = aliasSample - float(column) < span.accept;
    uint firstLight = self ? span.firstLight : span.aliasFirstLight;
    uint count = self ? (span.packedLightCounts & 0xffffu) : (span.packedLightCounts >> 16u);
    fogSelectSection(firstLight, count, seed, lightIndex);
}

float fogLightProposalPdf(WorldPush push, Light light, float3 radiance,
                          LightGridCell cell, int3 cellCoord, float localProbability) {
    float3 halfU = fogLightHalfU(light);
    float3 halfV = fogLightHalfV(light);
    float area = 4.0 * length(cross(halfU, halfV));
    float power = max(0.0, area * dot(radiance, FOG_LUMA));
    float globalPdf = pc.lightAliasAddr != 0 && push.lightRebase.w > 0.0
            ? power * push.lightRebase.w : 1.0 / float(max(push.lightCount, 1u));
    float localPdf = 0.0;
    if (localProbability > 0.0 && power > 0.0) {
        int3 delta = int3(light.section & 0x3ffu,
                (light.section >> 10u) & 0x3ffu,
                (light.section >> 20u) & 0x3ffu) - cellCoord;
        if (all(abs(delta) <= int3(2, 2, 2))) {
            float3 deltaF = float3(delta);
            localPdf = power * cell.invWeightSum / max(1.0, dot(deltaF, deltaF));
        }
    }
    return max(localProbability * localPdf + (1.0 - localProbability) * globalPdf, 1.0e-8);
}

// One area-emitter proposal. The selection PDF is reconstructed from the same global/grid aliases used by
// the surface path, so off-screen resident emitters retain support without a modulo light-index bias.
float3 fogEmitterSample(WorldPush push, float3 position, float3 viewDirection,
                        inout uint seed, bool useVisibility) {
    if (pc.lightBufAddr == 0 || push.lightCount == 0u) return float3(0.0);
    LightGridCell cell;
    int3 cellCoord;
    bool hasCell = fogFindLightCell(push, position, cell, cellCoord);
    float localProbability = hasCell ? 0.75 : 0.0;
    bool chooseLocal = hasCell && fogRandom(seed) < localProbability;
    uint lightIndex;
    if (chooseLocal) fogSelectCell(cell, seed, lightIndex);
    else fogSelectGlobal(push, seed, lightIndex);

    Light light = ConstPtr<Light>(pc.lightBufAddr)[lightIndex];
    float3 halfU = fogLightHalfU(light);
    float3 halfV = fogLightHalfV(light);
    float3 crossUv = cross(halfU, halfV);
    float area = 4.0 * length(crossUv);
    if (area <= 1.0e-6) return float3(0.0);
    float2 square = float2(fogRandom(seed), fogRandom(seed)) * 2.0 - 1.0;
    float3 samplePosition = light.pos + push.lightRebase.xyz
            + square.x * halfU + square.y * halfV;
    float3 toLight = samplePosition - position;
    float distanceSquared = dot(toLight, toLight);
    if (distanceSquared <= 1.0e-5) return float3(0.0);
    float distanceToLight = sqrt(distanceSquared);
    float3 lightDirection = toLight / distanceToLight;
    float emitterCosine = max(0.0, dot(fogLightNormal(light, crossUv), -lightDirection));
    float shadowDistance = max(FOG_RAY_TMIN + 0.001,
            min(FOG_LOCAL_MAX_DISTANCE, distanceToLight * 0.999));
    if (emitterCosine <= 0.0 || (useVisibility && !fogVisible(position, lightDirection, shadowDistance))) {
        return float3(0.0);
    }
    float3 radiance = fogLightRadiance(light);
    float selectionPdf = fogLightProposalPdf(push, light, radiance, cell, cellCoord, localProbability);
    float phase = fogHg(dot(viewDirection, -lightDirection), pc.settings.x);
    // The rectangle point is uniform, so area * cosine / distance² is the solid-angle estimate. Dividing by
    // the hierarchical light PDF makes this a valid proposal estimator rather than a uniform-light guess.
    return radiance * (area * emitterCosine / distanceSquared) * phase / selectionPdf;
}

float3 fogLocalRadiance(WorldPush push, float3 position, float3 viewDirection, inout uint seed) {
    if (pc.settings.y <= 0.0 || pc.lightBufAddr == 0 || push.lightCount == 0u) return float3(0.0);
    uint samples = clamp(uint(pc.settings.w), 1u, 8u);
    float3 result = float3(0.0);
    for (uint i = 0u; i < samples; i++) {
        result += fogEmitterSample(push, position, viewDirection, seed, true);
    }
    return result / float(samples);
}

// World-space keyed emitter reuse. The immutable hierarchy/grid is the proposal table; a quantized world
// cell gives neighboring froxels the same reservoir key and temporal history carries that low-frequency
// proposal across frames. This is intentionally an emissive/surface-light reuse path, not medium transport
// through arbitrary secondary rays.
float3 fogCacheRadiance(WorldPush push, float3 position, float3 viewDirection) {
    if (pc.giSettings.w <= 0.0 || pc.lightBufAddr == 0 || push.lightCount == 0u) return float3(0.0);
    float cellSize = max(pc.giSettings.w, 1.0);
    float3 hierarchyPosition = position - push.lightRebase.xyz;
    int3 cell = int3(floor(hierarchyPosition / cellSize));
    uint cellHash = fogPcg(uint(cell.x) * 73856093u ^ uint(cell.y) * 19349663u
            ^ uint(cell.z) * 83492791u);
    // A small world-cell reservoir is shared by every froxel that lands in this cell. Four candidates are
    // rotated over a four-frame low-discrepancy cycle, then one survivor is selected by its contribution
    // weight. The W = sum(w)/(M*w_survivor) factor is the usual one-step reservoir estimator; it is not a
    // second medium ray and it remains supported by the published global/grid proposal PDFs. Because the
    // key is world-space rather than screen-space, a resident emitter can influence off-screen-to-on-screen
    // transitions without needing a projected history pixel. The camera atlas history below only filters
    // this low-frequency world reservoir, and cache reuse is opt-in by default.
    const uint candidateCount = 4u;
    float3 selected = float3(0.0);
    float selectedWeight = 0.0;
    float weightSum = 0.0;
    // Advance the candidate stream every frame. Repeating the same four candidates for four frames makes
    // the cache look like a small periodic animation instead of a temporally reused reservoir.
    uint frameSeed = push.frameIndex + 1u;
    uint reservoirSeed = fogPcg(cellHash ^ frameSeed * 2246822519u);
    for (uint candidate = 0u; candidate < candidateCount; candidate++) {
        uint seed = fogPcg(cellHash ^ frameSeed * 2654435761u
                ^ (candidate + 1u) * 3266489917u);
        float3 proposal = max(fogEmitterSample(push, position, viewDirection, seed, true), float3(0.0));
        float proposalWeight = max(dot(proposal, FOG_LUMA), 0.0);
        weightSum += proposalWeight;
        if (proposalWeight > 0.0
                && fogRandom(reservoirSeed) * weightSum < proposalWeight) {
            selected = proposal;
            selectedWeight = proposalWeight;
        }
    }
    return selectedWeight > 1.0e-6
            ? selected * (weightSum / (float(candidateCount) * selectedWeight))
            : float3(0.0);
}

bool fogSceneRadiance(WorldPush push, float3 hitPosition, out float3 radiance) {
    float3 cameraRelative = hitPosition - push.camOffset;
    float4 clip = mul(push.curViewProj, float4(cameraRelative, 1.0));
    if (clip.w <= 0.0) {
        radiance = float3(0.0);
        return false;
    }
    float2 uv = clip.xy / clip.w * 0.5 + 0.5;
    if (any(uv < float2(0.0)) || any(uv > float2(1.0))) {
        radiance = float3(0.0);
        return false;
    }
    float hitDepth = clip.z / clip.w;
    float sceneDepth = depthImage.SampleLevel(uv, 0.0).r;
    // With reversed-Z, a value lower than the primary depth is farther away. Reject a back wall hidden by
    // the primary surface instead of copying the foreground pixel into the fog volume.
    if (hitDepth <= 0.0 || hitDepth > 1.001
            || (sceneDepth > 1.0e-5 && hitDepth < sceneDepth - 0.002)) {
        radiance = float3(0.0);
        return false;
    }
    radiance = max(sceneImage.SampleLevel(uv, 0.0).rgb / max(push.preExposure, 1.0e-6), float3(0.0));
    return true;
}

float4 fogHistorySample4(Sampler2D history, float2 uv, float z, float3 dimensions) {
    float gx = max(dimensions.x, 1.0);
    float gy = max(dimensions.y, 1.0);
    float gz = max(dimensions.z, 1.0);
    float zf = clamp(z, 0.0, 1.0) * (gz - 1.0);
    uint z0 = uint(floor(zf));
    uint z1 = min(z0 + 1u, uint(gz - 1.0));
    float zBlend = zf - float(z0);
    float2 atlasSize = float2(gx, gy * gz);
    // Keep the lookup on the current froxel row at the frustum edge; sampling exactly on an atlas edge
    // would linearly bleed into the next packed depth slice.
    float2 localTexel = clamp(uv * float2(gx, gy), 0.5, float2(gx, gy) - 0.5);
    float2 uv0 = (localTexel + float2(0.0, float(z0) * gy)) / atlasSize;
    float2 uv1 = (localTexel + float2(0.0, float(z1) * gy)) / atlasSize;
    return lerp(history.SampleLevel(uv0, 0.0), history.SampleLevel(uv1, 0.0), zBlend);
}

float4 fogHistorySpatialSample4(Sampler2D history, float2 uv, float z, float3 dimensions, float radius) {
    float4 center = fogHistorySample4(history, uv, z, dimensions);
    if (radius <= 0.001) return center;
    float2 texel = radius / max(dimensions.xy, float2(1.0));
    float4 filtered = center * 4.0;
    filtered += fogHistorySample4(history, clamp(uv + float2(texel.x, 0.0), 0.0, 1.0), z, dimensions);
    filtered += fogHistorySample4(history, clamp(uv - float2(texel.x, 0.0), 0.0, 1.0), z, dimensions);
    filtered += fogHistorySample4(history, clamp(uv + float2(0.0, texel.y), 0.0, 1.0), z, dimensions);
    filtered += fogHistorySample4(history, clamp(uv - float2(0.0, texel.y), 0.0, 1.0), z, dimensions);
    return filtered / 8.0;
}

float fogGuideWeight(float2 currentGuide, float4 historyGuide) {
    // Hit coverage is a Monte Carlo estimate, not a material ID. With two GI rays it can move between
    // 0, 0.5, and 1 even when the camera and scene are static; treating those bins as a hard rejection
    // prevents the volume from ever accumulating in exactly the noisy regions we want to reconstruct.
    float currentCoverage = saturate(currentGuide.y);
    float historyCoverage = saturate(historyGuide.y);
    float coverageDelta = abs(currentCoverage - historyCoverage);
    float coverageWeight = exp(-0.75 * coverageDelta);

    // Hit distance is useful when both estimates actually contain a surface hit, but a sky estimate has no
    // meaningful distance. Use it as a soft confidence term with a tolerance proportional to the hit
    // coverage. Primary reversed-Z depth below remains the hard foreground/sky discontinuity test.
    if (currentCoverage > 0.05 && historyCoverage > 0.05) {
        float tolerance = 0.10 + 0.42 * max(currentCoverage, historyCoverage);
        float hitWeight = exp(-abs(currentGuide.x - historyGuide.x) / max(tolerance, 0.08));
        return coverageWeight * max(hitWeight, 0.15);
    }
    return coverageWeight;
}

// gDepth is reversed-Z projected depth: sky is near zero and a nearer surface has a larger value. The
// direct field's alpha stores the previous primary depth alongside its RGB history, so every channel can
// reject a reprojection that crosses a primary surface or changes depth by more than a conservative guide
// tolerance without treating the non-linear value as linear distance.
float fogPrimaryDepthWeight(float currentDepth, float historyDepth) {
    bool currentSurface = currentDepth > 1.0e-5;
    bool historySurface = historyDepth > 1.0e-5;
    if (currentSurface != historySurface) return 0.0;
    if (!currentSurface) return 1.0;
    float tolerance = 0.02 + 0.12 * max(max(currentDepth, historyDepth), 0.05);
    float delta = abs(currentDepth - historyDepth);
    return delta > tolerance ? 0.0 : saturate(1.0 - delta / tolerance);
}

float4 fogGiHistorySpatialSample4(Sampler2D history, Sampler2D auxHistory,
                                  float2 uv, float z, float3 dimensions,
                                  float2 currentGuide, float currentFroxelZ,
                                  float radius, out float guideWeight,
                                  out float neighborhoodMinLuma, out float neighborhoodMaxLuma) {
    float4 center = fogHistorySample4(history, uv, z, dimensions);
    float4 centerGuide = fogHistorySample4(auxHistory, uv, z, dimensions);
    // The reprojected logarithmic slice is a cheap froxel-depth guide. A color-only reprojection can
    // otherwise pull a far-cell surface into a newly exposed near cell even when occupancy happens to
    // match; keep a broad tolerance for camera motion, but reject an actual depth-cell jump.
    float depthDelta = abs(currentFroxelZ - z);
    float depthWeight = depthDelta > 0.22 ? 0.0 : saturate(1.0 - depthDelta * 3.0);
    guideWeight = fogGuideWeight(currentGuide, centerGuide) * depthWeight;
    float centerLuma = dot(max(center.rgb, float3(0.0)), FOG_LUMA);
    neighborhoodMinLuma = centerLuma;
    neighborhoodMaxLuma = centerLuma;
    if (radius <= 0.001) return center;
    float2 texel = radius / max(dimensions.xy, float2(1.0));
    float4 taps[4];
    taps[0] = fogHistorySample4(history, clamp(uv + float2(texel.x, 0.0), 0.0, 1.0), z, dimensions);
    taps[1] = fogHistorySample4(history, clamp(uv - float2(texel.x, 0.0), 0.0, 1.0), z, dimensions);
    taps[2] = fogHistorySample4(history, clamp(uv + float2(0.0, texel.y), 0.0, 1.0), z, dimensions);
    taps[3] = fogHistorySample4(history, clamp(uv - float2(0.0, texel.y), 0.0, 1.0), z, dimensions);
    float2 tapUv[4] = {
        clamp(uv + float2(texel.x, 0.0), 0.0, 1.0), clamp(uv - float2(texel.x, 0.0), 0.0, 1.0),
        clamp(uv + float2(0.0, texel.y), 0.0, 1.0), clamp(uv - float2(0.0, texel.y), 0.0, 1.0)
    };
    float4 result = center * (4.0 * guideWeight);
    float weight = 4.0 * guideWeight;
    for (uint i = 0u; i < 4u; i++) {
        float4 tapGuide = fogHistorySample4(auxHistory, tapUv[i], z, dimensions);
        float tapWeight = fogGuideWeight(currentGuide, tapGuide) * depthWeight;
        result += taps[i] * tapWeight;
        weight += tapWeight;
        if (tapWeight > 0.0) {
            float tapLuma = dot(max(taps[i].rgb, float3(0.0)), FOG_LUMA);
            neighborhoodMinLuma = min(neighborhoodMinLuma, tapLuma);
            neighborhoodMaxLuma = max(neighborhoodMaxLuma, tapLuma);
        }
    }
    guideWeight = weight / 8.0;
    return weight > 1.0e-4 ? result / weight : center;
}

float3 fogClampHistoryLuminance(float3 history, float currentLuma, float variance,
                                 float neighborhoodMinLuma, float neighborhoodMaxLuma) {
    float historyLuma = max(dot(history, FOG_LUMA), 0.0);
    if (historyLuma <= 1.0e-5) return float3(0.0);

    // Clamp against the reprojected neighborhood, not just this frame's single stochastic estimate. The
    // latter collapses a bright history to almost zero whenever the two GI rays miss on one frame, which
    // looks like grain instead of temporal reconstruction. The variance term keeps a high-energy but
    // genuinely plausible history inside a bounded range when a neighboring ray is currently dark.
    float sigma = sqrt(max(variance, 0.0) + 0.0025
            + max(currentLuma, neighborhoodMaxLuma) * max(currentLuma, neighborhoodMaxLuma) * 0.10);
    float low = max(0.0, min(currentLuma, neighborhoodMinLuma) - 2.5 * sigma);
    float high = max(currentLuma, neighborhoodMaxLuma) + 2.5 * sigma;
    return history * (clamp(historyLuma, low, high) / historyLuma);
}

float3 fogFroxelPositionAt(WorldPush push, float2 uv, float slice, out float3 rayDirection) {
    float2 ndc = uv * 2.0 - 1.0;
    float4 nearH = mul(push.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(push.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    rayDirection = normalize(farP - nearP);
    float distance = exp2(lerp(log2(FOG_NEAR_DISTANCE),
            log2(max(pc.grid.w, FOG_NEAR_DISTANCE + 1.0)), slice));
    float nearAlongRay = length(nearP);
    return nearP + rayDirection * max(distance - nearAlongRay, 0.0) + push.camOffset;
}

bool fogPreviousCoordinates(WorldPush push, float3 position, out float2 uv, out float z) {
    float3 currentCameraRelative = position - push.camOffset;
    float3 previousCameraRelative = currentCameraRelative + push.camDelta;
    float4 clip = mul(push.prevViewProj, float4(previousCameraRelative, 1.0));
    if (clip.w <= 0.0) {
        uv = float2(0.0);
        z = 0.0;
        return false;
    }
    uv = clip.xy / clip.w * 0.5 + 0.5;
    float distance = length(previousCameraRelative);
    float nearLog = log2(FOG_NEAR_DISTANCE);
    float farLog = log2(max(pc.grid.w, FOG_NEAR_DISTANCE + 1.0));
    z = (log2(max(distance, FOG_NEAR_DISTANCE)) - nearLog) / max(farLog - nearLog, 1.0e-5);
    return all(uv >= float2(0.0)) && all(uv <= float2(1.0)) && z >= 0.0 && z <= 1.0;
}

float3 fogIndirectRadiance(WorldPush push, SkyState sky, float3 position, float3 viewDirection,
                           uint cellHash, out float guideHitDistance, out float guideCoverage) {
    guideHitDistance = 0.0;
    guideCoverage = 0.0;
    if (pc.giSettings.x <= 0.0) return float3(0.0);
    uint samples = clamp(uint(pc.giSettings.x), 1u, 4u);
    float3 radiance = float3(0.0);
    float maxRayDistance = max(FOG_RAY_TMIN + 0.001, min(pc.grid.w, FOG_LOCAL_MAX_DISTANCE));
    for (uint sample = 0u; sample < samples; sample++) {
        uint sequence = push.frameIndex * samples + sample + (cellHash & 7u);
        float phasePdf;
        float3 direction = fogPhaseDirection(viewDirection, pc.settings.x, fogSequence2(sequence, cellHash), phasePdf);
        // The secondary direction is sampled from the HG phase itself. Keep the explicit phase/PDF ratio
        // in the estimator rather than silently assuming a uniform direction distribution.
        float phaseWeight = fogHg(dot(direction, -viewDirection), pc.settings.x)
                / max(phasePdf, 1.0e-6);
        float3 origin = position + direction * FOG_RAY_TMIN;
        RayDesc ray;
        ray.Origin = origin;
        ray.TMin = FOG_RAY_TMIN;
        ray.Direction = direction;
        ray.TMax = maxRayDistance;

        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
        query.TraceRayInline(topLevelAS, RAY_FLAG_NONE, FOG_SECONDARY_MASK, ray);
        while (query.Proceed()) {
        }
        if (query.CommittedStatus() == COMMITTED_NOTHING) {
            radiance += fogSkyRadiance(sky, direction) * phaseWeight;
        } else {
            float hitDistance = query.CommittedRayT();
            float3 hitPosition = origin + direction * hitDistance;
            float3 surface;
            if (fogSceneRadiance(push, hitPosition, surface)) {
                radiance += surface * phaseWeight;
                guideHitDistance += saturate(hitDistance / maxRayDistance);
                guideCoverage += 1.0;
            } else {
                // A surface outside the primary view has no stable screen radiance; classify it as a sky-like
                // miss rather than allowing a stale surface history to cross the frustum boundary.
                radiance += fogSkyRadiance(sky, direction) * phaseWeight;
            }
        }
    }
    guideHitDistance /= float(samples);
    guideCoverage /= float(samples);
    return radiance / float(samples);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    uint gridX = max(uint(pc.grid.x), 1u);
    uint gridY = max(uint(pc.grid.y), 1u);
    uint gridZ = max(uint(pc.grid.z), 1u);
    uint atlasHeight = gridY * gridZ;
    if (dispatchId.x >= gridX || dispatchId.y >= atlasHeight) return;

    uint3 froxel = uint3(dispatchId.x, dispatchId.y % gridY, dispatchId.y / gridY);
    WorldPush push = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    SkyState sky = skyState(push);
    uint froxelHash = fogPcg((froxel.x + froxel.y * gridX + froxel.z * gridX * gridY)
            ^ push.frameIndex * 2654435761u);
    float2 centerUv = (float2(froxel.xy) + 0.5) / float2(gridX, gridY);
    float currentPrimaryDepth = saturate(depthImage.SampleLevel(centerUv, 0.0).r);
    float3 viewRay;
    float3 position = fogFroxelPositionAt(push, centerUv,
            float(froxel.z) / max(float(gridZ - 1u), 1.0), viewRay);
    float3 viewDirection = -viewRay;

    float3 direct = float3(0.0);
    if (pc.settings.z > 0.0 && pc.giSettings.z > 0.0) {
        uint directSamples = clamp(uint(pc.giSettings.z), 1u, 4u);
        float3 sunRadiance = fogCelestialRadiance(sky, true);
        float3 moonRadiance = fogCelestialRadiance(sky, false);
        for (uint sample = 0u; sample < directSamples; sample++) {
            float2 xi = fogSequence2(push.frameIndex * directSamples + sample, froxelHash);
            float2 sampleOffset = float2(0.0);
            if (directSamples == 2u) {
                sampleOffset = sample == 0u ? float2(-0.25, 0.0) : float2(0.25, 0.0);
            } else if (directSamples >= 4u) {
                sampleOffset = sample == 0u ? float2(-0.25, -0.25)
                        : sample == 1u ? float2(0.25, -0.25)
                        : sample == 2u ? float2(-0.25, 0.25) : float2(0.25, 0.25);
            }
            // The same xi drives XY and logarithmic-Z. That keeps the sample a coherent point in the froxel
            // rather than independently scrambling the shaft's screen and depth coordinates.
            float2 directUv = clamp((float2(froxel.xy) + 0.5 + sampleOffset + (xi - 0.5) * 0.5)
                    / float2(gridX, gridY), 0.0, 1.0);
            float sliceStep = 1.0 / max(float(gridZ - 1u), 1.0);
            float slice = clamp(float(froxel.z) * sliceStep + (xi.y - 0.5) * sliceStep, 0.0, 1.0);
            float3 directViewRay;
            float3 directPosition = fogFroxelPositionAt(push, directUv, slice, directViewRay);
            float3 directViewDirection = -directViewRay;
            if (max(sunRadiance.r, max(sunRadiance.g, sunRadiance.b)) > 0.0
                    && fogVisible(directPosition, sky.sunDir, 10000.0)) {
                direct += sunRadiance * fogHg(dot(directViewDirection, -sky.sunDir), pc.settings.x);
            }
            if (max(moonRadiance.r, max(moonRadiance.g, moonRadiance.b)) > 0.0
                    && fogVisible(directPosition, sky.moonDir, 10000.0)) {
                direct += moonRadiance * fogHg(dot(directViewDirection, -sky.moonDir), pc.settings.x);
            }
        }
        direct /= float(directSamples);
    }

    // Local emitter visibility is kept separate from the directional field. It is disabled in the production
    // default and uses the published power/grid hierarchy when explicitly enabled for diagnostics.
    uint localSeed = fogPcg(froxelHash ^ 0x68bc21ebu);
    float3 local = fogLocalRadiance(push, position, viewDirection, localSeed);

    // The cache proposal is world-cell keyed, so it can represent resident/off-screen emitters without
    // depending on whether their projected pixels were visible this frame.
    float3 cache = fogCacheRadiance(push, position, viewDirection);

    // Keep the low-discrepancy phase shift unique per froxel. Sharing one 2x2x2 hash makes eight
    // neighboring cells trace the same angular samples and turns a low sample count into visible blocks;
    // spatial filtering can provide correlation without making the sampling pattern itself blocky.
    uint giLinear = froxel.x + froxel.y * gridX + froxel.z * gridX * gridY;
    uint giHash = fogPcg(giLinear ^ 0x9e3779b9u);
    float giHitDistance;
    float giCoverage;
    float3 indirect = fogIndirectRadiance(push, sky, position, viewDirection,
            giHash, giHitDistance, giCoverage);
    // Keep the second moment from the unfiltered current estimate. Storing the square of the already
    // blended mean makes the variance collapse toward zero and turns the moment lane into decoration.
    float rawGiLuma = dot(max(indirect, float3(0.0)) / FOG_RADIANCE_SCALE, FOG_LUMA);
    float rawGiMoment2 = rawGiLuma * rawGiLuma;
    float2 currentUv = centerUv;
    float2 previousUv;
    float previousZ;
    bool hasHistoryCoordinates = fogPreviousCoordinates(push, position, previousUv, previousZ);
    float motionInFroxels = hasHistoryCoordinates
            ? length((previousUv - currentUv) * pc.grid.xy) : 1000.0;
    float translationPenalty = exp(-0.35 * length(push.camDelta));
    float primaryDepthWeight = 0.0;
    if (hasHistoryCoordinates) {
        float previousPrimaryDepth = fogHistorySample4(directHistory, previousUv, previousZ, pc.grid.xyz).a;
        primaryDepthWeight = fogPrimaryDepthWeight(currentPrimaryDepth, previousPrimaryDepth);
    }

    if (hasHistoryCoordinates && pc.temporalSettings.y > 0.0) {
        float4 history = fogHistorySpatialSample4(directHistory, previousUv, previousZ,
                pc.grid.xyz, max(pc.spatialSettings.y, 0.0));
        float3 historyRadiance = history.rgb * FOG_RADIANCE_SCALE;
        // Directional visibility is a hard discontinuity. A bounded luminance clamp plus motion attenuation
        // keeps a bright shaft from being dragged into a newly revealed occluder.
        float3 directLimit = direct * 2.0 + max(direct * 0.10, float3(0.01));
        historyRadiance = min(historyRadiance, directLimit);
        float currentFroxelZ = float(froxel.z) / max(float(gridZ - 1u), 1.0);
        float depthValidation = abs(previousZ - currentFroxelZ) > 0.18 ? 0.0
                : saturate(1.0 - abs(previousZ - currentFroxelZ) * 3.0);
        float directConfidence = dot(max(direct, float3(0.0)), FOG_LUMA) > 1.0e-5 ? 1.0 : 0.0;
        float weight = clamp(pc.temporalSettings.y * exp(-0.80 * motionInFroxels)
                * translationPenalty * depthValidation * primaryDepthWeight * directConfidence, 0.0, 0.90);
        direct = lerp(direct, historyRadiance, weight);
    }

    if (hasHistoryCoordinates && pc.temporalSettings.z > 0.0 && pc.settings.y > 0.0) {
        float4 history = fogHistorySpatialSample4(localHistory, previousUv, previousZ,
                pc.grid.xyz, max(pc.spatialSettings.z, 0.0));
        float3 historyRadiance = history.rgb * FOG_RADIANCE_SCALE;
        historyRadiance = min(historyRadiance, local * 3.0 + max(local * 0.25, float3(0.01)));
        float weight = clamp(pc.temporalSettings.z * exp(-0.65 * motionInFroxels)
                * translationPenalty * primaryDepthWeight, 0.0, 0.85);
        local = lerp(local, historyRadiance, weight);
    }

    if (hasHistoryCoordinates && pc.temporalSettings.w > 0.0 && pc.giSettings.w > 0.0) {
        float4 history = fogHistorySpatialSample4(cacheHistory, previousUv, previousZ,
                pc.grid.xyz, max(pc.spatialSettings.w, 0.0));
        float3 historyRadiance = history.rgb * FOG_RADIANCE_SCALE;
        historyRadiance = min(historyRadiance, cache * 3.0 + max(cache * 0.25, float3(0.01)));
        float weight = clamp(pc.temporalSettings.w * exp(-0.45 * motionInFroxels)
                * translationPenalty * primaryDepthWeight, 0.0, 0.90);
        cache = lerp(cache, historyRadiance, weight);
    }

    float2 currentGuide = float2(giHitDistance, giCoverage);
    float giHistoryGuideWeight = 0.0;
    float giHistoryBlendWeight = 0.0;
    float giNeighborhoodMinLuma = 0.0;
    float giNeighborhoodMaxLuma = 0.0;
    float4 giHistorySample = float4(0.0);
    float4 previousGiGuide = float4(0.0);
    if (hasHistoryCoordinates && pc.temporalSettings.x > 0.0 && pc.giSettings.x > 0.0) {
        previousGiGuide = fogHistorySample4(giAuxHistory, previousUv, previousZ, pc.grid.xyz);
        giHistorySample = fogGiHistorySpatialSample4(giHistory, giAuxHistory,
                previousUv, previousZ, pc.grid.xyz, currentGuide,
                float(froxel.z) / max(float(gridZ - 1u), 1.0),
                max(pc.spatialSettings.x, 0.0), giHistoryGuideWeight,
                giNeighborhoodMinLuma, giNeighborhoodMaxLuma);
        float3 historyRadiance = giHistorySample.rgb * FOG_RADIANCE_SCALE;
        float historyStoredLuma = dot(max(giHistorySample.rgb, float3(0.0)), FOG_LUMA);
        float historyVariance = max(giHistorySample.a - historyStoredLuma * historyStoredLuma, 0.0);
        historyRadiance = fogClampHistoryLuminance(historyRadiance,
                rawGiLuma * FOG_RADIANCE_SCALE,
                historyVariance * FOG_RADIANCE_SCALE * FOG_RADIANCE_SCALE,
                giNeighborhoodMinLuma * FOG_RADIANCE_SCALE,
                giNeighborhoodMaxLuma * FOG_RADIANCE_SCALE);
        float historyConfidence = saturate(previousGiGuide.w);
        float historyAgeWeight = exp(-max(previousGiGuide.z - 1.0, 0.0) * 0.02);
        float weight = clamp(pc.temporalSettings.x * giHistoryGuideWeight
                * exp(-0.55 * motionInFroxels) * translationPenalty * primaryDepthWeight
                * max(0.25, historyConfidence) * historyAgeWeight, 0.0, 0.92);
        giHistoryBlendWeight = weight;
        indirect = lerp(indirect, historyRadiance, weight);
    }

    float3 storedDirect = min(max(direct, float3(0.0)) / FOG_RADIANCE_SCALE, float3(HALF_MAX));
    float3 storedLocal = min(max(local, float3(0.0)) / FOG_RADIANCE_SCALE, float3(HALF_MAX));
    float3 storedCache = min(max(cache, float3(0.0)) / FOG_RADIANCE_SCALE, float3(HALF_MAX));
    float3 storedGi = min(max(indirect, float3(0.0)) / FOG_RADIANCE_SCALE, float3(HALF_MAX));
    float historyMeanLuma = dot(max(giHistorySample.rgb, float3(0.0)), FOG_LUMA);
    float historyMoment2 = max(giHistorySample.a, historyMeanLuma * historyMeanLuma);
    // Mix M2 with the raw current sample, not the square of the resolved mean. The latter loses the
    // between-sample term and makes variance-guided clamping ineffective after the first blend.
    float resolvedMoment2 = min(max(lerp(rawGiMoment2, historyMoment2, giHistoryBlendWeight), 0.0), HALF_MAX);
    float resolvedGuideHitDistance = giHitDistance;
    float resolvedGuideCoverage = giCoverage;
    float historyAge = 1.0;
    float historyConfidenceOut = giCoverage > 0.0 ? 0.75 : 0.5;
    if (giHistoryBlendWeight > 0.001 && previousGiGuide.w > 0.0) {
        // Keep the stochastic hit guide itself low frequency. Reusing it with the same confidence as the
        // radiance prevents the next frame from rejecting a valid history solely because two rays changed
        // their sky/surface classification bins.
        float guideReuse = min(giHistoryBlendWeight * 0.75, 0.75);
        resolvedGuideHitDistance = lerp(giHitDistance, previousGiGuide.x, guideReuse);
        resolvedGuideCoverage = lerp(giCoverage, previousGiGuide.y, guideReuse);
        historyAge = min(previousGiGuide.z + 1.0, 64.0);
        historyConfidenceOut = saturate(lerp(giCoverage > 0.0 ? 0.75 : 0.5,
                previousGiGuide.w, guideReuse));
    }
    giVolume[dispatchId.xy] = float4(storedGi, resolvedMoment2);
    giAuxVolume[dispatchId.xy] = float4(resolvedGuideHitDistance, resolvedGuideCoverage,
            historyAge, historyConfidenceOut);
    // Keep the primary reversed-Z depth in the direct history's spare alpha lane. The same guide is
    // consumed by direct, local, cache, and GI reprojection tests; no linear-depth assumption is made.
    directVolume[dispatchId.xy] = float4(storedDirect, currentPrimaryDepth);
    localVolume[dispatchId.xy] = float4(storedLocal, 1.0);
    cacheVolume[dispatchId.xy] = float4(storedCache, 1.0);
}
