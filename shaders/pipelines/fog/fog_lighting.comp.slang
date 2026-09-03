// Froxel lighting for the primary camera view.
//
// The two output fields intentionally have different estimators. Direct sunlight, moonlight, and local
// emissive rectangles are evaluated at the froxel with hard TLAS visibility and a forward HG phase, so
// shafts retain their directional shadow boundaries. GI is a one-bounce radiance-cache estimate: a small
// inline ray query lands on the already rendered surface image, then a reprojected, clamped history makes
// that low-frequency field quiet. Neither channel carries extinction; fog_integrate applies one shared
// medium model to both channels.
import world_common;
import sky;
import fog_common;

// Keep the descriptors in this entry-point source instead of a generic `bindings` module. The build
// contains several pipeline-local bindings files, and Slang's basename module lookup can otherwise resolve
// a different pipeline's module when generating reflection. Direct declarations make this shader's ABI
// unambiguous to both compileShaders and GenerateRtBindings.
[[vk::binding(0, 0)]] public RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture2D<float4> directVolume;
[[vk::binding(2, 0)]] [format("rgba16f")] public RWTexture2D<float4> giVolume;
[[vk::binding(3, 0)]] public Sampler2D giHistory;
[[vk::binding(4, 0)]] public Sampler2D sceneImage;
[[vk::binding(5, 0)]] public Sampler2D skyViewLut;
[[vk::binding(6, 0)]] public Sampler2D transmittanceLut;

[[vk::push_constant]] FogLightingPush pc;

static const float FOG_PI = 3.14159265359;
static const float FOG_NEAR_DISTANCE = 0.5;
static const float FOG_LOCAL_MAX_DISTANCE = 192.0;
static const uint FOG_SECONDARY_MASK = 0x01u;
static const float FOG_RAY_TMIN = 0.01;

uint fogPcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint word = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (word >> 22u) ^ word;
}

float fogRandom(inout uint state) {
    state = fogPcg(state);
    return float(state) / 4294967296.0;
}

float fogHg(float cosTheta, float g) {
    float gg = clamp(g, -0.95, 0.95);
    float denom = max(1.0 + gg * gg - 2.0 * gg * cosTheta, 1.0e-4);
    return (1.0 - gg * gg) / (4.0 * FOG_PI * pow(denom, 1.5));
}

float3 fogUniformSphere(inout uint state) {
    float z = fogRandom(state) * 2.0 - 1.0;
    float phi = fogRandom(state) * (2.0 * FOG_PI);
    float radius = sqrt(max(1.0 - z * z, 0.0));
    return float3(radius * cos(phi), z, radius * sin(phi));
}

float3 fogUnpackUnsignedFloat(uint bits, uint mantissaBits) {
    uint mantissaMask = (1u << mantissaBits) - 1u;
    uint mantissa = bits & mantissaMask;
    uint exponent = (bits >> mantissaBits) & 31u;
    if (exponent == 0u) {
        return float3(float(mantissa) * exp2(float(1 - 15 - int(mantissaBits))));
    }
    return float3((1.0 + float(mantissa) / float(1u << mantissaBits))
            * exp2(float(int(exponent) - 15)));
}

float3 fogLightRadiance(Light light) {
    uint packed = light.le;
    float r = fogUnpackUnsignedFloat(packed & 0x7ffu, 6u).x;
    float g = fogUnpackUnsignedFloat((packed >> 11u) & 0x7ffu, 6u).x;
    float b = fogUnpackUnsignedFloat((packed >> 22u) & 0x3ffu, 5u).x;
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

    // Inline queries deliberately force opaque traversal. The world TLAS still supplies the exact current
    // terrain/entity silhouette, while alpha-tested media are left to the surface path's dedicated shadow
    // SBT. This keeps the high-rate shaft pass bounded and deterministic.
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

float3 fogLocalRadiance(WorldPush push, float3 position, float3 viewDirection, inout uint seed) {
    if (pc.lightBufAddr == 0 || pc.lightCount == 0u || pc.settings.y <= 0.0) {
        return float3(0.0);
    }

    // Uniform light selection is intentionally capped at two samples. The light list is already a power
    // hierarchy for the surface path; fog uses its result as a broad, temporally filtered emissive field
    // rather than paying a second RIS reservoir for every froxel.
    const uint sampleCount = 2u;
    float3 result = float3(0.0);
    for (uint sample = 0u; sample < sampleCount; sample++) {
        seed = fogPcg(seed);
        uint lightIndex = seed % pc.lightCount;
        Light light = ConstPtr<Light>(pc.lightBufAddr)[lightIndex];
        float3 halfU = fogLightHalfU(light);
        float3 halfV = fogLightHalfV(light);
        float3 crossUv = cross(halfU, halfV);
        float area = 4.0 * length(crossUv);
        if (area <= 1.0e-6) {
            continue;
        }
        float2 square = float2(fogRandom(seed), fogRandom(seed)) * 2.0 - 1.0;
        float3 samplePosition = light.pos + push.lightRebase.xyz
                + square.x * halfU + square.y * halfV;
        float3 toLight = samplePosition - position;
        float distanceSquared = dot(toLight, toLight);
        if (distanceSquared <= 1.0e-5) {
            continue;
        }
        float distanceToLight = sqrt(distanceSquared);
        float3 lightDirection = toLight / distanceToLight;
        float emitterCosine = max(0.0, dot(fogLightNormal(light, crossUv), -lightDirection));
        float shadowDistance = max(FOG_RAY_TMIN + 0.001,
                min(FOG_LOCAL_MAX_DISTANCE, distanceToLight * 0.999));
        if (emitterCosine <= 0.0 || !fogVisible(position, lightDirection, shadowDistance)) {
            continue;
        }
        float solidAngleEstimate = area * emitterCosine / distanceSquared;
        result += fogLightRadiance(light) * solidAngleEstimate
                * fogHg(dot(viewDirection, lightDirection), pc.settings.x);
    }
    return result * (float(pc.lightCount) / float(sampleCount)) * pc.settings.y;
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
    // The primary path stores pre-exposed radiance. Return to absolute scene units before it enters the
    // fog history so a change in display exposure cannot become a false volumetric lighting change.
    radiance = max(sceneImage.SampleLevel(uv, 0.0).rgb / max(push.preExposure, 1.0e-6), float3(0.0));
    return true;
}

float3 fogHistorySample(Sampler2D history, float2 uv, float z, float3 dimensions) {
    float gx = max(dimensions.x, 1.0);
    float gy = max(dimensions.y, 1.0);
    float gz = max(dimensions.z, 1.0);
    float zf = clamp(z, 0.0, 1.0) * (gz - 1.0);
    uint z0 = uint(floor(zf));
    uint z1 = min(z0 + 1u, uint(gz - 1.0));
    float zBlend = zf - float(z0);
    float2 atlasSize = float2(gx, gy * gz);
    float2 localTexel = float2(uv.x * gx, uv.y * gy);
    float2 uv0 = (localTexel + float2(0.5, 0.5 + float(z0) * gy)) / atlasSize;
    float2 uv1 = (localTexel + float2(0.5, 0.5 + float(z1) * gy)) / atlasSize;
    return lerp(history.SampleLevel(uv0, 0.0).rgb,
            history.SampleLevel(uv1, 0.0).rgb, zBlend);
}

float3 fogIndirectRadiance(WorldPush push, SkyState sky, float3 position, inout uint seed) {
    uint samples = clamp(uint(pc.settings.w), 1u, 2u);
    float3 radiance = float3(0.0);
    for (uint sample = 0u; sample < samples; sample++) {
        float3 direction = fogUniformSphere(seed);
        float3 origin = position + direction * FOG_RAY_TMIN;
        RayDesc ray;
        ray.Origin = origin;
        ray.TMin = FOG_RAY_TMIN;
        ray.Direction = direction;
        ray.TMax = min(pc.grid.w, FOG_LOCAL_MAX_DISTANCE);

        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
        query.TraceRayInline(topLevelAS, RAY_FLAG_NONE, FOG_SECONDARY_MASK, ray);
        while (query.Proceed()) {
        }
        if (query.CommittedStatus() == COMMITTED_NOTHING) {
            // Sky view already contains atmospheric single + multiple scattering. It is sampled directly
            // instead of being sent through the camera-frustum medium a second time.
            radiance += fogSkyRadiance(sky, direction);
        } else {
            float hitDistance = query.CommittedRayT();
            float3 hitPosition = origin + direction * hitDistance;
            float3 surface;
            if (fogSceneRadiance(push, hitPosition, surface)) {
                radiance += surface;
            } else {
                // A surface outside the current primary view has no valid scene sample. Falling back to
                // the sky cache is low-frequency and prevents a black halo at the camera frustum boundary.
                radiance += fogSkyRadiance(sky, direction);
            }
        }
    }
    return radiance / float(samples);
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

float3 fogFroxelPosition(WorldPush push, uint3 id, uint3 dimensions, out float3 rayDirection) {
    float2 uv = (float2(id.xy) + 0.5) / float2(dimensions.xy);
    float2 ndc = uv * 2.0 - 1.0;
    float4 nearH = mul(push.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(push.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    rayDirection = normalize(farP - nearP);

    float slice = float(id.z) / max(float(dimensions.z - 1u), 1.0);
    float distance = exp2(lerp(log2(FOG_NEAR_DISTANCE), log2(max(pc.grid.w, FOG_NEAR_DISTANCE + 1.0)), slice));
    float nearAlongRay = length(nearP);
    return nearP + rayDirection * max(distance - nearAlongRay, 0.0) + push.camOffset;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    uint gridX = max(uint(pc.grid.x), 1u);
    uint gridY = max(uint(pc.grid.y), 1u);
    uint gridZ = max(uint(pc.grid.z), 1u);
    uint atlasHeight = gridY * gridZ;
    if (dispatchId.x >= gridX || dispatchId.y >= atlasHeight) {
        return;
    }

    uint3 froxel = uint3(dispatchId.x, dispatchId.y % gridY, dispatchId.y / gridY);
    WorldPush push = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    SkyState sky = skyState(push);
    float3 viewRay;
    float3 position = fogFroxelPosition(push, froxel, uint3(gridX, gridY, gridZ), viewRay);
    float3 viewDirection = -viewRay;
    uint seed = fogPcg((froxel.x + froxel.y * gridX + froxel.z * gridX * gridY)
            ^ pc.frameIndex * 2654435761u);

    float3 direct = float3(0.0);
    float3 sunRadiance = fogCelestialRadiance(sky, true);
    float3 moonRadiance = fogCelestialRadiance(sky, false);
    if (max(sunRadiance.r, max(sunRadiance.g, sunRadiance.b)) > 0.0
            && fogVisible(position, sky.sunDir, 10000.0)) {
        direct += sunRadiance * fogHg(dot(viewDirection, sky.sunDir), pc.settings.x);
    }
    if (max(moonRadiance.r, max(moonRadiance.g, moonRadiance.b)) > 0.0
            && fogVisible(position, sky.moonDir, 10000.0)) {
        direct += moonRadiance * fogHg(dot(viewDirection, sky.moonDir), pc.settings.x);
    }
    direct += fogLocalRadiance(push, position, viewDirection, seed);

    float3 indirect = fogIndirectRadiance(push, sky, position, seed);
    float2 previousUv;
    float previousZ;
    if (pc.settings.z > 0.0 && fogPreviousCoordinates(push, position, previousUv, previousZ)) {
        float3 history = fogHistorySample(giHistory, previousUv, previousZ, pc.grid.xyz)
                * FOG_RADIANCE_SCALE;
        // History rejection is deliberately conservative. GI is the soft channel; clamping only extreme
        // excursions preserves accumulation while stopping a newly revealed emissive block from trailing
        // a bright ribbon behind the camera.
        history = min(history, indirect * 4.0 + float3(8.0));
        indirect = lerp(indirect, history, clamp(pc.settings.z, 0.0, 0.98));
    }

    // Keep the shared fields in the fp16-safe representation. Integration multiplies by the same constant
    // exactly once, so the scene and the medium stay in absolute units at the final blend.
    directVolume[dispatchId.xy] = float4(min(max(direct, float3(0.0)) / FOG_RADIANCE_SCALE,
            float3(HALF_MAX)), 1.0);
    giVolume[dispatchId.xy] = float4(min(max(indirect, float3(0.0)) / FOG_RADIANCE_SCALE,
            float3(HALF_MAX)), 1.0);
}
