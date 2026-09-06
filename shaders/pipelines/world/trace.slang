// Ray dispatch: SBT/cull constants, payload construction, ordinary/reordered radiance, guide probes,
// and shadow visibility. Depends on core.

// receiver billboards only. The first-person camera owner uses 0x01 (secondary only).

import world_common;
import world_core;
import bindings;

public static const uint CULL_SECONDARY = 0x01u;
public static const uint CULL_PRIMARY = 0x02u;
// Fog-march rays (fog.slang). Instance mask bits used only by the fog march:
//  * CULL_FOG_ENTITY (0x04): entity-only occluder ray used when no exact probe fires. Terrain must not
//    carry it (terrain transmittance already comes from the per-frame fog grid cache and would be applied
//    twice), while ordinary entities (0xFF) and the first-person self (0x05) do.
//  * CULL_FOG_EXACT (0x08): the exact-colour probe — fired once per march at a tinted, still-lit column.
//    Carried by terrain (0xFB) and ordinary/block entities (0xFF) so the ray measures per-channel tint
//    AND entity occlusion; NOT carried by the first-person self (a camera-adjacent body hit would blank
//    the probe) or particles (0x02).
public static const uint CULL_FOG_ENTITY = 0x04u;
public static const uint CULL_FOG_EXACT = 0x08u;
public static const uint TERRAIN_BUCKETS = 4u;
public static const uint SBT_RADIANCE = 0u;
public static const uint SBT_SHADOW = TERRAIN_BUCKETS;
public static const uint SBT_STRIDE_BUCKET = 1u;
public static const uint MISS_RADIANCE = 0u;
public static const uint MISS_GUIDE = 1u;

public RayDesc makeRay(float3 origin, float tmin, float3 dir, float tmax) {
    RayDesc r;
    r.Origin = origin;
    r.TMin = tmin;
    r.Direction = dir;
    r.TMax = tmax;
    return r;
}

public float3 offsetSurfaceOrigin(float3 position, float3 surfaceNormal,
                           float3 outgoingDirection, float bias) {
    float3 sideNormal = dot(surfaceNormal, outgoingDirection) >= 0.0
            ? surfaceNormal : -surfaceNormal;
    return position + sideNormal * bias;
}

// A radiance payload carrying only the two words the invoked hit/miss shader READS: `flags`
// (show-celestial gate, consumed by world.rmiss) and the packed ray cone (consumed by world.rchit for
// texture LOD). Every other member is an OUTPUT that world.rchit/world.rmiss write, so it carries no
// information into a trace and is initialized to a neutral value here.
public Payload makeRadiancePayload(uint flags, uint rayCone) {
    Payload p;
    // Zeroes the packed albedo/normal words directly: both the hit and the miss view read zero out of
    // them (half 0 and float 0 are the same bit pattern), so this is neutral for either case.
    p.surface0 = 0u;
    p.surface1 = 0u;
    p.surface2 = 0u;
    p.hitT = -1.0; // miss sentinel: world.rmiss writes only the surface words, so a miss leaves this negative
    p.motionPrev = half3(0.0h, 0.0h, 0.0h);
    p.f0 = half3(0.0h, 0.0h, 0.0h);
    p.flags = flags;
    p.roughMetal = 0u;
    p.emissionSss = 0u;
    p.iorTransmission = 0u;
    p.rayCone = rayCone;
    return p;
}

// Ordinary radiance trace for latency-sensitive, coherent work such as Pass A. It invokes hit/miss
// shaders directly and therefore requires no invocation-reorder capability.
public void traceRadiance(uint cullMask, float3 ro, float tmin, float3 rd, float tmax,
                    bool showCelestial, float rayConeWidth, float rayConeSpread) {
    payload = makeRadiancePayload(showCelestial ? PAYLOAD_SHOW_CELESTIAL : 0u,
            packHalf2(float2(rayConeWidth, rayConeSpread)));
    TraceRay(topLevelAS, RAY_FLAG_NONE, cullMask,
            SBT_RADIANCE, SBT_STRIDE_BUCKET, MISS_RADIANCE,
            makeRay(ro, tmin, rd, tmax), payload);
}

// Auxiliary RR probes deliberately use ordinary TraceRay. They are one-off, incoherent rays whose
// result is consumed immediately, so paying a reorder barrier and carrying guide state across it only
// adds latency. The dedicated miss record also avoids the atmosphere march: guides need hit-vs-miss,
// not sky radiance.
public void traceGuide(uint cullMask, float3 ro, float tmin, float3 rd, float tmax,
                float rayConeWidth, float rayConeSpread) {
    payload = makeRadiancePayload(0u, packHalf2(float2(rayConeWidth, rayConeSpread)));
    TraceRay(topLevelAS, RAY_FLAG_NONE, cullMask, SBT_RADIANCE, SBT_STRIDE_BUCKET,
            MISS_GUIDE, makeRay(ro, tmin, rd, tmax), payload);
}

public Payload makeShadowPayload() {
    Payload shadowPayload;
    shadowPayload.surface0 = 0u;
    shadowPayload.surface1 = 0u;
    shadowPayload.surface2 = 0u;
    // The albedo view is this ray's accumulated transmittance; the any-hits multiply into it.
    packAlbedo(shadowPayload, float3(1.0, 1.0, 1.0));
    shadowPayload.hitT = -1.0; // water-crossing sentinel, filled by the water any-hit
    shadowPayload.motionPrev = half3(0.0h, 0.0h, 0.0h);
    shadowPayload.f0 = half3(0.0h, 0.0h, 0.0h);
    // world_guide.rmiss clears this sentinel. An accepted opaque hit skips closest-hit and therefore
    // leaves it set, while ignored translucent/water hits continue until either a later opaque hit or miss.
    shadowPayload.flags = 1u;
    shadowPayload.roughMetal = 0u;
    shadowPayload.emissionSss = 0u;
    shadowPayload.iorTransmission = 0u;
    shadowPayload.rayCone = 0u;
    return shadowPayload;
}

public VisibilityResult visibility(float3 origin, float3 dir, float tmax) {
    return visibilityMasked(origin, dir, tmax, CULL_SECONDARY);
}

/**
 * Entity-only fog-occlusion ray. The fog march already has full terrain transmittance from the per-frame
 * voxel cache, so terrain must not be re-tested here (it would be applied twice, darkening shafts). The
 * ray's cull mask is CULL_FOG_ENTITY, carried only by entity instances — including translucent/water
 * entity materials, which the shadow any-hits tint and pass through exactly like the old full ray.
 */
public VisibilityResult visibilityEntities(float3 origin, float3 dir, float tmax) {
    return visibilityMasked(origin, dir, tmax, CULL_FOG_ENTITY);
}

/**
 * Fog-march exact-colour probe. Fired once per march at a tinted cutout/translucent column (bake flag
 * FOG_PACK_EXACT_COLOR) that the cache still shows as lit: the real any-hit path restores per-channel
 * glass tint + texture-alpha dapple and measures entity occlusion in the same ray. Cull mask
 * CULL_FOG_EXACT is carried by terrain and ordinary/block entities but NOT by the first-person self
 * (body hits would blank the probe) or particles.
 */
public VisibilityResult visibilityFogExact(float3 origin, float3 dir, float tmax) {
    return visibilityMasked(origin, dir, tmax, CULL_FOG_EXACT);
}

VisibilityResult visibilityMasked(float3 origin, float3 dir, float tmax, uint cullMask) {
    // Vulkan requires an identical payload structure for every stage reachable by this trace. The shadow
    // path uses only the albedo view as accumulated transmittance and hitT as the nearest-water crossing.
    Payload shadowPayload = makeShadowPayload();
    // Shadow SBT records run any-hit only for cutout/translucent/water. Cutout alpha-tests; translucent
    // and water tint that transmittance and pass through. Solid blocks terminate traversal. There is no
    // closest shader worth executing. The lightweight guide miss clears the flags sentinel so this path
    // needs only ordinary TraceRay and no invocation-reorder capability.
    TraceRay(topLevelAS,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
            cullMask, SBT_SHADOW, SBT_STRIDE_BUCKET, MISS_GUIDE,
            makeRay(origin, RAY_TMIN, dir, tmax), shadowPayload);
    VisibilityResult result;
    result.transmittance = shadowPayload.flags == 0u
            ? unpackAlbedo(shadowPayload) : float3(0.0, 0.0, 0.0);
    result.waterHitT = shadowPayload.hitT;
    return result;
}

// `valid` is false when the point projects behind the previous camera, where the perspective divide
// would otherwise manufacture an arbitrarily large NDC. Mirror images reach that case easily: a virtual
// image sits as far behind the reflector as the source is in front, so a reflection seen at a grazing
