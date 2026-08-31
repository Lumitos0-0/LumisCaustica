// Shared definitions for all ray-tracing stages. The generateShaderRecords task generates typed Java
// records and serializers from Slang reflection.
module world_common;

// Read-only device pointer with explicit std430 layout. The layout parameter is LOAD-BEARING: a bare
// Ptr<T> gets Slang's "natural" layout (no vec4 16-byte alignment → e.g. WorldPush.sunDir lands at 200
// instead of the Java writer's 208, silently reading garbage) and additionally requires the
// scalarBlockLayout device feature. Keep std430 unless RtDeviceBringup explicitly enables that feature;
// this alias is the single layout switch and generated Java serialization stays in lock-step.
public typealias ConstPtr<T> = Ptr<T, Access.Read, AddressSpace.Device, Std430DataLayout>;
public typealias DevicePtr<T> = Ptr<T, Access.ReadWrite, AddressSpace.Device, Std430DataLayout>;

// Push constant block. The hot inline lanes avoid dereferencing WorldPush for values that are read in
// hit shaders or used for raygen control flow — every 64-bit device address lives here for exactly that
// reason, rather than behind the worldPushAddr indirection. tableAddr/entityTableAddr/materialTableAddr
// are read in world.rahit/world.rchit, which never load WorldPush at all; the light-buffer addresses are
// only read in world.rgen, which loads WorldPush anyway, but keeping every Addr in one place means there
// is exactly one struct to check when adding a new device buffer, instead of two.
public struct WorldPushConstants {
    public uint64_t worldPushAddr;
    public uint64_t tableAddr;
    public uint64_t entityTableAddr;
    public uint64_t materialTableAddr;
    public uint64_t lightBufAddr;        // RIS emitter-NEE global light buffer (0 = none published)
    public uint64_t lightAliasAddr;      // power-weighted O(1) alias table (0 = uniform fallback)
    public uint64_t lightLocalAliasAddr; // power aliases relative to each section's range
    public uint64_t lightGridCellAddr;   // dense light grid cell headers (0 = global proposals only)
    public uint64_t lightGridSpanAddr;   // packed weighted section spans referenced by cell headers
    public uint64_t pathQueueAddr;       // packed primary -> indirect continuation records
    public uint frameIndex;
    // ---- ReSTIR DI history. Read and write are separate buffers of one ping-pong pair so a lane can
    // never read a cell another lane is storing in the same dispatch. Either address being 0 disables
    // temporal reuse, which is also how a cold start after a resize or a reallocation is signalled.
    public uint64_t diHistoryReadAddr;
    public uint64_t diHistoryWriteAddr;
};

// Minimal push block for the compute passes that only need to dereference the frame's WorldPush: the sky
// LUT bakes read sun/moon/look state from the same BDA slot the trace does, so there is exactly one
// per-frame source of sky state and no way for the LUT and the frame it shades to disagree.
public struct PushAddr {
    public uint64_t worldPushAddr;
};

// Block-breaking overlay entry: xyz = breaking block position (rebased, same space as gl_WorldRay*),
// w = bindless entityAlbedoTex slot of the destroy-stage texture.
public struct BreakEntry {
    public int4 posSlot;
};

// Per-frame world data, read through a 64-bit device address. WorldPushData.java is generated from this
// reflected type, including its serializer, byte size, vector/matrix types, and fixed-array capacities.
public struct WorldPush {
    public float4x4 invViewProj;
    public float3   camOffset;
    public uint     frameIndex;
    public float4x4 prevViewProj;
    public float3   camDelta;
    public uint     spp;
    public float2   jitter;
    public uint     flags;            // bit0 submerged, bit4 waves
    public uint     maxBounces;
    // ---- Sky state. Only what the CPU alone can know: Minecraft's four eased celestial angles (the
    // 26.2 timeline drives them through a cubic-bezier ease and a datapack may replace the track, so
    // they are read from the attribute probe rather than re-derived from the tick), its star brightness,
    // the moon phase, and the look package's immutable constants. Every direction, colour, level and
    // transmittance is derived in sky.slang from these — no atmosphere maths runs on the CPU any more.
    public float4   celestial;        // x sun angle (rad), y moon angle, z star angle, w MC star brightness 0..1
    // x sun illuminance (lux, top of atmosphere), y full-moon illuminance (lux), z night airglow
    // (cd/m²), w star luminance anchor (cd/m²).
    public float4   skyLook0;
    // x noon south tilt (rad), y sun NEE half-angle (rad), z moon NEE half-angle (rad),
    // w moon-phase fixed light fraction.
    public float4   skyLook1;
    // x drawn sun half-angle (rad), y drawn moon half-angle (rad), z viewer altitude (km),
    // w moon phase index (0 full .. 4 new).
    public float4   skyLook2;
    public float4   skyLook3;         // x ground albedo, yzw reserved
    public float4   sunUv;            // vanilla sun sprite atlas rect (u0,v0,u1,v1)
    public float4   moonUv;           // vanilla current-moon-phase sprite atlas rect
    public float4   waterParams;      // xyz linear ACEScg camera-biome water tint, w wave time (s)
    public float4   waterAnchor;      // xy world-stable wave-domain anchor, z previous wave time
    public float4x4 curViewProj;      // forward camera-relative view-projection
    public uint     breakCount;
    public BreakEntry breaking[8];    // Java capacity and breakCount serialization are generated from this
    public float4   lightRebase;      // xyz hierarchy-to-current rebase, w inverse total light power
    public float4   lightGridOrigin;     // xyz minimum cell origin in rebased blocks, w cell size
    public int4     lightGridDims;       // xyz dense grid dimensions, w reserved
    public uint     lightCount;       // packed Light records in pc.lightBufAddr
    public uint     risCandidates;    // RIS candidate count M at the primary vertex (0 = emitter NEE off)
    public uint     risCandidatesSecondary; // M at every bounce vertex; -1 in the config keeps it equal to the above
    // Pre-exposure scalar applied to radiance at the outImage write
    // ONLY, so all light transport above stays in absolute scene units. Keeps stored fp16 values
    // near mid-grey at any scene brightness; the display pass divides it back out, so the two cancel
    // exactly and this cannot change the image. 1.0 disables it.
    public float    preExposure;
    // ---- ReSTIR DI temporal reuse of block-emitter direct light. Whether it happens at all is decided by
    // pc.diHistoryReadAddr/diHistoryWriteAddr, which the shader checks before consuming any RNG: with
    // reuse off, no random number is drawn here and no history is touched, so the frame is bit-identical to
    // one with ReSTIR DI absent. These lanes are the parameters of a pass that is already on.
    // x history divisor (render pixels per history cell), y temporal M cap, z how far a stored hit position
    // may drift and still count as the same surface, in blocks, w unused.
    public float4   diParams;
    // x minimum dot(normal, stored normal) for a temporal match, y spatial rounds, z spatial radius in
    // history cells, w unused. A round count of 0 leaves the spatial pass out entirely.
    public float4   diValidate;
    public uint     diGenTag;       // the published light generation; a mismatch means reject
    public uint     diHistoryWidth; // history grid extent in cells, after the divisor
    public uint     diHistoryHeight;
    // ---- ReSTIR GI world grid. Published by Java in the same rebased block space every position in this
    // shader lives in, so a cell index is one subtraction and a floor divide, and a cell means the same
    // piece of the world for every frame of one terrain rebuild. Step one of the GI work only publishes it:
    // nothing reads a cell yet, so these lanes cannot change a lit pixel.
    // xyz = the window's minimum corner in rebased blocks, already aligned down to the cell lattice;
    // w = cell edge in blocks. Both come from one Java decision, never from the config directly.
    public float4   giGrid;
    // xyz = cells per axis, and any zero component means the grid is off, which is the only off switch the
    // shader looks at. w is reserved.
    public int4     giDims;
};

// 32-byte hot area-light record. Linear ACEScg radiance uses packed R11G11B10; the
// grid-relative owner section uses three unsigned 10-bit coordinates; the geometric normal is
// reconstructed from the half axes.
//
// 32 divides the 64-byte cache line, so a record never straddles one. RIS fetches random records M times
// per shading vertex, making this alignment important. float3 forces 16-byte struct alignment and 32 is
// a multiple of it, so this lands with no tail padding.
//
// The rectangle spans 2U x 2V, so its area is derived as 4*|U x V|. lightGeometricNormal already
// needs that cross product, allowing the compiler to share the length with normalization.
public struct Light {
    public float3 pos;      // rebased world centre, f32 (the RIS target divides by distance to it)
    public uint   le;       // bitcast R11G11B10 radiance
    public uint   halfUxy;  // half2(U.x, U.y)
    public uint   halfUzVx; // half2(U.z, V.x)
    public uint   halfVyz;  // half2(V.y, V.z)
    public uint   section;  // 10:10:10 grid-relative section coord + bit 30 normal-flip
};

// Eight-byte Vose alias-table column. PDFs are reconstructed from the selected Light.
public struct LightAlias {
    public uint  aliasIndex;
    public float accept;
};

// Dense light grid cell entry. Empty coordinates have spanCount=0; populated section cells point into
// an immutable list of weighted section light ranges built and published alongside this grid.
public struct LightGridCell {
    public uint spanOffset;
    public uint spanCount;
    public float invWeightSum;
};

// One 16-byte Vose alias column for a cell's section distribution. The two light counts are uint16s.
public struct LightGridSpan {
    public uint firstLight;
    public uint aliasFirstLight;
    public uint packedLightCounts; // low16 self, high16 alias
    public float accept;
};

// One persisted ReSTIR DI reservoir, deliberately packed as twelve 4-byte scalars rather than vectors: a
// float3 member would push std430 alignment to 16 and round the whole record up, while this shape is
// exactly 48 bytes and every field is a plain scalar the generated Java record can write and read. The
// record is one read and one write per primary diffuse vertex, so unlike the light buffer it is fetched by
// coherent 2D tiles rather than at random, which is why straddling a cache line here is not worth the
// gymnastics a 32-byte form would need.
//
// The position and normal are the surface the reservoir was gathered ON, not part of the sample: they exist
// so the next frame can tell whether it is still the same surface. genTag records the published light
// generation, so a rebase or a republish invalidates every stored index instead of reusing a stale light.
// One direction bin of one cell of the ReSTIR GI radiance grid: a low-frequency answer to "what light is
// arriving here from that general direction", gathered from every surface that fell in this cell this frame
// and the frames before it. Like the DI record it is trusted only after its tag and its validation data
// agree with the surface asking, and it is never cleared: a fresh allocation holds whatever its previous
// owner left, and a record that survives the checks is usable regardless of where it came from.
//
// The radiance is stored PRE-exposed, in halves, because the grid is written before the frame's exposure
// scalar is final and read after: the display pass divides the same latched value back out. Storing it
// post-exposure would bake a number that changes from frame to frame into a cache that outlives the frame.
public struct GiRadianceRecord {
    // Written as three scalars rather than a float3, the way DiReservoirRecord does it: a vec3's alignment
    // decides whether the members after it pack into its tail or start on the next 16-byte boundary, which is
    // a 16-byte difference in a stride both Java's allocation and the debug pass's word map depend on.
    public float  surfaceX;         // rebased position the radiance was gathered at — the entry describes
    public float  surfaceY;         // one surface in this cell rather than the middle of it, which is the
    public float  surfaceZ;         // difference between a radiance cache and a smear
    public uint   normalOct;        // packHalf2(octEncode(surface normal)), for the same reason
    public uint   packedRG;         // packHalf2(r, g) of the pre-exposed radiance
    public uint   packedBHT;        // packHalf2(b, hit distance / cell size): the cache's own occlusion test
    public float  weight;           // finalized W = wSum / (M * phat)
    public float  m;                // accumulated effective sample count, capped by the temporal ceiling
    public uint   genTag;           // 0 = never written; any other mismatch = reject
    public uint   pad0;             // 36 used of 48 bytes: three words of room before this record has to be
    public uint   pad1;             // resized, which is a reflected-stride change in Java and a buffer-size
    public uint   pad2;             // change in the allocation policy, not an edit in one file
};

public struct DiReservoirRecord {
    public float  hitX, hitY, hitZ; // rebased shading position at store time
    public uint   normalOct;      // packHalf2(octEncode(surface normal))
    public uint   lightIndex;     // index into the published Light buffer of that generation
    public uint   packedST;       // packHalf2(emitter rectangle parameters)
    public float  weight;         // finalized W = wSum / (M * phat)
    public float  m;              // accumulated effective sample count
    public float  phat;           // target function at that surface, for diagnostics only
    public uint   genTag;         // 0 = never written; any other mismatch = reject
    public float  spatial;        // neighbours the spatial round merged this frame: diagnostics, not reuse
    // The radiance-grid cell this surface falls in, plus one (0 = the grid is off or the surface is outside
    // its window). Nothing that decides a lighting value reads it: it is here because this record is the only
    // per-surface state the trace already validates and the debug pass already knows how to read, which is
    // what makes the grid's addressing observable before any cell has content. See gi_grid.slang.
    public uint   giCell;
};

// TerrainPrim.flags bit 0: this emissive quad is in the light buffer (RtLightCollector membership).
public static const uint TERRAIN_PRIM_IN_LIGHT_BUFFER = 1u;

// A reservoir with no selected emitter yet. Kept out of the representable index range so reuse can tell
// "nothing was sampled" from "light zero was sampled" without inspecting any other field.
public static const uint DI_NO_LIGHT = 0xffffffffu;

// Largest finite half. Producers of fp16 HDR storage clamp to this instead of letting the conversion
// round up to +inf and propagate as NaN — world.rgen's rgba16f outImage store most of all, which is the
// one place the whole path integral's absolute radiance meets a half.
public static const float HALF_MAX = 65504.0;

// Radiance-ray payload (location 0). Member order and types are a cross-stage ABI.
//
// Three-component quantities are packed into half pairs rather than declared as float3: they cost 6 bytes
// instead of 12, and payload storage is reserved per trace call site, so every byte here is paid twice per
// radiance trace (once for TraceRay, once for Invoke) and is preserved across the SER reorder. hitT stays
// f32 — it reaches 10000 blocks and the hit position is reconstructed from it, where half's ~4-block
// spacing at that magnitude would be visible.
public struct Payload {
    // surface0..2 are a hand-packed UNION of the two things a radiance ray can come back with. Reach them
    // only through unpackAlbedo/packAlbedo, unpackNormal/packNormal, unpackSky/packSky — the raw words
    // have no meaning without knowing which case you are in.
    //
    //   hit  — half3 albedo + half3 normal
    //   miss — float3 sky radiance, full fp32
    //
    // The miss case is the entire reason these are hand-packed. Sky is in absolute photometric units and
    // the sun disc alone is ~3.6e5 cd/m², well past half's 65504 ceiling — a
    // half3 could not carry it without first normalising by the frame's exposure, which made a storage
    // format's limitation into a dependency on the exposure controller. A hit has no sky and a miss has
    // neither albedo nor normal, so reinterpreting the same twelve bytes buys the range for free and keeps
    // the payload exactly the size it was.
    //
    // The albedo lane doubles as the shadow ray's accumulated transmittance (see makeShadowPayload).
    public uint   surface0;    // hit: half2(albedo.r, albedo.g) | miss: asuint(sky.r)
    public uint   surface1;    // hit: half2(albedo.b, normal.x) | miss: asuint(sky.g)
    public uint   surface2;    // hit: half2(normal.y, normal.z) | miss: asuint(sky.b)
    public float  hitT;        // >= 0 on hit, < 0 on miss.
    public half3  motionPrev;  // per-vertex world displacement since last frame.
    public half3  f0;          // specular F0.
    public uint   flags;       // bits 0..1 material, bit 2 celestial, bit 3 water-entering, bits 4..6 emission source.
    public uint   roughMetal;  // packHalf2x16(roughness, metalness)
    public uint   emissionSss; // packHalf2x16(emission, sss)
    public uint   iorTransmission; // packHalf2x16(IOR, transmission factor)
    public uint   rayCone;     // packHalf2x16(width, spread)
};

public static const uint PAYLOAD_SHOW_CELESTIAL = 4u;
// Set by world.rchit on a terrain hit whose prim carries TERRAIN_PRIM_IN_LIGHT_BUFFER: this emitter is
// RIS-sampled, so raygen gates its direct-hit emission on diffuse continuation rays (no double count).
public static const uint PAYLOAD_EMITTER_IN_LIST = 128u;
// Set by world.rchit on any dielectric hit (water or glass/ice): true when the incoming ray travels
// against the prim's outward face normal (entering the volume), false when it exits. Derived from face
// orientation rather than toggled, so a stray or missing face cannot corrupt the medium for the rest of
// the path — each crossing re-derives which side it is on. Bit 2 (PAYLOAD_SHOW_CELESTIAL) is only
// meaningful on a miss and this bit only on a hit, so they never need to be read together, but they get
// distinct bits anyway to keep the two concerns unambiguous.
public static const uint PAYLOAD_DIELECTRIC_ENTERING = 8u;
public static const uint PAYLOAD_EMISSION_SOURCE_SHIFT = 4u;
public static const uint PAYLOAD_EMISSION_SOURCE_MASK = 7u;
public static const uint EMISSION_SOURCE_NONE = 0u;
public static const uint EMISSION_SOURCE_UNIFORM = 1u;
public static const uint EMISSION_SOURCE_HEURISTIC = 2u;
public static const uint EMISSION_SOURCE_LAB_PBR = 3u;

public static const uint PAYLOAD_MATERIAL_MASK = 3u;
public static const uint MATERIAL_OPAQUE = 0u;
// Both dielectrics are volumes: they refract and push a participating medium. They stay distinct ids
// because water owns behaviour nothing else has — an animated wave surface, the caustic term, biome-tint
// absorption calibrated per block of depth, and a chit path fed by the fluid mesher rather than the
// (inset) translucent terrain layer. MATERIAL_DIELECTRIC is everything else transparent: glass, ice, and
// any solid transparent block a pack defines.
public static const uint MATERIAL_WATER = 1u;
public static const uint MATERIAL_PARTICLE = 2u;
public static const uint MATERIAL_DIELECTRIC = 3u;

// Entity per-triangle record (48 B). The final lane mirrors TerrainPrim's integer material metadata.
public struct Prim {
    public float4 normal; // xyz = geometric normal, w = per-primitive emission strength
    public float4 tint;
    public uint materialId;
    public uint flags;
    public uint aux0;
    public uint aux1;
};

// Terrain per-triangle record (48 B). The final 16-byte lane contains integer geometry-local metadata;
// shared surface parameters live once in MaterialHeader instead of being copied into every triangle.
public struct TerrainPrim {
    public float4 normal;
    public float4 tint;
    public uint materialId;
    public uint flags;
    public uint aux0;
    public uint aux1;
};

// Compiled material record (80 B). Source-format data has already been decoded into canonical pages;
// texturePage indexes the dedicated canonical page arrays. materialUv maps sprite-local UV into the packed
// page, while albedoUv maps the vanilla atlas rectangle back to sprite-local UV.
public struct MaterialHeader {
    public uint model;
    public uint features;        // low feature bits; bits 24..31 = maximum valid material-page LOD
    public uint texturePage;
    public uint extensionOffset; // zero when no optional extension record is present
    public float4 materialUv; // packed-page uvMin.xy + uvExtent.xy
    public float4 albedoUv;   // vanilla atlas uvMin.xy + inverse uvExtent.xy
    public float4 params;  // roughness, metalness, IOR, transmission
    public float4 average; // whole-sprite linear-input average RGBA for translucent shadow filtering
};

public static const uint MATERIAL_FEATURE_SPEC = 1u;
public static const uint MATERIAL_FEATURE_NORMAL = 2u;
public static const uint MATERIAL_FEATURE_HEURISTIC_EMISSION = 4u;
public static const uint MATERIAL_FEATURE_STOCHASTIC_ALPHA = 16u;
// Final HDR emitting-surface luminance (look-package baseline or absolute JSON cd/m² override), baked in Java
// at material-compile time (RtMaterialRegistry) and packed here as a 16-bit fraction of the max — every
// emissive material carries a value (0 for non-emissive), not just resource-pack-overridden ones, so the
// full features word's spare 16 bits (8..23) hold the encoded value.
public static const uint MATERIAL_EMISSION_STRENGTH_SHIFT = 8u;
public static const uint MATERIAL_EMISSION_STRENGTH_MASK = 65535u;
// Mirrors RtMaterialRegistry.MAX_EMISSION_STRENGTH — HALF_MAX, the ceiling emission can survive through
// Payload.emissionSss (half2) and Light.le (R11G11B10). Emission is measured in cd/m².
public static const float MATERIAL_MAX_EMISSION_STRENGTH = 65504.0;
public static const uint MATERIAL_MAX_LOD_SHIFT = 24u;
public static const uint MATERIAL_MAX_LOD_MASK = 255u;

// Per-section geometry record: packed buffer addresses + per-bucket triangle offsets
// (gl_PrimitiveID restarts at 0 per geometry, so hit shaders re-add triBase[geometryIndex]).
public struct Section {
    public uint64_t primAddr;
    public uint64_t uvAddr;
    public uint triBase[4]; // fixed bucket order: solid/cutout/translucent/water
};

// Entity geometry record: packed buffer addresses, motion, and per-geometry triangle bases. Entity BLAS
// geometry order is opaque/any-hit; PrimitiveIndex restarts at zero for each geometry.
public struct EntityGeom {
    public uint64_t primAddr;
    public uint64_t idxAddr;
    public uint64_t uvAddr;
    public uint64_t dispAddr;
    public float4 rigidDisp;
    public uint triBase[2];
    public uint2 reserved;
};

// TLAS instance-custom-index encoding + terrain geometry buckets (shared by rahit/rchit).
public static const uint ENTITY_BIT = 0x800000u;
public static const uint PARTICLE_BIT = 0x400000u; // particles share the entity cutout path
public static const uint IDX_MASK = 0x3FFFFFu;     // low 22 bits = geom-table index
public static const uint BUCKET_CUTOUT = 1u;
public static const uint BUCKET_TRANSLUCENT = 2u;
public static const uint BUCKET_WATER = 3u;

// packHalf2x16 equivalent (Slang core module has no GLSL pack builtin; f32tof16 returns the half
// bit-pattern in the low 16 bits).
public uint packHalf2(float2 v) {
    return (f32tof16(v.y) << 16) | (f32tof16(v.x) & 0xFFFFu);
}

public float2 unpackHalf2(uint p) {
    return float2(f16tof32(p & 0xFFFFu), f16tof32(p >> 16));
}

// ---- Payload.surface0..2 accessors. Two views of the same twelve bytes; see the struct for why.
//
// The hit view straddles surface1 (albedo.b in its low half, normal.x in its high half), so the two
// setters are read-modify-write on that word. That is a couple of ALU ops on a value already in
// registers, and it buys a payload that did not have to grow. Each setter touches only its own halves,
// so writing one before the other is initialised is safe — nothing here does arithmetic on the half it
// is preserving.
public float3 unpackAlbedo(Payload p) {
    return float3(unpackHalf2(p.surface0), unpackHalf2(p.surface1).x);
}

public void packAlbedo(inout Payload p, float3 albedo) {
    p.surface0 = packHalf2(albedo.xy);
    p.surface1 = (p.surface1 & 0xFFFF0000u) | (f32tof16(albedo.z) & 0xFFFFu);
}

public float3 unpackNormal(Payload p) {
    return float3(unpackHalf2(p.surface1).y, unpackHalf2(p.surface2));
}

public void packNormal(inout Payload p, float3 normal) {
    p.surface1 = (p.surface1 & 0x0000FFFFu) | (f32tof16(normal.x) << 16);
    p.surface2 = packHalf2(normal.yz);
}

// Miss view: full fp32, so absolute photometric sky needs no pre-exposure round trip through the payload
// and no clamp to keep it finite. The fp16 ceiling still applies at the outImage store in world.rgen,
// where the finite-half clamp guards every radiance source.
public float3 unpackSky(Payload p) {
    return float3(asfloat(p.surface0), asfloat(p.surface1), asfloat(p.surface2));
}

public void packSky(inout Payload p, float3 sky) {
    p.surface0 = asuint(sky.x);
    p.surface1 = asuint(sky.y);
    p.surface2 = asuint(sky.z);
}

// The ray tracer's transport contract is scene-linear ACEScg (AP1/D60). Minecraft assets and
// captured vertex colours are authored as sRGB/BT.709 (D65), so colour inputs cross this OCIO-derived
// chromatic-adaptation + primary-conversion seam exactly once before they enter any BRDF, throughput,
// medium, light, guide, or radiance calculation.
public static const float3 ACESCG_LUMA = float3(0.27222872, 0.67408177, 0.05368952);
public static const float3 BT709_TO_ACESCG_R = float3(0.61309743, 0.33952314, 0.04737945);
public static const float3 BT709_TO_ACESCG_G = float3(0.07019372, 0.91635388, 0.01345240);
public static const float3 BT709_TO_ACESCG_B = float3(0.02061559, 0.10956977, 0.86981463);

public float3 bt709ToAcesCg(float3 color) {
    return float3(
            dot(color, BT709_TO_ACESCG_R),
            dot(color, BT709_TO_ACESCG_G),
            dot(color, BT709_TO_ACESCG_B));
}

public float3 srgbToLinear(float3 color) {
    float3 lo = color / 12.92;
    float3 hi = pow((color + 0.055) / 1.055, float3(2.4));
    return lerp(lo, hi, step(float3(0.04045), color));
}
