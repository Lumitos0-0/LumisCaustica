// Volumetric fog shared math and froxel mapping.
// This file centralizes the froxel <-> view/world mapping so every pass uses the SAME
// deterministic identity, satisfying Invariant: a froxel is identified by its stable center.
//
// COORDINATE-SPACE GLOSSARY (used throughout the volumetric pipeline):
//
//   world         : absolute Minecraft double-precision world space (CPU only, never in shaders).
//   rebased-world : camera-relative world in which the current terrain origin (terrain.blockX/Y/Z)
//                   is subtracted. camOffset = cam - terrain.block* is the camera position in
//                   this space. All GPU world positions are in rebased-world so fp32 precision
//                   stays sufficient far from origin. Previous-frame camera offset is given
//                   relative to the CURRENT rebase origin (prevCamOffset = camOffset - camDelta),
//                   which means positions can be reprojected across origin rebases without drift.
//   view          : this pipeline works with camera-relative rebased-world vectors (camera at
//                   origin, axes NOT rotated into the camera frame): everything UnprojectNdc
//                   returns is of that form. There is NO rotated view space anywhere below.
//   view depth    : linear depth measured ALONG THE CAMERA'S FORWARD AXIS, positive in front
//                   (blocks). For a camera-relative rebased-world point p or unit ray dir,
//                     viewDepth = dot(p, camForward)  resp.  viewDepth = t * dot(dir, camForward)
//                   with camForward = CameraForwardWorld(invViewProj). Taking the WORLD -z
//                   component instead (as if world axes doubled as camera axes) is only correct
//                   while the camera happens to face world -Z; for other yaw/pitch the error
//                   grows slowly and flips sign across the vertical dir.z = 0 plane, which is
//                   visible as a view-dependent vertical screen split.
//   clip          : post-projection homogeneous coordinates (curViewProj * camera-relative
//                   rebased-world point). clip.w equals the point's view depth (depth along the
//                   camera forward axis, positive in front): the projection w-row is
//                   (0,0,-1,0) in camera space and frameViewRotation is rotation-only, for both
//                   curVP and prevVP. WorldToFroxelUvw's w mapping relies on exactly this.
//   NDC           : clip.xyz / clip.w, x,y in [-1,1] with Y up; z is reversed-Z (1 at near, 0 at far).
//   screen UV     : (u,v) with u in [0,1] left-to-right, v in [0,1] TOP-to-BOTTOM (matches image
//                   Load/sample conventions). uv = (ndc*0.5 + 0.5, (1-ndc.y)*0.5).
//   froxel        : (u,v,w) with u,v,w in [0,1] mapping to the frustum volume. w=0 is the near
//                   plane, w=1 is the far plane; the depth distribution is exponential so slices
//                   are denser near the camera. A froxel's canonical stable CENTER in [0,1]^3 is
//                   ((ix+0.5)/W, (iy+0.5)/H, (iz+0.5)/D). Physical world position is derived from
//                   that canonical center; stochastic sampling is applied ONLY to the light ray,
//                   never to the froxel identity.
//
// The camera matrices plus per-frame medium parameters live in a small storage buffer accessed
// via device address so that push constants stay under the 128-byte minimum-max-guaranteed
// Vulkan limit.
//
// NOTE: this file must NOT import bindings.slang. The build system reflects every
// pipeline's bindings by compiling its "source" shader in isolation; including
// bindings here would cause every other pipeline's `import common;` (or implicit
// -I resolution) to pick up a bindings.slang from the wrong pipeline directory,
// resulting in "undefined identifier" errors for resources like `tlas`/`depthImage`.
// Functions that need descriptor-declared resources (tlas, depthImage, etc.) live
// in the .comp that imports bindings, or take the resource as a parameter.

// -----------------------------------------------------------------------------
// Matrix buffer (BDA, std430)
// -----------------------------------------------------------------------------
public struct FroxelMatrices
{
    // Clip-space -> camera-relative rebased world (inverse of projection * viewRotation).
    public float4x4 invViewProj;
    // Camera-relative rebased world -> current clip space (projection * viewRotation, jitter-free).
    public float4x4 curViewProj;
    // Camera-relative rebased world (in CURRENT rebase frame) -> previous-frame clip space
    // (jitter-free, for reprojection). The previous camera position relative to the current
    // rebase origin is VolumetricPush.prevCamOffset.
    public float4x4 prevViewProj;

    // The following float4s are frame/medium state carried in the same BDA buffer as the matrices
    // so VolumetricPush remains inside Vulkan's 128-byte guaranteed push-constant limit. Units:
    //   density        ~= 1/block
    //   sigma_s/a/t    = 1/block
    //   radiance       = renderer scene-linear radiance units
    //   distance       = blocks
    // airParams: x = air single-scattering albedo scalar, y = ambient sky strength,
    //            z = fixed sky-probe ray length in blocks, w = water absorption weight for surfaceT
    //                (0 when the primary path already applied water absorption, 1 otherwise).
    public float4 airParams;
    // waterParams: x = water density scale, y = water scattering scale,
    //              z = water absorption scale, w = water HG anisotropy.
    public float4 waterParams;
    // waterTintFlags: rgb = camera-biome water tint in the renderer's linear transport basis,
    //                 a = 1 when the camera is inside water and the froxel volume should use water.
    public float4 waterTintFlags;
    // weatherParams: x = rain level, y = thunder level, z = density multiplier, w = direct-light multiplier.
    public float4 weatherParams;
    // ambientRadiance: rgb = stable sky/airglow ambient radiance scale,
    //                  a = multiplier for TLAS-tested sky visibility probes.
    public float4 ambientRadiance;
    // worldOrigin: current terrain rebase origin in absolute block coordinates, used only to anchor
    // procedural density noise in world space while all ray/query positions remain rebased-world.
    public float4 worldOrigin;
};

// -----------------------------------------------------------------------------
// Push constants (must stay <= 128 bytes)
// -----------------------------------------------------------------------------
//
// debugView selects the visualization mode (all writes go through integrate.comp):
//    0 = final composite (volumeRadiance + volumeTransmittance * scene)
//    1 = DEBUG_VOLUME_DENSITY
//    2 = DEBUG_VOLUME_SCATTERING
//    3 = DEBUG_VOLUME_ABSORPTION
//    4 = DEBUG_VOLUME_TRANSMITTANCE
//    5 = DEBUG_VOLUME_LIGHT_VISIBILITY
//    6 = DEBUG_VOLUME_HISTORY (temporally reconstructed source radiance)
//    7 = DEBUG_VOLUME_HISTORY_WEIGHT (history confidence)
//    8 = DEBUG_VOLUME_FROXEL_SLICE
//    9 = DEBUG_VOLUME_REPROJECTION_ERROR
//   10 = world-position checker (verifies density is world-anchored; should not swim)
//   11 = integrated in-scattered radiance over black
//   12 = final volumetric contribution (L + (T-1)*scene) over neutral gray
public struct VolumetricPush
{
    public uint64_t matricesAddr;                              // offset  0, size  8
    // vec3 fields are 16-byte aligned in std430; each is followed by a float to pack tightly.
    public float3   camOffset;        // 16, camera position in CURRENT rebased world
    public float    nearPlane;        // 28, froxel near plane (blocks)
    public float3   prevCamOffset;    // 32, prev camera position in CURRENT rebased world
    public float    farPlane;         // 44, froxel far plane (blocks)
    public float3   sunDir;           // 48, normalized, toward dominant celestial light (field name kept ABI-stable)
    public float    sunAngularRadius; // 60, light-disc half-angle in radians
    public float3   sunColor;         // 64, phase-independent direct radiance (preExposure*E/pi)
    public float    heightFogDensity; // 76, sigma_t at Y=heightFogBase (1/blocks)
    public float    heightFogFalloff; // 80
    public float    heightFogBase;    // 84, Y at which density = heightFogDensity, shifted into the
                                      // current rebase frame (absolute world base - rebase origin Y,
                                      // matching the rebased worldPos fed to ExtinctionAt)
    public float    globalDensity;    // 88
    public float    anisotropyG;      // 92, Henyey-Greenstein g
    public float    historyBaseWeight;// 96, baseline history lerp weight (e.g. 0.92)
    public float    historyValid;     // 100, 0 on grid/reset, 1 otherwise
    public uint     froxelSizeX;      // 104
    public uint     froxelSizeY;      // 108
    public uint     froxelSizeZ;      // 112
    public uint     frameIndex;       // 116
    public uint     debugView;        // 120
    public uint     stochasticLight;  // 124
    // sizeof(VolumetricPush) = 128.
};

public static const float kEpsilon = 1.0e-6f;
public static const float kPi = 3.14159265f;

// Read-only device pointer matching world_common.slang convention.
public typealias ConstPtr<T> = Ptr<T, Access.Read, AddressSpace.Device, Std430DataLayout>;

// Froxel grid dimensions as a uint3.
public uint3 FroxelSize(VolumetricPush pc)
{
    return uint3(pc.froxelSizeX, pc.froxelSizeY, pc.froxelSizeZ);
}

public FroxelMatrices GetMatrices(VolumetricPush pc)
{
    return ConstPtr<FroxelMatrices>(pc.matricesAddr)[0];
}

// -----------------------------------------------------------------------------
// Low-discrepancy hash / R2-like sampling (deterministic for (froxel, frame)).
// Uses the golden-ratio additive recurrence for temporal blue-noise-like
// distribution, decorrelated per-froxel via a Wang hash.
// -----------------------------------------------------------------------------
public float Hash1(uint x)
{
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x ^= x >> 4;
    x *= 0x27d4eb2du;
    x ^= x >> 15;
    return float(x & 0x00ffffffu) / float(0x01000000u);
}

// Interleaved gradient noise / blue-noise-like 2D hash (deterministic for (froxel, frame)).
// Decorrelated between neighboring froxels in all three axes and across frames.
public float BlueNoise(uint3 froxel, uint frame)
{
    uint h = froxel.x * 19349663u
           + froxel.y * 83492791u
           + froxel.z * 2654435761u
           + frame    * 9742139u;
    return Hash1(h);
}

public float2 BlueNoise2(uint3 froxel, uint frame)
{
    return float2(
        BlueNoise(froxel, frame),
        BlueNoise(froxel + uint3(17u, 41u, 97u), frame ^ 0x12345678u)
    );
}

// R2 (golden ratio) additive recurrence for the per-frame temporal offset, producing a
// low-discrepancy 1D sequence that distributes samples evenly across frames. This is
// preferred over pure hashing for shadow-accumulation convergence.
public float R2Offset(uint3 froxel, uint frame)
{
    const float phi = 1.61803398875f;
    float seed = BlueNoise(froxel, 0u);   // per-froxel random phase
    return fract(seed + float(frame) / phi);
}

public float2 R2Offset2D(uint3 froxel, uint frame)
{
    // 2D golden ratio set (Cranley-Patterson rotation of (0,0)+n*(1/phi2, 1/phi3)).
    const float g2 = 1.32471795724f;  // plastic constant, good 2D lattice
    float a1 = 1.0f / g2;
    float a2 = 1.0f / (g2 * g2);
    float2 seed = BlueNoise2(froxel, 0u);
    return float2(
        fract(seed.x + float(frame) * a1),
        fract(seed.y + float(frame) * a2)
    );
}

// -----------------------------------------------------------------------------
// Froxel <-> depth mapping (exponential).
//   depth(t) = near * (far/near)^t for t in [0,1].
// All "view depth" values are linear POSITIVE distances along the camera -Z axis,
// i.e. -view.z (blocks).
// -----------------------------------------------------------------------------
public float LogRatio(float nearPlane, float farPlane)
{
    return log(max(farPlane / max(nearPlane, kEpsilon), 1.0001f));
}

// Authoritative exponential froxel Z distribution. Every pass must call this (or a wrapper
// that calls this) rather than reconstructing the distribution independently. `w` is in [0,1],
// with w=0 at the near edge and w=1 at the far edge.
public float froxelSliceDistance(float w, float nearPlane, float farPlane)
{
    w = clamp(w, 0.0f, 1.0f);
    return nearPlane * exp(w * LogRatio(nearPlane, farPlane));
}

public float ViewDepthFromFroxelCoord(float t, float nearPlane, float farPlane)
{
    return froxelSliceDistance(t, nearPlane, farPlane);
}

// Canonical stable center depth for slice z (integer) -- center at t=(z+0.5)/D.
public float FroxelCenterViewDepth(VolumetricPush pc, uint zSlice)
{
    float t = (float(zSlice) + 0.5f) / float(FroxelSize(pc).z);
    return ViewDepthFromFroxelCoord(t, pc.nearPlane, pc.farPlane);
}

// Continuous w-coordinate in [0,1] for a given linear view depth.
// Authoritative inverse of froxelSliceDistance(). Returns a clamped coordinate in [0,1].
public float froxelSliceCoord(float viewDepth, float nearPlane, float farPlane)
{
    float ratio = max(viewDepth, nearPlane) / max(nearPlane, kEpsilon);
    float w = log(max(ratio, 1.0001f)) / LogRatio(nearPlane, farPlane);
    return clamp(w, 0.0f, 1.0f);
}

public float SliceCoordFromViewDepth(float viewDepth, float nearPlane, float farPlane)
{
    return froxelSliceCoord(viewDepth, nearPlane, farPlane);
}

// Continuous w-coordinate for a given linear view depth, NOT clamped to [0,1]: values below 0
// or above 1 mean the depth lies outside the [near, far] range of the grid the coordinate
// belongs to. World->froxel mapping uses this so bounds tests reject depths the target grid
// never covered (nearer than its near plane, farther than its far plane) instead of sampling
// the clamped boundary slice and treating it as history.
public float SliceCoordFromViewDepthUnclamped(float viewDepth, float nearPlane, float farPlane)
{
    float ratio = max(viewDepth, nearPlane) / max(nearPlane, kEpsilon);
    return log(max(ratio, 1.0001f)) / LogRatio(nearPlane, farPlane);
}

// Near and far edge view-depths of integer slice z (z in [0..D-1]).
public void FroxelSliceDepthRange(uint z, VolumetricPush pc, out float zNear, out float zFar)
{
    float invD = 1.0f / float(FroxelSize(pc).z);
    float tNear = float(z)        * invD;
    float tFar  = float(z + 1u)   * invD;
    zNear = ViewDepthFromFroxelCoord(tNear, pc.nearPlane, pc.farPlane);
    zFar  = ViewDepthFromFroxelCoord(tFar,  pc.nearPlane, pc.farPlane);
}

// Ray-parameter t (distance along a NORMALIZED ray starting at the camera, rebased-world axes)
// that corresponds to a linear view depth. The t <-> depth factor per ray is the angle cosine
// dot(rayDir, camForward): view depth is defined along the CAMERA forward axis while rayDir is
// in world axes. For every in-frustum ray this cosine is strictly positive, so the epsilon
// clamp is a numerical tripwire only, not a reachable path.
public float RayTFromViewDepth(float viewDepth, float3 rayDir, float3 camForward)
{
    return viewDepth / max(dot(rayDir, camForward), kEpsilon);
}

// -----------------------------------------------------------------------------
// NDC / UV / unprojection helpers.
// All unprojections are through the inverse VP, so off-axis perspective is exact.
// -----------------------------------------------------------------------------
// Unproject a clip-space point (u,v,ndcZ) -> camera-relative rebased-world point.
public float3 UnprojectNdc(float4x4 invViewProj, float u, float v, float ndcZ)
{
    float4 clip = float4(u, v, ndcZ, 1.0f);
    float4 w = mul(invViewProj, clip);
    return w.xyz / w.w;
}

// Camera forward direction in rebased-world axes, recovered from the same inverse VP the rays
// use (principal-point probe with FroxelRayDir's exact ndcZ convention), so the view-depth
// scale can never drift from the matrix actually shipped in this frame's matrix slot.
public float3 CameraForwardWorld(float4x4 invViewProj)
{
    float3 nearSide = UnprojectNdc(invViewProj, 0.0f, 0.0f, 1.0f);
    float3 farSide  = UnprojectNdc(invViewProj, 0.0f, 0.0f, 0.0f);
    return normalize(farSide - nearSide);
}

// Given a pixel NDC and ndcZ, return the linear view depth (depth along the camera forward
// axis, positive, blocks).
// Reversed-Z: ndcZ=1 at near plane, ndcZ=0 at sky/far.
// Returns a sentinel > farPlane for sky pixels.
public float LinearizeSample(VolumetricPush pc, float ndcZ, float u, float v)
{
    if (ndcZ <= 0.0001f) return pc.farPlane * 2.0f;
    FroxelMatrices m = GetMatrices(pc);
    float3 p = UnprojectNdc(m.invViewProj, u, v, ndcZ);
    return max(dot(p, CameraForwardWorld(m.invViewProj)), 0.0f);
}

// NDC u,v (x in [-1,1], y in [-1,1] with y UP) for a froxel XY index.
public void FroxelNdc(VolumetricPush pc, uint fx, uint fy, out float u, out float v)
{
    u = ((float(fx) + 0.5f) / float(FroxelSize(pc).x)) * 2.0f - 1.0f;
    v = 1.0f - ((float(fy) + 0.5f) / float(FroxelSize(pc).y)) * 2.0f;
}

// NDC u,v for a continuous froxel UV coordinate (fcx,fcy in [0,froxelSize]).
public void ContinuousFroxelNdc(VolumetricPush pc, float2 fxy, out float u, out float v)
{
    u = (fxy.x / float(FroxelSize(pc).x)) * 2.0f - 1.0f;
    v = 1.0f - (fxy.y / float(FroxelSize(pc).y)) * 2.0f;
}

// Screen-space UV (top-left origin, [0,1]) for a pixel integer id given dims.
public float2 PixelUv(uint2 xy, uint2 dims)
{
    return (float2(xy) + 0.5f) / float2(dims);
}

// NDC u,v ([-1,1], y UP) for a pixel integer id given dims.
public void PixelNdc(uint2 xy, uint2 dims, out float u, out float v)
{
    float2 uv = PixelUv(xy, dims);
    u = uv.x * 2.0f - 1.0f;
    v = (1.0f - uv.y) * 2.0f - 1.0f;
}

// Reconstruct NORMALIZED ray direction (rebased-world axes) through a froxel center.
// Also returns invViewProj for the caller's convenience.
public float3 FroxelRayDir(VolumetricPush pc, uint3 froxel, out float4x4 invViewProj)
{
    FroxelMatrices m = GetMatrices(pc);
    invViewProj = m.invViewProj;
    float u, v;
    FroxelNdc(pc, froxel.x, froxel.y, u, v);
    float3 wNear = UnprojectNdc(m.invViewProj, u, v, 1.0f);
    float3 wFar  = UnprojectNdc(m.invViewProj, u, v, 0.0f);
    return normalize(wFar - wNear);
}

// Stable camera-relative (rebased-world axes) point at the froxel's canonical center.
// Lies exactly on the slab plane of constant camera view depth for slice z, i.e. the same
// depth metric WorldToFroxelUvw derives from clip.w, so current-frame placement and
// previous-frame reprojection agree for every camera orientation.
public float3 FroxelCenterViewSpace(VolumetricPush pc, uint3 froxel)
{
    float4x4 invVP;
    float3 dir = FroxelRayDir(pc, froxel, invVP);
    float viewDepth = FroxelCenterViewDepth(pc, froxel.z);
    float t = RayTFromViewDepth(viewDepth, dir, CameraForwardWorld(invVP));
    return dir * t;
}

// Stable rebased-world point at the froxel's canonical center.
// This is the identity anchor for temporal reprojection: the same (ix,iy,iz)
// refers to the same physical point in the world for a given camera/frustum.
public float3 FroxelCenterWorldSpace(VolumetricPush pc, uint3 froxel)
{
    return FroxelCenterViewSpace(pc, froxel) + pc.camOffset;
}

// Render-res pixel ray direction (normalized), also returns NDC u,v.
public float3 PixelViewRay(VolumetricPush pc, uint2 xy, uint2 dims, out float outU, out float outV)
{
    FroxelMatrices m = GetMatrices(pc);
    PixelNdc(xy, dims, outU, outV);
    float3 wNear = UnprojectNdc(m.invViewProj, outU, outV, 1.0f);
    float3 wFar  = UnprojectNdc(m.invViewProj, outU, outV, 0.0f);
    return normalize(wFar - wNear);
}

// -----------------------------------------------------------------------------
// World -> froxel mapping (for temporal reprojection and history sampling).
// Uses clip.w (= -view.z = linear view depth, since view matrices are rotation-only).
//
// The camera offset is passed in explicitly so the same routine handles both
// current (camOffset) and previous (prevCamOffset) frames; the caller chooses
// the matching view-projection matrix.
//
// Returns float3(u_froxel_01, v_froxel_01, w_froxel) where (u,v) are in
// screen-UV convention (origin top-left, like the Load coordinate), and w is
// the continuous slice coordinate with w=0 at the near plane and w=1 at the far
// plane. ALL three axes are UNCLAMPED: any component outside [0,1] means the
// point is off the froxel grid, and callers must bounds-test before sampling.
// (Behind/inside-near points return (-1,-1,-1) via the clip.w test above.)
// -----------------------------------------------------------------------------
public float3 WorldToFroxelUvw(VolumetricPush pc, float3 worldPos, float3 camOff, float4x4 viewProj)
{
    float3 rel = worldPos - camOff;
    float4 clip = mul(viewProj, float4(rel, 1.0f));
    if (clip.w <= kEpsilon) return float3(-1.0f, -1.0f, -1.0f);
    float3 ndc = clip.xyz / clip.w;
    float viewDepth = clip.w;  // depth along the camera forward axis (see glossary: clip)
    float u = ndc.x * 0.5f + 0.5f;
    float v = (1.0f - ndc.y) * 0.5f;
    float w = SliceCoordFromViewDepthUnclamped(viewDepth, pc.nearPlane, pc.farPlane);
    return float3(u, v, w);
}

public float3 WorldToFroxelCurrent(VolumetricPush pc, float3 worldPos)
{
    FroxelMatrices m = GetMatrices(pc);
    return WorldToFroxelUvw(pc, worldPos, pc.camOffset, m.curViewProj);
}

public float3 WorldToFroxelPrevious(VolumetricPush pc, float3 worldPos)
{
    FroxelMatrices m = GetMatrices(pc);
    return WorldToFroxelUvw(pc, worldPos, pc.prevCamOffset, m.prevViewProj);
}

// Map a froxel XY (in froxel-grid space) to the corresponding gDepth pixel
// coordinate (render res). Returns integer pixel coords, already clamped.
public int2 FroxelXyToDepthPx(uint2 froxelXy, uint2 froxelXySize, uint2 depthDims)
{
    float2 uv = (float2(froxelXy) + 0.5f) / float2(froxelXySize);
    int2 px = int2(uv * float2(depthDims));
    return int2(
        clamp(px.x, 0, (int)depthDims.x - 1),
        clamp(px.y, 0, (int)depthDims.y - 1)
    );
}

// Convenience for sampling gDepth at a given froxel XY: returns the integer
// pixel and the corresponding NDC u,v (for LinearizeSample).
public void FroxelDepthSampleLoc(uint2 froxelXy, uint2 froxelXySize, uint2 depthDims,
                                 out int2 px, out float u, out float v)
{
    px = FroxelXyToDepthPx(froxelXy, froxelXySize, depthDims);
    float2 uv = (float2(px) + 0.5f) / float2(depthDims);
    u = uv.x * 2.0f - 1.0f;
    v = (1.0f - uv.y) * 2.0f - 1.0f;
}

// -----------------------------------------------------------------------------
// Phase function
// -----------------------------------------------------------------------------
public float HenyeyGreenstein(float cosTheta, float g)
{
    // g > 0 = forward scattering, g = 0 = isotropic, g < 0 = backward scattering.
    // cosTheta is evaluated between directions with a single convention throughout this pipeline:
    //   rayDirCameraToSample = camera -> sample
    //   sampleToLightDir     = sample -> light/sky probe
    // dot(rayDirCameraToSample, sampleToLightDir) equals dot(incoming photon direction,
    // outgoing direction toward the camera), because both physical directions are negated.
    g = clamp(g, -0.98f, 0.98f);
    cosTheta = clamp(cosTheta, -1.0f, 1.0f);
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosTheta;
    return (1.0f - g2) / (4.0f * kPi * pow(max(denom, kEpsilon), 1.5f));
}

// -----------------------------------------------------------------------------
// Explicit participating-medium model.
// -----------------------------------------------------------------------------
public struct FroxelMedium
{
    public float  density;       // approximately 1/block, before RGB albedo/tint splitting
    public float3 sigmaS;        // scattering coefficient, 1/block
    public float3 sigmaA;        // absorption coefficient, 1/block
    public float3 sigmaT;        // extinction coefficient = sigmaS + sigmaA, 1/block
    public float  anisotropyG;   // HG g
    public float  surfaceAbsorptionWeight; // 0 for water absorption already applied by the primary path
    public float  water;         // 1 for water medium, 0 for air
};

public float Luminance(float3 v)
{
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

// Integer-cell hash and smooth value noise. Density modulation is intentionally world-space only:
// the same static Minecraft position yields the same medium density regardless of camera motion,
// dynamic resolution, DLSS jitter, or froxel-grid movement.
public float HashCell(int3 p)
{
    uint h = uint(p.x) * 0x8da6b343u ^ uint(p.y) * 0xd8163841u ^ uint(p.z) * 0xcb1ab31fu;
    return Hash1(h);
}

public float ValueNoise3D(float3 p)
{
    int3 i = int3(floor(p));
    float3 f = frac(p);
    f = f * f * (3.0f - 2.0f * f);

    float n000 = HashCell(i + int3(0,0,0));
    float n100 = HashCell(i + int3(1,0,0));
    float n010 = HashCell(i + int3(0,1,0));
    float n110 = HashCell(i + int3(1,1,0));
    float n001 = HashCell(i + int3(0,0,1));
    float n101 = HashCell(i + int3(1,0,1));
    float n011 = HashCell(i + int3(0,1,1));
    float n111 = HashCell(i + int3(1,1,1));

    float n00 = lerp(n000, n100, f.x);
    float n10 = lerp(n010, n110, f.x);
    float n01 = lerp(n001, n101, f.x);
    float n11 = lerp(n011, n111, f.x);
    float n0 = lerp(n00, n10, f.y);
    float n1 = lerp(n01, n11, f.y);
    return lerp(n0, n1, f.z);
}

// Artistic Minecraft fog density -> renderer coefficients conversion. The only place that converts
// height/global/weather density controls into sigma_t for air is here. Extinction is evaluated on
// rebased-world positions so it stays anchored to terrain as the camera moves: pc.heightFogBase is
// pre-shifted into this same rebase frame by the host.
public float AirDensityAt(VolumetricPush pc, float3 worldPos)
{
    FroxelMatrices f = GetMatrices(pc);
    float height = worldPos.y - pc.heightFogBase;
    float heightTerm = exp(-max(0.0f, height) * max(pc.heightFogFalloff, 0.0f));
    float baseDensity = pc.heightFogDensity * heightTerm + pc.globalDensity;

    // Large-scale, world-anchored density variation: subtle enough not to become cloud rendering,
    // strong enough to prevent perfectly flat slabs at long distances. The noise coordinate adds the
    // current terrain rebase origin back to the rebased froxel position, so a static world point keeps
    // the same density through camera motion and origin rebases.
    float3 noisePos = worldPos + f.worldOrigin.xyz;
    float n0 = ValueNoise3D(noisePos * 0.01875f);
    float n1 = ValueNoise3D(noisePos * 0.05500f + float3(17.0f, 3.0f, 29.0f));
    float modulation = 0.82f + 0.28f * n0 + 0.10f * n1;

    return max(baseDensity * modulation * max(f.weatherParams.z, 0.0f), 0.0f);
}

// Back-compatible name used by older passes/debug. Returns scalar air sigma_t.
public float ExtinctionAt(VolumetricPush pc, float3 worldPos)
{
    return AirDensityAt(pc, worldPos);
}

public float3 AirScatteringTint()
{
    // Slight blue tint in sigma_s keeps air shafts neutral/warm near the sun and faintly blue in skylight.
    return float3(0.90f, 0.95f, 1.00f);
}

public FroxelMedium EvaluateMedium(VolumetricPush pc, float3 worldPos)
{
    FroxelMatrices f = GetMatrices(pc);
    FroxelMedium m;

    bool inWater = f.waterTintFlags.w > 0.5f;
    if (inWater)
    {
        float3 tint = clamp(f.waterTintFlags.xyz, float3(0.02f, 0.02f, 0.02f), float3(1.0f, 1.0f, 1.0f));
        float density = max(f.waterParams.x, 0.0f);

        // Water absorption matches the surface path's Beer-Lambert convention: biome tint removes
        // red/green faster than blue. Scattering is separate and does not change the surface refraction
        // or caustic model.
        float3 absorbFloor = float3(0.015f, 0.010f, 0.008f);
        m.sigmaA = max(f.waterParams.z, 0.0f) * (absorbFloor + density * (float3(1.0f, 1.0f, 1.0f) - tint));
        m.sigmaS = max(f.waterParams.y, 0.0f) * density * lerp(float3(0.45f, 0.45f, 0.45f), tint, 0.65f);
        m.sigmaT = m.sigmaS + m.sigmaA;
        m.density = max(Luminance(m.sigmaT), 0.0f);
        m.anisotropyG = clamp(f.waterParams.w, -0.98f, 0.98f);
        m.surfaceAbsorptionWeight = clamp(f.airParams.w, 0.0f, 1.0f);
        m.water = 1.0f;
        return m;
    }

    float sigma = AirDensityAt(pc, worldPos);
    float airAlbedo = clamp(f.airParams.x, 0.0f, 0.999f);
    m.sigmaS = sigma * airAlbedo * AirScatteringTint();
    m.sigmaA = max(float3(sigma, sigma, sigma) - m.sigmaS, float3(0.0f, 0.0f, 0.0f));
    m.sigmaT = m.sigmaS + m.sigmaA;
    m.density = sigma;
    m.anisotropyG = clamp(pc.anisotropyG, -0.98f, 0.98f);
    m.surfaceAbsorptionWeight = 1.0f;
    m.water = 0.0f;
    return m;
}

public float FroxelMediumSigmaLuma(FroxelMedium m)
{
    return max(Luminance(m.sigmaT), 0.0f);
}

public float3 FroxelSkyProbeDirection(uint probeIndex)
{
    // Fixed world-space sky probes. They are not randomized per froxel/frame, so caves and forests
    // converge without temporal flashing. Keep the count small because each probe is a TLAS ray query.
    if (probeIndex == 0u) return float3(0.0f, 1.0f, 0.0f);
    return normalize(float3(0.62f, 0.70f, 0.34f));
}

public static const uint kFroxelSkyProbeCount = 2u;

// -----------------------------------------------------------------------------
// Sun-disk sampling.
// xi is a 2D uniform point in [0,1]^2 that is mapped to a point on the sun disk
// (concentric disk mapping projected onto the unit sphere about sunDir).
// When xi=(0.5,0.5) (i.e. fixed-sun mode) this returns exactly sunDir.
// -----------------------------------------------------------------------------
public float3 SampleSunDisk(float3 sunDir, float angularRadius, float2 xi)
{
    // Concentric disk.
    float2 t = xi - 0.5f;
    if (angularRadius <= 1.0e-5f || (abs(t.x) < 1.0e-5f && abs(t.y) < 1.0e-5f))
        return sunDir;
    float r, theta;
    if (abs(t.x) > abs(t.y)) {
        r = t.x;
        theta = kPi * 0.25f * (t.y / t.x);
    } else {
        r = t.y;
        theta = kPi * 0.5f - kPi * 0.25f * (t.x / t.y);
    }
    float diskX = r * cos(theta) * sin(angularRadius);
    float diskY = r * sin(theta) * sin(angularRadius);
    float diskZ = cos(angularRadius);

    // Build a tangent frame around sunDir.
    float3 up = abs(sunDir.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 T = normalize(cross(up, sunDir));
    float3 B = cross(sunDir, T);
    return normalize(T * diskX + B * diskY + sunDir * diskZ);
}

// -----------------------------------------------------------------------------
// Exact transmittance / analytic in-scattered source integration.
// T(s) = exp(-sigma * s)
// integ = S * (1 - T)/sigma  (scalar sigma; S already includes sigma_s).
// Integrator inlines these for clarity.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Debug visualization helpers.
// A world-anchored repeating 4-block checker pattern -- if this swims as the
// camera translates, coordinate spaces are being mixed.
// -----------------------------------------------------------------------------
public float WorldChecker(float3 worldPos, float scale)
{
    float3 p = floor(worldPos * scale);
    return (fract((p.x + p.y + p.z) * 0.5f) > 0.25f) ? 1.0f : 0.0f;
}
