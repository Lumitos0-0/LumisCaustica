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
//   view          : camera-relative space looking down -Z (same orientation as rebased-world but
//                   translated so the camera is at origin). view.z is negative for points in front;
//                   "linear view depth" = -view.z (positive, in blocks).
//   clip          : post-projection homogeneous coordinates (curViewProj * view). clip.w = -view.z
//                   for a rotation-only view times perspective projection (true for both curVP and
//                   prevVP because frameViewRotation is rotation-only).
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
// The three camera matrices (invViewProj, curViewProj, prevViewProj) live in a small storage
// buffer accessed via device address so that push constants stay under the 128-byte
// minimum-max-guaranteed Vulkan limit.
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
};

// -----------------------------------------------------------------------------
// Push constants (must stay <= 128 bytes)
// -----------------------------------------------------------------------------
//
// debugView selects the visualization mode (all writes go through integrate.comp):
//    0 = final composite (L + T * scene)
//    1 = Z-slice index (color-coded)
//    2 = world-position checker (verifies density is world-anchored; should not swim)
//    3 = extinction / density only
//    4 = raw binary RT visibility (0 or 1, before spatial filter)
//    5 = fogCurrent content at integrate time (post-temporal accumulated radiance;
//        temporal overwrites the pre-spatial injection in its place)
//    6 = fogVolume tap at integrate time (post-temporal; the pre-temporal spatial
//        output is overwritten by the fogCurrent -> fogVolume copy)
//    7 = history confidence (black=0, white=1)
//    8 = disocclusion / history-rejected mask (white=rejected)
//    9 = integrated in-scattered radiance (pre-composite over scene)
//   10 = final volumetric contribution (L + (T-1)*scene) over neutral gray
public struct VolumetricPush
{
    public uint64_t matricesAddr;                              // offset  0, size  8
    // vec3 fields are 16-byte aligned in std430; each is followed by a float to pack tightly.
    public float3   camOffset;        // 16, camera position in CURRENT rebased world
    public float    nearPlane;        // 28, froxel near plane (blocks)
    public float3   prevCamOffset;    // 32, prev camera position in CURRENT rebased world
    public float    farPlane;         // 44, froxel far plane (blocks)
    public float3   sunDir;           // 48, normalized, toward the sun (translation-invariant)
    public float    sunAngularRadius; // 60, radians
    public float3   sunColor;         // 64, phase-ready direct radiance (preExposure*E/pi)
    public float    heightFogDensity; // 76, sigma_t at Y=heightFogBase (1/blocks)
    public float    heightFogFalloff; // 80
    public float    heightFogBase;    // 84, world Y at which density = heightFogDensity
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

public float ViewDepthFromFroxelCoord(float t, float nearPlane, float farPlane)
{
    t = clamp(t, 0.0f, 1.0f);
    return nearPlane * exp(t * LogRatio(nearPlane, farPlane));
}

// Canonical stable center depth for slice z (integer) -- center at t=(z+0.5)/D.
public float FroxelCenterViewDepth(VolumetricPush pc, uint zSlice)
{
    float t = (float(zSlice) + 0.5f) / float(FroxelSize(pc).z);
    return ViewDepthFromFroxelCoord(t, pc.nearPlane, pc.farPlane);
}

// Continuous w-coordinate in [0,1] for a given linear view depth.
public float SliceCoordFromViewDepth(float viewDepth, float nearPlane, float farPlane)
{
    float ratio = max(viewDepth, nearPlane) / max(nearPlane, kEpsilon);
    float w = log(max(ratio, 1.0001f)) / LogRatio(nearPlane, farPlane);
    return clamp(w, 0.0f, 1.0f);
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

// Ray-parameter t (distance along a NORMALIZED ray starting at origin) that corresponds
// to a given linear view depth. For a camera ray, viewDepth = -rayDir.z * t when the
// ray starts at the camera. Used to map depth-slab ranges to ray segments.
public float RayTFromViewDepth(float viewDepth, float3 rayDir)
{
    return viewDepth / max(-rayDir.z, kEpsilon);
}

// -----------------------------------------------------------------------------
// NDC / UV / unprojection helpers.
// All unprojections are through the inverse VP, so off-axis perspective is exact.
// -----------------------------------------------------------------------------
// Unproject a clip-space point (u,v,ndcZ) -> camera-relative point.
public float3 UnprojectNdc(float4x4 invViewProj, float u, float v, float ndcZ)
{
    float4 clip = float4(u, v, ndcZ, 1.0f);
    float4 w = mul(invViewProj, clip);
    return w.xyz / w.w;
}

// Given a pixel NDC and ndcZ, return linear view depth (-p.z, positive, blocks).
// Reversed-Z: ndcZ=1 at near plane, ndcZ=0 at sky/far.
// Returns a sentinel > farPlane for sky pixels.
public float LinearizeSample(VolumetricPush pc, float ndcZ, float u, float v)
{
    if (ndcZ <= 0.0001f) return pc.farPlane * 2.0f;
    FroxelMatrices m = GetMatrices(pc);
    float3 p = UnprojectNdc(m.invViewProj, u, v, ndcZ);
    return max(-p.z, 0.0f);
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

// Reconstruct NORMALIZED view-space ray direction through a froxel center.
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

// Stable camera-relative (view-space) point at the froxel's canonical center.
public float3 FroxelCenterViewSpace(VolumetricPush pc, uint3 froxel)
{
    float4x4 invVP;
    float3 dir = FroxelRayDir(pc, froxel, invVP);
    float viewDepth = FroxelCenterViewDepth(pc, froxel.z);
    float t = RayTFromViewDepth(viewDepth, dir);
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
    float viewDepth = clip.w;  // = -rel.z (view space)
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
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosTheta;
    return (1.0f - g2) / (4.0f * kPi * pow(max(denom, kEpsilon), 1.5f));
}

// -----------------------------------------------------------------------------
// Density / extinction model.
// Extinction is evaluated in WORLD space (rebased) so it stays anchored to
// terrain as the camera moves: height falloff uses worldPos.y, not view.y.
// -----------------------------------------------------------------------------
public float ExtinctionAt(VolumetricPush pc, float3 worldPos)
{
    float height = worldPos.y - pc.heightFogBase;
    float heightTerm = exp(-max(0.0f, height) * max(pc.heightFogFalloff, 0.0f));
    return pc.heightFogDensity * heightTerm + pc.globalDensity;
}

// Single-scattering albedo ~0.95-1.0 (conservative; fog is highly scattering).
// Slight forward-blue tint gives sun shafts a subtle Rayleigh-like warmth forward/blue backward.
public float3 ScatteringAlbedo()
{
    return float3(0.9f, 0.95f, 1.0f);
}

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
