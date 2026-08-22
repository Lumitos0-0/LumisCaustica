// Volumetric fog shared math and froxel mapping.
// This file centralizes the froxel <-> view/world mapping so every pass uses the SAME
// deterministic identity, satisfying Invariant 1 (a froxel is identified by its stable center).
//
// The three camera matrices (invViewProj, curViewProj, prevViewProj) live in a small storage
// buffer accessed via device address so that push constants stay under the 128-byte
// minimum-max-guarantee on every Vulkan device.
//
// No explicit `module` declaration: Slang uses the file stem ("common") as the
// module name, so `import common;` from a volumetric pass resolves to this file
// (the same-directory -I entry is searched first).
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
    // Camera-relative rebased world -> previous-frame clip space (jitter-free, for reprojection).
    public float4x4 prevViewProj;
};

// -----------------------------------------------------------------------------
// Push constants (must stay <= 128 bytes)
// -----------------------------------------------------------------------------
// Layout notes:
//   Froxel grid is camera-frustum aligned:
//     froxelSize.xy = renderRes / xyDivisor
//     froxelSize.z  = depthSlices (typically 64)
//
//   depth distribution is exponential/logarithmic between nearPlane and farPlane.
//   Distribution uses zSlice in [0, depthSlices) with stable center at (z + 0.5) / depthSlices.
//
//   To keep history valid across frames, nearPlane/farPlane/depthSlices/xyDivisor
//   should be STABLE; when they change, historyValid=0 invalidates all of it.
//
//   debugView selects the visualization mode:
//     0  off (final composite)
//     1  Z-slice index
//     2  world-position checker (verifies density is world-anchored)
//     3  extinction / density only
//     4  raw RT visibility (0/1)
//     5  current injected lighting (before spatial/temporal)
//     6  reprojected history RGB
//     7  history confidence (black=0, white=1)
//     8  disocclusion / history rejected
//     9  integrated scattering (before composite)
//    10  final volumetric contribution composited over scene
public struct VolumetricPush
{
    // Device pointer to a FroxelMatrices struct (std430).
    public uint64_t matricesAddr;                              // offset  0, size  8

    // vec3 fields must be 16-byte aligned in std430. Each is followed by ONE float
    // to avoid wasting the 4-byte tail, then the next vec3 starts at the next 16-byte
    // boundary. This keeps the total struct exactly 128 bytes (Vulkan's min-max-guaranteed
    // push constant size). Order matters — do not insert new scalars between a vec3 and
    // its trailing scalar; that adds 12 bytes of implicit padding per insertion.
    public float3   camOffset;        // 16, camera position in rebased world (current frame)
    public float    nearPlane;        // 28, froxel near plane (blocks)
    public float3   prevCamOffset;    // 32, previous frame camera position in rebased world
    public float    farPlane;         // 44, froxel far plane (blocks)
    public float3   sunDir;           // 48, normalized, toward the sun (direction, translation-invariant)
    public float    sunAngularRadius; // 60, radians
    public float3   sunColor;         // 64, direct radiance scale, phase-ready (includes preExposure*E/pi)
    public float    heightFogDensity; // 76, sigma_t at Y=heightFogBase (1/blocks)

    // Mixed scalars from here on (4-byte aligned; no more vec3s). Std430 packs them
    // back-to-back, so ordering here directly controls final size.
    public float    heightFogFalloff; // 80
    public float    heightFogBase;    // 84, world Y at which density = heightFogDensity
    public float    globalDensity;    // 88
    public float    anisotropyG;      // 92, Henyey-Greenstein g parameter
    public float    historyBaseWeight;// 96, baseline history lerp weight (e.g. 0.9)
    public float    historyValid;     // 100, 0 on grid/reset, 1 otherwise
    public uint     froxelSizeX;      // 104, froxel grid width
    public uint     froxelSizeY;      // 108, froxel grid height
    public uint     froxelSizeZ;      // 112, froxel depth-slice count
    public uint     frameIndex;       // 116
    public uint     debugView;        // 120
    public uint     stochasticLight;  // 124
    // sizeof(VolumetricPush) = 128 (Vulkan's min-max-guaranteed push constant size).
    // Do NOT add new fields here without re-verifying the offset math; if more data is
    // needed, extend FroxelMatrices (BDA storage buffer) instead.
};

// [[vk::push_constant]] is declared in each .comp.slang file that consumes this,
// because Slang allows only one push-constant block per compilation unit and the
// layout probe imports this module separately.

static const float kEpsilon = 1.0e-6f;
static const float kPi = 3.14159265f;

// Read-only device pointer matching the codebase convention in world_common.slang.
public typealias ConstPtr<T> = Ptr<T, Access.Read, AddressSpace.Device, Std430DataLayout>;

// Convenience accessor: froxel grid dimensions as a uint3. The push constant stores
// X/Y/Z as three separate scalars to avoid std430 vec3 alignment padding that would push
// the struct beyond the 128-byte Vulkan push-constant minimum.
public uint3 FroxelSize(VolumetricPush pc)
{
    return uint3(pc.froxelSizeX, pc.froxelSizeY, pc.froxelSizeZ);
}

// -----------------------------------------------------------------------------
// Matrix buffer accessor
// -----------------------------------------------------------------------------
public FroxelMatrices GetMatrices(VolumetricPush pc)
{
    return ConstPtr<FroxelMatrices>(pc.matricesAddr)[0];
}

// -----------------------------------------------------------------------------
// Hash / blue-noise-like dither (deterministic for (froxel, frame))
// -----------------------------------------------------------------------------
float BlueNoise(uint3 froxel, uint frame)
{
    uint x = froxel.x * 19349663u + froxel.y * 83492791u + froxel.z * 2654435761u + frame * 9742139u;
    x ^= x >> 13;
    x *= 0x5bd1e995u;
    x ^= x >> 15;
    return float(x & 0x00FFFFFFu) / float(0x01000000u);
}

float2 BlueNoise2(uint3 froxel, uint frame)
{
    return float2(
        BlueNoise(froxel, frame),
        BlueNoise(froxel + uint3(17u, 41u, 97u), frame ^ 0x12345678u)
    );
}

// -----------------------------------------------------------------------------
// Froxel <-> depth mapping (logarithmic/exponential)
// -----------------------------------------------------------------------------
// Exponential depth distribution: viewDepth(z) = nearPlane * (farPlane/nearPlane)^(z/depthSlices)
// This is camera-anchored in Z, not world-anchored, which is correct for a frustum-aligned froxel
// grid. Z is measured in linear VIEW SPACE depth (positive, along camera -Z) in blocks.
float ViewDepthFromFroxelSlice(float zSlice, float nearPlane, float farPlane, uint depthSlices)
{
    float t = clamp(zSlice, 0.0f, 1.0f);
    float ratio = max(farPlane / max(nearPlane, kEpsilon), 1.0001f);
    return nearPlane * pow(ratio, t);
}

float FroxelCenterViewDepth(VolumetricPush pc, uint zSlice)
{
    float z = (float(zSlice) + 0.5f) / float(FroxelSize(pc).z);
    return ViewDepthFromFroxelSlice(z, pc.nearPlane, pc.farPlane, FroxelSize(pc).z);
}

float SliceCoordFromViewDepth(float viewDepth, float nearPlane, float farPlane, uint depthSlices)
{
    float ratio = max(viewDepth, nearPlane) / max(nearPlane, kEpsilon);
    float z = log(max(ratio, 1.0001f)) / log(max(farPlane / max(nearPlane, kEpsilon), 1.0001f));
    return clamp(z, 0.0f, 1.0f);
}

void FroxelSliceDepthRange(uint z, float nearPlane, float farPlane, uint depthSlices,
                           out float zNear, out float zFar)
{
    float tNear = (float(z) + 0.0f) / float(depthSlices);
    float tFar  = (float(z) + 1.0f) / float(depthSlices);
    float ratio = max(farPlane / max(nearPlane, kEpsilon), 1.0001f);
    zNear = nearPlane * pow(ratio, tNear);
    zFar  = nearPlane * pow(ratio, tFar);
}

// -----------------------------------------------------------------------------
// Froxel center (view space and camera-relative world)
// -----------------------------------------------------------------------------
// Unproject a clip-space point (u,v,ndcZ) through the inverse view-projection into
// camera-relative world. Handles reversed-Z and perspective W correctly.
// u,v are in [-1,1] NDC X/Y; ndcZ is reversed-Z (1=near, 0=sky/far).
float3 UnprojectNdc(float4x4 invViewProj, float u, float v, float ndcZ)
{
    float4 clip = float4(u, v, ndcZ, 1.0f);
    float4 w = mul(invViewProj, clip);
    return w.xyz / w.w;
}

// Given a pixel's NDC (u,v) and an ndcZ value, return the linear VIEW-SPACE depth
// (positive, measured from camera along the -Z axis in camera-relative space, in blocks).
// Reversed-Z: ndcZ=1 at near, ndcZ=0 at far/sky.
float LinearizeDepthNdc(float4x4 invViewProj, float u, float v, float ndcZ)
{
    // ndcZ <= 0 means sky/far plane; return a sentinel larger than froxel far.
    if (ndcZ <= 0.0001f) return 1.0e9f;
    float3 p = UnprojectNdc(invViewProj, u, v, ndcZ);
    return max(-p.z, 0.0f);
}

// Reconstruct NORMALIZED view-space direction for a froxel center (camera-relative).
// Uses inverse-VP unprojection of clip-near and clip-far points so perspective is exact
// off-axis (no central-axis approximation).
float3 FroxelRayDir(VolumetricPush pc, uint3 froxel, out float4x4 invViewProj)
{
    FroxelMatrices m = GetMatrices(pc);
    invViewProj = m.invViewProj;
    float u = ((float(froxel.x) + 0.5f) / float(FroxelSize(pc).x)) * 2.0f - 1.0f;
    float v = 1.0f - ((float(froxel.y) + 0.5f) / float(FroxelSize(pc).y)) * 2.0f;
    float3 wNear = UnprojectNdc(m.invViewProj, u, v, 1.0f);
    float3 wFar  = UnprojectNdc(m.invViewProj, u, v, 0.0f);
    return normalize(wFar - wNear);
}

// Camera-relative world point at the froxel's stable center.
float3 FroxelCenterViewSpace(VolumetricPush pc, uint3 froxel)
{
    float4x4 invVP;
    float3 dir = FroxelRayDir(pc, froxel, invVP);
    float viewDepth = FroxelCenterViewDepth(pc, froxel.z);
    float t = viewDepth / max(-dir.z, kEpsilon);
    return dir * t;
}

float3 FroxelCenterWorldSpace(VolumetricPush pc, uint3 froxel)
{
    return FroxelCenterViewSpace(pc, froxel) + pc.camOffset;
}

// Reconstruct view-ray direction AND origin for a RENDER-RES pixel.
// xy = integer pixel id; dims = render resolution. Returns normalized direction;
// also gives the NDC u,v for depth linearization.
float3 PixelViewRay(VolumetricPush pc, uint2 xy, uint2 dims, out float outU, out float outV)
{
    FroxelMatrices m = GetMatrices(pc);
    float u = ((float(xy.x) + 0.5f) / float(dims.x)) * 2.0f - 1.0f;
    float v = 1.0f - ((float(xy.y) + 0.5f) / float(dims.y)) * 2.0f;
    outU = u; outV = v;
    float3 wNear = UnprojectNdc(m.invViewProj, u, v, 1.0f);
    float3 wFar  = UnprojectNdc(m.invViewProj, u, v, 0.0f);
    return normalize(wFar - wNear);
}

// -----------------------------------------------------------------------------
// World-to-froxel mapping (for temporal reprojection).
// Uses clip.w (which equals linear view depth for a perspective projection where the
// view matrix has no translation relative to the camera-relative origin) rather than
// -rel.z, which is only correct when the camera is axis-aligned. This matters because
// the player can look up/down freely: using -rel.z would give a view-space Z that changes
// with rotation, producing sloshing depth slices when turning the camera.
// -----------------------------------------------------------------------------
float3 WorldToFroxel(VolumetricPush pc, float3 worldPos, float3 camOff, float4x4 viewProj)
{
    float3 rel = worldPos - camOff;
    float4 clip = mul(viewProj, float4(rel, 1.0f));
    // Behind the camera: clip.w <= 0 means the point projects behind the viewer.
    if (clip.w <= kEpsilon) return float3(-1.0f, -1.0f, -1.0f);
    float3 ndc = clip.xyz / clip.w;
    // Linear view depth = -rel_z_view = clip.w for a standard perspective projection whose
    // view matrix has no translation relative to the camera-relative origin (which is our case:
    // viewProj = frameProjection * frameViewRotation where frameViewRotation is rotation only).
    float viewDepth = clip.w;
    float u = ndc.x * 0.5f + 0.5f;
    float v = (1.0f - ndc.y) * 0.5f;
    float w = SliceCoordFromViewDepth(viewDepth, pc.nearPlane, pc.farPlane, FroxelSize(pc).z);
    return float3(u, v, w);
}

float3 WorldToFroxelCurrent(VolumetricPush pc, float3 worldPos)
{
    FroxelMatrices m = GetMatrices(pc);
    return WorldToFroxel(pc, worldPos, pc.camOffset, m.curViewProj);
}

float3 WorldToFroxelPrevious(VolumetricPush pc, float3 worldPos)
{
    FroxelMatrices m = GetMatrices(pc);
    return WorldToFroxel(pc, worldPos, pc.prevCamOffset, m.prevViewProj);
}

// -----------------------------------------------------------------------------
// Phase function
// -----------------------------------------------------------------------------
float HenyeyGreenstein(float cosTheta, float g)
{
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosTheta;
    return (1.0f - g2) / (4.0f * kPi * pow(max(denom, kEpsilon), 1.5f));
}

// -----------------------------------------------------------------------------
// Density / extinction model
// -----------------------------------------------------------------------------
// Scalar sigma_t at a world position. Coloured extinction is derived from it by
// multiplying by a fixed Rayleigh-like (0.6, 0.8, 1.1) tint.
float ExtinctionAt(VolumetricPush pc, float3 worldPos)
{
    float height = worldPos.y - pc.heightFogBase;
    float heightTerm = exp(-max(0.0f, height) * max(pc.heightFogFalloff, 0.0f));
    float sigma = pc.heightFogDensity * heightTerm + pc.globalDensity;
    return sigma;
}

float3 SigmaTFromLuma(float sigmaLuma)
{
    return float3(0.6f, 0.8f, 1.1f) * sigmaLuma;
}

float3 ScatteringSigma(float3 sigma_t)
{
    // Scattering albedo ~0.95-1.0; slight spectral variation keeps dense fog from
    // glowing pure white and gives a soft blue tint (Rayleigh-like).
    return sigma_t * float3(0.95f, 0.97f, 1.0f);
}

// -----------------------------------------------------------------------------
// Sun-disk sampling
// -----------------------------------------------------------------------------
float3 SampleSunDisk(float3 sunDir, float angularRadius, float2 xi)
{
    float theta = angularRadius * sqrt(xi.x);
    float phi   = 2.0f * kPi * xi.y;
    float3 up = abs(sunDir.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 t = normalize(cross(up, sunDir));
    float3 b = cross(sunDir, t);
    return normalize(sunDir * cos(theta) + (t * cos(phi) + b * sin(phi)) * sin(theta));
}

// -----------------------------------------------------------------------------
// Depth helpers (NDC math only; gDepth is read per-pass via bindings so common.slang
// doesn't need to import bindings.slang — see note at the top of this file).
// -----------------------------------------------------------------------------

// Map a froxel XY (in froxel-grid space) to the nearest render-res pixel coordinate
// for sampling gDepth. Returns integer pixel coords (nearest center), already clamped.
int2 FroxelXyToDepthPx(uint2 froxelXy, uint2 froxelXySize, uint2 depthDims)
{
    float2 uv = (float2(froxelXy) + 0.5f) / float2(froxelXySize);
    int2 px = int2(uv * float2(depthDims));
    return int2(clamp(px.x, 0, (int)depthDims.x - 1), clamp(px.y, 0, (int)depthDims.y - 1));
}

// Linearize a single gDepth sample (reversed-Z ndcZ = clip.z / clip.w, 1 at near,
// 0 at sky/far) at the given integer pixel position. Caller provides ndcZ and the
// pixel's NDC (u,v); this unprojects through invViewProj so view depth is correct
// off-axis. Returns linear view-space depth in blocks, or a sentinel > farPlane
// for sky pixels.
float LinearizeSample(VolumetricPush pc, float ndcZ, float u, float v)
{
    if (ndcZ <= 0.0001f) return pc.farPlane * 2.0f;
    FroxelMatrices m = GetMatrices(pc);
    float3 p = UnprojectNdc(m.invViewProj, u, v, ndcZ);
    return max(-p.z, 0.0f);
}

// -----------------------------------------------------------------------------
// Transmittance / exact volumetric integration helpers
// -----------------------------------------------------------------------------
float3 Transmittance(float3 sigma_t, float ds)
{
    return exp(-sigma_t * ds);
}

// Analytic in-scattered source for a segment with constant S and sigma_t:
//   integ = S * (1 - exp(-sigma_t * ds)) / sigma_t
// with a linear fallback as sigma_t -> 0 (S * ds) to avoid division by zero.
float3 IntegratedSource(float3 S, float3 sigma_t, float ds)
{
    float3 T = Transmittance(sigma_t, ds);
    float3 safeSigma = max(sigma_t, float3(kEpsilon, kEpsilon, kEpsilon));
    float3 integ = S * (float3(1.0f, 1.0f, 1.0f) - T) / safeSigma;
    // Per-component linear fallback for extremely low density.
    if (sigma_t.x < kEpsilon) integ.x = S.x * ds;
    if (sigma_t.y < kEpsilon) integ.y = S.y * ds;
    if (sigma_t.z < kEpsilon) integ.z = S.z * ds;
    return integ;
}
