// Shared froxel-grid definitions: the frustum-aligned volume layout, the fog density model, the
// scattering phase function, and the push-constant block both froxel compute passes read. This module is
// self-contained; its consumers (lighting/integrate) import world_common directly for the WorldPush
// device-pointer aliases and the ACEScg seam.
module froxel_common;

// The exponential depth distribution's near base, in blocks, is carried as FroxelPush.params.x by the
// Java side (RtFroxel pushes 0.05). This fixed tiny distance makes the first slice begin essentially at
// the near plane (the ray origin the world pass uses), giving dense cells near the camera and coarse cells
// far out — the whole point of a frustum-aligned volume.

// Push-constant block for both froxel compute passes. Unlike the world RT pipeline, these passes read
// WorldPush by device address (the same indirection the sky LUT uses) and carry the scaled-scalar
// parameter set directly, because a compute pipeline has no per-stage WorldPushConstants push block on
// which to piggyback. Field order is chosen so every 16-byte float4 lands on a 16-byte boundary; the
// probe in layout_probe.slang reflects this and generates the matching Java record at build time.
public struct FroxelPush {
    public float4  params;  // x near (blocks), y far (blocks), z froxel tile size (px), w depth slice count
    public float4  look;    // x base density (per-block extinction), y height falloff, z phase g, w enabled (0/1)
    public float4  aux;     // x scatter fraction (albedo), y noise intensity, z noise scale, w temporal blend 0..1
    public float4  ref;     // x global scattering intensity multiplier, yzw linear tint (1 = neutral)
    public float4  sky;     // x ambient skylight weight (0 = direct sun/moon shafts only), y submerged density multiplier
    public uint64_t worldPushAddr;
    public uint64_t froxelVolumeAddr;   // current-frame froxel buffer, DevicePtr<float4>
    public uint64_t froxelHistoryAddr;  // previous-frame froxel buffer for temporal blend (0 = no history)
    public uint    packedRenderSize;    // low16 render width, high16 render height
    public uint    packedGridDims;      // low16 froxel tile width, high16 froxel tile height
};

public uint unpackRenderWidth(uint packed) { return packed & 0xffffu; }
public uint unpackRenderHeight(uint packed) { return packed >> 16; }
public uint unpackGridWidth(uint packed) { return packed & 0xffffu; }
public uint unpackGridHeight(uint packed) { return packed >> 16; }

// Exponential (in distance) slice band boundary, normalised to [0,1] over the fog far distance. A large
// far/near base puts many thin slices right at the camera and thick slices out at the sky, where the
// phase/lobe has already converged to a smooth gradient.
public float froxelWeight(float u, float far, float near_) {
    float bias = max(far / max(near_, 1.0e-3), 1.0001);
    return (pow(bias, u) - 1.0) / (bias - 1.0);
}

public float sliceNearDist(uint slice, uint slices, float far, float near_) {
    return froxelWeight(float(slice) / float(slices), far, near_) * far;
}

public float sliceFarDist(uint slice, uint slices, float far, float near_) {
    return froxelWeight((float(slice) + 1.0) / float(slices), far, near_) * far;
}

public float sliceMidDist(uint slice, uint slices, float far, float near_) {
    return 0.5 * (sliceNearDist(slice, slices, far, near_) + sliceFarDist(slice, slices, far, near_));
}

// Reconstruct the camera-space ray through a tile/pixel centre from the inverse view-projected near and
// far planes — the exact near/far split world.primary.rgen uses, so the grid is built on precisely the
// frustum the path tracer traces. nearP/farP are camera-relative; the caller adds camOffset for world.
public void frustumRay(float2 ndc, float4x4 invViewProj, out float3 nearP, out float3 farP, out float3 dir) {
    float4 nearH = mul(invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    nearP = nearH.xyz / nearH.w;
    farP = farH.xyz / farH.w;
    dir = normalize(farP - nearP);
}

// Cheap, spatially-stable 3D hash for the density noise and the occlusion jitter. It is a pure function
// of position (no frame index), so a froxel's value is deterministic across frames and the shadow
// averaging cannot shimmer.
public float hash3(float3 p) {
    p = frac(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return frac((p.x + p.y) * p.z);
}

// One octave of trilinear value noise; the fog field is smooth to begin with, so a single octave plus the
// hash gives per-region mist variation without aliasing on the coarse grid.
public float valueNoise3(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);
    float n000 = hash3(i);
    float n100 = hash3(i + float3(1, 0, 0));
    float n010 = hash3(i + float3(0, 1, 0));
    float n110 = hash3(i + float3(1, 1, 0));
    float n001 = hash3(i + float3(0, 0, 1));
    float n101 = hash3(i + float3(1, 0, 1));
    float n011 = hash3(i + float3(0, 1, 1));
    float n111 = hash3(i + float3(1, 1, 1));
    float x00 = lerp(n000, n100, u.x);
    float x10 = lerp(n010, n110, u.x);
    float x01 = lerp(n001, n101, u.x);
    float x11 = lerp(n011, n111, u.x);
    return lerp(lerp(x00, x10, u.y), lerp(x01, x11, u.y), u.z);
}

// Per-block fog extinction at a rebased world position. Base density falls off with altitude above the
// camera's own (rebased) height, so the fog hugs the ground and thins overhead; a gentle value-noise
// field adds per-region mist variation. The height delta is computed from rebased coordinates, whose
// difference equals the difference of the absolute Ys, so it is resolution/rebase-safe in the shader.
public float fogDensity(float3 worldPos, float cameraRebasedY, float baseDensity,
                        float heightFalloff, float noiseIntensity, float noiseScale) {
    float heightDelta = max(0.0, worldPos.y - cameraRebasedY);
    float density = baseDensity * exp(-heightDelta * heightFalloff);
    if (noiseIntensity > 0.0) {
        float n = valueNoise3(worldPos * noiseScale);
        density *= 1.0 + noiseIntensity * (n * 2.0 - 1.0);
    }
    return max(density, 0.0);
}

// Henyey-Greenstein phase, normalised to integrate to 1 over the sphere exactly like the sky's phase
// functions (isotropic = 1/4π, Cornette-Shanks = 1). Forward-scattering (g > 0) brightens the volume in
// the sun's direction, which is what makes visible light shafts.
public float hgPhase(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2)
            / (4.0 * 3.14159265359 * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1.0e-4), 1.5));
}
