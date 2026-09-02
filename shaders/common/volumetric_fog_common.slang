// Push-constant structs for volumetric fog passes.
// Probed in layout_probe.slang so Java serializers are generated from Slang reflection.
// Layout is Std430: vec3 aligns to 16 bytes. Keep total size <= 256 bytes (Vulkan push constant limit).

public struct FogInjectionPush {
    public uint64_t worldPushAddr;   // 0
    public int3 froxelDim;           // 16 (aligned to 16)
    public float nearPlane;          // 28
    public float farPlane;           // 32
    public float density;            // 36
    public float anisotropy;         // 40
    public float scattering;         // 44
    public float extinction;         // 48
    public float heightFalloff;      // 52
    public float sunIntensity;       // 56
    public float moonIntensity;      // 60
    public float jitterStrength;     // 64
    public float maxDistance;        // 68
    public uint frameIndex;          // 72
    public uint sampleSeed;          // 76
    public int temporalEnabled;      // 80
    public int colorTransmissionEnabled; // 84
    public float2 jitterOffset;      // 88 (8-byte align)
    public float3 terrainOrigin;     // 96 (16-byte align)
    public float3 camWorldPos;       // 112 (16-byte align)
    // total 124 -> padded to 128
};

public struct FogIntegrationPush {
    public uint64_t worldPushAddr;   // 0
    public uint64_t tableAddr;       // 8
    public uint64_t entityTableAddr; // 16
    public uint64_t materialTableAddr; // 24
    public int2 displaySize;         // 32
    public int2 froxelDim;           // 40 (xy of froxel)
    public int froxelDepth;          // 48
    public float nearPlane;          // 52
    public float farPlane;           // 56
    public float density;            // 60
    public float anisotropy;         // 64
    public float scattering;         // 68
    public float extinction;         // 72
    public float heightFalloff;      // 76
    public float sunIntensity;       // 80
    public float moonIntensity;      // 84
    public float jitterStrength;     // 88
    public float maxDistance;        // 92
    public uint frameIndex;          // 96
    public int sampleCount;          // 100
    public int temporalEnabled;      // 104
    public int colorTransmissionEnabled; // 108
    public float exposure;           // 112
    public float2 jitterOffset;      // 120 (8-byte align, after padding)
    public float2 invDisplaySize;    // 128
    public float3 terrainOrigin;     // 144 (16-byte align)
    public float3 camWorldPos;       // 160
    public float3 sunDir;            // 176
    public float3 sunIlluminance;    // 192
    public float3 moonDir;           // 208
    public float3 moonIlluminance;   // 224
    // total ~236 -> padded to 240
};

// Pure logarithmic froxel depth gave excellent near-camera precision, but with the new cheaper lighting
// architecture it also made distant shafts break into very thick depth slabs. A mild power curve keeps the
// frustum-aligned volume front-loaded without exploding far-slice thickness the way a strict exponential does.
static const float FROXEL_DEPTH_CURVE = 1.5;

public float froxelDepthFromUnit(float unit01, float nearPlane, float farPlane) {
    unit01 = saturate(unit01);
    float span = max(farPlane - nearPlane, 1.0e-4);
    return nearPlane + span * pow(unit01, FROXEL_DEPTH_CURVE);
}

public float froxelUnitFromDepth(float depth, float nearPlane, float farPlane) {
    float span = max(farPlane - nearPlane, 1.0e-4);
    float linear01 = saturate((depth - nearPlane) / span);
    return pow(linear01, 1.0 / FROXEL_DEPTH_CURVE);
}

// Small tiled progressive blue-noise-style sequence shared by fog passes. The per-frame golden-ratio
// Cranley-Patterson rotation makes it spatiotemporal: each pixel/froxel keeps the same good spatial
// distribution every frame, but the sample advances through the unit interval over time instead of
// freezing a visible lattice into the volume.
static const float FOG_BLUE_NOISE_8X8[64] = {
    0.281250, 0.968750, 0.375000, 0.859375, 0.250000, 0.937500, 0.406250, 0.812500,
    0.593750, 0.187500, 0.734375, 0.046875, 0.578125, 0.218750, 0.703125, 0.015625,
    0.453125, 0.765625, 0.328125, 0.921875, 0.484375, 0.796875, 0.343750, 0.890625,
    0.640625, 0.093750, 0.546875, 0.156250, 0.671875, 0.078125, 0.515625, 0.125000,
    0.265625, 0.953125, 0.421875, 0.828125, 0.296875, 0.984375, 0.390625, 0.843750,
    0.562500, 0.234375, 0.687500, 0.000000, 0.609375, 0.203125, 0.718750, 0.031250,
    0.468750, 0.781250, 0.359375, 0.875000, 0.437500, 0.750000, 0.312500, 0.906250,
    0.656250, 0.062500, 0.500000, 0.140625, 0.625000, 0.109375, 0.531250, 0.171875
};

public float fogBlueNoiseScalar(uint2 pixel, uint frame, uint salt) {
    uint2 p = (pixel + uint2(salt * 3u, salt * 5u)) & uint2(7u, 7u);
    float base = FOG_BLUE_NOISE_8X8[int(p.y * 8u + p.x)];
    return frac(base + float((frame + salt * 17u) & 255u) * 0.61803398875);
}

public float fogBlueNoiseScalar3D(uint3 cell, uint frame, uint salt) {
    uint2 folded = cell.xy + uint2(cell.z * 13u + salt * 7u, cell.z * 17u + salt * 11u);
    return fogBlueNoiseScalar(folded, frame, salt);
}

public float2 fogBlueNoiseOffset2D(uint3 cell, uint frame, uint salt) {
    float x = fogBlueNoiseScalar3D(cell, frame, salt * 2u + 0u);
    float y = fogBlueNoiseScalar3D(cell.yzx, frame, salt * 2u + 1u);
    return float2(x, y) - 0.5;
}
