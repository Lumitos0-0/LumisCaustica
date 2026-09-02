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
