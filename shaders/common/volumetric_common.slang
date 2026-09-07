// Shared volumetric constants, coordinate projections, and phase functions.
module volumetric_common;

import world_common;

// Froxel grid resolution: 160x90x64 covers 16:9 viewports with dense z-slices.
public static const int3 FROXEL_DIMS = int3(160, 90, 64);
public static const float FROXEL_NEAR = 0.5;
public static const float FROXEL_FAR = 192.0;

public struct VolumetricPush {
    public uint64_t worldPushAddr;
    public float fogDensity;        // Base density (default ~0.015)
    public float fogHeightFalloff;   // Height falloff scale in blocks (default 16.0)
    public float fogHeightBaseline;  // Ground height anchor (default 64.0)
    public float anisotropy;         // Henyey-Greenstein g (default 0.75 for forward god rays)
    public float temporalWeight;     // Temporal history weight (default 0.92)
    public float giStrength;         // Multiplier for sky/indirect volumetric light
    public float directStrength;     // Multiplier for direct sun/moon shafts
    public uint frameIndex;
};

// Exponential depth distribution: concentrates precision near the camera.
public float froxelSliceToDepth(float slice, int numSlices) {
    return FROXEL_NEAR * pow(FROXEL_FAR / FROXEL_NEAR, (slice + 0.5) / float(numSlices));
}

public float depthToFroxelSlice(float linearDepth, int numSlices) {
    float d = clamp(linearDepth, FROXEL_NEAR, FROXEL_FAR);
    return (log(d / FROXEL_NEAR) / log(FROXEL_FAR / FROXEL_NEAR)) * float(numSlices) - 0.5;
}

// Henyey-Greenstein phase function for forward-scattering atmospheric aerosols.
public float henyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159265359 * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1.0e-4), 1.5));
}
