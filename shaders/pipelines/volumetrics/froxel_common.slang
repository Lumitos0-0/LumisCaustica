// Shared data structures and optical physics functions for Froxel-Frustum Volumetric Lighting.
module froxel_common;

import world_common;

public static const float VOLUMETRICS_PI = 3.14159265358979323846;
public static const float VOLUMETRICS_INV_4PI = 0.07957747154594767;

// Instance cull mask: bit 0 (0x01) for secondary shadow/volumetric rays, 0xFF for all opaque geometry
public static const uint CULL_SECONDARY = 0x01u;
public static const uint CULL_PRIMARY = 0x02u;
public static const uint CULL_ALL = 0xFFu;

// Structs for push constants across all volumetric passes
public struct FroxelMediaPush {
    public uint4 gridDims;        // x=Nx, y=Ny, z=Nz, w=flags (bit0=submerged, bit1=hasWater)
    public float4 params0;        // x=baseDensity, y=heightFalloff, z=groundAltitude, w=maxDistance
    public float4 params1;        // x=zNear, y=zFar, z=fogColorR, w=fogColorG
    public float4 params2;        // x=fogColorB, y=waterDensity, z=waterExtinctionFloor, w=reserved
    public float4 waterParams;    // camera biome water params (ACEScg RGB + wave time)
    public float4x4 invViewProj;   // camera inverse VP to reconstruct true world altitude Y
    public float3 camPos;         // camera world position
    public float  camY;
};

public struct FroxelLightPush {
    public uint4 gridDims;             // x=Nx, y=Ny, z=Nz, w=frameIndex
    public float4 params0;             // x=zNear, y=zFar, z=maxDistance, w=preExposure
    public float4 params1;             // x=anisotropyG1 (forward), y=anisotropyG2 (backward), z=lobeWeight, w=sunShaftMultiplier
    public float4 sunDirIllum;         // xyz = sun/moon light direction, w = illuminance (lux)
    public float4 lightGridOrigin;     // xyz = light grid cell origin in rebased blocks, w = cell size
    public int4   lightGridDims;       // xyz = light grid dimensions, w = enableBlockLights
    public float4 lightRebase;         // xyz = light buffer rebase offset in world blocks, w = invGlobalPowerSum
    public float4x4 invViewProj;       // camera inverse view-projection (jitter-free)
    public float3 camOffset;           // terrain rebase offset
    public uint   lightCount;
    public float3 camPos;
    public float  sunAngularRadius;
    public uint64_t lightBufAddr;        // RIS global light buffer address
    public uint64_t lightGridCellAddr;   // Dense light grid cell headers
    public uint64_t lightGridSpanAddr;   // Packed section spans
    public uint64_t lightAliasAddr;      // Global power alias buffer address
    public uint64_t lightLocalAliasAddr; // Local section alias buffer address
    public uint   enableShadows;
    public uint   risCandidates;
};

public struct FroxelTemporalPush {
    public uint4 gridDims;        // x=Nx, y=Ny, z=Nz, w=resetHistory (1=reset, 0=accumulate)
    public float4 params0;        // x=temporalWeight (e.g. 0.90), y=varianceGamma (e.g. 1.5), z=zNear, w=zFar
    public float4 params1;        // x=maxDistance, yzw reserved
    public float4x4 invViewProj;  // current inverse VP
    public float4x4 prevViewProj; // previous forward VP
    public float3 camDelta;       // camera motion delta since last frame
    public float  spatialBilateralWeight;
};

public struct FroxelIntegratePush {
    public uint4 gridDims;        // x=Nx, y=Ny, z=Nz, w=flags
    public float4 params0;        // x=zNear, y=zFar, z=maxDistance, w=sliceScale
};

public struct FroxelCompositePush {
    public uint4 screenAndDepth;  // x=displayWidth, y=displayHeight, z=renderWidth, w=renderHeight
    public uint4 gridDims;        // x=gridNx, y=gridNy, z=gridNz, w=flags
    public float4 params0;        // x=zNear, y=zFar, z=maxDistance, w=sliceScale
    public float4 params1;        // x=preExposure, y=fogDensityMult, z=sunShaftMult, w=enableBilateral
    public float4x4 invViewProj;
    public float3 camOffset;
    public float  padding;
};

// ================================================================================================
// Depth slice exponential parameterization
// ================================================================================================

public float linearDepthToSlice(float z, float zNear, float zFar, float numSlices) {
    float nZ = max(zNear, 0.75);
    float fZ = max(zFar, nZ + 1.0);
    float clampedZ = clamp(z, nZ, fZ);
    return numSlices * (log(clampedZ / nZ) / log(fZ / nZ));
}

public float sliceToLinearDepth(float slice, float zNear, float zFar, float numSlices) {
    float normSlice = clamp(slice / numSlices, 0.0, 1.0);
    float nZ = max(zNear, 0.75);
    float fZ = max(zFar, nZ + 1.0);
    return nZ * pow(fZ / nZ, normSlice);
}

public float computeSliceThickness(uint sliceIdx, float zNear, float zFar, float numSlices) {
    float z0 = sliceToLinearDepth(float(sliceIdx), zNear, zFar, numSlices);
    float z1 = sliceToLinearDepth(float(sliceIdx + 1u), zNear, zFar, numSlices);
    return max(z1 - z0, 1.0e-4);
}

// Low-discrepancy 3D PCG integer hash (eliminates diagonal banding and moiré grating patterns)
public float froxelJitter(uint3 coord, uint frameIndex) {
    uint v = coord.x * 1664525u + coord.y * 1013904223u + coord.z * 22695477u + frameIndex * 2654435761u;
    v ^= v >> 16u;
    v *= 0x45d9f3bu;
    v ^= v >> 16u;
    return float(v & 0x00FFFFFFu) * (1.0 / 16777216.0);
}

// Reconstruct unprojected camera ray direction from inverted projection (Vulkan reverse-Z)
public void froxelRay(float2 ndc, float4x4 invViewProj, out float3 dir) {
    float4 farH = mul(invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float4 nearH = mul(invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float3 farP = farH.xyz / farH.w;
    float3 nearP = nearH.xyz / nearH.w;
    dir = normalize(farP - nearP);
}

// Reconstruct camera-relative world position for a froxel coordinate
public float3 froxelToCamRelative(float3 froxelId, float3 gridDims, float4x4 invViewProj,
                                 float zNear, float zFar, float jitterZ) {
    float2 uv = (froxelId.xy + 0.5) / gridDims.xy;
    float2 ndc = uv * 2.0 - 1.0;
    float linearZ = sliceToLinearDepth(froxelId.z + jitterZ, zNear, zFar, gridDims.z);
    
    float3 dir;
    froxelRay(ndc, invViewProj, dir);
    return dir * linearZ;
}

// Convert camera-relative position back into continuous (u, v, slice) froxel space
public float3 camRelativeToFroxel(float3 camRelPos, float4x4 curViewProj,
                                 float3 gridDims, float zNear, float zFar) {
    float4 clip = mul(curViewProj, float4(camRelPos, 1.0));
    if (clip.w <= 0.0) {
        return float3(-1.0, -1.0, -1.0);
    }
    float2 ndc = clip.xy / clip.w;
    float2 uv = ndc * 0.5 + 0.5;
    float linearZ = length(camRelPos);
    float slice = linearDepthToSlice(linearZ, zNear, zFar, gridDims.z);
    
    return float3(uv, slice / gridDims.z);
}

// ================================================================================================
// Dual-Lobe Henyey-Greenstein & Rayleigh Phase Functions
// ================================================================================================

// Standard Henyey-Greenstein single lobe
public float henyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    float denom = max(1.0 + g2 - 2.0 * g * cosTheta, 1.0e-5);
    return VOLUMETRICS_INV_4PI * ((1.0 - g2) / (denom * sqrt(denom)));
}

// Dual-Lobe Henyey-Greenstein Phase Function
// cosTheta = dot(rayDir, lightDir) where rayDir points away from camera towards scene,
// and lightDir points towards the light source.
// g1 > 0: Forward scattering (creates sharp, bright god-ray light shafts towards light)
// g2 < 0: Backscattering (creates subtle halo glow when looking away from light)
// w: Weighting between forward and backward lobes (e.g. 0.85 forward, 0.15 backward)
public float dualLobePhase(float cosTheta, float g1, float g2, float w) {
    float forwardLobe = henyeyGreenstein(cosTheta, g1);
    float backLobe = henyeyGreenstein(cosTheta, g2);
    return lerp(backLobe, forwardLobe, clamp(w, 0.0, 1.0));
}

// Rayleigh phase function for molecular atmosphere
public float rayleighPhaseVolumetric(float cosTheta) {
    return (3.0 / (16.0 * VOLUMETRICS_PI)) * (1.0 + cosTheta * cosTheta);
}

// ================================================================================================
// Anti-Boiling & Singularity Regularization for Point & Block Emitters
// ================================================================================================

// Smooth distance regularization: prevents 1/r^2 singularity fireflies when froxel hits emitter center
public float regularizedInverseDistanceSquared(float distSq, float emitterRadius) {
    float r0Sq = max(emitterRadius * emitterRadius, 0.01);
    return 1.0 / (distSq + r0Sq);
}

// Voxel maximum energy capacity bound: caps light influx to what a finite volume element can physically sustain
public float3 clampVoxelMaxRadiance(float3 radiance, float voxelDiameter) {
    float maxEnergyCapacity = 25000.0 / max(voxelDiameter, 0.1);
    return min(radiance, float3(maxEnergyCapacity));
}

// ================================================================================================
// Color Space Conversions (YCoCg for 3D AABB Variance Clamping)
// ================================================================================================

public float3 rgbToYCoCg(float3 c) {
    float Y  = dot(c, float3( 0.25, 0.50,  0.25));
    float Co = dot(c, float3( 0.50, 0.00, -0.50));
    float Cg = dot(c, float3(-0.25, 0.50, -0.25));
    return float3(Y, Co, Cg);
}

public float3 ycocgToRgb(float3 ycocg) {
    float Y  = ycocg.x;
    float Co = ycocg.y;
    float Cg = ycocg.z;
    float R = Y + Co - Cg;
    float G = Y + Cg;
    float B = Y - Co - Cg;
    return max(float3(R, G, B), float3(0.0));
}
