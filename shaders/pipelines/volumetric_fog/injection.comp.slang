// Froxel density injection pass.
// Fills a 3D frustum-aligned froxel volume with fog density / extinction.
// No ray tracing here — density is procedural height fog + optional noise.
// The integration pass samples this volume and does hardware RT shadows per sample.

import world_common;
import volumetric_fog_common;

[[vk::push_constant]] FogInjectionPush pc;

[[vk::binding(0, 0)]] [format("rgba16f")] RWTexture3D<float4> froxelVolume;

static const float PI = 3.14159265359;

// PCG hash
uint pcgHash(uint v) {
    v = v * 747796405u + 2891336453u;
    uint w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}
float rnd(inout uint s) {
    return float(pcgHash(s) >> 8u) * (1.0 / 16777216.0);
}

// Interleaved Gradient Noise (Jimenez)
float ign(float2 p) {
    float3 magic = float3(0.06711056, 0.00583715, 52.9829189);
    return frac(magic.z * frac(dot(p, magic.xy)));
}

float3 primaryRayDirFroxel(float2 ndc) {
    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    return normalize(farP - nearP);
}

[shader("compute")]
[numthreads(8, 8, 4)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    int3 dim = pc.froxelDim;
    if (int(dispatchId.x) >= dim.x || int(dispatchId.y) >= dim.y || int(dispatchId.z) >= dim.z) return;

    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];

    // Stratified jitter inside froxel block.
    // Use frameIndex for temporal variation + spatial hash for decorrelation.
    uint seed = dispatchId.x * 1973u + dispatchId.y * 9277u + dispatchId.z * 26699u + pc.frameIndex * 2654435761u + pc.sampleSeed;
    float jitterX = (rnd(seed) - 0.5) * pc.jitterStrength;
    float jitterY = (rnd(seed) - 0.5) * pc.jitterStrength;
    float jitterZ = (rnd(seed) - 0.5) * pc.jitterStrength;

    // Temporal sub-pixel jitter from main camera jitter (DLSS) plus our own.
    float2 uv = (float2(dispatchId.xy) + float2(0.5, 0.5) + float2(jitterX, jitterY) + pc.jitterOffset) / float2(dim.xy);
    float2 ndc = uv * 2.0 - 1.0;

    float3 rayDir = primaryRayDirFroxel(ndc);

    // Exponential depth distribution: more slices near camera.
    float linear01 = (float(dispatchId.z) + 0.5 + jitterZ) / float(dim.z);
    // Use exponential: depth = near * (far/near)^(linear)
    float nearP = max(pc.nearPlane, 0.1);
    float farP = max(pc.farPlane, nearP + 1.0);
    float depth = nearP * pow(farP / nearP, linear01);
    depth = clamp(depth, nearP, farP);
    depth = min(depth, pc.maxDistance);

    // World position in rebased space
    float3 rebasedPos = worldPush.camOffset + rayDir * depth;
    // Absolute world position
    float3 worldPos = rebasedPos + pc.terrainOrigin;

    // Height fog: density falls off with height above sea level.
    // Minecraft sea level ~63, but we use camera-relative for simplicity.
    // Use world Y.
    float height = worldPos.y;
    // Base height fog: dense near ground, exponential falloff.
    float heightFactor = exp(-max(height - 64.0, 0.0) * pc.heightFalloff);
    // Also add a little extra density in valleys / lower areas, and reduce at very high altitude.
    float highAltitudeFade = saturate(1.0 - max(height - 128.0, 0.0) * 0.01);
    
    // Distance fade: reduce density at far distance to avoid washing out.
    float distanceFade = 1.0 / (1.0 + depth * 0.001);

    // Base density
    float baseDensity = pc.density * heightFactor * highAltitudeFade * distanceFade;

    // Add some low-frequency noise for variation (cheap hash noise)
    float3 noisePos = worldPos * 0.02;
    float n = frac(sin(dot(noisePos, float3(12.9898, 78.233, 45.164))) * 43758.5453);
    n = lerp(0.8, 1.2, n);
    baseDensity *= n;

    // Clamp
    baseDensity = max(baseDensity, 0.0);

    // Scattering and extinction stored in volume.
    // We'll store density in alpha, and scattering color (white) scaled by density in rgb for quick lookup.
    // For now, scattering rgb = density * scatteringCoeff (white fog). Could be tinted later.
    float scatteringCoeff = pc.scattering;
    float extinctionCoeff = pc.extinction;

    float3 scattering = float3(baseDensity * scatteringCoeff);
    float extinction = baseDensity * extinctionCoeff;

    // Store: rgb = scattering, a = extinction
    froxelVolume[dispatchId] = float4(scattering, extinction);
}
