// Volumetric fog integration pass.
// Consumes a pre-lit froxel volume where rgb already stores directional-light single-scattering
// contribution per unit length and a stores extinction. This pass now only integrates along the view ray.

import world_common;
import volumetric_fog_common;

[[vk::push_constant]] FogIntegrationPush pc;

// Bindings - set 0
[[vk::binding(0, 0)]] Sampler3D froxelVolume; // prelit single-scattering rgb, extinction a
[[vk::binding(1, 0)]] Sampler2D fogStopDepth;
[[vk::binding(2, 0)]] RWTexture2D<float4> sceneImage;

static const float PI = 3.14159265359;
static const float MIN_STEP = 0.1;

float henyeyGreenstein(float cosTheta, float g) {
    g = clamp(g, 0.0, 0.8);
    float gg = g * g;
    float denom = 1.0 + gg - 2.0 * g * cosTheta;
    return (1.0 - gg) / (4.0 * PI * pow(max(denom, 1.0e-4), 1.5));
}

float3 activeLightDir() {
    float3 sunIllum = pc.sunIlluminance * pc.sunIntensity;
    float3 moonIllum = pc.moonIlluminance * pc.moonIntensity;
    bool sunEnabled = max(sunIllum.r, max(sunIllum.g, sunIllum.b)) > 1.0e-6;
    bool moonEnabled = max(moonIllum.r, max(moonIllum.g, moonIllum.b)) > 1.0e-6;
    if (sunEnabled) return normalize(pc.sunDir);
    if (moonEnabled) return normalize(pc.moonDir);
    return float3(0.0, 1.0, 0.0);
}

float3 primaryRayDir(float2 ndc, WorldPush worldPush) {
    float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    return normalize(farP - nearP);
}

float3 reconstructRebasedPos(float2 ndc, float ndcDepth, WorldPush worldPush) {
    float4 clip = float4(ndc.x, ndc.y, ndcDepth, 1.0);
    float4 view = mul(worldPush.invViewProj, clip);
    view /= view.w;
    return view.xyz + worldPush.camOffset;
}

float linearDepthFromRebased(float3 rebasedPos, WorldPush worldPush) {
    float3 camRel = rebasedPos - worldPush.camOffset;
    return length(camRel);
}

static const float SKY_DEPTH_EPSILON = 1.0e-6;

float depthToMarchDistance(float2 ndc, float depth, WorldPush worldPush) {
    if (depth <= SKY_DEPTH_EPSILON) {
        return pc.maxDistance;
    }
    float3 rebasedHitPos = reconstructRebasedPos(ndc, depth, worldPush);
    float tMax = linearDepthFromRebased(rebasedHitPos, worldPush);
    if (!(tMax > 0.1 && tMax <= 10000.0)) {
        return pc.maxDistance;
    }
    return min(tMax, pc.maxDistance);
}

float sampleDepthBilinear(float2 uv) {
    float2 pix = uv * pc.displaySize - 0.5;
    int2 base = int2(floor(pix));
    float2 f = frac(pix);

    int2 p00 = clamp(base + int2(0, 0), int2(0, 0), int2(pc.displaySize) - 1);
    int2 p10 = clamp(base + int2(1, 0), int2(0, 0), int2(pc.displaySize) - 1);
    int2 p01 = clamp(base + int2(0, 1), int2(0, 0), int2(pc.displaySize) - 1);
    int2 p11 = clamp(base + int2(1, 1), int2(0, 0), int2(pc.displaySize) - 1);

    float d00 = fogStopDepth[p00].x;
    float d10 = fogStopDepth[p10].x;
    float d01 = fogStopDepth[p01].x;
    float d11 = fogStopDepth[p11].x;

    float dx0 = lerp(d00, d10, f.x);
    float dx1 = lerp(d01, d11, f.x);
    return lerp(dx0, dx1, f.y);
}

float3 integrateFog(float3 rayDir, float maxDistance, float2 uv, out float viewTransmittance) {
    int3 dim = int3(pc.froxelDim.x, pc.froxelDim.y, pc.froxelDepth);
    float3 accum = 0.0;
    float transmittance = 1.0;

    float viewCos = dot(-rayDir, activeLightDir());
    float phase = henyeyGreenstein(viewCos, pc.anisotropy);

    float prevDepth = 0.0;

    for (int z = 0; z < dim.z; ++z) {
        float z0 = float(z) / float(dim.z);
        float z1 = float(z + 1) / float(dim.z);

        float depth0 = pc.nearPlane * pow(pc.farPlane / pc.nearPlane, z0);
        float depth1 = pc.nearPlane * pow(pc.farPlane / pc.nearPlane, z1);
        depth0 = min(depth0, pc.maxDistance);
        depth1 = min(depth1, pc.maxDistance);

        if (depth0 >= maxDistance) break;

        float segmentNear = max(prevDepth, depth0);
        float segmentFar = min(depth1, maxDistance);
        float stepLen = segmentFar - segmentNear;
        if (stepLen <= MIN_STEP) {
            prevDepth = depth1;
            continue;
        }

        int3 coord = int3(
                clamp(int(uv.x * dim.x), 0, dim.x - 1),
                clamp(int(uv.y * dim.y), 0, dim.y - 1),
                z);
        float3 froxelUv = (float3(coord) + 0.5) / float3(dim);
        float4 froxel = froxelVolume.SampleLevel(froxelUv, 0.0);
        float extinction = max(froxel.a, 0.0);
        if (extinction <= 1.0e-6) {
            prevDepth = depth1;
            continue;
        }

        float3 litScatterPerMeter = max(froxel.rgb, float3(0.0));
        float tau = extinction * stepLen;
        float segmentTrans = exp(-tau);
        float weight = (extinction > 1.0e-6) ? (1.0 - segmentTrans) / extinction : stepLen;
        accum += transmittance * litScatterPerMeter * phase * weight;
        transmittance *= segmentTrans;

        if (transmittance < 1.0e-3) break;
        prevDepth = depth1;
    }

    viewTransmittance = transmittance;
    return accum;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    int2 pixel = int2(tid.xy);
    if (pixel.x >= int(pc.displaySize.x) || pixel.y >= int(pc.displaySize.y)) return;

    float2 uv = (float2(pixel) + 0.5) * pc.invDisplaySize;
    float2 ndc = uv * 2.0 - 1.0;
    float sceneDepth = sampleDepthBilinear(uv);
    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];

    float3 rayDir = primaryRayDir(ndc, worldPush);
    float maxDist = depthToMarchDistance(ndc, sceneDepth, worldPush);

    float viewTrans = 1.0;
    float3 fog = integrateFog(rayDir, maxDist, uv, viewTrans);
    float3 scene = max(sceneImage[pixel].rgb, 0.0);

    sceneImage[pixel] = float4(scene * viewTrans + fog, 1.0);
}
