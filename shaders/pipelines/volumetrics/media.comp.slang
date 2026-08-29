// Computes and injects participating medium properties (scattering and extinction) into the Froxel 3D grid.
import froxel_common;
import world_common;

[[vk::push_constant]] FroxelMediaPush pc;
[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture3D<float4> froxelMedia;

[shader("compute")]
[numthreads(8, 8, 4)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    uint3 gridDims = pc.gridDims.xyz;
    if (any(dispatchId >= gridDims)) {
        return;
    }

    float zNear = pc.params1.x;
    float zFar = pc.params1.y;
    float baseDensity = pc.params0.x;
    float heightFalloff = pc.params0.y;
    float groundAltitude = pc.params0.z;
    float maxDistance = pc.params0.w;

    // Linear depth of the center of this slice
    float sliceDepth = sliceToLinearDepth(float(dispatchId.z) + 0.5, zNear, zFar, float(gridDims.z));
    
    // Smooth boundary fade at the far distance to avoid harsh frustum edge outlines
    float distFade = smoothstep(maxDistance, maxDistance * 0.85, sliceDepth);
    if (distFade <= 0.0) {
        froxelMedia[dispatchId] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float2 uv = (float2(dispatchId.xy) + 0.5) / float2(gridDims.xy);
    float2 ndc = uv * 2.0 - 1.0;

    // Exact unprojected camera ray to calculate true world altitude Y
    float3 camOrigin, rayDir;
    froxelRay(ndc, pc.invViewProj, camOrigin, rayDir);
    float3 camRelPos = camOrigin + rayDir * sliceDepth;
    float worldY = camRelPos.y + pc.camPos.y;

    // Continuous atmosphere + cave medium: clear baseline medium plus smooth ground height modulation
    float baseHaze = baseDensity * 0.20;
    float heightScale = exp(-clamp((worldY - groundAltitude) * heightFalloff, -2.5, 4.0));
    float density = (baseHaze + baseDensity * 0.80 * heightScale) * distFade;

    // Natural atmosphere Rayleigh/Mie base scattering
    float3 fogColor = float3(pc.params1.z, pc.params1.w, pc.params2.x);
    if (dot(fogColor, fogColor) < 1.0e-5) {
        fogColor = float3(0.9, 0.95, 1.0); // Natural cool white / bluish haze default
    }

    float3 scattering = density * fogColor;
    float extinction = density;

    // Submerged / water volume detection
    bool cameraSubmerged = (pc.gridDims.w & 1u) != 0u;
    if (cameraSubmerged) {
        float waterDensity = pc.params2.y > 0.0 ? pc.params2.y : 0.06;
        float3 waterExtinctionFloor = float3(0.015, 0.010, 0.008);
        float3 waterTint = pc.waterParams.xyz;
        if (dot(waterTint, waterTint) < 1.0e-5) {
            waterTint = float3(0.2, 0.5, 0.9);
        }
        float3 waterExt = waterExtinctionFloor + waterDensity * (float3(1.0) - clamp(waterTint, 0.0, 1.0));
        
        scattering += waterDensity * waterTint * 0.5 * distFade;
        extinction += max(waterExt.r, max(waterExt.g, waterExt.b)) * distFade;
    }

    // Store in froxelMedia: RGB = scattering coefficient sigma_s, A = extinction coefficient sigma_t
    froxelMedia[dispatchId] = float4(scattering, extinction);
}
