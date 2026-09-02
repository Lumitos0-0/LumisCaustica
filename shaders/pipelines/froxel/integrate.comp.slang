// Froxel integrate / fog composite pass: accumulate the volume's in-scatter along each view ray and
// blend it over the path-traced surface radiance.
//
// One thread == one pixel of the render-resolution scene image. The ray is reconstructed from the same
// inverse view-projected near/far planes the primary pass uses, and the surface distance comes from the
// guide depth (reversed-Z). Each slice of the froxel column contributes its precomputed in-scatter
// weighted by the surviving transmittance; the output is `fogInScatter + surfaceTransmittance * radiance`,
// which runs through DLSS-RR / denoise / exposure / bloom along with everything else.
import froxel_common;
import world_common;

[[vk::push_constant]] FroxelPush pc;

[[vk::binding(0, 0)]] [format("rgba16f")] RWTexture2D<float4> sourceColor;
[[vk::binding(1, 0)]] [format("r32f")] RWTexture2D<float> depthImage;
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture2D<float4> outFog;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    int2 pix = int2(tid.xy);
    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];

    // Master switch: when the feature is disabled the pass is a 1:1 copy, so the frame graph always has
    // a valid fogged image and toggling the setting needs no pipeline/image rebuild.
    if (pc.look.w < 0.5) {
        outFog[pix] = sourceColor[pix];
        return;
    }

    uint renderW = max(unpackRenderWidth(pc.packedRenderSize), 1u);
    uint renderH = max(unpackRenderHeight(pc.packedRenderSize), 1u);
    uint tileSize = max(uint(pc.params.z), 1u);
    uint slices = max(uint(pc.params.w), 1u);
    float nearDist = pc.params.x;
    float farDist = pc.params.y;
    uint tileW = max(unpackGridWidth(pc.packedGridDims), 1u);
    uint tileH = max(unpackGridHeight(pc.packedGridDims), 1u);

    // Pixel-centre ray (no jitter — the fog is deterministic, so no sampling noise to cancel).
    float2 size = float2(float(renderW), float(renderH));
    float2 uv = (float2(pix) + 0.5) / size;
    float2 ndc = uv * 2.0 - 1.0;
    float3 nearP, farP, rd;
    frustumRay(ndc, worldPush.invViewProj, nearP, farP, rd);

    // Surface distance from the reversed-Z guide depth. The inverse projection with (ndc.xy, depth, 1)
    // and a homogeneous divide is exactly the point the near/far split reconstructs, so it is resolution-
    // safe and needs no knowledge of the projection's own near/far.
    float depthVal = depthImage[pix];
    float4 hitH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, depthVal, 1.0));
    float hitDist = hitH.w > 0.0 ? dot(hitH.xyz / hitH.w - nearP, rd) : farDist;

    uint tileX = min(uint((uint(pix.x) / tileSize)), tileW - 1u);
    uint tileY = min(uint((uint(pix.y) / tileSize)), tileH - 1u);

    float3 scattered = float3(0.0, 0.0, 0.0);
    float transmittance = 1.0;
    ConstPtr<float4> volume = ConstPtr<float4>(pc.froxelVolumeAddr);
    for (uint k = 0u; k < slices; k++) {
        float lo = max(sliceNearDist(k, slices, farDist, nearDist), 0.0);
        float hi = min(sliceFarDist(k, slices, farDist, nearDist), hitDist);
        if (lo >= hi) {
            continue;
        }
        uint lin = (k * tileH + tileY) * tileW + tileX;
        float4 frox = volume[lin];
        float sigmaT = frox.w;
        float3 inScatter = frox.rgb;
        float dt = hi - lo;
        float segT = sigmaT > 1.0e-6 ? exp(-sigmaT * dt) : 1.0;
        // Exact integration of constant scattering over the slice: (1-exp(-sigma * dt)) / sigma.
        float c = sigmaT > 1.0e-6 ? (1.0 - segT) / sigmaT : dt;
        scattered += transmittance * inScatter * c;
        transmittance *= segT;
    }

    float3 fogged = scattered + transmittance * sourceColor[pix].rgb;
    outFog[pix] = float4(fogged, 1.0);
}
