// Froxel integrate / fog composite pass: accumulate the volume's in-scatter along each view ray and
// blend it over the path-traced surface radiance.
//
// One thread == one pixel of the render-resolution scene image. The ray is reconstructed from the same
// inverse view-projected near/far planes the primary pass uses, and the surface distance comes from the
// guide depth (reversed-Z). Each slice of the froxel column contributes its precomputed in-scatter
// weighted by the surviving transmittance; the output is `fogInScatter + surfaceTransmittance * radiance`,
// which runs through DLSS-RR / denoise / exposure / bloom along with everything else.
//
// The volume is sampled TRILINEALLY — bilinearly across the four neighbouring tiles and linearly across
// the two neighbouring depth slices — never with a nearest fetch of the containing froxel. The grid is
// coarse by design (one froxel per 8x8 pixel tile per slice), so a nearest lookup would stamp visible
// pixel blocks into the fog; filtering the volume is what every shipping froxel implementation does
// (Frostbite and UE sample a 3D texture with hardware filtering; the storage buffer here gets the same
// filter by hand). The per-slice lookup is evaluated at the segment midpoint's continuous slice
// coordinate (froxel_common.sliceCoord), which also smooths the exponential staircase between slices.
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
    int tileW = int(max(unpackGridWidth(pc.packedGridDims), 1u));
    int tileH = int(max(unpackGridHeight(pc.packedGridDims), 1u));
    int sliceCount = int(slices);

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

    // Continuous tile coordinates, in the texture convention where tile centres sit on integers: tile i
    // spans [i, i+1) in tile units and its centre is i+0.5, so the sample position of pixel p is
    // (p + 0.5) / tileSize - 0.5. Bilinear weights come from the fractional part; the neighbouring tile
    // indices clamp at the grid edge (edge tiles of a partial final tile are already the right values).
    float2 tilePos = (float2(pix) + 0.5) / float(tileSize) - 0.5;
    int2 tileBase = int2(floor(tilePos));
    float2 tileFrac = tilePos - float2(tileBase);
    int tx0 = clamp(tileBase.x, 0, tileW - 1);
    int tx1 = clamp(tileBase.x + 1, 0, tileW - 1);
    int ty0 = clamp(tileBase.y, 0, tileH - 1);
    int ty1 = clamp(tileBase.y + 1, 0, tileH - 1);

    ConstPtr<float4> volume = ConstPtr<float4>(pc.froxelVolumeAddr);

    float3 scattered = float3(0.0, 0.0, 0.0);
    float transmittance = 1.0;
    for (uint k = 0u; k < slices; k++) {
        float lo = max(sliceNearDist(k, slices, farDist, nearDist), 0.0);
        float hi = min(sliceFarDist(k, slices, farDist, nearDist), hitDist);
        if (lo >= hi) {
            continue;
        }
        // Trilinear froxel lookup at the segment midpoint's continuous slice coordinate.
        float s = sliceCoord(0.5 * (lo + hi), slices, farDist, nearDist);
        int k0 = clamp(int(floor(s)), 0, sliceCount - 1);
        int k1 = min(k0 + 1, sliceCount - 1);
        float fz = clamp(s - float(k0), 0.0, 1.0);
        float4 sliceA = lerp(
                lerp(volume[(k0 * tileH + ty0) * tileW + tx0], volume[(k0 * tileH + ty0) * tileW + tx1], tileFrac.x),
                lerp(volume[(k0 * tileH + ty1) * tileW + tx0], volume[(k0 * tileH + ty1) * tileW + tx1], tileFrac.x),
                tileFrac.y);
        float4 sliceB = lerp(
                lerp(volume[(k1 * tileH + ty0) * tileW + tx0], volume[(k1 * tileH + ty0) * tileW + tx1], tileFrac.x),
                lerp(volume[(k1 * tileH + ty1) * tileW + tx0], volume[(k1 * tileH + ty1) * tileW + tx1], tileFrac.x),
                tileFrac.y);
        float4 frox = lerp(sliceA, sliceB, fz);

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
