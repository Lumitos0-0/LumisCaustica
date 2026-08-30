// Froxel raymarch integration. For each render pixel, walk the camera ray through the frustum-aligned
// froxel grid (froxel.rgen populated it), accumulating each cell's stored in-scattered source with
// Beer-Lambert transmittance, and composite the resulting fog over the path-traced scene colour — which
// it also attenuates by the fog's transmittance to the surface.
//
// The grid is frustum-aligned: its XY are screen UVs (so one pixel stays in one grid column) and its Z
// is an exponential depth parameterisation from the near plane to the froxel far distance, matching the
// cell positions the light pass used. Each cell holds RGB in-scattered source per unit length and A =
// extinction density. The analytic exp(-sigma*ds) segment integral removes stepping bias that a naive
// source*ds accumulation would introduce.
//
// This pass intentionally imports only world_common (no world RT bindings): it is a plain compute pass
// with its own three-bind descriptor set and a PushAddr dereference to the same per-frame WorldPush slot
// the RT stages read.
import world_common;
import froxel_common;

[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture2D<float4> outImage;
[[vk::binding(1, 0)]] public Texture2D<float> depthImage;
[[vk::binding(2, 0)]] public Sampler3D froxelVolume;

[[vk::push_constant]] PushAddr pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    WorldPush push = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint w, h;
    outImage.GetDimensions(w, h);
    int2 pix = int2(tid.xy);
    if (pix.x >= int(w) || pix.y >= int(h)) {
        return;
    }
    float2 uv = (float2(pix) + 0.5) / float2(w, h);
    float2 ndc = uv * 2.0 - 1.0;

    // Reconstruct the camera ray the same way primary.rgen does (unjittered, pixel-centre NDC).
    float4 nearH = mul(push.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(push.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    float3 rd = normalize(farP - nearP);
    float t0 = length(nearP);

    float near = max(push.froxel0.x, 1.0e-4);
    float far = max(push.froxel0.y, near + 1.0);
    float logRatio = log(far / near);

    // Surface distance from the reversed-Z hardware depth guide (near=1, far=0); unprojecting it gives
    // the surface's camera-relative position, whose length is the distance along this ray to geometry.
    // A sky pixel (depth ~ 0) has no surface, so let the fog span the whole grid and fade the far view;
    // otherwise stop the march at the first surface so fog never draws through an occluder.
    float ndcZ = depthImage.Load(int3(pix, 0)).x;
    float tSurface;
    if (ndcZ < 1.0e-3) {
        tSurface = far;
    } else {
        float4 surfH = mul(push.invViewProj, float4(ndc.x, ndc.y, ndcZ, 1.0));
        tSurface = clamp(length(surfH.xyz / surfH.w), t0, far);
    }

    float T = 1.0;
    float3 fog = float3(0.0, 0.0, 0.0);
    for (uint z = 0u; z < FROXEL_DEPTH; ++z) {
        float zf0 = float(z) / float(FROXEL_DEPTH);
        float zf1 = float(z + 1u) / float(FROXEL_DEPTH);
        float tz0 = near * exp(zf0 * logRatio);
        float tz1 = near * exp(zf1 * logRatio);
        if (tz1 <= t0) {
            continue;
        }
        if (tz0 >= tSurface) {
            break;
        }
        float tA = max(tz0, t0);
        float tB = min(tz1, tSurface);
        float ds = tB - tA;
        if (ds <= 0.0) {
            continue;
        }
        float4 cell = froxelVolume.SampleLevel(
                float3(uv, (float(z) + 0.5) / float(FROXEL_DEPTH)), 0.0);
        float3 source = cell.rgb;
        float sigma = max(cell.a, 0.0);
        float segT = exp(-sigma * ds);
        float3 integ = sigma > 1.0e-6 ? source * ((1.0 - segT) / sigma) : source * ds;
        fog += T * integ;
        T *= segT;
        if (T < 1.0e-4) {
            break;
        }
    }

    // The scene colour is already pre-exposed (indirect.rgen's store); the fog must ride the same seam,
    // so it is scaled by the same preExposure factor the display pass divides back out.
    float3 sceneCol = outImage[pix].rgb;
    outImage[pix] = float4(sceneCol * T + fog * push.preExposure, 1.0);
}
