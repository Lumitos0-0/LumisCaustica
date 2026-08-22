// Pass 1 — Density + lighting injection.
//
// For each froxel:
//   * compute its STABLE world-space center
//   * evaluate base extinction from world position (deterministic, world-anchored)
//   * sample directional sun light with one shadow ray
//   * write RGBA = (inScatteredRGB, sigma_t_scalar) into fogCurrent
//
// Invariant: jitter does NOT move the froxel center; only the light-ray sample
// direction is stochastic (and only when stochasticLight is enabled).

import bindings;
import common;

[[vk::push_constant]] VolumetricPush pc;

// Trace a shadow ray from origin toward dir (normalized). Uses FORCE_OPAQUE so all
// geometry is treated as opaque for shadow (water/translucent any-hits skipped),
// and ACCEPT_FIRST_HIT_AND_END_SEARCH so traversal terminates on first hit.
// No backface culling: terrain meshes are single-sided from the viewer's perspective
// but a shadow ray can hit a backfacing triangle from behind, and culling backfaces
// would leak light through walls/leaves.
float TraceShadow(float3 origin, float3 dir)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.TMin   = 0.05f;
    ray.Direction = dir;
    ray.TMax   = max(pc.farPlane - 0.1f, 0.1f);
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, 0xFF, 0xFF, ray);
    q.Proceed();
    return q.CommittedStatus() == COMMITTED_NOTHING ? 1.0f : 0.0f;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= FroxelSize(pc))) return;

    float3 worldPos = FroxelCenterWorldSpace(pc, id);
    float sigmaLuma = ExtinctionAt(pc, worldPos);

    // Early out: if density is negligible, store zero to speed up later passes.
    if (sigmaLuma < 1.0e-6f)
    {
        fogCurrent[id] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float3 sigma_t = SigmaTFromLuma(sigmaLuma);

    // View direction: from camera through the froxel (away from camera, into the scene).
    float3 viewDir = normalize(FroxelCenterViewSpace(pc, id));

    // Normalize sun direction defensively (CPU always pushes a unit vector, but a near-
    // zero direction would produce NaN phase/ray).
    float3 sunDirNorm = length(pc.sunDir) > 1.0e-4f
            ? normalize(pc.sunDir)
            : float3(0.0f, 1.0f, 0.0f);

    // cos(theta) between outgoing view ray and incoming light direction. HG phase
    // uses the cosine of the angle between view (out-scatter direction, away from
    // camera) and light (in-scatter direction, toward the sun).
    float cosTheta = dot(viewDir, sunDirNorm);
    float phase = HenyeyGreenstein(cosTheta, pc.anisotropyG);

    float2 xi = float2(0.5f, 0.5f);
    if (pc.stochasticLight != 0u)
    {
        xi = BlueNoise2(id, pc.frameIndex);
    }

    // Shadow ray direction is AWAY from the froxel toward the sun. SampleSunDisk
    // returns a unit vector on the sun cone around sunDir.
    float3 rayDir;
    if (pc.sunAngularRadius > 1.0e-5f && pc.stochasticLight != 0u)
        rayDir = SampleSunDisk(sunDirNorm, pc.sunAngularRadius, xi);
    else
        rayDir = sunDirNorm;

    float visibility = TraceShadow(worldPos, rayDir);

    float3 S = ScatteringSigma(sigma_t) * phase * visibility * pc.sunColor;

    fogCurrent[id] = float4(S, sigmaLuma);
}
