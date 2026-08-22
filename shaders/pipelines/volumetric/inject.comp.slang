// Pass 1 — Density + lighting injection.
//
// For each froxel:
//   * compute its STABLE world-space center (deterministic; no jitter)
//   * evaluate extinction from world position (deterministic, world-anchored)
//   * evaluate HG phase for the view-to-light angle at the center
//   * sample ONE shadow ray; when stochasticLight is enabled the ray direction
//     is jittered across the sun disk using a low-discrepancy R2 sequence
//     (deterministic per-(froxel,frame)), otherwise it shoots straight at
//     the sun center.
//   * write RGBA = (inScatteredRGB, sigma_t_scalar) into fogCurrent.
//
// Design invariants:
//   - FROXEL IDENTITY is the stable center; stochastic sampling moves ONLY
//     the light-ray direction, never the point we consider to be the froxel.
//   - Visibility is binary (0 or 1); we never blur/pack it into extinction.
//     Temporal + spatial reconstruction integrates samples over time/space.
//   - Exactly 1 shadow ray per froxel per frame. The pipeline is architected
//     to permit raising this to N rays in the future (the trace loop is
//     factored to make that straightforward), but 1 is the baseline.

import bindings;
import common;

[[vk::push_constant]] VolumetricPush pc;

// Binary shadow test from origin toward normalized direction dir.
// FORCE_OPAQUE treats all geometry as opaque for shadows; ACCEPT_FIRST_HIT
// terminates on the first hit. No backface culling: terrain meshes are
// single-sided and a shadow ray can hit a backfacing triangle from behind.
float TraceShadow(float3 origin, float3 dir)
{
    RayDesc ray;
    ray.Origin    = origin;
    ray.TMin      = 0.05f;
    ray.Direction = dir;
    ray.TMax      = max(pc.farPlane - 0.1f, 0.1f);
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

    // --- Stable froxel identity: world center is DETERMINISTIC for (id, camera/frustum).
    float3 worldPos = FroxelCenterWorldSpace(pc, id);
    float sigmaLuma = ExtinctionAt(pc, worldPos);

    if (sigmaLuma < 1.0e-6f)
    {
        fogCurrent[id] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // View direction: from camera through the froxel center (away from camera).
    float3 viewPos  = FroxelCenterViewSpace(pc, id);
    float3 viewDir  = normalize(viewPos);

    float3 sunDirNorm = length(pc.sunDir) > 1.0e-4f
        ? normalize(pc.sunDir)
        : float3(0.0f, 1.0f, 0.0f);

    // Henyey-Greenstein phase at the stable center.
    float cosTheta = dot(viewDir, sunDirNorm);
    float phase = HenyeyGreenstein(cosTheta, pc.anisotropyG);

    // Light-ray sample. The R2 sequence distributes samples across the sun disk
    // with a Cranley-Patterson rotation so consecutive frames fill the disk
    // without repeating, while remaining deterministic for (froxel, frame).
    float3 rayDir;
    if (pc.stochasticLight != 0u && pc.sunAngularRadius > 1.0e-5f)
    {
        float2 xi = R2Offset2D(id, pc.frameIndex);
        rayDir = SampleSunDisk(sunDirNorm, pc.sunAngularRadius, xi);
    }
    else
    {
        rayDir = sunDirNorm;
    }

    float visibility = TraceShadow(worldPos, rayDir);

    // Source term S = sigma_s * phase * V * L_sun.
    // Transmittance uses sigma_t (luma); we store sigma_t in alpha for the integrator.
    float3 S = ScatteringAlbedo() * sigmaLuma * phase * visibility * pc.sunColor;

    fogCurrent[id] = float4(S, sigmaLuma);
}
