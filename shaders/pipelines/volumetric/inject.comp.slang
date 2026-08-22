// Pass 1 — participating-medium coefficient + lighting injection.
//
// For each froxel:
//   * compute its stable world-space center (deterministic; no XY jitter),
//   * evaluate an explicit medium sample in world space (air or water),
//   * write physically meaningful coefficients:
//       fogMediumScattering.rgb = sigma_s (1/blocks)
//       fogMediumScattering.a   = density (approximately 1/block)
//       fogMediumAbsorption.rgb = sigma_a (1/blocks)
//       fogMediumAbsorption.a   = direct-light visibility for debug,
//   * evaluate direct celestial lighting from a directional source independent of the visible sun/moon
//     sprite, with one temporally coherent low-discrepancy sample across the light disc,
//   * evaluate stable ambient sky illumination using a fixed set of TLAS visibility probes,
//   * write fogCurrent.rgb = source radiance S = sigma_s * incidentRadiance * phase and
//     fogCurrent.a = luminance(sigma_t) for temporal validation.
//
// The stored source term is later integrated exactly over each froxel interval by integrate.comp:
//   radiance += T_before * S * (1 - exp(-sigma_t * ds)) / sigma_t.

import bindings;
import common;

[[vk::push_constant]] VolumetricPush pc;

// Binary shadow/visibility test from origin toward normalized direction dir.
// This intentionally uses the renderer TLAS in rebased-world space: neither the occluder nor the
// celestial source needs to be visible on screen to affect volumetric lighting. The instance mask matches
// secondary rays so particles and first-person-only geometry do not cast unstable shafts. Terrain water
// and translucent geometry are not treated as opaque shadow casters; solid and cutout buckets are.
float TraceVisibility(float3 origin, float3 dir, float tMax)
{
    RayDesc ray;
    ray.Origin    = origin;
    ray.TMin      = 0.05f;
    ray.Direction = dir;
    ray.TMax      = max(tMax, 0.1f);

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0x01, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            uint geometryIndex = q.CandidateGeometryIndex();
            // Terrain geometry indices are fixed by RtAccel: 2=translucent, 3=water.
            // Without material any-hit in inline queries, pass these through rather than making
            // glass/water fully opaque to volumetric light.
            if (geometryIndex == 2u || geometryIndex == 3u)
                continue;
            q.CommitNonOpaqueTriangleHit();
        }
    }
    return q.CommittedStatus() == COMMITTED_NOTHING ? 1.0f : 0.0f;
}

float DirectLightVisibility(float3 worldPos, float3 sampleToLightDir)
{
    return TraceVisibility(worldPos, sampleToLightDir, max(pc.farPlane - 0.1f, 0.1f));
}

float SkyProbeVisibility(float3 worldPos, float3 sampleToSkyDir)
{
    FroxelMatrices f = GetMatrices(pc);
    return TraceVisibility(worldPos, sampleToSkyDir, max(f.airParams.z, 1.0f));
}

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= FroxelSize(pc))) return;

    // Stable froxel identity: world center is deterministic for (id, camera/frustum).
    float3 worldPos = FroxelCenterWorldSpace(pc, id);
    FroxelMedium medium = EvaluateMedium(pc, worldPos);
    float sigmaLuma = FroxelMediumSigmaLuma(medium);

    fogMediumScattering[id] = float4(medium.sigmaS, medium.density);

    if (sigmaLuma < 1.0e-7f || max(max(medium.sigmaS.x, medium.sigmaS.y), medium.sigmaS.z) <= 1.0e-8f)
    {
        fogMediumAbsorption[id] = float4(medium.sigmaA, 0.0f);
        fogCurrent[id] = float4(0.0f, 0.0f, 0.0f, sigmaLuma);
        return;
    }

    FroxelMatrices frame = GetMatrices(pc);

    // viewDir is camera -> sample. sampleToLightDir/sampleToSkyDir point from sample toward the source.
    // HenyeyGreenstein documents why dot(viewDir, sampleToLightDir) is the correct physical cosTheta.
    float3 viewPos  = FroxelCenterViewSpace(pc, id);
    float3 viewDir  = normalize(viewPos);

    float3 primaryDir = length(pc.sunDir) > 1.0e-4f
        ? normalize(pc.sunDir)
        : float3(0.0f, 1.0f, 0.0f);

    // Low-discrepancy temporal light-disc sampling. Spatial froxel identity is not jittered.
    float3 sampleToLightDir;
    if (pc.stochasticLight != 0u && pc.sunAngularRadius > 1.0e-5f)
    {
        float2 xi = R2Offset2D(id, pc.frameIndex);
        sampleToLightDir = SampleSunDisk(primaryDir, pc.sunAngularRadius, xi);
    }
    else
    {
        sampleToLightDir = primaryDir;
    }

    float directVisibility = DirectLightVisibility(worldPos, sampleToLightDir);
    float directPhase = HenyeyGreenstein(dot(viewDir, sampleToLightDir), medium.anisotropyG);
    float3 source = medium.sigmaS * (directVisibility * pc.sunColor * frame.weatherParams.w) * directPhase;

    // Stable sky/ambient illumination. The probe set is fixed in world space and reused every frame;
    // temporal reconstruction accumulates the same estimator instead of chasing random hemisphere noise.
    float3 ambientIncident = float3(0.0f, 0.0f, 0.0f);
    if (frame.ambientRadiance.a > 1.0e-5f && frame.airParams.y > 1.0e-5f)
    {
        [unroll]
        for (uint i = 0u; i < kFroxelSkyProbeCount; ++i)
        {
            float3 skyDir = FroxelSkyProbeDirection(i);
            float skyVisibility = SkyProbeVisibility(worldPos, skyDir);
            float skyPhase = HenyeyGreenstein(dot(viewDir, skyDir), medium.anisotropyG);
            ambientIncident += skyVisibility * frame.ambientRadiance.xyz * skyPhase;
        }
        ambientIncident *= (frame.ambientRadiance.a * frame.airParams.y) / float(kFroxelSkyProbeCount);
        source += medium.sigmaS * ambientIncident;
    }

    fogMediumAbsorption[id] = float4(medium.sigmaA, directVisibility);
    fogCurrent[id] = float4(max(source, float3(0.0f, 0.0f, 0.0f)), sigmaLuma);
}
