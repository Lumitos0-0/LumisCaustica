// Froxel lighting pass: fill the frustum-aligned volume with extinction + in-scattered radiance.
//
// One thread == one froxel (tileX, tileY, depth slice). Each froxel samples the fog density field,
// casts ray-queried occlusion rays toward the dominant celestial light (the amortized "path-traced
// volumetric shadow" that yields god rays / light shafts), and writes rgb = in-scatter, w = extinction
// into the froxel volume buffer. Because the (expensive, opacity-tested) shadow work is projected once
// per froxel instead of once per pixel step, the per-pixel fog cost collapses to a fixed march in
// integrate.comp — the spatial-accumulation savings a froxel grid exists to deliver.
//
// In-scatter integrates two light sources, the same decomposition Wronski's volumetric fog and
// Frostbite's unified volumetric rendering use:
//   - the dominant celestial light (below), ray-occluded so shafts appear where light gets through, and
//   - the ambient sky dome, evaluated from the same sky-view LUT the miss shader draws the sky from.
// Without the ambient term, shadowed columns and moonless nights scatter nothing: the fog keeps dimming
// the scene (extinction) while adding no light of its own, which reads as black haze instead of fog.
// The dominant light is derived from the same sky module (and the same transmittance LUT) the path
// tracer uses, so volumetric fog, terrain lighting and the drawn sun/moon can never disagree.
import froxel_common;
import world_common;
import sky;

[[vk::push_constant]] FroxelPush pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(0, 1)]] Sampler2D transmittanceLut;
[[vk::binding(1, 1)]] Sampler2D skyViewLut;

static const uint CULL_SECONDARY = 0x01u;  // match trace.slang: shadows see terrain/entities, not the player
static const int SHADOW_SAMPLES = 4;       // jittered occlusion rays per froxel -> soft-edged shafts
static const float RAY_TMIN = 0.001;

// Ambient skylight: the average sky radiance over the sphere (sky above, the atmosphere model's ground
// plate below), from the same sky-view LUT the miss shader draws the dome from. Six fixed world-space
// taps — zenith, nadir, four horizon points — with equal weights. The average is a pure function of the
// frame's sky state, never of the froxel, so it cannot shimmer; every thread evaluates the same taps, so
// the texture cache serves each fetch once and the cost is negligible next to the shadow rays. Fog
// scatters this field with an isotropic phase — the standard stand-in for the multiple-scattering orders
// the single-scattering model drops (Wronski 2014's "constant ambient term"; Frostbite integrates its SH
// sky probes the same way).
float3 averageSkyRadiance(SkyState state) {
    float3 sum = skyDomeRadiance(state, skyViewLut, float3(0.0, 1.0, 0.0))
            + skyDomeRadiance(state, skyViewLut, float3(0.0, -1.0, 0.0))
            + skyDomeRadiance(state, skyViewLut, float3(1.0, 0.0, 0.0))
            + skyDomeRadiance(state, skyViewLut, float3(-1.0, 0.0, 0.0))
            + skyDomeRadiance(state, skyViewLut, float3(0.0, 0.0, 1.0))
            + skyDomeRadiance(state, skyViewLut, float3(0.0, 0.0, -1.0));
    return sum / 6.0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint tileX = tid.x;
    uint tileY = tid.y;
    uint slice = tid.z;

    // Trail a partial final tile's threads off the grid before any density or ray work. tileW/tileH are
    // the cell counts actually resident in the volume; the dispatch rounds the grid up to a multiple of
    // GROUP_SIZE, and the excess threads must neither compute a shadow ray nor index out of bounds.
    uint tileW = max(unpackGridWidth(pc.packedGridDims), 1u);
    uint tileH = max(unpackGridHeight(pc.packedGridDims), 1u);
    if (tileX >= tileW || tileY >= tileH) {
        return;
    }

    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];

    // The scene's one directional light, from the shared sky state. Convert the BT.709 illuminance to
    // the transport basis (ACEScg) exactly where world.indirect.rgen does, so fog and surfaces agree.
    SkyState state = skyState(worldPush);
    CelestialLight light = dominantCelestialLight(state, transmittanceLut);
    float3 toLight = light.dir;
    float halfAngle = light.halfAngle;

    uint renderW = max(unpackRenderWidth(pc.packedRenderSize), 1u);
    uint renderH = max(unpackRenderHeight(pc.packedRenderSize), 1u);
    uint tileSize = max(uint(pc.params.z), 1u);
    uint slices = max(uint(pc.params.w), 1u);
    float nearDist = pc.params.x;
    float farDist = pc.params.y;

    // Tile-centre NDC. The froxel is aligned to the frustum, so the tile centre's ray is the grid's
    // representative view direction for the whole tile (the coarse-resolution approximation froxels use).
    float2 ndc = float2((float(tileX) + 0.5) * tileSize / float(renderW),
                        (float(tileY) + 0.5) * tileSize / float(renderH)) * 2.0 - 1.0;
    float3 nearP, farP, dir;
    frustumRay(ndc, worldPush.invViewProj, nearP, farP, dir);

    float d = sliceMidDist(slice, slices, farDist, nearDist);
    float3 rebasedPos = nearP + dir * d;
    float3 worldPos = rebasedPos + worldPush.camOffset;

    float density = fogDensity(worldPos, worldPush.camOffset.y,
            pc.look.x, pc.look.y, pc.aux.y, pc.aux.z);
    if ((worldPush.flags & 1u) != 0u) {
        // Water scatters far more per block than air, so a submerged camera's volume is much murkier. The
        // path tracer's water medium already handles absorption; this multiplier adds the scattered
        // component that makes underwater views hazy and light shafts through the surface visible.
        // (WorldPush.flags bit0 = submerged.)
        density *= max(pc.sky.y, 0.0);
    }
    float sigmaT = density;             // per-block extinction (absorption + scattering)
    float sigmaS = sigmaT * pc.aux.x;   // scattering coefficient = sigma_t * albedo
    // The user-facing intensity and tint scale every in-scattered term uniformly (direct and ambient), so
    // "intensity 0" is a pure-extinction fog and the tint (ref.yzw, 1 = neutral) shades shafts and haze
    // together. sky.x then sets how much ambient skylight the fog scatters relative to the direct
    // sun/moon term: 1 is the physical balance, 0 recovers shafts-only behaviour.
    float3 scatterScale = pc.ref.x * pc.ref.yzw;
    float3 inScatter = float3(0.0, 0.0, 0.0);

    if (sigmaT > 1.0e-4) {
        // Ambient sky in-scatter first: unoccluded (a froxel cannot afford occlusion rays for the whole
        // dome; Frostbite's SH ambient is unshadowed too) and isotropic, per averageSkyRadiance above.
        if (pc.sky.x > 0.0) {
            inScatter += sigmaS * ISOTROPIC_PHASE * averageSkyRadiance(state) * (pc.sky.x * scatterScale);
        }

        // Direct sun/moon in-scatter, occluded by the scene.
        // Phase between the view direction and the light's incoming direction. Looking toward the sun
        // through the fog puts cosTheta ≈ 1 and the forward lobe peaks -> bright shafts.
        float cosTheta = dot(dir, toLight);
        float phase = hgPhase(cosTheta, pc.look.z);

        // Angular frame for the light's square, used to jitter the occlusion ray within its half angle.
        float3 right;
        float3 up;
        celestialSquareFrame(toLight, right, up);
        float tanA = tan(halfAngle);

        // Deterministic jitter per froxel (no frame index), so the averaged occlusion is stable and the
        // fog cannot shimmer. Averaging several rays across the light's extent turns hard shadow edges
        // into penumbra-shaped gradients across the body's disc.
        //
        // RAY_FLAG_FORCE_OPAQUE is a deliberate approximation: inline ray queries never execute any-hit
        // shaders (the Vulkan spec returns non-opaque candidates to the caller for manual confirmation),
        // so the cutout alpha test the shadow SBT's rahit performs for the path tracer cannot run here
        // without re-binding the block atlas and walking the section/material tables per ray. Treating
        // all geometry as opaque errs toward solid shadows — foliage canopies and glass block fully —
        // which is the conservative choice for a shadow query; the disc jitter above still softens the
        // penumbra the scene's solid geometry casts.
        float base = hash3(float3(float(tileX), float(tileY), float(slice)));
        float occluded = 0.0;
        for (int s = 0; s < SHADOW_SAMPLES; s++) {
            float2 local = float2(hash3(float3(base, float(s), 0.37)),
                                  hash3(float3(base, float(s), 7.13))) * 2.0 - 1.0;
            float3 towardLight = normalize(toLight + right * local.x * tanA + up * local.y * tanA);
            RayDesc ray;
            ray.Origin = worldPos;
            ray.TMin = RAY_TMIN;
            ray.Direction = towardLight;
            ray.TMax = 10000.0;
            RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
            query.TraceRayInline(tlas, RAY_FLAG_NONE, CULL_SECONDARY, ray);
            while (query.Proceed()) {
            }
            occluded += query.CommittedStatus() != COMMITTED_NOTHING ? 1.0 : 0.0;
        }
        // Fraction of jittered rays that reached the light: 1 in open air, 0 in full shadow. The froxel
        // in-scatters only the visible fraction -- shafts appear where light gets through, and shadowed
        // columns of air keep only their ambient sky in-scatter (occluded rays must not *add* light).
        float visibility = 1.0 - occluded / float(SHADOW_SAMPLES);

        inScatter += sigmaS * phase * bt709ToAcesCg(light.illuminance) * visibility * scatterScale;
    }

    uint lin = (slice * tileH + tileY) * tileW + tileX;

    float4 value = float4(inScatter, sigmaT);
    // Temporal blend (same-index EMA) only while the camera is essentially stationary, so it reduces
    // shadow/fog flicker on a still view without smearing translated content. Disabled entirely when no
    // history buffer is bound or the blend weight is zero.
    if (pc.froxelHistoryAddr != 0u && pc.aux.w > 0.0 && length(worldPush.camDelta) < 0.01) {
        float4 prev = ConstPtr<float4>(pc.froxelHistoryAddr)[lin];
        value = lerp(prev, value, pc.aux.w);
    }
    DevicePtr<float4>(pc.froxelVolumeAddr)[lin] = value;
    if (pc.froxelHistoryAddr != 0u) {
        DevicePtr<float4>(pc.froxelHistoryAddr)[lin] = value; // feed next frame's temporal blend
    }
}
