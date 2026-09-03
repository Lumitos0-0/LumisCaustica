// Froxel lighting pass: fill the frustum-aligned volume with extinction + in-scattered radiance.
//
// One thread == one froxel (tileX, tileY, depth slice). Each froxel samples the fog density field,
// casts ray-queried occlusion rays toward the dominant celestial light (the amortized "path-traced
// volumetric shadow" that yields god rays / light shafts), and writes rgb = in-scatter, w = extinction
// into the froxel volume buffer. Because the (expensive, opacity-tested) shadow work is projected once
// per froxel instead of once per pixel step, the per-pixel fog cost collapses to a filtered march in
// integrate.comp — the spatial-accumulation savings a froxel grid exists to deliver.
//
// In-scatter integrates two light sources, the same decomposition Wronski's volumetric fog and
// Frostbite's unified volumetric rendering use: the dominant celestial light (ray-occluded, so shafts
// appear where light gets through) and the ambient sky dome (unshadowed, like Frostbite's SH ambient).
//
// BOTH terms are anchored to the sky dome's radiance, not to the lights' physical illuminance. The
// look package's sun delivers 128,000 lux and its moon 5 — a 25,600:1 ratio — so an illuminance-linear
// fog is either a white flashbang by day or invisible by night at any single setting (exactly the
// failure mode raw lux produced here). The dome's own radiance already tracks time of day AND the
// scene's auto exposure: whenever the player can see the sky, sky-anchored fog reads at a comparable
// brightness, day or night. This is also how shipping engines steer it — UE's fog light contribution is
// an artist scalar ("volumetric scattering intensity") over authored colors, never raw solar lux.
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

// Direct shaft brightness, in multiples of the dome average. 12x keeps sun-facing fog near the
// brightness of sun-lit terrain (a readable glow with clear god rays) instead of the 5x-terrain
// whiteout raw illuminance produced. Scales with the dome, so it holds day and night alike.
static const float SHAFT_SKY_RATIO = 12.0;
// Ambient share kept in full shadow (caves, overhangs): the occlusion rays only probe the dominant
// light's direction, so they are a one-direction proxy for sky visibility. The floor keeps sealed
// caves from filling with daylight haze while sunlit/shadowed contrast makes the shafts read harder.
static const float AMBIENT_SHADOW_FLOOR = 0.15;

// Ambient skylight: the average sky radiance over the sphere (sky above, the atmosphere model's ground
// plate below), from the same sky-view LUT the miss shader draws the dome from. Six fixed world-space
// taps — zenith, nadir, four horizon points — with equal weights. The average is a pure function of the
// frame's sky state, never of the froxel, so it cannot shimmer; every thread evaluates the same taps, so
// the texture cache serves each fetch once and the cost is negligible next to the shadow rays.
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

    // The dominant celestial light supplies the shaft DIRECTION and the disc's angular extent (for the
    // occlusion jitter). Its illuminance is deliberately not used for magnitude — see the header.
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

    // Fade the density noise out with distance: a far froxel spans many periods of the noise field, so
    // sampling it at the froxel centre turns the variation into per-froxel randomness — patchy, shimming
    // far fog. Near the camera the froxels resolve the field properly, so the fade is invisible there.
    float noiseFade = 1.0 - smoothstep(0.3 * farDist, 0.7 * farDist, d);
    float density = fogDensity(worldPos, worldPush.camOffset.y,
            pc.look.x, pc.look.y, pc.aux.y * noiseFade, pc.aux.z);
    if ((worldPush.flags & 1u) != 0u) {
        // Water scatters far more per block than air, so a submerged camera's volume is much murkier. The
        // path tracer's water medium already handles absorption; this multiplier adds the scattered
        // component that makes underwater views hazy and light shafts through the surface visible.
        // (WorldPush.flags bit0 = submerged.)
        density *= max(pc.sky.y, 0.0);
    }
    float sigmaT = density;             // per-block extinction (absorption + scattering)
    float sigmaS = sigmaT * pc.aux.x;   // scattering coefficient = sigma_t * albedo
    // ref.x scales the sun/moon shafts (in multiples of the dome anchor), ref.yzw tints all scattered
    // light linearly (1 = neutral); sky.x sets the ambient share. Both terms ride the same dome anchor,
    // so every knob behaves the same by day and by night.
    float3 tint = pc.ref.yzw;
    float3 inScatter = float3(0.0, 0.0, 0.0);

    if (sigmaT > 1.0e-4) {
        float3 avgSky = averageSkyRadiance(state);

        // Phase between the view direction and the light's incoming direction. Looking toward the light
        // through the fog puts cosTheta ≈ 1 and the forward lobe peaks -> bright shafts.
        float cosTheta = dot(dir, toLight);
        float phase = hgPhase(cosTheta, pc.look.z);

        // Angular frame for the light's square, used to jitter the occlusion ray within its half angle.
        float3 right;
        float3 up;
        celestialSquareFrame(toLight, right, up);
        float tanA = tan(halfAngle);

        // A rotated four-point grid across the light's disc: the taps are always 90° apart (the
        // per-froxel hash only spins the grid and staggers the radii), so four samples can never
        // cluster and the averaged visibility is far less quantized than four independent random taps.
        // Deterministic per froxel (no frame index), so the averaged occlusion cannot shimmer.
        //
        // RAY_FLAG_FORCE_OPAQUE is a deliberate approximation: inline ray queries never execute any-hit
        // shaders (the Vulkan spec returns non-opaque candidates to the caller for manual confirmation),
        // so the cutout alpha test the shadow SBT's rahit performs for the path tracer cannot run here
        // without re-binding the block atlas and walking the section/material tables per ray. Treating
        // all geometry as opaque errs toward solid shadows — foliage canopies and glass block fully —
        // which is the conservative choice for a shadow query.
        float base = hash3(float3(float(tileX), float(tileY), float(slice)));
        float rot = base * 6.2831853;
        float occluded = 0.0;
        for (int s = 0; s < SHADOW_SAMPLES; s++) {
            float ang = rot + float(s) * 1.5707963;
            float rad = 0.25 + 0.7 * hash3(float3(base, float(s), 3.71));
            float2 local = float2(cos(ang), sin(ang)) * rad;
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
        // in-scatters only the visible fraction — shafts appear where light gets through, and shadowed
        // columns of air keep only their dimmed ambient share (occluded rays must not *add* light).
        float visibility = 1.0 - occluded / float(SHADOW_SAMPLES);

        // Direct sun/moon shafts, occluded by the scene and anchored to the dome (see header).
        inScatter += sigmaS * phase * (avgSky * SHAFT_SKY_RATIO) * visibility * (pc.ref.x * tint);

        // Ambient skylight. NOTE the phase cancellation: in-scatter is the phase-weighted integral of
        // radiance over the sphere, and with an isotropic phase (1/4π) against the sphere-AVERAGE
        // radiance the 4π solid angle cancels the 1/4π — uniform radiance L̄ scatters at σs·L̄ per unit
        // length, NOT σs·L̄/4π. (Writing σs·(1/4π)·L̄ leaves the term 4π ≈ 12.6× too dim, which is why
        // the first cut of this term was invisible at night.) The occlusion floor darkens cave and
        // overhang haze and hardens the shaft contrast.
        float skyWeight = lerp(AMBIENT_SHADOW_FLOOR, 1.0, visibility);
        inScatter += sigmaS * avgSky * skyWeight * (pc.sky.x * tint);
    }

    uint lin = (slice * tileH + tileY) * tileW + tileX;

    float4 value = float4(inScatter, sigmaT);
    // Temporal blend (same-index EMA) only while the camera is essentially stationary in BOTH position
    // and orientation. The volume is frustum-aligned, so once the view rotates the same index addresses
    // a different world region and blending across that smears shafts into ghost trails. A w=0
    // direction through the view-projection cancels the translation (only the rotation block
    // contributes), so comparing curViewProj and prevViewProj on one isolates the orientation change.
    float4 fwdCur = mul(worldPush.curViewProj, float4(0.0, 0.0, -1.0, 0.0));
    float4 fwdPrev = mul(worldPush.prevViewProj, float4(0.0, 0.0, -1.0, 0.0));
    bool cameraRotated = distance(fwdCur.xyz, fwdPrev.xyz) > 1.0e-4;
    if (pc.froxelHistoryAddr != 0u && pc.aux.w > 0.0
            && length(worldPush.camDelta) < 0.01 && !cameraRotated) {
        float4 prev = ConstPtr<float4>(pc.froxelHistoryAddr)[lin];
        value = lerp(prev, value, pc.aux.w);
    }
    DevicePtr<float4>(pc.froxelVolumeAddr)[lin] = value;
    if (pc.froxelHistoryAddr != 0u) {
        DevicePtr<float4>(pc.froxelHistoryAddr)[lin] = value; // feed next frame's temporal blend
    }
}
