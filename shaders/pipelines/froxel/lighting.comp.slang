// Froxel lighting pass: fill the frustum-aligned volume with extinction + in-scattered radiance.
//
// One thread == one froxel (tileX, tileY, depth slice). Each froxel samples the fog density field,
// casts ray-queried occlusion rays toward the dominant celestial light (the amortized "path-traced
// volumetric shadow" that yields god rays / light shafts), and writes rgb = in-scatter, w = extinction
// into the froxel volume buffer. Because the (expensive, opacity-tested) shadow work is projected once
// per froxel instead of once per pixel step, the per-pixel fog cost collapses to a fixed march in
// integrate.comp — the spatial-accumulation savings a froxel grid exists to deliver.
//
// The dominant light is derived from the same sky module (and the same transmittance LUT) the path
// tracer uses, so volumetric fog, terrain lighting and the drawn sun/moon can never disagree.
import froxel_common;
import world_common;
import sky;

[[vk::push_constant]] FroxelPush pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(0, 1)]] Sampler2D transmittanceLut;

static const uint CULL_SECONDARY = 0x01u;  // match trace.slang: shadows see terrain/entities, not the player
static const int SHADOW_SAMPLES = 4;       // jittered occlusion rays per froxel -> soft, dappled shafts
static const float RAY_TMIN = 0.001;
// Water scatters far more per block than air, so a submerged camera's volume is much murkier. The path
// tracer's water medium already handles absorption; this multiplier adds the scattered component that
// makes underwater views hazy and light shafts through the surface visible. (WorldPush.flags bit0 = submerged.)
static const float SUBMERGED_DENSITY_MULTIPLIER = 8.0;

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
    CelestialLight light = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    float3 sunRadiance = bt709ToAcesCg(light.illuminance) * pc.ref.x; // ref.x = global intensity
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
        density *= SUBMERGED_DENSITY_MULTIPLIER; // underwater: thick, light-scattering murk
    }
    float sigmaT = density;             // per-block extinction (absorption + scattering)
    float sigmaS = sigmaT * pc.aux.x;   // scattering coefficient = sigma_t * albedo
    float3 inScatter = float3(0.0, 0.0, 0.0);

    if (sigmaT > 1.0e-4) {
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
        // fog cannot shimmer. Averaging several rays across the light's extent turns hard opaque edges
        // into soft, partially-lit shafts through cutout foliage and windows.
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
        float occl = occluded / float(SHADOW_SAMPLES);

        inScatter = sigmaS * phase * sunRadiance * occl * pc.ref.yzw;
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
