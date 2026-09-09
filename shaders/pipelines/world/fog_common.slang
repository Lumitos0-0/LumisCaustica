// Volumetric fog: the analytic medium, the froxel slice parameterisation, and the camera-ray
// reconstruction shared by the froxel lighting raygen and the fog compute passes.
//
// Deliberately parameter-passing and free of world_core/math imports: the compute passes reach
// WorldPush through a PushAddr device address rather than the raygen's module-level global, and
// importing world_core would drag the world pipeline's descriptor bindings into their reflection.
// The phase function therefore mirrors math.hg locally instead of importing it.
//
// The medium is an exponential height fog: density is flat at/below a reference altitude and decays
// with a scale height above it. Density is a function of ABSOLUTE world Y (bedrock-relative altitude),
// while every position in these shaders is rebased to the terrain origin, so the conversion goes
// through fogScatter.w (the rebase origin's block Y) — the same reason waterAnchor carries a
// world-stable anchor, except altitude feeds an exponential and therefore cannot be reduced mod 4096.
//
// Froxels are camera-frustum voxels: the lateral axes tile the same [0,1] NDC square the trace uses,
// and the depth axis distributes slice count nonlinearly (t = maxDistance * u^exponent) so slice
// thickness grows with distance and near-camera shafts keep resolution.

import world_common;

public struct FogRay {
    public float3 origin;  // rebased world space
    public float3 dir;
};

/** Scattering/extinction density (1/block) at an absolute world altitude. */
public float fogDensityAtY(WorldPush push, float absoluteY) {
    return push.fogParams.x
            * exp(-max(absoluteY - push.fogParams.z, 0.0) / max(push.fogParams.y, 1.0e-3));
}

/** Density at a rebased world position, via the pushed absolute-Y offset of rebased zero. */
public float fogDensityAt(WorldPush push, float3 rebasedPos) {
    return fogDensityAtY(push, rebasedPos.y + push.fogScatter.w);
}

public bool fogEnabled(WorldPush push) {
    return push.fogGrid.w > 0.0 && push.fogParams.x > 0.0;
}

/**
 * Transmittance of the medium itself along a light ray from a point in the volume, i.e. the fog's own
 * aerial-perspective self-shadowing. Without it every sunlit froxel out to the volume's far plane
 * contributes at full strength and the result is a uniform white veil with dark wedges cut into it,
 * rather than beams standing out against fog that the light has already been spent crossing.
 *
 * Integrated numerically over {@code distance} rather than in closed form: the exponential height
 * profile has an analytic optical depth only for rays that stay above the reference altitude, and the
 * piecewise case (ray diving below it into the flat-density region) is not worth a branch per froxel
 * when eight trapezoid taps cost less than the visibility ray this multiplies.
 */
public float fogLightTransmittance(WorldPush push, float3 rebasedPos, float3 lightDir, float distance) {
    float opticalDepth = 0.0;
    float step = distance / 8.0;
    float previous = fogDensityAt(push, rebasedPos);
    for (int tap = 1; tap <= 8; tap++) {
        float here = fogDensityAt(push, rebasedPos + lightDir * (step * float(tap)));
        opticalDepth += 0.5 * (previous + here) * step;
        previous = here;
    }
    return exp(-opticalDepth);
}

// ---- Froxel depth parameterisation -----------------------------------------------------------------

/** Distance along a view ray from a normalised slice coordinate u in [0,1]. */
public float fogSliceDistance(WorldPush push, float u) {
    return push.fogGrid.x * pow(max(u, 0.0), push.fogGrid.y);
}

/** Inverse of fogSliceDistance: normalised slice coordinate for a distance. */
public float fogSliceCoord(WorldPush push, float distance) {
    return pow(clamp(distance / max(push.fogGrid.x, 1.0e-3), 0.0, 1.0),
               1.0 / max(push.fogGrid.y, 1.0e-3));
}

// ---- Camera-ray reconstruction ---------------------------------------------------------------------
//
// Unjittered, unlike the trace: the march and the froxel grid describe the resolved frame, whose pixels
// sit at pixel centres. Rebuilt from two unprojected points because Minecraft bakes view bobbing into
// the projection, so the effective eye comes from the matrix rather than the origin.

public FogRay fogCameraRay(WorldPush push, float2 uv01) {
    float2 ndc = uv01 * 2.0 - 1.0;
    float4 nearH = mul(push.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(push.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    FogRay r;
    r.origin = nearP + push.camOffset;
    r.dir = normalize(farP - nearP);
    return r;
}

/**
 * Linear view-ray distance for a reversed-Z hardware depth guide value (gDepth: curViewProj z/w,
 * near=1 far=0). For q(t) = origin + dir*t the projection gives d = (A + B t)/(C + E t), so the distance
 * is one rational inverse. d = 0 (sky) yields the far-plane distance along the ray, which the caller
 * clamps to the volume's own extent.
 */
public float fogDistanceFromDepth(WorldPush push, FogRay ray, float depth) {
    float3 origin = ray.origin - push.camOffset;  // curViewProj is camera-relative
    // Both coefficients come from ordinary matrix-vector multiplies rather than row indexing: the
    // translation-only part is the w=1 product and the direction derivative is the w=0 product, so this
    // reads exactly like the guides' own curClip = mul(curViewProj, hitCamRel) that produced the depth.
    float4 originClip = mul(push.curViewProj, float4(origin, 1.0));
    float4 dirClip = mul(push.curViewProj, float4(ray.dir, 0.0));
    float A = originClip.z;
    float B = dirClip.z;
    float C = originClip.w;
    float E = dirClip.w;
    float denominator = depth * E - B;
    if (abs(denominator) < 1.0e-9) {
        return push.fogGrid.x;
    }
    return max(0.0, (A - depth * C) / denominator);
}

/**
 * Two-lobe Henyey-Greenstein phase at the configured anisotropy, 4pi-normalised: mostly the forward
 * lobe that makes shafts read as beams when the view aligns with the light, blended with a weak
 * backward lobe. A single forward lobe at g = 0.6 peaks near eleven times isotropic, which turns
 * looking toward the sun into a white-out that swallows the whole frame; the second lobe spends a
 * third of that peak while keeping the forward bias that the beams depend on, and lifts the
 * back-scattered haze on the side of the volume away from the light.
 */
public float fogPhase(WorldPush push, float cosTheta) {
    float g = push.fogParams.w;
    float g2 = g * g;
    float forward = (1.0 - g2) / pow(max(0.0, 1.0 + g2 - 2.0 * g * cosTheta), 1.5);
    float gb = -0.25;
    float gb2 = gb * gb;
    float backward = (1.0 - gb2) / pow(max(0.0, 1.0 + gb2 - 2.0 * gb * cosTheta), 1.5);
    return (0.65 * forward + 0.35 * backward) / (4.0 * 3.14159265359);
}
