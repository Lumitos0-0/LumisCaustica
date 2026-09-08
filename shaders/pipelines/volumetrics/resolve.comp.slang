// Temporal resolve for the shaft buffer: reproject, validate, blend.
//
// The signal is one binary visibility sample per pixel, so its variance is p*(1-p) — up to 0.25 — and
// there is no way around accumulating a lot of frames. What makes that affordable is that the sun is
// the only light involved and it barely moves, so history is almost always valid and almost always
// still correct; camera motion and moving occluders are the only invalidation sources.
//
// Reprojection uses the deterministic scattering centroid the shaft pass stored, not surface motion
// vectors. In-scatter has no surface, so the surface motion field is simply the wrong vector field for
// it — that mismatch is also exactly why this cannot be fed to DLSS-RR, which is guided by surface
// depth, normal and motion and would smear the beams against them.
import world_common;
import volumetrics;
import volumetrics_common;
import bindings;

[[vk::push_constant]] VolumePush pc;

// History is rejected when the reprojected pixel's scattering centroid disagrees with this pixel's by
// more than this fraction. Both values are deterministic functions of the segment length, so in
// practice this fires on depth discontinuities — an occluder edge sweeping across the pixel — and not
// on Monte Carlo noise.
static const float DISTANCE_REJECT_FRACTION = 0.2;

// Neighbourhood clamp width, in standard deviations of the raw 3x3. Deliberately generous: with one
// sample per pixel the raw neighbourhood is itself noisy, so a tight clamp would reject valid history
// and reintroduce the noise it is meant to suppress. Its job here is to catch gross ghosting behind a
// moving occluder, which is many sigma out.
static const float CLAMP_SIGMA = 4.0;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 pixel = tid.xy;
    if (pixel.x >= pc.renderW || pixel.y >= pc.renderH) {
        return;
    }
    WorldPush push = ConstPtr<WorldPush>(pc.worldPushAddr)[0];

    float4 raw = shaftRawImage[pixel];
    float3 current = raw.rgb;
    float meanDistance = raw.a;

    // Colour moments of the raw 3x3, for the clamp below.
    float3 sum = float3(0.0, 0.0, 0.0);
    float3 sumSquares = float3(0.0, 0.0, 0.0);
    int taps = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 neighbour = int2(pixel) + int2(dx, dy);
            if (neighbour.x < 0 || neighbour.y < 0
                    || neighbour.x >= int(pc.renderW) || neighbour.y >= int(pc.renderH)) {
                continue;
            }
            float3 value = shaftRawImage[uint2(neighbour)].rgb;
            sum += value;
            sumSquares += value * value;
            taps++;
        }
    }
    float inverseTaps = 1.0 / float(max(taps, 1));
    float3 mean = sum * inverseTaps;
    float3 variance = max(sumSquares * inverseTaps - mean * mean, float3(0.0, 0.0, 0.0));
    float3 deviation = sqrt(variance) * CLAMP_SIGMA;

    float3 resolved = current;
    float frames = 1.0;

    if (pc.historyValid != 0u) {
        float2 size = float2(float(pc.renderW), float(pc.renderH));
        float2 uv = (float2(pixel) + 0.5) / size + push.jitter / size;
        float3 origin;
        float3 dir;
        volumeCameraRay(push.invViewProj, uv, origin, dir);
        float3 reprojected = volumeReprojectUv(push, origin + dir * meanDistance);
        if (reprojected.z > 0.0) {
            float4 history = shaftAccumPrev.SampleLevel(reprojected.xy, 0.0);
            // Nearest fetch for the age: it is a counter, and interpolating it across a disocclusion
            // boundary would hand a freshly-revealed pixel a long history it never had.
            uint2 historyPixel = uint2(clamp(reprojected.xy * size, float2(0.0, 0.0), size - 1.0));
            float historyAge = shaftAgePrev[historyPixel];

            float distanceDelta = abs(history.a - meanDistance);
            bool consistent = distanceDelta
                    <= DISTANCE_REJECT_FRACTION * max(meanDistance, 1.0) && historyAge > 0.0;
            if (consistent) {
                float3 clamped = clamp(history.rgb, mean - deviation, mean + deviation);
                frames = min(historyAge + 1.0, float(max(pc.maxAccumFrames, 1u)));
                resolved = lerp(clamped, current, 1.0 / max(frames, 1.0));
            }
        }
    }

    shaftAccumImage[pixel] = float4(resolved, meanDistance);
    shaftAgeImage[pixel] = frames;
}
