// Spatial cleanup for the shaft buffer. A 3x3 cross-bilateral pass whose step size shrinks as the
// temporal history matures, guided by the scattering centroid distance.
//
// The kernel's footprint is capped on purpose. The physical penumbra of a sun shaft is at most about
// eight display pixels and often far less, so a filter wider than that stops removing noise and starts
// removing the thing the pass exists to preserve. Two render pixels of step is the widest setting here,
// giving a four-pixel footprint — inside the penumbra even at the sharp end.
import world_common;
import volumetrics;
import volumetrics_common;
import bindings;

[[vk::push_constant]] VolumePush pc;

// Below this many accumulated frames the noise still dominates and the wide step earns its blur.
static const float WIDE_STEP_AGE = 8.0;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 pixel = tid.xy;
    if (pixel.x >= pc.renderW || pixel.y >= pc.renderH) {
        return;
    }

    float4 centre = shaftAccumImage[pixel];
    float centreDistance = centre.a;
    float age = shaftAgeImage[pixel];
    int step = age < WIDE_STEP_AGE ? 2 : 1;

    // Distance tolerance scales with the distance itself: a two-block disagreement is a hard edge at
    // five blocks and nothing at two hundred.
    float tolerance = max(0.08 * centreDistance, 0.5);

    float3 sum = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 neighbour = int2(pixel) + int2(dx, dy) * step;
            if (neighbour.x < 0 || neighbour.y < 0
                    || neighbour.x >= int(pc.renderW) || neighbour.y >= int(pc.renderH)) {
                continue;
            }
            float4 tap = shaftAccumImage[uint2(neighbour)];
            float spatial = (dx == 0 && dy == 0) ? 1.0 : ((dx == 0 || dy == 0) ? 0.5 : 0.25);
            float depth = exp(-abs(tap.a - centreDistance) / tolerance);
            float weight = spatial * depth;
            sum += volumeSanitize(tap.rgb) * weight;
            weightSum += weight;
        }
    }

    float3 filtered = weightSum > 0.0 ? sum / weightSum : centre.rgb;
    shaftFilteredImage[pixel] = float4(volumeSanitize(filtered), centreDistance);
}
