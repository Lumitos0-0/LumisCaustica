// Low-pass the deterministic scattering field in screen-froxel space before the per-pixel raymarch.
// Because the volume has no temporal accumulation, a wider, colour-aware bilateral kernel is the primary
// denoiser. It must average the stochastic per-cell shot noise (emissive RIS selection, 2-ray directional)
// aggressively in smooth fog while preserving the extinction/colour discontinuities that define light-shaft
// and emissive edges. A 5x5 kernel with bilateral weights on BOTH extinction and peak-channel luminance
// achieves this; the sigma widens with distance so distant shafts denoise without blurring near fog.

import world_common;
import world_core;
import bindings;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint3 index = DispatchRaysIndex();
    uint3 dispatchDimensions = DispatchRaysDimensions();
    if (any(index >= dispatchDimensions)) {
        return;
    }

    int3 dimensions = worldPush.froxelGrid.xyz;
    int3 center = int3(index);
    float4 centerSample = froxelScattering[center];
    float centerExt = max(centerSample.a, 1.0e-6);
    // Peak-channel luminance drives the colour bilateral so a single bright emitter cell can't be the
    // reference that lets its neighbours survive unchallenged.
    float centerLum = max(max(centerSample.r, centerSample.g), centerSample.b);

    // Distance-adaptive sigma: wider at distance (where noise is worst and detail least), tighter up close
    // so near fog and near shafts stay sharp.
    float dist01 = saturate(float(center.z) / float(dimensions.z));
    float sigmaExt = lerp(0.45, 0.70, dist01 * dist01);           // extinction (1/block scale)
    float sigmaLum = lerp(0.06, 0.12, dist01 * dist01);           // colour luminance (scene units)
    float inv2Ext2 = 1.0 / (2.0 * sigmaExt * sigmaExt);
    float inv2Lum2 = 1.0 / (2.0 * sigmaLum * sigmaLum);

    float4 filtered = float4(0.0);
    float weightSum = 0.0;

    // 5x5 centre-weighted spatial kernel.
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            int3 samplePos = clamp(center + int3(x, y, 0), int3(0), dimensions - 1);
            float4 s = froxelScattering[samplePos];

            float radius2 = float(x * x + y * y);
            float spatialWeight = exp(-radius2 * 0.16);           // gaussian falloff, centre = 1

            float extDiff = s.a - centerExt;
            float lum = max(max(s.r, s.g), s.b);
            float lumDiff = lum - centerLum;
            float bilateral = exp(-extDiff * extDiff * inv2Ext2 - lumDiff * lumDiff * inv2Lum2);

            float weight = spatialWeight * bilateral;
            filtered += s * weight;
            weightSum += weight;
        }
    }

    // Fallback to pure spatial if all bilateral weights collapsed (extremely rare).
    if (weightSum < 1.0e-5) {
        weightSum = 0.0;
        filtered = float4(0.0);
        for (int y = -2; y <= 2; ++y) {
            for (int x = -2; x <= 2; ++x) {
                int3 samplePos = clamp(center + int3(x, y, 0), int3(0), dimensions - 1);
                float radius2 = float(x * x + y * y);
                float weight = exp(-radius2 * 0.16);
                filtered += froxelScattering[samplePos] * weight;
                weightSum += weight;
            }
        }
    }

    froxelFiltered[center] = filtered / max(weightSum, 1.0e-6);
}
