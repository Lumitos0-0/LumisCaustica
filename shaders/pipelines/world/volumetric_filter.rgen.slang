// Low-pass the temporally accumulated scattering field in screen-froxel space before Z integration.
// This filter operates on the continuous volume itself, not scene geometry. Final composition samples the
// 3D field at each pixel's exact depth. A centre-weighted kernel keeps denoising while preserving the
// higher-frequency visibility transitions that define light-shaft edges.
//
// Improvement: simple bilateral weighting on extinction difference (cheap, preserves sharp shaft boundaries
// while still smoothing stochastic RIS + temporal noise). The sigma is tuned for typical fog extinction range.

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

    float4 filtered = float4(0.0);
    float weightSum = 0.0;

    static const float bilateralSigma = 0.35;   // tuned for typical fog extinction (1/block scale)
    static const float bilateralInvSigma2 = 1.0 / (bilateralSigma * bilateralSigma);

    for (int y = -1; y <= 1; ++y) {
        float wy = y == 0 ? 4.0 : 1.0;
        for (int x = -1; x <= 1; ++x) {
            float wx = x == 0 ? 4.0 : 1.0;
            int3 samplePos = clamp(center + int3(x, y, 0), int3(0), dimensions - 1);
            float4 s = froxelScattering[samplePos];

            // Spatial weight (centre-biased 3x3)
            float spatialWeight = wx * wy;

            // Bilateral weight on extinction difference (preserves shaft edges)
            float extDiff = s.a - centerExt;
            float bilateralWeight = exp(-extDiff * extDiff * bilateralInvSigma2);

            float weight = spatialWeight * bilateralWeight;
            filtered += s * weight;
            weightSum += weight;
        }
    }

    // Fallback to pure spatial if all bilateral weights collapsed (extremely rare)
    if (weightSum < 1.0e-5) {
        weightSum = 0.0;
        filtered = float4(0.0);
        for (int y = -1; y <= 1; ++y) {
            float wy = y == 0 ? 4.0 : 1.0;
            for (int x = -1; x <= 1; ++x) {
                float wx = x == 0 ? 4.0 : 1.0;
                int3 samplePos = clamp(center + int3(x, y, 0), int3(0), dimensions - 1);
                float weight = wx * wy;
                filtered += froxelScattering[samplePos] * weight;
                weightSum += weight;
            }
        }
    }

    froxelFiltered[center] = filtered / max(weightSum, 1.0e-6);
}
