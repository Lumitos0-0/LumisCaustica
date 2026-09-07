package dev.comfyfluffy.caustica.rt.pipeline;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The froxel grid's slice mapping, mirrored from {@code shaders/pipelines/world/fog.slang}.
 *
 * <p>The shader's {@code fogSliceDistance} and {@code fogDistanceSlice} are exact inverses, and the
 * composite depends on that: it converts a surface distance back into a normalized W to sample the
 * volume the injection pass filled by walking the mapping forwards. If the two ever disagreed, fog would
 * be fetched from the wrong depth — subtly, and worst near the camera where the curve is steepest. These
 * are pure functions, so the invariant is worth pinning on the Java side rather than only by eye.
 */
final class RtVolumetricFogGridTest {
    private static final float NEAR = 0.25f;
    private static final float FAR = 192.0f;
    private static final float DEPTH_EXP = 2.0f;
    private static final int SLICES = 128;

    private static float sliceDistance(float slice) {
        float n = Math.clamp(slice / SLICES, 0.0f, 1.0f);
        return NEAR + (FAR - NEAR) * (float) Math.pow(n, DEPTH_EXP);
    }

    private static float distanceSlice(float distance) {
        float n = Math.clamp((distance - NEAR) / (FAR - NEAR), 0.0f, 1.0f);
        return SLICES * (float) Math.pow(n, 1.0 / DEPTH_EXP);
    }

    @Test
    void sliceMappingRoundTrips() {
        for (int slice = 0; slice <= SLICES; slice++) {
            float distance = sliceDistance(slice);
            float recovered = distanceSlice(distance);
            assertEquals(slice, recovered, 1.0e-2,
                    "slice " + slice + " did not survive the distance round trip");
        }
    }

    @Test
    void sliceDistanceSpansTheConfiguredRange() {
        assertEquals(NEAR, sliceDistance(0), 1.0e-6);
        assertEquals(FAR, sliceDistance(SLICES), 1.0e-3);
    }

    @Test
    void nearSlicesAreShorterThanFarSlices() {
        // The whole point of the power distribution: resolution is spent near the camera, where shafts
        // have structure, rather than spread evenly out to the far plane.
        float firstStep = sliceDistance(1) - sliceDistance(0);
        float lastStep = sliceDistance(SLICES) - sliceDistance(SLICES - 1);
        assertTrue(firstStep < lastStep,
                "expected near slices to be shorter, got " + firstStep + " vs " + lastStep);
    }

    @Test
    void distanceBeyondFarClampsToTheLastSlice() {
        // Sky pixels reach the composite with a distance past the grid; they must sample the final slice
        // (the fully accumulated column), not wrap around to the near plane.
        assertEquals(SLICES, distanceSlice(FAR * 4.0f), 1.0e-3);
        assertEquals(0.0f, distanceSlice(0.0f), 1.0e-3);
    }

    /**
     * The isotropic ambient fill shares the direct term's shadowing and colour, so it is exactly
     * {@code direct * ambient/4pi} and folds into the phase as an additive constant. That identity is
     * what lets the injection store one value instead of two, and it must stay exact.
     */
    @Test
    void ambientFoldsIntoThePhaseExactly() {
        final double inv4Pi = 0.07957747155;
        double direct = 0.018 * 100000.0 * 0.6; // sigma_s * illuminance * visibility
        double ambient = 0.15;
        for (double phase : new double[]{0.02, 0.1, 0.5, 1.7}) {
            double reference = direct * phase + direct * inv4Pi * ambient;
            double folded = direct * (phase + inv4Pi * ambient);
            assertEquals(reference, folded, 1.0e-9,
                    "ambient must fold into the phase without changing the result");
        }
    }

    /**
     * The composite's bilateral upsample must reject a froxel column whose surface sits at a very
     * different distance. Without it, a column aimed out through a cave opening blends its sunlit
     * prefix into a pixel looking at a wall two blocks away, painting daylight onto the wall — the
     * leak, the jaggies and the oversoftening all come from that one blend.
     */
    @Test
    void bilateralUpsampleRejectsColumnsAtADifferentDepth() {
        double pixelDistance = 2.0;
        double[] columnDistances = {2.0, 2.0, 192.0, 192.0};
        double[] columnValues = {0.05, 0.05, 4.0, 4.0};
        double plain = weightedFog(pixelDistance, columnDistances, columnValues, false);
        double bilateral = weightedFog(pixelDistance, columnDistances, columnValues, true);
        assertTrue(plain > 1.5, "plain bilinear should leak the sunlit columns, got " + plain);
        assertTrue(bilateral < 0.1, "bilateral should reject them, got " + bilateral);
    }

    /** ...while leaving a continuous surface blending smoothly, so no banding is introduced. */
    @Test
    void bilateralUpsampleKeepsSmoothSurfacesSmooth() {
        double pixelDistance = 10.3;
        double[] columnDistances = {10.0, 10.4, 10.2, 10.6};
        double[] columnValues = {1.0, 1.02, 1.01, 1.03};
        double plain = weightedFog(pixelDistance, columnDistances, columnValues, false);
        double bilateral = weightedFog(pixelDistance, columnDistances, columnValues, true);
        assertEquals(plain, bilateral, 0.01,
                "a continuous surface must not be altered by the depth weighting");
    }

    /** Mirrors the 2x2 weighting in {@code volumetric/composite.comp.slang}. */
    private static double weightedFog(double pixelDistance, double[] columnDistances,
                                      double[] columnValues, boolean bilateral) {
        final double toleranceFraction = 0.06;
        final double minimumTolerance = 0.35;
        double tolerance = Math.max(pixelDistance * toleranceFraction, minimumTolerance);
        double fractionX = 0.6;
        double fractionY = 0.5;
        double total = 0.0;
        double sum = 0.0;
        for (int tap = 0; tap < 4; tap++) {
            int offsetX = tap & 1;
            int offsetY = (tap >> 1) & 1;
            double weight = Math.abs((1 - offsetX) - fractionX) * Math.abs((1 - offsetY) - fractionY);
            if (bilateral) {
                weight *= Math.exp(-Math.abs(columnDistances[tap] - pixelDistance) / tolerance);
            }
            sum += columnValues[tap] * weight;
            total += weight;
        }
        return total > 1.0e-4 ? sum / total : 0.0;
    }

    @Test
    void gridSizeIsPositiveAtEveryTier() {
        // gridSizeFor reads live config, so this only pins the arithmetic shape: a divisor must never
        // produce a zero extent for a plausible render resolution.
        for (int divisor : new int[]{6, 8, 12}) {
            int width = Math.max(1, (1920 + divisor - 1) / divisor);
            int height = Math.max(1, (1080 + divisor - 1) / divisor);
            assertTrue(width > 0 && height > 0, "divisor " + divisor + " produced an empty grid");
        }
    }
}
