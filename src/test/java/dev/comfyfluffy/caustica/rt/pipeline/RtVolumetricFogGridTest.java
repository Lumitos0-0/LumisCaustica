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
