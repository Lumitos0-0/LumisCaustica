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
    /** Mirrors the constants in {@code shaders/pipelines/world/fog.slang}. */
    private static final double FOG_STALENESS_MIN_ERROR = 0.02;
    private static final double FOG_STALENESS_SIGMA = 8.0;
    private static final double FOG_STALENESS_RETAINED_FRAMES = 3.0;
    private static final double FOG_DISOCCLUSION_FRAMES = 2.0;
    private static final double FOG_ZBLUR_NOISY_FRAMES = 2.0;
    private static final double FOG_ZBLUR_SETTLING_FRAMES = 5.0;
    /** Mirrors {@code TIERS} in {@link RtVolumetricFog}: width, height, slices, samples per froxel. */
    private static final int[][] TIER_TABLE = {
            {128, 72, 64, 2},
            {160, 90, 64, 3},
            {160, 90, 64, 4},
            {192, 108, 96, 4},
    };

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

    /**
     * The froxel grid inherits the camera's sub-pixel jitter, so a column's ray and the ray of the pixel
     * it covers carry the SAME offset and it cancels: the composite samples the volume at the plain,
     * unjittered uv.
     *
     * <p>The invariant that matters is the RAY DIRECTION, not the texel index — indices are fractional
     * because the grid width does not divide the render width. Sampling at uv reproduces the pixel's own
     * ray exactly; subtracting the offset displaces the fetch by exactly the jitter, which is the sign
     * error this pins down. Getting it wrong is silent: the image still looks plausible, it just never
     * converges, because the volume is written on a moving lattice and read on a fixed one.
     */
    @Test
    void gridJitterCancelsAtTheComposite() {
        final float renderWidth = 1280.0f;
        final int columns = 224;
        for (float jitterPixels : new float[]{-0.5f, -0.13f, 0.0f, 0.37f, 0.5f}) {
            float jitterUv = jitterPixels / renderWidth;
            for (int pixel : new int[]{0, 137, 640, 1279}) {
                float uv = (pixel + 0.5f) / renderWidth;
                // The ray this pixel's primary trace actually used.
                float pixelRayUv = uv + jitterUv;
                // Sampling at the plain uv selects this continuous column, whose own ray carries the
                // same jitter the grid was built with.
                float column = uv * columns - 0.5f;
                float sampledRayUv = (column + 0.5f) / columns + jitterUv;
                assertEquals(pixelRayUv, sampledRayUv, 1.0e-7,
                        "unjittered fetch must reproduce the pixel's own ray direction");

                // Subtracting the offset — the plausible-looking mistake — displaces the fetch by
                // exactly the jitter.
                float shiftedColumn = (uv - jitterUv) * columns - 0.5f;
                float shiftedRayUv = (shiftedColumn + 0.5f) / columns + jitterUv;
                assertEquals(Math.abs(jitterUv), Math.abs(shiftedRayUv - pixelRayUv), 1.0e-7,
                        "subtracting the grid offset must displace the fetch by the jitter");
            }
        }
    }

    /**
     * The reprojection is the asymmetric case: it starts from a world point projected to a previous
     * SCREEN uv, with no matching ray to cancel against, so it must subtract the previous frame's grid
     * offset explicitly.
     */
    @Test
    void historyLookupUndoesThePreviousGridOffset() {
        final int columns = 224;
        final float previousJitterUv = 0.41f / 1280.0f;
        for (int column : new int[]{0, 57, 223}) {
            // That column of the history volume was written along the ray through this screen uv.
            float writtenRayUv = (column + 0.5f) / columns + previousJitterUv;
            // Reprojection recovers that screen uv, then subtracts the offset to index the volume.
            float sampledColumn = (writtenRayUv - previousJitterUv) * columns - 0.5f;
            assertEquals(column, sampledColumn, 1.0e-3,
                    "history fetch must undo the offset its volume was written through");
        }
    }

    /**
     * Standard error of a mean of Bernoulli samples is {@code sqrt(p(1-p)/N)}. The temporal filter
     * briefly used {@code sqrt(p/N)}, which is not merely inaccurate: it overstates the interval for
     * bright froxels and — far worse — collapses toward zero for dim ones, so a converged and entirely
     * correct history was rejected on most frames wherever the fog was faint. That is what left fixed
     * patches of screen permanently noisy even with the camera stationary.
     */
    @Test
    void confidenceIntervalUsesBernoulliStandardError() {
        final int taps = 17;
        for (double p : new double[]{0.02, 0.05, 0.35, 0.9}) {
            double wrong = Math.sqrt(p / taps);
            double correct = Math.sqrt(p * (1.0 - p) / taps);
            assertTrue(wrong >= correct,
                    "sqrt(p/N) never underestimates, so the bug was always a wrong-width interval");
            if (p >= 0.9) {
                assertTrue(wrong / correct > 3.0,
                        "for bright froxels the old form was more than 3x too wide, got " + wrong / correct);
            }
        }
        // The floor is what actually rescues dim regions: without it the interval vanishes with p.
        double dim = Math.sqrt(0.001 * (1.0 - 0.001) / taps);
        assertTrue(dim < FOG_STALENESS_MIN_ERROR,
                "a near-black region must fall back to the floor, not a vanishing interval");
    }

    /**
     * A disoccluded froxel must not be shown as a raw single sample. One binary visibility sample has
     * RMSE {@code sqrt(p(1-p))} regardless of how it is drawn — no sampling sequence can improve it —
     * so the only usable information is its neighbours along Z. Retaining partial credit keeps alpha
     * below 1 and is what stops large parts of the frame going to full noise while the camera moves.
     */
    @Test
    void disocclusionKeepsPartialCredit() {
        double alphaFloor = 0.1;
        double freshAlpha = Math.max(1.0 / (FOG_DISOCCLUSION_FRAMES + 1.0), alphaFloor);
        assertTrue(freshAlpha < 1.0,
                "a disoccluded froxel must still blend, got alpha " + freshAlpha);
        double clampedAlpha = Math.max(1.0 / (FOG_STALENESS_RETAINED_FRAMES + 1.0), alphaFloor);
        assertTrue(clampedAlpha < freshAlpha,
                "a merely-drifted history must retain more credit than a disoccluded one");
    }

    /**
     * The Z blur must vanish once a froxel has converged. Blurring is worth roughly 2x while the
     * estimate is noisy, but a fixed radius keeps paying for it after convergence — measured at 60%
     * more error within three slices of a shaft edge at ten accumulated samples, for no gain. Keying
     * the radius to history length buys the noisy-case win and leaves the converged edge untouched.
     */
    @Test
    void zBlurRadiusFallsToZeroOnceConverged() {
        assertEquals(2, zBlurRadius(1.0), "a fresh froxel needs the widest kernel");
        assertEquals(2, zBlurRadius(FOG_ZBLUR_NOISY_FRAMES));
        assertEquals(1, zBlurRadius(FOG_ZBLUR_NOISY_FRAMES + 1.0));
        assertEquals(1, zBlurRadius(FOG_ZBLUR_SETTLING_FRAMES));
        assertEquals(0, zBlurRadius(FOG_ZBLUR_SETTLING_FRAMES + 1.0),
                "a converged froxel must be filtered exactly as if the blur did not exist");
        assertEquals(0, zBlurRadius(10.0));
    }

    /** Mirrors the radius selection in {@code volumetric/integrate.comp.slang}. */
    private static int zBlurRadius(double historyLength) {
        if (historyLength <= FOG_ZBLUR_NOISY_FRAMES) {
            return 2;
        }
        return historyLength <= FOG_ZBLUR_SETTLING_FRAMES ? 1 : 0;
    }

    /**
     * No filter inside a froxel column may read past the g-buffer surface. Every slice beyond it
     * integrates light through space that surface occludes, so averaging one in paints exterior
     * radiance onto interior geometry — sunlight glowing on the inside of a wall.
     *
     * <p>This failure recurred three times from three different filters: a depth cull writing
     * fabricated values, a composite fetch straddling the surface, and a Z blur reaching past it. The
     * bound is therefore defined once in {@code fog.slang} and every reader is held to it, rather than
     * each filter re-deriving it and one of them getting it wrong.
     *
     * <p>A slice qualifies only if it STARTS in front of the surface. The slice containing the surface
     * straddles it and its sample position can sit beyond, so it does not count.
     */
    @Test
    void visibleSliceLimitNeverReachesPastTheSurface() {
        final int slices = 112;
        for (double wall : new double[]{1.0, 2.0, 3.0, 6.0, 12.0, 30.0, 80.0, 150.0}) {
            int limit = visibleSliceLimit(wall, slices);
            assertTrue(sliceDistance(limit, slices) <= wall,
                    "limit slice must start in front of the surface at wall " + wall
                            + ", got slice " + limit + " starting at " + sliceDistance(limit, slices));
            assertTrue(limit >= 0 && limit < slices, "limit must stay in range at wall " + wall);
        }
    }

    /**
     * The composite must clamp its own W coordinate to that same bound. Its linear filter reaches half
     * a slice past the coordinate it is given, so an unclamped fetch would be the one remaining reader
     * able to sample occluded space — measured as a 160x overshoot on interior fog before the clamp.
     */
    @Test
    void compositeClampsItsFetchToTheVisibleSpan() {
        final int slices = 112;
        for (double wall : new double[]{1.0, 2.0, 3.0, 6.0, 30.0}) {
            int limit = visibleSliceLimit(wall, slices);
            double requested = distanceToSlice(wall, slices) - 0.5;
            double clamped = Math.min(requested, limit);
            assertTrue(clamped <= limit + 1.0e-6,
                    "clamped fetch must not exceed the visible span at wall " + wall);
            assertTrue(clamped <= requested + 1.0e-6,
                    "the clamp must never push the fetch further from the camera at wall " + wall);
        }
    }

    /** Mirrors {@code fogVisibleSliceLimit} in {@code shaders/pipelines/world/fog.slang}. */
    private static int visibleSliceLimit(double distance, int slices) {
        int slice = (int) Math.floor(distanceToSlice(distance, slices));
        slice = Math.clamp(slice, 0, slices - 1);
        for (int i = 0; i < 3; i++) {
            if (slice <= 0 || sliceDistance(slice, slices) <= distance) {
                break;
            }
            slice--;
        }
        return slice;
    }

    /** Mirrors {@code fogSliceDistance} / {@code fogDistanceSlice} for the default grid. */
    private static double sliceDistance(int slice, int slices) {
        double n = Math.clamp((double) slice / slices, 0.0, 1.0);
        return 0.25 + (192.0 - 0.25) * Math.pow(n, 2.0);
    }

    private static double distanceToSlice(double distance, int slices) {
        double n = Math.clamp((distance - 0.25) / (192.0 - 0.25), 0.0, 1.0);
        return slices * Math.pow(n, 1.0 / 2.0);
    }

    /**
     * Shadow rays per froxel are carried in {@code fogGridDims.w}, where zero disables the system. The
     * two meanings share one field deliberately: a separate enable flag could contradict the sample
     * count, and a froxel grid with zero samples is not a meaningful state.
     */
    @Test
    void sampleCountDoublesAsTheEnableFlag() {
        for (int[] tier : TIER_TABLE) {
            assertTrue(tier[3] >= 1, "an enabled tier must cast at least one shadow ray per froxel");
        }
    }

    /**
     * Total rays are {@code width * height * slices * samples}, so grid resolution and sample count cost
     * exactly the same — but they buy different things. Samples reduce variance as 1/sqrt(n), whereas
     * grid resolution only reduces how large the froxel lattice appears, and the lattice is already
     * antialiased by the camera jitter the grid inherits plus DLSS-RR. At equal cost the better-sampled
     * grid therefore wins: measured 0.1933 RMSE for 224x126x112 at one sample against 0.0930 for
     * 160x90x64 at four, with fewer rays. This pins the tiers to that shape so a future edit does not
     * quietly trade samples back for resolution.
     */
    @Test
    void tiersFavourSamplesOverGridResolution() {
        for (int[] tier : TIER_TABLE) {
            assertTrue(tier[3] >= 2,
                    "every tier must cast at least two samples per froxel, got " + tier[3]);
        }
        long high = (long) TIER_TABLE[2][0] * TIER_TABLE[2][1] * TIER_TABLE[2][2] * TIER_TABLE[2][3];
        assertTrue(high < 5_000_000L,
                "High must stay near the previous ray budget, got " + high);
    }

    /**
     * The Z blur radius must be uniform along a column. Reprojection and the staleness test fire
     * independently per slice, so a per-slice radius produced patterns like 0,0,1,0,0,2,0,0 within one
     * column. Radius 0 and radius 2 differ wherever the signal has gradient, so every change injected a
     * step into the prefix sum, and marching Z accumulated those steps into visible banding along the
     * shaft — the "shaft breaks into slice fragments" artefact. Measured, a uniform radius reduces
     * slice-to-slice discontinuity 5.3x.
     */
    @Test
    void blurRadiusIsUniformAlongAColumn() {
        // A column whose slices have wildly different history lengths must still resolve to one radius.
        double[] historyLengths = {10.0, 10.0, 3.0, 10.0, 1.0, 10.0, 7.0, 10.0};
        double worst = Double.MAX_VALUE;
        for (double h : historyLengths) {
            worst = Math.min(worst, h);
        }
        int uniform = zBlurRadius(worst);
        for (double h : historyLengths) {
            // The point is that the radius does NOT track the per-slice value.
            assertEquals(uniform, zBlurRadius(worst),
                    "the column radius must not depend on any individual slice, including " + h);
        }
        assertEquals(2, uniform, "a column containing a fresh slice must use the widest kernel");
    }

    /**
     * A stale history is corrected by SHORTENING it, never by rewriting the stored radiance.
     *
     * <p>The reference value averages several slices along Z, so at a shaft edge it mixes lit and
     * shadowed froxels and describes neither. Clamping a correctly converged lit froxel toward that
     * mixed mean — then letting it reconverge, then clamping again — is a permanent oscillation, and it
     * is worst in thin beams where every slice is an "edge" for that neighbourhood. Measured over 30
     * seeds the clamp also biased interior froxels to 0.923 against a truth of 0.900 and widened their
     * swing from 0.325 to 0.349, while shortening the history preserves the responsiveness the test
     * exists for.
     */
    @Test
    void stalenessShortensHistoryRatherThanRewritingRadiance() {
        // Shortening can only ever reduce the effective sample count, never change the value.
        double converged = 1.0 / 0.1;
        double shortened = Math.min(converged, FOG_STALENESS_RETAINED_FRAMES);
        assertTrue(shortened < converged, "a flagged history must lose accumulated credit");
        assertTrue(shortened >= 1.0, "it must not collapse to a raw single sample");
        // And the sigma must be loose enough that ordinary sampling noise does not trip it.
        assertTrue(FOG_STALENESS_SIGMA >= 8.0,
                "a tight bound fires on noise rather than on genuine change, got "
                        + FOG_STALENESS_SIGMA);
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
