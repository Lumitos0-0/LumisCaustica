package dev.comfyfluffy.caustica.rt.volumetric;

import dev.comfyfluffy.caustica.CausticaConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class RtFroxelGridTest {
    @Test
    void roundsScreenTilesUpAtEdges() {
        RtFroxelGrid grid = RtFroxelGrid.forRenderSize(1921, 1081, 16, 48);

        assertEquals(121, grid.width());
        assertEquals(68, grid.height());
        assertEquals(48, grid.depth());
        assertEquals(394_944L, grid.cellCount());
    }

    @Test
    void nonlinearDepthMappingRoundTripsEveryBoundary() {
        RtFroxelGrid grid = RtFroxelGrid.forRenderSize(1280, 720, 16, 48);
        float previous = -1.0f;
        for (int boundary = 0; boundary <= grid.depth(); boundary++) {
            float distance = grid.boundaryDistance(boundary, 192.0f, 2.0f);
            assertTrue(distance >= previous);
            assertEquals(boundary, grid.sliceForDistance(distance, 192.0f, 2.0f), 1.0e-4f);
            previous = distance;
        }
        assertEquals(192.0f, previous, 1.0e-5f);
    }

    @Test
    void qualityPresetsScaleTheBalancedGrid() {
        CausticaConfig.IntSetting quality = CausticaConfig.Rt.Volumetrics.QUALITY;
        CausticaConfig.IntSetting pixelSize = CausticaConfig.Rt.Volumetrics.GRID_PIXEL_SIZE;
        CausticaConfig.IntSetting slices = CausticaConfig.Rt.Volumetrics.DEPTH_SLICES;
        int oldQuality = quality.value();
        int oldPixelSize = pixelSize.value();
        int oldSlices = slices.value();
        try {
            pixelSize.set(16);
            slices.set(48);
            int[][] expected = {
                    {96, 54, 40},
                    {120, 68, 48},
                    {160, 90, 64},
                    {240, 135, 80},
                    {320, 180, 96}
            };
            for (int preset = 0; preset < expected.length; preset++) {
                quality.set(preset);
                RtFroxelGrid grid = RtVolumetrics.wantedGrid(1920, 1080);
                assertEquals(expected[preset][0], grid.width());
                assertEquals(expected[preset][1], grid.height());
                assertEquals(expected[preset][2], grid.depth());
            }
        } finally {
            quality.set(oldQuality);
            pixelSize.set(oldPixelSize);
            slices.set(oldSlices);
        }
    }

    @Test
    void qualityPresetsIncreaseEmitterProposalsButPreserveExplicitDisable() {
        CausticaConfig.IntSetting quality = CausticaConfig.Rt.Volumetrics.QUALITY;
        CausticaConfig.IntSetting candidates = CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_CANDIDATES;
        int oldQuality = quality.value();
        int oldCandidates = candidates.value();
        try {
            candidates.set(2);
            int[] expected = {2, 4, 6, 8, 8};
            for (int preset = 0; preset < expected.length; preset++) {
                quality.set(preset);
                assertEquals(expected[preset], RtVolumetrics.effectiveLocalLightCandidates());
            }
            candidates.set(0);
            assertEquals(0, RtVolumetrics.effectiveLocalLightCandidates());
        } finally {
            quality.set(oldQuality);
            candidates.set(oldCandidates);
        }
    }

    @Test
    void rejectsInvalidGridAndDepthParameters() {
        assertThrows(IllegalArgumentException.class,
                () -> RtFroxelGrid.forRenderSize(0, 720, 16, 48));
        RtFroxelGrid grid = RtFroxelGrid.forRenderSize(1280, 720, 16, 48);
        assertThrows(IllegalArgumentException.class,
                () -> grid.boundaryDistance(49, 192.0f, 2.0f));
        assertThrows(IllegalArgumentException.class,
                () -> grid.sliceForDistance(1.0f, 0.0f, 2.0f));
    }
}
