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
    void wantedGridUsesConfiguredResolution() {
        CausticaConfig.IntSetting pixelSize = CausticaConfig.Rt.Volumetrics.GRID_PIXEL_SIZE;
        CausticaConfig.IntSetting slices = CausticaConfig.Rt.Volumetrics.DEPTH_SLICES;
        int oldPixelSize = pixelSize.value();
        int oldSlices = slices.value();
        try {
            pixelSize.set(16);
            slices.set(48);
            RtFroxelGrid grid = RtVolumetrics.wantedGrid(1920, 1080);
            assertEquals(120, grid.width());
            assertEquals(68, grid.height());
            assertEquals(48, grid.depth());
        } finally {
            pixelSize.set(oldPixelSize);
            slices.set(oldSlices);
        }
    }

    @Test
    void candidatesPreservesExplicitDisable() {
        CausticaConfig.IntSetting candidates = CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_CANDIDATES;
        int oldCandidates = candidates.value();
        try {
            candidates.set(4);
            assertEquals(4, RtVolumetrics.effectiveLocalLightCandidates());
            candidates.set(0);
            assertEquals(0, RtVolumetrics.effectiveLocalLightCandidates());
        } finally {
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
