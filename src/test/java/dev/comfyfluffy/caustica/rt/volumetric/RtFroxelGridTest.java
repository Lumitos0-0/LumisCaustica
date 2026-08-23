package dev.comfyfluffy.caustica.rt.volumetric;

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
