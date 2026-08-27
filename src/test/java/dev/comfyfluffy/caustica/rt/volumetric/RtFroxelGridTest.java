package dev.comfyfluffy.caustica.rt.volumetric;

import dev.comfyfluffy.caustica.CausticaConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class RtFroxelGridTest {
    @Test
    void fixedPresetsNeverExceedRequiredMaximum() {
        assertEquals(new RtFroxelGrid.Dimensions(64, 32, 64),
                RtFroxelGrid.dimensions(RtFroxelGrid.Quality.LOW));
        assertEquals(new RtFroxelGrid.Dimensions(96, 48, 96),
                RtFroxelGrid.dimensions(RtFroxelGrid.Quality.MEDIUM));
        assertEquals(new RtFroxelGrid.Dimensions(128, 64, 128),
                RtFroxelGrid.dimensions(RtFroxelGrid.Quality.HIGH));

        for (RtFroxelGrid.Quality quality : RtFroxelGrid.Quality.values()) {
            RtFroxelGrid.Dimensions dimensions = RtFroxelGrid.dimensions(quality);
            assertTrue(dimensions.width() <= RtFroxelGrid.MAX_WIDTH);
            assertTrue(dimensions.height() <= RtFroxelGrid.MAX_HEIGHT);
            assertTrue(dimensions.depth() <= RtFroxelGrid.MAX_DEPTH);
        }
    }

    @Test
    void highPresetHasOneBlockAverageLinearDepthAtDefaultRange() {
        RtFroxelGrid.Dimensions high = RtFroxelGrid.dimensions(RtFroxelGrid.Quality.HIGH);
        assertEquals(1.0f, 128.0f / high.depth(), 0.0f);
        assertEquals(1.0f, CausticaConfig.Rt.Volumetrics.DEPTH_EXPONENT.defaultValue(), 0.0f);
        assertEquals(128.0f, CausticaConfig.Rt.Volumetrics.MAX_DISTANCE.defaultValue(), 0.0f);
    }

    @Test
    void runtimeQualityMapsDirectlyToFixedDimensions() {
        CausticaConfig.IntSetting quality = CausticaConfig.Rt.Volumetrics.QUALITY;
        int previous = quality.value();
        try {
            RtFroxelGrid.Dimensions[] expected = {
                    new RtFroxelGrid.Dimensions(64, 32, 64),
                    new RtFroxelGrid.Dimensions(96, 48, 96),
                    new RtFroxelGrid.Dimensions(128, 64, 128)
            };
            for (int preset = 0; preset < expected.length; ++preset) {
                quality.set(preset);
                assertEquals(expected[preset], RtVolumetrics.wantedDimensions());
            }
        } finally {
            quality.set(previous);
        }
    }

    @Test
    void dimensionsRejectZeroAndValuesAboveMaximum() {
        assertThrows(IllegalArgumentException.class,
                () -> new RtFroxelGrid.Dimensions(0, 32, 64));
        assertThrows(IllegalArgumentException.class,
                () -> new RtFroxelGrid.Dimensions(129, 64, 128));
        assertThrows(IllegalArgumentException.class,
                () -> new RtFroxelGrid.Dimensions(128, 65, 128));
        assertThrows(IllegalArgumentException.class,
                () -> new RtFroxelGrid.Dimensions(128, 64, 129));
    }
}
