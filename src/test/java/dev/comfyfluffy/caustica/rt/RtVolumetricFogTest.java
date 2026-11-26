package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class RtVolumetricFogTest {
    @Test
    void gridBudgetIsIndependentOfDisplayResolution() {
        for (int quality = 0; quality < 3; quality++) {
            var hd = RtVolumetricFog.Grid.fit(1920, 1080, quality);
            var uhd = RtVolumetricFog.Grid.fit(3840, 2160, quality);
            assertEquals(hd, uhd);
            long cells = (long) hd.columns() * hd.rows() * hd.slices();
            long lastDepthEnd = cells * (8L + 8L + 16L) + 4L * hd.columns() * hd.rows() + 4L * 1920 * 1080;
            assertEquals(lastDepthEnd, hd.bytes(1920, 1080));
        }
    }

    @Test
    void gridRetainsAspectAndSupportsTinyTargets() {
        var small = RtVolumetricFog.Grid.fit(1, 1, 0);
        assertEquals(1, small.columns());
        assertEquals(1, small.rows());
        var wide = RtVolumetricFog.Grid.fit(5120, 1440, 1);
        assertTrue(Math.abs(5120.0 / 1440.0 - (double) wide.columns() / wide.rows()) < 0.05);
        assertTrue(wide.columns() <= 192 && wide.rows() <= 108);
    }
}
