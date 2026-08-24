package dev.comfyfluffy.caustica;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

final class CausticaConfigTest {
    @Test
    void invalidPeakNitsFallsBackToDefault() {
        CausticaConfig.IntSetting setting = CausticaConfig.Rt.Hdr.PEAK_NITS;
        int previous = setting.value();
        try {
            setting.set(2000);
            assertEquals(2000, setting.value());

            setting.set(900);
            assertEquals(1000, setting.value());
        } finally {
            setting.set(previous);
        }
    }

    @Test
    void fogHistoryWeightStaysInsideTheResponsiveRange() {
        CausticaConfig.FloatSetting setting = CausticaConfig.Rt.Volumetrics.TEMPORAL_WEIGHT;
        float previous = setting.value();
        try {
            setting.set(0.75f);
            assertEquals(0.75f, setting.value(), 1.0e-6f);

            // A config file still holding the old 0.95 must not restore an average of about 39 frames,
            // which delayed lighting changes and smeared the volume by resampling history every frame.
            setting.set(0.95f);
            assertEquals(0.90f, setting.value(), 1.0e-6f);

            setting.set(-1.0f);
            assertEquals(0.0f, setting.value(), 1.0e-6f);
        } finally {
            setting.set(previous);
        }
    }
}
