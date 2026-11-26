package dev.comfyfluffy.caustica;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

final class CausticaConfigTest {
    @Test
    void solarAndLunarTintAcceptCustomColorsIndependently() {
        var sun = CausticaConfig.Rt.Fog.SUN_TINT;
        var moon = CausticaConfig.Rt.Fog.MOON_TINT;
        String oldSun = sun.get(), oldMoon = moon.get();
        try {
            sun.set(" #FFc080 ");
            moon.set("809AaF");
            assertEquals("ffc080", sun.get());
            assertEquals("809aaf", moon.get());
            moon.set("invalid");
            assertEquals("ffffff", moon.get());
            assertEquals("ffc080", sun.get());
        } finally {
            sun.set(oldSun);
            moon.set(oldMoon);
        }
    }

    @Test
    void waterDensityCanBeDisabledAndRejectsNonFiniteInputs() {
        var density = CausticaConfig.Rt.Fog.WATER_DENSITY;
        float old = density.value();
        try {
            density.set(-1f);
            assertEquals(0f, density.value());
            density.set(Float.NaN);
            assertEquals(density.defaultValue().floatValue(), density.value());
        } finally {
            density.set(old);
        }
    }

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
}
