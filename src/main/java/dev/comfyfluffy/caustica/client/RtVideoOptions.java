package dev.comfyfluffy.caustica.client;

import com.mojang.serialization.Codec;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaConfig.BooleanSetting;
import dev.comfyfluffy.caustica.CausticaConfig.FloatSetting;
import dev.comfyfluffy.caustica.CausticaConfig.IntSetting;
import dev.comfyfluffy.caustica.CausticaConfig.StringSetting;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import net.minecraft.client.Minecraft;
import net.minecraft.client.OptionInstance;
import net.minecraft.client.Options;
import net.minecraft.network.chat.Component;

/**
 * Builds the {@link OptionInstance} widgets shown in the RT section of the vanilla Video Settings screen
 * (injected by {@code VideoSettingsScreenMixin}). Each option is bound straight to a {@link CausticaConfig}
 * runtime setting: the initial value is read from the current config, and the value-update listener writes
 * back through {@code set(...)} so changes take effect on the next frame.
 *
 * <p>Only settings the renderer re-reads per-frame are exposed here — toggles that would require a device or
 * buffer-pool rebuild (worker threads, OMM, max-entity capacities, PBR material flags) are intentionally
 * left to the {@code -Dcaustica.*} startup surface. DLSS-RR quality is the exception: the render resolution
 * is queried from NGX for the chosen quality mode on every resize (see
 * {@code RtDlssRr.queryOptimalRenderSize}), and the RR feature itself is recreated live whenever
 * {@code quality} changes (see {@code RtDlssRr.ensureFeature}), so it is safe to expose here.
 */
public final class RtVideoOptions {
    private RtVideoOptions() {
    }

    /**
     * Runtime-tunable RT options, in display order. Paired two-per-row by {@code OptionsList.addSmall}.
     * The HDR entries are omitted entirely (not just disabled) when this session's swapchain isn't
     * PQ-capable ({@code CausticaConfig.Rt.Hdr.swapchainPqAvailable()}) — offering a toggle/sliders that
     * can never do anything is worse than not showing them, and unlike most settings here this one is
     * fixed by hardware/OS/compositor at surface-creation time. The current swapchain may still be native
     * SDR; changing the toggle invalidates its configuration and recreates it in the selected format.
     */
    public static OptionInstance<?>[] runtimeOptions() {
        List<OptionInstance<?>> options = new ArrayList<>(List.of(
            exposureMode(),
            manualEv(),
            gamma(),
            spp(),
            maxBounces(),
            entities(),
            particles(),
            waterWaves(),
            volumetricFog(),
            fogDensity(),
            fogHeight(),
            fogHeightFalloff(),
            fogMaxDistance(),
            fogAnisotropy(),
            fogAlbedo(),
            fogNoiseAmount(),
            fogNoiseScale(),
            lightShaftStrength(),
            lightShaftFocus(),
            fogTemporalWeight(),
            underwaterFog(),
            underwaterCaustics(),
            fogUnderwaterDistance(),
            fogQuality(),
            dlssQuality()
        ));
        if (CausticaConfig.Rt.Hdr.swapchainPqAvailable()) {
            options.add(hdrEnabled());
            options.add(hdrUiBrightness());
            options.add(hdrPeak());
        }
        options.add(debugView());
        return options.toArray(OptionInstance<?>[]::new);
    }

    private static OptionInstance<String> exposureMode() {
        StringSetting setting = CausticaConfig.Rt.Exposure.MODE;
        return new OptionInstance<>(
            "caustica.options.rt.exposureMode",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.exposureMode.tooltip")),
            // CycleButton (used for Enum values) already prepends "caption: " itself (DisplayState.
            // NAME_AND_VALUE), so this must return only the value's text, not caption + value again.
            (caption, value) -> Component.translatable("caustica.options.rt.exposureMode." + value),
            new OptionInstance.Enum<>(List.of("auto", "manual"), Codec.STRING),
            setting.get(),
            setting::set);
    }

    private static OptionInstance<Integer> manualEv() {
        FloatSetting setting = CausticaConfig.Rt.Exposure.MANUAL_EV;
        return new OptionInstance<>(
            "caustica.options.rt.manualEv",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.manualEv.tooltip")),
            (caption, tenths) -> {
                float ev = tenths / 10.0f;
                String sign = ev > 0.0f ? "+" : "";
                return Options.genericValueLabel(caption,
                        Component.literal(sign + String.format(Locale.ROOT, "%.1f EV", ev)));
            },
            new OptionInstance.IntRange(-150, 150),
            Math.clamp(Math.round(setting.value() * 10.0f), -150, 150),
            tenths -> setting.set(tenths / 10.0f));
    }

    private static OptionInstance<Integer> gamma() {
        FloatSetting setting = CausticaConfig.Rt.Tonemap.GAMMA;
        return new OptionInstance<>(
            "caustica.options.rt.gamma",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.gamma.tooltip")),
            (caption, hundredths) -> Options.genericValueLabel(caption,
                    Component.literal(String.format(Locale.ROOT, "%.2f", hundredths / 100.0f))),
            new OptionInstance.IntRange(50, 150),
            Math.clamp(Math.round(setting.value() * 100.0f), 50, 150),
            hundredths -> setting.set(hundredths / 100.0f));
    }

    private static OptionInstance<Integer> spp() {
        IntSetting setting = CausticaConfig.Rt.Composite.SPP;
        return new OptionInstance<>(
            "caustica.options.rt.spp",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.spp.tooltip")),
            (caption, value) -> Options.genericValueLabel(caption, value),
            new OptionInstance.IntRange(1, 8),
            Math.clamp(setting.value(), 1, 8),
            setting::set);
    }

    private static OptionInstance<Integer> maxBounces() {
        IntSetting setting = CausticaConfig.Rt.Composite.MAX_BOUNCES;
        return new OptionInstance<>(
            "caustica.options.rt.maxBounces",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.maxBounces.tooltip")),
            (caption, value) -> Options.genericValueLabel(caption, value),
            new OptionInstance.IntRange(2, 8),
            Math.clamp(setting.value(), 2, 8),
            setting::set);
    }

    private static OptionInstance<Boolean> entities() {
        return bool("caustica.options.rt.entities", CausticaConfig.Rt.Entities.ENABLED);
    }

    private static OptionInstance<Boolean> particles() {
        return bool("caustica.options.rt.particles", CausticaConfig.Rt.Entities.PARTICLES_ENABLED);
    }

    private static OptionInstance<Boolean> waterWaves() {
        return bool("caustica.options.rt.waterWaves", CausticaConfig.Rt.Composite.WATER_WAVES);
    }

    private static OptionInstance<Boolean> volumetricFog() {
        return bool("caustica.options.rt.volumetricFog", CausticaConfig.Rt.Volumetrics.ENABLED);
    }

    private static OptionInstance<Integer> fogDensity() {
        FloatSetting setting = CausticaConfig.Rt.Volumetrics.EXTINCTION;
        return new OptionInstance<>(
            "caustica.options.rt.fogDensity",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.fogDensity.tooltip")),
            (caption, value) -> Options.genericValueLabel(caption,
                    Component.literal(value + "%")),
            new OptionInstance.IntRange(0, 100),
            Math.clamp(Math.round(setting.value() * 10_000.0f), 0, 100),
            value -> setting.set(value / 10_000.0f));
    }

    // Fog medium shaping. Every setting below is read per frame by RtVolumetrics.prepareFrame and is
    // part of the optical signature, so changing one invalidates froxel history rather than ghosting.

    /** Altitude of the fog layer's base, as an offset from sea level in blocks. */
    private static OptionInstance<Integer> fogHeight() {
        return floatSlider("caustica.options.rt.fogHeight",
                CausticaConfig.Rt.Volumetrics.HEIGHT_OFFSET, -64, 128, 1.0f, "");
    }

    /** Exponential decay rate of fog density with altitude, x1000. 0 spreads fog evenly at all heights. */
    private static OptionInstance<Integer> fogHeightFalloff() {
        return floatSlider("caustica.options.rt.fogHeightFalloff",
                CausticaConfig.Rt.Volumetrics.HEIGHT_FALLOFF, 0, 120, 1000.0f, "");
    }

    /** How far the froxel volume reaches in blocks. Beyond this the fog is not simulated. */
    private static OptionInstance<Integer> fogMaxDistance() {
        return floatSlider("caustica.options.rt.fogMaxDistance",
                CausticaConfig.Rt.Volumetrics.MAX_DISTANCE, 8, 2048, 1.0f, "");
    }

    /** Scattering phase anisotropy of the air medium, x100. Negative values scatter backward. */
    private static OptionInstance<Integer> fogAnisotropy() {
        return floatSlider("caustica.options.rt.fogAnisotropy",
                CausticaConfig.Rt.Volumetrics.ANISOTROPY, -95, 95, 100.0f, "");
    }

    /** Fraction of light the medium scatters rather than absorbs. Below 100% the fog reads darker. */
    private static OptionInstance<Integer> fogAlbedo() {
        return floatSlider("caustica.options.rt.fogAlbedo",
                CausticaConfig.Rt.Volumetrics.SINGLE_SCATTERING_ALBEDO, 0, 100, 100.0f, "%");
    }

    /** Strength of the density variation that breaks fog into wisps. */
    private static OptionInstance<Integer> fogNoiseAmount() {
        return floatSlider("caustica.options.rt.fogNoiseAmount",
                CausticaConfig.Rt.Volumetrics.NOISE_AMOUNT, 0, 100, 100.0f, "%");
    }

    /** Spatial frequency of the fog noise field, x1000. Higher values give finer, busier wisps. */
    private static OptionInstance<Integer> fogNoiseScale() {
        return floatSlider("caustica.options.rt.fogNoiseScale",
                CausticaConfig.Rt.Volumetrics.NOISE_SCALE, 1, 200, 1000.0f, "");
    }

    /**
     * How much of the previous frame's resolved fog carries over. Higher is smoother but slower to
     * react; Fog Quality caps this, so the top of the range may be unreachable on High and above.
     */
    private static OptionInstance<Integer> fogTemporalWeight() {
        return floatSlider("caustica.options.rt.fogTemporalWeight",
                CausticaConfig.Rt.Volumetrics.TEMPORAL_WEIGHT, 0, 99, 100.0f, "%");
    }

    /** How far the froxel volume reaches while the camera is submerged, in blocks. */
    private static OptionInstance<Integer> fogUnderwaterDistance() {
        return floatSlider("caustica.options.rt.fogUnderwaterDistance",
                CausticaConfig.Rt.Volumetrics.UNDERWATER_MAX_DISTANCE, 8, 256, 1.0f, "");
    }

    private static OptionInstance<Integer> lightShaftStrength() {
        FloatSetting setting = CausticaConfig.Rt.Volumetrics.DIRECTIONAL_STRENGTH;
        return new OptionInstance<>(
            "caustica.options.rt.lightShaftStrength",
            OptionInstance.cachedConstantTooltip(
                    Component.translatable("caustica.options.rt.lightShaftStrength.tooltip")),
            (caption, percent) -> Options.genericValueLabel(caption,
                    Component.literal(percent + "%")),
            new OptionInstance.IntRange(0, 400),
            Math.clamp(Math.round(setting.value() * 100.0f), 0, 400),
            percent -> setting.set(percent / 100.0f));
    }

    private static OptionInstance<Integer> lightShaftFocus() {
        FloatSetting setting = CausticaConfig.Rt.Volumetrics.DIRECTIONAL_FOCUS;
        return new OptionInstance<>(
            "caustica.options.rt.lightShaftFocus",
            OptionInstance.cachedConstantTooltip(
                    Component.translatable("caustica.options.rt.lightShaftFocus.tooltip")),
            (caption, percent) -> Options.genericValueLabel(caption,
                    Component.literal(percent + "%")),
            new OptionInstance.IntRange(0, 90),
            Math.clamp(Math.round(setting.value() * 100.0f), 0, 90),
            percent -> setting.set(percent / 100.0f));
    }

    private static OptionInstance<Integer> underwaterFog() {
        FloatSetting setting = CausticaConfig.Rt.Volumetrics.UNDERWATER_SCATTERING;
        return new OptionInstance<>(
            "caustica.options.rt.underwaterFog",
            OptionInstance.cachedConstantTooltip(
                    Component.translatable("caustica.options.rt.underwaterFog.tooltip")),
            (caption, percent) -> Options.genericValueLabel(caption,
                    Component.literal(percent + "%")),
            new OptionInstance.IntRange(0, 150),
            Math.clamp(Math.round(setting.value() * 1000.0f), 0, 150),
            percent -> setting.set(percent / 1000.0f));
    }

    private static OptionInstance<Integer> underwaterCaustics() {
        FloatSetting setting = CausticaConfig.Rt.Volumetrics.UNDERWATER_CAUSTIC_STRENGTH;
        return new OptionInstance<>(
            "caustica.options.rt.underwaterCaustics",
            OptionInstance.cachedConstantTooltip(
                    Component.translatable("caustica.options.rt.underwaterCaustics.tooltip")),
            (caption, percent) -> Options.genericValueLabel(caption,
                    Component.literal(percent + "%")),
            new OptionInstance.IntRange(0, 300),
            Math.clamp(Math.round(setting.value() * 100.0f), 0, 300),
            percent -> setting.set(percent / 100.0f));
    }

    private static OptionInstance<Integer> fogQuality() {
        IntSetting setting = CausticaConfig.Rt.Volumetrics.QUALITY;
        return new OptionInstance<>(
            "caustica.options.rt.fogQuality",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.fogQuality.tooltip")),
            (caption, value) -> Options.genericValueLabel(caption,
                    Component.translatable("caustica.options.rt.fogQuality." + value)),
            new OptionInstance.IntRange(0, 4),
            Math.clamp(setting.value(), 0, 4),
            setting::set);
    }

    private static OptionInstance<Integer> dlssQuality() {
        IntSetting setting = CausticaConfig.Rt.DlssRr.QUALITY;
        List<Integer> steps = CausticaConfig.Rt.DlssRr.QUALITY_STEPS;
        int initialQuality = steps.contains(setting.value()) ? setting.value() : 0;
        int initialPosition = steps.indexOf(initialQuality);
        return new OptionInstance<>(
            "caustica.options.rt.dlssQuality",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.dlssQuality.tooltip")),
            (caption, position) -> Options.genericValueLabel(caption,
                    Component.translatable("caustica.options.rt.dlssQuality." + steps.get(position))),
            new OptionInstance.IntRange(0, steps.size() - 1),
            initialPosition,
            position -> setting.set(steps.get(position)));
    }

    private static OptionInstance<Boolean> hdrEnabled() {
        BooleanSetting setting = CausticaConfig.Rt.Hdr.ENABLED;
        return OptionInstance.createBoolean(
            "caustica.options.rt.hdr",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.hdr.tooltip")),
            setting.value(),
            enabled -> {
                if (setting.value() != enabled) {
                    setting.set(enabled);
                    // Reuse the framebuffer-resize path at the next safe frame boundary. GpuSurface
                    // refuses configure() while an image is acquired, so doing it directly here is unsafe.
                    Minecraft.getInstance().invalidateSurfaceConfiguration();
                }
            });
    }

    private static OptionInstance<Integer> hdrUiBrightness() {
        FloatSetting setting = CausticaConfig.Rt.Hdr.UI_NITS;
        return new OptionInstance<>(
            "caustica.options.rt.hdrUiBrightness",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.hdrUiBrightness.tooltip")),
            (caption, nits) -> Options.genericValueLabel(caption, Component.literal(nits + " nits")),
            new OptionInstance.IntRange(80, 500),
            Math.clamp(Math.round(setting.value()), 80, 500),
            nits -> setting.set(nits.floatValue()));
    }

    // Each step selects a baked ACES HDR mastering target. Changes take effect on the next frame.
    private static OptionInstance<Integer> hdrPeak() {
        IntSetting setting = CausticaConfig.Rt.Hdr.PEAK_NITS;
        List<Integer> steps = CausticaConfig.Rt.Hdr.PEAK_NITS_STEPS;
        int initialPeak = steps.contains(setting.value()) ? setting.value() : 1000;
        int initialPosition = steps.indexOf(initialPeak);
        return new OptionInstance<>(
            "caustica.options.rt.hdrPeak",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.hdrPeak.tooltip")),
            (caption, position) -> Options.genericValueLabel(caption, Component.literal(steps.get(position) + " nits")),
            new OptionInstance.IntRange(0, steps.size() - 1),
            Math.max(initialPosition, 0),
            position -> setting.set(steps.get(position)));
    }

    private static OptionInstance<Integer> debugView() {
        IntSetting setting = CausticaConfig.Rt.Composite.DEBUG_VIEW;
        return new OptionInstance<>(
            "caustica.options.rt.debugView",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.debugView.tooltip")),
            // CycleButton (used for Enum values) already prepends "caption: " itself (DisplayState.
            // NAME_AND_VALUE), so this must return only the value's text, not caption + value again.
            (caption, value) -> Component.translatable("caustica.options.rt.debugView." + value),
            new OptionInstance.Enum<>(List.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), Codec.INT),
            Math.clamp(setting.value(), 0, 9),
            setting::set);
    }

    private static OptionInstance<Boolean> bool(String captionKey, BooleanSetting setting) {
        return OptionInstance.createBoolean(
            captionKey,
            OptionInstance.cachedConstantTooltip(Component.translatable(captionKey + ".tooltip")),
            setting.value(),
            setting::set);
    }

    /** Integer slider over an integer setting, shown as the raw value. */
    private static OptionInstance<Integer> intSlider(String captionKey, IntSetting setting,
            int min, int max) {
        return new OptionInstance<>(
            captionKey,
            OptionInstance.cachedConstantTooltip(Component.translatable(captionKey + ".tooltip")),
            (caption, value) -> Options.genericValueLabel(caption, value),
            new OptionInstance.IntRange(min, max),
            Math.clamp(setting.value(), min, max),
            setting::set);
    }

    /**
     * Integer slider over a float setting. Widget position {@code p} maps to {@code p / scale}, so
     * {@code scale} sets both the precision and the displayed magnitude. {@code unit} is appended to the
     * number: "%" for a percentage, "" for a bare value.
     *
     * <p>The range is intentionally a useful subset of what the backing setting clamps to — the full
     * range stays reachable from the TOML, and a slider spanning every legal value would leave the
     * interesting ones in the first pixel.
     */
    private static OptionInstance<Integer> floatSlider(String captionKey, FloatSetting setting,
            int min, int max, float scale, String unit) {
        return new OptionInstance<>(
            captionKey,
            OptionInstance.cachedConstantTooltip(Component.translatable(captionKey + ".tooltip")),
            (caption, value) -> Options.genericValueLabel(caption,
                    Component.literal(unit.isEmpty() ? String.valueOf(value) : value + unit)),
            new OptionInstance.IntRange(min, max),
            Math.clamp(Math.round(setting.value() * scale), min, max),
            value -> setting.set(value / scale));
    }
}
