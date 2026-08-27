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
 * left to the {@code -Dcaustica.*} startup surface. DLSS-RR and froxel quality are explicit exceptions:
 * their images/features are recreated at the next safe frame boundary when the selected preset changes.
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
            fogQuality(),
            fogDensity(),
            fogDistance(),
            fogDepthDistribution(),
            fogAlbedo(),
            fogHeightOffset(),
            fogHeightFalloff(),
            fogNoiseAmount(),
            fogNoiseScale(),
            fogAmbientStrength(),
            lightShaftStrength(),
            lightShaftFocus(),
            localFogStrength(),
            localFogFocus(),
            localFogCandidates(),
            localFogSamples(),
            localFogClamp(),
            fogFilterEdges(),
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

    private static OptionInstance<Integer> fogQuality() {
        IntSetting setting = CausticaConfig.Rt.Volumetrics.QUALITY;
        return new OptionInstance<>(
            "caustica.options.rt.fogQuality",
            tooltip("caustica.options.rt.fogQuality"),
            (caption, value) -> Options.genericValueLabel(caption,
                    Component.translatable("caustica.options.rt.fogQuality." + value)),
            new OptionInstance.IntRange(0, 2),
            Math.clamp(setting.value(), 0, 2),
            setting::set);
    }

    private static OptionInstance<Integer> fogDensity() {
        FloatSetting setting = CausticaConfig.Rt.Volumetrics.EXTINCTION;
        return new OptionInstance<>(
            "caustica.options.rt.fogDensity", tooltip("caustica.options.rt.fogDensity"),
            (caption, value) -> Options.genericValueLabel(caption,
                    Component.literal(String.format(Locale.ROOT, "%.4f / block", value / 10_000.0f))),
            new OptionInstance.IntRange(0, 500),
            Math.clamp(Math.round(setting.value() * 10_000.0f), 0, 500),
            value -> setting.set(value / 10_000.0f));
    }

    private static OptionInstance<Integer> fogDistance() {
        FloatSetting setting = CausticaConfig.Rt.Volumetrics.MAX_DISTANCE;
        return new OptionInstance<>(
            "caustica.options.rt.fogDistance", tooltip("caustica.options.rt.fogDistance"),
            (caption, blocks) -> Options.genericValueLabel(caption, Component.literal(blocks + " blocks")),
            new OptionInstance.IntRange(16, 512),
            Math.clamp(Math.round(setting.value()), 16, 512),
            blocks -> setting.set(blocks.floatValue()));
    }

    private static OptionInstance<Integer> fogDepthDistribution() {
        FloatSetting setting = CausticaConfig.Rt.Volumetrics.DEPTH_EXPONENT;
        return decimalHundredths("caustica.options.rt.fogDepthDistribution", setting, 100, 400);
    }

    private static OptionInstance<Integer> fogAlbedo() {
        return percent("caustica.options.rt.fogAlbedo",
                CausticaConfig.Rt.Volumetrics.SINGLE_SCATTERING_ALBEDO, 0, 100);
    }

    private static OptionInstance<Integer> fogHeightOffset() {
        FloatSetting setting = CausticaConfig.Rt.Volumetrics.HEIGHT_OFFSET;
        return new OptionInstance<>(
            "caustica.options.rt.fogHeightOffset", tooltip("caustica.options.rt.fogHeightOffset"),
            (caption, blocks) -> Options.genericValueLabel(caption, Component.literal(blocks + " blocks")),
            new OptionInstance.IntRange(-64, 128),
            Math.clamp(Math.round(setting.value()), -64, 128),
            blocks -> setting.set(blocks.floatValue()));
    }

    private static OptionInstance<Integer> fogHeightFalloff() {
        FloatSetting setting = CausticaConfig.Rt.Volumetrics.HEIGHT_FALLOFF;
        return new OptionInstance<>(
            "caustica.options.rt.fogHeightFalloff", tooltip("caustica.options.rt.fogHeightFalloff"),
            (caption, thousandths) -> Options.genericValueLabel(caption,
                    Component.literal(String.format(Locale.ROOT, "%.3f / block", thousandths / 1000.0f))),
            new OptionInstance.IntRange(0, 250),
            Math.clamp(Math.round(setting.value() * 1000.0f), 0, 250),
            thousandths -> setting.set(thousandths / 1000.0f));
    }

    private static OptionInstance<Integer> fogNoiseAmount() {
        return percent("caustica.options.rt.fogNoiseAmount",
                CausticaConfig.Rt.Volumetrics.NOISE_AMOUNT, 0, 100);
    }

    private static OptionInstance<Integer> fogNoiseScale() {
        FloatSetting setting = CausticaConfig.Rt.Volumetrics.NOISE_SCALE;
        return new OptionInstance<>(
            "caustica.options.rt.fogNoiseScale", tooltip("caustica.options.rt.fogNoiseScale"),
            (caption, thousandths) -> Options.genericValueLabel(caption,
                    Component.literal(String.format(Locale.ROOT, "%.3f / block", thousandths / 1000.0f))),
            new OptionInstance.IntRange(1, 1000),
            Math.clamp(Math.round(setting.value() * 1000.0f), 1, 1000),
            thousandths -> setting.set(thousandths / 1000.0f));
    }

    private static OptionInstance<Integer> fogAmbientStrength() {
        return percent("caustica.options.rt.fogAmbientStrength",
                CausticaConfig.Rt.Volumetrics.AMBIENT_STRENGTH, 0, 400);
    }

    private static OptionInstance<Integer> lightShaftStrength() {
        return percent("caustica.options.rt.lightShaftStrength",
                CausticaConfig.Rt.Volumetrics.DIRECTIONAL_STRENGTH, 0, 400);
    }

    private static OptionInstance<Integer> lightShaftFocus() {
        return signedPercent("caustica.options.rt.lightShaftFocus",
                CausticaConfig.Rt.Volumetrics.DIRECTIONAL_FOCUS);
    }

    private static OptionInstance<Integer> localFogStrength() {
        return percent("caustica.options.rt.localFogStrength",
                CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_STRENGTH, 0, 800);
    }

    private static OptionInstance<Integer> localFogFocus() {
        return signedPercent("caustica.options.rt.localFogFocus",
                CausticaConfig.Rt.Volumetrics.ANISOTROPY);
    }

    private static OptionInstance<Integer> localFogCandidates() {
        IntSetting setting = CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_CANDIDATES;
        return integer("caustica.options.rt.localFogCandidates", setting, 0, 16);
    }

    private static OptionInstance<Integer> localFogSamples() {
        IntSetting setting = CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_SAMPLES;
        return integer("caustica.options.rt.localFogSamples", setting, 1, 4);
    }

    private static OptionInstance<Integer> localFogClamp() {
        FloatSetting setting = CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_CLAMP;
        return new OptionInstance<>(
            "caustica.options.rt.localFogClamp", tooltip("caustica.options.rt.localFogClamp"),
            (caption, value) -> Options.genericValueLabel(caption, value),
            new OptionInstance.IntRange(1, 512),
            Math.clamp(Math.round(setting.value()), 1, 512),
            value -> setting.set(value.floatValue()));
    }

    private static OptionInstance<Integer> fogFilterEdges() {
        FloatSetting setting = CausticaConfig.Rt.Volumetrics.FILTER_EDGE_SHARPNESS;
        return new OptionInstance<>(
            "caustica.options.rt.fogFilterEdges", tooltip("caustica.options.rt.fogFilterEdges"),
            (caption, value) -> Options.genericValueLabel(caption, value),
            new OptionInstance.IntRange(0, 32),
            Math.clamp(Math.round(setting.value()), 0, 32),
            value -> setting.set(value.floatValue()));
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

    private static OptionInstance<Integer> percent(String key, FloatSetting setting, int min, int max) {
        return new OptionInstance<>(
            key, tooltip(key),
            (caption, value) -> Options.genericValueLabel(caption, Component.literal(value + "%")),
            new OptionInstance.IntRange(min, max),
            Math.clamp(Math.round(setting.value() * 100.0f), min, max),
            value -> setting.set(value / 100.0f));
    }

    private static OptionInstance<Integer> signedPercent(String key, FloatSetting setting) {
        return new OptionInstance<>(
            key, tooltip(key),
            (caption, value) -> Options.genericValueLabel(caption, Component.literal(value + "%")),
            new OptionInstance.IntRange(-90, 90),
            Math.clamp(Math.round(setting.value() * 100.0f), -90, 90),
            value -> setting.set(value / 100.0f));
    }

    private static OptionInstance<Integer> decimalHundredths(
            String key, FloatSetting setting, int min, int max) {
        return new OptionInstance<>(
            key, tooltip(key),
            (caption, value) -> Options.genericValueLabel(caption,
                    Component.literal(String.format(Locale.ROOT, "%.2f", value / 100.0f))),
            new OptionInstance.IntRange(min, max),
            Math.clamp(Math.round(setting.value() * 100.0f), min, max),
            value -> setting.set(value / 100.0f));
    }

    private static OptionInstance<Integer> integer(String key, IntSetting setting, int min, int max) {
        return new OptionInstance<>(
            key, tooltip(key),
            (caption, value) -> Options.genericValueLabel(caption, value),
            new OptionInstance.IntRange(min, max),
            Math.clamp(setting.value(), min, max),
            setting::set);
    }

    private static <T> OptionInstance.TooltipSupplier<T> tooltip(String key) {
        return OptionInstance.cachedConstantTooltip(Component.translatable(key + ".tooltip"));
    }

    private static OptionInstance<Boolean> bool(String captionKey, BooleanSetting setting) {
        return OptionInstance.createBoolean(
            captionKey,
            OptionInstance.cachedConstantTooltip(Component.translatable(captionKey + ".tooltip")),
            setting.value(),
            setting::set);
    }
}
