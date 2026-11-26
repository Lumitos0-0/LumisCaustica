package dev.comfyfluffy.caustica.rt;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.CausticaConfig.Rt.Fog;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData.Float4;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData.Int4;
import dev.comfyfluffy.caustica.rt.pipeline.RtPipeline;
import java.nio.ByteBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;

final class RtVolumetricFog {
    record Grid(int columns, int rows, int slices) {
        static Grid fit(int width, int height, int quality) {
            int maxColumns = 128 + quality * 64;
            int maxRows = maxColumns * 9 / 16;
            double tile = Math.max(9 - 2 * quality,
                    Math.max((double) width / maxColumns, (double) height / maxRows));
            return new Grid(Math.clamp((int) Math.ceil(width / tile), 1, maxColumns),
                    Math.clamp((int) Math.ceil(height / tile), 1, maxRows), 40 + quality * 16);
        }

        long bytes(int width, int height) {
            long columns2d = (long) columns * rows;
            return 32L * columns2d * slices + 4L * (columns2d + (long) width * height);
        }
    }

    record Frame(long address, Float4 options, Int4 grid, Float4 temporal, Float4 sun, Float4 moon) {}
    private static final Float4 ZERO = new Float4(0f, 0f, 0f, 0f);
    private static final Frame OFF = new Frame(0L, ZERO, new Int4(0,0,0,0), ZERO, ZERO, ZERO);

    private record History(Object level, int width, int height, Grid grid, boolean submerged,
                           float range, float waterDensity, float focus, boolean sun, boolean moon) {}

    private static final class Tint {
        private String hex;
        private float r, g, b;

        Float4 withGain(String value, float gain) {
            if (!value.equals(hex)) {
                int rgb = Integer.parseInt(value, 16);
                r = channel(rgb >> 16);
                g = channel(rgb >> 8);
                b = channel(rgb);
                hex = value;
            }
            return new Float4(r, g, b, gain);
        }

        private static float channel(int rgb) {
            float s = (rgb & 255) / 255f;
            return s <= 0.04045f ? s / 12.92f : (float) Math.pow((s + 0.055f) / 1.055f, 2.4);
        }
    }

    private final Tint sunTint = new Tint(), moonTint = new Tint();
    private RtBuffer scratch;
    private History history;
    private long submittedSerial = Long.MIN_VALUE, lastTime;
    private Float4 previousSky, previousWater;

    Frame prepare(RtContext ctx, int width, int height, Object overworld, int flags, int altitudeOffset,
                  long serial, float cameraMoveSquared, Float4 celestial, Float4 water) {
        boolean submerged = (flags & 1) != 0;
        float density = submerged ? Fog.WATER_DENSITY.value() : Fog.DENSITY.value();
        if (overworld == null || !Fog.ENABLED.value() || density <= 0f) {
            invalidate();
            return OFF;
        }

        Grid grid = Grid.fit(width, height, Fog.QUALITY.value());
        long bytes = grid.bytes(width, height);
        if (scratch == null || scratch.size < bytes) {
            if (scratch != null) {
                ctx.waitIdle();
                destroy();
            }
            scratch = ctx.createBuffer(bytes, VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, "volumetric media");
        }

        float range = submerged ? Fog.WATER_RANGE.value() : Fog.RANGE.value();
        float focus = submerged && (flags & 16) != 0 ? Fog.WAVE_FOCUS.value() : 0f;
        Float4 sun = sunTint.withGain(Fog.SUN_TINT.get(), Fog.SUN_STRENGTH.value());
        Float4 moon = moonTint.withGain(Fog.MOON_TINT.get(), Fog.MOON_STRENGTH.value());
        History next = new History(overworld, width, height, grid, submerged, range,
                submerged ? density : 0f, focus, sun.w() > 0, moon.w() > 0);
        long now = System.nanoTime();
        float elapsed = (now - lastTime) * 1.0e-9f;
        float reuse = 0f;
        if (submittedSerial == serial - 1 && next.equals(history) && elapsed > 0f && elapsed <= 0.25f
                && cameraMoveSquared <= 16f
                && Math.cos(previousSky.x() - celestial.x()) > 0.99995
                && Math.cos(previousSky.y() - celestial.y()) > 0.99995
                && (!submerged || (water.w() >= previousWater.w()
                        && Math.abs(water.x() - previousWater.x()) < 0.08f
                        && Math.abs(water.y() - previousWater.y()) < 0.08f
                        && Math.abs(water.z() - previousWater.z()) < 0.08f))) {
            float weight = Fog.HISTORY.value();
            if (submerged) weight = Math.min(weight, 0.45f);
            reuse = (float) Math.pow(weight, Math.max(0.35f, elapsed * 60f));
        }
        history = next;
        previousSky = celestial;
        previousWater = water;
        lastTime = now;
        return new Frame(scratch.deviceAddress,
                new Float4(density, range, submerged ? Fog.WATER_PHASE.value() : Fog.ANISOTROPY.value(),
                        submerged ? 0f : Fog.HEIGHT_FALLOFF.value()),
                new Int4(grid.columns, grid.rows, grid.slices, (int) (serial & 1)),
                new Float4(reuse, focus, altitudeOffset, elapsed < 0.05f ? 1f : 0f), sun, moon);
    }

    void submitted(long serial) {
        if (history != null) submittedSerial = serial;
    }

    void invalidate() {
        history = null;
        submittedSerial = Long.MIN_VALUE;
    }

    void record(RtContext ctx, RtPipeline pipeline, VkCommandBuffer cmd, MemoryStack stack,
                ByteBuffer push, Int4 grid, int width, int height) {
        pass(ctx, pipeline, cmd, stack, push, 2, grid.x(), grid.y(), "frame.volumeBounds");
        pass(ctx, pipeline, cmd, stack, push, 3, grid.x(), grid.y() * grid.z(), "frame.volumeLighting");
        pass(ctx, pipeline, cmd, stack, push, 4, grid.x(), grid.y(), "frame.volumeIntegration");
        pass(ctx, pipeline, cmd, stack, push, 5, width, height, "frame.volumeResolve");
    }

    private static void pass(RtContext ctx, RtPipeline pipeline, VkCommandBuffer cmd, MemoryStack stack,
                             ByteBuffer push, int index, int width, int height, String name) {
        try (RtDebugLabels.Scope label = RtDebugLabels.scope(ctx, cmd, name);
             RtFrameStats.Scope stage = RtFrameStats.FRAME.stage(name)) {
            pipeline.trace(cmd, width, height, push, index);
        }
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
    }

    void destroy() {
        invalidate();
        if (scratch != null) {
            scratch.destroy();
            scratch = null;
        }
    }
}
