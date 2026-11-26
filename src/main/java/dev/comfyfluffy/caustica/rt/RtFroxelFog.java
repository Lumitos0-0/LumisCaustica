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

final class RtFroxelFog {
    private static final int INTEGRATE_RAYGEN = 2;
    private static final int COMPOSITE_RAYGEN = 3;

    record Frame(long bufferAddress, Float4 options, Int4 froxelGrid, float altitudeOffset) {}

    private static final Float4 ZERO = new Float4(0, 0, 0, 0);
    private static final Frame DISABLED =
            new Frame(0L, ZERO, new Int4(0, 0, 0, 0), 0);

    private RtBuffer scratch;

    Frame prepare(RtContext ctx, int width, int height, boolean airOverworld, int rebaseY) {
        float density = Fog.DENSITY.value();
        if (!airOverworld || !Fog.ENABLED.value() || density == 0f) {
            return DISABLED;
        }

        int quality = Fog.QUALITY.value();
        int tile = 8 - 2 * quality;
        // Bound the number of shadow-ray columns even at 4K and on ultrawide displays.
        int maxColumns = 160 + 80 * quality;
        int maxRows = maxColumns * 9 / 16;
        double stride = Math.max(tile, Math.max((double) width / maxColumns, (double) height / maxRows));
        Int4 froxels = new Int4((int) Math.ceil(width / stride), (int) Math.ceil(height / stride),
                40 + 24 * quality, 0);
        // One integrated RGBA value per column, followed by first-hit distances at render resolution.
        long bytes = 16L * froxels.x() * froxels.y() + 4L * width * height;
        if (scratch == null || scratch.size < bytes) {
            if (scratch != null) {
                ctx.waitIdle();
                destroy();
            }
            scratch = ctx.createBuffer(bytes, VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, "froxel fog scratch");
        }

        return new Frame(scratch.deviceAddress,
                new Float4(density, Fog.RANGE.value(), Fog.ANISOTROPY.value(), Fog.HEIGHT_FALLOFF.value()),
                froxels, rebaseY - 64f);
    }

    void record(RtContext ctx, RtPipeline pipeline, VkCommandBuffer cmd, MemoryStack stack,
                ByteBuffer push, Int4 froxels, int width, int height) {
        try (RtDebugLabels.Scope label = RtDebugLabels.scope(ctx, cmd, "fog froxel integration");
             RtFrameStats.Scope stage = RtFrameStats.FRAME.stage("frame.fogIntegrate")) {
            pipeline.trace(cmd, froxels.x(), froxels.y(), push, INTEGRATE_RAYGEN);
        }
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
        try (RtDebugLabels.Scope label = RtDebugLabels.scope(ctx, cmd, "fog composite");
             RtFrameStats.Scope stage = RtFrameStats.FRAME.stage("frame.fogComposite")) {
            pipeline.trace(cmd, width, height, push, COMPOSITE_RAYGEN);
        }
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
    }

    void destroy() {
        if (scratch != null) {
            scratch.destroy();
            scratch = null;
        }
    }
}
