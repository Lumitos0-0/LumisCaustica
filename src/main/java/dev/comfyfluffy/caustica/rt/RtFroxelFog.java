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
    private static final Int4 NEAR_GRID = new Int4(64, 48, 64, 0);
    private static final Int4 FAR_GRID = new Int4(48, 48, 48, 0);
    private static final int LIGHT_CELLS = cellCount(NEAR_GRID) + cellCount(FAR_GRID);
    private static final int LIGHT_DISPATCH_WIDTH = 256;

    private static final int LIGHT_RAYGEN = 2;
    private static final int INTEGRATE_RAYGEN = 3;
    private static final int COMPOSITE_RAYGEN = 4;

    record Frame(long bufferAddress, Float4 options, Float4 nearOrigin, Float4 farOrigin,
                 Int4 nearGrid, Int4 farGrid, Int4 froxelGrid, float altitudeOffset) {}

    private static final Float4 ZERO = new Float4(0, 0, 0, 0);
    private static final Frame DISABLED =
            new Frame(0L, ZERO, ZERO, ZERO, NEAR_GRID, FAR_GRID, new Int4(0, 0, 0, 0), 0);

    private RtBuffer scratch;

    Frame prepare(RtContext ctx, int width, int height, boolean airOverworld, double camX, double camY,
                  double camZ, int rebaseX, int rebaseY, int rebaseZ) {
        float density = Fog.DENSITY.value();
        if (!airOverworld || !Fog.ENABLED.value() || density == 0f) {
            return DISABLED;
        }

        int quality = Fog.QUALITY.value();
        int tile = 10 - 3 * quality;
        Int4 froxels = new Int4(Math.min(384, Math.ceilDiv(width, tile)), Math.min(216, Math.ceilDiv(height, tile)),
                40 + 16 * quality, 0);
        // Match the shader's scratch layout: RGB light, integrated froxels, then per-pixel float depth.
        long bytes = 16L * (LIGHT_CELLS + cellCount(froxels)) + 4L * width * height;
        if (scratch == null || scratch.size < bytes) {
            if (scratch != null) {
                ctx.waitIdle();
                destroy();
            }
            scratch = ctx.createBuffer(bytes, VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, "froxel fog scratch");
        }

        return new Frame(scratch.deviceAddress,
                new Float4(density, Fog.RANGE.value(), Fog.ANISOTROPY.value(), Fog.HEIGHT_FALLOFF.value()),
                gridOrigin(camX, camY, camZ, rebaseX, rebaseY, rebaseZ, NEAR_GRID, 0.75f),
                gridOrigin(camX, camY, camZ, rebaseX, rebaseY, rebaseZ, FAR_GRID, 4f),
                NEAR_GRID, FAR_GRID, froxels, rebaseY - 64f);
    }

    void record(RtContext ctx, RtPipeline pipeline, VkCommandBuffer cmd, MemoryStack stack,
                ByteBuffer push, Int4 froxels, int width, int height) {
        try (RtDebugLabels.Scope label = RtDebugLabels.scope(ctx, cmd, "fog world-space visibility");
             RtFrameStats.Scope stage = RtFrameStats.FRAME.stage("frame.fogLight")) {
            pipeline.trace(cmd, LIGHT_DISPATCH_WIDTH, Math.ceilDiv(LIGHT_CELLS, LIGHT_DISPATCH_WIDTH), push, LIGHT_RAYGEN);
        }
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
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

    private static Float4 gridOrigin(double x, double y, double z, int rebaseX, int rebaseY, int rebaseZ,
                                     Int4 grid, float spacing) {
        return new Float4(gridStart(x, rebaseX, grid.x(), spacing), gridStart(y, rebaseY, grid.y(), spacing),
                gridStart(z, rebaseZ, grid.z(), spacing), spacing);
    }

    private static float gridStart(double camera, int rebase, int cells, float spacing) {
        // Snap in absolute world space before rebasing to keep overlapping cells invariant across chunk boundaries.
        return (float) ((Math.floor(camera / spacing) - cells / 2) * spacing - rebase);
    }

    private static int cellCount(Int4 grid) {
        return grid.x() * grid.y() * grid.z();
    }
}
