package dev.comfyfluffy.caustica.rt.cache;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.RtFrameStats;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData.Float4;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData.Int4;
import dev.comfyfluffy.caustica.rt.pipeline.RtPipeline;
import java.nio.ByteBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;

/**
 * Owns the 3D World Radiance Cache (WRC) volume and the ray-generation pass that updates it.
 * The cache maintains a persistent world-anchored irradiance field around the player, providing
 * O(1) multi-scattered ambient and emissive lookups for both volumetrics and secondary path tracing.
 */
public final class RtWorldCache {
    public static final int WRC_UPDATE_RAYGEN = 5;

    private RtImage radianceVolume;
    private int gridDim = 48;
    private long lastRecordedFrame = Long.MIN_VALUE;
    private double lastOriginX = Double.NaN;
    private double lastOriginY = Double.NaN;
    private double lastOriginZ = Double.NaN;

    public record FrameData(Float4 origin, Int4 dims, boolean enabled, long frameIndex) {
    }

    public boolean matches() {
        int wantedDim = CausticaConfig.Rt.Wrc.GRID_DIM.value();
        return radianceVolume != null && gridDim == wantedDim;
    }

    public void ensure(RtContext ctx) {
        int wantedDim = CausticaConfig.Rt.Wrc.GRID_DIM.value();
        if (radianceVolume != null && gridDim == wantedDim) {
            return;
        }
        destroy();
        gridDim = wantedDim;
        String extent = gridDim + "x" + gridDim + "x" + gridDim;
        radianceVolume = ctx.createStorageVolume(gridDim, gridDim, gridDim,
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "world radiance cache " + extent);
        invalidateHistory();
    }

    public void bindAll(RtPipeline pipeline) {
        requireReady();
        pipeline.setWorldRadianceCache(radianceVolume.view);
    }

    public FrameData prepareFrame(long frameIndex, double camX, double camY, double camZ,
                                  int rebaseX, int rebaseY, int rebaseZ) {
        requireReady();
        boolean enabled = CausticaConfig.Rt.Wrc.ENABLED.value() && CausticaConfig.Rt.Wrc.MODE.value() > 0;
        float cellSize = CausticaConfig.Rt.Wrc.CELL_SIZE.value();
        float halfSpan = (gridDim * cellSize) * 0.5f;

        // World anchor snapped to cell size for stability
        double worldOriginX = Math.floor((camX - halfSpan) / cellSize) * cellSize;
        double worldOriginY = Math.floor((camY - halfSpan) / cellSize) * cellSize;
        double worldOriginZ = Math.floor((camZ - halfSpan) / cellSize) * cellSize;

        // Rebased coordinates for shader use
        float rebasedOriginX = (float) (worldOriginX - rebaseX);
        float rebasedOriginY = (float) (worldOriginY - rebaseY);
        float rebasedOriginZ = (float) (worldOriginZ - rebaseZ);

        double originDelta2 = (worldOriginX - lastOriginX) * (worldOriginX - lastOriginX)
                + (worldOriginY - lastOriginY) * (worldOriginY - lastOriginY)
                + (worldOriginZ - lastOriginZ) * (worldOriginZ - lastOriginZ);

        boolean historyValid = enabled && lastRecordedFrame == frameIndex - 1
                && originDelta2 < (cellSize * cellSize * 4.0);

        int flags = enabled ? 1 : 0;
        if (historyValid) {
            flags |= 2;
        }

        lastOriginX = worldOriginX;
        lastOriginY = worldOriginY;
        lastOriginZ = worldOriginZ;

        return new FrameData(
                new Float4(rebasedOriginX, rebasedOriginY, rebasedOriginZ, cellSize),
                new Int4(gridDim, gridDim, gridDim, flags),
                enabled,
                frameIndex);
    }

    public void record(RtContext ctx, VkCommandBuffer cmd, RtPipeline pipeline,
                       ByteBuffer pushConstants, FrameData frame) {
        if (!frame.enabled()) {
            return;
        }
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "world radiance cache update");
             RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.wrcUpdate")) {
            pipeline.trace(cmd, gridDim, gridDim, gridDim, pushConstants, WRC_UPDATE_RAYGEN);
            barrier(cmd);
        }
        lastRecordedFrame = frame.frameIndex();
    }

    public void invalidateHistory() {
        lastRecordedFrame = Long.MIN_VALUE;
        lastOriginX = Double.NaN;
        lastOriginY = Double.NaN;
        lastOriginZ = Double.NaN;
    }

    public void destroy() {
        if (radianceVolume != null) {
            radianceVolume.destroy();
            radianceVolume = null;
        }
        invalidateHistory();
    }

    public RtImage radianceVolume() {
        return radianceVolume;
    }

    private void requireReady() {
        if (radianceVolume == null) {
            throw new IllegalStateException("World Radiance Cache volume has not been created");
        }
    }

    private static void barrier(VkCommandBuffer cmd) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
    }
}
