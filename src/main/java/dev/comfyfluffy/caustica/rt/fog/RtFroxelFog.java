package dev.comfyfluffy.caustica.rt.fog;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtGpuExecutor;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.FogIntegratePushData;
import dev.comfyfluffy.caustica.rt.gen.FogLightingPushData;
import dev.comfyfluffy.caustica.rt.pipeline.RtFroxelFogPipeline;
import dev.comfyfluffy.caustica.rt.pipeline.RtSkyLut;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;

/**
 * Lifecycle and frame-graph owner for the camera-view hybrid fog volume.
 *
 * <p>The lighting fields are packed 2D atlases ({@code x = froxel column, y = zSlice * gridY + row})
 * rather than true 3D images. This keeps the implementation compatible with the existing storage-image
 * allocator and sampled-image path while retaining logarithmic-Z froxel semantics. Direct and GI fields
 * share that domain, but only GI owns a temporal history. The density/extinction model is evaluated once
 * by the integration pass and is therefore common to both lighting contributions.
 *
 * <p>This is intentionally a primary-view system. Its GI rays are a low-rate radiance-cache probe for the
 * visible frustum; arbitrary secondary path rays do not consult this camera volume. World-space medium
 * transport remains a separate future extension.
 */
public final class RtFroxelFog {
    // Keep the packed atlas within Vulkan's minimum guaranteed maxImageDimension2D. Devices with larger
    // limits still get the same predictable shape; the Z budget is reduced instead of failing allocation.
    private static final int MAX_ATLAS_HEIGHT = 4096;
    private static final int HDR_FORMAT = VK10.VK_FORMAT_R16G16B16A16_SFLOAT;

    private RtFroxelFogPipeline pipeline;
    private RtImage directVolume;
    private RtImage giVolume;
    private RtImage giHistory;
    private int gridX = -1;
    private int gridY = -1;
    private int gridZ = -1;
    private int atlasWidth = -1;
    private int atlasHeight = -1;
    private boolean historyValid;
    private boolean destroyed;
    private boolean failed;
    private boolean activeLastFrame;

    public boolean enabled() {
        return !failed && CausticaConfig.Rt.Fog.ENABLED.value();
    }

    public boolean ready() {
        return pipeline != null && directVolume != null && giVolume != null && giHistory != null;
    }

    public int gridX() {
        return gridX;
    }

    public int gridY() {
        return gridY;
    }

    public int gridZ() {
        return gridZ;
    }

    public RtImage directVolume() {
        return directVolume;
    }

    public RtImage giVolume() {
        return giVolume;
    }

    public RtImage giHistory() {
        return giHistory;
    }

    /** Invalidate the soft GI cache after a camera cut, resource reset, or material epoch change. */
    public void invalidateHistory() {
        historyValid = false;
    }

    /**
     * Allocate the froxel atlases for the current RT render resolution. Reallocation is rare and waits for
     * the device only when a live fog setting changed the atlas shape; ordinary frames do not pay that cost.
     */
    public void ensureResources(RtContext ctx, int renderWidth, int renderHeight) {
        if (!enabled() || CausticaConfig.Rt.Fog.DENSITY.value() <= 0.0f) {
            activeLastFrame = false;
            return;
        }
        destroyed = false;
        if (!activeLastFrame) {
            historyValid = false;
        }
        activeLastFrame = true;
        int tile = CausticaConfig.Rt.Fog.FROXEL_TILE_SIZE.value();
        int wantedX = Math.max(1, (renderWidth + tile - 1) / tile);
        int wantedY = Math.max(1, (renderHeight + tile - 1) / tile);
        int requestedZ = CausticaConfig.Rt.Fog.Z_SLICES.value();
        // Vulkan's minimum guaranteed 2D max height is 4096, while most desktop devices expose 16384.
        // Keep the requested logarithmic depth where possible and reduce it rather than rejecting a 4K
        // render on a conservative implementation.
        int maxZ = Math.max(1, MAX_ATLAS_HEIGHT / wantedY);
        int wantedZ = Math.max(1, Math.min(requestedZ, maxZ));
        int wantedHeight = Math.multiplyExact(wantedY, wantedZ);
        if (ready() && wantedX == gridX && wantedY == gridY && wantedZ == gridZ) {
            return;
        }

        try {
            if (ready() || directVolume != null || giVolume != null || giHistory != null) {
                // The caller normally already waited for a resize. This path also handles a live tile/Z setting
                // change from the video screen, where the previous frame may still reference the old images.
                ctx.waitIdle();
                destroyImages();
            }
            if (pipeline == null) {
                pipeline = RtFroxelFogPipeline.create(ctx);
            }
            directVolume = ctx.createStorageImage(wantedX, wantedHeight, HDR_FORMAT,
                    "fog direct froxel atlas " + wantedX + "x" + wantedHeight);
            giVolume = ctx.createStorageImage(wantedX, wantedHeight, HDR_FORMAT,
                    "fog GI froxel atlas " + wantedX + "x" + wantedHeight);
            giHistory = ctx.createStorageImage(wantedX, wantedHeight, HDR_FORMAT,
                    "fog GI history atlas " + wantedX + "x" + wantedHeight);
            gridX = wantedX;
            gridY = wantedY;
            gridZ = wantedZ;
            atlasWidth = wantedX;
            atlasHeight = wantedHeight;
            historyValid = false;
        } catch (Throwable t) {
            if (pipeline != null) {
                pipeline.destroy();
                pipeline = null;
            }
            destroyImages();
            failed = true;
            CausticaMod.LOGGER.error("Primary-view fog unavailable; continuing without fog", t);
        }
    }

    /**
     * Record direct/GI froxel lighting, shared-medium integration, and the GI history copy. The caller
     * places this after the primary + indirect world traces and before DLSS-RR, so the lighting pass can
     * sample the finished pre-exposed scene while the integration remains at render resolution.
     */
    public void record(VkCommandBuffer cmd, long tlas, RtImage scene, RtImage depth,
                       long worldPushAddr, long lightBufAddr, int lightCount, RtSkyLut skyLut,
                       float baseHeight, long frameIndex,
                       RtGpuExecutor.GraphicsUse graphicsUse,
                       RtGpuExecutor.GraphicsUseWaiter graphicsUseWaiter) {
        if (!enabled() || !ready() || CausticaConfig.Rt.Fog.DENSITY.value() <= 0.0f) {
            return;
        }
        float maxDistance = CausticaConfig.Rt.Fog.MAX_DISTANCE.value();
        float historyBlend = historyValid
                ? CausticaConfig.Rt.Fog.GI_TEMPORAL_BLEND.value() : 0.0f;
        pipeline.bindFrame(tlas, scene, depth, directVolume, giVolume, giHistory,
                skyLut.skyViewView(), skyLut.transmittanceView(), skyLut.sampler(),
                graphicsUse, graphicsUseWaiter);

        FogLightingPushData.Float4 grid = new FogLightingPushData.Float4(
                gridX, gridY, gridZ, maxDistance);
        FogLightingPushData.Float4 settings = new FogLightingPushData.Float4(
                CausticaConfig.Rt.Fog.DIRECT_ANISOTROPY.value(),
                CausticaConfig.Rt.Fog.LOCAL_STRENGTH.value(),
                historyBlend,
                CausticaConfig.Rt.Fog.GI_SAMPLES.value());
        pipeline.dispatchLighting(cmd, atlasWidth, atlasHeight,
                new FogLightingPushData(worldPushAddr, lightBufAddr, grid, settings,
                        lightCount, (int) frameIndex));

        try (MemoryStack stack = MemoryStack.stackPush()) {
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }

        FogIntegratePushData.Float4 integrateGrid = new FogIntegratePushData.Float4(
                gridX, gridY, gridZ, maxDistance);
        FogIntegratePushData.Float4 medium = new FogIntegratePushData.Float4(
                CausticaConfig.Rt.Fog.DENSITY.value(),
                CausticaConfig.Rt.Fog.HEIGHT_FALLOFF.value(),
                baseHeight,
                0.5f);
        FogIntegratePushData.Float4 lighting = new FogIntegratePushData.Float4(
                CausticaConfig.Rt.Fog.DIRECT_STRENGTH.value(),
                CausticaConfig.Rt.Fog.GI_STRENGTH.value(),
                CausticaConfig.Rt.Fog.SCATTERING_STRENGTH.value(),
                0.0f);
        FogIntegratePushData.Float4 scatteringColor = new FogIntegratePushData.Float4(
                CausticaConfig.Rt.Fog.SCATTER_R.value(),
                CausticaConfig.Rt.Fog.SCATTER_G.value(),
                CausticaConfig.Rt.Fog.SCATTER_B.value(),
                0.0f);
        pipeline.dispatchIntegrate(cmd, scene.width, scene.height,
                new FogIntegratePushData(worldPushAddr, integrateGrid, medium,
                        lighting, scatteringColor));

        try (MemoryStack stack = MemoryStack.stackPush()) {
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
            pipeline.copyHistory(cmd, giVolume, giHistory);
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
        historyValid = true;
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        // Drop descriptor sets/pipelines before their image handles, matching Vulkan object lifetime
        // requirements even though the caller normally has already waited for the device.
        if (pipeline != null) {
            pipeline.destroy();
            pipeline = null;
        }
        destroyImages();
        activeLastFrame = false;
        destroyed = true;
    }

    private void destroyImages() {
        if (directVolume != null) {
            directVolume.destroy();
            directVolume = null;
        }
        if (giVolume != null) {
            giVolume.destroy();
            giVolume = null;
        }
        if (giHistory != null) {
            giHistory.destroy();
            giHistory = null;
        }
        gridX = -1;
        gridY = -1;
        gridZ = -1;
        atlasWidth = -1;
        atlasHeight = -1;
        historyValid = false;
    }
}
