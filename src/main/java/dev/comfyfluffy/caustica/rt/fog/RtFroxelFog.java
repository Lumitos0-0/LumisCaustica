package dev.comfyfluffy.caustica.rt.fog;

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
import org.lwjgl.vulkan.VkPhysicalDeviceProperties;

/**
 * Owns the primary-view hybrid fog volume. Packed 2D atlases use x for the froxel column and
 * y = zSlice * gridY + row; the integration pass maps the atlas back to logarithmic camera depth.
 * Direct shafts, local diagnostic emitters, GI, and the world-space emitter-reuse field each have a
 * current atlas and an independent history where temporal reconstruction is useful.
 *
 * <p>The system is intentionally limited to the primary camera view. GI rays recover visible surface
 * radiance and sky radiance; the world-space reuse path samples the published emitter hierarchy. Neither
 * path adds participating-medium transport to arbitrary secondary world rays.
 */
public final class RtFroxelFog {
    // Keep the packed atlas below the device's Vulkan 2D-image limit while retaining the configured 32
    // slices for ordinary render targets when the device supports that extent.
    private static final int MAX_ATLAS_HEIGHT = 8192;
    private static final int HDR_FORMAT = VK10.VK_FORMAT_R16G16B16A16_SFLOAT;

    private RtFroxelFogPipeline pipeline;
    private RtImage directVolume;
    private RtImage localVolume;
    private RtImage giVolume;
    private RtImage cacheVolume;
    private RtImage giAuxVolume;
    private RtImage directHistory;
    private RtImage localHistory;
    private RtImage giHistory;
    private RtImage giAuxHistory;
    private RtImage cacheHistory;
    private int gridX = -1;
    private int gridY = -1;
    private int gridZ = -1;
    private int atlasWidth = -1;
    private int atlasHeight = -1;
    private boolean historyValid;
    private float historyFarDistance = -1.0f;
    private float historyBaseHeight = Float.NaN;
    private boolean destroyed;
    private boolean failed;
    private boolean activeLastFrame;
    private boolean settingsInitialized;
    private int settingsSignature;

    public boolean enabled() {
        return !failed && CausticaConfig.Rt.Fog.ENABLED.value();
    }

    public boolean ready() {
        return pipeline != null && directVolume != null && localVolume != null
                && giVolume != null && cacheVolume != null && giAuxVolume != null
                && directHistory != null && localHistory != null && giHistory != null
                && giAuxHistory != null && cacheHistory != null;
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

    /** True when the current allocation can be kept for this render resolution and fog configuration. */
    public boolean matchesShape(RtContext ctx, int renderWidth, int renderHeight) {
        if (!enabled() || CausticaConfig.Rt.Fog.DENSITY.value() <= 0.0f) return !ready();
        int tile = CausticaConfig.Rt.Fog.FROXEL_TILE_SIZE.value();
        int wantedX = Math.max(1, (renderWidth + tile - 1) / tile);
        int wantedY = Math.max(1, (renderHeight + tile - 1) / tile);
        int wantedZ = desiredZSlices(ctx, wantedY);
        return ready() && wantedX == gridX && wantedY == gridY && wantedZ == gridZ;
    }

    public RtImage directVolume() {
        return directVolume;
    }

    public RtImage localVolume() {
        return localVolume;
    }

    public RtImage giVolume() {
        return giVolume;
    }

    public RtImage cacheVolume() {
        return cacheVolume;
    }

    public RtImage giAuxVolume() {
        return giAuxVolume;
    }

    public RtImage giHistory() {
        return giHistory;
    }

    /** Invalidate all camera-volume histories after a scene, camera, light, or resource discontinuity. */
    public void invalidateHistory() {
        historyValid = false;
    }

    /** Allow an explicit render-state reset to retry a transient allocation/compiler failure. */
    public void resetFailureLatch() {
        failed = false;
    }

    /**
     * Allocate the froxel atlases for the current RT render resolution. A shape change destroys the
     * descriptor/pipeline owner before destroying image views, so no live descriptor set can retain a
     * handle that has already been freed.
     */
    public void ensureResources(RtContext ctx, int renderWidth, int renderHeight) {
        int currentSignature = settingsSignature();
        if (settingsInitialized && settingsSignature != currentSignature) {
            // Any temporal-reconstruction setting change invalidates the correspondence assumptions used by
            // the stored fields; the next dispatch must start from current-frame samples.
            historyValid = false;
        }
        settingsInitialized = true;
        settingsSignature = currentSignature;
        if (!enabled() || CausticaConfig.Rt.Fog.DENSITY.value() <= 0.0f) {
            activeLastFrame = false;
            if (pipeline != null || directVolume != null || localVolume != null || giVolume != null
                    || cacheVolume != null || giAuxVolume != null) {
                ctx.waitIdle();
                if (pipeline != null) {
                    pipeline.destroy();
                    pipeline = null;
                }
                destroyImages();
            }
            return;
        }
        destroyed = false;
        if (!activeLastFrame) historyValid = false;
        activeLastFrame = true;
        float maxDistance = CausticaConfig.Rt.Fog.MAX_DISTANCE.value();
        if (Float.floatToIntBits(maxDistance) != Float.floatToIntBits(historyFarDistance)) {
            historyValid = false;
            historyFarDistance = maxDistance;
        }
        int tile = CausticaConfig.Rt.Fog.FROXEL_TILE_SIZE.value();
        int wantedX = Math.max(1, (renderWidth + tile - 1) / tile);
        int wantedY = Math.max(1, (renderHeight + tile - 1) / tile);
        int wantedZ = desiredZSlices(ctx, wantedY);
        int wantedHeight = Math.multiplyExact(wantedY, wantedZ);
        if (ready() && wantedX == gridX && wantedY == gridY && wantedZ == gridZ) return;

        try {
            if (pipeline != null || directVolume != null || localVolume != null || giVolume != null
                    || cacheVolume != null || giAuxVolume != null) {
                ctx.waitIdle();
                // Descriptor sets are owned by the pipeline. Retire them before the image views they reference.
                if (pipeline != null) {
                    pipeline.destroy();
                    pipeline = null;
                }
                destroyImages();
            }
            pipeline = RtFroxelFogPipeline.create(ctx);
            directVolume = createAtlas(ctx, wantedX, wantedHeight, "direct");
            localVolume = createAtlas(ctx, wantedX, wantedHeight, "local diagnostic");
            giVolume = createAtlas(ctx, wantedX, wantedHeight, "GI");
            cacheVolume = createAtlas(ctx, wantedX, wantedHeight, "world emitter reuse");
            giAuxVolume = createAtlas(ctx, wantedX, wantedHeight, "GI guides");
            directHistory = createAtlas(ctx, wantedX, wantedHeight, "direct history");
            localHistory = createAtlas(ctx, wantedX, wantedHeight, "local history");
            giHistory = createAtlas(ctx, wantedX, wantedHeight, "GI history");
            giAuxHistory = createAtlas(ctx, wantedX, wantedHeight, "GI guide history");
            cacheHistory = createAtlas(ctx, wantedX, wantedHeight, "world emitter reuse history");
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

    private static int desiredZSlices(RtContext ctx, int gridY) {
        int deviceMaxDimension;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPhysicalDeviceProperties properties = VkPhysicalDeviceProperties.calloc(stack);
            VK10.vkGetPhysicalDeviceProperties(ctx.vk().getPhysicalDevice(), properties);
            deviceMaxDimension = properties.limits().maxImageDimension2D();
        }
        int atlasHeightLimit = Math.max(1, Math.min(MAX_ATLAS_HEIGHT, deviceMaxDimension));
        int maxZ = Math.max(1, atlasHeightLimit / gridY);
        return Math.max(1, Math.min(CausticaConfig.Rt.Fog.Z_SLICES.value(), maxZ));
    }

    private static int settingsSignature() {
        int hash = 1;
        hash = 31 * hash + Boolean.hashCode(CausticaConfig.Rt.Fog.ENABLED.value());
        hash = 31 * hash + Boolean.hashCode(CausticaConfig.Rt.Fog.DIRECT_ENABLED.value());
        hash = 31 * hash + Boolean.hashCode(CausticaConfig.Rt.Fog.GI_ENABLED.value());
        hash = 31 * hash + Boolean.hashCode(CausticaConfig.Rt.Fog.LOCAL_LIGHTING_ENABLED.value());
        hash = 31 * hash + Boolean.hashCode(CausticaConfig.Rt.Fog.CACHE_REUSE_ENABLED.value());
        hash = 31 * hash + Boolean.hashCode(CausticaConfig.Rt.Fog.HISTORY_ENABLED.value());
        hash = 31 * hash + Boolean.hashCode(CausticaConfig.Rt.Fog.GI_HISTORY_ENABLED.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.DENSITY.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.HEIGHT_FALLOFF.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.DIRECT_ANISOTROPY.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.DIRECT_STRENGTH.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.GI_STRENGTH.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.LOCAL_STRENGTH.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.CACHE_STRENGTH.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.SCATTER_R.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.SCATTER_G.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.SCATTER_B.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.SCATTERING_STRENGTH.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.MAX_DISTANCE.value());
        hash = 31 * hash + CausticaConfig.Rt.Fog.FROXEL_TILE_SIZE.value();
        hash = 31 * hash + CausticaConfig.Rt.Fog.Z_SLICES.value();
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.GI_TEMPORAL_BLEND.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.DIRECT_TEMPORAL_BLEND.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.LOCAL_TEMPORAL_BLEND.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.CACHE_TEMPORAL_BLEND.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.GI_SPATIAL_RADIUS.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.DIRECT_SPATIAL_RADIUS.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.LOCAL_SPATIAL_RADIUS.value());
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.CACHE_SPATIAL_RADIUS.value());
        hash = 31 * hash + CausticaConfig.Rt.Fog.GI_SAMPLES.value();
        hash = 31 * hash + CausticaConfig.Rt.Fog.DIRECT_SAMPLES.value();
        hash = 31 * hash + CausticaConfig.Rt.Fog.LOCAL_SAMPLES.value();
        hash = 31 * hash + Float.floatToIntBits(CausticaConfig.Rt.Fog.CACHE_CELL_SIZE.value());
        return hash;
    }

    private static RtImage createAtlas(RtContext ctx, int width, int height, String channel) {
        return ctx.createStorageImage(width, height, HDR_FORMAT,
                "fog " + channel + " froxel atlas " + width + "x" + height);
    }

    /**
     * Record the independent lighting fields, exact medium integration, and history copies. The caller
     * invokes this after world tracing and before DLSS-RR so GI reads the completed pre-exposed scene.
     */
    public void record(VkCommandBuffer cmd, long tlas, RtImage scene, RtImage depth,
                       long worldPushAddr, long lightBufAddr, long lightAliasAddr,
                       long lightLocalAliasAddr, long lightGridCellAddr, long lightGridSpanAddr,
                       RtSkyLut skyLut, float baseHeight,
                       RtGpuExecutor.GraphicsUse graphicsUse,
                       RtGpuExecutor.GraphicsUseWaiter graphicsUseWaiter) {
        if (!enabled() || !ready() || CausticaConfig.Rt.Fog.DENSITY.value() <= 0.0f) return;
        float maxDistance = CausticaConfig.Rt.Fog.MAX_DISTANCE.value();
        if (Float.floatToIntBits(baseHeight) != Float.floatToIntBits(historyBaseHeight)) {
            historyValid = false;
            historyBaseHeight = baseHeight;
        }
        boolean historyOn = historyValid && CausticaConfig.Rt.Fog.HISTORY_ENABLED.value();
        float giHistoryBlend = historyOn && CausticaConfig.Rt.Fog.GI_HISTORY_ENABLED.value()
                ? CausticaConfig.Rt.Fog.GI_TEMPORAL_BLEND.value() : 0.0f;
        float directHistoryBlend = historyOn ? CausticaConfig.Rt.Fog.DIRECT_TEMPORAL_BLEND.value() : 0.0f;
        float localHistoryBlend = historyOn ? CausticaConfig.Rt.Fog.LOCAL_TEMPORAL_BLEND.value() : 0.0f;
        float cacheHistoryBlend = historyOn ? CausticaConfig.Rt.Fog.CACHE_TEMPORAL_BLEND.value() : 0.0f;

        pipeline.bindFrame(tlas, scene, depth, directVolume, localVolume, giVolume, cacheVolume, giAuxVolume,
                directHistory, localHistory, giHistory, giAuxHistory, cacheHistory,
                skyLut.skyViewView(), skyLut.transmittanceView(), skyLut.sampler(),
                graphicsUse, graphicsUseWaiter);

        FogLightingPushData.Float4 grid = new FogLightingPushData.Float4(
                gridX, gridY, gridZ, maxDistance);
        FogLightingPushData.Float4 settings = new FogLightingPushData.Float4(
                CausticaConfig.Rt.Fog.DIRECT_ANISOTROPY.value(),
                CausticaConfig.Rt.Fog.LOCAL_LIGHTING_ENABLED.value() ? 1.0f : 0.0f,
                CausticaConfig.Rt.Fog.DIRECT_ENABLED.value() ? 1.0f : 0.0f,
                CausticaConfig.Rt.Fog.LOCAL_SAMPLES.value());
        int giSamples = CausticaConfig.Rt.Fog.GI_ENABLED.value()
                ? CausticaConfig.Rt.Fog.GI_SAMPLES.value() : 0;
        int directSamples = CausticaConfig.Rt.Fog.DIRECT_ENABLED.value()
                ? CausticaConfig.Rt.Fog.DIRECT_SAMPLES.value() : 0;
        float cacheCellSize = CausticaConfig.Rt.Fog.CACHE_REUSE_ENABLED.value()
                ? CausticaConfig.Rt.Fog.CACHE_CELL_SIZE.value() : 0.0f;
        FogLightingPushData.Float4 giSettings = new FogLightingPushData.Float4(
                giSamples,
                CausticaConfig.Rt.Fog.GI_SPATIAL_RADIUS.value(),
                directSamples,
                cacheCellSize);
        FogLightingPushData.Float4 temporalSettings = new FogLightingPushData.Float4(
                giHistoryBlend, directHistoryBlend, localHistoryBlend, cacheHistoryBlend);
        FogLightingPushData.Float4 spatialSettings = new FogLightingPushData.Float4(
                CausticaConfig.Rt.Fog.GI_SPATIAL_RADIUS.value(),
                CausticaConfig.Rt.Fog.DIRECT_SPATIAL_RADIUS.value(),
                CausticaConfig.Rt.Fog.LOCAL_SPATIAL_RADIUS.value(),
                CausticaConfig.Rt.Fog.CACHE_SPATIAL_RADIUS.value());
        pipeline.dispatchLighting(cmd, atlasWidth, atlasHeight,
                new FogLightingPushData(worldPushAddr, lightBufAddr, lightAliasAddr, lightLocalAliasAddr,
                        lightGridCellAddr, lightGridSpanAddr, grid, settings, giSettings,
                        temporalSettings, spatialSettings));

        pipeline.barrierLightingToIntegration(cmd, directVolume, localVolume, giVolume, cacheVolume, giAuxVolume);

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
                CausticaConfig.Rt.Fog.LOCAL_STRENGTH.value(),
                CausticaConfig.Rt.Fog.CACHE_STRENGTH.value());
        FogIntegratePushData.Float4 scatteringColor = new FogIntegratePushData.Float4(
                CausticaConfig.Rt.Fog.SCATTER_R.value(),
                CausticaConfig.Rt.Fog.SCATTER_G.value(),
                CausticaConfig.Rt.Fog.SCATTER_B.value(),
                CausticaConfig.Rt.Fog.SCATTERING_STRENGTH.value());
        pipeline.dispatchIntegrate(cmd, scene.width, scene.height,
                new FogIntegratePushData(worldPushAddr, integrateGrid, medium, lighting, scatteringColor));

        RtImage[] currentImages = {directVolume, localVolume, giVolume, giAuxVolume, cacheVolume};
        RtImage[] historyImages = {directHistory, localHistory, giHistory, giAuxHistory, cacheHistory};
        pipeline.barrierForHistoryCopy(cmd, currentImages, historyImages);
        pipeline.copyHistory(cmd, directVolume, directHistory);
        pipeline.copyHistory(cmd, localVolume, localHistory);
        pipeline.copyHistory(cmd, giVolume, giHistory);
        pipeline.copyHistory(cmd, giAuxVolume, giAuxHistory);
        pipeline.copyHistory(cmd, cacheVolume, cacheHistory);
        pipeline.barrierAfterHistoryCopy(cmd, currentImages, historyImages);
        historyValid = true;
    }

    public void destroy() {
        if (destroyed) return;
        if (pipeline != null) {
            pipeline.destroy();
            pipeline = null;
        }
        destroyImages();
        activeLastFrame = false;
        failed = false;
        destroyed = true;
    }

    private void destroyImages() {
        destroy(directVolume);
        destroy(localVolume);
        destroy(giVolume);
        destroy(cacheVolume);
        destroy(giAuxVolume);
        destroy(directHistory);
        destroy(localHistory);
        destroy(giHistory);
        destroy(giAuxHistory);
        destroy(cacheHistory);
        directVolume = null;
        localVolume = null;
        giVolume = null;
        cacheVolume = null;
        giAuxVolume = null;
        directHistory = null;
        localHistory = null;
        giHistory = null;
        giAuxHistory = null;
        cacheHistory = null;
        gridX = -1;
        gridY = -1;
        gridZ = -1;
        atlasWidth = -1;
        atlasHeight = -1;
        historyValid = false;
        historyFarDistance = -1.0f;
        historyBaseHeight = Float.NaN;
        settingsInitialized = false;
    }

    private static void destroy(RtImage image) {
        if (image != null) image.destroy();
    }
}
