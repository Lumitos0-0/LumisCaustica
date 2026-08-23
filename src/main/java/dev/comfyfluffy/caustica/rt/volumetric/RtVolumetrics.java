package dev.comfyfluffy.caustica.rt.volumetric;

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
 * Owns the frustum-froxel resources and the two ray-generation passes that update them. Lighting and
 * extinction are injected into one of two history volumes, then integrated camera-outward into a volume
 * of cumulative in-scattering and transmittance. The path tracer samples that integrated volume before
 * pre-exposure, so volumetrics remain scene-linear and work unchanged with DLSS-RR, SDR, and HDR output.
 */
public final class RtVolumetrics {
    public static final int INJECT_RAYGEN = 2;
    public static final int INTEGRATE_RAYGEN = 3;

    private RtFroxelGrid grid;
    private RtImage volumeDepth;
    private final RtImage[] scattering = new RtImage[2];
    private RtImage integrated;
    private long lastRecordedFrame = Long.MIN_VALUE;
    private long lastOpticalSignature = Long.MIN_VALUE;

    /** Frame-local values serialized into {@code WorldPush}. */
    public record FrameData(Int4 gridAndFlags, Float4 distanceParams, Float4 optics,
                            Float4 shape, Float4 worldOffsetAndTime, boolean enabled,
                            int writeIndex, long opticalSignature, long frameIndex) {
    }

    public boolean matches(int renderWidth, int renderHeight) {
        if (grid == null || volumeDepth == null || scattering[0] == null || scattering[1] == null
                || integrated == null) {
            return false;
        }
        return grid.equals(wantedGrid(renderWidth, renderHeight))
                && volumeDepth.width == renderWidth && volumeDepth.height == renderHeight;
    }

    /** Recreate size-dependent images. The caller must have made the device idle before this method. */
    public void ensure(RtContext ctx, int renderWidth, int renderHeight) {
        RtFroxelGrid wanted = wantedGrid(renderWidth, renderHeight);
        if (matches(renderWidth, renderHeight)) {
            return;
        }
        destroyImages();
        grid = wanted;
        volumeDepth = ctx.createStorageImage(renderWidth, renderHeight, VK10.VK_FORMAT_R32_SFLOAT,
                "volumetric first-surface depth " + renderWidth + "x" + renderHeight);
        String extent = wanted.width() + "x" + wanted.height() + "x" + wanted.depth();
        scattering[0] = ctx.createStorageVolume(wanted.width(), wanted.height(), wanted.depth(),
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel scattering A " + extent);
        scattering[1] = ctx.createStorageVolume(wanted.width(), wanted.height(), wanted.depth(),
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel scattering B " + extent);
        integrated = ctx.createStorageVolume(wanted.width(), wanted.height(), wanted.depth(),
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel integrated lighting " + extent);
        invalidateHistory();
    }

    /** Initialize every descriptor-ring slot after pipeline creation or image recreation. */
    public void bindAll(RtPipeline pipeline) {
        requireReady();
        pipeline.setVolumetricImages(volumeDepth.view, scattering[0].view, scattering[1].view,
                integrated.view);
    }

    /**
     * Build this frame's physically based medium parameters. The base height is represented in rebased
     * shader space while the noise offset restores a small, world-stable coordinate after terrain rebases.
     */
    public FrameData prepareFrame(long frameIndex, int rebaseX, int rebaseY, int rebaseZ,
                                  int seaLevel, boolean submerged, float timeSeconds,
                                  float cameraDeltaX, float cameraDeltaY, float cameraDeltaZ) {
        requireReady();
        float maxDistance = CausticaConfig.Rt.Volumetrics.MAX_DISTANCE.value();
        float distribution = CausticaConfig.Rt.Volumetrics.DEPTH_EXPONENT.value();
        float extinction = CausticaConfig.Rt.Volumetrics.EXTINCTION.value();
        float heightFalloff = CausticaConfig.Rt.Volumetrics.HEIGHT_FALLOFF.value();
        float albedo = CausticaConfig.Rt.Volumetrics.SINGLE_SCATTERING_ALBEDO.value();
        float anisotropy = CausticaConfig.Rt.Volumetrics.ANISOTROPY.value();
        float noiseAmount = CausticaConfig.Rt.Volumetrics.NOISE_AMOUNT.value();
        float noiseScale = CausticaConfig.Rt.Volumetrics.NOISE_SCALE.value();
        float temporalWeight = CausticaConfig.Rt.Volumetrics.TEMPORAL_WEIGHT.value();
        int localCandidates = CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_CANDIDATES.value();

        boolean enabled = CausticaConfig.Rt.Volumetrics.ENABLED.value() && !submerged;
        long signature = opticalSignature(maxDistance, distribution, extinction, heightFalloff,
                albedo, anisotropy, noiseAmount, noiseScale, temporalWeight, localCandidates, seaLevel);
        float cameraTravel2 = cameraDeltaX * cameraDeltaX + cameraDeltaY * cameraDeltaY
                + cameraDeltaZ * cameraDeltaZ;
        boolean historyValid = enabled && lastRecordedFrame == frameIndex - 1
                && lastOpticalSignature == signature
                && cameraTravel2 < maxDistance * maxDistance * 0.25f;
        int flags = enabled ? 1 : 0;
        if (historyValid) {
            flags |= 2;
        }
        flags |= (Math.clamp(localCandidates, 0, 255) << 8);
        int writeIndex = (int) (frameIndex & 1L);
        float baseHeightRebased = seaLevel + CausticaConfig.Rt.Volumetrics.HEIGHT_OFFSET.value() - rebaseY;

        return new FrameData(
                new Int4(grid.width(), grid.height(), grid.depth(), flags),
                new Float4(maxDistance, distribution, extinction, heightFalloff),
                new Float4(albedo, albedo, albedo, anisotropy),
                new Float4(baseHeightRebased, noiseAmount, noiseScale, temporalWeight),
                new Float4(wrappedWorldCoordinate(rebaseX), rebaseY,
                        wrappedWorldCoordinate(rebaseZ), timeSeconds),
                enabled, writeIndex, signature, frameIndex);
    }

    /**
     * Bind this frame's ping-pong direction into the descriptor slot selected by {@code setTlas}, inject
     * lighting with ray-traced visibility, then prefix-integrate each XY froxel column.
     */
    public void record(RtContext ctx, VkCommandBuffer cmd, RtPipeline pipeline,
                       ByteBuffer pushConstants, FrameData frame) {
        if (!frame.enabled()) {
            return;
        }
        RtImage current = scattering[frame.writeIndex()];
        RtImage history = scattering[1 - frame.writeIndex()];
        pipeline.setCurrentVolumetricHistory(current.view, history.view);
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "froxel volumetrics");
             RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.volumetrics")) {
            pipeline.trace(cmd, grid.width(), grid.height(), grid.depth(), pushConstants, INJECT_RAYGEN);
            barrier(cmd);
            pipeline.trace(cmd, grid.width(), grid.height(), 1, pushConstants, INTEGRATE_RAYGEN);
            barrier(cmd);
        }
        lastRecordedFrame = frame.frameIndex();
        lastOpticalSignature = frame.opticalSignature();
    }

    public void invalidateHistory() {
        lastRecordedFrame = Long.MIN_VALUE;
        lastOpticalSignature = Long.MIN_VALUE;
    }

    public void destroy() {
        destroyImages();
        grid = null;
        invalidateHistory();
    }

    public RtFroxelGrid grid() {
        return grid;
    }

    private static void barrier(VkCommandBuffer cmd) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
    }

    private static RtFroxelGrid wantedGrid(int renderWidth, int renderHeight) {
        return RtFroxelGrid.forRenderSize(renderWidth, renderHeight,
                CausticaConfig.Rt.Volumetrics.GRID_PIXEL_SIZE.value(),
                CausticaConfig.Rt.Volumetrics.DEPTH_SLICES.value());
    }

    private static float wrappedWorldCoordinate(int coordinate) {
        return Math.floorMod(coordinate, 65536);
    }

    private static long opticalSignature(float... values) {
        long hash = 0xcbf29ce484222325L;
        for (float value : values) {
            hash ^= Integer.toUnsignedLong(Float.floatToIntBits(value));
            hash *= 0x100000001b3L;
        }
        return hash;
    }

    private static long opticalSignature(float maxDistance, float distribution, float extinction,
                                         float heightFalloff, float albedo, float anisotropy,
                                         float noiseAmount, float noiseScale, float temporalWeight,
                                         int localCandidates, int seaLevel) {
        long hash = opticalSignature(maxDistance, distribution, extinction, heightFalloff, albedo,
                anisotropy, noiseAmount, noiseScale, temporalWeight);
        hash = (hash ^ Integer.toUnsignedLong(localCandidates)) * 0x100000001b3L;
        return (hash ^ Integer.toUnsignedLong(seaLevel)) * 0x100000001b3L;
    }

    private void requireReady() {
        if (grid == null || volumeDepth == null || scattering[0] == null || scattering[1] == null
                || integrated == null) {
            throw new IllegalStateException("Froxel resources have not been created");
        }
    }

    private void destroyImages() {
        if (volumeDepth != null) {
            volumeDepth.destroy();
            volumeDepth = null;
        }
        for (int i = 0; i < scattering.length; i++) {
            if (scattering[i] != null) {
                scattering[i].destroy();
                scattering[i] = null;
            }
        }
        if (integrated != null) {
            integrated.destroy();
            integrated = null;
        }
    }
}
