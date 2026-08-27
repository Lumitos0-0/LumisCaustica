package dev.comfyfluffy.caustica.rt.volumetric;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
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

/**
 * Owns a history-free camera-frustum participating-media volume.
 *
 * <p>Every enabled frame runs injection, deterministic bilateral filtering, and analytic Beer-Lambert
 * prefix integration. A fourth ray-generation dispatch composites the result into the display-resolution
 * DLSS-RR/fallback image. There are exactly three 3D images: source/extinction, filtered source/extinction,
 * and cumulative scattering/transmittance; no ping-pong or temporal image exists.</p>
 */
public final class RtVolumetrics {
    public static final int INJECT_RAYGEN = 2;
    public static final int FILTER_RAYGEN = 3;
    public static final int INTEGRATE_RAYGEN = 4;
    public static final int COMPOSITE_RAYGEN = 5;

    private RtFroxelGrid.Dimensions dimensions;
    private RtImage volumeDepth;
    private RtImage scattering;
    private RtImage filtered;
    private RtImage integrated;

    /** Frame-local values serialized into {@code WorldPush}. */
    public record FrameData(Int4 gridAndFlags, Float4 distanceParams, Float4 optics,
                            Float4 shape, Float4 worldOffsetAndStrength, Float4 lighting,
                            boolean enabled) {
    }

    public boolean matches(int renderWidth, int renderHeight) {
        return dimensions != null && dimensions.equals(wantedDimensions())
                && volumeDepth != null && scattering != null && filtered != null && integrated != null
                && volumeDepth.width == renderWidth && volumeDepth.height == renderHeight;
    }

    /** Recreate size/quality-dependent images. The caller has already made the device idle. */
    public void ensure(RtContext ctx, int renderWidth, int renderHeight) {
        RtFroxelGrid.Dimensions wanted = wantedDimensions();
        if (matches(renderWidth, renderHeight)) {
            return;
        }
        destroyImages();
        dimensions = wanted;
        volumeDepth = ctx.createStorageImage(renderWidth, renderHeight, VK10.VK_FORMAT_R32_SFLOAT,
                "volumetric first-interface depth " + renderWidth + "x" + renderHeight);
        String extent = wanted.width() + "x" + wanted.height() + "x" + wanted.depth();
        scattering = ctx.createStorageVolume(wanted.width(), wanted.height(), wanted.depth(),
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel source and extinction " + extent);
        filtered = ctx.createStorageVolume(wanted.width(), wanted.height(), wanted.depth(),
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel filtered source and extinction " + extent);
        integrated = ctx.createStorageVolume(wanted.width(), wanted.height(), wanted.depth(),
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel integrated scattering " + extent);
    }

    /** Initialize all descriptor-ring slots after pipeline creation or volume recreation. */
    public void bindAll(RtPipeline pipeline) {
        requireReady();
        pipeline.setVolumetricImages(volumeDepth.view, scattering.view, filtered.view, integrated.view);
    }

    /** Bind the current display-resolution RR/fallback output used only by the final composite stage. */
    public void bindCompositeOutput(RtPipeline pipeline, RtImage compositeOutput) {
        requireReady();
        pipeline.setVolumetricCompositeImage(compositeOutput.view);
    }

    /**
     * Build the neutral atmospheric coefficients. Rebase X/Z are wrapped before conversion to float so
     * fBm stays world-stable without losing precision in distant worlds. A submerged camera explicitly
     * disables this atmospheric volume; water continues to use the path tracer's existing surface model.
     */
    public FrameData prepareFrame(int rebaseX, int rebaseY, int rebaseZ,
                                  int seaLevel, boolean submerged) {
        requireReady();
        boolean enabled = CausticaConfig.Rt.Volumetrics.ENABLED.value() && !submerged;
        int flags = enabled ? 1 : 0;
        if (submerged) {
            flags |= 2;
        }
        int localCandidates = Math.clamp(
                CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_CANDIDATES.value(), 0, 255);
        int emitterSamples = Math.clamp(
                CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_SAMPLES.value(), 1, 15);
        flags |= localCandidates << 8;
        flags |= emitterSamples << 16;

        float albedo = CausticaConfig.Rt.Volumetrics.SINGLE_SCATTERING_ALBEDO.value();
        float referenceHeight = seaLevel + CausticaConfig.Rt.Volumetrics.HEIGHT_OFFSET.value() - rebaseY;
        return new FrameData(
                new Int4(dimensions.width(), dimensions.height(), dimensions.depth(), flags),
                new Float4(
                        CausticaConfig.Rt.Volumetrics.MAX_DISTANCE.value(),
                        CausticaConfig.Rt.Volumetrics.DEPTH_EXPONENT.value(),
                        CausticaConfig.Rt.Volumetrics.EXTINCTION.value(),
                        CausticaConfig.Rt.Volumetrics.HEIGHT_FALLOFF.value()),
                new Float4(albedo, albedo, albedo,
                        CausticaConfig.Rt.Volumetrics.ANISOTROPY.value()),
                new Float4(referenceHeight,
                        CausticaConfig.Rt.Volumetrics.NOISE_AMOUNT.value(),
                        CausticaConfig.Rt.Volumetrics.NOISE_SCALE.value(),
                        CausticaConfig.Rt.Volumetrics.AMBIENT_STRENGTH.value()),
                new Float4(wrappedWorldCoordinate(rebaseX), rebaseY,
                        wrappedWorldCoordinate(rebaseZ),
                        CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_STRENGTH.value()),
                new Float4(
                        CausticaConfig.Rt.Volumetrics.DIRECTIONAL_STRENGTH.value(),
                        CausticaConfig.Rt.Volumetrics.DIRECTIONAL_FOCUS.value(),
                        CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_CLAMP.value(),
                        CausticaConfig.Rt.Volumetrics.FILTER_EDGE_SHARPNESS.value()),
                enabled);
    }

    /** Rebuild the complete volume after primary first-interface depth is available. */
    public void recordUpdate(RtContext ctx, VkCommandBuffer cmd, RtPipeline pipeline,
                             ByteBuffer pushConstants, FrameData frame) {
        if (!frame.enabled()) {
            return;
        }
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "froxel volume rebuild");
             RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.volumetricsUpdate")) {
            pipeline.trace(cmd, dimensions.width(), dimensions.height(), dimensions.depth(),
                    pushConstants, INJECT_RAYGEN);
            barrier(cmd);
            pipeline.trace(cmd, dimensions.width(), dimensions.height(), dimensions.depth(),
                    pushConstants, FILTER_RAYGEN);
            barrier(cmd);
            pipeline.trace(cmd, dimensions.width(), dimensions.height(), 1,
                    pushConstants, INTEGRATE_RAYGEN);
        }
    }

    /** Composite cumulative scattering into the post-RR display-resolution image. */
    public void recordComposite(RtContext ctx, VkCommandBuffer cmd, RtPipeline pipeline,
                                ByteBuffer pushConstants, FrameData frame,
                                int displayWidth, int displayHeight) {
        if (!frame.enabled()) {
            return;
        }
        barrier(cmd); // integration and DLSS-RR/fallback writes visible to the final raygen
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "post-RR volumetric composite");
             RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.volumetricsComposite")) {
            pipeline.trace(cmd, displayWidth, displayHeight, pushConstants, COMPOSITE_RAYGEN);
        }
    }

    public void destroy() {
        destroyImages();
        dimensions = null;
    }

    public RtFroxelGrid.Dimensions dimensions() {
        return dimensions;
    }

    static RtFroxelGrid.Dimensions wantedDimensions() {
        int quality = Math.clamp(CausticaConfig.Rt.Volumetrics.QUALITY.value(), 0, 2);
        RtFroxelGrid.Quality preset = switch (quality) {
            case 0 -> RtFroxelGrid.Quality.LOW;
            case 1 -> RtFroxelGrid.Quality.MEDIUM;
            default -> RtFroxelGrid.Quality.HIGH;
        };
        return RtFroxelGrid.dimensions(preset);
    }

    private static float wrappedWorldCoordinate(int coordinate) {
        return Math.floorMod(coordinate, 65536);
    }

    private static void barrier(VkCommandBuffer cmd) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
    }

    private void requireReady() {
        if (dimensions == null || volumeDepth == null || scattering == null
                || filtered == null || integrated == null) {
            throw new IllegalStateException("Froxel resources have not been created");
        }
    }

    private void destroyImages() {
        if (volumeDepth != null) {
            volumeDepth.destroy();
            volumeDepth = null;
        }
        if (scattering != null) {
            scattering.destroy();
            scattering = null;
        }
        if (filtered != null) {
            filtered.destroy();
            filtered = null;
        }
        if (integrated != null) {
            integrated.destroy();
            integrated = null;
        }
    }
}
