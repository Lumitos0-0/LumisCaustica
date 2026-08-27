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
 * Owns the camera-frustum participating-media volume.
 *
 * <p>Every enabled frame runs deterministic injection, geometry-depth-guided spatial filtering, a low
 * depth-validated temporal stabilization pass, and analytic Beer-Lambert prefix integration. The final
 * raygen composites into the display-resolution DLSS-RR/fallback image. Source sampling has no frame seed;
 * history feedback defaults to 0.15 and is hard-capped at 0.35 to suppress movement noise without trails.</p>
 */
public final class RtVolumetrics {
    public static final int INJECT_RAYGEN = 2;
    public static final int FILTER_RAYGEN = 3;
    public static final int TEMPORAL_RAYGEN = 4;
    public static final int INTEGRATE_RAYGEN = 5;
    public static final int COMPOSITE_RAYGEN = 6;

    private RtFroxelGrid.Dimensions dimensions;
    private RtImage volumeDepth;
    private RtImage scattering;
    private RtImage filtered;
    private RtImage integrated;
    private final RtImage[] resolved = new RtImage[2];
    private final RtImage[] depth = new RtImage[2];
    private long lastRecordedFrame = Long.MIN_VALUE;
    private long lastOpticalSignature = Long.MIN_VALUE;

    /** Frame-local values serialized into {@code WorldPush}. */
    public record FrameData(Int4 gridAndFlags, Float4 distanceParams, Float4 optics,
                            Float4 shape, Float4 worldOffsetAndStrength, Float4 lighting,
                            Float4 temporal, boolean enabled, int writeIndex,
                            long frameIndex, long opticalSignature) {
    }

    public boolean matches(int renderWidth, int renderHeight) {
        return dimensions != null && dimensions.equals(wantedDimensions())
                && volumeDepth != null && scattering != null && filtered != null && integrated != null
                && resolved[0] != null && resolved[1] != null && depth[0] != null && depth[1] != null
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
        scattering = createColorVolume(ctx, "froxel source and extinction ", extent);
        filtered = createColorVolume(ctx, "froxel depth-filtered source and extinction ", extent);
        integrated = createColorVolume(ctx, "froxel integrated scattering ", extent);
        resolved[0] = createColorVolume(ctx, "froxel resolved A ", extent);
        resolved[1] = createColorVolume(ctx, "froxel resolved B ", extent);
        depth[0] = createDepthVolume(ctx, "froxel geometry depth A ", extent);
        depth[1] = createDepthVolume(ctx, "froxel geometry depth B ", extent);
        invalidateHistory();
    }

    private RtImage createColorVolume(RtContext ctx, String label, String extent) {
        return ctx.createStorageVolume(dimensions.width(), dimensions.height(), dimensions.depth(),
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, label + extent);
    }

    private RtImage createDepthVolume(RtContext ctx, String label, String extent) {
        return ctx.createStorageVolume(dimensions.width(), dimensions.height(), dimensions.depth(),
                VK10.VK_FORMAT_R32_SFLOAT, label + extent);
    }

    /** Initialize every descriptor-ring slot after pipeline creation or volume recreation. */
    public void bindAll(RtPipeline pipeline) {
        requireReady();
        pipeline.setVolumetricImages(volumeDepth.view, scattering.view, filtered.view, integrated.view,
                resolved[0].view, resolved[1].view, depth[0].view, depth[1].view);
    }

    /** Bind the current display-resolution RR/fallback output used by the final composite stage. */
    public void bindCompositeOutput(RtPipeline pipeline, RtImage compositeOutput) {
        requireReady();
        pipeline.setVolumetricCompositeImage(compositeOutput.view);
    }

    /** Build neutral atmospheric coefficients and decide whether the previous volume is safe to sample. */
    public FrameData prepareFrame(long frameIndex, int rebaseX, int rebaseY, int rebaseZ,
                                  int seaLevel, boolean submerged,
                                  float cameraDeltaX, float cameraDeltaY, float cameraDeltaZ) {
        requireReady();
        boolean enabled = CausticaConfig.Rt.Volumetrics.ENABLED.value() && !submerged;
        int quality = CausticaConfig.Rt.Volumetrics.QUALITY.value();
        int localCandidates = effectiveLocalLightCandidates(quality);
        int emitterSamples = effectiveEmitterSamples(quality);
        float maxDistance = CausticaConfig.Rt.Volumetrics.MAX_DISTANCE.value();
        float temporalWeight = Math.min(CausticaConfig.Rt.Volumetrics.TEMPORAL_WEIGHT.value(), 0.35f);
        float referenceHeight = seaLevel + CausticaConfig.Rt.Volumetrics.HEIGHT_OFFSET.value() - rebaseY;
        long opticalSignature = opticalSignature(seaLevel, quality, localCandidates, emitterSamples,
                maxDistance, temporalWeight);
        float cameraTravel2 = cameraDeltaX * cameraDeltaX + cameraDeltaY * cameraDeltaY
                + cameraDeltaZ * cameraDeltaZ;
        boolean historyValid = enabled && temporalWeight > 0.0f
                && lastRecordedFrame == frameIndex - 1
                && lastOpticalSignature == opticalSignature
                && cameraTravel2 <= 64.0f;

        int flags = enabled ? 1 : 0;
        if (submerged) {
            flags |= 2;
        }
        if (historyValid) {
            flags |= 4;
        }
        flags |= Math.clamp(localCandidates, 0, 255) << 8;
        flags |= Math.clamp(emitterSamples, 1, 15) << 16;
        float albedo = CausticaConfig.Rt.Volumetrics.SINGLE_SCATTERING_ALBEDO.value();
        return new FrameData(
                new Int4(dimensions.width(), dimensions.height(), dimensions.depth(), flags),
                new Float4(maxDistance,
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
                        CausticaConfig.Rt.Volumetrics.DEPTH_TOLERANCE.value()),
                new Float4(temporalWeight, 0.0f, 0.0f, 0.0f),
                enabled, (int) (frameIndex & 1L), frameIndex, opticalSignature);
    }

    /** Rebuild and stabilize the volume after primary first-interface depth is available. */
    public void recordUpdate(RtContext ctx, VkCommandBuffer cmd, RtPipeline pipeline,
                             ByteBuffer pushConstants, FrameData frame) {
        if (!frame.enabled()) {
            return;
        }
        int current = frame.writeIndex();
        int previous = 1 - current;
        // setTlas selected and waited for this exact descriptor slot earlier in the frame.
        pipeline.setCurrentVolumetricImages(resolved[current].view, resolved[previous].view,
                depth[current].view, depth[previous].view);
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "froxel volume rebuild");
             RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.volumetricsUpdate")) {
            pipeline.trace(cmd, dimensions.width(), dimensions.height(), dimensions.depth(),
                    pushConstants, INJECT_RAYGEN);
            barrier(cmd);
            pipeline.trace(cmd, dimensions.width(), dimensions.height(), dimensions.depth(),
                    pushConstants, FILTER_RAYGEN);
            barrier(cmd);
            pipeline.trace(cmd, dimensions.width(), dimensions.height(), dimensions.depth(),
                    pushConstants, TEMPORAL_RAYGEN);
            barrier(cmd);
            pipeline.trace(cmd, dimensions.width(), dimensions.height(), 1,
                    pushConstants, INTEGRATE_RAYGEN);
        }
        lastRecordedFrame = frame.frameIndex();
        lastOpticalSignature = frame.opticalSignature();
    }

    /** Composite cumulative scattering into the post-RR display-resolution image. */
    public void recordComposite(RtContext ctx, VkCommandBuffer cmd, RtPipeline pipeline,
                                ByteBuffer pushConstants, FrameData frame,
                                int displayWidth, int displayHeight) {
        if (!frame.enabled()) {
            return;
        }
        barrier(cmd);
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "post-RR volumetric composite");
             RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.volumetricsComposite")) {
            pipeline.trace(cmd, displayWidth, displayHeight, pushConstants, COMPOSITE_RAYGEN);
        }
    }

    public void destroy() {
        destroyImages();
        dimensions = null;
        invalidateHistory();
    }

    public RtFroxelGrid.Dimensions dimensions() {
        return dimensions;
    }

    static RtFroxelGrid.Dimensions wantedDimensions() {
        int quality = Math.clamp(CausticaConfig.Rt.Volumetrics.QUALITY.value(), 0, 3);
        RtFroxelGrid.Quality preset = switch (quality) {
            case 0 -> RtFroxelGrid.Quality.LOW;
            case 1 -> RtFroxelGrid.Quality.MEDIUM;
            case 2 -> RtFroxelGrid.Quality.HIGH;
            default -> RtFroxelGrid.Quality.ULTRA;
        };
        return RtFroxelGrid.dimensions(preset);
    }

    static int effectiveLocalLightCandidates(int quality) {
        int configured = CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_CANDIDATES.value();
        return quality >= 3 && configured > 0 ? Math.max(configured, 8) : configured;
    }

    static int effectiveEmitterSamples(int quality) {
        int configured = CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_SAMPLES.value();
        return quality >= 3 ? Math.max(configured, 2) : configured;
    }

    private long opticalSignature(int seaLevel, int quality, int localCandidates, int emitterSamples,
                                  float maxDistance, float temporalWeight) {
        long hash = 0xcbf29ce484222325L;
        hash = mix(hash, seaLevel);
        hash = mix(hash, quality);
        hash = mix(hash, dimensions.width());
        hash = mix(hash, dimensions.height());
        hash = mix(hash, dimensions.depth());
        hash = mix(hash, localCandidates);
        hash = mix(hash, emitterSamples);
        hash = mix(hash, maxDistance);
        hash = mix(hash, temporalWeight);
        hash = mix(hash, CausticaConfig.Rt.Volumetrics.DEPTH_EXPONENT.value());
        hash = mix(hash, CausticaConfig.Rt.Volumetrics.EXTINCTION.value());
        hash = mix(hash, CausticaConfig.Rt.Volumetrics.SINGLE_SCATTERING_ALBEDO.value());
        hash = mix(hash, CausticaConfig.Rt.Volumetrics.ANISOTROPY.value());
        hash = mix(hash, CausticaConfig.Rt.Volumetrics.AMBIENT_STRENGTH.value());
        hash = mix(hash, CausticaConfig.Rt.Volumetrics.DIRECTIONAL_STRENGTH.value());
        hash = mix(hash, CausticaConfig.Rt.Volumetrics.DIRECTIONAL_FOCUS.value());
        hash = mix(hash, CausticaConfig.Rt.Volumetrics.HEIGHT_OFFSET.value());
        hash = mix(hash, CausticaConfig.Rt.Volumetrics.HEIGHT_FALLOFF.value());
        hash = mix(hash, CausticaConfig.Rt.Volumetrics.NOISE_AMOUNT.value());
        hash = mix(hash, CausticaConfig.Rt.Volumetrics.NOISE_SCALE.value());
        hash = mix(hash, CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_STRENGTH.value());
        hash = mix(hash, CausticaConfig.Rt.Volumetrics.LOCAL_LIGHT_CLAMP.value());
        return mix(hash, CausticaConfig.Rt.Volumetrics.DEPTH_TOLERANCE.value());
    }

    private static long mix(long hash, int value) {
        return (hash ^ Integer.toUnsignedLong(value)) * 0x100000001b3L;
    }

    private static long mix(long hash, float value) {
        return mix(hash, Float.floatToIntBits(value));
    }

    private static float wrappedWorldCoordinate(int coordinate) {
        return Math.floorMod(coordinate, 65536);
    }

    private static void barrier(VkCommandBuffer cmd) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
    }

    private void invalidateHistory() {
        lastRecordedFrame = Long.MIN_VALUE;
        lastOpticalSignature = Long.MIN_VALUE;
    }

    private void requireReady() {
        if (dimensions == null || volumeDepth == null || scattering == null
                || filtered == null || integrated == null
                || resolved[0] == null || resolved[1] == null
                || depth[0] == null || depth[1] == null) {
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
        for (int i = 0; i < 2; ++i) {
            if (resolved[i] != null) {
                resolved[i].destroy();
                resolved[i] = null;
            }
            if (depth[i] != null) {
                depth[i].destroy();
                depth[i] = null;
            }
        }
    }
}
