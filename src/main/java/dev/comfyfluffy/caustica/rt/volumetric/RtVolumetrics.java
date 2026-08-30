package dev.comfyfluffy.caustica.rt.volumetric;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.RtFrameStats;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.accel.RtVolume;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData.Float4;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData.Int4;
import dev.comfyfluffy.caustica.rt.pipeline.RtPipeline;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * Frustum-froxel participating media, path-traced per froxel. Owns the 3D volumes and the three
 * ray-generation passes that update them:
 *
 * <ol>
 *   <li>inject — one thread per lighting-probe cell (the probe grid is decimated in XY and Z relative
 *       to the fog grid, see {@code froxelInjectDimensions}) computes extinction (world-stable density
 *       field) and the single-scattering term {sigma_s·L per unit length, sigma_t}; sun/moon light
 *       reaches the media through the same path-traced {@code visibility()} rays as surface lighting,
 *       block emitters enter through a pdf-correct next-event sample over the light grid, and the
 *       remaining radiance is sampled by tracing the scene's own path tracer ({@code tracePath}) from
 *       each probe cell under a restricted probe path (no surface RIS/emission gather/SSS, diffuse-only
 *       continuation) — so terrain GI, skylight and tinted glass/water/translucent geometry filter
 *       everything that enters the fog without the single-sample HDR spikes those restrictions
 *       remove;</li>
 *   <li>filter — resamples the decimated probe grid to the fog grid, temporal reprojection of the
 *       previous frame's filtered result into the current camera, gamma-encoded temporal accumulation
 *       (the DDGI/RTXGI probe curve: high-DR single samples are compressed before blending so they
 *       cannot read as per-cell dots), plus variance clipping against the current injection (ping-pong
 *       RGBA16F lanes);</li>
 *   <li>integrate — camera-outward march applying the analytic Beer–Lambert slice integral, producing a
 *       per-column {cumulative in-scatter, transmittance} volume that {@code applyFroxelFog} composites
 *       scene-linearly before pre-exposure.</li>
 * </ol>
 *
 * <p>All frame values are serialized into {@code WorldPush} by the caller using {@link FrameData};
 * the passes are extra raygen records (index 2/3/4) of the existing world pipeline, so they share the
 * TLAS, descriptors, and push constants with the trace itself. Volumes are recreated on render-size or
 * quality changes with the device idle (same discipline as every other {@code RtComposite} image).
 */
public final class RtVolumetrics {
    public static final int INJECT_RAYGEN = 2;
    public static final int FILTER_RAYGEN = 3;
    public static final int INTEGRATE_RAYGEN = 4;

    private static final int[] QUALITY_PIXEL_SIZE = {4, 4, 3, 2, 2};
    private static final int[] QUALITY_DEPTH_SLICES = {32, 48, 64, 96, 112};
    // Path-traced radiance probes per cell, bounce depth each probe continues past its first hit, and
    // the probe-grid decimation relative to the fog grid (XY and Z). Every extra bounce costs a full
    // trace + shadow rays per probe, so higher qualities raise the sample budget before the depth (the
    // fog is a blurry medium — deep chains add more cost than visible detail). Decimation trades rays
    // for lighting resolution; the temporal accumulator + trilinear resample keep lower settings smooth.
    private static final int[] QUALITY_GI_SAMPLES = {1, 1, 1, 1, 2};
    private static final int[] QUALITY_GI_BOUNCES = {0, 0, 1, 1, 1};
    private static final int[] QUALITY_PROBE_DIV_XY = {4, 3, 2, 2, 2};
    private static final int[] QUALITY_PROBE_DIV_Z = {2, 2, 2, 2, 2};

    private RtFroxelGrid grid;
    private int deviceExtentCap;
    private RtImage volumeDepth;
    private RtVolume scattering;
    private final RtVolume[] filtered = new RtVolume[2];
    private RtVolume integrated;
    private long lastRecordedFrame = Long.MIN_VALUE;
    private long opticalSignature = Long.MIN_VALUE;
    private boolean historyValid;

    /**
     * Frame-local values serialized into {@code WorldPush}. {@code enabled} also gates the pass
     * dispatch; the caller records the passes only when it is true.
     */
    public record FrameData(Int4 gridAndFlags, Float4 distanceParams, Float4 optics, Float4 shape,
                            Float4 worldOffsetAndTime, Float4 lighting,
                            boolean enabled, long opticalSignature) {
    }

    public boolean matches(int renderWidth, int renderHeight) {
        if (grid == null || volumeDepth == null || scattering == null || filtered[0] == null
                || filtered[1] == null || integrated == null) {
            return false;
        }
        // Recompute the same clamped grid ensure() derives; a device-extent clamp changes the stored
        // dims, so comparing the raw wanted grid would force a rebuild on every frame.
        RtFroxelGrid wanted = wantedGrid(renderWidth, renderHeight, deviceExtentCap);
        return grid.equals(wanted)
                && volumeDepth.width == renderWidth && volumeDepth.height == renderHeight;
    }

    /**
     * Recreate size-dependent volumes. The caller must have made the device idle before this method.
     * The grid's XY/depth are clamped to the device's 3D storage-image extent so a very large render
     * resolution degrades the fog's spatial sampling instead of failing the whole RT path; every
     * shader derives cell/pixel mapping from the pushed grid dimensions, so a clamped grid stays
     * self-consistent.
     */
    public void ensure(RtContext ctx, int renderWidth, int renderHeight) {
        int maxExtent = ctx.storage3DMaxExtent(VK10.VK_FORMAT_R16G16B16A16_SFLOAT);
        RtFroxelGrid wanted = wantedGrid(renderWidth, renderHeight, maxExtent);
        deviceExtentCap = Math.min(maxExtent, 0x7fffffff); // clamped grid dims are stored in `grid`
        if (matches(renderWidth, renderHeight)) {
            return;
        }
        destroyImages();
        grid = wanted;
        volumeDepth = ctx.createStorageImage(renderWidth, renderHeight, VK10.VK_FORMAT_R32_SFLOAT,
                "volumetric first-surface depth " + renderWidth + "x" + renderHeight);
        String extent = wanted.width() + "x" + wanted.height() + "x" + wanted.depth();
        // Lighting probe grid decimation matches the quality flags the shader reads (see
        // froxelInjectDimensions); the filter pass trilinearly resamples it to the full fog grid, so the
        // inject ray count drops by the decimation factor without losing fog density detail.
        int quality = Math.clamp(CausticaConfig.Rt.Volumetrics.QUALITY.value(), 0, 4);
        int divXY = QUALITY_PROBE_DIV_XY[quality];
        int divZ = QUALITY_PROBE_DIV_Z[quality];
        int probeWidth = (wanted.width() + divXY - 1) / divXY;
        int probeHeight = (wanted.height() + divXY - 1) / divXY;
        int probeDepth = (wanted.depth() + divZ - 1) / divZ;
        String probeExtent = probeWidth + "x" + probeHeight + "x" + probeDepth;
        scattering = ctx.createStorageVolume(probeWidth, probeHeight, probeDepth,
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel scattering " + probeExtent);
        filtered[0] = ctx.createStorageVolume(wanted.width(), wanted.height(), wanted.depth(),
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel filtered A " + extent);
        filtered[1] = ctx.createStorageVolume(wanted.width(), wanted.height(), wanted.depth(),
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel filtered B " + extent);
        integrated = ctx.createStorageVolume(wanted.width(), wanted.height(), wanted.depth(),
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel integrated " + extent);
        invalidateHistory();
    }

    /** Initialize every descriptor-ring slot after pipeline creation or volume recreation. */
    public void bindAll(RtPipeline pipeline) {
        requireReady();
        pipeline.setVolumetricImages(volumeDepth.view, scattering.view,
                filtered[0].view, filtered[1].view, integrated.view);
    }

    /**
     * Build this frame's physically based medium parameters. Base height is expressed in rebased shader
     * space while the noise offset restores a small, world-stable coordinate after terrain rebases;
     * underwater settings replace the ordinary ones when the camera is submerged, so the same shader
     * code serves both with no branching on the worldPush side beyond the flag.
     */
    public FrameData prepareFrame(long frameIndex, int rebaseX, int rebaseY, int rebaseZ,
                                  int seaLevel, boolean submerged, float timeSeconds) {
        requireReady();
        boolean underwaterEnabled = CausticaConfig.Rt.Volumetrics.UNDERWATER_ENABLED.value();
        boolean enabled = CausticaConfig.Rt.Volumetrics.ENABLED.value()
                && (!submerged || underwaterEnabled);
        boolean historyWasValid = historyValid && enabled;
        // A new frame invalidates whatever record() may have marked; record() re-arms it on success.
        historyValid = false;
        if (!enabled) {
            return new FrameData(new Int4(grid.width(), grid.height(), grid.depth(), 0),
                    new Float4(0f, 0f, 0f, 0f), new Float4(0f, 0f, 0f, 0f),
                    new Float4(0f, 0f, 0f, 0f), new Float4(0f, 0f, 0f, 0f),
                    new Float4(0f, 0f, 0f, 0f), false, Long.MIN_VALUE);
        }
        if (lastRecordedFrame != Long.MIN_VALUE && frameIndex - lastRecordedFrame > 4L) {
            // Long gaps (pause/menu or disabled frames) make the previous filtered volume stale enough
            // that reprojection is not worth the ghosting; drop it and restart accumulation.
            historyWasValid = false;
        }

        int quality = Math.clamp(CausticaConfig.Rt.Volumetrics.QUALITY.value(), 0, 4);
        float maxDistance = CausticaConfig.Rt.Volumetrics.MAX_DISTANCE.value();
        float distribution = CausticaConfig.Rt.Volumetrics.DEPTH_EXPONENT.value();
        float extinction = CausticaConfig.Rt.Volumetrics.EXTINCTION.value();
        float heightFalloff = CausticaConfig.Rt.Volumetrics.HEIGHT_FALLOFF.value();
        float heightOffset = CausticaConfig.Rt.Volumetrics.HEIGHT_OFFSET.value();
        float noiseAmount = CausticaConfig.Rt.Volumetrics.NOISE_AMOUNT.value();
        float noiseScale = CausticaConfig.Rt.Volumetrics.NOISE_SCALE.value();
        float albedo = CausticaConfig.Rt.Volumetrics.SCATTERING_ALBEDO.value();
        float anisotropy = CausticaConfig.Rt.Volumetrics.ANISOTROPY.value();
        float directional = CausticaConfig.Rt.Volumetrics.GOD_RAYS.value() ? 1.0f : 0.0f;
        float giStrength = CausticaConfig.Rt.Volumetrics.AMBIENT_STRENGTH.value();
        float temporalWeight = CausticaConfig.Rt.Volumetrics.TEMPORAL_WEIGHT.value();
        float causticStrength = 0f;
        float[] tint = parseFloat3(CausticaConfig.Rt.Volumetrics.TINT.get(), new float[]{1f, 1f, 1f});

        if (submerged) {
            maxDistance = CausticaConfig.Rt.Volumetrics.UNDERWATER_MAX_DISTANCE.value();
            extinction = CausticaConfig.Rt.Volumetrics.UNDERWATER_EXTINCTION.value();
            heightFalloff = 0f;
            albedo = 1f;
            anisotropy = CausticaConfig.Rt.Volumetrics.UNDERWATER_ANISOTROPY.value();
            causticStrength = CausticaConfig.Rt.Volumetrics.UNDERWATER_CAUSTIC_STRENGTH.value();
            noiseAmount = Math.min(noiseAmount, 0.15f);
            tint = parseFloat3(CausticaConfig.Rt.Volumetrics.UNDERWATER_TINT.get(),
                    new float[]{0.15f, 0.45f, 0.5f});
        }
        float[] albedoTint = {Math.clamp(albedo * tint[0], 0f, 1f),
                Math.clamp(albedo * tint[1], 0f, 1f), Math.clamp(albedo * tint[2], 0f, 1f)};

        long signature = signature(maxDistance, distribution, extinction, heightFalloff, albedoTint,
                anisotropy, noiseAmount, noiseScale, temporalWeight, giStrength, causticStrength);
        if (signature != opticalSignature) {
            opticalSignature = signature;
            historyWasValid = false;
        }

        int flags = 0;
        if (enabled) flags |= 0b01;
        if (historyWasValid) flags |= 0b10;
        if (submerged) flags |= 0b100;
        flags |= (QUALITY_GI_SAMPLES[quality] & 0xff) << 8;
        flags |= (QUALITY_GI_BOUNCES[quality] & 0xf) << 16;
        flags |= (QUALITY_PROBE_DIV_XY[quality] & 0xf) << 20;
        flags |= (QUALITY_PROBE_DIV_Z[quality] & 0xf) << 24;

        float baseHeightRebased = seaLevel + heightOffset - rebaseY;
        return new FrameData(
                new Int4(grid.width(), grid.height(), grid.depth(), flags),
                new Float4(maxDistance, distribution, extinction, heightFalloff),
                new Float4(albedoTint[0], albedoTint[1], albedoTint[2], anisotropy),
                new Float4(baseHeightRebased, noiseAmount, noiseScale, temporalWeight),
                new Float4(rebaseX, rebaseY, rebaseZ, timeSeconds),
                new Float4(directional, giStrength, causticStrength, 0f),
                true, signature);
    }

    /**
     * Record the three volumetric passes between Pass A and Pass B. Pass A wrote volumeDepth (fog clip)
     * immediately before; the guards and barriers here keep volume writes ordered before the read by the
     * next pass and by the final trace's {@code applyFroxelFog}.
     */
    public void record(RtContext ctx, VkCommandBuffer cmd, RtPipeline active,
                       ByteBuffer pushConstants, long frameIndex) {
        requireReady();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "volumetric inject");
                 RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.volumetricInject")) {
                // Inject at the lighting probe grid (half XY), filter/integrate at the full fog grid.
                active.trace(cmd, scattering.width, scattering.height, scattering.depth,
                        pushConstants, INJECT_RAYGEN);
            }
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
            try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "volumetric filter");
                 RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.volumetricFilter")) {
                // One thread per cell (z included): each cell reprojects itself into the previous
                // camera, so the dispatch must be 3D even though integrate below is 1D.
                active.trace(cmd, grid.width(), grid.height(), grid.depth(), pushConstants, FILTER_RAYGEN);
            }
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
            try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "volumetric integrate");
                 RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.volumetricIntegrate")) {
                active.trace(cmd, grid.width(), grid.height(), 1, pushConstants, INTEGRATE_RAYGEN);
            }
        }
        lastRecordedFrame = frameIndex;
        // The passes ran (or were recorded as no-ops for cells behind interfaces); the next frame may
        // use the lane written now as its history.
        historyValid = true;
    }

    /** Drop temporal history (F3+A, resource epoch, parameter changes). */
    public void invalidateHistory() {
        historyValid = false;
    }

    public void destroy() {
        destroyImages();
        grid = null;
        deviceExtentCap = 0;
        historyValid = false;
        lastRecordedFrame = Long.MIN_VALUE;
    }

    public boolean available() {
        return grid != null;
    }

    public RtFroxelGrid grid() {
        return grid;
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
        for (int i = 0; i < filtered.length; i++) {
            if (filtered[i] != null) {
                filtered[i].destroy();
                filtered[i] = null;
            }
        }
        if (integrated != null) {
            integrated.destroy();
            integrated = null;
        }
        lastRecordedFrame = Long.MIN_VALUE;
    }

    private void requireReady() {
        if (grid == null || volumeDepth == null || scattering == null
                || filtered[0] == null || filtered[1] == null || integrated == null) {
            throw new IllegalStateException("Volumetric resources are not ready");
        }
    }

    private static RtFroxelGrid wantedGrid(int renderWidth, int renderHeight, int extentCap) {
        int quality = Math.clamp(CausticaConfig.Rt.Volumetrics.QUALITY.value(), 0, 4);
        RtFroxelGrid wanted = RtFroxelGrid.forRenderSize(renderWidth, renderHeight,
                QUALITY_PIXEL_SIZE[quality], QUALITY_DEPTH_SLICES[quality]);
        if (extentCap > 0) {
            wanted = new RtFroxelGrid(Math.min(wanted.width(), extentCap),
                    Math.min(wanted.height(), extentCap), Math.min(wanted.depth(), extentCap),
                    wanted.pixelSize());
        }
        return wanted;
    }

    private static long signature(float maxDistance, float distribution, float extinction,
                                  float heightFalloff, float[] albedoTint, float anisotropy,
                                  float noiseAmount, float noiseScale, float temporalWeight,
                                  float giStrength, float causticStrength) {
        long h = 0xC1;
        long[] values = {
                Float.floatToRawIntBits(maxDistance) & 0xffffffffL,
                Float.floatToRawIntBits(distribution) & 0xffffffffL,
                Float.floatToRawIntBits(extinction) & 0xffffffffL,
                Float.floatToRawIntBits(heightFalloff) & 0xffffffffL,
                Float.floatToRawIntBits(albedoTint[0]) & 0xffffffffL,
                Float.floatToRawIntBits(albedoTint[1]) & 0xffffffffL,
                Float.floatToRawIntBits(albedoTint[2]) & 0xffffffffL,
                Float.floatToRawIntBits(anisotropy) & 0xffffffffL,
                Float.floatToRawIntBits(noiseAmount) & 0xffffffffL,
                Float.floatToRawIntBits(noiseScale) & 0xffffffffL,
                Float.floatToRawIntBits(temporalWeight) & 0xffffffffL,
                Float.floatToRawIntBits(giStrength) & 0xffffffffL,
                Float.floatToRawIntBits(causticStrength) & 0xffffffffL
        };
        for (long value : values) {
            h ^= value + 0x9e3779b97f4a7c15L + (h << 6L) + (h >>> 2L);
        }
        return h;
    }

    private static float[] parseFloat3(String value, float[] fallback) {
        if (value == null) {
            return fallback;
        }
        String[] parts = value.trim().split("[,\\s]+");
        if (parts.length != 3) {
            return fallback;
        }
        float[] out = new float[3];
        try {
            for (int i = 0; i < 3; i++) {
                out[i] = Float.parseFloat(parts[i]);
            }
        } catch (NumberFormatException e) {
            CausticaMod.LOGGER.warn("Ignoring invalid volumetric tint string '{}'", value);
            return fallback;
        }
        return out;
    }
}
