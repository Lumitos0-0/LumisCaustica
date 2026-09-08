package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.RtFrameStats;
import dev.comfyfluffy.caustica.rt.RtGpuExecutor;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.VolumePushData;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkSamplerCreateInfo;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;
import org.lwjgl.vulkan.VkWriteDescriptorSetAccelerationStructureKHR;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;

/**
 * Volumetric fog: sun and moon shafts plus the sky's contribution to a participating medium.
 *
 * <p>The work is split across two representations because a sun shaft and a fog bed are not the same
 * kind of signal. The sky's contribution is smooth in every direction and is integrated in a froxel
 * grid at a fraction of the render resolution, where it is cheap and where its blur is invisible. The
 * sun and moon are not smooth: the solar disc subtends 0.53 degrees, so a beam's penumbra can be under
 * a pixel wide near its occluder, and a grid coarse enough to be affordable would smear that to a dozen
 * pixels of uniform softness. That half runs per pixel, in screen space, with its own denoiser.
 *
 * <p>Six dispatches, in order:
 *
 * <ol>
 *   <li><b>grid</b> — one phase-sampled sky direction and one visibility ray per froxel, accumulated
 *       into the froxel volume.</li>
 *   <li><b>scan</b> — prefix-integrates that volume along the view axis so the composite can read it
 *       with one trilinear fetch.</li>
 *   <li><b>shaft</b> — one importance-sampled scattering event and one solar-disc visibility ray per
 *       render pixel.</li>
 *   <li><b>resolve</b> — temporal accumulation of the shaft buffer, reprojected on the scattering
 *       centroid.</li>
 *   <li><b>filter</b> — a narrow cross-bilateral pass whose width shrinks as history matures.</li>
 *   <li><b>composite</b> — applies transmittance and adds both terms to the display-resolution
 *       image.</li>
 * </ol>
 *
 * <p>The first two run before the trace, the next three after it, and the composite after DLSS-RR: RR
 * is guided by surface depth, normal and motion, and in-scatter has no surface, so anything volumetric
 * fed through it is reprojected by the wrong vector field.
 *
 * <p>Two descriptor sets are allocated and alternated per frame. The pairs that alternate are the
 * froxel volume, the shaft accumulation buffer and its age buffer — this is the temporal ping-pong. The
 * TLAS is written into the frame's set on the same two-slot rotation the trace pipeline uses, guarded
 * by the same graphics-use timeline.
 */
public final class RtVolumetrics {
    private static final String SHADER_DIR = "/caustica/shaders/pipelines/volumetrics/";
    private static final int SET_RING = 2;
    private static final int GRID_GROUP = 4;
    private static final int SCREEN_GROUP = 8;
    /** Froxel volumes and the shaft buffers; rgba16f is a mandatory storage format at every image type. */
    private static final int COLOR_FORMAT = VK10.VK_FORMAT_R16G16B16A16_SFLOAT;
    /** Shaft history age. r32f rather than a narrower format because only r32f is mandatory for storage. */
    private static final int AGE_FORMAT = VK10.VK_FORMAT_R32_SFLOAT;

    /**
     * Grid resolution ladder. Only the froxel grid moves: the shaft pass stays at render resolution at
     * every level, since halving it would cost precisely the sharpness it exists to deliver.
     *
     * <p>Cost is {@code slices / scale^2} rays per render pixel for the grid, plus a flat one for the
     * shafts — so 1.08 at Low and 2.0 at Ultra.
     */
    public enum QualityPreset {
        LOW(24, 48),
        MEDIUM(16, 64),
        HIGH(12, 64),
        ULTRA(8, 64);

        public final int gridScale;
        public final int slices;

        QualityPreset(int gridScale, int slices) {
            this.gridScale = gridScale;
            this.slices = slices;
        }

        public static QualityPreset of(int ordinal) {
            QualityPreset[] all = values();
            return all[Math.clamp(ordinal, 0, all.length - 1)];
        }
    }

    private final RtContext ctx;
    private final long linear2d;
    private final long linear3d;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long[] descriptorSets = new long[SET_RING];
    private final RtGpuExecutor.TrackedGraphicsUse[] descriptorSetUses =
            new RtGpuExecutor.TrackedGraphicsUse[SET_RING];
    private final long pipelineLayout;
    private final long gridPipeline;
    private final long scanPipeline;
    private final long shaftPipeline;
    private final long resolvePipeline;
    private final long filterPipeline;
    private final long compositePipeline;

    // Ping-pong pairs, indexed by the descriptor set they are the CURRENT target of.
    private final RtImage[] froxel = new RtImage[SET_RING];
    private final RtImage[] shaftAccum = new RtImage[SET_RING];
    private final RtImage[] shaftAge = new RtImage[SET_RING];
    private RtImage froxelIntegrated;
    private RtImage shaftRaw;
    private RtImage shaftFiltered;

    private int gridX;
    private int gridY;
    private int gridZ;
    private int renderW;
    private int renderH;
    private int displayW;
    private int displayH;
    private int currentSet;
    // Views currently written into both descriptor sets, so a target recreated at an unchanged size
    // cannot leave a dangling descriptor behind.
    private long boundSceneView;
    private long boundDepthView;
    private long boundSkyViewLut;
    private long boundTransmittanceLut;
    /** Cleared whenever anything that invalidates every temporal buffer at once changes. */
    private boolean historyValid;
    /** Rebased world Y of the previous frame, to detect the terrain rebase that teleports the fog. */
    private int previousRebaseY = Integer.MIN_VALUE;
    private boolean destroyed;

    private RtVolumetrics(RtContext ctx, long linear2d, long linear3d, long descriptorSetLayout,
                          long descriptorPool, long[] sets, long pipelineLayout, long gridPipeline,
                          long scanPipeline, long shaftPipeline, long resolvePipeline,
                          long filterPipeline, long compositePipeline) {
        this.ctx = ctx;
        this.linear2d = linear2d;
        this.linear3d = linear3d;
        this.descriptorSetLayout = descriptorSetLayout;
        this.descriptorPool = descriptorPool;
        System.arraycopy(sets, 0, this.descriptorSets, 0, SET_RING);
        this.pipelineLayout = pipelineLayout;
        this.gridPipeline = gridPipeline;
        this.scanPipeline = scanPipeline;
        this.shaftPipeline = shaftPipeline;
        this.resolvePipeline = resolvePipeline;
        this.filterPipeline = filterPipeline;
        this.compositePipeline = compositePipeline;
        for (int i = 0; i < SET_RING; i++) {
            this.descriptorSetUses[i] = new RtGpuExecutor.TrackedGraphicsUse();
        }
    }

    public static RtVolumetrics create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer handle = stack.mallocLong(1);
            long linear2d = createSampler(ctx, stack, handle, "volumetrics 2D sampler");
            long linear3d = createSampler(ctx, stack, handle, "volumetrics 3D sampler");

            VkDescriptorSetLayoutBinding.Buffer bindings =
                    VkDescriptorSetLayoutBinding.calloc(VOLUME_BINDING_COUNT, stack);
            for (int i = 0; i < VOLUME_BINDING_COUNT; i++) {
                bindings.get(i).binding(i).descriptorCount(1)
                        .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                        .descriptorType(descriptorTypeOf(i));
            }
            VkDescriptorSetLayoutCreateInfo layoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(bindings);
            check(VK10.vkCreateDescriptorSetLayout(vk, layoutInfo, null, handle),
                    "vkCreateDescriptorSetLayout(volumetrics)");
            long descriptorSetLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                    descriptorSetLayout, "volumetrics descriptor set layout");

            int storageCount = 0;
            int sampledCount = 0;
            int tlasCount = 0;
            for (int i = 0; i < VOLUME_BINDING_COUNT; i++) {
                int type = descriptorTypeOf(i);
                if (type == VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                    storageCount++;
                } else if (type == VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                    sampledCount++;
                } else {
                    tlasCount++;
                }
            }
            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(3, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(storageCount * SET_RING);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(sampledCount * SET_RING);
            poolSizes.get(2).type(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                    .descriptorCount(tlasCount * SET_RING);
            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(SET_RING).pPoolSizes(poolSizes);
            check(VK10.vkCreateDescriptorPool(vk, poolInfo, null, handle),
                    "vkCreateDescriptorPool(volumetrics)");
            long descriptorPool = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                    descriptorPool, "volumetrics descriptor pool");

            LongBuffer layouts = stack.mallocLong(SET_RING);
            for (int i = 0; i < SET_RING; i++) {
                layouts.put(i, descriptorSetLayout);
            }
            VkDescriptorSetAllocateInfo allocateInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(descriptorPool).pSetLayouts(layouts);
            LongBuffer setHandles = stack.mallocLong(SET_RING);
            check(VK10.vkAllocateDescriptorSets(vk, allocateInfo, setHandles),
                    "vkAllocateDescriptorSets(volumetrics)");
            long[] sets = new long[SET_RING];
            for (int i = 0; i < SET_RING; i++) {
                sets[i] = setHandles.get(i);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, sets[i],
                        "volumetrics descriptor set " + i);
            }

            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
            pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .offset(0).size(VolumePushData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo pipelineLayoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default().pSetLayouts(stack.longs(descriptorSetLayout))
                    .pPushConstantRanges(pushRange);
            check(VK10.vkCreatePipelineLayout(vk, pipelineLayoutInfo, null, handle),
                    "vkCreatePipelineLayout(volumetrics)");
            long pipelineLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                    pipelineLayout, "volumetrics pipeline layout");

            return new RtVolumetrics(ctx, linear2d, linear3d, descriptorSetLayout, descriptorPool, sets,
                    pipelineLayout,
                    createComputePipeline(ctx, stack, pipelineLayout, "grid.comp.spv", "volumetric grid"),
                    createComputePipeline(ctx, stack, pipelineLayout, "scan.comp.spv", "volumetric scan"),
                    createComputePipeline(ctx, stack, pipelineLayout, "shaft.comp.spv", "volumetric shaft"),
                    createComputePipeline(ctx, stack, pipelineLayout, "resolve.comp.spv", "volumetric resolve"),
                    createComputePipeline(ctx, stack, pipelineLayout, "filter.comp.spv", "volumetric filter"),
                    createComputePipeline(ctx, stack, pipelineLayout, "composite.comp.spv", "volumetric composite"));
        }
    }

    /** Descriptor type of a binding, mirroring shaders/pipelines/volumetrics/bindings.slang. */
    private static int descriptorTypeOf(int binding) {
        if (binding == VOLUME_TLAS) {
            return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        }
        if (binding == VOLUME_ACCUM_PREV || binding == VOLUME_ACCUM_CUR
                || binding == VOLUME_INTEGRATED_LUT || binding == VOLUME_SHAFT_ACCUM_PREV
                || binding == VOLUME_SKY_VIEW || binding == VOLUME_TRANSMITTANCE) {
            return VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        }
        return VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }

    /**
     * (Re)creates the volumes and screen-resolution buffers and rebinds both descriptor sets.
     *
     * <p>The image views are part of the guard, not just the dimensions: the caller recreates its scene
     * and depth targets on changes that do not always move the resolution, and a stale view left in
     * these descriptor sets is a use-after-free the validation layers would only catch at dispatch.
     *
     * <p>The caller must have waited for the device to be idle, as it does for every other sized
     * resource.
     */
    public void ensureResources(int renderWidth, int renderHeight, int displayWidth, int displayHeight,
                                long sceneView, long depthView, long skyViewLut, long transmittanceLut,
                                long skyLutSampler) {
        QualityPreset requested = QualityPreset.of(CausticaConfig.Rt.Volumetrics.QUALITY.value());
        int wantGridX = Math.max(1, (renderWidth + requested.gridScale - 1) / requested.gridScale);
        int wantGridY = Math.max(1, (renderHeight + requested.gridScale - 1) / requested.gridScale);
        boolean geometryChanged = froxelIntegrated == null || wantGridX != gridX || wantGridY != gridY
                || requested.slices != gridZ || renderWidth != renderW || renderHeight != renderH
                || displayWidth != displayW || displayHeight != displayH;
        boolean viewsChanged = sceneView != boundSceneView || depthView != boundDepthView
                || skyViewLut != boundSkyViewLut || transmittanceLut != boundTransmittanceLut;
        if (!geometryChanged && !viewsChanged) {
            return;
        }

        if (geometryChanged) {
            destroyImages();
            gridX = wantGridX;
            gridY = wantGridY;
            gridZ = requested.slices;
            renderW = renderWidth;
            renderH = renderHeight;
            displayW = displayWidth;
            displayH = displayHeight;
            historyValid = false;

            for (int i = 0; i < SET_RING; i++) {
                froxel[i] = ctx.createStorageImage3D(gridX, gridY, gridZ, COLOR_FORMAT,
                        "volumetric froxel " + i + " " + gridX + "x" + gridY + "x" + gridZ);
                shaftAccum[i] = ctx.createStorageImage(renderW, renderH, COLOR_FORMAT,
                        "volumetric shaft accum " + i + " " + renderW + "x" + renderH);
                shaftAge[i] = ctx.createStorageImage(renderW, renderH, AGE_FORMAT,
                        "volumetric shaft age " + i + " " + renderW + "x" + renderH);
            }
            froxelIntegrated = ctx.createStorageImage3D(gridX, gridY, gridZ, COLOR_FORMAT,
                    "volumetric froxel integrated " + gridX + "x" + gridY + "x" + gridZ);
            shaftRaw = ctx.createStorageImage(renderW, renderH, COLOR_FORMAT,
                    "volumetric shaft raw " + renderW + "x" + renderH);
            shaftFiltered = ctx.createStorageImage(renderW, renderH, COLOR_FORMAT,
                    "volumetric shaft filtered " + renderW + "x" + renderH);
        }

        boundSceneView = sceneView;
        boundDepthView = depthView;
        boundSkyViewLut = skyViewLut;
        boundTransmittanceLut = transmittanceLut;
        for (int set = 0; set < SET_RING; set++) {
            writeSet(set, sceneView, depthView, skyViewLut, transmittanceLut, skyLutSampler);
        }
    }

    private void writeSet(int set, long sceneView, long depthView, long skyViewLut,
                          long transmittanceLut, long skyLutSampler) {
        int other = 1 - set;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer images =
                    VkDescriptorImageInfo.calloc(VOLUME_BINDING_COUNT, stack);
            VkWriteDescriptorSet.Buffer writes =
                    VkWriteDescriptorSet.calloc(VOLUME_BINDING_COUNT - 1, stack);
            int[] bindingOf = {
                    VOLUME_ACCUM_IMAGE, VOLUME_ACCUM_PREV, VOLUME_ACCUM_CUR, VOLUME_INTEGRATED_IMAGE,
                    VOLUME_INTEGRATED_LUT, VOLUME_SHAFT_RAW, VOLUME_SHAFT_ACCUM, VOLUME_SHAFT_ACCUM_PREV,
                    VOLUME_SHAFT_AGE, VOLUME_SHAFT_AGE_PREV, VOLUME_SHAFT_FILTERED, VOLUME_SCENE,
                    VOLUME_DEPTH, VOLUME_SKY_VIEW, VOLUME_TRANSMITTANCE,
            };
            long[] viewOf = {
                    froxel[set].view, froxel[other].view, froxel[set].view, froxelIntegrated.view,
                    froxelIntegrated.view, shaftRaw.view, shaftAccum[set].view, shaftAccum[other].view,
                    shaftAge[set].view, shaftAge[other].view, shaftFiltered.view, sceneView,
                    depthView, skyViewLut, transmittanceLut,
            };
            for (int i = 0; i < bindingOf.length; i++) {
                int binding = bindingOf[i];
                int type = descriptorTypeOf(binding);
                images.get(binding).imageView(viewOf[i]).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                if (type == VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                    images.get(binding).sampler(samplerFor(binding, skyLutSampler));
                }
                writes.get(i).sType$Default().dstSet(descriptorSets[set]).dstBinding(binding)
                        .descriptorCount(1).descriptorType(type)
                        .pImageInfo(VkDescriptorImageInfo.create(images.address(binding), 1));
            }
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
    }

    private long samplerFor(int binding, long skyLutSampler) {
        if (binding == VOLUME_SKY_VIEW || binding == VOLUME_TRANSMITTANCE) {
            return skyLutSampler;
        }
        return binding == VOLUME_SHAFT_ACCUM_PREV ? linear2d : linear3d;
    }

    /**
     * Rotate to the next descriptor set and bind this frame's TLAS into it. Must be called once per
     * frame, after the TLAS build is prepared and before {@link #record}: the rotation is also the
     * temporal ping-pong, so skipping it would make a pass read the buffer it is writing.
     */
    public void beginFrame(long tlas, RtGpuExecutor.GraphicsUse graphicsUse,
                           RtGpuExecutor.GraphicsUseWaiter graphicsUseWaiter, int rebaseY) {
        currentSet = (currentSet + 1) % SET_RING;
        RtGpuExecutor.TrackedGraphicsUse slotUse = descriptorSetUses[currentSet];
        graphicsUseWaiter.await(slotUse);
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkWriteDescriptorSetAccelerationStructureKHR asWrite =
                    VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack)
                            .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR)
                            .pAccelerationStructures(stack.longs(tlas));
            VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(1, stack);
            write.get(0).sType$Default().pNext(asWrite.address()).dstSet(descriptorSets[currentSet])
                    .dstBinding(VOLUME_TLAS).descriptorCount(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
            VK10.vkUpdateDescriptorSets(ctx.vk(), write, null);
        }
        slotUse.mark(graphicsUse);

        // A terrain rebase moves the world origin, and the fog's height profile is anchored in rebased
        // Y, so every stored froxel suddenly describes a different altitude. Nothing in the reprojection
        // can see that, hence the explicit reset.
        if (rebaseY != previousRebaseY) {
            historyValid = false;
            previousRebaseY = rebaseY;
        }
    }

    /** Froxel-grid half: the two passes that must run before the trace overwrites nothing they read. */
    public void recordGrid(VkCommandBuffer cmd, MemoryStack stack, long worldPushAddress) {
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "volumetric froxel grid");
             RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.volumeGrid")) {
            bind(cmd, stack, worldPushAddress);
            dispatch(cmd, stack, gridPipeline,
                    groups(gridX, GRID_GROUP), groups(gridY, GRID_GROUP), groups(gridZ, GRID_GROUP));
            dispatch(cmd, stack, scanPipeline,
                    groups(gridX, SCREEN_GROUP), groups(gridY, SCREEN_GROUP), 1);
        }
    }

    /**
     * Screen-space half: the shaft estimator and its denoiser. Needs gDepth, so it runs after the trace
     * and before the composite.
     */
    public void recordShafts(VkCommandBuffer cmd, MemoryStack stack, long worldPushAddress) {
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "volumetric shafts");
             RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.volumeShafts")) {
            bind(cmd, stack, worldPushAddress);
            int groupsX = groups(renderW, SCREEN_GROUP);
            int groupsY = groups(renderH, SCREEN_GROUP);
            dispatch(cmd, stack, shaftPipeline, groupsX, groupsY, 1);
            dispatch(cmd, stack, resolvePipeline, groupsX, groupsY, 1);
            dispatch(cmd, stack, filterPipeline, groupsX, groupsY, 1);
        }
    }

    /**
     * Applies transmittance and both in-scatter terms to the display-resolution image, in place. Runs
     * after DLSS-RR (which cannot reproject a surfaceless signal) but before the exposure histogram and
     * bloom, so that fog meters and blooms like any other part of the image.
     *
     * <p>This is also where history becomes valid: every buffer the next frame reads has been written by
     * the time this returns.
     */
    public void recordComposite(VkCommandBuffer cmd, MemoryStack stack, long worldPushAddress) {
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "volumetric composite");
             RtFrameStats.Scope ignoredStats = RtFrameStats.FRAME.stage("frame.volumeComposite")) {
            bind(cmd, stack, worldPushAddress);
            dispatch(cmd, stack, compositePipeline,
                    groups(displayW, SCREEN_GROUP), groups(displayH, SCREEN_GROUP), 1);
        }
        historyValid = true;
    }

    /** Drops every temporal buffer, e.g. after a teleport or a dimension change. */
    public void invalidateHistory() {
        historyValid = false;
    }

    /**
     * The medium as the shaders consume it, for {@code WorldPush.fog}. Built here rather than in the
     * frame recorder so the derivation from the artist-facing settings lives next to the passes that
     * integrate it.
     *
     * @param rebaseY the terrain rebase origin's Y, which the configured absolute world height is
     *                expressed relative to
     */
    public static WorldPushData.VolumetricMedium medium(int rebaseY) {
        float[] extinction = CausticaConfig.Rt.Volumetrics.extinctionPerBlock();
        float albedo = CausticaConfig.Rt.Volumetrics.ALBEDO.value();
        return new WorldPushData.VolumetricMedium(
                new WorldPushData.Float3(extinction[0], extinction[1], extinction[2]),
                CausticaConfig.Rt.Volumetrics.ANISOTROPY.value(),
                new WorldPushData.Float3(albedo, albedo, albedo),
                CausticaConfig.Rt.Volumetrics.HEIGHT_FALLOFF.value(),
                CausticaConfig.Rt.Volumetrics.HEIGHT_BASE.value() - rebaseY,
                CausticaConfig.Rt.Volumetrics.MAX_DISTANCE.value(),
                CausticaConfig.Rt.Volumetrics.SHAFT_INTENSITY.value(),
                CausticaConfig.Rt.Volumetrics.AMBIENT_INTENSITY.value());
    }

    private void bind(VkCommandBuffer cmd, MemoryStack stack, long worldPushAddress) {
        VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipelineLayout, 0, stack.longs(descriptorSets[currentSet]), null);
        ByteBuffer push = stack.malloc(VolumePushData.BYTE_SIZE);
        new VolumePushData(worldPushAddress, gridX, gridY, gridZ, renderW, renderH, displayW, displayH,
                historyValid ? 1 : 0,
                CausticaConfig.Rt.Volumetrics.SKY_OCCLUSION.value() ? 1 : 0,
                CausticaConfig.Rt.Volumetrics.MAX_ACCUM_FRAMES.value(),
                CausticaConfig.Rt.Volumetrics.DEBUG_VIEW.value()).write(push);
        VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
    }

    private void dispatch(VkCommandBuffer cmd, MemoryStack stack, long pipeline, int x, int y, int z) {
        VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        VK10.vkCmdDispatch(cmd, x, y, z);
        // Every pass consumes the previous one's image, so none of them may overlap.
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
    }

    private static int groups(int extent, int groupSize) {
        return (extent + groupSize - 1) / groupSize;
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, compositePipeline, null);
        VK10.vkDestroyPipeline(vk, filterPipeline, null);
        VK10.vkDestroyPipeline(vk, resolvePipeline, null);
        VK10.vkDestroyPipeline(vk, shaftPipeline, null);
        VK10.vkDestroyPipeline(vk, scanPipeline, null);
        VK10.vkDestroyPipeline(vk, gridPipeline, null);
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        VK10.vkDestroySampler(vk, linear3d, null);
        VK10.vkDestroySampler(vk, linear2d, null);
        destroyImages();
        destroyed = true;
    }

    private void destroyImages() {
        for (int i = 0; i < SET_RING; i++) {
            if (froxel[i] != null) {
                froxel[i].destroy();
                froxel[i] = null;
            }
            if (shaftAccum[i] != null) {
                shaftAccum[i].destroy();
                shaftAccum[i] = null;
            }
            if (shaftAge[i] != null) {
                shaftAge[i].destroy();
                shaftAge[i] = null;
            }
        }
        if (froxelIntegrated != null) {
            froxelIntegrated.destroy();
            froxelIntegrated = null;
        }
        if (shaftRaw != null) {
            shaftRaw.destroy();
            shaftRaw = null;
        }
        if (shaftFiltered != null) {
            shaftFiltered.destroy();
            shaftFiltered = null;
        }
    }

    private static long createSampler(RtContext ctx, MemoryStack stack, LongBuffer handle, String label) {
        // CLAMP_TO_EDGE on all three axes. Reprojected fetches routinely land just outside the volume,
        // and a wrapped tap there would pull in-scatter from the opposite side of the screen.
        VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                .magFilter(VK10.VK_FILTER_LINEAR).minFilter(VK10.VK_FILTER_LINEAR)
                .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                .minLod(0.0f).maxLod(0.0f);
        check(VK10.vkCreateSampler(ctx.vk(), samplerInfo, null, handle), "vkCreateSampler(" + label + ")");
        long sampler = handle.get(0);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, label);
        return sampler;
    }

    private static long createComputePipeline(RtContext ctx, MemoryStack stack, long layout,
                                              String shader, String label) {
        VkDevice vk = ctx.vk();
        long module = loadModule(vk, stack, SHADER_DIR + shader);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, module, label + " module");
        VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                .sType$Default().stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                .module(module).pName(stack.UTF8("main"));
        VkComputePipelineCreateInfo.Buffer info = VkComputePipelineCreateInfo.calloc(1, stack);
        info.get(0).sType$Default().stage(stage).layout(layout);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, info, null, handle),
                "vkCreateComputePipelines(" + shader + ")");
        VK10.vkDestroyShaderModule(vk, module, null);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, handle.get(0), label);
        return handle.get(0);
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String resource) {
        byte[] bytes;
        try (InputStream input = RtVolumetrics.class.getResourceAsStream(resource)) {
            if (input == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + resource);
            }
            bytes = input.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + resource, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo moduleInfo = VkShaderModuleCreateInfo.calloc(stack)
                    .sType$Default().pCode(code);
            LongBuffer module = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, moduleInfo, null, module),
                    "vkCreateShaderModule(" + resource + ")");
            return module.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}
