package dev.comfyfluffy.caustica.rt.pipeline;

import org.joml.Matrix4fc;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkClearColorValue;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkImageCopy;
import org.lwjgl.vulkan.VkImageSubresourceRange;
import org.lwjgl.vulkan.VkMemoryBarrier;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.rt.RtComposite;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.VolumetricPushData;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.VOLUME_DEPTH_IMAGE;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.VOLUME_FOG_CURRENT;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.VOLUME_FOG_HISTORY;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.VOLUME_FOG_HISTORY_CONF;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.VOLUME_FOG_VOLUME;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.VOLUME_OUTPUT_IMAGE;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.VOLUME_TLAS;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VkWriteDescriptorSetAccelerationStructureKHR;

/**
 * Volumetric fog (froxel) pipeline.
 *
 * <p>Implements the four-pass architecture:
 * <ol>
 *   <li><b>inject</b> — per-froxel density + one shadow ray, writes {@code fogCurrent}.
 *   <li><b>spatial</b> — edge-aware 3x3 XY bilateral filter, writes {@code fogVolume}.
 *   <li><b>temporal</b> — stable-center reprojection into {@code fogHistory},
 *       neighborhood clamping, confidence accumulation, writes {@code fogVolume}.
 *   <li><b>integrate</b> — per-pixel front-to-back exponential integration,
 *       composites L + T * scene directly back into the render-res RT output
 *       (so DLSS-RR denoises fog + surfaces together).
 * </ol>
 *
 * <p>Camera matrices live in a small host-visible storage buffer (BDA-addressed via a
 * pointer in push constants) so the scalar push block stays within the 128-byte guaranteed
 * minimum. Triple-buffered (PUSH_RING slots) to avoid writing a buffer the GPU is still reading.
 *
 * <p>All four passes share one descriptor set so a single VkDescriptorSet is re-bound across
 * dispatches. After integrate we copy {@code fogVolume} into {@code fogHistory} for the next frame.
 */
public final class RtVolumetricFog {

    private static final String SHADER_DIR = "/caustica/shaders/pipelines/volumetric/";
    private static final int FOG_FORMAT = VK10.VK_FORMAT_R16G16B16A16_SFLOAT;
    private static final int CONF_FORMAT = VK10.VK_FORMAT_R16_SFLOAT;

    // FroxelMatrices: 3 float4x4 = 3 * 64 = 192 bytes.
    private static final long MATRICES_BUF_SIZE = 192;
    private static final int PUSH_RING = 3;

    private final RtContext ctx;

    private long descriptorSetLayout;
    private long descriptorPool;
    private long descriptorSet;
    private long pipelineLayout;

    private long injectPipeline;
    private long spatialPipeline;
    private long temporalPipeline;
    private long integratePipeline;

    private final RtBuffer[] matricesRing = new RtBuffer[PUSH_RING];
    private int matricesSlot;

    private RtImage fogCurrent;
    private RtImage fogVolume;
    private RtImage fogHistory;
    private RtImage fogHistoryConfidence;

    private int gridW = -1;
    private int gridH = -1;
    private int gridZ = 64;
    private int renderW = -1;
    private int renderH = -1;
    private boolean destroyed;

    private long lastTlas;
    private long lastDepthView;
    private long lastOutputView;

    private float lastNear = -1f;
    private float lastFar = -1f;
    private int lastGridZ = -1;
    private int lastDivisor = -1;

    private RtVolumetricFog(RtContext ctx) {
        this.ctx = ctx;
    }

    public static RtVolumetricFog create(RtContext ctx) {
        RtVolumetricFog self = new RtVolumetricFog(ctx);
        self.init();
        return self;
    }

    private void init() {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            // ---- Descriptor set layout: TLAS(AS) + 6 storage images.
            VkDescriptorSetLayoutBinding.Buffer binds = VkDescriptorSetLayoutBinding.calloc(7, stack);
            binds.get(VOLUME_TLAS).binding(VOLUME_TLAS)
                    .descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                    .descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            int[] storageBindings = {
                    VOLUME_FOG_VOLUME, VOLUME_FOG_CURRENT, VOLUME_FOG_HISTORY,
                    VOLUME_FOG_HISTORY_CONF, VOLUME_DEPTH_IMAGE, VOLUME_OUTPUT_IMAGE
            };
            for (int b : storageBindings) {
                binds.get(b).binding(b)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1)
                        .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }

            VkDescriptorSetLayoutCreateInfo dslci = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default()
                    .pBindings(binds);
            LongBuffer p = stack.mallocLong(1);
            check(VK10.vkCreateDescriptorSetLayout(vk, dslci, null, p),
                    "vkCreateDescriptorSetLayout(rt volume fog)");
            descriptorSetLayout = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                    descriptorSetLayout, "volume fog descriptor set layout");

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(1);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(6);
            VkDescriptorPoolCreateInfo dpci = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default()
                    .maxSets(1).pPoolSizes(poolSizes);
            check(VK10.vkCreateDescriptorPool(vk, dpci, null, p), "vkCreateDescriptorPool(rt volume fog)");
            descriptorPool = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                    descriptorPool, "volume fog descriptor pool");

            VkDescriptorSetAllocateInfo dsai = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                    .descriptorPool(descriptorPool).pSetLayouts(stack.longs(descriptorSetLayout));
            LongBuffer pSet = stack.mallocLong(1);
            check(VK10.vkAllocateDescriptorSets(vk, dsai, pSet), "vkAllocateDescriptorSets(rt volume fog)");
            descriptorSet = pSet.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET,
                    descriptorSet, "volume fog descriptor set");

            VkPushConstantRange.Buffer pcr = VkPushConstantRange.calloc(1, stack)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .offset(0).size(VolumetricPushData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo plci = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                    .pSetLayouts(stack.longs(descriptorSetLayout)).pPushConstantRanges(pcr);
            check(VK10.vkCreatePipelineLayout(vk, plci, null, p), "vkCreatePipelineLayout(rt volume fog)");
            pipelineLayout = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                    pipelineLayout, "volume fog pipeline layout");

            injectPipeline = createComputePipeline(vk, stack, "inject.comp.spv", "volume fog inject");
            spatialPipeline = createComputePipeline(vk, stack, "spatial.comp.spv", "volume fog spatial");
            temporalPipeline = createComputePipeline(vk, stack, "temporal.comp.spv", "volume fog temporal");
            integratePipeline = createComputePipeline(vk, stack, "integrate.comp.spv", "volume fog integrate");
        }
    }

    private long createComputePipeline(VkDevice vk, MemoryStack stack, String name, String label) {
        long module = loadModule(vk, stack, name);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, module, label + " module");
        VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack).sType$Default()
                .stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(module).pName(stack.UTF8("main"));
        VkComputePipelineCreateInfo cpci = VkComputePipelineCreateInfo.calloc(stack).sType$Default()
                .stage(stage).layout(pipelineLayout);
        LongBuffer p = stack.mallocLong(1);
        check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, cpci, null, p),
                "vkCreateComputePipelines(" + name + ")");
        long pipeline = p.get(0);
        VK10.vkDestroyShaderModule(vk, module, null);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, pipeline, label);
        return pipeline;
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String name) {
        byte[] bytes;
        try (InputStream in = RtVolumetricFog.class.getResourceAsStream(SHADER_DIR + name)) {
            if (in == null) throw new IllegalStateException("missing SPIR-V: " + SHADER_DIR + name);
            bytes = in.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V: " + SHADER_DIR + name, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo mci = VkShaderModuleCreateInfo.calloc(stack).sType$Default().pCode(code);
            LongBuffer p = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, mci, null, p), "vkCreateShaderModule(" + name + ")");
            return p.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }

    private void ensureMatricesBuffer() {
        if (matricesRing[0] != null) return;
        for (int i = 0; i < PUSH_RING; i++) {
            matricesRing[i] = ctx.createBuffer(MATRICES_BUF_SIZE,
                    VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    true, "volumetric matrices " + i);
        }
    }

    private void ensureResources(int renderWidth, int renderHeight, int divisor) {
        int w = Math.max(1, renderWidth / divisor);
        int h = Math.max(1, renderHeight / divisor);
        int z = CausticaConfig.Rt.Fog.DEPTH_SLICES.value();
        if (fogCurrent != null && gridW == w && gridH == h && gridZ == z
                && renderW == renderWidth && renderH == renderHeight) {
            return;
        }
        ctx.waitIdle();
        destroyImages();
        renderW = renderWidth;
        renderH = renderHeight;
        gridW = w;
        gridH = h;
        gridZ = z;

        fogCurrent = ctx.createStorageImage3D(w, h, gridZ, FOG_FORMAT,
                "volumetric fog injected " + w + "x" + h + "x" + gridZ);
        fogVolume = ctx.createStorageImage3D(w, h, gridZ, FOG_FORMAT,
                "volumetric fog accumulated " + w + "x" + h + "x" + gridZ);
        fogHistory = ctx.createStorageImage3D(w, h, gridZ, FOG_FORMAT,
                "volumetric fog history " + w + "x" + h + "x" + gridZ);
        fogHistoryConfidence = ctx.createStorageImage3D(w, h, gridZ, CONF_FORMAT,
                "volumetric fog confidence " + w + "x" + h + "x" + gridZ);

        // Clear history and history-confidence to zero so the first frame does not read
        // undefined contents. fogVolume/fogCurrent are written this frame by inject/spatial
        // before being read, but clear them too for deterministic debugging.
        ctx.submitSync(cmd -> clearFroxelImages(cmd));

        lastDepthView = 0;
        lastOutputView = 0;
    }

    private void clearFroxelImages(VkCommandBuffer cmd) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkImageSubresourceRange.Buffer range = VkImageSubresourceRange.calloc(1, stack);
            range.get(0).aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                    .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);

            VkClearColorValue black = VkClearColorValue.calloc(stack);
            black.float32(0, 0f).float32(1, 0f).float32(2, 0f).float32(3, 0f);

            RtImage[] toClear = { fogCurrent, fogVolume, fogHistory, fogHistoryConfidence };
            for (RtImage img : toClear) {
                if (img == null) continue;
                VK10.vkCmdClearColorImage(cmd, img.image, VK10.VK_IMAGE_LAYOUT_GENERAL, black, range);
            }

            VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, stack);
            mb.get(0).sType$Default()
                    .srcAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                    .dstAccessMask(VK10.VK_ACCESS_SHADER_READ_BIT | VK10.VK_ACCESS_SHADER_WRITE_BIT);
            VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, mb, null, null);
        }
    }

    private void destroyImages() {
        if (fogCurrent != null) { fogCurrent.destroy(); fogCurrent = null; }
        if (fogVolume != null) { fogVolume.destroy(); fogVolume = null; }
        if (fogHistory != null) { fogHistory.destroy(); fogHistory = null; }
        if (fogHistoryConfidence != null) { fogHistoryConfidence.destroy(); fogHistoryConfidence = null; }
    }

    private void writeDescriptors(MemoryStack stack, long tlas, long depthView, long outputView) {
        boolean needWrite = (tlas != lastTlas) || (depthView != lastDepthView) || (outputView != lastOutputView)
                || lastDepthView == 0;
        if (!needWrite) return;
        VkDevice vk = ctx.vk();

        // 6 storage image descriptors + 1 AS descriptor.
        VkDescriptorImageInfo.Buffer img = VkDescriptorImageInfo.calloc(6, stack);
        img.get(0).imageView(fogVolume.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
        img.get(1).imageView(fogCurrent.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
        img.get(2).imageView(fogHistory.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
        img.get(3).imageView(fogHistoryConfidence.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
        img.get(4).imageView(depthView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
        img.get(5).imageView(outputView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

        VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(7, stack);
        int w = 0;
        // TLAS
        VkWriteDescriptorSetAccelerationStructureKHR asWrite = VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR)
                .pAccelerationStructures(stack.longs(tlas));
        writes.get(w).sType$Default().pNext(asWrite).dstSet(descriptorSet).dstBinding(VOLUME_TLAS)
                .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
        w++;
        // Storage images (FOG_VOLUME, FOG_CURRENT, FOG_HISTORY, FOG_HISTORY_CONF, DEPTH_IMAGE, OUTPUT_IMAGE)
        int[] storageSlots = { VOLUME_FOG_VOLUME, VOLUME_FOG_CURRENT, VOLUME_FOG_HISTORY,
                VOLUME_FOG_HISTORY_CONF, VOLUME_DEPTH_IMAGE, VOLUME_OUTPUT_IMAGE };
        for (int i = 0; i < storageSlots.length; i++) {
            writes.get(w).sType$Default().dstSet(descriptorSet).dstBinding(storageSlots[i])
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(VkDescriptorImageInfo.create(img.address() + i * VkDescriptorImageInfo.SIZEOF));
            w++;
        }

        VK10.vkUpdateDescriptorSets(vk, VkWriteDescriptorSet.create(writes.address(), w), null);
        lastTlas = tlas;
        lastDepthView = depthView;
        lastOutputView = outputView;
    }

    /**
     * Dispatch all four fog passes and record the volume->history copy.
     *
     * <p>When done, the integrate pass will have overwritten {@code outputView} with
     * {@code L + T * scene}.
     *
     * @param cmd             command buffer in recording state
     * @param renderWidth     render-resolution width (matches gDepth, output)
     * @param renderHeight    render-resolution height
     * @param divisor         froxel XY downsample divisor (e.g. 8)
     * @param tlas            top-level AS for shadow rays
     * @param depthView       image view of the depth guide buffer (R32F, GENERAL)
     * @param outputView      image view of the render-res RT output (RGBA16F, GENERAL),
     *                        already populated with the pre-fog RT color; overwritten by integrate
     * @param invViewProj     inverse view-projection (jitter-free), camera-relative rebased world
     * @param curViewProj     current view-projection (jitter-free)
     * @param prevViewProj    previous-frame jitter-free view-projection (for temporal reprojection)
     * @param camX,camY,camZ  camera position in rebased (camera-relative) coordinates this frame
     * @param camDeltaX,camDeltaY,camDeltaZ camera-space translation from PREVIOUS camera position to
     *                        current camera position, in blocks (world units). Rebase-agnostic:
     *                        previous camera offset = (camX - camDeltaX, camY - camDeltaY, camZ - camDeltaZ).
     * @param sunDirX,sunDirY,sunDirZ normalized sun direction (toward sun; direction is translation-invariant)
     * @param sunAngularRadius sun disk half-angle in radians
     * @param sunRadiance     directional in-scattered radiance scale for the sun (already in the
     *                        pre-exposed scene-linear domain; includes HG-phase-independent
     *                        factors like E_lux * preExposure / pi). Phase is applied per-froxel.
     * @param historyValid    true if a valid history volume exists from last frame (same grid
     *                        dimensions/distribution and camera continuity) so temporal accumulation
     *                        is safe; false to reset to current-frame-only this dispatch.
     */
    public void dispatch(VkCommandBuffer cmd, int renderWidth, int renderHeight, int divisor,
                         long tlas, long depthView, long outputView,
                         Matrix4fc invViewProj, Matrix4fc curViewProj, Matrix4fc prevViewProj,
                         float camX, float camY, float camZ,
                         float camDeltaX, float camDeltaY, float camDeltaZ,
                         float sunDirX, float sunDirY, float sunDirZ,
                         float sunAngularRadius, float sunRadiance,
                         boolean historyValid) {
        if (!CausticaConfig.Rt.Fog.ENABLED.value()) return;

        ensureMatricesBuffer();
        ensureResources(renderWidth, renderHeight, divisor);

        // Determine history validity: caller-supplied continuity AND grid stability.
        boolean gridValid = lastNear == CausticaConfig.Rt.Fog.NEAR.value()
                && lastFar == CausticaConfig.Rt.Fog.FAR.value()
                && lastGridZ == gridZ
                && lastDivisor == divisor;
        boolean useHistory = historyValid && gridValid;

        // Previous-frame camera offset in THIS frame's rebased coords = current offset - camera delta.
        // camDelta is the world-space translation from prev camera to current (rebase-agnostic), so
        // subtracting it from the current (rebased) offset yields the previous camera position in
        // the current rebase frame — which is exactly what prevCamOffset must be for WorldToFroxelPrevious
        // to reproject correctly across rebase-origin jumps.
        float pcPrevX = useHistory ? camX - camDeltaX : camX;
        float pcPrevY = useHistory ? camY - camDeltaY : camY;
        float pcPrevZ = useHistory ? camZ - camDeltaZ : camZ;

        // Rotate matrices ring slot and write the three matrices (column-major; Slang's float4x4 matches).
        matricesSlot = (matricesSlot + 1) % PUSH_RING;
        RtBuffer matBuf = matricesRing[matricesSlot];
        {
            ByteBuffer mapped = MemoryUtil.memByteBuffer(matBuf.mapped, (int) MATRICES_BUF_SIZE);
            FloatBuffer fb = mapped.asFloatBuffer();
            fb.rewind();
            invViewProj.get(fb); fb.position(16);
            curViewProj.get(fb); fb.position(32);
            prevViewProj.get(fb); fb.position(48);
            matBuf.flush(0L, MATRICES_BUF_SIZE);
        }

        // NOTE: parameter order must match the field declaration order in
        // volumetric/common.slang VolumetricPush exactly. The generated
        // VolumetricPushData record constructor mirrors Slang-reflected field order.
        VolumetricPushData push = new VolumetricPushData(
                matBuf.deviceAddress,
                new VolumetricPushData.Float3(camX, camY, camZ),
                CausticaConfig.Rt.Fog.NEAR.value(),
                new VolumetricPushData.Float3(pcPrevX, pcPrevY, pcPrevZ),
                CausticaConfig.Rt.Fog.FAR.value(),
                new VolumetricPushData.Float3(sunDirX, sunDirY, sunDirZ),
                sunAngularRadius,
                new VolumetricPushData.Float3(sunRadiance, sunRadiance, sunRadiance),
                CausticaConfig.Rt.Fog.HEIGHT_DENSITY.value(),
                CausticaConfig.Rt.Fog.HEIGHT_FALLOFF.value(),
                CausticaConfig.Rt.Fog.HEIGHT_BASE.value(),
                CausticaConfig.Rt.Fog.GLOBAL_DENSITY.value(),
                CausticaConfig.Rt.Fog.ANISOTROPY.value(),
                CausticaConfig.Rt.Fog.HISTORY_WEIGHT.value(),
                useHistory ? 1f : 0f,
                gridW,
                gridH,
                gridZ,
                (int) RtComposite.frameCounter(),
                0, // debugView (0 = composite; extended later)
                CausticaConfig.Rt.Fog.STOCHASTIC_LIGHT.value() ? 1 : 0
        );

        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "volumetric fog")) {
            writeDescriptors(stack, tlas, depthView, outputView);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipelineLayout, 0, stack.longs(descriptorSet), null);

            ByteBuffer pcBuf = stack.malloc(VolumetricPushData.BYTE_SIZE);
            push.write(pcBuf);
            VK10.vkCmdPushConstants(cmd, pipelineLayout,
                    VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, pcBuf);

            try (RtDebugLabels.Scope s = RtDebugLabels.scope(ctx, cmd, "volumetric inject")) {
                VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, injectPipeline);
                VK10.vkCmdDispatch(cmd, (gridW + 3) / 4, (gridH + 3) / 4, (gridZ + 3) / 4);
            }
            computeBarrier(cmd, stack);

            try (RtDebugLabels.Scope s = RtDebugLabels.scope(ctx, cmd, "volumetric spatial")) {
                VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, spatialPipeline);
                VK10.vkCmdDispatch(cmd, (gridW + 3) / 4, (gridH + 3) / 4, (gridZ + 3) / 4);
            }
            computeBarrier(cmd, stack);

            try (RtDebugLabels.Scope s = RtDebugLabels.scope(ctx, cmd, "volumetric temporal")) {
                VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, temporalPipeline);
                VK10.vkCmdDispatch(cmd, (gridW + 3) / 4, (gridH + 3) / 4, (gridZ + 3) / 4);
            }
            // Temporal writes to fogCurrent (out-of-place vs. its fogVolume reads) to avoid
            // in-place feedback hazards across neighboring workgroups. Copy fogCurrent -> fogVolume
            // so integrate reads the post-temporal froxel grid.
            computeBarrier(cmd, stack);
            try (RtDebugLabels.Scope s = RtDebugLabels.scope(ctx, cmd, "volumetric temporal->volume copy")) {
                VkImageCopy.Buffer region = VkImageCopy.calloc(1, stack);
                region.get(0).srcSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                        .mipLevel(0).baseArrayLayer(0).layerCount(1);
                region.get(0).dstSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                        .mipLevel(0).baseArrayLayer(0).layerCount(1);
                region.get(0).extent().set(gridW, gridH, gridZ);
                VK10.vkCmdCopyImage(cmd, fogCurrent.image, VK10.VK_IMAGE_LAYOUT_GENERAL,
                        fogVolume.image, VK10.VK_IMAGE_LAYOUT_GENERAL, region);
            }
            transferBarrier(cmd, stack);

            try (RtDebugLabels.Scope s = RtDebugLabels.scope(ctx, cmd, "volumetric integrate")) {
                VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, integratePipeline);
                VK10.vkCmdDispatch(cmd, (renderWidth + 7) / 8, (renderHeight + 7) / 8, 1);
            }
            computeBarrier(cmd, stack);

            // Copy fogVolume -> fogHistory for next frame (3D image copy; same format/size).
            try (RtDebugLabels.Scope s = RtDebugLabels.scope(ctx, cmd, "volumetric history copy")) {
                VkImageCopy.Buffer region = VkImageCopy.calloc(1, stack);
                region.get(0).srcSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                        .mipLevel(0).baseArrayLayer(0).layerCount(1);
                region.get(0).dstSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                        .mipLevel(0).baseArrayLayer(0).layerCount(1);
                region.get(0).extent().set(gridW, gridH, gridZ);
                VK10.vkCmdCopyImage(cmd, fogVolume.image, VK10.VK_IMAGE_LAYOUT_GENERAL,
                        fogHistory.image, VK10.VK_IMAGE_LAYOUT_GENERAL, region);
            }
            transferBarrier(cmd, stack);
        }

        // Persist grid state for next frame's validity check.
        lastNear = CausticaConfig.Rt.Fog.NEAR.value();
        lastFar = CausticaConfig.Rt.Fog.FAR.value();
        lastGridZ = gridZ;
        lastDivisor = divisor;
    }

    private static void computeBarrier(VkCommandBuffer cmd, MemoryStack stack) {
        VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, stack);
        mb.get(0).sType$Default()
                .srcAccessMask(VK10.VK_ACCESS_SHADER_WRITE_BIT)
                .dstAccessMask(VK10.VK_ACCESS_SHADER_READ_BIT);
        VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, mb, null, null);
    }

    private static void transferBarrier(VkCommandBuffer cmd, MemoryStack stack) {
        VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, stack);
        mb.get(0).sType$Default()
                .srcAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                .dstAccessMask(VK10.VK_ACCESS_SHADER_READ_BIT);
        VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, mb, null, null);
    }

    public void destroy() {
        if (destroyed) return;
        destroyImages();
        for (int i = 0; i < PUSH_RING; i++) {
            if (matricesRing[i] != null) { matricesRing[i].destroy(); matricesRing[i] = null; }
        }
        VkDevice vk = ctx.vk();
        if (injectPipeline != 0L) { VK10.vkDestroyPipeline(vk, injectPipeline, null); injectPipeline = 0L; }
        if (spatialPipeline != 0L) { VK10.vkDestroyPipeline(vk, spatialPipeline, null); spatialPipeline = 0L; }
        if (temporalPipeline != 0L) { VK10.vkDestroyPipeline(vk, temporalPipeline, null); temporalPipeline = 0L; }
        if (integratePipeline != 0L) { VK10.vkDestroyPipeline(vk, integratePipeline, null); integratePipeline = 0L; }
        if (pipelineLayout != 0L) { VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null); pipelineLayout = 0L; }
        if (descriptorPool != 0L) { VK10.vkDestroyDescriptorPool(vk, descriptorPool, null); descriptorPool = 0L; }
        if (descriptorSetLayout != 0L) { VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null); descriptorSetLayout = 0L; }
        destroyed = true;
    }
}
