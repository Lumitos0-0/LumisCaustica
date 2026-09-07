package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.PushAddrData;
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

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/**
 * Froxel volumetric fog: the three frustum-aligned volumes and the two compute passes that turn the
 * injection raygen's raw per-froxel in-scatter into a camera-to-slice prefix the composite can sample.
 * See {@code shaders/pipelines/world/fog.slang} for the grid parameterisation and the medium.
 *
 * <p>The injection itself is NOT here — it is a raygen in the world ray-tracing pipeline, so that it
 * reuses the same shader binding table and the same {@code visibility()} helper as surface next-event
 * estimation. That is what makes a shaft through stained glass carry the pane's colour without a second
 * implementation of the any-hit tint chain. This class owns the images that raygen writes into, and
 * publishes the scatter volume's view so {@link RtPipeline#setFogScatter} can bind it.
 *
 * <p>Volumes:
 * <ul>
 *   <li>{@code scatter} — this frame's raw in-scatter per unit length, absolute ACEScg. Written by the
 *       injection raygen, consumed by the integration pass.</li>
 *   <li>{@code history[2]} — ping-pong temporal accumulation. RGB is the blended in-scatter, A is a
 *       running second moment of its luminance, which is what gives the blend a real per-froxel
 *       variance to adapt to instead of a fixed alpha.</li>
 *   <li>{@code integrated} — RGB in-scatter accumulated camera-to-slice, A accumulated transmittance.</li>
 * </ul>
 *
 * <p>Extinction is deliberately not stored: it is a pure function of position and the density field, so
 * recomputing it in both passes is cheaper than a fourth volume, and it frees the alpha channel for the
 * second moment.
 *
 * <p>The volumes are single-buffered apart from the history pair, on the same argument as every other
 * per-frame GPU image here: the passes that write and read them are separated by barriers inside one
 * submission, and the composite's end-of-frame barrier orders the next frame's overwrite against this
 * frame's reads.
 */
public final class RtVolumetricFog {
    private static final String SHADER_DIR = "/caustica/shaders/pipelines/volumetric/";
    private static final int GROUP_SIZE = 8;
    /** Quality tiers: {screen divisor, slice count}. Index is {@code Rt.Fog.QUALITY}. */
    private static final int[][] TIERS = {
            {12, 64},   // low
            {8, 96},    // medium
            {8, 128},   // high (default)
            {6, 160},   // ultra
    };
    /**
     * Slice counts below this band at Minecraft's scale; the plan's measurement is that 64 is visibly
     * ringed along the sun direction and 128 is clean. Kept as a floor so a tier edit cannot silently
     * reintroduce banding.
     */
    private static final int MIN_SLICES = 32;

    private final RtContext ctx;
    private final long sampler;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    // Two sets: [0] reads history[0] and writes history[1], [1] the reverse. Ping-ponging by descriptor
    // set rather than by a dynamically indexed image array keeps the shader free of storage-image
    // dynamic indexing, which the device bring-up does not request.
    private final long[] descriptorSets;
    private final long pipelineLayout;
    private final long integratePipeline;
    private final long compositePipeline;

    private RtImage scatter;
    private final RtImage[] history = new RtImage[2];
    private RtImage integrated;
    private int gridWidth;
    private int gridHeight;
    private int gridDepth;
    private int accumIndex;
    private boolean historyValid;
    private boolean destroyed;

    private RtVolumetricFog(RtContext ctx, long sampler, long descriptorSetLayout, long descriptorPool,
                            long[] descriptorSets, long pipelineLayout, long integratePipeline,
                            long compositePipeline) {
        this.ctx = ctx;
        this.sampler = sampler;
        this.descriptorSetLayout = descriptorSetLayout;
        this.descriptorPool = descriptorPool;
        this.descriptorSets = descriptorSets;
        this.pipelineLayout = pipelineLayout;
        this.integratePipeline = integratePipeline;
        this.compositePipeline = compositePipeline;
    }

    public static RtVolumetricFog create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            // CLAMP on every axis. The history fetch reprojects to arbitrary coordinates and the
            // composite clamps its W to the grid's far slice; a REPEAT would wrap a far sample back to
            // the near plane and inject foreground scattering into the sky.
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR).minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f).maxLod(0.0f);
            LongBuffer handle = stack.mallocLong(1);
            check(VK10.vkCreateSampler(vk, samplerInfo, null, handle), "vkCreateSampler(volumetric fog)");
            long sampler = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "volumetric fog sampler");

            VkDescriptorSetLayoutBinding.Buffer bindings =
                    VkDescriptorSetLayoutBinding.calloc(VOLUMETRIC_BINDING_COUNT, stack);
            int[] storageBindings = {VOLUMETRIC_SCATTER, VOLUMETRIC_HISTORY, VOLUMETRIC_INTEGRATED,
                    VOLUMETRIC_SCENE, VOLUMETRIC_SCENE_DEPTH};
            for (int binding : storageBindings) {
                bindings.get(binding).binding(binding)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            int[] sampledBindings = {VOLUMETRIC_HISTORY_PREV, VOLUMETRIC_INTEGRATED_SAMPLER};
            for (int binding : sampledBindings) {
                bindings.get(binding).binding(binding)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            VkDescriptorSetLayoutCreateInfo layoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(bindings);
            check(VK10.vkCreateDescriptorSetLayout(vk, layoutInfo, null, handle),
                    "vkCreateDescriptorSetLayout(volumetric fog)");
            long descriptorSetLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                    descriptorSetLayout, "volumetric fog descriptor set layout");

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(storageBindings.length * 2);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(sampledBindings.length * 2);
            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(2).pPoolSizes(poolSizes);
            check(VK10.vkCreateDescriptorPool(vk, poolInfo, null, handle),
                    "vkCreateDescriptorPool(volumetric fog)");
            long descriptorPool = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                    descriptorPool, "volumetric fog descriptor pool");

            LongBuffer setLayouts = stack.mallocLong(2);
            setLayouts.put(0, descriptorSetLayout).put(1, descriptorSetLayout);
            VkDescriptorSetAllocateInfo allocateInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(descriptorPool).pSetLayouts(setLayouts);
            LongBuffer setHandles = stack.mallocLong(2);
            check(VK10.vkAllocateDescriptorSets(vk, allocateInfo, setHandles),
                    "vkAllocateDescriptorSets(volumetric fog)");
            long[] descriptorSets = {setHandles.get(0), setHandles.get(1)};
            for (int i = 0; i < descriptorSets.length; i++) {
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, descriptorSets[i],
                        "volumetric fog descriptor set " + i);
            }

            // Same inline address block the sky LUT bakes use: both passes dereference the frame's
            // WorldPush through pcAddr.worldPushAddr, so the fog and the trace read one source of truth.
            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
            pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .offset(0).size(PushAddrData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo pipelineLayoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default().pSetLayouts(stack.longs(descriptorSetLayout))
                    .pPushConstantRanges(pushRange);
            check(VK10.vkCreatePipelineLayout(vk, pipelineLayoutInfo, null, handle),
                    "vkCreatePipelineLayout(volumetric fog)");
            long pipelineLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                    pipelineLayout, "volumetric fog pipeline layout");

            long integratePipeline = createComputePipeline(ctx, stack, pipelineLayout,
                    "integrate.comp.spv", "volumetric fog integrate pipeline");
            long compositePipeline = createComputePipeline(ctx, stack, pipelineLayout,
                    "composite.comp.spv", "volumetric fog composite pipeline");

            return new RtVolumetricFog(ctx, sampler, descriptorSetLayout, descriptorPool, descriptorSets,
                    pipelineLayout, integratePipeline, compositePipeline);
        }
    }

    public static boolean enabled() {
        return CausticaConfig.Rt.Fog.ENABLED.value();
    }

    /**
     * Froxel grid dimensions for a render resolution at the configured tier, or a 1x1x1 placeholder when
     * fog is disabled.
     *
     * <p>The placeholder is deliberate rather than skipping allocation entirely: the world descriptor set
     * declares the scatter volume unconditionally, and a descriptor left unwritten is undefined at trace
     * time even for a raygen that never runs (VUID-vkCmdTraceRaysKHR-None-08114). Keeping a real but
     * trivial image satisfies the binding for a few bytes, and re-enabling fog is then an ordinary grid
     * change that reallocates through the same path as a resolution change.
     */
    public static int[] gridSizeFor(int renderWidth, int renderHeight) {
        if (!enabled()) {
            return new int[]{1, 1, 1};
        }
        int[] tier = TIERS[Math.clamp(CausticaConfig.Rt.Fog.QUALITY.value(), 0, TIERS.length - 1)];
        int divisor = Math.max(1, tier[0]);
        return new int[]{
                Math.max(1, (renderWidth + divisor - 1) / divisor),
                Math.max(1, (renderHeight + divisor - 1) / divisor),
                Math.max(MIN_SLICES, tier[1])};
    }

    public int gridWidth() {
        return gridWidth;
    }

    public int gridHeight() {
        return gridHeight;
    }

    public int gridDepth() {
        return gridDepth;
    }

    /** View of the volume the injection raygen writes; bound into the world descriptor set. */
    public long scatterView() {
        return scatter != null ? scatter.view : 0L;
    }

    public boolean ready() {
        return scatter != null && integrated != null && history[0] != null && history[1] != null;
    }

    /**
     * True when this frame's temporal history describes the same grid this frame will write. The
     * integration pass falls back to the raw sample wherever this is false, so a resize, a tier change or
     * a teleport cannot smear a stale volume into the new one.
     */
    public boolean historyValid() {
        return historyValid;
    }

    /** Drop the accumulated history for the next recorded frame (teleport, rebase shift, config change). */
    public void invalidateHistory() {
        historyValid = false;
    }

    /**
     * Allocate (or reallocate) the volumes for a render resolution and the scene images the composite
     * writes. Returns true when the grid changed, which the caller must treat as a reason to rebind the
     * world pipeline's scatter binding.
     */
    public boolean ensureResources(int renderWidth, int renderHeight, RtImage sceneImage, RtImage sceneDepth) {
        int[] size = gridSizeFor(renderWidth, renderHeight);
        boolean sameGrid = scatter != null && gridWidth == size[0] && gridHeight == size[1]
                && gridDepth == size[2];
        if (sameGrid) {
            // The scene images are recreated on resize, and the grid may legitimately be unchanged when
            // only the display resolution moved, so the composite's two bindings are refreshed anyway.
            writeSceneBindings(sceneImage, sceneDepth);
            return false;
        }
        ctx.waitIdle(); // reallocation is rare; no in-flight frame may still reference the old volumes
        destroyVolumes();
        gridWidth = size[0];
        gridHeight = size[1];
        gridDepth = size[2];
        String extent = gridWidth + "x" + gridHeight + "x" + gridDepth;
        scatter = ctx.createStorageImage3D(gridWidth, gridHeight, gridDepth,
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "fog scatter " + extent);
        for (int i = 0; i < history.length; i++) {
            history[i] = ctx.createStorageImage3D(gridWidth, gridHeight, gridDepth,
                    VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "fog history " + i + " " + extent);
        }
        integrated = ctx.createStorageImage3D(gridWidth, gridHeight, gridDepth,
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "fog integrated " + extent);
        accumIndex = 0;
        historyValid = false; // freshly allocated volumes contain no usable history
        writeDescriptorSets(sceneImage, sceneDepth);
        return true;
    }

    private void writeDescriptorSets(RtImage sceneImage, RtImage sceneDepth) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            for (int set = 0; set < descriptorSets.length; set++) {
                // Set i writes history[i] and samples history[1 - i].
                long writeView = history[set].view;
                long readView = history[1 - set].view;
                VkDescriptorImageInfo.Buffer infos =
                        VkDescriptorImageInfo.calloc(VOLUMETRIC_BINDING_COUNT, stack);
                VkWriteDescriptorSet.Buffer writes =
                        VkWriteDescriptorSet.calloc(VOLUMETRIC_BINDING_COUNT, stack);
                long[] views = new long[VOLUMETRIC_BINDING_COUNT];
                views[VOLUMETRIC_SCATTER] = scatter.view;
                views[VOLUMETRIC_HISTORY] = writeView;
                views[VOLUMETRIC_HISTORY_PREV] = readView;
                views[VOLUMETRIC_INTEGRATED] = integrated.view;
                views[VOLUMETRIC_INTEGRATED_SAMPLER] = integrated.view;
                views[VOLUMETRIC_SCENE] = sceneImage.view;
                views[VOLUMETRIC_SCENE_DEPTH] = sceneDepth.view;
                for (int binding = 0; binding < VOLUMETRIC_BINDING_COUNT; binding++) {
                    boolean sampled = binding == VOLUMETRIC_HISTORY_PREV
                            || binding == VOLUMETRIC_INTEGRATED_SAMPLER;
                    infos.get(binding).imageView(views[binding])
                            .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                    if (sampled) {
                        infos.get(binding).sampler(sampler);
                    }
                    writes.get(binding).sType$Default().dstSet(descriptorSets[set]).dstBinding(binding)
                            .descriptorCount(1)
                            .descriptorType(sampled ? VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                    : VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                            .pImageInfo(VkDescriptorImageInfo.create(infos.address(binding), 1));
                }
                VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
            }
        }
    }

    private void writeSceneBindings(RtImage sceneImage, RtImage sceneDepth) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer infos = VkDescriptorImageInfo.calloc(2, stack);
            infos.get(0).imageView(sceneImage.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            infos.get(1).imageView(sceneDepth.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(4, stack);
            for (int set = 0; set < descriptorSets.length; set++) {
                writes.get(set * 2).sType$Default().dstSet(descriptorSets[set])
                        .dstBinding(VOLUMETRIC_SCENE).descriptorCount(1)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(VkDescriptorImageInfo.create(infos.address(0), 1));
                writes.get(set * 2 + 1).sType$Default().dstSet(descriptorSets[set])
                        .dstBinding(VOLUMETRIC_SCENE_DEPTH).descriptorCount(1)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(VkDescriptorImageInfo.create(infos.address(1), 1));
            }
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
    }

    /**
     * Temporal blend + depth integration, one thread per froxel column. Must be recorded after the
     * injection raygen with a barrier between; the caller owns the barrier after it.
     */
    public void recordIntegrate(VkCommandBuffer cmd, long worldPushAddress) {
        if (!ready()) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "fog integrate")) {
            bind(cmd, stack, integratePipeline, worldPushAddress);
            VK10.vkCmdDispatch(cmd, groups(gridWidth), groups(gridHeight), 1);
        }
        // The next frame's integration may read what this one just accumulated.
        historyValid = CausticaConfig.Rt.Fog.TEMPORAL.value();
        accumIndex ^= 1;
    }

    /**
     * Apply the integrated volume to the reconstructed scene image, in place, at display resolution.
     * Records after DLSS-RR and before the exposure histogram.
     */
    public void recordComposite(VkCommandBuffer cmd, long worldPushAddress, int displayWidth,
                                int displayHeight) {
        if (!ready()) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "fog composite")) {
            // Either set works here: the two differ only in which history image is bound for read and
            // write, and the composite touches neither — it reads the integrated volume and the scene.
            bind(cmd, stack, compositePipeline, worldPushAddress);
            VK10.vkCmdDispatch(cmd, groups(displayWidth), groups(displayHeight), 1);
        }
    }

    private void bind(VkCommandBuffer cmd, MemoryStack stack, long pipeline, long worldPushAddress) {
        VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                stack.longs(descriptorSets[accumIndex]), null);
        ByteBuffer push = stack.malloc(PushAddrData.BYTE_SIZE);
        new PushAddrData(worldPushAddress).write(push);
        VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
    }

    private static int groups(int extent) {
        return (extent + GROUP_SIZE - 1) / GROUP_SIZE;
    }

    private void destroyVolumes() {
        if (scatter != null) {
            scatter.destroy();
            scatter = null;
        }
        for (int i = 0; i < history.length; i++) {
            if (history[i] != null) {
                history[i].destroy();
                history[i] = null;
            }
        }
        if (integrated != null) {
            integrated.destroy();
            integrated = null;
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, compositePipeline, null);
        VK10.vkDestroyPipeline(vk, integratePipeline, null);
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        VK10.vkDestroySampler(vk, sampler, null);
        destroyVolumes();
        destroyed = true;
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
        try (InputStream input = RtVolumetricFog.class.getResourceAsStream(resource)) {
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
