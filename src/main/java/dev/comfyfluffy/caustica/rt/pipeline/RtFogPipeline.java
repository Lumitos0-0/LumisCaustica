package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.gen.FogFilterPushData;
import dev.comfyfluffy.caustica.rt.gen.FogFilterPushData.Int2;
import dev.comfyfluffy.caustica.rt.gen.FogMarchPushData;
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
 * The two compute passes of the volumetric fog: the froxel temporal/spatial filter and the display-res
 * march that integrates the filtered cache and composites it over the denoised scene.
 *
 * <p>The per-froxel light sampling itself is not here — it is the world pipeline's third raygen record
 * ({@code froxel_light.rgen}), because it needs the TLAS, the light grid and the sky LUTs, all of which
 * the world descriptor set already binds. This class only consumes the raw source term that pass writes.
 *
 * <p>The resolved cache is a ping-pong pair of 3D images. Each is bound into both descriptor sets — as a
 * sampled history input for the filter and as a storage write target for whichever half the parity
 * selects — so the sets are written once per (re)allocation instead of once per frame. Updating a set
 * per frame would race the previous frame's still-executing command buffer.
 */
public final class RtFogPipeline {
    private static final String FILTER_SHADER = "/caustica/shaders/pipelines/fog/filter.comp.spv";
    private static final String MARCH_SHADER = "/caustica/shaders/pipelines/fog/march.comp.spv";

    private final RtContext ctx;
    private final long filterSetLayout;
    private final long marchSetLayout;
    private final long descriptorPool;
    private final long filterSet;
    private final long marchSet;
    private final long filterPipelineLayout;
    private final long marchPipelineLayout;
    private final long filterPipeline;
    private final long marchPipeline;
    private final long sampler;
    // Last views written into the two sets, so the per-frame bind call in RtComposite is a no-op unless
    // an allocation actually changed. Re-writing a set every frame would race the previous frame's
    // still-executing command buffer, which reads the same descriptors.
    private long boundRawView;
    private long boundCacheAView;
    private long boundCacheBView;
    private long boundSceneView;
    private long boundDepthView;
    private long boundFoggedView;
    private boolean destroyed;

    private RtFogPipeline(RtContext ctx, long filterSetLayout, long marchSetLayout, long descriptorPool,
                          long filterSet, long marchSet, long filterPipelineLayout, long marchPipelineLayout,
                          long filterPipeline, long marchPipeline, long sampler) {
        this.ctx = ctx;
        this.filterSetLayout = filterSetLayout;
        this.marchSetLayout = marchSetLayout;
        this.descriptorPool = descriptorPool;
        this.filterSet = filterSet;
        this.marchSet = marchSet;
        this.filterPipelineLayout = filterPipelineLayout;
        this.marchPipelineLayout = marchPipelineLayout;
        this.filterPipeline = filterPipeline;
        this.marchPipeline = marchPipeline;
        this.sampler = sampler;
    }

    public static RtFogPipeline create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer handle = stack.mallocLong(1);

            VkDescriptorSetLayoutBinding.Buffer filterBindings =
                    VkDescriptorSetLayoutBinding.calloc(FOG_FILTER_BINDING_COUNT, stack);
            storageBinding(filterBindings.get(FOG_FILTER_RAW), FOG_FILTER_RAW);
            sampledBinding(filterBindings.get(FOG_FILTER_CACHE_A), FOG_FILTER_CACHE_A);
            sampledBinding(filterBindings.get(FOG_FILTER_CACHE_B), FOG_FILTER_CACHE_B);
            storageBinding(filterBindings.get(FOG_FILTER_WRITE_A), FOG_FILTER_WRITE_A);
            storageBinding(filterBindings.get(FOG_FILTER_WRITE_B), FOG_FILTER_WRITE_B);
            long filterSetLayout = createSetLayout(ctx, vk, stack, filterBindings, handle,
                    "fog filter descriptor set layout");

            VkDescriptorSetLayoutBinding.Buffer marchBindings =
                    VkDescriptorSetLayoutBinding.calloc(FOG_MARCH_BINDING_COUNT, stack);
            sampledBinding(marchBindings.get(FOG_MARCH_FROXEL_A), FOG_MARCH_FROXEL_A);
            sampledBinding(marchBindings.get(FOG_MARCH_FROXEL_B), FOG_MARCH_FROXEL_B);
            storageBinding(marchBindings.get(FOG_MARCH_SCENE), FOG_MARCH_SCENE);
            storageBinding(marchBindings.get(FOG_MARCH_DEPTH), FOG_MARCH_DEPTH);
            storageBinding(marchBindings.get(FOG_MARCH_OUTPUT), FOG_MARCH_OUTPUT);
            long marchSetLayout = createSetLayout(ctx, vk, stack, marchBindings, handle,
                    "fog march descriptor set layout");

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(3 + 3);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(2 + 2);
            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(2).pPoolSizes(poolSizes);
            check(VK10.vkCreateDescriptorPool(vk, poolInfo, null, handle),
                    "vkCreateDescriptorPool(rt fog)");
            long descriptorPool = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, descriptorPool, "fog descriptor pool");

            VkDescriptorSetAllocateInfo allocateInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(descriptorPool)
                    .pSetLayouts(stack.longs(filterSetLayout, marchSetLayout));
            LongBuffer setHandles = stack.mallocLong(2);
            check(VK10.vkAllocateDescriptorSets(vk, allocateInfo, setHandles),
                    "vkAllocateDescriptorSets(rt fog)");
            long filterSet = setHandles.get(0);
            long marchSet = setHandles.get(1);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, filterSet, "fog filter descriptor set");
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, marchSet, "fog march descriptor set");

            // Linear, CLAMP_TO_EDGE on all three axes: trilinear interpolation between slices is what
            // turns a 32-slice cache into a smooth volume, and clamping keeps edge froxels from bleeding
            // in whatever sits outside the frustum-aligned grid.
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR)
                    .minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f)
                    .maxLod(0.0f);
            check(VK10.vkCreateSampler(vk, samplerInfo, null, handle), "vkCreateSampler(rt fog)");
            long sampler = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "fog trilinear sampler");

            long filterPipelineLayout = createPipelineLayout(ctx, vk, stack, filterSetLayout,
                    FogFilterPushData.BYTE_SIZE, handle, "fog filter pipeline layout");
            long marchPipelineLayout = createPipelineLayout(ctx, vk, stack, marchSetLayout,
                    FogMarchPushData.BYTE_SIZE, handle, "fog march pipeline layout");
            long filterPipeline = createComputePipeline(ctx, vk, stack, FILTER_SHADER,
                    filterPipelineLayout, handle, "fog filter compute pipeline");
            long marchPipeline = createComputePipeline(ctx, vk, stack, MARCH_SHADER,
                    marchPipelineLayout, handle, "fog march compute pipeline");

            return new RtFogPipeline(ctx, filterSetLayout, marchSetLayout, descriptorPool, filterSet,
                    marchSet, filterPipelineLayout, marchPipelineLayout, filterPipeline, marchPipeline,
                    sampler);
        }
    }

    private static void storageBinding(VkDescriptorSetLayoutBinding binding, int index) {
        binding.binding(index).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
    }

    private static void sampledBinding(VkDescriptorSetLayoutBinding binding, int index) {
        binding.binding(index).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
    }

    private static long createSetLayout(RtContext ctx, VkDevice vk, MemoryStack stack,
                                        VkDescriptorSetLayoutBinding.Buffer bindings, LongBuffer handle,
                                        String label) {
        VkDescriptorSetLayoutCreateInfo info =
                VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(bindings);
        check(VK10.vkCreateDescriptorSetLayout(vk, info, null, handle), "vkCreateDescriptorSetLayout(rt fog)");
        long layout = handle.get(0);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, layout, label);
        return layout;
    }

    private static long createPipelineLayout(RtContext ctx, VkDevice vk, MemoryStack stack, long setLayout,
                                             int pushSize, LongBuffer handle, String label) {
        VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
        pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(pushSize);
        VkPipelineLayoutCreateInfo info = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                .pSetLayouts(stack.longs(setLayout)).pPushConstantRanges(pushRange);
        check(VK10.vkCreatePipelineLayout(vk, info, null, handle), "vkCreatePipelineLayout(rt fog)");
        long layout = handle.get(0);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, label);
        return layout;
    }

    private static long createComputePipeline(RtContext ctx, VkDevice vk, MemoryStack stack, String shader,
                                              long pipelineLayout, LongBuffer handle, String label) {
        long module = loadModule(vk, stack, shader, label);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, module, label + " module");
        VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                .sType$Default().stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                .module(module).pName(stack.UTF8("main"));
        VkComputePipelineCreateInfo.Buffer info = VkComputePipelineCreateInfo.calloc(1, stack);
        info.get(0).sType$Default().stage(stage).layout(pipelineLayout);
        LongBuffer pipelineHandle = stack.mallocLong(1);
        check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, info, null, pipelineHandle),
                "vkCreateComputePipelines(rt fog)");
        VK10.vkDestroyShaderModule(vk, module, null);
        long pipeline = pipelineHandle.get(0);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, pipeline, label);
        return pipeline;
    }

    /**
     * Bind this allocation's images into both sets. {@code cacheA} and {@code cacheB} are the two halves
     * of the resolved-cache ping-pong pair: each is bound as the filter's sampled history input and as
     * the other parity's storage write target, and both are bound as the march's sampled inputs.
     */
    public void setImages(long rawView, long cacheAView, long cacheBView, long sceneView, long depthView,
                          long foggedView) {
        if (boundRawView == rawView && boundCacheAView == cacheAView && boundCacheBView == cacheBView
                && boundSceneView == sceneView && boundDepthView == depthView
                && boundFoggedView == foggedView) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer images = VkDescriptorImageInfo.calloc(10, stack);
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(10, stack);
            int index = 0;
            index = write(index, images, writes, filterSet, FOG_FILTER_RAW,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, rawView, 0L);
            index = write(index, images, writes, filterSet, FOG_FILTER_CACHE_A,
                    VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, cacheAView, sampler);
            index = write(index, images, writes, filterSet, FOG_FILTER_CACHE_B,
                    VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, cacheBView, sampler);
            index = write(index, images, writes, filterSet, FOG_FILTER_WRITE_A,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, cacheBView, 0L);
            index = write(index, images, writes, filterSet, FOG_FILTER_WRITE_B,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, cacheAView, 0L);
            index = write(index, images, writes, marchSet, FOG_MARCH_FROXEL_A,
                    VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, cacheAView, sampler);
            index = write(index, images, writes, marchSet, FOG_MARCH_FROXEL_B,
                    VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, cacheBView, sampler);
            index = write(index, images, writes, marchSet, FOG_MARCH_SCENE,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, sceneView, 0L);
            index = write(index, images, writes, marchSet, FOG_MARCH_DEPTH,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, depthView, 0L);
            write(index, images, writes, marchSet, FOG_MARCH_OUTPUT,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, foggedView, 0L);
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundRawView = rawView;
        boundCacheAView = cacheAView;
        boundCacheBView = cacheBView;
        boundSceneView = sceneView;
        boundDepthView = depthView;
        boundFoggedView = foggedView;
    }

    private static int write(int index, VkDescriptorImageInfo.Buffer images,
                             VkWriteDescriptorSet.Buffer writes, long set, int binding, int type,
                             long imageView, long sampler) {
        images.get(index).imageView(imageView).sampler(sampler)
                .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
        writes.get(index).sType$Default().dstSet(set).dstBinding(binding)
                .descriptorCount(1).descriptorType(type)
                .pImageInfo(VkDescriptorImageInfo.create(images.address(index), 1));
        return index + 1;
    }

    /**
     * Record the froxel filter over the whole grid. {@code parity} 0 reads cacheA and writes cacheB;
     * 1 does the reverse. {@code worldPushAddress} is this frame's WorldPush slot: the pass needs the
     * previous-frame view-projection and camera delta to reproject history.
     */
    public void recordFilter(VkCommandBuffer cmd, int gridWidth, int gridHeight, int slices, int parity,
                             float maxHistory, float fireflyClamp, long worldPushAddress) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "fog froxel filter")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, filterPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    filterPipelineLayout, 0, stack.longs(filterSet), null);
            ByteBuffer push = stack.malloc(FogFilterPushData.BYTE_SIZE);
            new FogFilterPushData(worldPushAddress, parity, new Int2(gridWidth, gridHeight),
                    maxHistory, fireflyClamp).write(push);
            VK10.vkCmdPushConstants(cmd, filterPipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (gridWidth + 3) / 4, (gridHeight + 3) / 4, slices);
            VulkanCommandEncoder.memoryBarrier(cmd, stack); // resolved cache visible to the march
        }
    }

    /**
     * Record the display-res march and composite. {@code enabled} 0 makes the pass copy the scene
     * through untouched, so the display chain can read one image whether fog is on or off.
     */
    public void recordMarch(VkCommandBuffer cmd, boolean enabled, int parity, long frameIndex,
                            int displayWidth, int displayHeight, int renderWidth, int renderHeight,
                            long worldPushAddress) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "fog march")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, marchPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    marchPipelineLayout, 0, stack.longs(marchSet), null);
            ByteBuffer push = stack.malloc(FogMarchPushData.BYTE_SIZE);
            new FogMarchPushData(worldPushAddress, enabled ? 1 : 0, parity, (int) frameIndex,
                    new FogMarchPushData.Int2(displayWidth, displayHeight),
                    new FogMarchPushData.Int2(renderWidth, renderHeight)).write(push);
            VK10.vkCmdPushConstants(cmd, marchPipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (displayWidth + 7) / 8, (displayHeight + 7) / 8, 1);
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, filterPipeline, null);
        VK10.vkDestroyPipeline(vk, marchPipeline, null);
        VK10.vkDestroySampler(vk, sampler, null);
        VK10.vkDestroyPipelineLayout(vk, filterPipelineLayout, null);
        VK10.vkDestroyPipelineLayout(vk, marchPipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, filterSetLayout, null);
        VK10.vkDestroyDescriptorSetLayout(vk, marchSetLayout, null);
        destroyed = true;
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String shader, String label) {
        byte[] bytes;
        try (InputStream input = RtFogPipeline.class.getResourceAsStream(shader)) {
            if (input == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + shader);
            }
            bytes = input.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + shader, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo moduleInfo = VkShaderModuleCreateInfo.calloc(stack)
                    .sType$Default().pCode(code);
            LongBuffer module = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, moduleInfo, null, module), "vkCreateShaderModule(" + label + ")");
            return module.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}
