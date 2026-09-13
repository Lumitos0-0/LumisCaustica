package dev.comfyfluffy.caustica.rt.pipeline;

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

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.gen.PushAddrData;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/**
 * Fog temporal accumulation + apply passes, sharing one 13-binding descriptor set (both shaders
 * import {@code pipelines/fog/bindings.slang}, so the layout is identical by construction).
 *
 * <p>Volume slots: FRESH holds this frame's jittered bake (written by the world pipeline's
 * {@code fog.rgen}); A/B hold the temporally accumulated volumes, ping-ponged by frame parity --
 * even frames blend FRESH + B into A, odd frames blend FRESH + A into B. Ages count accumulated
 * frames per voxel; prevDepth snapshots the hardware depth for the occlusion check. The accumulate
 * pass reprojects each voxel's world position into the previous frame (full 3D), so the volume
 * stays converged through camera motion; the apply pass is purely spatial (bilateral upsample +
 * composite) and can never smear or ghost. Slot selection happens in-shader by frame parity, so
 * descriptors are written once per resize and never rewritten per frame.
 *
 * <p>Reads camera + fog state from the frame's {@code WorldPush} slot through the shared
 * {@code PushAddr} block, like the sky LUT bakes.
 */
public final class RtFogPipeline {
    private static final String SHADER_DIR = "/caustica/shaders/pipelines/fog/";

    private final RtContext ctx;
    private final long linearSampler;
    private final long nearestSampler;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long descriptorSet;
    private final long pipelineLayout;
    private final long pipeline;
    private final long accumulatePipeline;
    private long boundOutputView;
    private long boundDepthView;
    private long boundFreshView;
    private long boundVolAView;
    private long boundVolBView;
    private long boundAgeAView;
    private long boundAgeBView;
    private long boundPrevDepthView;
    private boolean destroyed;

    private RtFogPipeline(RtContext ctx, long linearSampler, long nearestSampler, long dsl, long pool,
            long set, long layout, long pipeline, long accumulatePipeline) {
        this.ctx = ctx;
        this.linearSampler = linearSampler;
        this.nearestSampler = nearestSampler;
        this.descriptorSetLayout = dsl;
        this.descriptorPool = pool;
        this.descriptorSet = set;
        this.pipelineLayout = layout;
        this.pipeline = pipeline;
        this.accumulatePipeline = accumulatePipeline;
    }

    public static RtFogPipeline create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            // Linear: the accumulated volumes reproject to arbitrary sub-voxel positions, so
            // history must be reconstructed by filtering, not snapped to the nearest voxel.
            long linearSampler = createSampler(vk, stack, ctx, VK10.VK_FILTER_LINEAR,
                    "fog volume linear sampler");
            // Nearest: the fresh bake (exact texels for the change-detection box), the age
            // counters (per-voxel integers, never interpolated), and the previous depth
            // (each voxel validates against its own texel, not a blend).
            long nearestSampler = createSampler(vk, stack, ctx, VK10.VK_FILTER_NEAREST,
                    "fog volume nearest sampler");

            VkDescriptorSetLayoutBinding.Buffer binds = VkDescriptorSetLayoutBinding.calloc(FOG_BINDING_COUNT, stack);
            binds.get(FOG_OUTPUT).binding(FOG_OUTPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(FOG_DEPTH).binding(FOG_DEPTH).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(FOG_FRESH).binding(FOG_FRESH).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(FOG_VOL_A).binding(FOG_VOL_A).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(FOG_VOL_B).binding(FOG_VOL_B).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(FOG_VOL_A_RW).binding(FOG_VOL_A_RW).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(FOG_VOL_B_RW).binding(FOG_VOL_B_RW).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(FOG_AGE_A).binding(FOG_AGE_A).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(FOG_AGE_B).binding(FOG_AGE_B).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(FOG_AGE_A_RW).binding(FOG_AGE_A_RW).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(FOG_AGE_B_RW).binding(FOG_AGE_B_RW).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(FOG_PREV_DEPTH).binding(FOG_PREV_DEPTH).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            binds.get(FOG_PREV_DEPTH_RW).binding(FOG_PREV_DEPTH_RW).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);

            VkDescriptorSetLayoutCreateInfo dslci = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(binds);
            LongBuffer p = stack.mallocLong(1);
            check(VK10.vkCreateDescriptorSetLayout(vk, dslci, null, p), "vkCreateDescriptorSetLayout(fog)");
            long dsl = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, dsl, "fog descriptor set layout");

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            // Storage: output + depth + the A/B volume, age and prev-depth write bindings.
            // Sampled: fresh + the A/B volume, age and prev-depth read bindings.
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(7);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(6);
            VkDescriptorPoolCreateInfo dpci = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default().maxSets(1).pPoolSizes(poolSizes);
            check(VK10.vkCreateDescriptorPool(vk, dpci, null, p), "vkCreateDescriptorPool(fog)");
            long pool = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, pool, "fog descriptor pool");

            VkDescriptorSetAllocateInfo dsai = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                    .descriptorPool(pool).pSetLayouts(stack.longs(dsl));
            LongBuffer pSet = stack.mallocLong(1);
            check(VK10.vkAllocateDescriptorSets(vk, dsai, pSet), "vkAllocateDescriptorSets(fog)");
            long set = pSet.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, set, "fog descriptor set");

            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
            pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(PushAddrData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo plci = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                    .pSetLayouts(stack.longs(dsl)).pPushConstantRanges(pushRange);
            check(VK10.vkCreatePipelineLayout(vk, plci, null, p), "vkCreatePipelineLayout(fog)");
            long layout = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, "fog pipeline layout");

            long pipeline = createComputePipeline(vk, stack, ctx, layout, "apply.comp.spv", "fog apply");
            long accumulatePipeline = createComputePipeline(vk, stack, ctx, layout, "accumulate.comp.spv",
                    "fog accumulate");

            return new RtFogPipeline(ctx, linearSampler, nearestSampler, dsl, pool, set, layout,
                    pipeline, accumulatePipeline);
        }
    }

    /**
     * Bind the trace color target (read-write), the guide depth, the fresh bake, the accumulated
     * A/B volumes, the A/B age counters, and the previous-depth snapshot. Read-only/read-write
     * pairs share one view each; the set is rewritten only when a view actually changes (resize /
     * quality switch), never per frame.
     */
    public void setImages(long outputImageView, long depthImageView, long freshView, long volAView,
            long volBView, long ageAView, long ageBView, long prevDepthView) {
        if (boundOutputView == outputImageView && boundDepthView == depthImageView
                && boundFreshView == freshView && boundVolAView == volAView
                && boundVolBView == volBView && boundAgeAView == ageAView
                && boundAgeBView == ageBView && boundPrevDepthView == prevDepthView) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer outputInfo = VkDescriptorImageInfo.calloc(1, stack);
            outputInfo.get(0).imageView(outputImageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer depthInfo = VkDescriptorImageInfo.calloc(1, stack);
            depthInfo.get(0).imageView(depthImageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer freshInfo = VkDescriptorImageInfo.calloc(1, stack);
            freshInfo.get(0).imageView(freshView).sampler(nearestSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer volAInfo = VkDescriptorImageInfo.calloc(1, stack);
            volAInfo.get(0).imageView(volAView).sampler(linearSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer volBInfo = VkDescriptorImageInfo.calloc(1, stack);
            volBInfo.get(0).imageView(volBView).sampler(linearSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer volARwInfo = VkDescriptorImageInfo.calloc(1, stack);
            volARwInfo.get(0).imageView(volAView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer volBRwInfo = VkDescriptorImageInfo.calloc(1, stack);
            volBRwInfo.get(0).imageView(volBView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer ageARoInfo = VkDescriptorImageInfo.calloc(1, stack);
            ageARoInfo.get(0).imageView(ageAView).sampler(nearestSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer ageBRoInfo = VkDescriptorImageInfo.calloc(1, stack);
            ageBRoInfo.get(0).imageView(ageBView).sampler(nearestSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer ageARwInfo = VkDescriptorImageInfo.calloc(1, stack);
            ageARwInfo.get(0).imageView(ageAView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer ageBRwInfo = VkDescriptorImageInfo.calloc(1, stack);
            ageBRwInfo.get(0).imageView(ageBView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer prevDepthRoInfo = VkDescriptorImageInfo.calloc(1, stack);
            prevDepthRoInfo.get(0).imageView(prevDepthView).sampler(nearestSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer prevDepthRwInfo = VkDescriptorImageInfo.calloc(1, stack);
            prevDepthRwInfo.get(0).imageView(prevDepthView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(FOG_BINDING_COUNT, stack);
            writes.get(FOG_OUTPUT).sType$Default().dstSet(descriptorSet).dstBinding(FOG_OUTPUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(outputInfo);
            writes.get(FOG_DEPTH).sType$Default().dstSet(descriptorSet).dstBinding(FOG_DEPTH)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(depthInfo);
            writes.get(FOG_FRESH).sType$Default().dstSet(descriptorSet).dstBinding(FOG_FRESH)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(freshInfo);
            writes.get(FOG_VOL_A).sType$Default().dstSet(descriptorSet).dstBinding(FOG_VOL_A)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(volAInfo);
            writes.get(FOG_VOL_B).sType$Default().dstSet(descriptorSet).dstBinding(FOG_VOL_B)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(volBInfo);
            writes.get(FOG_VOL_A_RW).sType$Default().dstSet(descriptorSet).dstBinding(FOG_VOL_A_RW)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(volARwInfo);
            writes.get(FOG_VOL_B_RW).sType$Default().dstSet(descriptorSet).dstBinding(FOG_VOL_B_RW)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(volBRwInfo);
            writes.get(FOG_AGE_A).sType$Default().dstSet(descriptorSet).dstBinding(FOG_AGE_A)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(ageARoInfo);
            writes.get(FOG_AGE_B).sType$Default().dstSet(descriptorSet).dstBinding(FOG_AGE_B)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(ageBRoInfo);
            writes.get(FOG_AGE_A_RW).sType$Default().dstSet(descriptorSet).dstBinding(FOG_AGE_A_RW)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(ageARwInfo);
            writes.get(FOG_AGE_B_RW).sType$Default().dstSet(descriptorSet).dstBinding(FOG_AGE_B_RW)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(ageBRwInfo);
            writes.get(FOG_PREV_DEPTH).sType$Default().dstSet(descriptorSet).dstBinding(FOG_PREV_DEPTH)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(prevDepthRoInfo);
            writes.get(FOG_PREV_DEPTH_RW).sType$Default().dstSet(descriptorSet).dstBinding(FOG_PREV_DEPTH_RW)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(prevDepthRwInfo);
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundOutputView = outputImageView;
        boundDepthView = depthImageView;
        boundFreshView = freshView;
        boundVolAView = volAView;
        boundVolBView = volBView;
        boundAgeAView = ageAView;
        boundAgeBView = ageBView;
        boundPrevDepthView = prevDepthView;
    }

    public void dispatch(VkCommandBuffer cmd, int width, int height, long worldPushAddress) {
        dispatchPipeline(cmd, pipeline, (width + 15) / 16, (height + 15) / 16, 1, worldPushAddress,
                "fog apply");
    }

    public void dispatchAccumulate(VkCommandBuffer cmd, int width, int height, int depth,
            long worldPushAddress) {
        dispatchPipeline(cmd, accumulatePipeline, (width + 3) / 4, (height + 3) / 4, (depth + 3) / 4,
                worldPushAddress, "fog accumulate");
    }

    private void dispatchPipeline(VkCommandBuffer cmd, long pipeline, int groupsX, int groupsY,
            int groupsZ, long worldPushAddress, String label) {
        try (MemoryStack stack = MemoryStack.stackPush();
                RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, label)) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                    stack.longs(descriptorSet), null);
            ByteBuffer push = stack.malloc(PushAddrData.BYTE_SIZE);
            new PushAddrData(worldPushAddress).write(push);
            VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, groupsX, groupsY, groupsZ);
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, pipeline, null);
        VK10.vkDestroyPipeline(vk, accumulatePipeline, null);
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        VK10.vkDestroySampler(vk, linearSampler, null);
        VK10.vkDestroySampler(vk, nearestSampler, null);
        destroyed = true;
    }

    private static long createSampler(VkDevice vk, MemoryStack stack, RtContext ctx, int filter,
            String label) {
        // Clamped: the volume's XY spans the NDC frustum and Z spans the slice range, so
        // out-of-range fetches (rounding at the frustum edge) must hold the boundary texel.
        VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                .magFilter(filter).minFilter(filter)
                .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                .minLod(0.0f).maxLod(0.0f);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreateSampler(vk, samplerInfo, null, handle), "vkCreateSampler(" + label + ")");
        long sampler = handle.get(0);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, label);
        return sampler;
    }

    private static long createComputePipeline(VkDevice vk, MemoryStack stack, RtContext ctx, long layout,
            String spvName, String label) {
        long module = loadModule(vk, stack, spvName);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, module, label + " shader module");
        VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack).sType$Default()
                .stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(module).pName(stack.UTF8("main"));
        VkComputePipelineCreateInfo.Buffer cpci = VkComputePipelineCreateInfo.calloc(1, stack);
        cpci.get(0).sType$Default().stage(stage).layout(layout);
        LongBuffer pPipeline = stack.mallocLong(1);
        check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, cpci, null, pPipeline),
                "vkCreateComputePipelines(" + label + ")");
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, pPipeline.get(0), label + " compute pipeline");
        VK10.vkDestroyShaderModule(vk, module, null);
        return pPipeline.get(0);
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String name) {
        byte[] bytes;
        try (InputStream in = RtFogPipeline.class.getResourceAsStream(SHADER_DIR + name)) {
            if (in == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + SHADER_DIR + name);
            }
            bytes = in.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + SHADER_DIR + name, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo smci = VkShaderModuleCreateInfo.calloc(stack).sType$Default().pCode(code);
            LongBuffer pModule = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, smci, null, pModule), "vkCreateShaderModule(" + name + ")");
            return pModule.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}
