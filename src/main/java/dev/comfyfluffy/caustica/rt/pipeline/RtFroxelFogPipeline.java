package dev.comfyfluffy.caustica.rt.pipeline;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.RtGpuExecutor;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.FogIntegratePushData;
import dev.comfyfluffy.caustica.rt.gen.FogLightingPushData;
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
import org.lwjgl.vulkan.VkImageCopy;
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
 * Owns the standalone compute pipelines for the primary-view hybrid fog volume.
 *
 * <p>Lighting and integration use six descriptor slots just like the world RT pipeline. A slot is only
 * rewritten after its exact prior graphics use completes; this matters because the TLAS changes every
 * frame while the fog descriptors are consumed later in the same graphics submission. The fog pipeline
 * never modifies the world RT SBT or its descriptor ABI.
 */
public final class RtFroxelFogPipeline {
    private static final String LIGHTING_SHADER = "/caustica/shaders/pipelines/fog/fog_lighting.comp.spv";
    private static final String INTEGRATE_SHADER = "/caustica/shaders/pipelines/fog/fog_integrate.comp.spv";
    private static final int RING = 6;
    private static final int GROUP_SIZE = 8;

    private final RtContext ctx;
    private final long lightingDescriptorSetLayout;
    private final long lightingDescriptorPool;
    private final long[] lightingDescriptorSets;
    private final long lightingPipelineLayout;
    private final long lightingPipeline;
    private final long integrateDescriptorSetLayout;
    private final long integrateDescriptorPool;
    private final long[] integrateDescriptorSets;
    private final long integratePipelineLayout;
    private final long integratePipeline;
    private final RtGpuExecutor.TrackedGraphicsUse[] descriptorSetUses;
    private final long linearSampler;
    private final long pointSampler;
    private int currentSet;
    private boolean destroyed;

    private RtFroxelFogPipeline(RtContext ctx,
                                long lightingDescriptorSetLayout, long lightingDescriptorPool,
                                long[] lightingDescriptorSets, long lightingPipelineLayout, long lightingPipeline,
                                long integrateDescriptorSetLayout, long integrateDescriptorPool,
                                long[] integrateDescriptorSets, long integratePipelineLayout, long integratePipeline,
                                long linearSampler, long pointSampler) {
        this.ctx = ctx;
        this.lightingDescriptorSetLayout = lightingDescriptorSetLayout;
        this.lightingDescriptorPool = lightingDescriptorPool;
        this.lightingDescriptorSets = lightingDescriptorSets;
        this.lightingPipelineLayout = lightingPipelineLayout;
        this.lightingPipeline = lightingPipeline;
        this.integrateDescriptorSetLayout = integrateDescriptorSetLayout;
        this.integrateDescriptorPool = integrateDescriptorPool;
        this.integrateDescriptorSets = integrateDescriptorSets;
        this.integratePipelineLayout = integratePipelineLayout;
        this.integratePipeline = integratePipeline;
        this.descriptorSetUses = new RtGpuExecutor.TrackedGraphicsUse[RING];
        for (int i = 0; i < RING; i++) {
            descriptorSetUses[i] = new RtGpuExecutor.TrackedGraphicsUse();
        }
        this.linearSampler = linearSampler;
        this.pointSampler = pointSampler;
    }

    public static RtFroxelFogPipeline create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer handle = stack.mallocLong(1);
            long linearSampler = createSampler(vk, stack, VK10.VK_FILTER_LINEAR, "fog linear sampler");
            long pointSampler = createSampler(vk, stack, VK10.VK_FILTER_NEAREST, "fog point sampler");

            VkDescriptorSetLayoutBinding.Buffer lightingBindings =
                    VkDescriptorSetLayoutBinding.calloc(FOG_LIGHTING_BINDING_COUNT, stack);
            lightingBindings.get(FOG_LIGHTING_TLAS).binding(FOG_LIGHTING_TLAS)
                    .descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            lightingBindings.get(FOG_LIGHTING_DIRECT).binding(FOG_LIGHTING_DIRECT)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            lightingBindings.get(FOG_LIGHTING_GI).binding(FOG_LIGHTING_GI)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            for (int binding = FOG_LIGHTING_HISTORY; binding <= FOG_LIGHTING_TRANSMITTANCE; binding++) {
                lightingBindings.get(binding).binding(binding)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1)
                        .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            VkDescriptorSetLayoutCreateInfo lightingLayoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(lightingBindings);
            check(VK10.vkCreateDescriptorSetLayout(vk, lightingLayoutInfo, null, handle),
                    "vkCreateDescriptorSetLayout(fog lighting)");
            long lightingLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, lightingLayout,
                    "fog lighting descriptor set layout");

            long lightingPool = createPool(vk, stack, RING, RING * 2, RING * 4, "fog lighting");
            long[] lightingSets = allocateSets(vk, stack, lightingPool, lightingLayout, "fog lighting");

            VkPushConstantRange.Buffer lightingPushRange = VkPushConstantRange.calloc(1, stack);
            lightingPushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0)
                    .size(FogLightingPushData.BYTE_SIZE);
            long lightingPipelineLayout = createPipelineLayout(vk, stack, lightingLayout,
                    lightingPushRange, "fog lighting");
            long lightingModule = loadModule(vk, stack, LIGHTING_SHADER);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, lightingModule,
                    "fog lighting shader module");
            long lightingPipeline = createComputePipeline(vk, stack, lightingPipelineLayout,
                    lightingModule, "fog lighting");
            VK10.vkDestroyShaderModule(vk, lightingModule, null);

            VkDescriptorSetLayoutBinding.Buffer integrateBindings =
                    VkDescriptorSetLayoutBinding.calloc(FOG_INTEGRATE_BINDING_COUNT, stack);
            integrateBindings.get(FOG_INTEGRATE_OUTPUT).binding(FOG_INTEGRATE_OUTPUT)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            integrateBindings.get(FOG_INTEGRATE_DEPTH).binding(FOG_INTEGRATE_DEPTH)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            integrateBindings.get(FOG_INTEGRATE_DIRECT).binding(FOG_INTEGRATE_DIRECT)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            integrateBindings.get(FOG_INTEGRATE_GI).binding(FOG_INTEGRATE_GI)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorSetLayoutCreateInfo integrateLayoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(integrateBindings);
            check(VK10.vkCreateDescriptorSetLayout(vk, integrateLayoutInfo, null, handle),
                    "vkCreateDescriptorSetLayout(fog integrate)");
            long integrateLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, integrateLayout,
                    "fog integration descriptor set layout");

            long integratePool = createPool(vk, stack, RING, RING, RING * 3, "fog integrate");
            long[] integrateSets = allocateSets(vk, stack, integratePool, integrateLayout, "fog integrate");

            VkPushConstantRange.Buffer integratePushRange = VkPushConstantRange.calloc(1, stack);
            integratePushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0)
                    .size(FogIntegratePushData.BYTE_SIZE);
            long integratePipelineLayout = createPipelineLayout(vk, stack, integrateLayout,
                    integratePushRange, "fog integrate");
            long integrateModule = loadModule(vk, stack, INTEGRATE_SHADER);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, integrateModule,
                    "fog integration shader module");
            long integratePipeline = createComputePipeline(vk, stack, integratePipelineLayout,
                    integrateModule, "fog integrate");
            VK10.vkDestroyShaderModule(vk, integrateModule, null);

            return new RtFroxelFogPipeline(ctx, lightingLayout, lightingPool, lightingSets,
                    lightingPipelineLayout, lightingPipeline, integrateLayout, integratePool,
                    integrateSets, integratePipelineLayout, integratePipeline, linearSampler, pointSampler);
        }
    }

    /**
     * Select and bind one descriptor slot for this frame. The slot contains both passes, so the TLAS and
     * all sampled/storage images describe one coherent frame when lighting and integration are recorded.
     */
    public void bindFrame(long tlas, RtImage output, RtImage depth, RtImage direct, RtImage gi,
                          RtImage history, long skyView, long transmittance, long skySampler,
                          RtGpuExecutor.GraphicsUse graphicsUse,
                          RtGpuExecutor.GraphicsUseWaiter graphicsUseWaiter) {
        if (destroyed) {
            throw new IllegalStateException("fog pipeline is destroyed");
        }
        currentSet = (currentSet + 1) % RING;
        graphicsUseWaiter.await(descriptorSetUses[currentSet]);
        writeLightingDescriptors(currentSet, tlas, output, direct, gi, history,
                skyView, transmittance, skySampler);
        writeIntegrateDescriptors(currentSet, output, depth, direct, gi);
        descriptorSetUses[currentSet].mark(graphicsUse);
    }

    public void dispatchLighting(VkCommandBuffer cmd, int atlasWidth, int atlasHeight,
                                 FogLightingPushData pushData) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "fog froxel lighting")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, lightingPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    lightingPipelineLayout, 0, stack.longs(lightingDescriptorSets[currentSet]), null);
            ByteBuffer push = stack.malloc(FogLightingPushData.BYTE_SIZE);
            pushData.write(push);
            VK10.vkCmdPushConstants(cmd, lightingPipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (atlasWidth + GROUP_SIZE - 1) / GROUP_SIZE,
                    (atlasHeight + GROUP_SIZE - 1) / GROUP_SIZE, 1);
        }
    }

    public void dispatchIntegrate(VkCommandBuffer cmd, int width, int height,
                                  FogIntegratePushData pushData) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "fog camera integration")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, integratePipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    integratePipelineLayout, 0, stack.longs(integrateDescriptorSets[currentSet]), null);
            ByteBuffer push = stack.malloc(FogIntegratePushData.BYTE_SIZE);
            pushData.write(push);
            VK10.vkCmdPushConstants(cmd, integratePipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (width + GROUP_SIZE - 1) / GROUP_SIZE,
                    (height + GROUP_SIZE - 1) / GROUP_SIZE, 1);
        }
    }

    public void copyHistory(VkCommandBuffer cmd, RtImage current, RtImage history) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "fog GI history")) {
            VkImageCopy.Buffer region = VkImageCopy.calloc(1, stack);
            region.get(0).srcSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                    .mipLevel(0).baseArrayLayer(0).layerCount(1);
            region.get(0).dstSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                    .mipLevel(0).baseArrayLayer(0).layerCount(1);
            region.get(0).extent().set(current.width, current.height, 1);
            VK10.vkCmdCopyImage(cmd, current.image, VK10.VK_IMAGE_LAYOUT_GENERAL,
                    history.image, VK10.VK_IMAGE_LAYOUT_GENERAL, region);
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, integratePipeline, null);
        VK10.vkDestroyPipelineLayout(vk, integratePipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, integrateDescriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, integrateDescriptorSetLayout, null);
        VK10.vkDestroyPipeline(vk, lightingPipeline, null);
        VK10.vkDestroyPipelineLayout(vk, lightingPipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, lightingDescriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, lightingDescriptorSetLayout, null);
        VK10.vkDestroySampler(vk, linearSampler, null);
        VK10.vkDestroySampler(vk, pointSampler, null);
        destroyed = true;
    }

    private void writeLightingDescriptors(int setIndex, long tlas, RtImage output, RtImage direct,
                                          RtImage gi, RtImage history, long skyView, long transmittance,
                                          long skySampler) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkWriteDescriptorSetAccelerationStructureKHR asWrite =
                    VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack)
                            .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR)
                            .pAccelerationStructures(stack.longs(tlas));
            VkDescriptorImageInfo.Buffer directInfo = VkDescriptorImageInfo.calloc(1, stack);
            directInfo.get(0).imageView(direct.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer giInfo = VkDescriptorImageInfo.calloc(1, stack);
            giInfo.get(0).imageView(gi.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer historyInfo = VkDescriptorImageInfo.calloc(1, stack);
            historyInfo.get(0).sampler(linearSampler).imageView(history.view)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer sceneInfo = VkDescriptorImageInfo.calloc(1, stack);
            sceneInfo.get(0).sampler(linearSampler).imageView(output.view)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer skyInfo = VkDescriptorImageInfo.calloc(1, stack);
            skyInfo.get(0).sampler(skySampler).imageView(skyView)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer transInfo = VkDescriptorImageInfo.calloc(1, stack);
            transInfo.get(0).sampler(skySampler).imageView(transmittance)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(FOG_LIGHTING_BINDING_COUNT, stack);
            writes.get(FOG_LIGHTING_TLAS).sType$Default().pNext(asWrite.address())
                    .dstSet(lightingDescriptorSets[setIndex]).dstBinding(FOG_LIGHTING_TLAS)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
            writes.get(FOG_LIGHTING_DIRECT).sType$Default().dstSet(lightingDescriptorSets[setIndex])
                    .dstBinding(FOG_LIGHTING_DIRECT).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(directInfo);
            writes.get(FOG_LIGHTING_GI).sType$Default().dstSet(lightingDescriptorSets[setIndex])
                    .dstBinding(FOG_LIGHTING_GI).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(giInfo);
            writes.get(FOG_LIGHTING_HISTORY).sType$Default().dstSet(lightingDescriptorSets[setIndex])
                    .dstBinding(FOG_LIGHTING_HISTORY).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(historyInfo);
            writes.get(FOG_LIGHTING_SCENE).sType$Default().dstSet(lightingDescriptorSets[setIndex])
                    .dstBinding(FOG_LIGHTING_SCENE).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(sceneInfo);
            writes.get(FOG_LIGHTING_SKY_VIEW).sType$Default().dstSet(lightingDescriptorSets[setIndex])
                    .dstBinding(FOG_LIGHTING_SKY_VIEW).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(skyInfo);
            writes.get(FOG_LIGHTING_TRANSMITTANCE).sType$Default().dstSet(lightingDescriptorSets[setIndex])
                    .dstBinding(FOG_LIGHTING_TRANSMITTANCE).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(transInfo);
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
    }

    private void writeIntegrateDescriptors(int setIndex, RtImage output, RtImage depth,
                                           RtImage direct, RtImage gi) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer outputInfo = VkDescriptorImageInfo.calloc(1, stack);
            outputInfo.get(0).imageView(output.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer depthInfo = VkDescriptorImageInfo.calloc(1, stack);
            depthInfo.get(0).sampler(pointSampler).imageView(depth.view)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer directInfo = VkDescriptorImageInfo.calloc(1, stack);
            directInfo.get(0).sampler(linearSampler).imageView(direct.view)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer giInfo = VkDescriptorImageInfo.calloc(1, stack);
            giInfo.get(0).sampler(linearSampler).imageView(gi.view)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(FOG_INTEGRATE_BINDING_COUNT, stack);
            writes.get(FOG_INTEGRATE_OUTPUT).sType$Default().dstSet(integrateDescriptorSets[setIndex])
                    .dstBinding(FOG_INTEGRATE_OUTPUT).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(outputInfo);
            writes.get(FOG_INTEGRATE_DEPTH).sType$Default().dstSet(integrateDescriptorSets[setIndex])
                    .dstBinding(FOG_INTEGRATE_DEPTH).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(depthInfo);
            writes.get(FOG_INTEGRATE_DIRECT).sType$Default().dstSet(integrateDescriptorSets[setIndex])
                    .dstBinding(FOG_INTEGRATE_DIRECT).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(directInfo);
            writes.get(FOG_INTEGRATE_GI).sType$Default().dstSet(integrateDescriptorSets[setIndex])
                    .dstBinding(FOG_INTEGRATE_GI).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(giInfo);
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
    }

    private static long createSampler(VkDevice vk, MemoryStack stack, int filter, String label) {
        VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                .magFilter(filter).minFilter(filter)
                .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                .minLod(0.0f).maxLod(0.0f);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreateSampler(vk, samplerInfo, null, handle), "vkCreateSampler(" + label + ")");
        return handle.get(0);
    }

    private static long createPool(VkDevice vk, MemoryStack stack, int accelerationStructures,
                                   int storageImages, int combinedSamplers, String label) {
        int count = 0;
        if (accelerationStructures > 0) count++;
        if (storageImages > 0) count++;
        if (combinedSamplers > 0) count++;
        VkDescriptorPoolSize.Buffer sizes = VkDescriptorPoolSize.calloc(count, stack);
        int index = 0;
        if (accelerationStructures > 0) {
            sizes.get(index++).type(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                    .descriptorCount(accelerationStructures);
        }
        if (storageImages > 0) {
            sizes.get(index++).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(storageImages);
        }
        if (combinedSamplers > 0) {
            sizes.get(index).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(combinedSamplers);
        }
        VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                .sType$Default().maxSets(RING).pPoolSizes(sizes);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreateDescriptorPool(vk, poolInfo, null, handle),
                "vkCreateDescriptorPool(" + label + ")");
        return handle.get(0);
    }

    private static long[] allocateSets(VkDevice vk, MemoryStack stack, long pool, long layout, String label) {
        LongBuffer layouts = stack.mallocLong(RING);
        for (int i = 0; i < RING; i++) {
            layouts.put(i, layout);
        }
        VkDescriptorSetAllocateInfo allocateInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                .sType$Default().descriptorPool(pool).pSetLayouts(layouts);
        LongBuffer handles = stack.mallocLong(RING);
        check(VK10.vkAllocateDescriptorSets(vk, allocateInfo, handles),
                "vkAllocateDescriptorSets(" + label + ")");
        long[] result = new long[RING];
        for (int i = 0; i < RING; i++) {
            result[i] = handles.get(i);
        }
        return result;
    }

    private static long createPipelineLayout(VkDevice vk, MemoryStack stack, long descriptorSetLayout,
                                             VkPushConstantRange.Buffer pushRange, String label) {
        VkPipelineLayoutCreateInfo info = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                .pSetLayouts(stack.longs(descriptorSetLayout)).pPushConstantRanges(pushRange);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreatePipelineLayout(vk, info, null, handle),
                "vkCreatePipelineLayout(" + label + ")");
        return handle.get(0);
    }

    private static long createComputePipeline(VkDevice vk, MemoryStack stack, long pipelineLayout,
                                              long module, String label) {
        VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                .sType$Default().stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(module)
                .pName(stack.UTF8("main"));
        VkComputePipelineCreateInfo.Buffer info = VkComputePipelineCreateInfo.calloc(1, stack);
        info.get(0).sType$Default().stage(stage).layout(pipelineLayout);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, info, null, handle),
                "vkCreateComputePipelines(" + label + ")");
        return handle.get(0);
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String resource) {
        byte[] bytes;
        try (InputStream input = RtFroxelFogPipeline.class.getResourceAsStream(resource)) {
            if (input == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + resource);
            }
            bytes = input.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + resource, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes).flip();
        try {
            VkShaderModuleCreateInfo info = VkShaderModuleCreateInfo.calloc(stack).sType$Default().pCode(code);
            LongBuffer handle = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, info, null, handle),
                    "vkCreateShaderModule(" + resource + ")");
            return handle.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}
