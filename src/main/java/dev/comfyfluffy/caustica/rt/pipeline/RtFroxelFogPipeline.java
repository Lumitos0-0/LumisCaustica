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
import org.lwjgl.vulkan.VkImageMemoryBarrier;
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
 * Compute pipelines and descriptor-ring owner for the primary-view fog volume. A descriptor slot is not
 * rewritten until the graphics completion token that consumed it has signalled; this covers the TLAS,
 * current fields, histories, and the sky LUT views as one coherent frame manifest.
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
        for (int i = 0; i < RING; i++) descriptorSetUses[i] = new RtGpuExecutor.TrackedGraphicsUse();
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
            for (int binding = FOG_LIGHTING_DIRECT; binding <= FOG_LIGHTING_GI_AUX; binding++) {
                lightingBindings.get(binding).binding(binding)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1)
                        .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            for (int binding = FOG_LIGHTING_DIRECT_HISTORY; binding <= FOG_LIGHTING_TRANSMITTANCE; binding++) {
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
            long lightingPool = createPool(vk, stack, RING, RING * 5, RING * 9, "fog lighting");
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
            for (int binding = FOG_INTEGRATE_DEPTH; binding <= FOG_INTEGRATE_CACHE; binding++) {
                integrateBindings.get(binding).binding(binding)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1)
                        .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            VkDescriptorSetLayoutCreateInfo integrateLayoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(integrateBindings);
            check(VK10.vkCreateDescriptorSetLayout(vk, integrateLayoutInfo, null, handle),
                    "vkCreateDescriptorSetLayout(fog integrate)");
            long integrateLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, integrateLayout,
                    "fog integration descriptor set layout");
            long integratePool = createPool(vk, stack, RING, RING, RING * 5, "fog integrate");
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

    public void bindFrame(long tlas, RtImage output, RtImage depth,
                          RtImage direct, RtImage local, RtImage gi, RtImage cache, RtImage giAux,
                          RtImage directHistory, RtImage localHistory, RtImage giHistory,
                          RtImage giAuxHistory, RtImage cacheHistory,
                          long skyView, long transmittance, long skySampler,
                          RtGpuExecutor.GraphicsUse graphicsUse,
                          RtGpuExecutor.GraphicsUseWaiter graphicsUseWaiter) {
        if (destroyed) throw new IllegalStateException("fog pipeline is destroyed");
        currentSet = (currentSet + 1) % RING;
        graphicsUseWaiter.await(descriptorSetUses[currentSet]);
        writeLightingDescriptors(currentSet, tlas, output, depth, direct, local, gi, cache, giAux,
                directHistory, localHistory, giHistory, giAuxHistory, cacheHistory,
                skyView, transmittance, skySampler);
        writeIntegrateDescriptors(currentSet, output, depth, direct, local, gi, cache);
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

    /** Make lighting's storage writes visible to integration's sampled reads without changing GENERAL layout. */
    public void barrierLightingToIntegration(VkCommandBuffer cmd, RtImage... images) {
        imageBarrier(cmd, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK10.VK_ACCESS_SHADER_WRITE_BIT,
                VK10.VK_ACCESS_SHADER_READ_BIT | VK10.VK_ACCESS_SHADER_WRITE_BIT, images);
    }

    /** Make both the sampled lighting fields and read histories available to the transfer copy. */
    public void barrierForHistoryCopy(VkCommandBuffer cmd, RtImage[] current, RtImage[] histories) {
        imageBarrier(cmd, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK10.VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK10.VK_ACCESS_SHADER_READ_BIT | VK10.VK_ACCESS_SHADER_WRITE_BIT,
                VK10.VK_ACCESS_TRANSFER_READ_BIT, current);
        imageBarrier(cmd, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK10.VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK10.VK_ACCESS_SHADER_READ_BIT,
                VK10.VK_ACCESS_TRANSFER_WRITE_BIT, histories);
    }

    /** Make copied history and current fields visible after the transfer copy; all fog images stay GENERAL. */
    public void barrierAfterHistoryCopy(VkCommandBuffer cmd, RtImage[] current, RtImage[] histories) {
        imageBarrier(cmd, VK10.VK_PIPELINE_STAGE_TRANSFER_BIT, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK10.VK_ACCESS_TRANSFER_READ_BIT,
                VK10.VK_ACCESS_SHADER_READ_BIT | VK10.VK_ACCESS_SHADER_WRITE_BIT, current);
        imageBarrier(cmd, VK10.VK_PIPELINE_STAGE_TRANSFER_BIT, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK10.VK_ACCESS_TRANSFER_WRITE_BIT,
                VK10.VK_ACCESS_SHADER_READ_BIT, histories);
    }

    private static void imageBarrier(VkCommandBuffer cmd, int srcStage, int dstStage,
                                     int srcAccess, int dstAccess, RtImage... images) {
        if (images.length == 0) return;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkImageMemoryBarrier.Buffer barriers = VkImageMemoryBarrier.calloc(images.length, stack);
            for (int i = 0; i < images.length; i++) {
                barriers.get(i).sType$Default()
                        .srcAccessMask(srcAccess).dstAccessMask(dstAccess)
                        .oldLayout(VK10.VK_IMAGE_LAYOUT_GENERAL).newLayout(VK10.VK_IMAGE_LAYOUT_GENERAL)
                        .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                        .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                        .image(images[i].image);
                barriers.get(i).subresourceRange()
                        .aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                        .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
            }
            VK10.vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0,
                    null, null, barriers);
        }
    }

    public void copyHistory(VkCommandBuffer cmd, RtImage current, RtImage history) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "fog history copy")) {
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
        if (destroyed) return;
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

    private void writeLightingDescriptors(int setIndex, long tlas, RtImage output, RtImage depth,
                                          RtImage direct, RtImage local, RtImage gi, RtImage cache, RtImage giAux,
                                          RtImage directHistory, RtImage localHistory, RtImage giHistory,
                                          RtImage giAuxHistory, RtImage cacheHistory,
                                          long skyView, long transmittance, long skySampler) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkWriteDescriptorSetAccelerationStructureKHR asWrite =
                    VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack)
                            .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR)
                            .pAccelerationStructures(stack.longs(tlas));
            RtImage[] storageImages = {direct, local, gi, cache, giAux};
            RtImage[] historyImages = {directHistory, localHistory, giHistory, giAuxHistory, cacheHistory};
            VkDescriptorImageInfo.Buffer[] infos = new VkDescriptorImageInfo.Buffer[5];
            for (int i = 0; i < storageImages.length; i++) {
                infos[i] = VkDescriptorImageInfo.calloc(1, stack);
                infos[i].get(0).imageView(storageImages[i].view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            }
            VkDescriptorImageInfo.Buffer[] historyInfos = new VkDescriptorImageInfo.Buffer[5];
            for (int i = 0; i < historyImages.length; i++) {
                historyInfos[i] = VkDescriptorImageInfo.calloc(1, stack);
                historyInfos[i].get(0).sampler(linearSampler).imageView(historyImages[i].view)
                        .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            }
            VkDescriptorImageInfo.Buffer sceneInfo = VkDescriptorImageInfo.calloc(1, stack);
            sceneInfo.get(0).sampler(linearSampler).imageView(output.view)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer depthInfo = VkDescriptorImageInfo.calloc(1, stack);
            depthInfo.get(0).sampler(pointSampler).imageView(depth.view)
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
            for (int i = 0; i < storageImages.length; i++) {
                int binding = FOG_LIGHTING_DIRECT + i;
                writes.get(binding).sType$Default().dstSet(lightingDescriptorSets[setIndex]).dstBinding(binding)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(infos[i]);
            }
            for (int i = 0; i < historyImages.length; i++) {
                int binding = FOG_LIGHTING_DIRECT_HISTORY + i;
                writes.get(binding).sType$Default().dstSet(lightingDescriptorSets[setIndex]).dstBinding(binding)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(historyInfos[i]);
            }
            VkDescriptorImageInfo.Buffer[] sampled = {sceneInfo, depthInfo, skyInfo, transInfo};
            for (int i = 0; i < sampled.length; i++) {
                int binding = FOG_LIGHTING_SCENE + i;
                writes.get(binding).sType$Default().dstSet(lightingDescriptorSets[setIndex]).dstBinding(binding)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(sampled[i]);
            }
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
    }

    private void writeIntegrateDescriptors(int setIndex, RtImage output, RtImage depth,
                                           RtImage direct, RtImage local, RtImage gi, RtImage cache) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer outputInfo = VkDescriptorImageInfo.calloc(1, stack);
            outputInfo.get(0).imageView(output.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            RtImage[] volumes = {direct, local, gi, cache};
            VkDescriptorImageInfo.Buffer depthInfo = VkDescriptorImageInfo.calloc(1, stack);
            depthInfo.get(0).sampler(pointSampler).imageView(depth.view)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorImageInfo.Buffer[] sampled = new VkDescriptorImageInfo.Buffer[5];
            sampled[0] = depthInfo;
            for (int i = 0; i < volumes.length; i++) {
                sampled[i + 1] = VkDescriptorImageInfo.calloc(1, stack);
                sampled[i + 1].get(0).sampler(linearSampler).imageView(volumes[i].view)
                        .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            }
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(FOG_INTEGRATE_BINDING_COUNT, stack);
            writes.get(FOG_INTEGRATE_OUTPUT).sType$Default().dstSet(integrateDescriptorSets[setIndex])
                    .dstBinding(FOG_INTEGRATE_OUTPUT).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(outputInfo);
            for (int i = 0; i < sampled.length; i++) {
                int binding = FOG_INTEGRATE_DEPTH + i;
                writes.get(binding).sType$Default().dstSet(integrateDescriptorSets[setIndex]).dstBinding(binding)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(sampled[i]);
            }
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
        for (int i = 0; i < RING; i++) layouts.put(i, layout);
        VkDescriptorSetAllocateInfo allocateInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                .sType$Default().descriptorPool(pool).pSetLayouts(layouts);
        LongBuffer handles = stack.mallocLong(RING);
        check(VK10.vkAllocateDescriptorSets(vk, allocateInfo, handles),
                "vkAllocateDescriptorSets(" + label + ")");
        long[] result = new long[RING];
        for (int i = 0; i < RING; i++) result[i] = handles.get(i);
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
            if (input == null) throw new IllegalStateException("missing SPIR-V resource: " + resource);
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
