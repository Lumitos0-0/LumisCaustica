package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.VolumetricPushData;
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
 * 3D Froxel Frustum Volume Grid pipeline (Strategy A).
 * Evaluates hardware ray-queried direct sun/moon shafts and Hillaire sky-view volumetric GI in compute,
 * temporally accumulates across frames, and integrates along camera depth slices.
 */
public final class RtVolumetricPipeline {
    private static final String SHADER_DIR = "/caustica/shaders/pipelines/volumetric/";

    public static final int FROXEL_WIDTH = 160;
    public static final int FROXEL_HEIGHT = 90;
    public static final int FROXEL_DEPTH = 64;
    private static final int GROUP_SIZE = 8;

    private final RtContext ctx;
    private final RtImage[] froxelRadiance = new RtImage[2];
    private final RtImage froxelExtinction;
    private final RtImage froxelAccumulatedLut;
    private final long sampler;

    private final long injectDescriptorSetLayout;
    private final long injectDescriptorPool;
    private final long[] injectDescriptorSets = new long[2];
    private final long injectPipelineLayout;
    private final long injectPipeline;

    private final long accumulateDescriptorSetLayout;
    private final long accumulateDescriptorPool;
    private final long[] accumulateDescriptorSets = new long[2];
    private final long accumulatePipelineLayout;
    private final long accumulatePipeline;

    private long boundTlas;
    private long boundSkyView;
    private long boundSkySampler;
    private long boundTransmittance;
    private long boundTransmittanceSampler;
    private int pingPongIndex = 0;
    private boolean destroyed;

    private RtVolumetricPipeline(RtContext ctx, RtImage rad0, RtImage rad1, RtImage ext, RtImage accum,
                                 long sampler, long injDsl, long injPool, long[] injSets,
                                 long injLayout, long injPipeline,
                                 long accDsl, long accPool, long[] accSets,
                                 long accLayout, long accPipeline) {
        this.ctx = ctx;
        this.froxelRadiance[0] = rad0;
        this.froxelRadiance[1] = rad1;
        this.froxelExtinction = ext;
        this.froxelAccumulatedLut = accum;
        this.sampler = sampler;

        this.injectDescriptorSetLayout = injDsl;
        this.injectDescriptorPool = injPool;
        this.injectDescriptorSets[0] = injSets[0];
        this.injectDescriptorSets[1] = injSets[1];
        this.injectPipelineLayout = injLayout;
        this.injectPipeline = injPipeline;

        this.accumulateDescriptorSetLayout = accDsl;
        this.accumulateDescriptorPool = accPool;
        this.accumulateDescriptorSets[0] = accSets[0];
        this.accumulateDescriptorSets[1] = accSets[1];
        this.accumulatePipelineLayout = accLayout;
        this.accumulatePipeline = accPipeline;
    }

    public static RtVolumetricPipeline create(RtContext ctx) {
        VkDevice vk = ctx.vk();

        RtImage rad0 = ctx.createStorageImage3D(FROXEL_WIDTH, FROXEL_HEIGHT, FROXEL_DEPTH,
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel radiance history 0");
        RtImage rad1 = ctx.createStorageImage3D(FROXEL_WIDTH, FROXEL_HEIGHT, FROXEL_DEPTH,
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel radiance history 1");
        RtImage ext = ctx.createStorageImage3D(FROXEL_WIDTH, FROXEL_HEIGHT, FROXEL_DEPTH,
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel extinction");
        RtImage accum = ctx.createStorageImage3D(FROXEL_WIDTH, FROXEL_HEIGHT, FROXEL_DEPTH,
                VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel accumulated LUT");

        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR).minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f).maxLod(0.0f);
            LongBuffer pSampler = stack.mallocLong(1);
            check(VK10.vkCreateSampler(vk, samplerInfo, null, pSampler), "vkCreateSampler(volumetric)");
            long sampler = pSampler.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "volumetric sampler");

            // --- Pass 1: Inject pipeline layout & descriptors ---
            VkDescriptorSetLayoutBinding.Buffer injBinds = VkDescriptorSetLayoutBinding.calloc(VOLUMETRIC_INJECT_BINDING_COUNT, stack);
            injBinds.get(VOLUMETRIC_INJECT_TLAS).binding(VOLUMETRIC_INJECT_TLAS)
                    .descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            injBinds.get(VOLUMETRIC_INJECT_SKY_VIEW).binding(VOLUMETRIC_INJECT_SKY_VIEW)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            injBinds.get(VOLUMETRIC_INJECT_TRANSMITTANCE).binding(VOLUMETRIC_INJECT_TRANSMITTANCE)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            injBinds.get(VOLUMETRIC_INJECT_PREV_RADIANCE).binding(VOLUMETRIC_INJECT_PREV_RADIANCE)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            injBinds.get(VOLUMETRIC_INJECT_OUT_RADIANCE).binding(VOLUMETRIC_INJECT_OUT_RADIANCE)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            injBinds.get(VOLUMETRIC_INJECT_OUT_EXTINCTION).binding(VOLUMETRIC_INJECT_OUT_EXTINCTION)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);

            VkDescriptorSetLayoutCreateInfo injDslci = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(injBinds);
            LongBuffer pHandle = stack.mallocLong(1);
            check(VK10.vkCreateDescriptorSetLayout(vk, injDslci, null, pHandle), "vkCreateDescriptorSetLayout(volumetric inject)");
            long injDsl = pHandle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, injDsl, "volumetric inject descriptor set layout");

            VkDescriptorPoolSize.Buffer injPoolSizes = VkDescriptorPoolSize.calloc(3, stack);
            injPoolSizes.get(0).type(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(2);
            injPoolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(6);
            injPoolSizes.get(2).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(4);
            VkDescriptorPoolCreateInfo injDpci = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default()
                    .maxSets(2).pPoolSizes(injPoolSizes);
            check(VK10.vkCreateDescriptorPool(vk, injDpci, null, pHandle), "vkCreateDescriptorPool(volumetric inject)");
            long injPool = pHandle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, injPool, "volumetric inject descriptor pool");

            VkDescriptorSetAllocateInfo injDsai = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                    .descriptorPool(injPool).pSetLayouts(stack.longs(injDsl, injDsl));
            LongBuffer pInjSets = stack.mallocLong(2);
            check(VK10.vkAllocateDescriptorSets(vk, injDsai, pInjSets), "vkAllocateDescriptorSets(volumetric inject)");
            long[] injSets = {pInjSets.get(0), pInjSets.get(1)};

            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
            pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(VolumetricPushData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo injPlci = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                    .pSetLayouts(stack.longs(injDsl)).pPushConstantRanges(pushRange);
            check(VK10.vkCreatePipelineLayout(vk, injPlci, null, pHandle), "vkCreatePipelineLayout(volumetric inject)");
            long injLayout = pHandle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, injLayout, "volumetric inject pipeline layout");

            long injModule = loadModule(vk, stack, SHADER_DIR + "froxel_inject.comp.spv");
            VkPipelineShaderStageCreateInfo injStage = VkPipelineShaderStageCreateInfo.calloc(stack).sType$Default()
                    .stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(injModule).pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer injCpci = VkComputePipelineCreateInfo.calloc(1, stack);
            injCpci.get(0).sType$Default().stage(injStage).layout(injLayout);
            check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, injCpci, null, pHandle),
                    "vkCreateComputePipelines(volumetric inject)");
            long injPipeline = pHandle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, injPipeline, "volumetric inject compute pipeline");
            VK10.vkDestroyShaderModule(vk, injModule, null);

            // --- Pass 2: Accumulate pipeline layout & descriptors ---
            VkDescriptorSetLayoutBinding.Buffer accBinds = VkDescriptorSetLayoutBinding.calloc(VOLUMETRIC_ACCUMULATE_BINDING_COUNT, stack);
            accBinds.get(VOLUMETRIC_ACCUMULATE_IN_RADIANCE).binding(VOLUMETRIC_ACCUMULATE_IN_RADIANCE)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            accBinds.get(VOLUMETRIC_ACCUMULATE_IN_EXTINCTION).binding(VOLUMETRIC_ACCUMULATE_IN_EXTINCTION)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            accBinds.get(VOLUMETRIC_ACCUMULATE_OUT_SCATTERING).binding(VOLUMETRIC_ACCUMULATE_OUT_SCATTERING)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);

            VkDescriptorSetLayoutCreateInfo accDslci = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(accBinds);
            check(VK10.vkCreateDescriptorSetLayout(vk, accDslci, null, pHandle), "vkCreateDescriptorSetLayout(volumetric accumulate)");
            long accDsl = pHandle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, accDsl, "volumetric accumulate descriptor set layout");

            VkDescriptorPoolSize.Buffer accPoolSizes = VkDescriptorPoolSize.calloc(1, stack);
            accPoolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(6);
            VkDescriptorPoolCreateInfo accDpci = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default()
                    .maxSets(2).pPoolSizes(accPoolSizes);
            check(VK10.vkCreateDescriptorPool(vk, accDpci, null, pHandle), "vkCreateDescriptorPool(volumetric accumulate)");
            long accPool = pHandle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, accPool, "volumetric accumulate descriptor pool");

            VkDescriptorSetAllocateInfo accDsai = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                    .descriptorPool(accPool).pSetLayouts(stack.longs(accDsl, accDsl));
            LongBuffer pAccSets = stack.mallocLong(2);
            check(VK10.vkAllocateDescriptorSets(vk, accDsai, pAccSets), "vkAllocateDescriptorSets(volumetric accumulate)");
            long[] accSets = {pAccSets.get(0), pAccSets.get(1)};

            VkPipelineLayoutCreateInfo accPlci = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                    .pSetLayouts(stack.longs(accDsl)).pPushConstantRanges(pushRange);
            check(VK10.vkCreatePipelineLayout(vk, accPlci, null, pHandle), "vkCreatePipelineLayout(volumetric accumulate)");
            long accLayout = pHandle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, accLayout, "volumetric accumulate pipeline layout");

            long accModule = loadModule(vk, stack, SHADER_DIR + "froxel_accumulate.comp.spv");
            VkPipelineShaderStageCreateInfo accStage = VkPipelineShaderStageCreateInfo.calloc(stack).sType$Default()
                    .stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(accModule).pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer accCpci = VkComputePipelineCreateInfo.calloc(1, stack);
            accCpci.get(0).sType$Default().stage(accStage).layout(accLayout);
            check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, accCpci, null, pHandle),
                    "vkCreateComputePipelines(volumetric accumulate)");
            long accPipeline = pHandle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, accPipeline, "volumetric accumulate compute pipeline");
            VK10.vkDestroyShaderModule(vk, accModule, null);

            // Wire static image bindings for both ping-pong slots
            for (int i = 0; i < 2; i++) {
                int prev = 1 - i;
                VkDescriptorImageInfo.Buffer prevRadInfo = VkDescriptorImageInfo.calloc(1, stack);
                prevRadInfo.get(0).imageView(i == 0 ? rad1.view : rad0.view).sampler(sampler)
                        .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                VkDescriptorImageInfo.Buffer outRadInfo = VkDescriptorImageInfo.calloc(1, stack);
                outRadInfo.get(0).imageView(i == 0 ? rad0.view : rad1.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                VkDescriptorImageInfo.Buffer outExtInfo = VkDescriptorImageInfo.calloc(1, stack);
                outExtInfo.get(0).imageView(ext.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

                VkWriteDescriptorSet.Buffer injStaticWrites = VkWriteDescriptorSet.calloc(3, stack);
                injStaticWrites.get(0).sType$Default().dstSet(injSets[i]).dstBinding(VOLUMETRIC_INJECT_PREV_RADIANCE)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(prevRadInfo);
                injStaticWrites.get(1).sType$Default().dstSet(injSets[i]).dstBinding(VOLUMETRIC_INJECT_OUT_RADIANCE)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(outRadInfo);
                injStaticWrites.get(2).sType$Default().dstSet(injSets[i]).dstBinding(VOLUMETRIC_INJECT_OUT_EXTINCTION)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(outExtInfo);
                VK10.vkUpdateDescriptorSets(vk, injStaticWrites, null);

                // Accumulate descriptors
                VkDescriptorImageInfo.Buffer inRadInfo = VkDescriptorImageInfo.calloc(1, stack);
                inRadInfo.get(0).imageView(i == 0 ? rad0.view : rad1.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                VkDescriptorImageInfo.Buffer inExtInfo = VkDescriptorImageInfo.calloc(1, stack);
                inExtInfo.get(0).imageView(ext.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                VkDescriptorImageInfo.Buffer outScatInfo = VkDescriptorImageInfo.calloc(1, stack);
                outScatInfo.get(0).imageView(accum.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

                VkWriteDescriptorSet.Buffer accStaticWrites = VkWriteDescriptorSet.calloc(3, stack);
                accStaticWrites.get(0).sType$Default().dstSet(accSets[i]).dstBinding(VOLUMETRIC_ACCUMULATE_IN_RADIANCE)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(inRadInfo);
                accStaticWrites.get(1).sType$Default().dstSet(accSets[i]).dstBinding(VOLUMETRIC_ACCUMULATE_IN_EXTINCTION)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(inExtInfo);
                accStaticWrites.get(2).sType$Default().dstSet(accSets[i]).dstBinding(VOLUMETRIC_ACCUMULATE_OUT_SCATTERING)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(outScatInfo);
                VK10.vkUpdateDescriptorSets(vk, accStaticWrites, null);
            }

            return new RtVolumetricPipeline(ctx, rad0, rad1, ext, accum, sampler,
                    injDsl, injPool, injSets, injLayout, injPipeline,
                    accDsl, accPool, accSets, accLayout, accPipeline);
        }
    }

    public void setResources(long tlas, long skyViewView, long skyViewSampler,
                             long transmittanceView, long transmittanceSampler) {
        if (boundTlas == tlas && boundSkyView == skyViewView && boundSkySampler == skyViewSampler
                && boundTransmittance == transmittanceView && boundTransmittanceSampler == transmittanceSampler) {
            return;
        }
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            for (int i = 0; i < 2; i++) {
                VkWriteDescriptorSetAccelerationStructureKHR asWrite = VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack)
                        .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR)
                        .pAccelerationStructures(stack.longs(tlas));

                VkDescriptorImageInfo.Buffer skyInfo = VkDescriptorImageInfo.calloc(1, stack);
                skyInfo.get(0).imageView(skyViewView).sampler(skyViewSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

                VkDescriptorImageInfo.Buffer transInfo = VkDescriptorImageInfo.calloc(1, stack);
                transInfo.get(0).imageView(transmittanceView).sampler(transmittanceSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

                VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(3, stack);
                writes.get(0).sType$Default().pNext(asWrite.address()).dstSet(injectDescriptorSets[i])
                        .dstBinding(VOLUMETRIC_INJECT_TLAS)
                        .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
                writes.get(1).sType$Default().dstSet(injectDescriptorSets[i]).dstBinding(VOLUMETRIC_INJECT_SKY_VIEW)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(skyInfo);
                writes.get(2).sType$Default().dstSet(injectDescriptorSets[i]).dstBinding(VOLUMETRIC_INJECT_TRANSMITTANCE)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(transInfo);

                VK10.vkUpdateDescriptorSets(vk, writes, null);
            }
        }
        boundTlas = tlas;
        boundSkyView = skyViewView;
        boundSkySampler = skyViewSampler;
        boundTransmittance = transmittanceView;
        boundTransmittanceSampler = transmittanceSampler;
    }

    public void record(VkCommandBuffer cmd, long worldPushAddress, int frameIndex) {
        if (boundTlas == 0L || boundSkyView == 0L) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "volumetric froxel passes")) {

            int currentSlot = pingPongIndex;

            // --- Pass 1: Inject direct sun shafts & sky GI into 3D froxel volume ---
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, injectPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    injectPipelineLayout, 0, stack.longs(injectDescriptorSets[currentSlot]), null);

            ByteBuffer push = stack.malloc(VolumetricPushData.BYTE_SIZE);
            new VolumetricPushData(worldPushAddress,
                    CausticaConfig.Rt.Volumetrics.DENSITY.value(),
                    CausticaConfig.Rt.Volumetrics.HEIGHT_FALLOFF.value(),
                    64.0f,
                    CausticaConfig.Rt.Volumetrics.ANISOTROPY.value(),
                    CausticaConfig.Rt.Volumetrics.TEMPORAL_WEIGHT.value(),
                    CausticaConfig.Rt.Volumetrics.GI_STRENGTH.value(),
                    CausticaConfig.Rt.Volumetrics.DIRECT_STRENGTH.value(),
                    CausticaConfig.Rt.Volumetrics.DITHER_STRENGTH.value(),
                    CausticaConfig.Rt.Volumetrics.QUALITY.value(),
                    frameIndex).write(push);
            VK10.vkCmdPushConstants(cmd, injectPipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);

            VK10.vkCmdDispatch(cmd, (FROXEL_WIDTH + GROUP_SIZE - 1) / GROUP_SIZE,
                    (FROXEL_HEIGHT + GROUP_SIZE - 1) / GROUP_SIZE, FROXEL_DEPTH);

            VulkanCommandEncoder.memoryBarrier(cmd, stack);

            // --- Pass 2: Front-to-back Z-slice accumulation ---
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, accumulatePipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    accumulatePipelineLayout, 0, stack.longs(accumulateDescriptorSets[currentSlot]), null);

            VK10.vkCmdPushConstants(cmd, accumulatePipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);

            VK10.vkCmdDispatch(cmd, (FROXEL_WIDTH + GROUP_SIZE - 1) / GROUP_SIZE,
                    (FROXEL_HEIGHT + GROUP_SIZE - 1) / GROUP_SIZE, 1);

            VulkanCommandEncoder.memoryBarrier(cmd, stack);

            pingPongIndex = 1 - pingPongIndex;
        }
    }

    public long volumetricLutView() {
        return froxelAccumulatedLut.view;
    }

    public long sampler() {
        return sampler;
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, accumulatePipeline, null);
        VK10.vkDestroyPipelineLayout(vk, accumulatePipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, accumulateDescriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, accumulateDescriptorSetLayout, null);

        VK10.vkDestroyPipeline(vk, injectPipeline, null);
        VK10.vkDestroyPipelineLayout(vk, injectPipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, injectDescriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, injectDescriptorSetLayout, null);

        VK10.vkDestroySampler(vk, sampler, null);
        froxelAccumulatedLut.destroy();
        froxelExtinction.destroy();
        froxelRadiance[0].destroy();
        froxelRadiance[1].destroy();
        destroyed = true;
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String resource) {
        byte[] bytes;
        try (InputStream input = RtVolumetricPipeline.class.getResourceAsStream(resource)) {
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
