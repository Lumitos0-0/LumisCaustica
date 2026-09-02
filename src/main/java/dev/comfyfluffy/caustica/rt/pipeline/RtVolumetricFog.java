package dev.comfyfluffy.caustica.rt.pipeline;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.FogIntegrationPushData;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.KHRAccelerationStructure;
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

/**
 * Froxel frustum grid volumetric fog.
 */
public final class RtVolumetricFog {
    private static final String SHADER_DIR = "/caustica/shaders/pipelines/volumetric_fog/";

    private final RtContext ctx;
    private final long injectionDsl;
    private final long injectionPool;
    private final long injectionSet;
    private final long injectionLayout;
    private final long injectionPipeline;

    private final long resolveDsl;
    private final long resolvePool;
    private final long resolveSet;
    private final long resolveLayout;
    private final long resolvePipeline;

    private final long integrationDsl;
    private final long integrationPool;
    private final long integrationSet;
    private final long integrationLayout;
    private final long integrationPipeline;

    private final long compositeDsl;
    private final long compositePool;
    private final long compositeSet;
    private final long compositeLayout;
    private final long compositePipeline;

    private final long linearSampler;
    private final long nearestSampler;

    private RtImage rawFroxelVolume;
    private RtImage[] filteredFroxelVolumes = new RtImage[0];
    private RtImage integratedFroxelVolume;
    private int historyReadIndex;
    private int historyWriteIndex;
    private boolean historyValid;

    private long boundInjectionFroxelView;
    private long boundResolveInputView;
    private long boundResolveHistoryView;
    private long boundResolveOutputView;
    private long boundIntegrationInputView;
    private long boundIntegrationOutputView;
    private long boundCompositeLocalView;
    private long boundCompositeIntegratedView;
    private long boundFogDepthView;
    private long boundSceneColorView;
    private long boundBlockAlbedoView;
    private boolean destroyed;

    private RtVolumetricFog(RtContext ctx,
                            long injectionDsl, long injectionPool, long injectionSet, long injectionLayout, long injectionPipeline,
                            long resolveDsl, long resolvePool, long resolveSet, long resolveLayout, long resolvePipeline,
                            long integrationDsl, long integrationPool, long integrationSet, long integrationLayout, long integrationPipeline,
                            long compositeDsl, long compositePool, long compositeSet, long compositeLayout, long compositePipeline,
                            long linearSampler, long nearestSampler) {
        this.ctx = ctx;
        this.injectionDsl = injectionDsl;
        this.injectionPool = injectionPool;
        this.injectionSet = injectionSet;
        this.injectionLayout = injectionLayout;
        this.injectionPipeline = injectionPipeline;
        this.resolveDsl = resolveDsl;
        this.resolvePool = resolvePool;
        this.resolveSet = resolveSet;
        this.resolveLayout = resolveLayout;
        this.resolvePipeline = resolvePipeline;
        this.integrationDsl = integrationDsl;
        this.integrationPool = integrationPool;
        this.integrationSet = integrationSet;
        this.integrationLayout = integrationLayout;
        this.integrationPipeline = integrationPipeline;
        this.compositeDsl = compositeDsl;
        this.compositePool = compositePool;
        this.compositeSet = compositeSet;
        this.compositeLayout = compositeLayout;
        this.compositePipeline = compositePipeline;
        this.linearSampler = linearSampler;
        this.nearestSampler = nearestSampler;
    }

    public static RtVolumetricFog create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer p = stack.mallocLong(1);
            LongBuffer pSet = stack.mallocLong(1);
            LongBuffer pPipe = stack.mallocLong(1);

            VkDescriptorSetLayoutBinding.Buffer injBinds = VkDescriptorSetLayoutBinding.calloc(3, stack);
            injBinds.get(0).binding(0).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            injBinds.get(1).binding(1).descriptorType(KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            injBinds.get(2).binding(2).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorSetLayoutCreateInfo injDslCi = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(injBinds);
            check(VK10.vkCreateDescriptorSetLayout(vk, injDslCi, null, p), "vkCreateDescriptorSetLayout(fog injection)");
            long injectionDsl = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, injectionDsl, "fog injection dsl");

            VkDescriptorPoolSize.Buffer injPoolSizes = VkDescriptorPoolSize.calloc(3, stack);
            injPoolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
            injPoolSizes.get(1).type(KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(1);
            injPoolSizes.get(2).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1);
            VkDescriptorPoolCreateInfo injPoolCi = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default().maxSets(1).pPoolSizes(injPoolSizes);
            check(VK10.vkCreateDescriptorPool(vk, injPoolCi, null, p), "vkCreateDescriptorPool(fog injection)");
            long injectionPool = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, injectionPool, "fog injection pool");

            VkDescriptorSetAllocateInfo injAlloc = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                    .descriptorPool(injectionPool).pSetLayouts(stack.longs(injectionDsl));
            check(VK10.vkAllocateDescriptorSets(vk, injAlloc, pSet), "vkAllocateDescriptorSets(fog injection)");
            long injectionSet = pSet.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, injectionSet, "fog injection set");

            VkPushConstantRange.Buffer injPushRange = VkPushConstantRange.calloc(1, stack);
            injPushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(FogIntegrationPushData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo injPlCi = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                    .pSetLayouts(stack.longs(injectionDsl)).pPushConstantRanges(injPushRange);
            check(VK10.vkCreatePipelineLayout(vk, injPlCi, null, p), "vkCreatePipelineLayout(fog injection)");
            long injectionLayout = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, injectionLayout, "fog injection layout");

            long injModule = loadModule(vk, stack, "injection.comp.spv");
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, injModule, "fog injection module");
            VkPipelineShaderStageCreateInfo injStage = VkPipelineShaderStageCreateInfo.calloc(stack).sType$Default()
                    .stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(injModule).pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer injPipeCi = VkComputePipelineCreateInfo.calloc(1, stack);
            injPipeCi.get(0).sType$Default().stage(injStage).layout(injectionLayout);
            check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, injPipeCi, null, pPipe), "vkCreateComputePipelines(fog injection)");
            long injectionPipeline = pPipe.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, injectionPipeline, "fog injection pipeline");
            VK10.vkDestroyShaderModule(vk, injModule, null);

            VkDescriptorSetLayoutBinding.Buffer resolveBinds = VkDescriptorSetLayoutBinding.calloc(3, stack);
            resolveBinds.get(0).binding(0).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            resolveBinds.get(1).binding(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            resolveBinds.get(2).binding(2).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorSetLayoutCreateInfo resolveDslCi = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(resolveBinds);
            check(VK10.vkCreateDescriptorSetLayout(vk, resolveDslCi, null, p), "vkCreateDescriptorSetLayout(fog resolve)");
            long resolveDsl = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, resolveDsl, "fog resolve dsl");

            VkDescriptorPoolSize.Buffer resolvePoolSizes = VkDescriptorPoolSize.calloc(2, stack);
            resolvePoolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(2);
            resolvePoolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
            VkDescriptorPoolCreateInfo resolvePoolCi = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default()
                    .maxSets(1).pPoolSizes(resolvePoolSizes);
            check(VK10.vkCreateDescriptorPool(vk, resolvePoolCi, null, p), "vkCreateDescriptorPool(fog resolve)");
            long resolvePool = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, resolvePool, "fog resolve pool");

            VkDescriptorSetAllocateInfo resolveAlloc = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                    .descriptorPool(resolvePool).pSetLayouts(stack.longs(resolveDsl));
            check(VK10.vkAllocateDescriptorSets(vk, resolveAlloc, pSet), "vkAllocateDescriptorSets(fog resolve)");
            long resolveSet = pSet.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, resolveSet, "fog resolve set");

            VkPushConstantRange.Buffer resolvePushRange = VkPushConstantRange.calloc(1, stack);
            resolvePushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(FogIntegrationPushData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo resolvePlCi = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                    .pSetLayouts(stack.longs(resolveDsl)).pPushConstantRanges(resolvePushRange);
            check(VK10.vkCreatePipelineLayout(vk, resolvePlCi, null, p), "vkCreatePipelineLayout(fog resolve)");
            long resolveLayout = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, resolveLayout, "fog resolve layout");

            long resolveModule = loadModule(vk, stack, "resolve.comp.spv");
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, resolveModule, "fog resolve module");
            VkPipelineShaderStageCreateInfo resolveStage = VkPipelineShaderStageCreateInfo.calloc(stack).sType$Default()
                    .stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(resolveModule).pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer resolvePipeCi = VkComputePipelineCreateInfo.calloc(1, stack);
            resolvePipeCi.get(0).sType$Default().stage(resolveStage).layout(resolveLayout);
            check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, resolvePipeCi, null, pPipe), "vkCreateComputePipelines(fog resolve)");
            long resolvePipeline = pPipe.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, resolvePipeline, "fog resolve pipeline");
            VK10.vkDestroyShaderModule(vk, resolveModule, null);

            VkDescriptorSetLayoutBinding.Buffer intBinds = VkDescriptorSetLayoutBinding.calloc(2, stack);
            intBinds.get(0).binding(0).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            intBinds.get(1).binding(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorSetLayoutCreateInfo intDslCi = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(intBinds);
            check(VK10.vkCreateDescriptorSetLayout(vk, intDslCi, null, p), "vkCreateDescriptorSetLayout(fog integration)");
            long integrationDsl = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, integrationDsl, "fog integration dsl");

            VkDescriptorPoolSize.Buffer intPoolSizes = VkDescriptorPoolSize.calloc(2, stack);
            intPoolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1);
            intPoolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
            VkDescriptorPoolCreateInfo intPoolCi = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default()
                    .maxSets(1).pPoolSizes(intPoolSizes);
            check(VK10.vkCreateDescriptorPool(vk, intPoolCi, null, p), "vkCreateDescriptorPool(fog integration)");
            long integrationPool = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, integrationPool, "fog integration pool");

            VkDescriptorSetAllocateInfo intAlloc = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                    .descriptorPool(integrationPool).pSetLayouts(stack.longs(integrationDsl));
            check(VK10.vkAllocateDescriptorSets(vk, intAlloc, pSet), "vkAllocateDescriptorSets(fog integration)");
            long integrationSet = pSet.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, integrationSet, "fog integration set");

            VkPushConstantRange.Buffer intPushRange = VkPushConstantRange.calloc(1, stack);
            intPushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(FogIntegrationPushData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo intPlCi = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                    .pSetLayouts(stack.longs(integrationDsl)).pPushConstantRanges(intPushRange);
            check(VK10.vkCreatePipelineLayout(vk, intPlCi, null, p), "vkCreatePipelineLayout(fog integration)");
            long integrationLayout = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, integrationLayout, "fog integration layout");

            long intModule = loadModule(vk, stack, "integration.comp.spv");
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, intModule, "fog integration module");
            VkPipelineShaderStageCreateInfo intStage = VkPipelineShaderStageCreateInfo.calloc(stack).sType$Default()
                    .stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(intModule).pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer intPipeCi = VkComputePipelineCreateInfo.calloc(1, stack);
            intPipeCi.get(0).sType$Default().stage(intStage).layout(integrationLayout);
            check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, intPipeCi, null, pPipe), "vkCreateComputePipelines(fog integration)");
            long integrationPipeline = pPipe.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, integrationPipeline, "fog integration pipeline");
            VK10.vkDestroyShaderModule(vk, intModule, null);

            VkDescriptorSetLayoutBinding.Buffer compBinds = VkDescriptorSetLayoutBinding.calloc(4, stack);
            compBinds.get(0).binding(0).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            compBinds.get(1).binding(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            compBinds.get(2).binding(2).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            compBinds.get(3).binding(3).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorSetLayoutCreateInfo compDslCi = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(compBinds);
            check(VK10.vkCreateDescriptorSetLayout(vk, compDslCi, null, p), "vkCreateDescriptorSetLayout(fog composite)");
            long compositeDsl = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, compositeDsl, "fog composite dsl");

            VkDescriptorPoolSize.Buffer compPoolSizes = VkDescriptorPoolSize.calloc(2, stack);
            compPoolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(3);
            compPoolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
            VkDescriptorPoolCreateInfo compPoolCi = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default()
                    .maxSets(1).pPoolSizes(compPoolSizes);
            check(VK10.vkCreateDescriptorPool(vk, compPoolCi, null, p), "vkCreateDescriptorPool(fog composite)");
            long compositePool = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, compositePool, "fog composite pool");

            VkDescriptorSetAllocateInfo compAlloc = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                    .descriptorPool(compositePool).pSetLayouts(stack.longs(compositeDsl));
            check(VK10.vkAllocateDescriptorSets(vk, compAlloc, pSet), "vkAllocateDescriptorSets(fog composite)");
            long compositeSet = pSet.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, compositeSet, "fog composite set");

            VkPushConstantRange.Buffer compPushRange = VkPushConstantRange.calloc(1, stack);
            compPushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(FogIntegrationPushData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo compPlCi = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                    .pSetLayouts(stack.longs(compositeDsl)).pPushConstantRanges(compPushRange);
            check(VK10.vkCreatePipelineLayout(vk, compPlCi, null, p), "vkCreatePipelineLayout(fog composite)");
            long compositeLayout = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, compositeLayout, "fog composite layout");

            long compModule = loadModule(vk, stack, "composite.comp.spv");
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, compModule, "fog composite module");
            VkPipelineShaderStageCreateInfo compStage = VkPipelineShaderStageCreateInfo.calloc(stack).sType$Default()
                    .stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(compModule).pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer compPipeCi = VkComputePipelineCreateInfo.calloc(1, stack);
            compPipeCi.get(0).sType$Default().stage(compStage).layout(compositeLayout);
            check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, compPipeCi, null, pPipe), "vkCreateComputePipelines(fog composite)");
            long compositePipeline = pPipe.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, compositePipeline, "fog composite pipeline");
            VK10.vkDestroyShaderModule(vk, compModule, null);

            VkSamplerCreateInfo samplerCi = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR).minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_LINEAR)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f).maxLod(16.0f);
            check(VK10.vkCreateSampler(vk, samplerCi, null, p), "vkCreateSampler(fog linear)");
            long linearSampler = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, linearSampler, "fog linear sampler");

            VkSamplerCreateInfo nearestSamplerCi = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_NEAREST).minFilter(VK10.VK_FILTER_NEAREST)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f).maxLod(0.0f);
            check(VK10.vkCreateSampler(vk, nearestSamplerCi, null, p), "vkCreateSampler(fog nearest)");
            long nearestSampler = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, nearestSampler, "fog nearest sampler");

            return new RtVolumetricFog(ctx,
                    injectionDsl, injectionPool, injectionSet, injectionLayout, injectionPipeline,
                    resolveDsl, resolvePool, resolveSet, resolveLayout, resolvePipeline,
                    integrationDsl, integrationPool, integrationSet, integrationLayout, integrationPipeline,
                    compositeDsl, compositePool, compositeSet, compositeLayout, compositePipeline,
                    linearSampler, nearestSampler);
        }
    }

    public RtImage froxelVolume() {
        return filteredFroxelVolumes.length == 2 ? currentResolvedVolume() : null;
    }

    private RtImage historyVolume() {
        return filteredFroxelVolumes[historyReadIndex];
    }

    private RtImage currentResolvedVolume() {
        return filteredFroxelVolumes[historyWriteIndex];
    }

    private static int effectiveSampleCount() {
        // The integrated-volume rewrite no longer ray-marches the display image sample-by-sample, but the
        // shared reflected push ABI still carries sampleCount and the generated Java serializer is not
        // checked into source. Keep writing a sane, quality-tier-aware value until the layout can be
        // regenerated alongside a later dedicated ABI cleanup.
        int qualityFloor = switch (CausticaConfig.Rt.VolumetricFog.QUALITY.value()) {
            case 0 -> 32;
            case 2 -> 96;
            case 3 -> 128;
            default -> 64;
        };
        return Math.min(Math.max(CausticaConfig.Rt.VolumetricFog.SAMPLES.value(), qualityFloor), 192);
    }

    public void ensureImages(int displayW, int displayH) {
        ensureImages(displayW, displayH, displayW, displayH);
    }

    public void ensureImages(int displayW, int displayH, int guideW, int guideH) {
        int[] dims = CausticaConfig.Rt.VolumetricFog.froxelDimensions();
        int fw = dims[0], fh = dims[1], fd = dims[2];

        boolean froxelResize = rawFroxelVolume == null
                || rawFroxelVolume.width != fw || rawFroxelVolume.height != fh || rawFroxelVolume.depth != fd
                || filteredFroxelVolumes.length != 2
                || filteredFroxelVolumes[0] == null
                || filteredFroxelVolumes[1] == null
                || filteredFroxelVolumes[0].width != fw || filteredFroxelVolumes[0].height != fh || filteredFroxelVolumes[0].depth != fd
                || filteredFroxelVolumes[1].width != fw || filteredFroxelVolumes[1].height != fh || filteredFroxelVolumes[1].depth != fd
                || integratedFroxelVolume == null
                || integratedFroxelVolume.width != fw || integratedFroxelVolume.height != fh || integratedFroxelVolume.depth != fd;

        // Fog quality can be changed live from the video settings, which resizes the froxel volume without a
        // window resize and therefore outside ensureOutput()'s existing waitIdle path. Recreating it while
        // the previous frame is still sampling/writing the old view can hand Vulkan a dead image handle.
        if (froxelResize) {
            ctx.waitIdle();
            if (rawFroxelVolume != null) rawFroxelVolume.destroy();
            for (RtImage filteredFroxelVolume : filteredFroxelVolumes) {
                if (filteredFroxelVolume != null) filteredFroxelVolume.destroy();
            }
            if (integratedFroxelVolume != null) {
                integratedFroxelVolume.destroy();
            }
            rawFroxelVolume = ctx.createStorageImage3D(fw, fh, fd, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "fog raw froxel volume");
            filteredFroxelVolumes = new RtImage[]{
                    ctx.createStorageImage3D(fw, fh, fd, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "fog resolved froxel volume A"),
                    ctx.createStorageImage3D(fw, fh, fd, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "fog resolved froxel volume B")
            };
            integratedFroxelVolume = ctx.createStorageImage3D(fw, fh, fd, VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                    "fog integrated froxel volume");
            historyReadIndex = 0;
            historyWriteIndex = 1;
            historyValid = false;
            boundInjectionFroxelView = 0L;
            boundResolveInputView = 0L;
            boundResolveHistoryView = 0L;
            boundResolveOutputView = 0L;
            boundIntegrationInputView = 0L;
            boundIntegrationOutputView = 0L;
            boundCompositeLocalView = 0L;
            boundCompositeIntegratedView = 0L;
            boundFogDepthView = 0L;
            boundSceneColorView = 0L;
            boundBlockAlbedoView = 0L;
        }
    }

    // Compatibility overload used by old code path
    public void ensureImages(int displayW, int displayH, CausticaConfig.Rt.VolumetricFog ignored) {
        ensureImages(displayW, displayH);
    }

    public void setInjectionImage() {
        if (rawFroxelVolume == null) return;
        if (boundInjectionFroxelView == rawFroxelVolume.view) return;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer imgInfo = VkDescriptorImageInfo.calloc(1, stack);
            imgInfo.get(0).imageView(rawFroxelVolume.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(1, stack);
            writes.get(0).sType$Default().dstSet(injectionSet).dstBinding(0)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(imgInfo);
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundInjectionFroxelView = rawFroxelVolume.view;
    }

    public void setInjectionTracingResources(long tlas, long blockAlbedoView, long blockAlbedoSampler) {
        if (rawFroxelVolume == null) return;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkWriteDescriptorSetAccelerationStructureKHR tlasInfo = VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack)
                    .sType(KHRAccelerationStructure.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR)
                    .pAccelerationStructures(stack.longs(tlas));
            VkDescriptorImageInfo.Buffer blockInfo = VkDescriptorImageInfo.calloc(1, stack);
            blockInfo.get(0).imageView(blockAlbedoView).sampler(blockAlbedoSampler)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(2, stack);
            writes.get(0).sType$Default().dstSet(injectionSet).dstBinding(1)
                    .descriptorCount(1).descriptorType(KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                    .pNext(tlasInfo.address());
            writes.get(1).sType$Default().dstSet(injectionSet).dstBinding(2)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .pImageInfo(blockInfo);
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundBlockAlbedoView = blockAlbedoView;
    }

    public void setResolveImages() {
        if (rawFroxelVolume == null) return;
        long historyView = historyValid ? historyVolume().view : rawFroxelVolume.view;
        long outputView = currentResolvedVolume().view;
        if (boundResolveInputView == rawFroxelVolume.view
                && boundResolveHistoryView == historyView
                && boundResolveOutputView == outputView) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(3, stack);

            VkDescriptorImageInfo.Buffer currentInfo = VkDescriptorImageInfo.calloc(1, stack);
            currentInfo.get(0).imageView(rawFroxelVolume.view).sampler(linearSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(0).sType$Default().dstSet(resolveSet).dstBinding(0)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(currentInfo);

            VkDescriptorImageInfo.Buffer historyInfo = VkDescriptorImageInfo.calloc(1, stack);
            historyInfo.get(0).imageView(historyView).sampler(linearSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(1).sType$Default().dstSet(resolveSet).dstBinding(1)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(historyInfo);

            VkDescriptorImageInfo.Buffer outputInfo = VkDescriptorImageInfo.calloc(1, stack);
            outputInfo.get(0).imageView(outputView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(2).sType$Default().dstSet(resolveSet).dstBinding(2)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(outputInfo);

            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundResolveInputView = rawFroxelVolume.view;
        boundResolveHistoryView = historyView;
        boundResolveOutputView = outputView;
    }

    public void setIntegrationImages() {
        if (rawFroxelVolume == null || integratedFroxelVolume == null) return;
        long inputView = currentResolvedVolume().view;
        long outputView = integratedFroxelVolume.view;
        if (boundIntegrationInputView == inputView && boundIntegrationOutputView == outputView) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(2, stack);

            VkDescriptorImageInfo.Buffer inputInfo = VkDescriptorImageInfo.calloc(1, stack);
            inputInfo.get(0).imageView(inputView).sampler(linearSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(0).sType$Default().dstSet(integrationSet).dstBinding(0)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(inputInfo);

            VkDescriptorImageInfo.Buffer outputInfo = VkDescriptorImageInfo.calloc(1, stack);
            outputInfo.get(0).imageView(outputView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(1).sType$Default().dstSet(integrationSet).dstBinding(1)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(outputInfo);

            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundIntegrationInputView = inputView;
        boundIntegrationOutputView = outputView;
    }

    public void setCompositeImages(long fogDepthView, long sceneColorView) {
        if (rawFroxelVolume == null || integratedFroxelVolume == null) return;
        long localView = currentResolvedVolume().view;
        long integratedView = integratedFroxelVolume.view;
        if (boundCompositeLocalView == localView
                && boundCompositeIntegratedView == integratedView
                && boundFogDepthView == fogDepthView
                && boundSceneColorView == sceneColorView) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(4, stack);

            VkDescriptorImageInfo.Buffer localInfo = VkDescriptorImageInfo.calloc(1, stack);
            localInfo.get(0).imageView(localView).sampler(linearSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(0).sType$Default().dstSet(compositeSet).dstBinding(0)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(localInfo);

            VkDescriptorImageInfo.Buffer integratedInfo = VkDescriptorImageInfo.calloc(1, stack);
            integratedInfo.get(0).imageView(integratedView).sampler(linearSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(1).sType$Default().dstSet(compositeSet).dstBinding(1)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(integratedInfo);

            VkDescriptorImageInfo.Buffer depthInfo = VkDescriptorImageInfo.calloc(1, stack);
            // Depth-aware XY reconstruction should classify silhouettes against actual guide-depth texels,
            // not a linearly blurred depth field. Using a nearest/clamp sampler here sharply reduces the
            // bright-side fog bleed that otherwise appears when display-res reconstruction samples across a
            // lower-resolution fog-stop edge.
            depthInfo.get(0).imageView(fogDepthView).sampler(nearestSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(2).sType$Default().dstSet(compositeSet).dstBinding(2)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(depthInfo);

            VkDescriptorImageInfo.Buffer outputInfo = VkDescriptorImageInfo.calloc(1, stack);
            outputInfo.get(0).imageView(sceneColorView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(3).sType$Default().dstSet(compositeSet).dstBinding(3)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(outputInfo);

            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundCompositeLocalView = localView;
        boundCompositeIntegratedView = integratedView;
        boundFogDepthView = fogDepthView;
        boundSceneColorView = sceneColorView;
    }

    private FogIntegrationPushData buildPush(long worldPushAddr, long tableAddr, long entityTableAddr, long materialTableAddr,
                                             int renderW, int renderH,
                                             int frameIndex, float exposure,
                                             float[] terrainOrigin, float[] camWorldPos,
                                             float[] jitterOffset,
                                             float[] sunDir, float[] sunIllum,
                                             float[] moonDir, float[] moonIllum,
                                             boolean temporalEnabled) {
        int[] dims = CausticaConfig.Rt.VolumetricFog.froxelDimensions();
        float nearPlane = 0.1f;
        float farPlane = CausticaConfig.Rt.VolumetricFog.MAX_DISTANCE.value();
        return new FogIntegrationPushData(
                worldPushAddr, tableAddr, entityTableAddr, materialTableAddr,
                new FogIntegrationPushData.Int2(renderW, renderH),
                new FogIntegrationPushData.Int2(dims[0], dims[1]), dims[2],
                nearPlane, farPlane,
                CausticaConfig.Rt.VolumetricFog.DENSITY.value(),
                CausticaConfig.Rt.VolumetricFog.ANISOTROPY.value(),
                CausticaConfig.Rt.VolumetricFog.SCATTERING.value(),
                CausticaConfig.Rt.VolumetricFog.EXTINCTION.value(),
                CausticaConfig.Rt.VolumetricFog.HEIGHT_FALLOFF.value(),
                CausticaConfig.Rt.VolumetricFog.SUN_INTENSITY.value(),
                CausticaConfig.Rt.VolumetricFog.MOON_INTENSITY.value(),
                CausticaConfig.Rt.VolumetricFog.JITTER_STRENGTH.value(),
                CausticaConfig.Rt.VolumetricFog.MAX_DISTANCE.value(),
                frameIndex,
                effectiveSampleCount(),
                temporalEnabled ? 1 : 0,
                CausticaConfig.Rt.VolumetricFog.COLOR_TRANSMISSION.value() ? 1 : 0,
                exposure,
                new FogIntegrationPushData.Float2(jitterOffset[0], jitterOffset[1]),
                new FogIntegrationPushData.Float2(1.0f / renderW, 1.0f / renderH),
                new FogIntegrationPushData.Float3(terrainOrigin[0], terrainOrigin[1], terrainOrigin[2]),
                new FogIntegrationPushData.Float3(camWorldPos[0], camWorldPos[1], camWorldPos[2]),
                new FogIntegrationPushData.Float3(sunDir[0], sunDir[1], sunDir[2]),
                new FogIntegrationPushData.Float3(sunIllum[0], sunIllum[1], sunIllum[2]),
                new FogIntegrationPushData.Float3(moonDir[0], moonDir[1], moonDir[2]),
                new FogIntegrationPushData.Float3(moonIllum[0], moonIllum[1], moonIllum[2]));
    }

    public void dispatchInjection(VkCommandBuffer cmd,
                                  long worldPushAddr, long tableAddr, long entityTableAddr, long materialTableAddr,
                                  int renderW, int renderH,
                                  int frameIndex, float exposure,
                                  float[] terrainOrigin, float[] camWorldPos,
                                  float[] jitterOffset,
                                  float[] sunDir, float[] sunIllum,
                                  float[] moonDir, float[] moonIllum) {
        if (rawFroxelVolume == null) return;
        int[] dims = CausticaConfig.Rt.VolumetricFog.froxelDimensions();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, injectionPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, injectionLayout, 0, stack.longs(injectionSet), null);
            ByteBuffer push = stack.malloc(FogIntegrationPushData.BYTE_SIZE);
            buildPush(worldPushAddr, tableAddr, entityTableAddr, materialTableAddr,
                    renderW, renderH, frameIndex, exposure, terrainOrigin, camWorldPos,
                    jitterOffset, sunDir, sunIllum, moonDir, moonIllum,
                    CausticaConfig.Rt.VolumetricFog.TEMPORAL.value()).write(push);
            VK10.vkCmdPushConstants(cmd, injectionLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (dims[0] + 7) / 8, (dims[1] + 7) / 8, (dims[2] + 3) / 4);
        }
    }

    // old overload compatibility
    public void dispatchInjection(VkCommandBuffer cmd, CausticaConfig.Rt.VolumetricFog cfg,
                                  long worldPushAddr, long tableAddr, long entityTableAddr, long materialTableAddr,
                                  int renderW, int renderH,
                                  int frameIndex, float exposure,
                                  float[] terrainOrigin, float[] camWorldPos,
                                  float[] jitterOffset,
                                  float[] sunDir, float[] sunIllum,
                                  float[] moonDir, float[] moonIllum) {
        dispatchInjection(cmd, worldPushAddr, tableAddr, entityTableAddr, materialTableAddr,
                renderW, renderH, frameIndex, exposure, terrainOrigin, camWorldPos, jitterOffset, sunDir, sunIllum, moonDir, moonIllum);
    }

    public void dispatchResolve(VkCommandBuffer cmd,
                                long worldPushAddr, long tableAddr, long entityTableAddr, long materialTableAddr,
                                int renderW, int renderH,
                                int frameIndex, float exposure,
                                float[] terrainOrigin, float[] camWorldPos,
                                float[] jitterOffset,
                                float[] sunDir, float[] sunIllum,
                                float[] moonDir, float[] moonIllum) {
        if (rawFroxelVolume == null) return;
        int[] dims = CausticaConfig.Rt.VolumetricFog.froxelDimensions();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, resolvePipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, resolveLayout, 0, stack.longs(resolveSet), null);
            ByteBuffer push = stack.malloc(FogIntegrationPushData.BYTE_SIZE);
            buildPush(worldPushAddr, tableAddr, entityTableAddr, materialTableAddr,
                    renderW, renderH, frameIndex, exposure, terrainOrigin, camWorldPos,
                    jitterOffset, sunDir, sunIllum, moonDir, moonIllum,
                    CausticaConfig.Rt.VolumetricFog.TEMPORAL.value() && historyValid).write(push);
            VK10.vkCmdPushConstants(cmd, resolveLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (dims[0] + 7) / 8, (dims[1] + 7) / 8, (dims[2] + 3) / 4);
        }
    }

    public void dispatchIntegration(VkCommandBuffer cmd,
                                    long worldPushAddr, long tableAddr, long entityTableAddr, long materialTableAddr,
                                    int renderW, int renderH,
                                    int frameIndex, float exposure,
                                    float[] terrainOrigin, float[] camWorldPos,
                                    float[] jitterOffset,
                                    float[] sunDir, float[] sunIllum,
                                    float[] moonDir, float[] moonIllum) {
        if (rawFroxelVolume == null || integratedFroxelVolume == null) return;
        int[] dims = CausticaConfig.Rt.VolumetricFog.froxelDimensions();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, integrationPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, integrationLayout, 0, stack.longs(integrationSet), null);
            ByteBuffer push = stack.malloc(FogIntegrationPushData.BYTE_SIZE);
            buildPush(worldPushAddr, tableAddr, entityTableAddr, materialTableAddr,
                    renderW, renderH, frameIndex, exposure, terrainOrigin, camWorldPos,
                    jitterOffset, sunDir, sunIllum, moonDir, moonIllum,
                    CausticaConfig.Rt.VolumetricFog.TEMPORAL.value()).write(push);
            VK10.vkCmdPushConstants(cmd, integrationLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (dims[0] + 7) / 8, (dims[1] + 7) / 8, 1);
        }
    }

    // compatibility overload
    public void dispatchIntegration(VkCommandBuffer cmd, CausticaConfig.Rt.VolumetricFog cfg,
                                    long worldPushAddr, long tableAddr, long entityTableAddr, long materialTableAddr,
                                    int renderW, int renderH,
                                    int frameIndex, float exposure,
                                    float[] terrainOrigin, float[] camWorldPos,
                                    float[] jitterOffset,
                                    float[] sunDir, float[] sunIllum,
                                    float[] moonDir, float[] moonIllum) {
        dispatchIntegration(cmd, worldPushAddr, tableAddr, entityTableAddr, materialTableAddr,
                renderW, renderH, frameIndex, exposure, terrainOrigin, camWorldPos, jitterOffset, sunDir, sunIllum, moonDir, moonIllum);
    }

    public void dispatchComposite(VkCommandBuffer cmd,
                                  long worldPushAddr, long tableAddr, long entityTableAddr, long materialTableAddr,
                                  int renderW, int renderH,
                                  int frameIndex, float exposure,
                                  float[] terrainOrigin, float[] camWorldPos,
                                  float[] jitterOffset,
                                  float[] sunDir, float[] sunIllum,
                                  float[] moonDir, float[] moonIllum) {
        if (rawFroxelVolume == null || integratedFroxelVolume == null) return;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, compositePipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, compositeLayout, 0, stack.longs(compositeSet), null);
            ByteBuffer push = stack.malloc(FogIntegrationPushData.BYTE_SIZE);
            buildPush(worldPushAddr, tableAddr, entityTableAddr, materialTableAddr,
                    renderW, renderH, frameIndex, exposure, terrainOrigin, camWorldPos,
                    jitterOffset, sunDir, sunIllum, moonDir, moonIllum,
                    CausticaConfig.Rt.VolumetricFog.TEMPORAL.value()).write(push);
            VK10.vkCmdPushConstants(cmd, compositeLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (renderW + 15) / 16, (renderH + 15) / 16, 1);
        }
    }

    public void invalidateHistory() {
        historyValid = false;
        boundResolveHistoryView = 0L;
        boundIntegrationInputView = 0L;
        boundCompositeLocalView = 0L;
    }

    public void advanceHistory() {
        if (filteredFroxelVolumes.length != 2) {
            return;
        }
        historyValid = true;
        historyReadIndex = historyWriteIndex;
        historyWriteIndex = 1 - historyWriteIndex;
        boundResolveHistoryView = 0L;
        boundResolveOutputView = 0L;
        boundIntegrationInputView = 0L;
        boundCompositeLocalView = 0L;
        boundCompositeIntegratedView = 0L;
    }

    public void destroy() {
        if (destroyed) return;
        VkDevice vk = ctx.vk();
        if (rawFroxelVolume != null) rawFroxelVolume.destroy();
        for (RtImage filteredFroxelVolume : filteredFroxelVolumes) {
            if (filteredFroxelVolume != null) filteredFroxelVolume.destroy();
        }
        if (integratedFroxelVolume != null) {
            integratedFroxelVolume.destroy();
        }
        VK10.vkDestroyPipeline(vk, injectionPipeline, null);
        VK10.vkDestroyPipelineLayout(vk, injectionLayout, null);
        VK10.vkDestroyDescriptorPool(vk, injectionPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, injectionDsl, null);
        VK10.vkDestroyPipeline(vk, resolvePipeline, null);
        VK10.vkDestroyPipelineLayout(vk, resolveLayout, null);
        VK10.vkDestroyDescriptorPool(vk, resolvePool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, resolveDsl, null);
        VK10.vkDestroyPipeline(vk, integrationPipeline, null);
        VK10.vkDestroyPipelineLayout(vk, integrationLayout, null);
        VK10.vkDestroyDescriptorPool(vk, integrationPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, integrationDsl, null);
        VK10.vkDestroyPipeline(vk, compositePipeline, null);
        VK10.vkDestroyPipelineLayout(vk, compositeLayout, null);
        VK10.vkDestroyDescriptorPool(vk, compositePool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, compositeDsl, null);
        VK10.vkDestroySampler(vk, linearSampler, null);
        VK10.vkDestroySampler(vk, nearestSampler, null);
        destroyed = true;
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String name) {
        byte[] bytes;
        try (InputStream in = RtVolumetricFog.class.getResourceAsStream(SHADER_DIR + name)) {
            if (in == null) throw new IllegalStateException("missing SPIR-V resource: " + SHADER_DIR + name);
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
