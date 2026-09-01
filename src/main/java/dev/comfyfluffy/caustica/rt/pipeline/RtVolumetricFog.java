package dev.comfyfluffy.caustica.rt.pipeline;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.FogInjectionPushData;
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

    private final long integrationDsl;
    private final long integrationPool;
    private final long integrationSet;
    private final long integrationLayout;
    private final long integrationPipeline;
    private final long linearSampler;
    private final long nearestSampler;

    private RtImage froxelVolume;

    private long boundFroxelView;
    private long boundFogDepthView;
    private long boundSceneColorView;
    private long boundBlockAlbedoView;
    private boolean destroyed;

    private RtVolumetricFog(RtContext ctx,
                            long injectionDsl, long injectionPool, long injectionSet, long injectionLayout, long injectionPipeline,
                            long integrationDsl, long integrationPool, long integrationSet, long integrationLayout, long integrationPipeline,
                            long linearSampler, long nearestSampler) {
        this.ctx = ctx;
        this.injectionDsl = injectionDsl;
        this.injectionPool = injectionPool;
        this.injectionSet = injectionSet;
        this.injectionLayout = injectionLayout;
        this.injectionPipeline = injectionPipeline;
        this.integrationDsl = integrationDsl;
        this.integrationPool = integrationPool;
        this.integrationSet = integrationSet;
        this.integrationLayout = integrationLayout;
        this.integrationPipeline = integrationPipeline;
        this.linearSampler = linearSampler;
        this.nearestSampler = nearestSampler;
    }

    public static RtVolumetricFog create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer p = stack.mallocLong(1);
            LongBuffer pSet = stack.mallocLong(1);
            LongBuffer pPipe = stack.mallocLong(1);

            VkDescriptorSetLayoutBinding.Buffer injBinds = VkDescriptorSetLayoutBinding.calloc(1, stack);
            injBinds.get(0).binding(0).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorSetLayoutCreateInfo injDslCi = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(injBinds);
            check(VK10.vkCreateDescriptorSetLayout(vk, injDslCi, null, p), "vkCreateDescriptorSetLayout(fog injection)");
            long injectionDsl = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, injectionDsl, "fog injection dsl");

            VkDescriptorPoolSize.Buffer injPoolSizes = VkDescriptorPoolSize.calloc(1, stack);
            injPoolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
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
            injPushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(FogInjectionPushData.BYTE_SIZE);
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

            VkDescriptorSetLayoutBinding.Buffer intBinds0 = VkDescriptorSetLayoutBinding.calloc(5, stack);
            intBinds0.get(0).binding(0).descriptorType(KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            intBinds0.get(1).binding(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            intBinds0.get(2).binding(2).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            intBinds0.get(3).binding(3).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            intBinds0.get(4).binding(4).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);

            VkDescriptorSetLayoutCreateInfo intDsl0Ci = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(intBinds0);
            check(VK10.vkCreateDescriptorSetLayout(vk, intDsl0Ci, null, p), "vkCreateDescriptorSetLayout(fog integration set0)");
            long dsl0 = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, dsl0, "fog integration dsl0");

            VkDescriptorPoolSize.Buffer intPoolSizes = VkDescriptorPoolSize.calloc(3, stack);
            intPoolSizes.get(0).type(KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(1);
            intPoolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(3);
            intPoolSizes.get(2).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
            VkDescriptorPoolCreateInfo intPoolCi = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default()
                    .maxSets(1).pPoolSizes(intPoolSizes);
            check(VK10.vkCreateDescriptorPool(vk, intPoolCi, null, p), "vkCreateDescriptorPool(fog integration)");
            long integrationPool = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, integrationPool, "fog integration pool");

            VkDescriptorSetAllocateInfo intAlloc = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                    .descriptorPool(integrationPool).pSetLayouts(stack.longs(dsl0));
            check(VK10.vkAllocateDescriptorSets(vk, intAlloc, pSet), "vkAllocateDescriptorSets(fog integration)");
            long integrationSet = pSet.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, integrationSet, "fog integration set0");

            VkPushConstantRange.Buffer intPushRange = VkPushConstantRange.calloc(1, stack);
            intPushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(FogIntegrationPushData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo intPlCi = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                    .pSetLayouts(stack.longs(dsl0)).pPushConstantRanges(intPushRange);
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

            samplerCi.magFilter(VK10.VK_FILTER_NEAREST).minFilter(VK10.VK_FILTER_NEAREST).mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST);
            check(VK10.vkCreateSampler(vk, samplerCi, null, p), "vkCreateSampler(fog nearest)");
            long nearestSampler = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, nearestSampler, "fog nearest sampler");

            return new RtVolumetricFog(ctx,
                    injectionDsl, injectionPool, injectionSet, injectionLayout, injectionPipeline,
                    dsl0, integrationPool, integrationSet, integrationLayout, integrationPipeline,
                    linearSampler, nearestSampler);
        }
    }

    public RtImage froxelVolume() { return froxelVolume; }

    private static int effectiveSampleCount() {
        int qualityCap = switch (CausticaConfig.Rt.VolumetricFog.QUALITY.value()) {
            case 0 -> 16;
            case 2 -> 32;
            case 3 -> 48;
            default -> 24;
        };
        return Math.min(CausticaConfig.Rt.VolumetricFog.SAMPLES.value(), qualityCap);
    }

    public void ensureImages(int displayW, int displayH) {
        ensureImages(displayW, displayH, displayW, displayH);
    }

    public void ensureImages(int displayW, int displayH, int guideW, int guideH) {
        int[] dims = CausticaConfig.Rt.VolumetricFog.froxelDimensions();
        int fw = dims[0], fh = dims[1], fd = dims[2];

        boolean froxelResize = froxelVolume == null
                || froxelVolume.width != fw || froxelVolume.height != fh || froxelVolume.depth != fd;

        // Fog quality can be changed live from the video settings, which resizes the froxel volume without a
        // window resize and therefore outside ensureOutput()'s existing waitIdle path. Recreating it while
        // the previous frame is still sampling/writing the old view can hand Vulkan a dead image handle.
        if (froxelResize) {
            ctx.waitIdle();
            if (froxelVolume != null) froxelVolume.destroy();
            froxelVolume = ctx.createStorageImage3D(fw, fh, fd, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel volume");
            boundFroxelView = 0L;
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
        if (froxelVolume == null) return;
        if (boundFroxelView == froxelVolume.view) return;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer imgInfo = VkDescriptorImageInfo.calloc(1, stack);
            imgInfo.get(0).imageView(froxelVolume.view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(1, stack);
            writes.get(0).sType$Default().dstSet(injectionSet).dstBinding(0)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(imgInfo);
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundFroxelView = froxelVolume.view;
    }

    public void setIntegrationImages(long tlas, long fogDepthView, long sceneColorView, long blockAlbedoView, long blockAlbedoSampler) {
        if (froxelVolume == null) return;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkWriteDescriptorSetAccelerationStructureKHR tlasInfo = VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack)
                    .sType(KHRAccelerationStructure.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR)
                    .pAccelerationStructures(stack.longs(tlas));
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(5, stack);

            writes.get(0).sType$Default().dstSet(integrationSet).dstBinding(0)
                    .descriptorCount(1).descriptorType(KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                    .pNext(tlasInfo.address());

            VkDescriptorImageInfo.Buffer froxelInfo = VkDescriptorImageInfo.calloc(1, stack);
            froxelInfo.get(0).imageView(froxelVolume.view).sampler(linearSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(1).sType$Default().dstSet(integrationSet).dstBinding(1)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(froxelInfo);

            VkDescriptorImageInfo.Buffer depthInfo = VkDescriptorImageInfo.calloc(1, stack);
            depthInfo.get(0).imageView(fogDepthView).sampler(nearestSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(2).sType$Default().dstSet(integrationSet).dstBinding(2)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(depthInfo);

            VkDescriptorImageInfo.Buffer sceneColorInfo = VkDescriptorImageInfo.calloc(1, stack);
            sceneColorInfo.get(0).imageView(sceneColorView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            writes.get(3).sType$Default().dstSet(integrationSet).dstBinding(3)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(sceneColorInfo);

            VkDescriptorImageInfo.Buffer blockInfo = VkDescriptorImageInfo.calloc(1, stack);
            blockInfo.get(0).imageView(blockAlbedoView).sampler(blockAlbedoSampler).imageLayout(VK10.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            writes.get(4).sType$Default().dstSet(integrationSet).dstBinding(4)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(blockInfo);

            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundFogDepthView = fogDepthView;
        boundSceneColorView = sceneColorView;
        boundBlockAlbedoView = blockAlbedoView;
    }

    public void dispatchInjection(VkCommandBuffer cmd, long worldPushAddr, int frameIndex, float[] terrainOrigin, float[] camWorldPos, float[] jitterOffset) {
        if (froxelVolume == null) return;
        int[] dims = CausticaConfig.Rt.VolumetricFog.froxelDimensions();
        float nearPlane = 0.1f;
        float farPlane = CausticaConfig.Rt.VolumetricFog.MAX_DISTANCE.value();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, injectionPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, injectionLayout, 0, stack.longs(injectionSet), null);
            ByteBuffer push = stack.malloc(FogInjectionPushData.BYTE_SIZE);
            FogInjectionPushData data = new FogInjectionPushData(
                    worldPushAddr,
                    new FogInjectionPushData.Int3(dims[0], dims[1], dims[2]),
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
                    (int) (Integer.toUnsignedLong(frameIndex) * 2654435761L),
                    CausticaConfig.Rt.VolumetricFog.TEMPORAL.value() ? 1 : 0,
                    CausticaConfig.Rt.VolumetricFog.COLOR_TRANSMISSION.value() ? 1 : 0,
                    new FogInjectionPushData.Float2(jitterOffset[0], jitterOffset[1]),
                    new FogInjectionPushData.Float3(terrainOrigin[0], terrainOrigin[1], terrainOrigin[2]),
                    new FogInjectionPushData.Float3(camWorldPos[0], camWorldPos[1], camWorldPos[2])
            );
            data.write(push);
            VK10.vkCmdPushConstants(cmd, injectionLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (dims[0] + 7) / 8, (dims[1] + 7) / 8, (dims[2] + 3) / 4);
        }
    }

    // old overload compatibility
    public void dispatchInjection(VkCommandBuffer cmd, CausticaConfig.Rt.VolumetricFog cfg, long worldPushAddr, int frameIndex, float[] terrainOrigin, float[] camWorldPos, float[] jitterOffset) {
        dispatchInjection(cmd, worldPushAddr, frameIndex, terrainOrigin, camWorldPos, jitterOffset);
    }

    public void dispatchIntegration(VkCommandBuffer cmd,
                                    long worldPushAddr, long tableAddr, long entityTableAddr, long materialTableAddr,
                                    int renderW, int renderH,
                                    int frameIndex, float exposure,
                                    float[] terrainOrigin, float[] camWorldPos,
                                    float[] jitterOffset,
                                    float[] sunDir, float[] sunIllum,
                                    float[] moonDir, float[] moonIllum) {
        if (froxelVolume == null) return;
        int[] dims = CausticaConfig.Rt.VolumetricFog.froxelDimensions();
        float nearPlane = 0.1f;
        float farPlane = CausticaConfig.Rt.VolumetricFog.MAX_DISTANCE.value();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, integrationPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, integrationLayout, 0, stack.longs(integrationSet), null);
            ByteBuffer push = stack.malloc(FogIntegrationPushData.BYTE_SIZE);
            FogIntegrationPushData data = new FogIntegrationPushData(
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
                    CausticaConfig.Rt.VolumetricFog.TEMPORAL.value() ? 1 : 0,
                    CausticaConfig.Rt.VolumetricFog.COLOR_TRANSMISSION.value() ? 1 : 0,
                    exposure,
                    new FogIntegrationPushData.Float2(jitterOffset[0], jitterOffset[1]),
                    new FogIntegrationPushData.Float2(1.0f / renderW, 1.0f / renderH),
                    new FogIntegrationPushData.Float3(terrainOrigin[0], terrainOrigin[1], terrainOrigin[2]),
                    new FogIntegrationPushData.Float3(camWorldPos[0], camWorldPos[1], camWorldPos[2]),
                    new FogIntegrationPushData.Float3(sunDir[0], sunDir[1], sunDir[2]),
                    new FogIntegrationPushData.Float3(sunIllum[0], sunIllum[1], sunIllum[2]),
                    new FogIntegrationPushData.Float3(moonDir[0], moonDir[1], moonDir[2]),
                    new FogIntegrationPushData.Float3(moonIllum[0], moonIllum[1], moonIllum[2])
            );
            data.write(push);
            VK10.vkCmdPushConstants(cmd, integrationLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (renderW + 15) / 16, (renderH + 15) / 16, 1);
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

    public void destroy() {
        if (destroyed) return;
        VkDevice vk = ctx.vk();
        if (froxelVolume != null) froxelVolume.destroy();
        VK10.vkDestroyPipeline(vk, injectionPipeline, null);
        VK10.vkDestroyPipelineLayout(vk, injectionLayout, null);
        VK10.vkDestroyDescriptorPool(vk, injectionPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, injectionDsl, null);
        VK10.vkDestroyPipeline(vk, integrationPipeline, null);
        VK10.vkDestroyPipelineLayout(vk, integrationLayout, null);
        VK10.vkDestroyDescriptorPool(vk, integrationPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, integrationDsl, null);
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
