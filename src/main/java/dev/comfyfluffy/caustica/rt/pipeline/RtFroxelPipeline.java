package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.accel.RtImage3D;
import dev.comfyfluffy.caustica.rt.gen.FroxelCompositePushData;
import dev.comfyfluffy.caustica.rt.gen.FroxelIntegratePushData;
import dev.comfyfluffy.caustica.rt.gen.FroxelLightPushData;
import dev.comfyfluffy.caustica.rt.gen.FroxelMediaPushData;
import dev.comfyfluffy.caustica.rt.gen.FroxelTemporalPushData;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData;
import org.joml.Matrix4f;
import org.joml.Matrix4fc;
import org.joml.Vector3f;
import org.joml.Vector4f;
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
 * End-to-end GPU compute pipeline for Froxel-Frustum Volumetric Lighting.
 * Manages 3D froxel grids, compute stages (media, light injection, 3D spatio-temporal filtering,
 * exact exponential volume integration, and depth-aware composite).
 */
public final class RtFroxelPipeline {
    private static final String SHADER_DIR = "/caustica/shaders/pipelines/volumetrics/";

    public static final int QUALITY_LOW = 0;
    public static final int QUALITY_MEDIUM = 1;
    public static final int QUALITY_HIGH = 2;
    public static final int QUALITY_ULTRA = 3;
    public static final int QUALITY_CINEMATIC = 4;

    private final RtContext ctx;

    // Pipelines
    private final long mediaPipeline;
    private final long mediaLayout;
    private final long mediaDsl;
    private final long mediaPool;
    private final long mediaSet;

    private final long lightPipeline;
    private final long lightLayout;
    private final long lightDsl;
    private final long lightPool;
    private final long lightSet;

    private final long temporalPipeline;
    private final long temporalLayout;
    private final long temporalDsl;
    private final long temporalPool;
    private final long temporalSet;

    private final long integratePipeline;
    private final long integrateLayout;
    private final long integrateDsl;
    private final long integratePool;
    private final long integrateSet;

    private final long compositePipeline;
    private final long compositeLayout;
    private final long compositeDsl;
    private final long compositePool;
    private final long compositeSet;

    private final long sampler3D;

    // Sized 3D Volume Resources
    private RtImage3D froxelMedia;
    private RtImage3D froxelLight;
    private RtImage3D froxelFiltered;
    private RtImage3D froxelHistory0;
    private RtImage3D froxelHistory1;
    private RtImage3D froxelVolume;

    private int gridNx = -1;
    private int gridNy = -1;
    private int gridNz = -1;
    private int currentQuality = -1;
    private int historyIndex = 0;
    private boolean resetHistory = true;
    private boolean destroyed;

    private RtFroxelPipeline(RtContext ctx,
                            long mediaPipeline, long mediaLayout, long mediaDsl, long mediaPool, long mediaSet,
                            long lightPipeline, long lightLayout, long lightDsl, long lightPool, long lightSet,
                            long temporalPipeline, long temporalLayout, long temporalDsl, long temporalPool, long temporalSet,
                            long integratePipeline, long integrateLayout, long integrateDsl, long integratePool, long integrateSet,
                            long compositePipeline, long compositeLayout, long compositeDsl, long compositePool, long compositeSet,
                            long sampler3D) {
        this.ctx = ctx;
        this.mediaPipeline = mediaPipeline;
        this.mediaLayout = mediaLayout;
        this.mediaDsl = mediaDsl;
        this.mediaPool = mediaPool;
        this.mediaSet = mediaSet;

        this.lightPipeline = lightPipeline;
        this.lightLayout = lightLayout;
        this.lightDsl = lightDsl;
        this.lightPool = lightPool;
        this.lightSet = lightSet;

        this.temporalPipeline = temporalPipeline;
        this.temporalLayout = temporalLayout;
        this.temporalDsl = temporalDsl;
        this.temporalPool = temporalPool;
        this.temporalSet = temporalSet;

        this.integratePipeline = integratePipeline;
        this.integrateLayout = integrateLayout;
        this.integrateDsl = integrateDsl;
        this.integratePool = integratePool;
        this.integrateSet = integrateSet;

        this.compositePipeline = compositePipeline;
        this.compositeLayout = compositeLayout;
        this.compositeDsl = compositeDsl;
        this.compositePool = compositePool;
        this.compositeSet = compositeSet;

        this.sampler3D = sampler3D;
    }

    public static RtFroxelPipeline create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer p = stack.mallocLong(1);

            // 1. Media Pass
            VkDescriptorSetLayoutBinding.Buffer mediaBinds = VkDescriptorSetLayoutBinding.calloc(FROXEL_MEDIA_BINDING_COUNT, stack);
            mediaBinds.get(FROXEL_MEDIA_OUTPUT).binding(FROXEL_MEDIA_OUTPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            long mediaDsl = createDsl(vk, stack, mediaBinds, p, "froxel media dsl");
            long mediaPool = createPool(vk, stack, 1, 1, 0, 0, p, "froxel media pool");
            long mediaSet = allocSet(vk, stack, mediaPool, mediaDsl, p, "froxel media set");
            long mediaLayout = createPipelineLayout(vk, stack, mediaDsl, FroxelMediaPushData.BYTE_SIZE, p, "froxel media layout");
            long mediaPipeline = createComputePipeline(vk, stack, mediaLayout, "media.comp.spv", p, "froxel media pipeline");

            // 2. Light Pass
            VkDescriptorSetLayoutBinding.Buffer lightBinds = VkDescriptorSetLayoutBinding.calloc(FROXEL_LIGHT_BINDING_COUNT, stack);
            lightBinds.get(FROXEL_LIGHT_TLAS).binding(FROXEL_LIGHT_TLAS).descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            lightBinds.get(FROXEL_LIGHT_MEDIA).binding(FROXEL_LIGHT_MEDIA).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            lightBinds.get(FROXEL_LIGHT_OUTPUT).binding(FROXEL_LIGHT_OUTPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            lightBinds.get(FROXEL_LIGHT_SKY_VIEW).binding(FROXEL_LIGHT_SKY_VIEW).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            lightBinds.get(FROXEL_LIGHT_TRANSMITTANCE).binding(FROXEL_LIGHT_TRANSMITTANCE).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            long lightDsl = createDsl(vk, stack, lightBinds, p, "froxel light dsl");
            long lightPool = createPool(vk, stack, 1, 2, 2, 1, p, "froxel light pool");
            long lightSet = allocSet(vk, stack, lightPool, lightDsl, p, "froxel light set");
            long lightLayout = createPipelineLayout(vk, stack, lightDsl, FroxelLightPushData.BYTE_SIZE, p, "froxel light layout");
            long lightPipeline = createComputePipeline(vk, stack, lightLayout, "light.comp.spv", p, "froxel light pipeline");

            // 3. Temporal Pass
            VkDescriptorSetLayoutBinding.Buffer tempBinds = VkDescriptorSetLayoutBinding.calloc(FROXEL_TEMPORAL_BINDING_COUNT, stack);
            tempBinds.get(FROXEL_TEMPORAL_LIGHT_INPUT).binding(FROXEL_TEMPORAL_LIGHT_INPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            tempBinds.get(FROXEL_TEMPORAL_HISTORY_INPUT).binding(FROXEL_TEMPORAL_HISTORY_INPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            tempBinds.get(FROXEL_TEMPORAL_OUTPUT).binding(FROXEL_TEMPORAL_OUTPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            tempBinds.get(FROXEL_TEMPORAL_HISTORY_OUTPUT).binding(FROXEL_TEMPORAL_HISTORY_OUTPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            long temporalDsl = createDsl(vk, stack, tempBinds, p, "froxel temporal dsl");
            long temporalPool = createPool(vk, stack, 1, 3, 1, 0, p, "froxel temporal pool");
            long temporalSet = allocSet(vk, stack, temporalPool, temporalDsl, p, "froxel temporal set");
            long temporalLayout = createPipelineLayout(vk, stack, temporalDsl, FroxelTemporalPushData.BYTE_SIZE, p, "froxel temporal layout");
            long temporalPipeline = createComputePipeline(vk, stack, temporalLayout, "temporal.comp.spv", p, "froxel temporal pipeline");

            // 4. Integrate Pass
            VkDescriptorSetLayoutBinding.Buffer intBinds = VkDescriptorSetLayoutBinding.calloc(FROXEL_INTEGRATE_BINDING_COUNT, stack);
            intBinds.get(FROXEL_INTEGRATE_LIGHT_INPUT).binding(FROXEL_INTEGRATE_LIGHT_INPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            intBinds.get(FROXEL_INTEGRATE_MEDIA_INPUT).binding(FROXEL_INTEGRATE_MEDIA_INPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            intBinds.get(FROXEL_INTEGRATE_OUTPUT).binding(FROXEL_INTEGRATE_OUTPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            long integrateDsl = createDsl(vk, stack, intBinds, p, "froxel integrate dsl");
            long integratePool = createPool(vk, stack, 1, 3, 0, 0, p, "froxel integrate pool");
            long integrateSet = allocSet(vk, stack, integratePool, integrateDsl, p, "froxel integrate set");
            long integrateLayout = createPipelineLayout(vk, stack, integrateDsl, FroxelIntegratePushData.BYTE_SIZE, p, "froxel integrate layout");
            long integratePipeline = createComputePipeline(vk, stack, integrateLayout, "integrate.comp.spv", p, "froxel integrate pipeline");

            // 5. Composite Pass
            VkDescriptorSetLayoutBinding.Buffer compBinds = VkDescriptorSetLayoutBinding.calloc(FROXEL_COMPOSITE_BINDING_COUNT, stack);
            compBinds.get(FROXEL_COMPOSITE_SCENE).binding(FROXEL_COMPOSITE_SCENE).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            compBinds.get(FROXEL_COMPOSITE_VOLUME).binding(FROXEL_COMPOSITE_VOLUME).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            compBinds.get(FROXEL_COMPOSITE_DEPTH).binding(FROXEL_COMPOSITE_DEPTH).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            long compositeDsl = createDsl(vk, stack, compBinds, p, "froxel composite dsl");
            long compositePool = createPool(vk, stack, 1, 2, 1, 0, p, "froxel composite pool");
            long compositeSet = allocSet(vk, stack, compositePool, compositeDsl, p, "froxel composite set");
            long compositeLayout = createPipelineLayout(vk, stack, compositeDsl, FroxelCompositePushData.BYTE_SIZE, p, "froxel composite layout");
            long compositePipeline = createComputePipeline(vk, stack, compositeLayout, "composite.comp.spv", p, "froxel composite pipeline");

            // 3D Linear Clamp Sampler
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR)
                    .minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
            check(VK10.vkCreateSampler(vk, samplerInfo, null, p), "vkCreateSampler(froxel 3D)");
            long sampler3D = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler3D, "froxel 3D linear sampler");

            return new RtFroxelPipeline(ctx,
                    mediaPipeline, mediaLayout, mediaDsl, mediaPool, mediaSet,
                    lightPipeline, lightLayout, lightDsl, lightPool, lightSet,
                    temporalPipeline, temporalLayout, temporalDsl, temporalPool, temporalSet,
                    integratePipeline, integrateLayout, integrateDsl, integratePool, integrateSet,
                    compositePipeline, compositeLayout, compositeDsl, compositePool, compositeSet,
                    sampler3D);
        }
    }

    public void requestHistoryReset() {
        this.resetHistory = true;
    }

    public long volumeView() {
        return froxelVolume != null ? froxelVolume.view : 0L;
    }

    public long sampler3D() {
        return sampler3D;
    }

    public void ensureResources(RtContext ctx, int displayW, int displayH, int quality) {
        int[] dims = calculateGridDims(displayW, displayH, quality);
        int nx = dims[0], ny = dims[1], nz = dims[2];

        if (froxelMedia != null && gridNx == nx && gridNy == ny && gridNz == nz && currentQuality == quality) {
            return;
        }

        ctx.waitIdle();
        destroyImages();

        gridNx = nx;
        gridNy = ny;
        gridNz = nz;
        currentQuality = quality;

        froxelMedia = ctx.createStorageImage3D(nx, ny, nz, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel media " + nx + "x" + ny + "x" + nz);
        froxelLight = ctx.createStorageImage3D(nx, ny, nz, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel light " + nx + "x" + ny + "x" + nz);
        froxelFiltered = ctx.createStorageImage3D(nx, ny, nz, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel filtered " + nx + "x" + ny + "x" + nz);
        froxelHistory0 = ctx.createStorageImage3D(nx, ny, nz, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel history 0 " + nx + "x" + ny + "x" + nz);
        froxelHistory1 = ctx.createStorageImage3D(nx, ny, nz, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel history 1 " + nx + "x" + ny + "x" + nz);
        froxelVolume = ctx.createStorageImage3D(nx, ny, nz, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "froxel integrated volume " + nx + "x" + ny + "x" + nz);

        resetHistory = true;

        // Bind Media Pass
        writeStorageImage(mediaSet, FROXEL_MEDIA_OUTPUT, froxelMedia.view);

        // Bind Integrate Pass
        writeStorageImage(integrateSet, FROXEL_INTEGRATE_LIGHT_INPUT, froxelFiltered.view);
        writeStorageImage(integrateSet, FROXEL_INTEGRATE_MEDIA_INPUT, froxelMedia.view);
        writeStorageImage(integrateSet, FROXEL_INTEGRATE_OUTPUT, froxelVolume.view);
    }

    public void recordVolumetrics(VkCommandBuffer cmd, MemoryStack stack,
                                  long tlas, long skyView, long transmittance, long skySampler,
                                  RtImage sceneColor, RtImage guideDepth,
                                  Matrix4fc curProjView, Matrix4fc prevProjView, Matrix4fc invViewProj,
                                  double camX, double camY, double camZ,
                                  float camDeltaX, float camDeltaY, float camDeltaZ,
                                  int terrainBlockX, int terrainBlockY, int terrainBlockZ,
                                  float lightRebaseX, float lightRebaseY, float lightRebaseZ,
                                  long lightBufAddr, long lightGridCellAddr, long lightGridSpanAddr,
                                  long lightAliasAddr, long lightLocalAliasAddr, float invGlobalPowerSum,
                                  float lightGridOriginX, float lightGridOriginY, float lightGridOriginZ, float lightGridCellSize,
                                  int lightGridDimX, int lightGridDimY, int lightGridDimZ,
                                  int lightCount,
                                  float sunDirX, float sunDirY, float sunDirZ, float sunIlluminanceLux,
                                  float sunAngularRadiusRad, float preExposure,
                                  boolean isSubmerged, float waterR, float waterG, float waterB, float waterWaveTime,
                                  long frameIndex) {
        if (froxelMedia == null || !CausticaConfig.Rt.Volumetrics.ENABLED.value()) {
            return;
        }

        float zNear = 0.75f;
        float zFar = CausticaConfig.Rt.Volumetrics.MAX_DISTANCE.value();
        float baseDensity = CausticaConfig.Rt.Volumetrics.BASE_DENSITY.value();
        float heightFalloff = CausticaConfig.Rt.Volumetrics.HEIGHT_FALLOFF.value();
        float groundAltitude = CausticaConfig.Rt.Volumetrics.GROUND_ALTITUDE.value();
        float g1 = CausticaConfig.Rt.Volumetrics.ANISOTROPY_FORWARD.value();
        float g2 = CausticaConfig.Rt.Volumetrics.ANISOTROPY_BACKWARD.value();
        float lobeWeight = CausticaConfig.Rt.Volumetrics.LOBE_WEIGHT.value();

        int groupX = (gridNx + 7) / 8;
        int groupY = (gridNy + 7) / 8;
        int groupZ = (gridNz + 3) / 4;

        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "froxel volumetrics")) {

            // 1. Media Pass
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, mediaPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, mediaLayout, 0, stack.longs(mediaSet), null);
            ByteBuffer mediaPush = stack.malloc(FroxelMediaPushData.BYTE_SIZE);
            new FroxelMediaPushData(
                    new FroxelMediaPushData.Int4(gridNx, gridNy, gridNz, isSubmerged ? 1 : 0),
                    new FroxelMediaPushData.Float4(baseDensity, heightFalloff, groundAltitude, zFar),
                    new FroxelMediaPushData.Float4(zNear, zFar, 0.9f, 0.95f),
                    new FroxelMediaPushData.Float4(1.0f, 0.1f, 0.015f, 0.0f),
                    new FroxelMediaPushData.Float4(waterR, waterG, waterB, waterWaveTime),
                    invViewProj,
                    new FroxelMediaPushData.Float3((float) camX, (float) camY, (float) camZ),
                    0.0f
            ).write(mediaPush);
            VK10.vkCmdPushConstants(cmd, mediaLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, mediaPush);
            VK10.vkCmdDispatch(cmd, groupX, groupY, groupZ);

            VulkanCommandEncoder.memoryBarrier(cmd, stack);

            // 2. Light Injection Pass
            bindLightImages(tlas, froxelMedia.view, froxelLight.view, skyView, transmittance, skySampler);
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, lightPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, lightLayout, 0, stack.longs(lightSet), null);

            float sunShaftMultiplier = CausticaConfig.Rt.Volumetrics.SUN_SHAFT_MULTIPLIER.value();
            ByteBuffer lightPush = stack.malloc(FroxelLightPushData.BYTE_SIZE);
            new FroxelLightPushData(
                    new FroxelLightPushData.Int4(gridNx, gridNy, gridNz, (int) frameIndex),
                    new FroxelLightPushData.Float4(zNear, zFar, zFar, preExposure),
                    new FroxelLightPushData.Float4(g1, g2, lobeWeight, sunShaftMultiplier),
                    new FroxelLightPushData.Float4(sunDirX, sunDirY, sunDirZ, sunIlluminanceLux),
                    new FroxelLightPushData.Float4(lightGridOriginX, lightGridOriginY, lightGridOriginZ, lightGridCellSize),
                    new FroxelLightPushData.Int4(lightGridDimX, lightGridDimY, lightGridDimZ, CausticaConfig.Rt.Volumetrics.BLOCK_LIGHTS.value() ? 1 : 0),
                    new FroxelLightPushData.Float4(lightRebaseX, lightRebaseY, lightRebaseZ, invGlobalPowerSum),
                    invViewProj,
                    new FroxelLightPushData.Float3((float) (camX - terrainBlockX), (float) (camY - terrainBlockY), (float) (camZ - terrainBlockZ)),
                    lightCount,
                    new FroxelLightPushData.Float3((float) camX, (float) camY, (float) camZ),
                    sunAngularRadiusRad,
                    lightBufAddr,
                    lightGridCellAddr,
                    lightGridSpanAddr,
                    lightAliasAddr,
                    lightLocalAliasAddr,
                    CausticaConfig.Rt.Volumetrics.SUN_SHAFTS.value() ? 1 : 0,
                    8 // RIS candidates (M=8)
            ).write(lightPush);
            VK10.vkCmdPushConstants(cmd, lightLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, lightPush);
            VK10.vkCmdDispatch(cmd, groupX, groupY, groupZ);

            VulkanCommandEncoder.memoryBarrier(cmd, stack);

            // 3. Spatio-Temporal Filter Pass
            RtImage3D historyIn = historyIndex == 0 ? froxelHistory0 : froxelHistory1;
            RtImage3D historyOut = historyIndex == 0 ? froxelHistory1 : froxelHistory0;
            historyIndex = 1 - historyIndex;

            bindTemporalImages(froxelLight.view, historyIn.view, froxelFiltered.view, historyOut.view);
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, temporalPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, temporalLayout, 0, stack.longs(temporalSet), null);

            ByteBuffer tempPush = stack.malloc(FroxelTemporalPushData.BYTE_SIZE);
            new FroxelTemporalPushData(
                    new FroxelTemporalPushData.Int4(gridNx, gridNy, gridNz, (resetHistory || !CausticaConfig.Rt.Volumetrics.TEMPORAL_FILTER.value()) ? 1 : 0),
                    new FroxelTemporalPushData.Float4(CausticaConfig.Rt.Volumetrics.TEMPORAL_FILTER.value() ? 0.90f : 0.0f, 1.5f, zNear, zFar),
                    new FroxelTemporalPushData.Float4(zFar, 0.0f, 0.0f, 0.0f),
                    invViewProj,
                    prevProjView,
                    new FroxelTemporalPushData.Float3(camDeltaX, camDeltaY, camDeltaZ),
                    0.0f
            ).write(tempPush);
            VK10.vkCmdPushConstants(cmd, temporalLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, tempPush);
            VK10.vkCmdDispatch(cmd, groupX, groupY, groupZ);
            resetHistory = false;

            VulkanCommandEncoder.memoryBarrier(cmd, stack);

            // 4. Exact Exponential Integration Pass
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, integratePipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, integrateLayout, 0, stack.longs(integrateSet), null);
            ByteBuffer intPush = stack.malloc(FroxelIntegratePushData.BYTE_SIZE);
            new FroxelIntegratePushData(
                    new FroxelIntegratePushData.Int4(gridNx, gridNy, gridNz, 0),
                    new FroxelIntegratePushData.Float4(zNear, zFar, zFar, 1.0f)
            ).write(intPush);
            VK10.vkCmdPushConstants(cmd, integrateLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, intPush);
            VK10.vkCmdDispatch(cmd, (gridNx + 7) / 8, (gridNy + 7) / 8, 1);

            VulkanCommandEncoder.memoryBarrier(cmd, stack);

            // 5. Depth-Aware Composite Pass
            bindCompositeImages(sceneColor.view, froxelVolume.view, guideDepth.view);
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, compositePipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, compositeLayout, 0, stack.longs(compositeSet), null);
            ByteBuffer compPush = stack.malloc(FroxelCompositePushData.BYTE_SIZE);
            new FroxelCompositePushData(
                    new FroxelCompositePushData.Int4(sceneColor.width, sceneColor.height, guideDepth.width, guideDepth.height),
                    new FroxelCompositePushData.Int4(gridNx, gridNy, gridNz, 0),
                    new FroxelCompositePushData.Float4(zNear, zFar, zFar, 0.0f),
                    new FroxelCompositePushData.Float4(preExposure, 1.0f, 1.0f, 1.0f),
                    invViewProj,
                    new FroxelCompositePushData.Float3((float) (camX - terrainBlockX), (float) (camY - terrainBlockY), (float) (camZ - terrainBlockZ)),
                    0.0f
            ).write(compPush);
            VK10.vkCmdPushConstants(cmd, compositeLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, compPush);
            VK10.vkCmdDispatch(cmd, (sceneColor.width + 15) / 16, (sceneColor.height + 15) / 16, 1);
        }
    }

    private void bindLightImages(long tlas, long mediaView, long lightView, long skyView, long transmittance, long skySampler) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkWriteDescriptorSetAccelerationStructureKHR asWrite = VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR).pAccelerationStructures(stack.longs(tlas));

            VkDescriptorImageInfo.Buffer imgInfo = VkDescriptorImageInfo.calloc(4, stack);
            imgInfo.get(0).imageView(mediaView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            imgInfo.get(1).imageView(lightView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            imgInfo.get(2).sampler(skySampler).imageView(skyView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            imgInfo.get(3).sampler(skySampler).imageView(transmittance).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(5, stack);
            writes.get(0).sType$Default().pNext(asWrite.address()).dstSet(lightSet).dstBinding(FROXEL_LIGHT_TLAS)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
            writes.get(1).sType$Default().dstSet(lightSet).dstBinding(FROXEL_LIGHT_MEDIA)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(VkDescriptorImageInfo.create(imgInfo.address(0), 1));
            writes.get(2).sType$Default().dstSet(lightSet).dstBinding(FROXEL_LIGHT_OUTPUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(VkDescriptorImageInfo.create(imgInfo.address(1), 1));
            writes.get(3).sType$Default().dstSet(lightSet).dstBinding(FROXEL_LIGHT_SKY_VIEW)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .pImageInfo(VkDescriptorImageInfo.create(imgInfo.address(2), 1));
            writes.get(4).sType$Default().dstSet(lightSet).dstBinding(FROXEL_LIGHT_TRANSMITTANCE)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .pImageInfo(VkDescriptorImageInfo.create(imgInfo.address(3), 1));

            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
    }

    private void bindTemporalImages(long lightInView, long historyInView, long filteredOutView, long historyOutView) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer imgInfo = VkDescriptorImageInfo.calloc(4, stack);
            imgInfo.get(0).imageView(lightInView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            imgInfo.get(1).sampler(sampler3D).imageView(historyInView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            imgInfo.get(2).imageView(filteredOutView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            imgInfo.get(3).imageView(historyOutView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(4, stack);
            writes.get(0).sType$Default().dstSet(temporalSet).dstBinding(FROXEL_TEMPORAL_LIGHT_INPUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(VkDescriptorImageInfo.create(imgInfo.address(0), 1));
            writes.get(1).sType$Default().dstSet(temporalSet).dstBinding(FROXEL_TEMPORAL_HISTORY_INPUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .pImageInfo(VkDescriptorImageInfo.create(imgInfo.address(1), 1));
            writes.get(2).sType$Default().dstSet(temporalSet).dstBinding(FROXEL_TEMPORAL_OUTPUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(VkDescriptorImageInfo.create(imgInfo.address(2), 1));
            writes.get(3).sType$Default().dstSet(temporalSet).dstBinding(FROXEL_TEMPORAL_HISTORY_OUTPUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(VkDescriptorImageInfo.create(imgInfo.address(3), 1));

            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
    }

    private void bindCompositeImages(long sceneView, long volumeView, long depthView) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer imgInfo = VkDescriptorImageInfo.calloc(3, stack);
            imgInfo.get(0).imageView(sceneView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            imgInfo.get(1).sampler(sampler3D).imageView(volumeView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            imgInfo.get(2).imageView(depthView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(3, stack);
            writes.get(0).sType$Default().dstSet(compositeSet).dstBinding(FROXEL_COMPOSITE_SCENE)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(VkDescriptorImageInfo.create(imgInfo.address(0), 1));
            writes.get(1).sType$Default().dstSet(compositeSet).dstBinding(FROXEL_COMPOSITE_VOLUME)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .pImageInfo(VkDescriptorImageInfo.create(imgInfo.address(1), 1));
            writes.get(2).sType$Default().dstSet(compositeSet).dstBinding(FROXEL_COMPOSITE_DEPTH)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .pImageInfo(VkDescriptorImageInfo.create(imgInfo.address(2), 1));

            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
    }

    private void writeStorageImage(long set, int binding, long imageView) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer imgInfo = VkDescriptorImageInfo.calloc(1, stack);
            imgInfo.get(0).imageView(imageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(1, stack);
            write.get(0).sType$Default().dstSet(set).dstBinding(binding)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(imgInfo);
            VK10.vkUpdateDescriptorSets(ctx.vk(), write, null);
        }
    }

    private static int[] calculateGridDims(int displayW, int displayH, int quality) {
        float aspect = (float) displayW / Math.max(1, displayH);
        return switch (quality) {
            case QUALITY_LOW -> new int[] { Math.round(45 * aspect), 45, 32 };
            case QUALITY_MEDIUM -> new int[] { Math.round(68 * aspect), 68, 64 };
            case QUALITY_ULTRA -> new int[] { Math.round(112 * aspect), 112, 128 };
            case QUALITY_CINEMATIC -> new int[] { Math.round(135 * aspect), 135, 160 };
            default -> new int[] { Math.round(90 * aspect), 90, 96 }; // QUALITY_HIGH
        };
    }

    private void destroyImages() {
        if (froxelMedia != null) { froxelMedia.destroy(); froxelMedia = null; }
        if (froxelLight != null) { froxelLight.destroy(); froxelLight = null; }
        if (froxelFiltered != null) { froxelFiltered.destroy(); froxelFiltered = null; }
        if (froxelHistory0 != null) { froxelHistory0.destroy(); froxelHistory0 = null; }
        if (froxelHistory1 != null) { froxelHistory1.destroy(); froxelHistory1 = null; }
        if (froxelVolume != null) { froxelVolume.destroy(); froxelVolume = null; }
    }

    public void destroy() {
        if (destroyed) return;
        VkDevice vk = ctx.vk();

        destroyImages();

        VK10.vkDestroySampler(vk, sampler3D, null);

        VK10.vkDestroyPipeline(vk, mediaPipeline, null);
        VK10.vkDestroyPipelineLayout(vk, mediaLayout, null);
        VK10.vkDestroyDescriptorPool(vk, mediaPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, mediaDsl, null);

        VK10.vkDestroyPipeline(vk, lightPipeline, null);
        VK10.vkDestroyPipelineLayout(vk, lightLayout, null);
        VK10.vkDestroyDescriptorPool(vk, lightPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, lightDsl, null);

        VK10.vkDestroyPipeline(vk, temporalPipeline, null);
        VK10.vkDestroyPipelineLayout(vk, temporalLayout, null);
        VK10.vkDestroyDescriptorPool(vk, temporalPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, temporalDsl, null);

        VK10.vkDestroyPipeline(vk, integratePipeline, null);
        VK10.vkDestroyPipelineLayout(vk, integrateLayout, null);
        VK10.vkDestroyDescriptorPool(vk, integratePool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, integrateDsl, null);

        VK10.vkDestroyPipeline(vk, compositePipeline, null);
        VK10.vkDestroyPipelineLayout(vk, compositeLayout, null);
        VK10.vkDestroyDescriptorPool(vk, compositePool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, compositeDsl, null);

        destroyed = true;
    }

    private static long createDsl(VkDevice vk, MemoryStack stack, VkDescriptorSetLayoutBinding.Buffer binds, LongBuffer p, String label) {
        VkDescriptorSetLayoutCreateInfo dslci = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(binds);
        check(VK10.vkCreateDescriptorSetLayout(vk, dslci, null, p), "vkCreateDescriptorSetLayout(" + label + ")");
        return p.get(0);
    }

    private static long createPool(VkDevice vk, MemoryStack stack, int storageImg, int combinedSampler, int asCount, int dummy, LongBuffer p, String label) {
        int types = 0;
        if (storageImg > 0) types++;
        if (combinedSampler > 0) types++;
        if (asCount > 0) types++;

        VkDescriptorPoolSize.Buffer sizes = VkDescriptorPoolSize.calloc(types, stack);
        int idx = 0;
        if (storageImg > 0) {
            sizes.get(idx++).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(storageImg);
        }
        if (combinedSampler > 0) {
            sizes.get(idx++).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(combinedSampler);
        }
        if (asCount > 0) {
            sizes.get(idx++).type(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(asCount);
        }
        VkDescriptorPoolCreateInfo dpci = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default().maxSets(1).pPoolSizes(sizes);
        check(VK10.vkCreateDescriptorPool(vk, dpci, null, p), "vkCreateDescriptorPool(" + label + ")");
        return p.get(0);
    }

    private static long allocSet(VkDevice vk, MemoryStack stack, long pool, long dsl, LongBuffer p, String label) {
        VkDescriptorSetAllocateInfo dsai = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                .descriptorPool(pool).pSetLayouts(stack.longs(dsl));
        check(VK10.vkAllocateDescriptorSets(vk, dsai, p), "vkAllocateDescriptorSets(" + label + ")");
        return p.get(0);
    }

    private static long createPipelineLayout(VkDevice vk, MemoryStack stack, long dsl, int pushSize, LongBuffer p, String label) {
        VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
        pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(pushSize);
        VkPipelineLayoutCreateInfo plci = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                .pSetLayouts(stack.longs(dsl)).pPushConstantRanges(pushRange);
        check(VK10.vkCreatePipelineLayout(vk, plci, null, p), "vkCreatePipelineLayout(" + label + ")");
        return p.get(0);
    }

    private static long createComputePipeline(VkDevice vk, MemoryStack stack, long layout, String spvName, LongBuffer p, String label) {
        long module = loadModule(vk, stack, spvName);
        VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack).sType$Default()
                .stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(module).pName(stack.UTF8("main"));
        VkComputePipelineCreateInfo.Buffer cpci = VkComputePipelineCreateInfo.calloc(1, stack);
        cpci.get(0).sType$Default().stage(stage).layout(layout);
        check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, cpci, null, p), "vkCreateComputePipelines(" + label + ")");
        VK10.vkDestroyShaderModule(vk, module, null);
        return p.get(0);
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String name) {
        byte[] bytes;
        try (InputStream in = RtFroxelPipeline.class.getResourceAsStream(SHADER_DIR + name)) {
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
