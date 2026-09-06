package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.gen.FogGridPushData;
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
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.comfyfluffy.caustica.rt.RtContext.check;

/**
 * The per-frame light-space fog transmittance volume (see {@code shaders/pipelines/fog_grid/build.comp.slang}).
 * One descriptor-free compute bake per frame marches sun columns through the terrain mesher's occupancy
 * tiles (one 16^3 uint tile per section, addressed through RtTerrain's CPU-maintained 64^3 section grid)
 * and writes a four-level 128^3 RGBA8 transmittance cache; the fog march in {@code world/fog.slang} then
 * reads it with a handful of constant-time fetches instead of firing a TLAS visibility ray per sample.
 *
 * <p>The volume is single-buffered: the bake and the trace that reads it run in the same submission with a
 * barrier between, and the composite's end-of-frame barrier orders the next frame's overwrite against this
 * frame's trace — the same argument that lets every other GPU-written per-frame resource here be
 * single-buffered.
 *
 * <p>Level layout is an ABI with {@code fog.slang}: level {@code k} has cell size {@code 2^k} blocks, extent
 * {@code 64*2^k} blocks around the camera, and cell {@code (u, v, z)} packs RGB transmittance from that cell
 * to the level's {@code +z} (light-direction) face.
 */
public final class RtFogGrid {
    private static final String SHADER_DIR = "/caustica/shaders/pipelines/fog_grid/";
    // Keep in lock-step with the same-named constants in fog.slang / build.comp.slang.
    public static final int VOLUME_CELLS = 128;
    public static final int VOLUME_LEVELS = 4;
    private static final int VOLUME_BYTES = VOLUME_LEVELS * VOLUME_CELLS * VOLUME_CELLS * VOLUME_CELLS * Integer.BYTES;
    private static final int GROUP_SIZE = 8;

    private final RtContext ctx;
    private final RtBuffer volume;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long descriptorSet;
    private final long pipelineLayout;
    private final long pipeline;
    private boolean destroyed;

    private RtFogGrid(RtContext ctx, RtBuffer volume, long descriptorSetLayout, long descriptorPool,
                      long descriptorSet, long pipelineLayout, long pipeline) {
        this.ctx = ctx;
        this.volume = volume;
        this.descriptorSetLayout = descriptorSetLayout;
        this.descriptorPool = descriptorPool;
        this.descriptorSet = descriptorSet;
        this.pipelineLayout = pipelineLayout;
        this.pipeline = pipeline;
    }

    public static RtFogGrid create(RtContext ctx, long transmittanceView, long sampler) {
        VkDevice vk = ctx.vk();
        RtBuffer volume = ctx.createBuffer(VOLUME_BYTES, VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false,
                "fog transmittance volume");
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(1, stack);
            bindings.get(0).binding(0).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorSetLayoutCreateInfo layoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(bindings);
            LongBuffer handle = stack.mallocLong(1);
            check(VK10.vkCreateDescriptorSetLayout(vk, layoutInfo, null, handle),
                    "vkCreateDescriptorSetLayout(fog grid)");
            long descriptorSetLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                    descriptorSetLayout, "fog grid descriptor set layout");

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(1, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1);
            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(1).pPoolSizes(poolSizes);
            check(VK10.vkCreateDescriptorPool(vk, poolInfo, null, handle),
                    "vkCreateDescriptorPool(fog grid)");
            long descriptorPool = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                    descriptorPool, "fog grid descriptor pool");

            VkDescriptorSetAllocateInfo allocateInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(descriptorPool)
                    .pSetLayouts(stack.longs(descriptorSetLayout));
            LongBuffer setHandle = stack.mallocLong(1);
            check(VK10.vkAllocateDescriptorSets(vk, allocateInfo, setHandle),
                    "vkAllocateDescriptorSets(fog grid)");
            long descriptorSet = setHandle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET,
                    descriptorSet, "fog grid descriptor set");

            // The dominant-light choice must match world.rgen exactly, so the bake samples the SAME
            // transmittance LUT and sampler the raygen uses (shared from RtSkyLut).
            VkDescriptorImageInfo imageInfo = VkDescriptorImageInfo.calloc(stack).imageView(transmittanceView)
                    .sampler(sampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkWriteDescriptorSet write = VkWriteDescriptorSet.calloc(1, stack);
            write.get(0).sType$Default().dstSet(descriptorSet).dstBinding(0).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .pImageInfo(VkDescriptorImageInfo.create(imageInfo.address(), 1));
            VK10.vkUpdateDescriptorSets(vk, write, null);

            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
            pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .offset(0).size(FogGridPushData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo pipelineLayoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default().pSetLayouts(stack.longs(descriptorSetLayout))
                    .pPushConstantRanges(pushRange);
            check(VK10.vkCreatePipelineLayout(vk, pipelineLayoutInfo, null, handle),
                    "vkCreatePipelineLayout(fog grid)");
            long pipelineLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                    pipelineLayout, "fog grid pipeline layout");

            long pipeline = createComputePipeline(ctx, stack, pipelineLayout,
                    "build.comp.spv", "fog grid bake pipeline");
            return new RtFogGrid(ctx, volume, descriptorSetLayout, descriptorPool, descriptorSet,
                    pipelineLayout, pipeline);
        } catch (Throwable t) {
            volume.destroy();
            throw t;
        }
    }

    /** Device address of the packed volume the fog march samples (0 before creation). */
    public long volumeAddress() {
        return volume.deviceAddress;
    }

    /**
     * Record this frame's fog-grid bake: marching light columns through the CPU-published section grid
     * into {@code volume}. Must be recorded before the trace with a barrier after (the caller's); the
     * raygen stages sample the volume the trace itself will read. {@code fogGridAddress} may be 0 (no
     * terrain published yet) — every tile then reads as clear and the volume becomes all-transmissive,
     * exactly matching a world with no occluders.
     */
    public void record(VkCommandBuffer cmd, long worldPushAddress, long fogGridAddress,
                       int gridShiftX, int gridShiftY, int gridShiftZ) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "fog grid bake")) {
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipelineLayout, 0, stack.longs(descriptorSet), null);
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            ByteBuffer push = stack.malloc(FogGridPushData.BYTE_SIZE);
            new FogGridPushData(worldPushAddress, fogGridAddress, volume.deviceAddress,
                    gridShiftX, gridShiftY, gridShiftZ).write(push);
            VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (VOLUME_CELLS + GROUP_SIZE - 1) / GROUP_SIZE,
                    (VOLUME_CELLS + GROUP_SIZE - 1) / GROUP_SIZE, VOLUME_LEVELS);
            VulkanCommandEncoder.memoryBarrier(cmd, stack); // volume writes visible to raygen/miss
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, pipeline, null);
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        volume.destroy();
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
        try (InputStream input = RtFogGrid.class.getResourceAsStream(resource)) {
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
