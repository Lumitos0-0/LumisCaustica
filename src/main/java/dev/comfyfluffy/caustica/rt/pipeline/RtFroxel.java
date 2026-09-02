package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.RtGpuExecutor;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.FroxelPushData;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.KHRAccelerationStructure;
import org.lwjgl.vulkan.VK10;
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
import org.lwjgl.vulkan.VkWriteDescriptorSetAccelerationStructureKHR;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.comfyfluffy.caustica.rt.RtContext.check;

/**
 * The froxel-based volumetric fog and light-shaft pipeline (Wronski/Frostbite frustum-aligned volume).
 *
 * <p>Two compute passes share a frustum-aligned volume stored as a storage buffer of {@code float4}s (one
 * per froxel: rgb = single-scattered in-scatter toward the camera, w = per-block extinction):
 * <ul>
 *   <li><b>lighting</b> ({@code lighting.comp}) fills the volume. One thread per froxel samples the fog
 *       density field, casts several ray-queried occlusion rays toward the dominant sun/moon light (the
 *       amortized "path-traced volumetric shadow" that yields god rays / light shafts), and writes the
 *       result. The dominant light and its transmittance LUT come from the same sky module the path tracer
 *       uses, so fog, terrain light and the drawn disc never disagree.</li>
 *   <li><b>integrate</b> ({@code integrate.comp}) accumulates the volume along each view ray against the
 *       guide depth and composites the fog over the path-traced radiance. Its output feeds DLSS-RR (or the
 *       fallback upscale), so the fog is denoised, exposed and bloomed with the scene.</li>
 * </ul>
 *
 * <p>The volume is rebuilt every frame. Because the shadow work is done at froxel resolution (a tile of
 * {@code tileSize}x{@code tileSize} pixels per depth slice) rather than per-pixel per step, the per-pixel
 * fog cost is a fixed march over {@code depthSlices} records. A temporal EMA of the volume (gated on a
 * still camera) stabilizes the shadow rays without smearing a moving view.
 */
public final class RtFroxel {
    private static final String SHADER_DIR = "/caustica/shaders/pipelines/froxel/";
    private static final int GROUP_SIZE = 8;

    // Descriptor bindings, must stay in lock-step with the .comp.slang files.
    private static final int BIND_TLAS = 0;            // set 0 (lighting only)
    private static final int BIND_LUT = 0;             // set 1, lighting (combined image sampler)
    private static final int BIND_SOURCE = 0;          // set 1, integrate (storage image)
    private static final int BIND_DEPTH = 1;           // set 1, integrate (storage image)
    private static final int BIND_OUT = 2;             // set 1, integrate (storage image)
    private static final int TLAS_RING = 4;

    private final RtContext ctx;
    // TLAS descriptor-set ring for the ray-query occlusion rays (the handle changes most frames).
    private final long tlasLayout;
    private final long tlasPool;
    private final long[] tlasSets;
    private final RtGpuExecutor.TrackedGraphicsUse[] tlasUses;
    private int tlasCurrent = -1;

    private final long lightingSetLayout;
    private final long lightingSetPool;
    private final long lightingSet;
    private final long lightingPipelineLayout;
    private final long lightingPipeline;

    private final long integrateSetLayout;
    private final long integrateSetPool;
    private final long integrateSet;
    private final long integratePipelineLayout;
    private final long integratePipeline;

    private RtBuffer volume;
    private RtBuffer history;

    private int tileW;
    private int tileH;
    private int slices;
    // False until the first lighting dispatch has actually been recorded (and again after a resize
    // recreates the history buffer): the temporal EMA then must not read the not-yet-written history,
    // whose device memory is undefined on a fresh allocation.
    private boolean historyReady;
    private boolean destroyed;

    private RtFroxel(RtContext ctx, long tlasLayout, long tlasPool, long[] tlasSets,
                     long lightingSetLayout, long lightingSetPool, long lightingSet,
                     long lightingPipelineLayout, long lightingPipeline,
                     long integrateSetLayout, long integrateSetPool, long integrateSet,
                     long integratePipelineLayout, long integratePipeline) {
        this.ctx = ctx;
        this.tlasLayout = tlasLayout;
        this.tlasPool = tlasPool;
        this.tlasSets = tlasSets;
        this.tlasUses = new RtGpuExecutor.TrackedGraphicsUse[tlasSets.length];
        for (int i = 0; i < tlasUses.length; i++) {
            tlasUses[i] = new RtGpuExecutor.TrackedGraphicsUse();
        }
        this.lightingSetLayout = lightingSetLayout;
        this.lightingSetPool = lightingSetPool;
        this.lightingSet = lightingSet;
        this.lightingPipelineLayout = lightingPipelineLayout;
        this.lightingPipeline = lightingPipeline;
        this.integrateSetLayout = integrateSetLayout;
        this.integrateSetPool = integrateSetPool;
        this.integrateSet = integrateSet;
        this.integratePipelineLayout = integratePipelineLayout;
        this.integratePipeline = integratePipeline;
    }

    /** Builds the pipelines and empty descriptor slots. {@link #ensureGrid} binds the volume buffers first. */
    public static RtFroxel create(RtContext ctx, long transmittanceView, long transmittanceSampler) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer p = stack.mallocLong(1);

            // TLAS ring: binding 0 of set 0, compute stage (ray query).
            VkDescriptorSetLayoutBinding.Buffer tlasBinds = VkDescriptorSetLayoutBinding.calloc(1, stack);
            tlasBinds.get(0).binding(BIND_TLAS)
                    .descriptorType(KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorSetLayoutCreateInfo tlasDsl = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(tlasBinds);
            check(VK10.vkCreateDescriptorSetLayout(vk, tlasDsl, null, p), "vkCreateDescriptorSetLayout(froxel TLAS)");
            long tlasLayout = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, tlasLayout, "froxel TLAS set layout");

            VkDescriptorPoolSize.Buffer tlasPoolSize = VkDescriptorPoolSize.calloc(1, stack);
            tlasPoolSize.get(0).type(KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                    .descriptorCount(TLAS_RING);
            VkDescriptorPoolCreateInfo tlasPoolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(TLAS_RING).pPoolSizes(tlasPoolSize);
            check(VK10.vkCreateDescriptorPool(vk, tlasPoolInfo, null, p), "vkCreateDescriptorPool(froxel TLAS)");
            long tlasPool = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, tlasPool, "froxel TLAS pool");

            LongBuffer tlasLayouts = stack.mallocLong(TLAS_RING);
            for (int i = 0; i < TLAS_RING; i++) {
                tlasLayouts.put(i, tlasLayout);
            }
            VkDescriptorSetAllocateInfo tlasAlloc = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(tlasPool).pSetLayouts(tlasLayouts);
            LongBuffer tlasSetsBuf = stack.mallocLong(TLAS_RING);
            check(VK10.vkAllocateDescriptorSets(vk, tlasAlloc, tlasSetsBuf), "vkAllocateDescriptorSets(froxel TLAS)");
            long[] tlasSets = new long[TLAS_RING];
            tlasSetsBuf.get(tlasSets);
            for (int i = 0; i < TLAS_RING; i++) {
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, tlasSets[i], "froxel TLAS set " + i);
            }

            // Lighting set (set 1, binding 0): the sky transmittance LUT sampled by the occlusion light.
            VkDescriptorSetLayoutBinding.Buffer lutBinds = VkDescriptorSetLayoutBinding.calloc(1, stack);
            lutBinds.get(0).binding(BIND_LUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorSetLayoutCreateInfo lutDsl = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(lutBinds);
            check(VK10.vkCreateDescriptorSetLayout(vk, lutDsl, null, p), "vkCreateDescriptorSetLayout(froxel lighting)");
            long lightingSetLayout = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, lightingSetLayout, "froxel lighting set layout");

            VkDescriptorPoolSize.Buffer lutPoolSize = VkDescriptorPoolSize.calloc(1, stack);
            lutPoolSize.get(0).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1);
            VkDescriptorPoolCreateInfo lutPoolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(1).pPoolSizes(lutPoolSize);
            check(VK10.vkCreateDescriptorPool(vk, lutPoolInfo, null, p), "vkCreateDescriptorPool(froxel lighting)");
            long lightingSetPool = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, lightingSetPool, "froxel lighting pool");

            VkDescriptorSetAllocateInfo lutAlloc = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(lightingSetPool)
                    .pSetLayouts(stack.longs(lightingSetLayout));
            LongBuffer lightingSetBuf = stack.mallocLong(1);
            check(VK10.vkAllocateDescriptorSets(vk, lutAlloc, lightingSetBuf), "vkAllocateDescriptorSets(froxel lighting)");
            long lightingSet = lightingSetBuf.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, lightingSet, "froxel lighting set");

            // Bind the transmittance LUT once (the sky LUT lives for the device's lifetime).
            VkDescriptorImageInfo lutImg = VkDescriptorImageInfo.calloc(1, stack)
                    .sampler(transmittanceSampler).imageView(transmittanceView)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkWriteDescriptorSet lutWrite = VkWriteDescriptorSet.calloc(1, stack)
                    .sType$Default().dstSet(lightingSet).dstBinding(BIND_LUT)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .pImageInfo(VkDescriptorImageInfo.create(lutImg.address(0), 1));
            VK10.vkUpdateDescriptorSets(vk, lutWrite, null);

            // Integrate set (set 1, bindings 0..2): source color, guide depth, fogged output.
            VkDescriptorSetLayoutBinding.Buffer imgBinds = VkDescriptorSetLayoutBinding.calloc(3, stack);
            imgBinds.get(BIND_SOURCE).binding(BIND_SOURCE).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            imgBinds.get(BIND_DEPTH).binding(BIND_DEPTH).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            imgBinds.get(BIND_OUT).binding(BIND_OUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorSetLayoutCreateInfo imgDsl = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(imgBinds);
            check(VK10.vkCreateDescriptorSetLayout(vk, imgDsl, null, p), "vkCreateDescriptorSetLayout(froxel integrate)");
            long integrateSetLayout = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, integrateSetLayout, "froxel integrate set layout");

            VkDescriptorPoolSize.Buffer imgPoolSize = VkDescriptorPoolSize.calloc(1, stack);
            imgPoolSize.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(3);
            VkDescriptorPoolCreateInfo imgPoolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(1).pPoolSizes(imgPoolSize);
            check(VK10.vkCreateDescriptorPool(vk, imgPoolInfo, null, p), "vkCreateDescriptorPool(froxel integrate)");
            long integrateSetPool = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, integrateSetPool, "froxel integrate pool");

            VkDescriptorSetAllocateInfo imgAlloc = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(integrateSetPool)
                    .pSetLayouts(stack.longs(integrateSetLayout));
            LongBuffer integrateSetBuf = stack.mallocLong(1);
            check(VK10.vkAllocateDescriptorSets(vk, imgAlloc, integrateSetBuf), "vkAllocateDescriptorSets(froxel integrate)");
            long integrateSet = integrateSetBuf.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, integrateSet, "froxel integrate set");

            int pushBytes = FroxelPushData.BYTE_SIZE;
            // Lighting uses set 0 = TLAS (ray query) and set 1 = transmittance LUT; the shader declares
            // them as (0,0) and (0,1), so the layout order must match.
            long lightingPipelineLayout = createPipelineLayout(ctx, stack, tlasLayout, lightingSetLayout, pushBytes, "froxel lighting");
            long integratePipelineLayout = createPipelineLayout(ctx, stack, integrateSetLayout, 0L, pushBytes, "froxel integrate");
            long lightingPipeline = createComputePipeline(ctx, stack, lightingPipelineLayout, "lighting.comp.spv", "froxel lighting pipeline");
            long integratePipeline = createComputePipeline(ctx, stack, integratePipelineLayout, "integrate.comp.spv", "froxel integrate pipeline");

            return new RtFroxel(ctx, tlasLayout, tlasPool, tlasSets,
                    lightingSetLayout, lightingSetPool, lightingSet, lightingPipelineLayout, lightingPipeline,
                    integrateSetLayout, integrateSetPool, integrateSet, integratePipelineLayout, integratePipeline);
        }
    }

    private static long createPipelineLayout(RtContext ctx, MemoryStack stack, long set0, long set1,
                                             int pushBytes, String label) {
        VkDevice vk = ctx.vk();
        VkPipelineLayoutCreateInfo info = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default();
        if (set1 != 0L) {
            info.pSetLayouts(stack.longs(set0, set1));
        } else {
            info.pSetLayouts(stack.longs(set0));
        }
        if (pushBytes > 0) {
            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(pushBytes);
            info.pPushConstantRanges(pushRange);
        }
        LongBuffer p = stack.mallocLong(1);
        check(VK10.vkCreatePipelineLayout(vk, info, null, p), "vkCreatePipelineLayout(" + label + ")");
        long layout = p.get(0);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, label + " layout");
        return layout;
    }

    public int tileWidth() {
        return tileW;
    }

    public int tileHeight() {
        return tileH;
    }

    public int depthSlices() {
        return slices;
    }

    /** (Re)allocates the froxel volume (and optional history) buffers for the current render resolution. */
    public void ensureGrid(RtContext ctx, int renderW, int renderH) {
        int tileSize = CausticaConfig.Rt.Froxel.TILE_SIZE.value();
        int sliceCount = CausticaConfig.Rt.Froxel.DEPTH_SLICES.value();
        int newTileW = Math.max(1, (renderW + tileSize - 1) / tileSize);
        int newTileH = Math.max(1, (renderH + tileSize - 1) / tileSize);
        if (volume != null && newTileW == tileW && newTileH == tileH && sliceCount == slices) {
            return;
        }
        ctx.waitIdle(); // resize is rare; no in-flight frame may reference the old buffer
        if (volume != null) {
            volume.destroy();
            volume = null;
        }
        if (history != null) {
            history.destroy();
            history = null;
        }
        long count = Math.multiplyExact(Math.multiplyExact((long) newTileW, newTileH), sliceCount);
        long bytes = Math.multiplyExact(count, 16L); // float4 per froxel
        volume = ctx.createBuffer(bytes, VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, "froxel volume");
        if (CausticaConfig.Rt.Froxel.TEMPORAL.value()) {
            history = ctx.createBuffer(bytes, VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, "froxel history");
        }
        historyReady = false; // fresh history: undefined device memory until the first write-back
        tileW = newTileW;
        tileH = newTileH;
        slices = sliceCount;
    }

    /** Points the integrate set at the current frame's scene color, guide depth, and fog output. */
    public void setIntegrateImages(RtContext ctx, RtImage source, RtImage depth, RtImage out) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer images = VkDescriptorImageInfo.calloc(3, stack);
            long[] views = {source.view, depth.view, out.view};
            int[] bindings = {BIND_SOURCE, BIND_DEPTH, BIND_OUT};
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(3, stack);
            for (int i = 0; i < 3; i++) {
                images.get(i).imageView(views[i]).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                writes.get(i).sType$Default().dstSet(integrateSet).dstBinding(bindings[i])
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(VkDescriptorImageInfo.create(images.address(i), 1));
            }
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
    }

    /** Binds the next TLAS ring slot and records the froxel lighting dispatch. */
    public void recordLighting(VkCommandBuffer cmd, long worldPushAddress, long tlasHandle,
                               RtGpuExecutor.GraphicsUse graphicsUse,
                               RtGpuExecutor.GraphicsUseWaiter graphicsUseWaiter,
                               int renderW, int renderH) {
        if (volume == null || !CausticaConfig.Rt.Froxel.ENABLED.value()) {
            return;
        }
        tlasCurrent = (tlasCurrent + 1) % TLAS_RING;
        RtGpuExecutor.TrackedGraphicsUse slotUse = tlasUses[tlasCurrent];
        graphicsUseWaiter.await(slotUse);
        long tlasSet = tlasSets[tlasCurrent];
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkWriteDescriptorSetAccelerationStructureKHR asWrite = VkWriteDescriptorSetAccelerationStructureKHR
                    .calloc(stack)
                    .sType(KHRAccelerationStructure.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR)
                    .pAccelerationStructures(stack.longs(tlasHandle));
            VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(1, stack);
            write.get(0).sType$Default().pNext(asWrite.address()).dstSet(tlasSet).dstBinding(BIND_TLAS)
                    .descriptorCount(1).descriptorType(KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
            VK10.vkUpdateDescriptorSets(ctx.vk(), write, null);
        }
        slotUse.mark(graphicsUse);

        float near = 0.05f;
        float far = CausticaConfig.Rt.Froxel.FAR.value();
        float tileSize = CausticaConfig.Rt.Froxel.TILE_SIZE.value();
        float density = CausticaConfig.Rt.Froxel.DENSITY.value();
        float falloff = CausticaConfig.Rt.Froxel.HEIGHT_FALLOFF.value();
        float phaseG = CausticaConfig.Rt.Froxel.PHASE_G.value();
        float albedo = CausticaConfig.Rt.Froxel.SCATTER_ALBEDO.value();
        float noiseIntensity = CausticaConfig.Rt.Froxel.NOISE_INTENSITY.value();
        float noiseScale = CausticaConfig.Rt.Froxel.NOISE_SCALE.value();
        float intensity = CausticaConfig.Rt.Froxel.INTENSITY.value();
        // On the first frame (and after a resize) the history buffer is undefined device memory, so the
        // EMA is disabled: the shader writes fresh values to both volume and history, and historyReady
        // flips only once a complete frame's history actually exists to read next frame.
        float temporalBlend = historyReady && CausticaConfig.Rt.Froxel.TEMPORAL.value()
                ? CausticaConfig.Rt.Froxel.TEMPORAL_BLEND.value() : 0.0f;

        try (MemoryStack stack = MemoryStack.stackPush()) {
            ByteBuffer push = stack.malloc(FroxelPushData.BYTE_SIZE);
            new FroxelPushData(
                    new FroxelPushData.Float4(near, far, tileSize, (float) slices),
                    new FroxelPushData.Float4(density, falloff, phaseG, 1.0f),
                    new FroxelPushData.Float4(albedo, noiseIntensity, noiseScale, temporalBlend),
                    new FroxelPushData.Float4(intensity, 1.0f, 1.0f, 1.0f),
                    worldPushAddress,
                    volume.deviceAddress,
                    history != null ? history.deviceAddress : 0L,
                    pack16(renderW, renderH),
                    pack16(tileW, tileH)).write(push);
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, lightingPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, lightingPipelineLayout,
                    0, stack.longs(tlasSet, lightingSet), null);
            VK10.vkCmdPushConstants(cmd, lightingPipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (tileW + GROUP_SIZE - 1) / GROUP_SIZE,
                    (tileH + GROUP_SIZE - 1) / GROUP_SIZE, slices);
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
        historyReady = true; // this dispatch (once executed) wrote a complete history for next frame
    }

    /** Records the froxel integrate (fog composite) dispatch over the render-resolution scene. */
    public void recordIntegrate(VkCommandBuffer cmd, long worldPushAddress, int renderW, int renderH) {
        if (volume == null) {
            return;
        }
        float near = 0.05f;
        float far = CausticaConfig.Rt.Froxel.FAR.value();
        float tileSize = CausticaConfig.Rt.Froxel.TILE_SIZE.value();
        float enabled = CausticaConfig.Rt.Froxel.ENABLED.value() ? 1.0f : 0.0f;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            ByteBuffer push = stack.malloc(FroxelPushData.BYTE_SIZE);
            new FroxelPushData(
                    new FroxelPushData.Float4(near, far, tileSize, (float) slices),
                    new FroxelPushData.Float4(0.0f, 0.0f, 0.0f, enabled),
                    new FroxelPushData.Float4(0.0f, 0.0f, 0.0f, 0.0f),
                    new FroxelPushData.Float4(0.0f, 1.0f, 1.0f, 1.0f),
                    worldPushAddress,
                    volume.deviceAddress,
                    history != null ? history.deviceAddress : 0L,
                    pack16(renderW, renderH),
                    pack16(tileW, tileH)).write(push);
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, integratePipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, integratePipelineLayout,
                    0, stack.longs(integrateSet), null);
            VK10.vkCmdPushConstants(cmd, integratePipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (renderW + GROUP_SIZE - 1) / GROUP_SIZE,
                    (renderH + GROUP_SIZE - 1) / GROUP_SIZE, 1);
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
    }

    private static int pack16(int low, int high) {
        return (low & 0xffff) | ((high & 0xffff) << 16);
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        if (volume != null) {
            volume.destroy();
        }
        if (history != null) {
            history.destroy();
        }
        VK10.vkDestroyPipeline(vk, lightingPipeline, null);
        VK10.vkDestroyPipeline(vk, integratePipeline, null);
        VK10.vkDestroyPipelineLayout(vk, lightingPipelineLayout, null);
        VK10.vkDestroyPipelineLayout(vk, integratePipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, integrateSetPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, integrateSetLayout, null);
        VK10.vkDestroyDescriptorPool(vk, lightingSetPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, lightingSetLayout, null);
        VK10.vkDestroyDescriptorPool(vk, tlasPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, tlasLayout, null);
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
        try (InputStream input = RtFroxel.class.getResourceAsStream(resource)) {
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
