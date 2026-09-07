package dev.comfyfluffy.caustica.rt.pipeline;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.util.vma.Vma;
import org.lwjgl.util.vma.VmaAllocationCreateInfo;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkBufferImageCopy;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkImageCreateInfo;
import org.lwjgl.vulkan.VkImageMemoryBarrier;
import org.lwjgl.vulkan.VkImageViewCreateInfo;
import org.lwjgl.vulkan.VkSamplerCreateInfo;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;

/**
 * NVIDIA's spatiotemporal blue-noise mask (vec2 family, 128x128x64), used by the froxel fog to place its
 * sun-disc sample.
 *
 * <p>Baked from {@code NVIDIAGameWorks/SpatiotemporalBlueNoiseSDK}. Only the R and G channels carry the
 * vec2, so the resource stores two bytes per texel and uploads as {@code R8G8_UNORM} — 2 MB rather than
 * the 4 MB an RGBA copy would cost.
 *
 * <p>Why this rather than an analytic sequence: an STBN mask is blue in <em>both</em> space and time.
 * Verified on the shipped asset, the spatial high/low power ratio is 22.6x and the temporal ratio 8.5x,
 * which means the sampling error it produces is concentrated in exactly the frequencies a small
 * low-pass filter removes. That only pays off if such a filter exists and if the mask is indexed along
 * the axis it filters — see {@code fogBlueNoise} in {@code shaders/pipelines/world/fog.slang}, which
 * maps the mask's Y axis to the froxel SLICE for that reason.
 *
 * <p>Device lifetime, loaded once. The sampler is unnormalized-free and uses REPEAT: the mask tiles by
 * construction, so wrapping is the intended behaviour rather than an error case.
 */
public final class RtBlueNoise {
    private static final String RESOURCE = "/caustica/noise/stbn_vec2_128x128x64.bin";
    private static final int MAGIC = 0x4E425453; // "STBN" little-endian
    private static final int HEADER_BYTES = 4 * 5; // magic, version, width, height, depth

    private final VkDevice vk;
    private final long vma;
    private final long image;
    private final long allocation;
    private final long view;
    private final long sampler;
    public final int width;
    public final int height;
    public final int depth;
    private boolean destroyed;

    private RtBlueNoise(VkDevice vk, long vma, long image, long allocation, long view, long sampler,
                        int width, int height, int depth) {
        this.vk = vk;
        this.vma = vma;
        this.image = image;
        this.allocation = allocation;
        this.view = view;
        this.sampler = sampler;
        this.width = width;
        this.height = height;
        this.depth = depth;
    }

    public long view() {
        return view;
    }

    public long sampler() {
        return sampler;
    }

    public static RtBlueNoise load(RtContext ctx) {
        ByteBuffer data = readResource(RESOURCE);
        try {
            data.order(ByteOrder.LITTLE_ENDIAN);
            int magic = data.getInt(0);
            if (magic != MAGIC) {
                throw new IllegalStateException(RESOURCE + ": bad magic 0x" + Integer.toHexString(magic));
            }
            int version = data.getInt(4);
            if (version != 1) {
                throw new IllegalStateException(RESOURCE + ": unsupported version " + version);
            }
            int width = data.getInt(8);
            int height = data.getInt(12);
            int depth = data.getInt(16);
            if (width <= 0 || height <= 0 || depth <= 0) {
                throw new IllegalStateException(RESOURCE + ": invalid dimensions "
                        + width + "x" + height + "x" + depth);
            }
            long expected = HEADER_BYTES + (long) width * height * depth * 2L;
            if (data.remaining() != expected) {
                throw new IllegalStateException(RESOURCE + ": expected " + expected + " bytes, got "
                        + data.remaining());
            }
            ByteBuffer texels = data.slice(HEADER_BYTES, (int) (expected - HEADER_BYTES));
            return upload(ctx, width, height, depth, texels);
        } finally {
            MemoryUtil.memFree(data);
        }
    }

    private static RtBlueNoise upload(RtContext ctx, int width, int height, int depth, ByteBuffer texels) {
        VkDevice vk = ctx.vk();
        long vma = ctx.vma();
        long createdImage = 0L;
        long createdAllocation = 0L;
        long createdView = 0L;
        long createdSampler = 0L;
        RtBuffer staging = null;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack).sType$Default()
                    .imageType(VK10.VK_IMAGE_TYPE_3D).format(VK10.VK_FORMAT_R8G8_UNORM)
                    .mipLevels(1).arrayLayers(1).samples(VK10.VK_SAMPLE_COUNT_1_BIT)
                    .tiling(VK10.VK_IMAGE_TILING_OPTIMAL)
                    .usage(VK10.VK_IMAGE_USAGE_SAMPLED_BIT | VK10.VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                    .sharingMode(VK10.VK_SHARING_MODE_EXCLUSIVE)
                    .initialLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED);
            imageInfo.extent().set(width, height, depth);
            VmaAllocationCreateInfo allocationInfo = VmaAllocationCreateInfo.calloc(stack)
                    .usage(Vma.VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            LongBuffer imageOut = stack.mallocLong(1);
            PointerBuffer allocationOut = stack.mallocPointer(1);
            RtContext.check(Vma.vmaCreateImage(vma, imageInfo, allocationInfo, imageOut, allocationOut, null),
                    "vmaCreateImage(blue noise)");
            createdImage = imageOut.get(0);
            createdAllocation = allocationOut.get(0);
            RtDebugLabels.nameImage(ctx, createdImage, "STBN vec2 mask");

            VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack).sType$Default()
                    .image(createdImage).viewType(VK10.VK_IMAGE_VIEW_TYPE_3D)
                    .format(VK10.VK_FORMAT_R8G8_UNORM);
            viewInfo.subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                    .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
            LongBuffer viewOut = stack.mallocLong(1);
            RtContext.check(VK10.vkCreateImageView(vk, viewInfo, null, viewOut),
                    "vkCreateImageView(blue noise)");
            createdView = viewOut.get(0);
            RtDebugLabels.nameImageView(ctx, createdView, "STBN vec2 mask view");

            // NEAREST and REPEAT, both deliberate. Interpolating between mask texels would average
            // independent samples and destroy the blue-noise spectrum the mask exists to provide, and
            // the mask tiles by construction so wrapping is correct rather than a clamp case.
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_NEAREST).minFilter(VK10.VK_FILTER_NEAREST)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .minLod(0f).maxLod(0f);
            LongBuffer samplerOut = stack.mallocLong(1);
            RtContext.check(VK10.vkCreateSampler(vk, samplerInfo, null, samplerOut),
                    "vkCreateSampler(blue noise)");
            createdSampler = samplerOut.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, createdSampler, "STBN vec2 sampler");

            int totalBytes = texels.remaining();
            staging = ctx.createUploadBuffer(totalBytes, "blue noise upload");
            ByteBuffer mapped = MemoryUtil.memByteBuffer(staging.mapped, totalBytes);
            mapped.put(texels.duplicate());
            staging.flush();

            long uploadImage = createdImage;
            long uploadBuffer = staging.handle;
            ctx.submitSync(cmd -> {
                try (MemoryStack uploadStack = MemoryStack.stackPush()) {
                    VkImageMemoryBarrier.Buffer toTransfer = VkImageMemoryBarrier.calloc(1, uploadStack);
                    toTransfer.get(0).sType$Default()
                            .oldLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED)
                            .newLayout(VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                            .srcAccessMask(0).dstAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                            .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                            .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED).image(uploadImage);
                    toTransfer.get(0).subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                            .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
                    VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK10.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, null, null, toTransfer);

                    VkBufferImageCopy.Buffer copy = VkBufferImageCopy.calloc(1, uploadStack);
                    copy.get(0).bufferOffset(0).bufferRowLength(0).bufferImageHeight(0);
                    copy.get(0).imageSubresource().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                            .mipLevel(0).baseArrayLayer(0).layerCount(1);
                    copy.get(0).imageOffset().set(0, 0, 0);
                    copy.get(0).imageExtent().set(width, height, depth);
                    VK10.vkCmdCopyBufferToImage(cmd, uploadBuffer, uploadImage,
                            VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copy);

                    // GENERAL to match every other sampled image here; the descriptor write must use
                    // the same layout or validation flags the mismatch.
                    VkImageMemoryBarrier.Buffer toRead = VkImageMemoryBarrier.calloc(1, uploadStack);
                    toRead.get(0).sType$Default()
                            .oldLayout(VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                            .newLayout(VK10.VK_IMAGE_LAYOUT_GENERAL)
                            .srcAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                            .dstAccessMask(VK10.VK_ACCESS_SHADER_READ_BIT)
                            .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                            .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED).image(uploadImage);
                    toRead.get(0).subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                            .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
                    VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK10.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, null, null, toRead);
                }
            });
        } catch (Throwable t) {
            if (createdSampler != 0L) VK10.vkDestroySampler(vk, createdSampler, null);
            if (createdView != 0L) VK10.vkDestroyImageView(vk, createdView, null);
            if (createdImage != 0L) Vma.vmaDestroyImage(vma, createdImage, createdAllocation);
            throw t;
        } finally {
            if (staging != null) {
                staging.destroy();
            }
        }
        return new RtBlueNoise(vk, vma, createdImage, createdAllocation, createdView, createdSampler,
                width, height, depth);
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VK10.vkDestroySampler(vk, sampler, null);
        VK10.vkDestroyImageView(vk, view, null);
        Vma.vmaDestroyImage(vma, image, allocation);
        destroyed = true;
    }

    private static ByteBuffer readResource(String path) {
        try (InputStream in = RtBlueNoise.class.getResourceAsStream(path)) {
            if (in == null) {
                throw new IllegalStateException("missing blue-noise resource: " + path);
            }
            byte[] bytes = in.readAllBytes();
            ByteBuffer buffer = MemoryUtil.memAlloc(bytes.length);
            buffer.put(bytes).flip();
            return buffer;
        } catch (IOException e) {
            throw new IllegalStateException("failed to read blue-noise resource: " + path, e);
        }
    }
}
