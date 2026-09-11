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
import java.nio.LongBuffer;

/**
 * Spatiotemporal jitter for the fog bake: a 64x64x64 RGBA8 volume bound to the world set, sampled
 * nearest with repeat wrap (XY tiled over froxel columns, Z stepped per frame).
 *
 * <p>Texels come from Christoph Peters' CC0 3D blue noise ({@code 64_64_64/HDR_LA.raw}, mirrored at
 * <a href="https://github.com/Calinou/free-blue-noise-textures">Calinou/free-blue-noise-textures</a>),
 * repacked offline by {@code tools/repack_bluenoise64.py} into (L_f, A_f, L_f+32, A_f+32) per texel:
 * L/A is Peters' decorrelated channel pair and the +32-slice pair is ~independent, so one fetch
 * yields four usable jitter channels. See THIRD_PARTY_NOTICES.md.
 */
public final class RtStbn {
    public static final int SIZE = 64;
    private static final String RESOURCE = "/caustica/noise/bluenoise64_rgba.bin";
    private static final int EXPECTED_BYTES = SIZE * SIZE * SIZE * 4;

    private final VkDevice vk;
    private final long vma;
    private final long image;
    private final long allocation;
    private final long view;
    private final long sampler;
    private boolean destroyed;

    private RtStbn(VkDevice vk, long vma, long image, long allocation, long view, long sampler) {
        this.vk = vk;
        this.vma = vma;
        this.image = image;
        this.allocation = allocation;
        this.view = view;
        this.sampler = sampler;
    }

    public long view() {
        return view;
    }

    public long sampler() {
        return sampler;
    }

    /** Loads the jitter volume; device-lifetime, uploaded once via a staging buffer. */
    public static RtStbn create(RtContext ctx) {
        ByteBuffer data = readResource(RESOURCE);
        try {
            if (data.remaining() != EXPECTED_BYTES) {
                throw new IllegalStateException(RESOURCE + ": expected " + EXPECTED_BYTES + " bytes, got "
                        + data.remaining());
            }
            return upload(ctx, data, "fog jitter volume");
        } finally {
            MemoryUtil.memFree(data);
        }
    }

    private static RtStbn upload(RtContext ctx, ByteBuffer texels, String label) {
        VkDevice vk = ctx.vk();
        long vma = ctx.vma();
        long createdImage = 0L;
        long createdAllocation = 0L;
        long createdView = 0L;
        long createdSampler = 0L;
        RtBuffer staging = null;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack).sType$Default()
                    .imageType(VK10.VK_IMAGE_TYPE_3D).format(VK10.VK_FORMAT_R8G8B8A8_UNORM)
                    .mipLevels(1).arrayLayers(1).samples(VK10.VK_SAMPLE_COUNT_1_BIT)
                    .tiling(VK10.VK_IMAGE_TILING_OPTIMAL)
                    .usage(VK10.VK_IMAGE_USAGE_SAMPLED_BIT | VK10.VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                    .sharingMode(VK10.VK_SHARING_MODE_EXCLUSIVE)
                    .initialLayout(VK10.VK_IMAGE_LAYOUT_UNDEFINED);
            imageInfo.extent().set(SIZE, SIZE, SIZE);
            VmaAllocationCreateInfo allocationInfo = VmaAllocationCreateInfo.calloc(stack)
                    .usage(Vma.VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            LongBuffer imageOut = stack.mallocLong(1);
            PointerBuffer allocationOut = stack.mallocPointer(1);
            RtContext.check(Vma.vmaCreateImage(vma, imageInfo, allocationInfo, imageOut, allocationOut, null),
                    "vmaCreateImage(" + label + ")");
            createdImage = imageOut.get(0);
            createdAllocation = allocationOut.get(0);
            RtDebugLabels.nameImage(ctx, createdImage, label);

            VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack).sType$Default()
                    .image(createdImage).viewType(VK10.VK_IMAGE_VIEW_TYPE_3D)
                    .format(VK10.VK_FORMAT_R8G8B8A8_UNORM);
            viewInfo.subresourceRange().aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                    .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
            LongBuffer viewOut = stack.mallocLong(1);
            RtContext.check(VK10.vkCreateImageView(vk, viewInfo, null, viewOut),
                    "vkCreateImageView(" + label + " view)");
            createdView = viewOut.get(0);
            RtDebugLabels.nameImageView(ctx, createdView, label + " view");

            // Nearest + repeat: noise texels are addressed explicitly ((column + 0.5) / 64 tiles by
            // wrapping) and must never be filtered — interpolation would correlate the jitter.
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_NEAREST).minFilter(VK10.VK_FILTER_NEAREST)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .minLod(0f).maxLod(0f);
            LongBuffer samplerOut = stack.mallocLong(1);
            RtContext.check(VK10.vkCreateSampler(vk, samplerInfo, null, samplerOut),
                    "vkCreateSampler(" + label + " sampler)");
            createdSampler = samplerOut.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, createdSampler, label + " sampler");

            int totalBytes = texels.remaining();
            staging = ctx.createUploadBuffer(totalBytes, label + " upload");
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
                    copy.get(0).imageExtent().set(SIZE, SIZE, SIZE);
                    VK10.vkCmdCopyBufferToImage(cmd, uploadBuffer, uploadImage,
                            VK10.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copy);

                    // GENERAL, not SHADER_READ_ONLY_OPTIMAL, to match every other sampled/storage
                    // image in this codebase — the descriptor write must use the same layout or
                    // validation flags a mismatch.
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
            if (staging != null) staging.destroy();
        }
        return new RtStbn(vk, vma, createdImage, createdAllocation, createdView, createdSampler);
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
        try (InputStream in = RtStbn.class.getResourceAsStream(path)) {
            if (in == null) {
                throw new IllegalStateException("missing blue noise resource: " + path);
            }
            byte[] bytes = in.readAllBytes();
            ByteBuffer buf = MemoryUtil.memAlloc(bytes.length);
            buf.put(bytes);
            buf.flip();
            return buf;
        } catch (IOException e) {
            throw new IllegalStateException("failed to read blue noise resource: " + path, e);
        }
    }
}
