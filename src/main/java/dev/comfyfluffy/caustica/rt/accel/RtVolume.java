package dev.comfyfluffy.caustica.rt.accel;

import org.lwjgl.util.vma.Vma;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkDevice;

/**
 * A VMA-backed 3D storage image + 3D view, created in {@code VK_IMAGE_LAYOUT_GENERAL}. Used by the
 * frustum-froxel volumetric passes (scattering, history, integrated). Created via
 * {@link dev.comfyfluffy.caustica.rt.RtContext#createStorageVolume}; freed with {@link #destroy()}.
 */
public final class RtVolume {
    public final long image;
    public final long allocation;
    public final long view;
    public final int width;
    public final int height;
    public final int depth;

    private final long vma;
    private final VkDevice vk;
    private boolean destroyed;

    public RtVolume(long vma, VkDevice vk, long image, long allocation, long view,
                    int width, int height, int depth) {
        this.vma = vma;
        this.vk = vk;
        this.image = image;
        this.allocation = allocation;
        this.view = view;
        this.width = width;
        this.height = height;
        this.depth = depth;
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        if (view != 0L) {
            VK10.vkDestroyImageView(vk, view, null);
        }
        if (image != 0L) {
            Vma.vmaDestroyImage(vma, image, allocation);
        }
        destroyed = true;
    }
}
