package dev.comfyfluffy.caustica.rt.volumetric;

/**
 * Fixed camera-frustum froxel quality presets.
 *
 * <p>The maximum preset is exactly 128 x 64 x 128. At the default 128-block range and a linear depth
 * distribution its 128 longitudinal slices average one block each. Lower presets cover the same physical
 * frustum and reduce all axes together; dimensions never depend on display or ray-tracing resolution, so a
 * dynamic-resolution change cannot invalidate or reproject hidden temporal state.</p>
 */
public final class RtFroxelGrid {
    public static final int MAX_WIDTH = 128;
    public static final int MAX_HEIGHT = 64;
    public static final int MAX_DEPTH = 128;

    public enum Quality {
        LOW(64, 32, 64),
        MEDIUM(96, 48, 96),
        HIGH(MAX_WIDTH, MAX_HEIGHT, MAX_DEPTH);

        private final int width;
        private final int height;
        private final int depth;

        Quality(int width, int height, int depth) {
            this.width = width;
            this.height = height;
            this.depth = depth;
        }

        public int width() {
            return width;
        }

        public int height() {
            return height;
        }

        public int depth() {
            return depth;
        }
    }

    public record Dimensions(int width, int height, int depth) {
        public Dimensions {
            if (width < 1 || height < 1 || depth < 1
                    || width > MAX_WIDTH || height > MAX_HEIGHT || depth > MAX_DEPTH) {
                throw new IllegalArgumentException("Invalid froxel dimensions: "
                        + width + " x " + height + " x " + depth);
            }
        }
    }

    private RtFroxelGrid() {
    }

    public static Dimensions dimensions(Quality quality) {
        Quality preset = quality == null ? Quality.HIGH : quality;
        return new Dimensions(preset.width(), preset.height(), preset.depth());
    }
}
