package dev.comfyfluffy.caustica.rt.volumetric;

/**
 * Immutable dimensions and depth mapping for the camera-aligned fog volume. XY cells cover square
 * blocks of render pixels; Z boundaries follow a power distribution so nearby participating media get
 * more samples than distant haze.
 */
public record RtFroxelGrid(int width, int height, int depth, int pixelSize) {
    public RtFroxelGrid {
        if (width <= 0 || height <= 0 || depth <= 0 || pixelSize <= 0) {
            throw new IllegalArgumentException("Froxel dimensions and pixel size must be positive");
        }
    }

    public static RtFroxelGrid forRenderSize(int renderWidth, int renderHeight, int pixelSize,
                                             int depthSlices) {
        if (renderWidth <= 0 || renderHeight <= 0) {
            throw new IllegalArgumentException("Render dimensions must be positive");
        }
        return new RtFroxelGrid(ceilDiv(renderWidth, pixelSize), ceilDiv(renderHeight, pixelSize),
                depthSlices, pixelSize);
    }

    /** Distance at a Z boundary, where boundary zero is the camera and boundary {@code depth} is max. */
    public float boundaryDistance(int boundary, float maxDistance, float distributionExponent) {
        if (boundary < 0 || boundary > depth) {
            throw new IllegalArgumentException("Froxel boundary out of range: " + boundary);
        }
        requireDepthParameters(maxDistance, distributionExponent);
        float normalized = (float) boundary / depth;
        return (float) (Math.pow(normalized, distributionExponent) * maxDistance);
    }

    /** Continuous boundary coordinate corresponding to a camera-ray distance. */
    public float sliceForDistance(float distance, float maxDistance, float distributionExponent) {
        requireDepthParameters(maxDistance, distributionExponent);
        float normalized = Math.clamp(distance / maxDistance, 0.0f, 1.0f);
        return (float) (Math.pow(normalized, 1.0f / distributionExponent) * depth);
    }

    public long cellCount() {
        return Math.multiplyExact(Math.multiplyExact((long) width, height), depth);
    }

    private static int ceilDiv(int value, int divisor) {
        return Math.floorDiv(value - 1, divisor) + 1;
    }

    private static void requireDepthParameters(float maxDistance, float distributionExponent) {
        if (!Float.isFinite(maxDistance) || maxDistance <= 0.0f
                || !Float.isFinite(distributionExponent) || distributionExponent <= 0.0f) {
            throw new IllegalArgumentException("Froxel depth parameters must be finite and positive");
        }
    }
}
