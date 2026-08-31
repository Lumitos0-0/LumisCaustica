package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.rt.gen.DiReservoirRecordData;

/**
 * Addressing and invalidation for the ReSTIR DI reservoir history, kept out of {@code RtComposite} so the
 * arithmetic a unit test can pin down is not welded to a class whose static initializers bring up the whole
 * renderer.
 *
 * <p>A history cell is one {@link #RECORD_BYTES}-byte {@code DiReservoirRecord}, written by exactly one
 * pixel per frame — the anchor of the block the cell covers — and read by every pixel that projects into
 * it. The grid is indexed in render pixels divided by {@code HISTORY_DIVISOR}, which is why the extent is
 * a ceiling rather than a scale: cells beyond it are never addressed, and the shader clamps its lookup
 * against the extent reported here instead of against an image it has no binding for.
 *
 * <p>Nothing clears the history, on an allocation or on a reset. A stored record is only trusted after it
 * survives {@link #generationTag} equality, a light index range check, a positivity check on both reservoir
 * scalars and a position/normal comparison against the surface being shaded, so leftover memory from a
 * previous owner is rejected by the same machinery that rejects a light that has moved — and a record that
 * passes all of it describes a sample that is still valid for that surface.
 */
public final class RtRestirDi {
    /** One stored reservoir, reflected from the shader's own record type (see layout_probe.slang). */
    public static final int RECORD_BYTES = DiReservoirRecordData.BYTE_SIZE;

    // Word indices into a record, as raw uints. The debug present pass reads a history this way instead of
    // importing the world shader module, and these are the constants it hard-codes: changing the record in
    // world_common.slang moves the reflection and therefore this test, which is where the two are tied.
    public static final int LIGHT_INDEX_WORD = 4;
    public static final int WEIGHT_WORD = 6;
    public static final int M_WORD = 7;
    public static final int PHAT_WORD = 8;
    public static final int GEN_TAG_WORD = 9;
    public static final int SPATIAL_WORD = 10;
    /** The ReSTIR GI cell this surface falls in, plus one; 0 means the grid is off or the surface is out of
     * reach. Diagnostic only — no validator reads it — and {@code RtGiGrid} is what computes the same box. */
    public static final int GI_CELL_WORD = 11;

    /**
     * Ceiling on one dimension of the history grid. The divisor normally keeps the grid far below this; it
     * only bites at a divisor of 1 on a very wide render target, where an uncapped grid would cost more than
     * the entire DLSS-RR guide set. It is a ceiling and not a scale — pixels whose cell falls past it simply
     * neither store nor reuse, which the trace bounds against the reported extent on both paths.
     */
    public static final int MAX_DIMENSION = 8192;

    private RtRestirDi() {
    }

    /** History extent in cells for one render-target dimension. */
    public static int historyExtent(int renderExtent, int divisor) {
        if (renderExtent < 0) {
            throw new IllegalArgumentException("render extent must not be negative: " + renderExtent);
        }
        if (divisor < 1) {
            throw new IllegalArgumentException("history divisor must be at least 1: " + divisor);
        }
        return (renderExtent + divisor - 1) / divisor;
    }

    /** Bytes to allocate for a history grid of {@code cells} cells. */
    public static long historyBytes(int cells) {
        if (cells < 0) {
            throw new IllegalArgumentException("cell count must not be negative: " + cells);
        }
        return Math.multiplyExact((long) cells, (long) RECORD_BYTES);
    }

    /**
     * The generation tag a frame's records are written with and matched against. Zero is reserved for
     * "reuse is off", so it is folded out. The reset counter is mixed in because a light-generation number
     * cannot by itself say "this buffer was just reallocated": placing a block, an F3 invalidation and a
     * resize all change what a stored index means, and only some of them move the counter.
     */
    public static int generationTag(long generation, int resets) {
        long mixed = generation * 0x9E3779B97F4A7C15L
                + Integer.toUnsignedLong(resets) * 0xC2B2AE3D27D4EB4FL;
        int tag = (int) (mixed >>> 32);
        return tag == 0 ? 1 : tag;
    }
}
