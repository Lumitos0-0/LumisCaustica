package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.rt.gen.GiRadianceRecordData;

/**
 * Extent policy for the ReSTIR GI radiance grid, kept out of {@code RtComposite} for the same reason
 * {@link RtRestirDi} is: the arithmetic a unit test can pin down should not be welded to a class whose
 * static initializers bring up the whole renderer.
 *
 * <p>The grid is a dense box of {@link #BINS} records per cell, and its extent is decided here rather than in
 * the config because the honest constraint is memory, not a cell count. The config's cell size is a request:
 * {@link #cellBlocks(int, int)} doubles it until the box fits {@link #MAX_BYTES}, and the size it settles on
 * is what gets pushed to the shader, so a scene with a wide view distance gets coarser reuse instead of an
 * allocation nobody thought about.
 *
 * <p>The window is anchored wherever the camera was when it last had to slide, and its minimum corner is
 * aligned down to the cell lattice ({@link #windowOriginBlocks}). That pairing is the property the rest of
 * the design rests on, and it is visible in the arithmetic: a cell index is
 * {@code floorDiv(worldBlock - alignDown(anchor - radius, cell), cell)}, in which the rebase position the
 * shader's space is measured from cancels out entirely. So a rebase — which moves every shader-space
 * position by a whole number of blocks and moves the pushed origin by exactly the same amount — leaves every
 * cell naming the same piece of world, and a cache does not have to be thrown away because the player walked
 * past the threshold that rebase origins. {@link #anchorFor} is the hysteresis that keeps the anchor still
 * in the first place; {@code RtGiGridTest} pins both, since a lattice that quietly stops aligning is
 * invisible in a still image and shows up as a cache that never converges while walking.
 *
 * <p>No records are written or read yet, so nothing here carries a generation tag. When the first writer
 * arrives, the lattice alignment above is what lets it key off world <em>edits</em> instead of camera motion.
 */
public final class RtGiGrid {
    /** One direction bin of one cell, reflected from the shader's own record type. */
    public static final int RECORD_BYTES = GiRadianceRecordData.BYTE_SIZE;

    /**
     * Direction bins per cell, stored as a cube-face decomposition (dominant axis and its sign) rather than
     * an octahedral map: it needs no trigonometry, it is invertible by inspection, and it is the same six
     * faces an irradiance cache has always used. The count is a memory commitment before it is a shader
     * function — the allocation covers six bins per cell from the start, because growing a published
     * allocation is the resize-and-invalidate path you do not want a cache to be on — and the bin function
     * that reads them has to produce this many.
     */
    public static final int BINS = 6;

    /**
     * Ceiling on the whole grid, in bytes. A budget rather than a scale: exceeding it costs a frame of
     * allocation and a cache rebuild, which is worse than the reuse being coarse. The dense light grid next
     * door caps itself at 4M cells for the same reason.
     */
    public static final long MAX_BYTES = 24L << 20;

    private RtGiGrid() {
    }

    /** Cells along one axis for a window extending {@code radiusBlocks} on either side. At least 1. */
    public static int cellsPerAxis(int radiusBlocks, int cellBlocks) {
        if (radiusBlocks < 1) {
            throw new IllegalArgumentException("radius must be at least 1 block: " + radiusBlocks);
        }
        if (cellBlocks < 1) {
            throw new IllegalArgumentException("cell size must be at least 1 block: " + cellBlocks);
        }
        // +1 so the origin cell counts, and the box then spans 2*radius + cell blocks: a surface exactly
        // radiusBlocks from the anchor still lands inside it after the floor divide, which is the case a
        // reader would otherwise have to trust.
        return Math.toIntExact(Math.floorDiv(2L * radiusBlocks, cellBlocks) + 1);
    }

    /** Total cells in the window — a long, because a wide radius and a small cell overflow an int. */
    public static long cellCount(int radiusBlocks, int cellBlocks) {
        long axis = cellsPerAxis(radiusBlocks, cellBlocks);
        return Math.multiplyExact(axis * axis, axis);
    }

    /** Bytes the grid occupies: every cell carries one record per direction bin. */
    public static long gridBytes(int radiusBlocks, int cellBlocks) {
        return gridBytesForCells(cellCount(radiusBlocks, cellBlocks));
    }

    /** The same arithmetic with the cell count spelled out, which is what the budget is compared against. */
    public static long gridBytesForCells(long cells) {
        if (cells < 0) {
            throw new IllegalArgumentException("cell count must not be negative: " + cells);
        }
        return Math.multiplyExact(cells, (long) (BINS * RECORD_BYTES));
    }

    /**
     * The cell size actually used, grown by doubling from {@code requestedBlocks} until the grid fits
     * {@link #MAX_BYTES}. A request that already fits comes back unchanged, which is what keeps this
     * invisible in the common case.
     */
    public static int cellBlocks(int requestedBlocks, int radiusBlocks) {
        int cell = Math.max(1, requestedBlocks);
        // 24 doublings is far more than the budget needs to win at any allowed radius; the bound is here so
        // that a future radius cap being raised turns into a coarse grid rather than an infinite loop.
        for (int guard = 0; guard < 24 && gridBytes(radiusBlocks, cell) > MAX_BYTES; guard++) {
            cell <<= 1;
        }
        return cell;
    }

    /**
     * The camera block the window should stay centred on: {@code retainedAnchor} while it still covers
     * {@code camBlock}, and the camera itself once it does not. Sliding is what a whole-cell-aligned origin
     * cannot do continuously, so it is done rarely and in steps of the span rather than per frame — a window
     * that followed the camera exactly would renumber every cell as the player walked, which is a cache that
     * never accumulates.
     */
    public static int anchorFor(int retainedAnchor, int camBlock, int radiusBlocks, int cellBlocks) {
        int origin = Math.floorDiv(retainedAnchor - radiusBlocks, cellBlocks) * cellBlocks;
        long span = (long) cellsPerAxis(radiusBlocks, cellBlocks) * cellBlocks;
        // Half-open on purpose: the span is a whole number of cells and is at least 2*radius + cell, so the
        // extra cell the lattice rounds up to sits on the far side rather than being wasted on both.
        if (camBlock >= origin && (long) camBlock < (long) origin + span) {
            return retainedAnchor;
        }
        return camBlock;
    }

    /**
     * The window's minimum corner, in the shader's rebased block space — the space whose zero is the terrain
     * rebase origin, which is where every position a shader hands this arithmetic came from.
     *
     * <p>{@code rebaseBlock} is that origin's absolute block coordinate. Aligning the <em>absolute</em>
     * corner down to the lattice and then moving into shader space is the order that matters: doing it the
     * other way round would tie cell boundaries to wherever the rebase happened to land.
     */
    public static int windowOriginBlocks(int anchorBlock, int rebaseBlock, int radiusBlocks, int cellBlocks) {
        int alignedWorld = Math.floorDiv(anchorBlock - radiusBlocks, cellBlocks) * cellBlocks;
        return alignedWorld - rebaseBlock;
    }

    /**
     * Which cell an absolute world block falls in, as the shader computes it — from the shader's rebased
     * position and the pushed origin. Only a test needs it, and what it is for is the claim in the class
     * comment: the same world block must land in the same cell whatever the rebase position is.
     */
    public static int cellIndex(int worldBlock, int anchorBlock, int rebaseBlock,
                                int radiusBlocks, int cellBlocks) {
        int origin = windowOriginBlocks(anchorBlock, rebaseBlock, radiusBlocks, cellBlocks);
        return Math.floorDiv(worldBlock - rebaseBlock - origin, cellBlocks);
    }
}
