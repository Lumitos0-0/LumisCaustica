package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The arithmetic that decides where the ReSTIR GI grid is, which is the whole of what the grid is for at this
 * step: nothing reads a cell yet, so the properties worth pinning are the ones a cache cannot be built on top
 * of if they are wrong.
 */
final class RtGiGridTest {
    private static final int RADIUS = 64;
    private static final int CELL = 4;

    @Test
    void theWindowCoversTheRequestedRadiusOnBothSides() {
        // A 128-block-wide request becomes 33 cells of 4 blocks, which is 132 blocks: the cell count is
        // allowed to overshoot and never to undershoot, or the surfaces at the edge of the radius would have
        // no cell and the reach would be quietly shorter than the config says.
        assertEquals(33, RtGiGrid.cellsPerAxis(RADIUS, CELL));
        assertEquals(-64, RtGiGrid.windowOriginBlocks(0, 0, RADIUS, CELL));
        assertEquals(32, RtGiGrid.cellIndex(RADIUS, 0, 0, RADIUS, CELL));
        assertEquals(0, RtGiGrid.cellIndex(-RADIUS, 0, 0, RADIUS, CELL));
        // The box is [-64, 68), so the lattice's extra cell lands on the far side and index 33 is the first
        // one the grid does not have. anchorFor treats exactly that block as having left, which is the two
        // halves of the same rule agreeing rather than each picking an edge.
        assertEquals(33, RtGiGrid.cellIndex(RADIUS + CELL, 0, 0, RADIUS, CELL));
    }

    @Test
    void aRebaseRenumbersNothing() {
        // The property the design rests on. A rebase moves every position the shaders work in, so if the
        // pushed window origin moved with it rather than being anchored on the lattice, walking past the
        // rebase threshold would silently relabel every cached cell — invisible in a still frame, and a
        // cache that never converges for a player who moves.
        for (int rebase = -4096; rebase <= 4096; rebase += 97) {
            for (int world = rebase - RADIUS; world <= rebase + RADIUS; world += 13) {
                assertEquals(RtGiGrid.cellIndex(world, 0, 0, RADIUS, CELL),
                        RtGiGrid.cellIndex(world, 0, rebase, RADIUS, CELL),
                        "cell of world block " + world + " moved when the rebase was at " + rebase);
            }
        }
    }

    @Test
    void theAnchorStaysPutWhileTheCameraIsStillInsideAndSlidesToItWhenNot() {
        // The window covers [-64, 68) blocks around a retained anchor of 0, so a walk of a hundred blocks
        // does not move the grid out from under the cache; the moment it would, the anchor becomes the camera.
        assertEquals(0, RtGiGrid.anchorFor(0, 60, RADIUS, CELL));
        assertEquals(0, RtGiGrid.anchorFor(0, -64, RADIUS, CELL));
        // One block short of leaving is still inside: the box is half-open at the far end.
        assertEquals(0, RtGiGrid.anchorFor(0, 67, RADIUS, CELL));
        assertEquals(68, RtGiGrid.anchorFor(0, 68, RADIUS, CELL));
        assertEquals(-65, RtGiGrid.anchorFor(0, -65, RADIUS, CELL));
    }

    @Test
    void aSlideIsAlwaysAWholeNumberOfCells() {
        // Nothing has to know a slide happened, because a slide cannot split a cell: whatever the two anchors
        // are, the corners they produce differ by a multiple of the cell size.
        for (int a = -1000; a < 1000; a += 37) {
            for (int b = -1000; b < 1000; b += 53) {
                int delta = RtGiGrid.windowOriginBlocks(a, 0, RADIUS, CELL)
                        - RtGiGrid.windowOriginBlocks(b, 0, RADIUS, CELL);
                assertEquals(0, delta % CELL, "anchors " + a + " and " + b + " slid by " + delta + " blocks");
            }
        }
    }

    @Test
    void theMemoryBudgetCoarsensTheCellInsteadOfTheAllocation() {
        // The default settings fit, so the policy is invisible in the common case...
        assertEquals(CELL, RtGiGrid.cellBlocks(CELL, RADIUS));
        assertTrue(RtGiGrid.gridBytes(RADIUS, CELL) <= RtGiGrid.MAX_BYTES);
        // ...and an unreasonable one is absorbed by growing cells, not by a bigger allocation or a truncated
        // window. One-block cells over a 1024-block radius is a gigabyte of grid; the answer is coarser reuse.
        assertTrue(RtGiGrid.gridBytes(512, 16) > RtGiGrid.MAX_BYTES);
        int grown = RtGiGrid.cellBlocks(1, 512);
        assertTrue(grown > 1);
        assertTrue(RtGiGrid.gridBytes(512, grown) <= RtGiGrid.MAX_BYTES);
        assertTrue(RtGiGrid.gridBytes(512, grown >> 1) > RtGiGrid.MAX_BYTES);
    }

    @Test
    void recordsAreFortyEightBytesSoSixBinsPerCellIsWhatTheBudgetAssumes() {
        // The stride is reflected rather than declared, and every number above is a multiple of it: a record
        // that grew would silently grow the grid by whatever the budget lets it trade for.
        assertEquals(48, RtGiGrid.RECORD_BYTES);
        assertEquals(6, RtGiGrid.BINS);
        // The budget is spent on bins per cell, not on cells, and this is the one place that says so.
        assertEquals(1000L * RtGiGrid.BINS * RtGiGrid.RECORD_BYTES, RtGiGrid.gridBytesForCells(1000));
        assertEquals(RtGiGrid.gridBytes(RADIUS, CELL),
                RtGiGrid.gridBytesForCells(RtGiGrid.cellCount(RADIUS, CELL)));
    }
}
