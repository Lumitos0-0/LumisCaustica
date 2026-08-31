package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.rt.gen.DiReservoirRecordData;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The arithmetic the ReSTIR DI history is addressed and invalidated by. The shader side is not exercised
 * here — the point is that the numbers Java feeds it, and the word layout the debug pass reads raw, are the
 * numbers this class derives.
 */
final class RtRestirDiTest {
    @Test
    void dividesRenderExtentUpToCellsSoNoEdgePixelIsLeftWithoutACell() {
        assertEquals(640, RtRestirDi.historyExtent(1280, 2));
        assertEquals(360, RtRestirDi.historyExtent(720, 2));
        // An odd extent has to round up: a grid that is one cell short of the last column of pixels would
        // leave those pixels reusing a neighbour's reservoir forever instead of their own.
        assertEquals(1, RtRestirDi.historyExtent(1, 2));
        assertEquals(641, RtRestirDi.historyExtent(1281, 2));
        assertEquals(1280, RtRestirDi.historyExtent(1280, 1));
        assertEquals(320, RtRestirDi.historyExtent(1280, 4));
    }

    @Test
    void rejectsAnUndivisableHistoryExtentInsteadOfDividingByZero() {
        assertThrows(IllegalArgumentException.class, () -> RtRestirDi.historyExtent(1280, 0));
        assertThrows(IllegalArgumentException.class, () -> RtRestirDi.historyExtent(1280, -1));
        assertThrows(IllegalArgumentException.class, () -> RtRestirDi.historyExtent(-1, 1));
    }

    @Test
    void sizesTheHistoryFromTheCellCountAndRejectsAnImpossibleGrid() {
        // A quarter-resolution 1280x720 grid is 640x360 cells of 48 bytes: 11 MB per buffer, against the
        // 44 MB per buffer a per-pixel history at the same extent costs (and there are always two).
        assertEquals(640 * 360L * 48L, RtRestirDi.historyBytes(640 * 360));
        assertThrows(IllegalArgumentException.class, () -> RtRestirDi.historyBytes(-1));
    }

    @Test
    void tagsAreNeverZeroAndMoveWithEitherGenerationOrReset() {
        for (int resets = 0; resets < 4096; resets++) {
            for (long generation = 0; generation < 64; generation++) {
                assertNotEquals(0, RtRestirDi.generationTag(generation, resets));
                assertNotEquals(0, RtRestirDi.generationTag(-generation - 1, resets));
            }
        }
        // A reset has to invalidate a history even when the lights did not change, and vice versa: those
        // are the two independent reasons a stored index stops meaning what it meant.
        assertNotEquals(RtRestirDi.generationTag(7, 0), RtRestirDi.generationTag(7, 1));
        assertNotEquals(RtRestirDi.generationTag(7, 0), RtRestirDi.generationTag(8, 0));
        assertEquals(RtRestirDi.generationTag(7, 0), RtRestirDi.generationTag(7, 0));
    }

    /**
     * The debug present pass reads a history as raw words rather than as a struct, so the indices it
     * hard-codes have to be the offsets the reflected record actually puts those fields at.
     */
    @Test
    void debugViewWordMapMatchesTheReflectedRecordLayout() {
        ByteBuffer buffer = ByteBuffer.allocateDirect(RtRestirDi.RECORD_BYTES);
        new DiReservoirRecordData(1.5f, -2.25f, 300.0f, 0x11223344, 0x55667788, 0x99AABBCC,
                1.75f, 42.0f, 0.125f, 0x0BADF00D, 3.0f, 617).write(buffer);

        assertEquals(48, RtRestirDi.RECORD_BYTES);
        assertEquals(1.5f, buffer.getFloat(0), 0.0f);
        assertEquals(-2.25f, buffer.getFloat(Float.BYTES), 0.0f);
        assertEquals(300.0f, buffer.getFloat(2 * Float.BYTES), 0.0f);
        assertEquals(0x11223344, buffer.getInt(3 * Integer.BYTES)); // the oct-encoded normal, a plain half2
        assertEquals(0x55667788, buffer.getInt(RtRestirDi.LIGHT_INDEX_WORD * Integer.BYTES));
        assertEquals(1.75f, buffer.getFloat(RtRestirDi.WEIGHT_WORD * Integer.BYTES), 0.0f);
        assertEquals(42.0f, buffer.getFloat(RtRestirDi.M_WORD * Integer.BYTES), 0.0f);
        assertEquals(0.125f, buffer.getFloat(RtRestirDi.PHAT_WORD * Integer.BYTES), 0.0f);
        assertEquals(0x0BADF00D, buffer.getInt(RtRestirDi.GEN_TAG_WORD * Integer.BYTES));
        assertEquals(3.0f, buffer.getFloat(RtRestirDi.SPATIAL_WORD * Integer.BYTES), 0.0f);
        assertEquals(617, buffer.getInt(RtRestirDi.GI_CELL_WORD * Integer.BYTES));
    }

    /**
     * The ReSTIR GI cell a surface falls in rides in the DI record's last word, so the record the debug pass
     * reads has to stay 48 bytes with that word at 44 — the stride is reflected, and the word index is not.
     */
    @Test
    void theRadianceGridCellIsTheLastWordOfARecordThatHasNoRoomLeft() {
        assertEquals(RtRestirDi.RECORD_BYTES - Integer.BYTES, RtRestirDi.GI_CELL_WORD * Integer.BYTES);
        assertEquals(48, RtRestirDi.RECORD_BYTES);
    }

    @Test
    void roundTripsAStoredReservoirThroughTheGeneratedRecord() {
        ByteBuffer buffer = ByteBuffer.allocateDirect(RtRestirDi.RECORD_BYTES);
        DiReservoirRecordData written = new DiReservoirRecordData(0.0f, 0.0f, 0.0f, 0, 3, 0,
                2.0f, 8.0f, 0.5f, 1, 0.0f, 0);
        written.write(buffer);

        DiReservoirRecordData read = DiReservoirRecordData.read(buffer);
        assertEquals(written, read);
    }
}
