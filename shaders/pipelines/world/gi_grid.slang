// ReSTIR GI: addressing for the world-space radiance grid.
//
// A cell answers "what light is arriving in this neighbourhood of the world from that general direction".
// It is the world-space counterpart of the screen-space history in restir_di.slang, and it exists because a
// bounce vertex has no pixel to hang reuse state in: the only handle a secondary vertex has on its own
// location is where in the world it is.
//
// The grid is a fixed dense box, published by Java, in the same rebased block space every position in this
// shader already lives in. That gives the addressing three properties worth naming, because each is a bug
// class a world-space cache can otherwise have:
//  - One subtraction and a floor divide to index. No transform, no matrix, nothing to drift.
//  - The window's minimum corner is aligned down to the cell lattice before it is pushed, and the anchor it
//    is measured from is retained between frames rather than re-centred on the camera every time the player
//    moves, so a cell boundary falls on the same world block and a cell keeps its index. A rebase moves every shader-space position by a whole number of blocks
//    and moves the pushed origin by the same amount, which cancels: the rebase does not renumber the grid, so
//    cached radiance does not have to be thrown away because the player crossed the distance that rebases.
//  - Java reports the cell size it settled on rather than the one the config asked for, and the memory
//    budget may have made it larger. The shader divides by what it is told, so the two can never disagree
//    about where a boundary is; the cost of a clipped grid is coarser reuse, never wrong reuse.
//
// What is deliberately NOT here yet: the direction bins a cell is subdivided into, the visibility of a
// stored radiance against the surface asking for it, and the validation frames that bound how long a wrong
// answer can survive. Each lands with the first code that has to answer it, so nothing here is a parameter
// nothing reads.
import world_common;
import world_core;   // the per-thread `worldPush` every function below reads is world_core's global

// No cell for this position: the grid is off, or the surface is outside its window. Being outside is a
// normal answer rather than an error — the window is a box around wherever the camera was when it last had
// to slide, so a surface the player has walked away from simply has no cached radiance and the caller falls
// back to tracing. Losing reach is the cost of a bounded allocation; wrong radiance would be the alternative.
public static const uint GI_NO_CELL = 0xffffffffu;

// Whether a grid was published this frame. A zero extent is the only off switch there is: nothing here has a
// second condition to agree with, so Java turning the feature off cannot leave a half-populated grid behind.
public bool giGridActive() {
    return worldPush.giDims.x > 0 && worldPush.giDims.y > 0 && worldPush.giDims.z > 0
            && worldPush.giGrid.w > 0.0;
}

// Cell coordinate of a shading position in rebased blocks, or false when there is no cell to speak of. All
// six comparisons are load-bearing: the window is a box, so a position outside it on *any* axis — low or
// high — has no cell, and dropping the low side would index a neighbour's cell instead of reporting none.
public bool giCellCoord(float3 p, out int3 coord) {
    coord = int3(0, 0, 0);
    if (!giGridActive()) {
        return false;
    }
    int3 c = int3(floor((p - worldPush.giGrid.xyz) / worldPush.giGrid.w));
    if (c.x < 0 || c.y < 0 || c.z < 0
            || c.x >= worldPush.giDims.x || c.y >= worldPush.giDims.y || c.z >= worldPush.giDims.z) {
        return false;
    }
    coord = c;
    return true;
}

// Linear cell index, in the axis order findLightGridCell already uses (x fastest, then y, then z) so the
// light grid and the radiance grid can be read side by side in a debug view without one of them being
// transposed in your head.
public uint giCellIndex(float3 p) {
    int3 c;
    if (!giCellCoord(p, c)) {
        return GI_NO_CELL;
    }
    return (uint(c.z) * uint(worldPush.giDims.y) + uint(c.y)) * uint(worldPush.giDims.x) + uint(c.x);
}
