// Push-constant block shared by every volumetric pass. One struct for all six dispatches — each pass
// reads the subset it needs — so there is a single place to check when a dimension or a toggle changes,
// and one generated Java writer instead of six.
//
// Everything describing the MEDIUM lives in WorldPush.fog (see world_common.slang), reached through
// `worldPushAddr`. What is here is strictly per-dispatch geometry and control.

public struct VolumePush {
    public uint64_t worldPushAddr;
    // Froxel grid dimensions. x/y are the render resolution divided by the quality preset's scale
    // (rounded up), z is the slice count.
    public uint gridX;
    public uint gridY;
    public uint gridZ;
    public uint renderW;
    public uint renderH;
    public uint displayW;
    public uint displayH;
    // 0 rejects all temporal history this frame. Set on the first frame after the images are created,
    // after a resolution or quality change, and whenever the terrain rebase origin moves — the fog's
    // height profile is anchored in rebased world Y, so a rebase shift teleports the density field.
    public uint historyValid;
    // 0 skips the froxel sky-visibility ray, leaving the fog lit by the unoccluded sky everywhere.
    public uint skyOcclusion;
    // Temporal accumulation ceiling, in frames, for both the froxel grid and the shaft resolve.
    public uint maxAccumFrames;
    // Composite diagnostic selector; 0 composites normally. See volumetrics.slang VOLUME_DEBUG_*.
    public uint debugView;
};
