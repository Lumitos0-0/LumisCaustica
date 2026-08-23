// Capture Pass A's first-interface depth at each stable froxel centre. This runs once per XY cell (not
// once per Z slice) and ping-pongs with scattering history for temporal disocclusion tests.

import world_common;
import world_core;
import bindings;
import froxel;

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 index = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    if (any(index >= dimensions)) {
        return;
    }

    float2 sampleCoordinate;
    float minDepth;
    float maxDepth;
    buildFroxelBackgroundDepth(int2(index), sampleCoordinate, minDepth, maxDepth);
    froxelDepthCurrent[int2(index)] = float4(minDepth, maxDepth, sampleCoordinate);
}
