// ReSTIR DI: temporal reuse of the block-emitter direct-lighting reservoir.
// INDIRECT PASS ONLY, and only at the primary shading vertex. Like lighting.slang this is reached from
// world.rgen. The debug present pass reads the same records with its own decode rather than importing this
// module, because everything below needs the ray dispatch context to be meaningful.
//
// risInitial builds a ReSTIR DI initial reservoir — M candidates from the published light grid, weighted
// reservoir sampling, then exactly one shadow ray on the survivor in shadeReservoir. This module adds the
// other half of ReSTIR's temporal pass: the survivor is stored, reprojected onto the surface it was
// gathered on next frame, and merged back in. A primary vertex therefore accumulates a reservoir built from
// M candidates per frame for as long as that surface survives, which is what lets the configured candidate
// count act as a burn-in length rather than as the per-frame coverage of the light list.
//
// Deliberate limits of this implementation, each for a reason:
//  - primary vertex only. A reservoir is indexed by screen position, so reprojecting a reservoir that was
//    gathered at the end of a reflection would need that surface to be visible in the previous frame's
//    depth, and Pass A does not write one for what a reflected ray hits.
//  - temporal only, no spatial round. The indirect raygen is register-pressure-bound (see its header),
//    and neighbours' reservoirs read inline would add that pressure to every lane, coherently or not.
//    A spatial round belongs in a compute pass over the packed records, which is also where it gets to
//    run at history resolution.
//  - only the survivor is stored, and no visibility. The record is the reservoir, not the answer; the
//    one shadow ray is still cast against it by shadeReservoir every frame, so a light that becomes
//    occluded stops contributing the frame after it moves rather than leaking through a stale sample.
//  - no frame-gap token. The swap happens once per recorded frame, and a frame the renderer skipped leaves
//    the surface where it was, which is what the position and normal comparison is for; carrying a frame
//    index into the record would only re-derive that conclusion more expensively.
//
// With SPP above 1 every sample of a pixel reads the same stored reservoir, so the samples are correlated
// through the history rather than independent. That is the same trade the guide buffers already make —
// they are per pixel too — and the reason SPP stays at its default of 1 with DLSS-RR accumulating.
//
// The read buffer is last frame's write buffer. Two buffers rather than one means a lane can never read a
// cell that another lane is storing in the same dispatch, so reuse stays deterministic frame to frame.

import world_common;
import world_core;
import bindings;
import lighting;
import math;
import segment;

// A stored reservoir is usable when it names a real light of the CURRENT published generation. genTag
// changes with the light grid's published generation and with every history reset, so a rebase, a
// republish, a resize or an F3 invalidation invalidates every record at once without a clear pass.
public bool diRecordUsable(DiReservoirRecord rec, float3 hitPos, float3 n) {
    if (rec.genTag != worldPush.diGenTag) {
        return false;
    }
    if (rec.lightIndex >= worldPush.lightCount || rec.m <= 0.0 || rec.weight <= 0.0) {
        return false;
    }
    // The record was gathered on a surface, and the only guarantee temporal reuse gets is that the surface
    // is still there. Position and normal are what a screen-space index can actually verify; a moved wall,
    // a placed block and a camera that crossed a block boundary all show up here as a mismatch.
    float3 storedPos = float3(rec.hitX, rec.hitY, rec.hitZ);
    if (distance(storedPos, hitPos) > worldPush.diParams.z) {
        return false;
    }
    return dot(n, octDecode(unpackHalf2(rec.normalOct))) >= worldPush.diValidate.x;
}

// One history cell covers divisor x divisor render pixels, and only its anchor pixel stores. Every cell
// therefore has exactly one writer per frame, which is what lets the whole pass stay lock-free.
public uint diHistoryDivisor() {
    // Floored rather than trusted: this divides an index, and a zero from a push slot written before the
    // history was allocated would be a divide-by-zero in the middle of the trace.
    return max(1u, uint(worldPush.diParams.x));
}

public bool diIsHistoryAnchor(uint2 pix) {
    uint d = diHistoryDivisor();
    return pix.x % d == 0u && pix.y % d == 0u;
}

public uint2 diHistoryCell(uint2 pix) {
    return pix / diHistoryDivisor();
}

public uint diHistoryIndex(uint2 cell) {
    return cell.y * worldPush.diHistoryWidth + cell.x;
}

// Where this surface was on the previous frame's screen. The stored history cell is a screen index, so a
// moving camera has to be undone before it can be read; the same prevViewProj the DLSS-RR guides use is
// what makes that index mean the same surface rather than the same pixel.
public uint2 diReprojectedHistoryCell(float3 hitPos, out bool valid) {
    valid = false;
    float4 clip = mul(worldPush.prevViewProj, float4(hitPos + worldPush.camDelta, 1.0));
    if (clip.w <= 0.0) {
        return uint2(0u, 0u); // behind the previous camera: the divide would invent a huge NDC
    }
    float2 size = float2(DispatchRaysDimensions().xy);
    // No y flip: Slang's Vulkan clip space and the history index share a y-down convention, so the same
    // [0,1] names the same cell on both frames.
    float2 uv = clip.xy / clip.w * 0.5 + 0.5;
    if (any(lessThan(uv, float2(0.0, 0.0))) || any(greaterEqual(uv, float2(1.0, 1.0)))) {
        return uint2(0u, 0u); // off screen last frame — a fresh view, nothing was ever stored here
    }
    uint2 prevPix = uint2(uv * size);
    valid = true;
    return diHistoryCell(prevPix);
}

// Merge this frame's initial reservoir with the reprojected previous one. Returns the reservoir to shade.
// Nothing here reads the shadow map or traces, so the survivor selection is exactly the selection the
// single following shadow ray will be applied to.
public Reservoir restirDiTemporal(Reservoir cur, uint2 pix, float3 hitPos, float3 n, float3 v, float3 rd,
                                  float3 diffAlb, float3 F0, float rough, float sss, inout uint seed) {
    // One gate, on the addresses, and before anything else this pass could perturb: with reuse off the
    // reservoir returned is the one risInitial built, and no RNG has been drawn. Both are checked even
    // though Java sets them together, because a half-wired pair is otherwise a read of address zero.
    if (pc.diHistoryReadAddr == 0 || pc.diHistoryWriteAddr == 0) {
        return cur;
    }

    bool hasHistory;
    uint2 cell = diReprojectedHistoryCell(hitPos, hasHistory);
    if (hasHistory) {
        uint2 dims = uint2(worldPush.diHistoryWidth, worldPush.diHistoryHeight);
        hasHistory = cell.x < dims.x && cell.y < dims.y;
    }
    if (hasHistory) {
        DevicePtr<DiReservoirRecord> history = DevicePtr<DiReservoirRecord>(pc.diHistoryReadAddr);
        DiReservoirRecord rec = history[diHistoryIndex(cell)];
        if (diRecordUsable(rec, hitPos, n)) {
            Light lg = ConstPtr<Light>(pc.lightBufAddr)[rec.lightIndex];
            float2 st = unpackHalf2(rec.packedST);
            float3 sp;
            float3 lnrm;
            float3 le;
            float area;
            evalLightSample(lg, st.x, st.y, sp, lnrm, le, area);
            float phat = reservoirPhat(sp, lnrm, le, area, hitPos, n, v, rd, diffAlb, F0, rough,
                    false, sss);
            // The stored W already carries the previous frame's p-hat at the previous surface. Rebuilding
            // the accumulated weight with THIS frame's p-hat is what makes the merge an importance-resampled
            // combination of two frames rather than a comparison of two differently-scaled quantities.
            float mHist = rec.m;
            float wSumHist = rec.weight * mHist * phat;
            if (phat > 0.0 && wSumHist > 0.0) {
                float cap = worldPush.diParams.y;
                // Cap the inherited count and rescale its weight by the same factor: without a ceiling the
                // history dominates permanently, a light that stops being visible keeps lighting the
                // surface for as long as the surface exists, and the reservoir stops responding to
                // anything new. Capping bounds that lag at ~cap frames while still killing most of the
                // per-frame variance.
                if (cap > 0.0 && cur.M + mHist > cap) {
                    float allowed = max(0.0, cap - cur.M);
                    wSumHist *= allowed / mHist;
                    mHist = allowed;
                }
                if (mHist > 0.0) {
                    float wSum = cur.wSum + wSumHist;
                    float m = cur.M + mHist;
                    if (rndf(seed) * wSum < wSumHist) {
                        cur.pos = sp;
                        cur.lnrm = lnrm;
                        cur.le = le;
                        cur.area = area;
                        cur.phat = phat;
                        cur.lightIndex = rec.lightIndex;
                        cur.s = st.x;
                        cur.t = st.y;
                    }
                    cur.wSum = wSum;
                    cur.M = m;
                    cur.W = cur.phat > 0.0 ? cur.wSum / (cur.M * cur.phat) : 0.0;
                }
            }
        }
    }

    // Store for next frame from the pixel that owns this cell. This is the same reservoir the survivor is
    // shaded from, minus the shadow ray — reuse must never inherit visibility. The bound matters: Java may
    // have clamped the allocation to a smaller grid than this render target divides to, and a write past it
    // would land outside the buffer rather than in an unused cell.
    uint2 ownCell = diHistoryCell(pix);
    bool canStore = ownCell.x < worldPush.diHistoryWidth && ownCell.y < worldPush.diHistoryHeight;
    if (canStore && diIsHistoryAnchor(pix) && cur.lightIndex != DI_NO_LIGHT) {
        DevicePtr<DiReservoirRecord> out = DevicePtr<DiReservoirRecord>(pc.diHistoryWriteAddr);
        DiReservoirRecord rec;
        rec.hitX = hitPos.x;
        rec.hitY = hitPos.y;
        rec.hitZ = hitPos.z;
        rec.normalOct = packHalf2(octEncode(n));
        rec.lightIndex = cur.lightIndex;
        rec.packedST = packHalf2(float2(cur.s, cur.t));
        rec.weight = cur.W;
        rec.m = cur.M;
        rec.phat = cur.phat;
        rec.genTag = worldPush.diGenTag;
        rec.pad0 = 0.0;
        rec.pad1 = 0.0;
        out[diHistoryIndex(ownCell)] = rec;
    }
    return cur;
}
