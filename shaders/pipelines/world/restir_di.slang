// ReSTIR DI: temporal and spatial reuse of the block-emitter direct-lighting reservoir.
// INDIRECT PASS ONLY, and only at the primary shading vertex. Like lighting.slang this is reached from
// world.rgen. The debug present pass reads the same records with its own decode rather than importing this
// module, because everything below needs the ray dispatch context to be meaningful.
//
// risInitial builds a ReSTIR DI initial reservoir — M candidates from the published light grid, weighted
// reservoir sampling, then exactly one shadow ray on the survivor in shadeReservoir. This module adds the
// two reuse rounds that make that reservoir worth more than one frame: the survivor is stored, found again
// on the same surface next frame (temporal), and combined with what the surrounding surfaces settled on
// (spatial). A primary vertex therefore accumulates a reservoir built from M candidates per frame for as
// long as that surface survives, which is what lets the configured candidate count act as a burn-in length
// rather than as the per-frame coverage of the light list.
//
// Both rounds read the previous frame's buffer and write this frame's. Nothing here traces a ray: borrowed
// samples are re-scored against this frame's surface and the single shadow ray in shadeReservoir is still
// what decides visibility, so a light that becomes occluded stops contributing the frame after it moves
// instead of leaking through a stored answer.
//
// Deliberate limits of this implementation, each for a reason:
//  - primary vertex only. A reservoir is indexed by screen position, so reprojecting a reservoir that was
//    gathered at the end of a reflection would need that surface to be visible in the previous frame's
//    depth, and Pass A does not write one for what a reflected ray hits. Bounce vertices get a separate,
//    independently configurable candidate budget instead (see risBudget in world.rgen).
//  - the spatial round is biased, in the way ReSTIR DI's spatial round is biased: a borrowed sample is
//    weighted by the uniform area it was drawn from rather than by a combined PDF with the light proposal
//    that produced it, and it is not visibility-tested against this surface. Both are the documented trade
//    for one extra lookup per round; setting the round count to 0 removes the round and the bias with it.
//  - no frame-gap token. The swap happens once per recorded frame, and a frame the renderer skipped leaves
//    the surface where it was, which is what the position and normal comparison is for; carrying a frame
//    index into the record would only re-derive that conclusion more expensively.
//
// With SPP above 1 every sample of a pixel reads the same stored reservoir, so the samples are correlated
// through the history rather than independent. That is the same trade the guide buffers already make — they
// are per pixel too — and the reason SPP stays at its default of 1 with DLSS-RR accumulating.

import world_common;
import world_core;
import bindings;
import lighting;
import math;
import segment;

public uint diHistoryDivisor() {
    // Floored rather than trusted: this divides an index, and a zero from a push slot written before the
    // history was allocated would be a divide-by-zero in the middle of the trace.
    return max(1u, uint(worldPush.diParams.x));
}

// One history cell covers divisor x divisor render pixels, and only its anchor pixel stores. Every cell
// therefore has exactly one writer per frame, which is what lets the whole pass stay lock-free.
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

public bool diHistoryHasCell(uint2 cell) {
    return cell.x < worldPush.diHistoryWidth && cell.y < worldPush.diHistoryHeight;
}

// Every reuse path starts here: the record has to describe a real light of the CURRENT published
// generation. genTag moves with the light grid's published generation and with every history reset, so a
// republish, a resize or an F3 invalidation invalidates every stored index at once without a clear pass.
// The two scalars are checked for the same reason a fresh buffer's contents are not trusted on faith:
// a reservoir with no weight or no samples is not an answer, whatever tag it carries.
public bool diRecordLive(DiReservoirRecord rec) {
    return rec.genTag == worldPush.diGenTag && rec.lightIndex < worldPush.lightCount
            && rec.m > 0.0 && rec.weight > 0.0;
}

// Temporal validation: the record has to describe THIS surface, because the reservoir being merged carries
// an accumulated history that only means something if the surface it accumulated on is still there. A
// moved wall, a placed block and a camera that crossed a block boundary (which re-bases the stored
// position) all show up as a mismatch.
public bool diSurfaceMatchesTemporal(DiReservoirRecord rec, float3 hitPos, float3 n) {
    float3 storedPos = float3(rec.hitX, rec.hitY, rec.hitZ);
    if (distance(storedPos, hitPos) > worldPush.diParams.z) {
        return false;
    }
    return dot(n, octDecode(unpackHalf2(rec.normalOct))) >= worldPush.diValidate.x;
}

// Spatial validation: the neighbour is a DIFFERENT surface, so requiring its position to match is both
// wrong (it is laterally displaced by design) and the only leak guard that matters — how far apart the two
// surfaces are ALONG THE NORMAL. Two pixels on one wall are coplanar however far apart they are on screen;
// a pixel across a corner is not, and that is the case that would smear light through geometry.
public bool diSurfaceMatchesSpatial(DiReservoirRecord rec, float3 hitPos, float3 n) {
    float3 storedPos = float3(rec.hitX, rec.hitY, rec.hitZ);
    if (abs(dot(hitPos - storedPos, n)) > worldPush.diParams.z) {
        return false;
    }
    return dot(n, octDecode(unpackHalf2(rec.normalOct))) >= worldPush.diValidate.x;
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

// Read one stored reservoir and re-score it at this surface. Both rounds borrow a (light, emitter point)
// pair rather than an answer, so everything comes back through the light table and the target function.
// Returns false when there is nothing to borrow; phat is the neighbour's value at THIS surface, which is
// the only quantity the combine needs from it.
public bool diBorrowSample(DiReservoirRecord rec, float3 hitPos, float3 n, float3 v, float3 rd,
                           float3 diffAlb, float3 F0, float rough, float sss,
                           out float3 sp, out float3 lnrm, out float3 le, out float area, out float phat) {
    Light lg = ConstPtr<Light>(pc.lightBufAddr)[rec.lightIndex];
    float2 st = unpackHalf2(rec.packedST);
    evalLightSample(lg, st.x, st.y, sp, lnrm, le, area);
    phat = reservoirPhat(sp, lnrm, le, area, hitPos, n, v, rd, diffAlb, F0, rough, false, sss);
    return phat > 0.0;
}

// Weighted-reservoir update for one incoming candidate, in the form both rounds share. The survivor fields
// are the reservoir's whole payload, so adopting means replacing all of them together.
public void diOfferCandidate(inout Reservoir r, float w, uint lightIndex, float s, float t,
                             float3 sp, float3 lnrm, float3 le, float area, float phat,
                             float mIncoming, inout uint seed) {
    r.wSum += w;
    r.M += mIncoming;
    if (rndf(seed) * r.wSum < w) {
        r.pos = sp;
        r.lnrm = lnrm;
        r.le = le;
        r.area = area;
        r.phat = phat;
        r.lightIndex = lightIndex;
        r.s = s;
        r.t = t;
    }
    r.W = r.phat > 0.0 ? (r.wSum / (r.M * r.phat)) : 0.0;
}

// Temporal round: merge the reprojected previous reservoir. Its stored W already carries the previous
// frame's p-hat at the previous surface, so the accumulated weight is rebuilt with THIS frame's p-hat —
// that is what turns two frames' reservoirs into one importance-resampled estimate instead of a
// comparison of two differently-scaled quantities.
public void diTemporalRound(inout Reservoir cur, float3 hitPos, float3 n, float3 v, float3 rd,
                            float3 diffAlb, float3 F0, float rough, float sss, inout uint seed) {
    bool hasHistory;
    uint2 cell = diReprojectedHistoryCell(hitPos, hasHistory);
    hasHistory = hasHistory && diHistoryHasCell(cell);
    if (!hasHistory) {
        return;
    }
    DevicePtr<DiReservoirRecord> history = DevicePtr<DiReservoirRecord>(pc.diHistoryReadAddr);
    DiReservoirRecord rec = history[diHistoryIndex(cell)];
    if (!diRecordLive(rec) || !diSurfaceMatchesTemporal(rec, hitPos, n)) {
        return;
    }
    float3 sp;
    float3 lnrm;
    float3 le;
    float area;
    float phat;
    if (!diBorrowSample(rec, hitPos, n, v, rd, diffAlb, F0, rough, sss, sp, lnrm, le, area, phat)) {
        return;
    }
    float mHist = rec.m;
    float wSumHist = rec.weight * mHist * phat;
    float cap = worldPush.diParams.y;
    // Cap the inherited count and rescale its weight by the same factor: without a ceiling the history
    // dominates permanently, a light that stops being visible keeps lighting the surface for as long as
    // the surface exists, and the reservoir stops responding to anything new. Capping bounds that lag at
    // ~cap frames while still killing most of the per-frame variance.
    if (cap > 0.0 && cur.M + mHist > cap) {
        float allowed = max(0.0, cap - cur.M);
        wSumHist *= allowed / mHist;
        mHist = allowed;
    }
    if (mHist <= 0.0) {
        return;
    }
    float2 st = unpackHalf2(rec.packedST);
    diOfferCandidate(cur, wSumHist, rec.lightIndex, st.x, st.y, sp, lnrm, le, area, phat, mHist, seed);
}

// Spatial round: combine this frame's reservoir with what the surrounding cells settled on. Reading
// last frame's records is what makes this legal mid-dispatch — the read buffer has no writer this frame, so
// gathering arbitrary neighbours cannot race, and no reservoir pass ahead of shading is needed.
//
// Each neighbour enters as ONE sample (m = 1) even though it may itself carry a long history: counting that
// depth a second time would let one frame's survivor dominate the frame through every cell that borrowed it.
// Its weight is p-hat divided by the uniform proposal density of the sampled square, which is the standard
// ReSTIR DI formulation and the source of this round's documented bias.
public void diSpatialRound(inout Reservoir cur, uint2 pix, float3 hitPos, float3 n, float3 v, float3 rd,
                           float3 diffAlb, float3 F0, float rough, float sss, inout uint seed,
                           out uint accepted) {
    accepted = 0u;
    uint rounds = uint(max(0.0, worldPush.diValidate.y));
    if (rounds == 0u) {
        return; // no RNG consumed and no loads issued, so a rounds-free frame is the frame from before this
    }
    float radius = max(1.0, worldPush.diValidate.z);
    float support = (2.0 * radius + 1.0) * (2.0 * radius + 1.0); // cells the neighbours are drawn uniformly from
    uint2 centre = diHistoryCell(pix);
    DevicePtr<DiReservoirRecord> history = DevicePtr<DiReservoirRecord>(pc.diHistoryReadAddr);
    for (uint i = 0u; i < rounds; i++) {
        int2 offset = int2(floor((float2(rndf(seed), rndf(seed)) * 2.0 - 1.0) * radius));
        int2 bounded = clamp(int2(centre) + offset, int2(0, 0),
                int2(worldPush.diHistoryWidth, worldPush.diHistoryHeight) - 1);
        if (uint2(bounded) == centre) {
            continue; // the centre is already in the reservoir; offering it twice would count it twice
        }
        uint2 cell = uint2(bounded);
        DiReservoirRecord rec = history[diHistoryIndex(cell)];
        if (!diRecordLive(rec) || !diSurfaceMatchesSpatial(rec, hitPos, n)) {
            continue;
        }
        float3 sp;
        float3 lnrm;
        float3 le;
        float area;
        float phat;
        if (!diBorrowSample(rec, hitPos, n, v, rd, diffAlb, F0, rough, sss, sp, lnrm, le, area, phat)) {
            continue; // this surface does not see what the neighbour sees: nothing to borrow
        }
        float2 st = unpackHalf2(rec.packedST);
        diOfferCandidate(cur, phat * support, rec.lightIndex, st.x, st.y,
                sp, lnrm, le, area, phat, 1.0, seed);
        accepted++;
    }
}

// Both rounds, then the store. Returns the reservoir to shade.
public Reservoir restirDiResample(Reservoir cur, uint2 pix, float3 hitPos, float3 n, float3 v, float3 rd,
                                  float3 diffAlb, float3 F0, float rough, float sss, inout uint seed) {
    // One gate, on the addresses, and before anything else this pass could perturb: with reuse off the
    // reservoir returned is the one risInitial built, and no RNG has been drawn. Both are checked even
    // though Java sets them together, because a half-wired pair is otherwise a read of address zero.
    if (pc.diHistoryReadAddr == 0 || pc.diHistoryWriteAddr == 0) {
        return cur;
    }

    diTemporalRound(cur, hitPos, n, v, rd, diffAlb, F0, rough, sss, seed);
    uint accepted;
    diSpatialRound(cur, pix, hitPos, n, v, rd, diffAlb, F0, rough, sss, seed, accepted);

    // Store for next frame from the pixel that owns this cell: the reservoir the survivor is shaded from,
    // minus the shadow ray. The bounds matter, not just the anchor test — Java may have clamped the
    // allocation to a smaller grid than this render target divides to, and a write past it would land
    // outside the buffer rather than in an unused cell.
    uint2 ownCell = diHistoryCell(pix);
    if (diHistoryHasCell(ownCell) && diIsHistoryAnchor(pix) && cur.lightIndex != DI_NO_LIGHT) {
        DevicePtr<DiReservoirRecord> outHistory = DevicePtr<DiReservoirRecord>(pc.diHistoryWriteAddr);
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
        rec.spatial = float(accepted);
        rec.pad1 = 0.0;
        outHistory[diHistoryIndex(ownCell)] = rec;
    }
    return cur;
}
