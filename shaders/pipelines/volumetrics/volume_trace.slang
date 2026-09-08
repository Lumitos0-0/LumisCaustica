// Inline visibility for the volumetric passes. These are compute dispatches, so they use ray query
// rather than the ray-tracing pipeline's shadow record — which means no any-hit shader runs and the
// alpha test has to be approximated here.
//
// The approximation is by geometry bucket, and it is a deliberate one. A terrain section's BLAS holds
// one geometry per bucket ({solid, cutout, translucent, water}, RtAccel.BUCKET_*), and only the solid
// bucket is flagged opaque, so everything else arrives as a candidate:
//
//   solid        never reported; the hardware commits it. Correct blocker.
//   cutout       committed here. With opacity micromaps on (the default) the hardware has already
//                discarded the fully-transparent micro-triangles of a leaf or a grass blade, so what
//                reaches this loop is the genuinely-covered part plus a thin "unknown" boundary —
//                close to the real alpha test. With micromaps off, foliage casts a solid quad.
//   translucent  skipped. Glass has to pass sunlight or the classic shaft-through-a-window is a
//                shadow instead. The tint it should impose is lost; the beam is slightly too neutral.
//   water        skipped, for the same reason plus the fact that water's own extinction is the path
//                tracer's job, not the fog's.
//
// Entity BLASes use their own two buckets (opaque, any-hit) at indices 0 and 1, so a non-opaque entity
// candidate lands outside the two skip cases and blocks — which is what an entity should do.
import volumetrics;
import bindings;

/** True if nothing between `origin` and `origin + dir * maxDistance` blocks the light. */
public bool volumeVisible(float3 origin, float3 dir, float maxDistance) {
    RayDesc ray;
    ray.Origin = origin;
    ray.TMin = 1.0e-3;
    ray.Direction = dir;
    ray.TMax = maxDistance;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, VOLUME_CULL_MASK, ray);
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            uint bucket = query.CandidateGeometryIndex();
            if (bucket != VOLUME_BUCKET_TRANSLUCENT && bucket != VOLUME_BUCKET_WATER) {
                query.CommitNonOpaqueTriangleHit();
            }
        }
    }
    return query.CommittedStatus() == COMMITTED_NOTHING;
}
