// Froxel lighting pass: raygen record 2 of the world pipeline. One invocation per froxel cell of the
// camera-frustum volume; writes the single-scatter source term (in-scatter radiance per unit distance in
// rgb, the froxel's extinction in alpha) into the froxel radiance cache input.
//
// This is a raygen rather than a compute pass on purpose: the visibility test that carves shafts out of
// the medium is a real traversal of the TLAS, and only the ray-tracing pipeline can issue one. Running
// it as a third record of the world pipeline reuses the entire SBT, so a froxel's sun ray alpha-tests
// leaf canopies through the same any-hit path that shadows surfaces — which is what makes beam edges
// follow foliage instead of block borders.
//
// Dispatched as (froxelWidth, froxelHeight * sliceCount): the lateral axes tile the trace frustum and
// the packed Y axis carries the depth slices, because the shared trace entry point dispatches 2D.
//
// Lighting is the dominant celestial body (the same derived light that shades terrain, so fog colour and
// surface colour cannot disagree) gated by one visibility ray, plus — when enabled — block emitters
// resampled from the same light-grid proposal machinery surfaces use, shaded with one further visibility
// ray. Two traced rays per froxel per frame is all this pass costs; the temporal filter in pipelines/fog
// converges the residual variance across frames.
import world_common;
import world_core;
import bindings;
import math;
import trace;
import lighting;
import sky;
import fog_common;

// Luminance ceiling for one froxel's source term. A bare emitter inside fog can otherwise write a
// single froxel at emission strength, which the temporal cache then smears across frames as a crawling
// bright blob; clamping per froxel keeps that to a local, stationary glow.
static const float FROXEL_FIREFLY_LUMINANCE = 4000.0;

float3 clampFroxelFirefly(float3 source) {
    float lum = luminance(source);
    return lum > FROXEL_FIREFLY_LUMINANCE ? source * (FROXEL_FIREFLY_LUMINANCE / lum) : source;
}

// Emitter in-scatter at a point in the medium, as initial RIS over M power-weighted candidates followed
// by ONE visibility ray for the survivor — the same shape as the surface path's risInitial/shadeReservoir
// pair. A volume has no normal, so the target pdf replaces the BRDF's cosine term with the phase function
// along the sample-to-light direction and keeps the emitter's geometry term: every emitter face oriented
// toward the sample contributes.
//
// One ray per froxel rather than one per candidate is what keeps the volume affordable: the grid is
// hundreds of thousands of froxels, and the temporal filter converges the residual variance of a single
// resampled sample per frame. The estimator carries exactly the bias the surface NEE already carries
// (visibility is evaluated for the survivor only, not per candidate).
float3 emitterInScatter(float3 p, float sigmaS, float3 viewDir, inout uint seed,
                        inout uint proposalSeed) {
    uint candidateCount = max(uint(worldPush.fogScatter.z), 1u);
    LightGridCell gridCell;
    int3 gridCellCoord;
    float3 gridLookup = p;
    if (pc.lightGridCellAddr != 0) {
        // Stochastically blend across hard section boundaries, as the surface path does: every cell
        // proposal retains full global support, so conditioning on the jittered lookup stays unbiased.
        gridLookup += (float3(rndf(proposalSeed), rndf(proposalSeed), rndf(proposalSeed)) - 0.5)
                * worldPush.lightGridOrigin.w;
    }
    bool hasGridCell = findLightGridCell(gridLookup, gridCell, gridCellCoord);
    // Same deterministic proposal stratification as risInitial: at M=8 that is six local and two global
    // candidates, with every lane taking the same branch for a given candidate index.
    uint globalCandidateCount = hasGridCell
            ? max(1u, (candidateCount + 2u) / 4u) : candidateCount;
    uint localCandidateCount = candidateCount - globalCandidateCount;
    float localProbability = float(localCandidateCount) / float(candidateCount);

    float weightSum = 0.0;
    float selectedPhat = 0.0;
    float3 selectedPos = float3(0.0, 0.0, 0.0);
    float3 selectedNormal = float3(0.0, 0.0, 0.0);
    float3 selectedRadiance = float3(0.0, 0.0, 0.0);
    float selectedArea = 0.0;
    for (uint c = 0u; c < candidateCount; c++) {
        uint li;
        uint globalsBefore = (c * globalCandidateCount) / candidateCount;
        uint globalsAfter = ((c + 1u) * globalCandidateCount) / candidateCount;
        bool useLocal = hasGridCell && globalsAfter == globalsBefore;
        selectLightGridLight(gridCell, useLocal, proposalSeed, li);
        Light lg = ConstPtr<Light>(pc.lightBufAddr)[li];
        // Uniform point on the emitter rectangle ((s,t) in [-1,1]^2, pdf 1/area).
        float s = rndf(seed) * 2.0 - 1.0;
        float t = rndf(seed) * 2.0 - 1.0;
        float3 sp = lg.pos + worldPush.lightRebase.xyz
                + s * lightHalfU(lg) + t * lightHalfV(lg);
        float3 toL = sp - p;
        float dist2 = dot(toL, toL);
        if (dist2 < 1.0e-6) {
            continue;
        }
        float3 wi = toL / sqrt(dist2);
        float cosL = max(0.0, dot(lightGeometricNormal(lg), -wi));
        if (cosL <= 0.0) {
            continue;
        }
        float3 le = lightRadiance(lg);
        float area = lightArea(lg);
        float phat = fogPhase(worldPush, dot(viewDir, wi)) * luminance(le) * (cosL * area / dist2);
        if (phat <= 0.0) {
            continue;
        }
        float sourcePdf = proposalPdf(lg, le, gridCell, gridCellCoord, localProbability);
        float w = phat / max(sourcePdf, 1.0e-20);
        weightSum += w;
        if (rndf(seed) * weightSum < w) { // weighted reservoir update
            selectedPos = sp;
            selectedNormal = lightGeometricNormal(lg);
            selectedRadiance = le;
            selectedArea = area;
            selectedPhat = phat;
        }
    }
    float W = selectedPhat > 0.0
            ? weightSum / (float(candidateCount) * selectedPhat) : 0.0;
    if (W <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    float3 toL = selectedPos - p;
    float dist = length(toL);
    float3 wi = toL / max(dist, 1.0e-6);
    float cosL = max(0.0, dot(selectedNormal, -wi));
    float3 contrib = sigmaS * fogPhase(worldPush, dot(viewDir, wi)) * selectedRadiance
            * (cosL * selectedArea / max(dist * dist, 1.0e-6));
    // Stop just short of the sample point so the ray cannot self-occlude on the emitter's own face.
    float3 vis = visibility(p + wi * SURF_BIAS, wi, dist * 0.999).transmittance;
    return contrib * vis * W;
}

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    // One derived directional light per dispatch, exactly as the indirect pass derives it: the same
    // transmittance LUT tints fog and terrain, so a sunset's fog cannot disagree with its surfaces.
    celestialLight = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);
    uint3 index = DispatchRaysIndex();
    uint3 dims = DispatchRaysDimensions();
    uint slices = max(uint(worldPush.fogGrid.z), 1u);
    uint froxelHeight = max(dims.y / slices, 1u);
    uint slice = index.y / froxelHeight;
    uint fy = index.y - slice * froxelHeight;

    uint2 froxelSize = uint2(dims.x, froxelHeight);
    float2 uv = (float2(index.x, fy) + 0.5) / float2(froxelSize);
    float u = (float(slice) + 0.5) / float(slices);
    FogRay ray = fogCameraRay(worldPush, uv);
    float3 p = ray.origin + ray.dir * fogSliceDistance(worldPush, u);

    float4 outValue = float4(0.0, 0.0, 0.0, 0.0);
    if (fogEnabled(worldPush)) {
        float sigmaT = fogDensityAt(worldPush, p);
        if (sigmaT > 1.0e-6) {
            float sigmaS = sigmaT * clamp(worldPush.fogScatter.x, 0.0, 1.0);
            uint seed = (index.x * 1973u + fy * 9277u + slice * 26699u)
                    ^ (worldPush.frameIndex * 2654435761u);
            seed = pcg(seed);
            uint proposalSeed = pcg(seed ^ 0x9e3779b9u);

            // The sun/moon term: a directional light's in-scatter is sigma_s * phase * illuminance,
            // gated by one visibility ray. Jittering the light direction within its angular extent is
            // unnecessary here — the disc is tiny and the temporal pass integrates what jitter buys.
            float3 sunVis = visibility(p + celestialLight.dir * SURF_BIAS,
                                       celestialLight.dir, 10000.0).transmittance;
            float3 source = sigmaS * fogPhase(worldPush, dot(ray.dir, celestialLight.dir))
                    * celestialLight.illuminance * sunVis;

            if (worldPush.fogScatter.z > 0.0 && worldPush.lightCount > 0u && pc.lightBufAddr != 0) {
                source += emitterInScatter(p, sigmaS, ray.dir, seed, proposalSeed);
            }
            outValue = float4(clampFroxelFirefly(source), sigmaT);
        }
    }
    froxelRaw[int3(index.x, fy, slice)] = outValue;
}
