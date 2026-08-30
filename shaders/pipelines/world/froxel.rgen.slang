// Froxel volumetric light pass: one raygen invocation per camera-frustum voxel. Each cell computes the
// in-scattered radiance it emits toward the camera and stores it (plus its extinction density) into the
// froxel 3D grid; a later screen-space raymarch (froxel.comp) integrates those sources along the view ray.
//
// The grid's light is found by PATH TRACING, not a shadow map. For the dominant celestial body
// (sun/moon) each cell casts the exact same shadow ray the surface NEE uses (trace.visibility): the
// any-hit records accumulate real spectral transmission through semitransparent surfaces — tinted glass,
// ice, water, entity dielectrics — and terminate on the first opaque hit. A cell behind a magenta pane
// therefore receives magenta light and scatters magenta fog, so the colour transmission that tints a
// shadow on terrain tints the volumetric glow the same way. Because the cell sits in free space, the
// light reaching it is the product of every surface it passed through since the source.
import world_common;
import world_core;
import bindings;
import math;
import trace;
import sky;

// Isotropic sky/ambient scattering colour (linear BT.709, D65). Scaled by the ambient factor and the
// dominant body's illuminance so shadowed and night-time volumes keep a soft sky fill instead of going
// pitch black; the day/night level comes from the light itself.
static const float3 FOG_SKY_AMBIENT = float3(0.55, 0.68, 1.0);

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    celestialLight = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);

    uint3 cell = DispatchRaysIndex();
    uint3 dims = DispatchRaysDimensions();

    float near = max(worldPush.froxel0.x, 1.0e-4);
    float far = max(worldPush.froxel0.y, near + 1.0);
    float baseDensity = max(worldPush.froxel0.z, 0.0);
    float phaseG = clamp(worldPush.froxel0.w, -1.0, 1.0);
    float3 fogColor = max(worldPush.froxel1.rgb, float3(0.0, 0.0, 0.0));
    float ambientFactor = max(worldPush.froxel1.w, 0.0);
    float fogFloorY = worldPush.froxel2.x;
    float fogFalloff = max(worldPush.froxel2.y, 1.0e-3);
    float fogStrength = max(worldPush.froxel2.z, 0.0);

    // Screen UV of this cell, then the depth of its slice centre along the camera ray.
    float2 uv = (float2(cell.xy) + 0.5) / float2(dims.xy);
    float zFrac = (float(cell.z) + 0.5) / float(dims.z);
    float3 rd = primaryRayDir(uv * 2.0 - 1.0);
    float depth = near * pow(far / near, zFrac);
    float3 cellPos = worldPush.camOffset + rd * depth;

    // Exponential height falloff: full density at/below fogFloorY, thinning with altitude so the world
    // clears above a low-lying fog bank while caves near the floor stay dense.
    float heightFactor = exp(-max(cellPos.y - fogFloorY, 0.0) / fogFalloff);
    float density = min(clamp(baseDensity * heightFactor, 0.0, 1.0) * fogStrength, 50.0);

    // Cells with no density need no shadow ray (the dominant cost of this pass); store a zero cell so the
    // raymarch sees transparent air.
    if (density < 1.0e-5) {
        froxelVolume[cell] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Per-channel scattering coefficient; the volume is treated as pure scattering (extinction = density,
    // scattering = density * colour) so the density lane also drives the view-direction transmittance.
    float3 scattering = density * fogColor;

    // Path-traced, colour-transmitting transmittance from the cell toward the light. Same ray as the
    // surface NEE shadow; see the file comment for why this is the whole point of the pass.
    float3 toLight = visibility(cellPos, celestialLight.dir, 10000.0).transmittance;

    float3 sunInscatter = scattering * hg(dot(-rd, celestialLight.dir), phaseG)
            * celestialLight.illuminance * toLight;

    // Isotropic sky/ambient fill. Source per unit length = scattering * ambientRadiance; luminance of
    // the dominant body scales it with time of day, the tint gives it a sky colour.
    float3 ambientRadiance = ambientFactor * luminance(celestialLight.illuminance) * FOG_SKY_AMBIENT;
    float3 ambientInscatter = scattering * ambientRadiance;

    froxelVolume[cell] = float4(sunInscatter + ambientInscatter, density);
}
