# Caustica — how it works, with a deep pass on the volumetrics

Review artifact (not part of the shipped docs). Source-level read of the repo at
`6d3cff1`; nothing was built (no `slangc`/Vulkan SDK in this sandbox).

---

## 1. What the mod is

Caustica is a Fabric client mod for Minecraft 26.2 that **replaces the vanilla world
renderer with a hardware path tracer** on Minecraft's Vulkan backend.

- Java side (`src/main/java/.../rt/**`) owns Vulkan directly: it mixins into the
  vanilla `VulkanBackend`/`GpuDevice`, grabs the device/queue, and builds its own
  BLAS/TLAS, materials, light hierarchy, pipelines and descriptor rings.
- Terrain is re-meshed into RT geometry (`rt/terrain/**`), entities and particles are
  captured per-frame (`rt/entity/**`), materials come from a LabPBR-style registry
  (`rt/material/**`).
- Shaders are **Slang**, compiled to SPIR-V at build time; `buildSrc` generates the Java
  push-constant records and binding indices from shader reflection, so the ABI can't
  silently drift.
- Frame ends in DLSS Ray Reconstruction (denoise + upscale), optional DLSS Frame
  Generation, auto-exposure, bloom, ACES/tonemap → SDR or HDR10/PQ present.
- The UI, GUI and overlays (block outline, name tags, glow) stay vanilla-ish and get
  composited on top.

### Frame graph (from `RtComposite.renderFrame`)

```
build/refit TLAS ─ upload WorldPush ─ sky LUTs ──┐
                                                 ▼
   world primary trace (rgen A)  →  world indirect trace (rgen B)
                                                 │  scene color (render res) + guides
                                                 ▼
   ███ froxel fog: lighting pass → integration pass → history copy ███
                                                 ▼
   DLSS-RR (denoise + upscale to display res)  [or blit fallback]
                                                 ▼
   exposure histogram → bloom → display map (ACES/LUT, SDR|PQ) → debug present → UI
```

The fog sits **after both world traces and before DLSS-RR**. That placement is
deliberate and load-bearing: the fog's GI channel samples the *finished* scene color
image, while its direct channel still queries the exact same TLAS the trace used.
It writes the composited result back into the render-resolution scene image, so DLSS-RR
upscales scene+fog together.

---

## 2. There are three separate volumetric systems

| System | Where | Scale | Method |
|---|---|---|---|
| **A. Atmosphere** | `shaders/pipelines/sky_lut/*`, `world/sky.slang` | planetary (km) | Hillaire 2020 LUT-based single+multiple scattering |
| **B. Froxel fog** | `shaders/pipelines/fog/*`, `rt/fog/RtFroxelFog.java` | camera frustum (≤512 blocks) | froxel grid, 4 independent lighting fields, exact per-slice analytic integration |
| **C. Path-traced media** | `world/medium.slang`, `world/water.slang` | per-ray | Beer–Lambert extinction on a depth-2 medium stack + analytic caustics |

They do not talk to each other much, which is both the design's strength (each is
independently debuggable/disableable) and the source of the main physical gaps (§6).

---

## A. Atmospheric volumetrics — `world/sky.slang` + `sky_lut/*`

A textbook, well-implemented Hillaire "Scalable and Production Ready Sky and
Atmosphere" model. Everything is **photometric** (lux in, cd/m² out), linear BT.709
inside the module, converted to ACEScg exactly once at each consumer, and — importantly —
**all lengths are kilometres**, because the LUT parameterisations take square roots of
differences of radii near 6360 vs 6460 km and metres would destroy the fp32 significand
(the code says so, and it's right).

Medium model (`sampleMedium`), Earth reference coefficients:

- Rayleigh `(0.005802, 0.013558, 0.033100)` /km, scale height 8 km
- Mie scattering `0.003996`, extinction `0.004440`, scale height 1.2 km, `g = 0.8`
- Ozone `(0.000650, 0.001881, 0.000085)`, 25 km-centred tent, ±15 km — absorption only.
  This is what makes noon zenith deep blue and twilight go purple.

Three LUTs, all compute passes:

| LUT | Size | Cadence | Content |
|---|---|---|---|
| `transmittance.comp` | 256×64 | static | transmittance point→space by (altitude, cos zenith), 40 steps |
| `multiscatter.comp` | 32×32 | static | 2nd+ order scattering, 20 steps × 64 sphere directions |
| `view.comp` | 192×**216** | per frame | the dome itself, 32 steps — **two stacked 192×108 slices, one for the sun, one for the moon** |

The two-body sky-view slice is a nice touch: moonlight and sunlight scatter through the
same atmosphere with no crossfade between "day" and "night" lighting solutions. Same for
`dominantCelestialLight()`, which picks whichever body delivers more lux *after*
extinction — no authored blend, because at the flip point both are ~70 air masses deep
and already numerically zero.

`integrateScatteredLuminance` uses the analytic constant-medium segment integral
`(S − S·T)/σt` rather than a midpoint sample, which is why 32 steps don't band. The
planet shadow is softened over the light's angular radius and uses each sample's *own*
geometric horizon (`horizonDipSin`), so high-altitude samples stay lit after the ground
below is dark — that's what produces real twilight without a fill term.

**Consumers:** `world.rmiss` (sky for escaped rays), NEE's directional light, and — the
relevant part here — the fog lighting pass, which samples both LUTs directly
(`fogSkyRadiance`, `fogCelestialRadiance`). So fog and sky can never disagree about the
sun's colour or intensity.

**Gap:** there is no *aerial perspective* LUT. Distance haze on terrain comes entirely
from system B, which is capped at 512 blocks and uses a grey exponential-height medium,
not the Rayleigh/Mie one. Nether/End skies are on the TODO list.

---

## B. The froxel fog — the main event

### B.1 Storage layout

`RtFroxelFog` owns **ten** `RGBA16F` images. They are *2D atlases*, not 3D textures:

```
x = froxel column          width  = ceil(renderW / tile)          (tile 8 default)
y = zSlice * gridY + row   height = ceil(renderH / tile) * zSlices (32 default)
```

Five "current" fields + five histories:

| Field | RGB | A |
|---|---|---|
| `direct` | sun + moon in-scattered radiance | **current primary reversed-Z depth** (shared reprojection guide) |
| `local` | local emitter field (diagnostic, off by default) | 1 |
| `gi` | surface + sky indirect | **2nd moment of GI luma** (for variance-guided clamping) |
| `cache` | world-cell emitter reuse (off by default) | 1 |
| `giAux` | x = mean hit distance, y = hit coverage | z = history age, w = confidence |

All four radiance fields are stored **divided by `FOG_RADIANCE_SCALE = 128`** so a
forward-scattering sun phase peak can't overflow fp16's 65504; the integration pass
multiplies it back. Nice, cheap, and documented in one place (`fog_common.slang`).

Depth distribution is **logarithmic**: slice *i* sits at
`exp2(lerp(log2(0.5), log2(maxDistance), i/(gridZ−1)))` blocks from the camera. Near
plane 0.5, far = `fog.max-distance` (default 192, range 16–512).

Z-slice count is clamped so `gridY * gridZ ≤ min(8192, device maxImageDimension2D)` —
at 4K native with tile 8 you silently get 30 slices instead of 32.

### B.2 Pass 1 — `fog_lighting.comp.slang` (768 lines, one thread per froxel)

Four **completely independent** radiance fields are computed. None of them applies
extinction; that happens once, later. This is the key architectural decision — you can
toggle or debug-view any channel without perturbing the others.

**Direct (sun/moon shafts)** — `fog.direct-samples` (default 2) stratified samples per
froxel. Each sample jitters both the screen position *and* the log-Z slice with the
same `xi`, so the sample stays a coherent point inside the froxel instead of scrambling
screen and depth independently. For each of sun and moon: one inline `RayQuery`
visibility ray with `TMax = 10000`, then `illuminance × HG(θ, g)`. Because
`transmittanceToSpace` multiplies by a hard `smoothstep` horizon term that hits exactly
zero below the horizon, the body that's down costs **no rays at all** — so it's
~2 rays/froxel in practice, not 4.

**GI (surface + sky)** — `fog.gi-samples` (default 2) rays sampled **from the HG phase
function itself** around the forward axis. `fogSceneRadiance` reprojects the hit point
into the current screen, rejects it against reversed-Z primary depth, and reads the
scene colour → a screen-space radiance cache. Misses (and off-screen/occluded hits) take
the sky LUT. Ray length is capped at `min(fogMaxDistance, 192)`.

> Note: `phaseWeight = HG(cosθ)/pdf` where `pdf` *is* `HG(cosθ)` — it is identically
> 1.0. Correct, just dead arithmetic.

**Local emitters (diagnostic, default off)** — `fog.local-samples` (default 4) area-light
samples. Uses the **same published light hierarchy as the surface path**: 75% chance to
pick from the light-grid cell (spatial alias table), 25% global power alias, and the
selection PDF is reconstructed as an MIS-style mixture in `fogLightProposalPdf` so
off-screen resident emitters keep support without a modulo-index bias. Estimator is
`Le · (area·cosθ_e/d²) · phase / pdf`, with a shadow ray each.

**Cache / world-space emitter reuse (default off)** — quantises the froxel to a
world-space cell (`fog.cache-cell-size`, default 8 blocks), derives a stable hash, and
runs a **4-candidate weighted reservoir** (`W = Σw / (M·w_survivor)`) seeded from
`cellHash ⊕ frameIndex`. The point is that the key is *world*-space, so an emitter can
influence off-screen→on-screen transitions without a valid history pixel.
⚠️ Despite the name there is **no cross-froxel sharing** — every froxel runs its own
4 candidates with 4 shadow rays. The cell key only correlates the RNG streams so the
temporal filter can average them. It's expensive, which is why it defaults off.

**Sampling** — `fogSequence2` = a Halton(2,3) pair with a per-cell Cranley–Patterson
shift. GI uses a *frame-independent* cell hash (correct: the shift is fixed, the frame
index advances the sequence). The direct channel's `froxelHash` folds in `frameIndex`,
so its shift re-randomises every frame and the Halton progressivity is lost — it behaves
like white noise. Minor inconsistency, easily fixed.

### B.3 Temporal reconstruction (the most intricate part)

Reprojection is `fogPreviousCoordinates`: take the froxel's world position, add
`camDelta`, project with `prevViewProj`, and recompute the previous log-Z slice from the
radial distance to the *previous* camera. All four fields share these coordinates plus a
common set of rejection terms:

- `motionInFroxels` → `exp(−k·motion)`, with k = 0.80 (direct) / 0.65 (local) /
  0.55 (GI) / 0.45 (cache): the sharper the field, the harsher the motion penalty.
- `translationPenalty = exp(−0.35·|camDelta|)`
- `primaryDepthWeight` — reads the **previous primary depth out of the direct history's
  alpha lane** and rejects if the sky/surface classification flipped or reversed-Z
  changed beyond a value-proportional tolerance. One guide, four consumers, and no
  linear-depth assumption is ever made on a reversed-Z value.
- a froxel-depth check `|prevZ − curZ| > 0.18…0.22 → reject`, so a far cell can't be
  pulled into a newly-exposed near cell.

Per-channel extras:

- **Direct** clamps history to `2×current + max(0.1×current, 0.01)` and multiplies by a
  binary `directConfidence` (0 if the current sample is black). Visibility is a hard
  discontinuity, so the filter deliberately refuses to accumulate anything in a froxel
  that is currently occluded — it converges to dark instantly but can't denoise a
  half-lit froxel. Cap 0.90.
- **GI** is the sophisticated one: a guide-weighted 5-tap cross filter where each tap's
  weight comes from `fogGuideWeight(hitDistance, coverage)` — coverage compared as a soft
  exponential rather than a hard material-ID reject (with 2 rays, coverage legitimately
  hops between 0/0.5/1 on a static camera, and hard rejection would kill accumulation
  exactly where it's needed). Then a **variance-guided luminance clamp** against the
  reprojected *neighbourhood* min/max (not this frame's single stochastic estimate),
  with σ from the stored 2nd moment. The moment is mixed from the **raw** current sample,
  not the square of the resolved mean — the comment explains why, and it's correct:
  using the resolved mean collapses variance to zero and turns the moment lane into
  decoration. Age (`exp(−0.02·age)`) and confidence terms feed back through `giAux`.
  Cap 0.92.
- **Local / cache** get simpler clamps (`3×current + …`) and caps 0.85 / 0.90.

History is invalidated on: resolution/projection change, any fog setting change
(`settingsSignature()` hashes 30 values), terrain scene generation change, light
hierarchy generation change, fog base height change, far-distance change, and
reactivation after being disabled.

### B.4 Pass 2 — `fog_integrate.comp.slang` (one thread per screen pixel)

This is where the single extinction model lives, applied **once** to all four fields.

```
σt = density · exp(clamp(−(y − baseHeight)·falloff, −4, 4))     // grey, height-exponential
for each of gridZ log slices, front to back:
    clip the slice against the reconstructed primary surface distance
    T_step   = exp(−σt · L)
    S_int    = scatterScale · (1 − T_step)          // exact ∫σs·e^(−σt·s) for constant density
    incident = direct·wD + gi·wG + local·wL + cache·wC
    inScatter += T · S_int · albedoRGB · incident
    T *= T_step
    if (T < 0.002) break
result = scene·T + inScatter
```

Three things worth calling out:

1. **It is a true per-pixel ray march, not a 3D prefix-integral fetch.** Most froxel-fog
   implementations bake scattering/transmittance into a 3D texture and do one lookup.
   Doing 32 analytic steps per pixel costs more, but the opaque endpoint
   (`surfaceDistance`) is clipped *exactly*, so you don't get the classic depth-slice
   quantisation halo around foreground geometry.
2. **The per-slice integral is analytic**, `σs/σt·(1 − e^(−σt·L))`, not `σs·L`. For long
   logarithmic slices the naive form is visibly biased; this stays correct and
   well-behaved as density → 0.
3. **`scatteringColor.w` is effectively the single-scattering albedo**, and RGB is a
   separate tint. Defaults `(0.82, 0.90, 1.00) × 1.0` — slightly blue-biased, conservative.
   The config allows each up to 2.0, i.e. you can make the medium emit energy if you want.

Atlas sampling (`fogVolumeSample`) manually does the trilinear: hardware bilinear in XY,
manual lerp between the two packed depth rows, with the XY clamped to
`[0.5, grid−0.5]` **before** the row offset is added. That clamp is essential — a linear
tap at the outer row edge would bleed the neighbouring depth slice in and read as
temporal noise at the frustum border. Same guard in the history sampler.

Everything is done in pre-exposure-divided linear space and re-multiplied at the end,
so it composites correctly with the auto-exposure pipeline.

### B.5 Pass 3 — history

Five `vkCmdCopyImage` full-atlas copies (current → history), all images staying in
`VK_IMAGE_LAYOUT_GENERAL`, with explicit `compute→transfer` / `transfer→compute` barriers
either side.

### B.6 Synchronisation & lifetime

`RtFroxelFogPipeline` keeps a **6-deep descriptor-set ring** and a
`TrackedGraphicsUse` per slot; `bindFrame` awaits the completion token that consumed a
slot before rewriting it. On a shape change the *pipeline* (descriptor-set owner) is
destroyed **before** the image views it referenced — correct ordering, and the comment
says exactly why. Allocation failure sets a `failed` latch so the renderer degrades to
no-fog instead of crashing, and `resetFailureLatch()` lets an explicit reset retry.

### B.7 Debug

`debug_present` has dedicated views for the direct / local / GI / cache fields and the
`giAux` guides (hit distance, coverage, age, confidence), sampled at slice 0.5 and
scaled back by `FOG_RADIANCE_SCALE`. Composited after tonemapping so inspecting a field
can't perturb exposure history.

### B.8 Config surface

| Setting | Default | Range |
|---|---|---|
| `fog.enabled` / `direct` / `gi` | on / on / on | |
| `fog.local-lighting-enabled` | **off** (diagnostic) | |
| `fog.cache-reuse-enabled` | **off** (opt-in) | |
| `fog.density` | 0.0018 /block | 0 – 0.05 |
| `fog.height-falloff` | 0.025 | 0 – 0.25 |
| `fog.max-distance` | 192 blocks | 16 – 512 |
| `fog.froxel-tile-size` | 8 px ("Quality") | 4 – 32 |
| `fog.z-slices` | 32 | 16 – 96 |
| `fog.direct-anisotropy` (HG g) | 0.78 | −0.95 – 0.95 |
| `fog.direct/gi/local-samples` | 2 / 2 / 4 | |
| `fog.*-strength` | 1.0 / 0.35 / 0.8 / 0.15 | |
| `fog.*-temporal-blend` | 0.72 / 0.78 / 0.55 / 0.88 | |
| `fog.*-spatial-radius` | 0.5 / 1.0 / 0.5 / 1.0 | |

Base height is `seaLevel − terrain.blockY` (rebase space), so the height falloff is
anchored to the world's sea level, not the camera.

---

## C. Path-traced participating media — `medium.slang`, `water.slang`

This is the *per-ray* volumetrics inside the path tracer, entirely separate from the fog.

**Medium stack.** `MediumStack` is depth 2 (`current` + `outer`) held in **named fields,
not an array** — a dynamically indexed local array would land in scratch memory and this
raygen is already register-bound. Depth 2 covers air→water→glass and air→glass→water;
anything deeper degrades to air on the way out, and because `entering` is re-derived per
face from geometry rather than toggled, the path **re-synchronises at the next crossing**
instead of staying corrupted. That's a genuinely good robustness property.

**Tint → extinction.** Two mappings:
- Water: `WATER_ABSORB_FLOOR (0.015, 0.010, 0.008) + 0.1·(1 − tint)` from the biome
  colour. Low red in an ocean tint ⇒ red absorbed fastest ⇒ bluer with depth. The floor
  keeps even white-tinted water slightly absorbing.
- Other dielectrics: invert Beer–Lambert over a 1-block reference, `σ = −ln(tint·transmission)`,
  clamped away from zero so a saturated texel doesn't give infinite σ.

Applied as `throughput *= exp(−σ · hitT)` on every segment travelled inside a medium
(both rgen passes). Air's σ is zero, so the common case is one comparison.

**Water surface** (`water.slang`) — a 10-component directional wave spectrum:
sharp-crested profile `h = a·e^(S(sin φ − 1))` (peaked crests, flat troughs), **real
deep-water dispersion `ω = √(gk)`** so 14 m swell strides past while 0.38 m chop flutters
in place, wind-biased headings with per-component spread, and crest-phase meander on the
long components chain-ruled into the gradient. Geometry stays flat; only the normal
tilts. World-anchored domain, plus a wavelength/footprint LOD weight and an analytic
`∂/∂t` used to rewind the gradient one frame for reflection reprojection in a single
spectrum walk.

**Caustics** (`waterCaustic`) — this is the mod's namesake and the nicest bit of physics
here. For an underwater NEE vertex whose shadow ray crossed water, it refracts the sun
ray through the *same* analytic wave field, computes the horizontal landing position on
the receiver plane, finite-differences it (ε = 0.25 m) and takes the **inverse Jacobian
determinant** of the surface→floor mapping. Area compression = brightening; real fold
caustics where det → 0, clamped at 6.0 to stop fireflies. Because the per-sample sun-quad
jitter shifts the pattern, caustics **physically blur with depth** under DLSS-RR
accumulation instead of being a scrolling texture. Grazing light fades to neutral 1.0.

**Shadow rays** use a dedicated SBT record with any-hit only: cutout alpha-tests,
translucent and water *tint* the transmittance and continue, opaque terminates — so
coloured shadows through stained glass work, and `waterHitT` comes back as the caustic
trigger.

---

## 6. Observations

### Strengths

- The four-field decomposition in the fog is the right call. Each channel has its own
  estimator, its own temporal policy tuned to its own frequency content, and its own
  debug view. That's rare and it makes the system tractable.
- The exact per-slice analytic integral + exact surface-distance clipping avoids the two
  most common froxel-fog artefacts (long-slice bias, depth-quantisation halos).
- The fog reuses the *published* light hierarchy and sky LUTs rather than duplicating
  them, so it cannot disagree with the surface path about what the lights are.
- Vulkan hygiene is genuinely careful: descriptor ring with completion tokens,
  destruction ordering, generated push-constant ABI, failure latches.
- The comments explain *why* (fp16 headroom, km vs m, scratch-memory avoidance, reversed-Z,
  atlas row bleed). Very little of it is restating the code.

### Physical approximations worth knowing about

1. **Off-screen GI hits become sky.** In `fogIndirectRadiance`, a ray that *hits a
   surface* which isn't visible on screen (off-frustum, or occluded by a nearer primary
   surface) falls back to `fogSkyRadiance`. Inside a cave or a sealed room, the fraction
   of GI rays that go sideways/backwards will inject full daylight sky radiance into the
   fog. Forward-biased HG (g = 0.78) keeps most rays on-screen and `gi-strength` is only
   0.35, so it's damped — but it's the largest light-leak vector in the system. A better
   fallback would be the previous frame's own GI field at the reprojected hit, or a
   sky-visibility-attenuated ambient, rather than the raw dome.
2. **Fog visibility is `FORCE_OPAQUE`.** Alpha-cutout foliage fully occludes shafts (a
   tree canopy blocks light with its whole quad, not its leaf mask), and stained
   glass/water neither tint nor transmit a shaft. The surface path does honour cutout
   and coloured shadow semantics, so the two disagree. The code acknowledges this
   explicitly as a cost bound, and it's a reasonable trade — just know it's there.
3. **Underwater is unhandled by the fog.** `fog_integrate` only knows a grey
   height-exponential medium; nothing reads `worldPush.flags` bit 0 (submerged) or
   `waterParams`. When the camera is submerged you get water absorption from system C
   *plus* atmospheric haze from system B stacked on top, and the "shafts" are traced as
   if through air. Underwater god rays — the obvious thing to want from a mod called
   Caustica — aren't there yet.
4. **No aerial perspective from the atmosphere model.** Distance haze is the fog's grey
   medium, capped at 512 blocks, not Rayleigh/Mie. Far terrain won't blue-shift correctly.
5. **`local` and `cache` are additive over the same emitter set.** Both sample the same
   light hierarchy with visibility, and `fog_integrate` adds `local·wL + cache·wC`. Both
   default off, but enabling both double-counts emitter energy. Worth either making them
   mutually exclusive or documenting the overlap.
6. **Fog is composited pre-DLSS-RR.** RR's guides (albedo, normal, depth) describe the
   surface, not the medium, so RR is denoising a signal it has no guide for. Smooth haze
   is fine; a hard shaft edge crossing a depth discontinuity is where you'd expect
   smearing. This is a real trade for getting fog upscaled with the scene, not a bug.

### Concrete efficiency opportunities

1. **Two of five atlases are dead weight in the default config.** `local` and `cache` are
   off by default, yet their images are allocated (2 × 8.3 MB at 1080p render res),
   written every frame, `vkCmdCopyImage`'d to history every frame, and sampled twice per
   slice per pixel in the integration loop. Gating allocation + copy + sampling on
   `LOCAL_STRENGTH/CACHE_STRENGTH > 0 && enabled` would remove ~17 MB of VRAM, 2 of 5
   history copies, and roughly a third of the integration pass's texture traffic for free.
2. **History copies instead of ping-pong.** Five full-atlas copies per frame is
   ~41 MB read + 41 MB write at 1080p (and ~155 MB each way at 4K native). Alternating
   current/history bindings across two descriptor sets eliminates all of it. The ring is
   already 6 deep, so the plumbing exists.
3. **VRAM scales worse than it looks.** Ten `RGBA16F` atlases: ~83 MB at 1080p render res,
   ~310 MB at 4K native with tile 8. Given DLSS-RR usually renders below display res this
   is normally fine, but a native-4K user with fog on pays a lot.
4. **`fogRadicalInverse3`** runs a 12-iteration integer divide loop per sample; a
   precomputed table or base-2 Sobol pair would be cheaper.
5. **Direct-channel Halton shift includes `frameIndex`** (`froxelHash`), which defeats the
   low-discrepancy progressivity the comment describes. GI does it correctly (stable
   `giHash`); making direct match should reduce shaft shimmer for free.
6. **`phaseWeight` in `fogIndirectRadiance` is provably 1.0** (HG-sampled, HG-weighted).
   Harmless, but it's two `pow`s per GI sample.

### Small latent couplings

- `FOG_NEAR_DISTANCE = 0.5` is hard-coded in **both** fog shaders while
  `pc.medium.w` (also 0.5) is pushed from Java for the slice-distance function only. The
  `volumeZ` mapping uses the constant, the segment bounds use the push value. They agree
  today because Java hard-codes `0.5f`; if anyone ever exposes a near distance, the
  lighting and integration passes will disagree about where slice *i* lives. Push it
  everywhere or make it a shared `static const`.
- `fog_integrate`'s segment bounds divide by `gridZ` while `fog_lighting`'s froxel
  centres divide by `gridZ − 1`. The continuous `volumeZ` lookup papers over it, so
  it's consistent in practice, but the two mappings being *almost* the same is the kind
  of thing that bites during a refactor.
- `historyValid = true` is set unconditionally at the end of `record()`, including on the
  very first frame after allocation. That's fine only because the lighting dispatch
  always covers the full atlas — worth a comment, since it's an invariant, not a
  coincidence.
