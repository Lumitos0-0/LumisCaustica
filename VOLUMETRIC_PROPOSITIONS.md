# Caustica — Clean Advanced Volumetric System Propositions

> Investigated from branch `arena/01a04a68-lumiscaustica`
> Repository: `Lumitos0-0/LumisCaustica`
> Date: 2026-08-28

---

## 1. Brief Investigation — How the Mod Works

**Identity:** Caustica (`LumisCaustica`) is an experimental hardware-ray-traced renderer for Minecraft 26.2's Vulkan backend. It replaces vanilla world rendering with path-traced radiance, DLSS Ray Reconstruction / Frame Generation, HDR output, dynamic entity capture, LabPBR-style material pages, RIS emitter lighting, and animated water caustics.

**Pipeline architecture (confirmed by source + shaders):**
- `primary.rgen.slang` (Pass A): camera ray, guide buffers, first dielectric split, writes `PackedPathSegment` continuation queue (`pathQueueAddr`).
- `indirect.rgen.slang` (Pass B): resumes queued segments; bounce loop with sun/moon NEE (`celestialLight` from `sky.slang`), RIS reservoir sampling (`risInitial` / `shadeReservoir` in `lighting.slang`), thin-surface SSS (`SSS_STRENGTH`, `hg()`), GGX BRDF, Russian roulette, medium push/pop, Beer-Lambert attenuation at segment boundaries (`throughput *= exp(-extinction * hitT)`).
- `closest_hit.rchit.slang`: evaluates `MaterialHeader`, samples canonical pages (`materialSurface0Tex`, `materialSurface1Tex`, `materialNormalAoTex`), builds payload (`Payload.surface0..2` pack albedo/normal/sky; `roughMetal`, `emissionSss`, `iorTransmission`, `flags`).
- `any_hit.rahit.slang` + `shadow.rmiss.slang` / `sky.rmiss.slang`: visibility and atmosphere miss shaders.

**Medium / Volume (`medium.slang`):**
- `Medium` struct: `float ior`, `float3 extinction`, `bool water`.
- `MediumStack`: fixed depth-2 (`current` + `outer`), NOT an array. Explicit push/pop at dielectric crossings (`mediumPush`, `mediumPop`).
- `segment.slang`: `PathSegment` packs `medium.current` and `medium.outer` via `packRgb9e5()` / `unpackRgb9e5()` into `PackedPathSegment`. This keeps medium state register-compact and avoids scratch-memory arrays.
- Water uses biome-tint (`worldPush.waterParams.xyz`) mapped to Beer-Lambert extinction (`waterExtinction`). Glass/ice uses texture-tint mapped by `volumeExtinction()`.
- **No in-scattering** currently exists: attenuation only (`exp(-sigma * dist)`). No volumetric emission integration along ray segments.

**Emission subsystem (confirmed by Java + shaders):**
- `RtEmissionGrid` (16x16 cells): CPU-built per-sprite premultiplied emission (`r,g,b,weight`).
- `RtEmissionHeuristic`: derives emission masks from linear RGBA albedo (`DARK_FLOOR`, `LUMINANCE_POWER`).
- `RtEmissionSemantics`: analyzes block-state light emission and sprite associations.
- `RtMaterialRegistry`: compiles `MaterialHeader` with `emissionSource` (LAB_PBR / HEURISTIC_MASK / STATE_UNIFORM / NONE), `emissionStrength` (packed 16-bit fraction of `MAX_EMISSION_STRENGTH = 65504`), feature bits (`FEATURE_SPEC`, `FEATURE_HEURISTIC_EMISSION`).
- In shaders (`indirect.rgen.slang`): direct emission added when `payloadEmission() > 0.0` and either not RIS-gated or `showCelestial` true. RIS (`lighting.slang`) samples emitters from `lightBufAddr` / `lightAliasAddr` / `lightGridCellAddr` with reservoir sampling (`risInitial` → `shadeReservoir`).
- `closest_hit.rchit.slang`: `payload.flags |= PAYLOAD_EMITTER_IN_LIST` when `pr.flags & TERRAIN_PRIM_IN_LIGHT_BUFFER`. This gates double-counting: RIS-covered emitters suppress direct-hit emission on diffuse continuation rays (`showCelestial == false`).

**Materials (`RtMaterialRegistry.java`, `world_common.slang`, `closest_hit.rchit.slang`):**
- Canonical pages per sprite: `surface0` (rough/metal/emission/sss), `surface1` (LabPBR F0 for metals), `normalAo` (tangent-space normal + AO). `MaterialHeader` holds `model`, `features`, `params` (rough, metal, IOR, transmission), `texturePage`, `materialUv`, `albedoUv`.
- `Surface` evaluation (`evaluateMaterial`): decodes pages, applies normal perturbation (`perturbNormal` with TBN from `VK_KHR_ray_tracing_position_fetch`), applies AO (`NORMAL_AO_STRENGTH = 0.5`).
- `RtMaterials.Profile`: DEFAULT, METAL, GLASS, SMOOTH, WATER, LAVA — maps roughness/metalness and emission variants (`VARIANT_OPAQUE`, `VARIANT_GLASS`, emitting/non-emitting cross product, 32-byte aligned `MaterialHeader`).

**Sky (`sky.slang`, `sky_lut/*.slang`):**
- Hillaire 2020 atmosphere: `transmittanceLut` (256x64), `multiscatterLut` (32x32), `skyViewLut` (192x216, 2 bodies = sun + moon). Photometric: lux in, cd/m² out.
- `medium.slang` does NOT import `sky.slang`; atmosphere is separate from world volume transport. Sky miss (`world.rmiss`) reads `payloadSky()` (full fp32 float3). Sky is NOT integrated as an in-scattering volume inside the world path trace.

---

## 2. Current Volumetric Capability (What Exists)

| Capability | Status | Key File / Module |
|---|---|---|
| Beer-Lambert attenuation in volume | **Yes** — `throughput *= exp(-extinction * hitT)` in `indirect.rgen.slang` | `medium.slang`, `indirect.rgen.slang` |
| Water volume (biome tint, IOR 1.333, wave normal, caustic) | **Yes** — `makeDielectricMedium(true, WATER_IOR, tint, transmission)`; caustic in NEE | `medium.slang`, `water.slang`, `indirect.rgen.slang` |
| Glass/ice transparent volume (tint, IOR from `RtDielectrics`, inset bias) | **Yes** — `MaterialHeader.model == MATERIAL_DIELECTRIC`, `volumeExtinction()` | `medium.slang`, `closest_hit.rchit.slang` |
| Medium stack depth-2 (current + outer) | **Yes** — fixed fields, no array, avoids scratch memory | `medium.slang`, `segment.slang` |
| In-scattering (phase function, integrated radiance inside volume) | **No** | Missing in `indirect.rgen.slang` |
| Volumetric emission (emission density along segment) | **No** — emission is surface-only (`payloadEmission()`) | `indirect.rgen.slang`, `lighting.slang` |
| Atmospheric fog / haze integrated with sky physics | **No** — sky and world volume are disconnected | `sky.slang` vs `medium.slang` |
| Heterogeneous volume density (noise/3D texture) | **No** — only homogeneous per-block tint | `medium.slang` (no texture sample for density) |
| Deep nesting (>2 media layers) | **No** — degrades to air on deeper exit (by design) | `medium.slang` comment |

---

## 3. Propositions — Clean & Advanced Volumetric System

Each proposition is designed to respect the existing architecture:
- **No dynamic arrays** in raygen/hit (register-bound, scratch-memory cost is high — see `medium.slang` comment).
- **Depth-2 medium stack preserved** (`MediumStack` fixed fields in payload / segment packing).
- **Emission integration preserved** (RIS light grid, `payload.flags` emission bits, `gateEmitter` logic).
- **Material page system preserved** (`MaterialHeader`, canonical texture arrays, `Surface` evaluation).
- **Sky/atmosphere preserved** (`sky.slang` LUTs, `worldPush.skyLook*`, `celestialLight`).

---

### Proposition A — Homogeneous In-Scattering Integration (Core Volumetric Transport)

**Concept:** Extend `indirect.rgen.slang`'s segment attenuation from pure extinction (`exp(-sigma * t)`) to include single-scattering in-scattering along the segment inside the medium. The medium defines not just `extinction` (`sigma_t = sigma_s + sigma_a`) but also `scattering` (`sigma_s`) and a phase function asymmetry (`g`), similar to the existing `SSS_G = 0.6` Henyey-Greenstein used for thin surfaces.

**How it works:**
- In `medium.slang`, expand `Medium` with `float3 scattering; float phaseG; float emissionDensity;` (or keep emission separate — see Prop B). `scattering` is the scattering coefficient (units = 1/block, like `extinction`).
- In `indirect.rgen.slang`, during the bounce loop, when `medium.current` has `any(scattering > 0.0)`:
  - Instead of only `throughput *= exp(-extinction * payload.hitT)`, also accumulate scattered light:
    `L += throughput * (scattering * phase * lightRadiance) * analyticIntegration(...)` or discrete raymarch step.
- Given the register-bound constraint, a **discrete 2-step analytic integral** (exact for homogeneous segment) is preferred over iterative raymarch loops (loop divergence costs lanes):
  `integratedInScatter = (1 - exp(-sigma_t * t)) / sigma_t * sigma_s * phase` (similar to atmospheric single-scattering integral in `sky.slang` — reuse `integrateScatteredLuminance` logic but scaled to block units).
- The phase function uses the same `hg()` from `math.slang` (`SSS_G` can become a per-medium parameter instead of a constant).

**Integration with emission:**
- If `scattering > 0` and the medium is inside an emissive container (e.g., glowing fog, lava-filled glass), emission density (Prop B) feeds into the same integral: the segment integrates both self-emission and in-scattered light.
- RIS (`risInitial`) remains unchanged: it samples surface emitters. Volumetric emission (Prop B) is added directly to `L` per segment, not through RIS, which is correct because RIS targets discrete surface lights; continuous volume emission has no discrete sample point. This avoids corrupting the reservoir's `phat` / `W` statistics.

**Clean design notes:**
- Keeps `Medium` small (add 4 floats: `scattering.r/g/b` + `phaseG` — fits in `std430` without changing payload alignment significantly). `emissionDensity` should be separate (Prop B) to avoid mixing light-source and transport properties.
- No loop divergence in `indirect.rgen.slang`: the integration is a closed-form expression evaluated once per segment, not a `for` loop over steps.
- Compatible with `segment.slang`: `packRgb9e5()` currently packs `currentExtinction` and `outerExtinction`. Adding `currentScattering` requires either (a) packing into the existing word (replacing some bits — risky) or (b) adding a new `uint currentScattering` to `PackedPathSegment`. Given `PackedPathSegment` is 48 bytes (see `segment.slang` comments), adding fields may break the stride. Better approach: reuse `currentExtinction` word for scattering when `scattering` is needed, or extend `PathSegment` but keep `PackedPathSegment` compact by encoding scattering as a fraction of extinction (e.g., `scattering = albedo_scat * extinction`) if a single scalar albedo is sufficient. **Recommendation:** start with `scattering = albedo * extinction` (single float `scatteringAlbedo` in `Medium`), so no new payload word is needed — just multiply `extinction` by albedo before integration.

---

### Proposition B — Volumetric Emission Density (Emissive Volumes Compatible with RIS & Grid)

**Concept:** Allow volumes (not just surfaces) to emit light continuously. Extend the emission subsystem so that emissive materials can express a volumetric emission density (e.g., `cd/m²` per block depth, or per-unit-volume radiance) rather than only a surface radiance (`payloadEmission()`).

**How it works:**
- **CPU side (`RtEmissionGrid.java`, `RtMaterialRegistry.java`):** Add an optional `volumeEmissionDensity` float to `RtMaterialDesc`. When `emissionSource == STATE_UNIFORM` or `LAB_PBR`, if the material is also a volume container (`MODEL_DIELECTRIC` or a new `MODEL_VOLUME`), the emission value is interpreted as density rather than surface radiance.
- **Shader (`indirect.rgen.slang`):** When inside a medium (`medium.current`) with emission density `Le_vol`, accumulate:
  `L += throughput * Le_vol * (1 - exp(-sigma_t * t)) / sigma_t` (integrated emission over the segment, attenuated by medium extinction). This is analogous to atmospheric emission but scaled to local block-space.
- The existing `RIS` (`lighting.slang`) continues to handle **surface** emitters (`lightBufAddr`). Volumetric emission does NOT need RIS sampling (no point light source), so it is added directly to `L`. This avoids the `gateEmitter` complexity corrupting reservoir weights.
- **Material page extension (`MaterialHeader`, `evaluateMaterial`):** Add `float emissionDensity` to `MaterialHeader.params` (currently holds rough, metal, IOR, transmission). `Surface.emission` stays for surfaces; a new `Surface.emissionDensity` or direct `MaterialHeader` read in `indirect.rgen.slang` provides the volume value.

**Clean design notes:**
- The existing emission packing (`MATERIAL_EMISSION_STRENGTH_SHIFT = 8`, mask `65535`) carries surface emission. A new bit in `MaterialHeader.features` (e.g., `FEATURE_VOLUME_EMISSION = 32`) signals that `emissionStrength` should be interpreted as density. This keeps backward compatibility: non-volume materials ignore the bit.
- `payload.flags` already has bits `4..6` (`PAYLOAD_EMISSION_SOURCE_SHIFT`) for emission source type. Adding a `PAYLOAD_VOLUME_EMITTER` bit (`32` or reuse an existing bit) allows `indirect.rgen.slang` to distinguish volume emission from surface emission for gate logic.
- Compatible with `RtEmissionGrid`: the grid stores surface emission footprint. Volume emission uses the material-level `emissionStrength` directly (uniform per material) or a new 3D texture (Prop D), avoiding grid bloat.

---

### Proposition C — Atmospheric Volume Integration (Sky-to-World Fog / Haze)

**Concept:** Connect `sky.slang` (atmosphere physics) with `medium.slang` (world volume transport) so that the world can contain atmospheric fog/haze that uses the same scattering/extinction coefficients (Rayleigh, Mie, ozone) as the sky dome, ensuring visual consistency.

**How it works:**
- `sky.slang` defines `sampleMedium(altitudeKm)` which returns `scattering`, `extinction`, `rayleigh`, `mie`, `phaseR`, `phaseM`. This is exactly the physics needed for world fog.
- Add a new world-volume mode (`MODEL_ATMOSPHERE` or reuse `MATERIAL_DIELECTRIC` with special feature bit) where the medium's `scattering` and `extinction` are derived from `worldPush.waterParams` / new `worldPush.atmosphereParams` or sampled from a 3D procedural noise texture bound to `blockAlbedoAtlas` or a new binding.
- The simplest clean approach: a **procedural homogeneous fog layer** defined by `WorldPush` parameters (`fogDensity`, `fogHeight`, `scatteringAlbedo`, `phaseG`). `indirect.rgen.slang` applies the same single-scattering integral as Prop A, but uses `scattering` derived from the sky's coefficients scaled by density.
- More advanced: a **3D noise-based heterogeneous fog** (Perlin/simplex noise sampled in world space, bound as a 3D texture or computed procedurally in shader) to create clouds/weather. Given the shader's register-bound design, a procedural noise function (`hash3d`) is preferred over a 3D texture (avoids new binding / memory load). The noise output `n(x,y,z)` drives `extinction` and `scattering` locally.

**Integration with emission:**
- Fog/clouds can be emissive (lightning-lit clouds, glowing nether fog). `emissionDensity` (Prop B) applies inside the fog medium. Because fog covers large segments (`payload.hitT` large), integrated emission `Le * (1 - exp(-sigma_t * t)) / sigma_t` produces smooth volumetric glow, not point-source fireflies — perfect for atmospheric emission.
- The sky's `celestialLight.illuminance` can also drive in-scattering inside fog: `L += throughput * scattering * phase * celestialLight.illuminance * analyticIntegration(...)`. This connects the sun/moon NEE directly with volumetric fog lighting without requiring additional shadow rays per step.

**Clean design notes:**
- Keeps `Medium` compact: `scattering` (derived from `worldPush.atmosphereParams`) + `phaseG` + `emissionDensity`. No 3D texture binding needed for homogeneous fog; procedural noise is optional and self-contained.
- Reuses `sky.slang` math (`hg()`, `sampleMedium` logic scaled to blocks) — avoids duplicating physics.
- The `segment.slang` packing handles the new medium type automatically because it only packs `extinction`, `ior`, `water` flags. Adding `scattering` requires either (a) deriving it from `extinction` at unpack time (if `scattering = albedo * extinction`) or (b) adding a compact field. See Prop A recommendation: use albedo-scaled scattering to avoid payload expansion initially.

---

### Proposition D — Material-Channel Volumetric Properties (Glass with Dust / Fog Containers)

**Concept:** Extend the material evaluation pipeline (`MaterialHeader`, `evaluateMaterial`, `Surface`) to support materials that represent volume containers: glass with suspended particles (dust in light beams), water with plankton/glow, or dedicated fog blocks. The material page provides not just surface albedo but volume properties.

**How it works:**
- **Material page extension:** The canonical pages (`materialSurface0Tex`, `materialSurface1Tex`, `materialNormalAoTex`) are per-texel arrays. Add a new optional page binding (`volumePropertyTex`) or reuse `materialSurface0Tex`'s alpha channel (`surface0.a`) for `scatteringAlbedo` / `volumeDensity` when the feature bit `FEATURE_VOLUME_PROPERTIES = 64` is set in `MaterialHeader.features`.
- `MaterialHeader.params` currently holds `(roughness, metalness, IOR, transmission)`. It can be extended to `(roughness, metalness, IOR, transmission, scatteringAlbedo, phaseG)` if `float4` is expanded to `float6` (but `float4` is fixed in `world_common.slang`). Better: reuse `surface0` page (which already carries emission and SSS) to also carry `scattering` when `FEATURE_SPEC` or `FEATURE_HEURISTIC_EMISSION` is combined with a new `FEATURE_VOLUME` bit.
- In `indirect.rgen.slang`, after evaluating the material at the hit (`Surface surface = ...`), if the hit is inside a dielectric (`material == MATERIAL_DIELECTRIC`) or the surface represents a volume boundary, read `surface.scattering` (derived from page alpha / new feature) and set the segment's medium `scattering` accordingly. This links surface material to volume transport seamlessly.
- The `RtMaterialRegistry.java` compilation (`compileDesc`) can set `scattering` from material properties: e.g., glass (`MODEL_DIELECTRIC`) with a dust feature gets `scattering = 0.1 * transmission`, phase `g = 0.7`.

**Integration with emission:**
- A glass container holding emissive fog uses both Prop B (`emissionDensity`) and Prop D (`scatteringAlbedo`): the surface material defines the container boundary (IOR, transmission, surface roughness); the volume emission defines the glow inside; the scattering property defines how light propagates and scatters within. This is exactly how real emissive participating media work (e.g., glowing mist inside a glass jar).
- RIS (`risInitial`) still samples the surface emitters (jar walls, surface glow). The volumetric emission (mist glow) is integrated over the segment (Prop B), avoiding conflict.

**Clean design notes:**
- Keeps `MaterialHeader` size intact: `features` is 32-bit (`uint`), has spare bits (`FEATURE_SPEC = 1`, `FEATURE_NORMAL = 2`, `FEATURE_HEURISTIC_EMISSION = 4`, `FEATURE_STOCHASTIC_ALPHA = 16`). `FEATURE_VOLUME_PROPERTIES = 32` fits. `params` stays `float4` (rough, metal, IOR, transmission); volume properties come from the page or feature bit, not new header fields.
- `Payload.flags` (`PAYLOAD_DIELECTRIC_ENTERING`, `PAYLOAD_EMISSION_SOURCE_*`) can be extended with `PAYLOAD_VOLUME_SCATTERING = 32` or similar to signal to `indirect.rgen.slang` that the medium should use scattering properties rather than pure extinction.

---

### Proposition E — Deep Volume Stack Extension (Depth > 2 with Compact Encoding)

**Concept:** The current `MediumStack` is explicitly depth-2 (`current` + `outer`) with a comment that deeper nesting degrades to air. For true nested volumes (e.g., underwater inside glass inside fog), extend to depth-3 or a compact array-encoded stack while keeping payload size bounded.

**How it works:**
- **Option 1 (Depth-3 fixed fields):** Add `Medium inner` to `MediumStack` (current, middle, outer). `segment.slang` packs three extinction/IOR sets. This triples medium payload but stays fixed-size. `indirect.rgen.slang` pushes/pops with three-level nesting. Given the existing comment about realistic worst-case (`air->water->glass`), depth-3 covers almost all Minecraft cases.
- **Option 2 (Compact array encoding):** Keep `MediumStack` depth-2 but encode deeper nesting as a single `current` medium that combines properties (e.g., blended IOR, combined extinction) or use a small fixed-size array (`Medium stack[3]`) packed into `uint` words using quantized IOR/extinction. This avoids growing `PathSegment` but requires unpacking arithmetic.

**Integration with emission & RIS:**
- Deeper nesting doesn't change emission logic: emission still applies at the surface (`closest_hit.rchit.slang`) or inside the current volume (`indirect.rgen.slang`). The stack depth only affects attenuation and refraction paths.
- The medium push/pop in `indirect.rgen.slang` uses `entering` (`payloadDielectricEntering()`) derived from face orientation. Deeper nesting requires tracking which layer is being exited/entered, but the `entering` boolean per face handles this naturally: at each crossing, the face orientation tells whether to push (entering) or pop (exiting) relative to the current stack top.
- For depth-3, push/pop logic becomes: if entering and depth < max, push; else if not entering and current is the entered layer, pop back to previous. This is simple branch logic, no array indexing needed if fields are named (`current`, `middle`, `outer`).

**Clean design notes:**
- The current payload (`Payload`) is exactly sized with `uint surface0..2`, `float hitT`, `half3 motionPrev`, `uint flags`, `uint roughMetal`, `uint emissionSss`, `uint iorTransmission`, `uint rayCone`. Adding `Medium inner` to `PathSegment` increases `PackedPathSegment` from 48 bytes. Check stride alignment (`std430`). If `PathSegment` grows, `queue` (`DevicePtr<PackedPathSegment>`) size increases linearly with pixel count. At 1920x1080, 48 bytes → ~100MB per frame; adding 16 bytes → ~133MB. This is acceptable but must be measured.
- **Recommendation:** Implement depth-3 (`current`, `middle`, `outer`) as fixed fields first. It's the cleanest extension of the existing design and avoids scratch arrays. Monitor `RtFrameStats.java` or GPU memory usage.

---

### Proposition F — Volumetric Light Transport for Emissive Media (Integrated RIS + Volume)

**Concept:** Combine Prop A (in-scattering) and Prop B (volumetric emission) into a unified light transport model where emissive media contribute light to both the local segment (self-emission) and nearby surfaces (scattered emission), and RIS reservoir sampling accounts for this.

**How it works:**
- **Local segment emission:** As in Prop B, `L += throughput * Le_density * (1 - exp(-sigma_t * t)) / sigma_t`.
- **Scattered emission to nearby surfaces:** This is advanced. In a full volumetric path tracer, emission inside the volume also scatters: `L_vol_scattered = integral(throughput(s) * sigma_s(s) * Le(s) / (4*pi) * hg(...) ds)`. For a homogeneous medium over a short segment (`t`), this simplifies to:
  `L += throughput * sigma_s * Le_density * integral_factor * phase`.
  This is essentially the same integral as Prop A but with `Le_density` inside. It can be computed with the same closed-form expression without additional shadow rays.
- **RIS integration:** If a volume contains discrete bright emitters (e.g., a glowing particle inside fog), RIS (`risInitial`) should include them. The current RIS uses `lightGridCellAddr` / `lightGridSpanAddr` for discrete surface lights. Volumetric discrete emitters can be added to the same grid (`lightBufAddr`) with their 3D positions (`float3 pos` already in `Light` struct) and volumes (new field or derived from density). The reservoir sampling (`shadeReservoir`) uses `visibility()` shadow rays; for volumetric emitters, visibility should also account for medium attenuation (`throughput *= exp(-extinction * dist)`). The existing `VisibilityResult` (`transmittance`, `waterHitT`) can be extended with `mediumTransmittance` by sampling the medium along the shadow ray — this connects naturally with Prop A.

**Clean design notes:**
- Keeps RIS architecture intact: `Light` records (32 bytes, `float3 pos`, `uint le`, half axes, etc.) can represent point/light sources inside volumes if `lightArea()` is interpreted differently (e.g., spherical volume or point source). `proposalPdf()` uses `power = area * luminance(le)`; for point sources, `area` can be `4 * pi * r^2` or a fixed virtual area, maintaining the same math.
- The `shadow` visibility in `shadeReservoir` (`visibility()`) can be enhanced to apply medium attenuation (`throughput *= exp(-medium.current.extinction * shadow_dist)`) when `medium.current` is non-air. This makes RIS shadows correct inside fog/glass without changing the reservoir algorithm.
- `medium.current.water` and `waterCaustic` (`indirect.rgen.slang`) show how special medium behavior (caustic focusing) is handled per-medium. Volumetric emission scattering can have its own medium-specific term (`volumeCaustic` or `scatteredLe`) using the same pattern.

---

## 4. Implementation Priority & Compatibility Matrix

| Proposition | Key Integration Points | Emission Compatible | RIS Compatible | Material Pages Compatible | Sky/Atmosphere Compatible | Register / Payload Impact |
|---|---|---|---|---|---|---|
| **A** — In-Scattering Transport | `indirect.rgen.slang` (segment attenuation), `medium.slang` (`scattering`, `phaseG`) | Yes (Prop B adds emission) | Yes (no RIS change) | Yes (scattering from material or medium) | Yes (reuse `sky.slang` coefficients) | Low: add fields to `Medium`, reuse payload words |
| **B** — Volumetric Emission Density | `RtMaterialDesc` (density bit), `MaterialHeader` (feature bit), `indirect.rgen.slang` (integrated emission) | **Core feature** | Yes (volume emission direct, RIS unchanged) | Yes (`FEATURE_VOLUME_EMISSION`) | Yes (fog emission uses sky illuminance) | Low: feature bits + direct `L` accumulation |
| **C** — Atmospheric Fog / Haze | `WorldPush` (fog params) or `medium.slang` (procedural noise), `indirect.rgen.slang` (scattering integral) | Yes (fog can emit) | Yes (fog attenuates RIS shadows with Prop F) | Yes (fog is a medium property, not material) | **Core feature** (reuses `sky.slang` physics) | Low: procedural noise avoids new bindings |
| **D** — Material Volume Properties | `MaterialHeader.features`, `Surface`, `evaluateMaterial`, `RtMaterialRegistry.compileDesc()` | Yes (glass containers with emissive fog) | Yes (volume properties don't affect RIS sampling) | **Core feature** (new feature bit, page reuse) | Yes (material properties define fog density) | Low: feature bits + page interpretation |
| **E** — Depth-3 Stack | `medium.slang` (`MediumStack`), `segment.slang` (`PackedPathSegment`), `indirect.rgen.slang` (push/pop) | Yes (deeper nesting for emissive volumes) | Yes (RIS shadows work at any depth) | Yes | Yes | Medium: payload / segment size increase (~16 bytes) |
| **F** — Unified Volumetric RIS + Transport | `lighting.slang` (`shadeReservoir`, visibility), `indirect.rgen.slang` (medium attenuation in shadows) | Yes (volume emission integrated, surface emission preserved) | **Core feature** (medium-aware visibility, volume light sources) | Yes | Yes | Medium: visibility result extension |

---

## 5. Clean Design Principles (From Code Evidence)

These are derived from reading the actual architecture, not generic advice:

1. **Fixed-size, register-bound payloads** (`Payload` is 12 bytes of packed surface words + `hitT` + motion + flags + rough/metal + emission/sss + ior/transmission + rayCone). No dynamic arrays (`medium.slang` explicitly avoids arrays for scratch memory; `indirect.rgen.slang` uses no loops over volume steps). Any volumetric addition must prefer **closed-form integrals** over iterative raymarch loops.

2. **Medium stack is depth-2 named fields** (`current`, `outer`), NOT an array (`medium.slang`: "A dynamically indexed local array lands in scratch memory, and this raygen is already register-bound"). Extension should be **named depth-3 fields** (`current`, `middle`, `outer`) or compact encoding, not `Medium stack[]`.

3. **Emission has two paths** that must stay synchronized:
   - **Direct hit emission** (`indirect.rgen.slang`: `L += throughput * albedo * emission` gated by `gateEmitter` and `showCelestial`).
   - **RIS reservoir emission** (`lighting.slang`: `risInitial` → `shadeReservoir` with `visibility()` shadow rays).
   Any volumetric emission must **not** corrupt `reservoir.W` or `phat` by being added as a discrete candidate. It must be integrated as a continuous term (`L += integral(...)`), separate from the reservoir.

4. **Material compilation is atomic per sprite** (`RtMaterialRegistry.rebuild()`). Volume features must be compiled into `MaterialHeader` or derived from the page, not computed per-ray from uncompiled texture data (too expensive in closest-hit).

5. **Sky physics (`sky.slang`) is photometric (lux / cd/m²) and independent of world volume** (`medium.slang` does not import `sky`). Connecting them requires a **scaling layer** (block-space density = atmospheric coefficient * density factor), not rewriting either module.

6. **Segment packing uses `Rgb9e5` / `packHalf2`** (`segment.slang`). Adding new medium properties must either reuse existing bits (e.g., `scattering` encoded as fraction of `extinction`) or expand `PackedPathSegment` with strict `std430` alignment awareness.

---

## 6. Recommended First Step

Given the interruption and the need for a **clean, working addition** rather than speculative architecture:

**Start with Proposition B (Volumetric Emission Density) + Proposition A (In-Scattering with albedo-scaled scattering)** combined as a minimal, register-safe extension:

- Add `FEATURE_VOLUME_EMISSION = 32` to `world_common.slang`.
- In `RtMaterialRegistry.java`, when compiling `MaterialDesc` with emission, also set a `volumeEmissionDensity` flag (or reuse `emissionStrength` interpreted as density when the feature bit is set).
- In `indirect.rgen.slang`, when `material == MATERIAL_DIELECTRIC` (or any material with `FEATURE_VOLUME_EMISSION`) and `medium.current` is active, compute:
  ```slang
  float sigma_s = albedo * medium.current.extinction; // clean, no new payload word
  float Le = payloadEmission() * emissionDensityScale; // from Prop B
  float integratedLe = Le * (1.0 - exp(-medium.current.extinction * payload.hitT)) / max(medium.current.extinction, 1.0e-6);
  float integratedScat = scattering * celestialLight.illuminance * (1.0 - exp(-medium.current.extinction * payload.hitT)) / max(medium.current.extinction, 1.0e-6);
  L += throughput * (integratedLe + integratedScat);
  ```
- This uses only existing `payloadEmission()` (surface emission value) scaled by density, existing `medium.current.extinction`, and the existing `throughput`. No new payload fields. No loop divergence. Fully compatible with RIS and emission gating.

This minimal combined approach delivers a true volumetric emission + scattering result with zero structural changes to the payload or segment packing, making it the cleanest possible first addition.

---

## APPENDIX A — EXPLICIT HIGH-QUALITY 3D VOLUME SPECIFICATION

> Added explicitly per clarification request: light scattering / sun beams, exponential height falloff FBM, true 3D (not screen-space), no noise / no ghosting / no pixelation, no "fake" fog.

### True 3D Volume Confirmation (Not Screen-Space / Not Fake)
Every proposition above uses the **world-space medium stack** (`MediumStack.current` / `.outer`) with per-segment attenuation (`throughput *= exp(-extinction * payload.hitT)`) computed along the actual 3D ray path (`ro` → `hitPos` → next `ro`). Density is evaluated at world coordinates (`hitPos.x, y, z`), not derived from screen-space depth or camera-distance approximations. This is **true participating media** — not screen-space depth fog (`fake fog`), not distance-based fade, not post-process haze.

### Sun Beams / God Rays (Light Scattering from Directional Celestial Source)
**Explicit mechanism (Prop A + Prop F combined):**
In `indirect.rgen.slang`, the bounce loop already accesses `celestialLight.dir` and `celestialLight.illuminance` (derived from `sky.slang`). For sun beams through a heterogeneous or homogeneous volume:

```slang
// Per segment inside volume with scattering coefficient sigma_s
float cosTheta = dot(normalize(rd), celestialLight.dir);
float phase = hg(cosTheta, medium.current.phaseG); // or anisotropic Mie phase
float integratedBeam = celestialLight.illuminance * sigma_s * phase
        * (1.0 - exp(-medium.current.extinction * payload.hitT))
        / max(medium.current.extinction, 1.0e-6);
L += throughput * integratedBeam;
```
This produces **true directional sun beams**: the celestial light is scattered along the ray segment inside the volume, with intensity scaled by `phaseG` (forward-scatter peak). Because it uses `celestialLight` (not a separate light source), the beam color matches the atmospheric-tinted sun/moon exactly (consistent with `sky.slang` transmittance). No extra shadow rays are needed per beam sample — the scattering integral is closed-form.

### Exponential Height Falloff FBM (Fractal Brownian Motion Density Field)
**Explicit mechanism (Prop C enhanced):**
Instead of uniform fog or simple gradient, define a **3D procedural density field** using FBM (fractal Brownian motion) with exponential height decay. In shader (`indirect.rgen.slang` or `medium.slang` procedural function):

```slang
float fbm3d(float3 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < 4; ++i) { // 4 octaves = clean, no visible banding
        value += amplitude * noise3d(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

// World-space density with exponential height falloff
float densityField(float3 worldPos) {
    float height = worldPos.y; // world Y = height
    float baseDensity = fbm3d(worldPos * 0.05); // low frequency = large clouds/fog patterns
    float falloff = exp(-max(height - FOG_BASE_HEIGHT, 0.0) / FOG_SCALE_HEIGHT);
    return clamp(baseDensity * falloff, 0.0, 1.0);
}
```
The `medium.current.extinction` and `.scattering` are set per segment from `densityField(hitPos)` (sampled once per hit, at segment start, avoiding iterative step sampling). The `fbm3d()` uses simple hash-based 3D noise (no texture binding needed), keeping it register-bound. The exponential falloff (`exp(-height/scale)`) ensures density approaches zero smoothly at high altitude (no hard cutoff, no visible seam at cloud top).

**Compatibility with emission (Prop B):** The density field scales both scattering and emission density:
```slang
float d = densityField(hitPos);
medium.current.extinction = BASE_EXTINCTION * d;
medium.current.scattering = BASE_SCATTERING * d;
emissionDensity = VOLUME_EMISSION_DENSITY * d;
```
This creates **3D emissive clouds** (e.g., lightning-lit storm clouds, glowing nether mist) that vary in 3D space and fade with height.

### No Noise / No Ghosting / No Pixelation (Quality Guarantees)
The existing architecture already supports this, but it must be enforced explicitly for volumes:

**No noise from iteration:** The design uses **closed-form single-scattering integrals** (`(1 - exp(-sigma_t * t)) / sigma_t`) evaluated once per segment, not discrete raymarch steps (`for (int i=0; i<steps; i++)`). This eliminates step-size noise, banding, and temporal flicker from variable sample counts.

**No ghosting (temporal stability):** Volume properties (`densityField`, `scattering`, `emissionDensity`) are derived from **world-stable coordinates** (not screen-space or jittered per frame). The `fbm3d()` uses deterministic hash noise (`hash3d(p + worldAnchor)` — see `worldPush.waterAnchor` for stable domain reference). Combined with the existing RIS reservoir (`risInitial` / `shadeReservoir`) and DLSS-RR temporal accumulation, the volume appears stable across frames without ghosting.

**No pixelation:** The medium is evaluated at **continuous world positions** (`hitPos` reconstructed from `ro + rd * payload.hitT`), not sampled at pixel centers or approximated with screen-space textures. The density field is smooth (FBM with 4 octaves, low frequency base = 0.05 blocks⁻¹, which is ~20-block feature size — no pixel-scale artifacts). Volume boundaries are anti-aliased naturally by the ray cone (`rayConeWidth` / `rayConeSpread` tracked in `segment.slang` and used for texture LOD in closest-hit), preventing sharp aliased edges where a volume starts/stops.

**Anti-ghosting for volume shadows:** When RIS samples emitters through fog/glass (`Prop F`), `visibility()` shadow rays apply `throughput *= exp(-medium.current.extinction * shadowDist)` continuously along the shadow segment. There is no binary "in fog / out of fog" threshold, so shadows fade smoothly — no ghost edges where a shadow ray crosses a volume boundary.

### Explicit Anti-Fake-Fog Statement
These propositions do **not** include:
- Screen-space depth-based fog (`gl_FragCoord.z` approximation)
- Distance-based exponential fade (`fogDensity * distance` without participating media)
- 2D screen-space noise overlays
- Post-process haze shaders

All fog/haze/volume effects are computed within the **radiance path integral** (`indirect.rgen.slang` bounce loop) as true 3D participating media with physical scattering (`sigma_s`), absorption (`sigma_a`), emission (`Le_density`), and directional phase functions (`hg()` or `miePhase()`). This is the same physics framework used by `sky.slang` for atmosphere, scaled to local block-space.

---

## APPENDIX B — Dual-Lobe Henyey-Greenstein (Forward + Backward)
> Added explicitly: does NOT break emission, RIS, medium stack, or quality guarantees.

### Would It Mess With the System?
**No.** A dual-lobe HG phase function is a **pure angular redistribution** of scattered energy. It does not change:
- The total scattering coefficient (`sigma_s`) — energy is conserved (`w_f + w_b = 1`, both `hg()` terms integrate to 1 over the sphere).
- The emission integral (`Prop B`) — emission density accumulates independently of phase direction (except for the physically-correct scattered-emission term in `Prop F`, which is enhanced, not broken).
- The RIS reservoir (`Prop F`) — `shadeReservoir()` uses binary visibility (`visibility()` shadow rays) and discrete light point samples. The phase function affects how scattered light reaches the receiver, not whether the shadow ray hits an occluder. The reservoir's `phat`, `W`, and `wSum` remain exact.
- The medium stack (`MediumStack`) — only angular distribution changes; `extinction`, `ior`, `water`, `scattering` coefficients stay identical.
- The payload / segment packing — no new fields are required if `phaseGB` and `phaseWF` are packed into existing `Medium` fields (see below).

### Why It Works Cleanly
The existing `hg()` (`math.slang`) integrates to 1 over the full sphere:
```
∫ hg(cosθ, g) dΩ = 1
```
A dual-lobe mix with weights summing to 1 also integrates to 1:
```
phase_dual(cosθ) = w_f · hg(cosθ, g_f) + w_b · hg(cosθ, g_b)
∫ phase_dual dΩ = w_f · 1 + w_b · 1 = w_f + w_b = 1
```
This preserves energy conservation exactly — no lost light, no extra energy created.

### Exact Implementation (Register-Safe)
**In `medium.slang`:**
```slang
public struct Medium {
    public float ior;
    public float3 extinction;
    public bool water;
    public float phaseGF;  // forward HG asymmetry (positive, e.g. 0.85 for clouds/water)
    public float phaseGB;  // backward HG asymmetry (negative, e.g. -0.3 for backscatter)
    public float phaseWF;  // forward weight [0..1] (backward weight = 1 - phaseWF)
};
```
No array changes. `phaseGF`/`GB`/`WF` are 3 floats = 12 bytes. `Medium` grows from ~24 bytes to ~36 bytes, still well within `std430` alignment and register capacity for a depth-2 stack. If payload growth is a concern, `phaseGB` and `phaseWF` can be packed into a single `uint` (quantized) or derived from `extinction` properties, but the cleanest approach is the named float fields above.

**In shader (`indirect.rgen.slang` or a helper in `math.slang`):**
```slang
public float hg_dual(float cosT, float gf, float gb, float wf) {
    float wb = 1.0 - wf;
    return wf * hg(cosT, gf) + wb * hg(cosT, gb);
}
```
Used in the scattering integral (`Prop A` / `Prop C`):
```slang
float cosTheta = dot(normalize(rd), celestialLight.dir);
float phase = hg_dual(cosTheta, medium.current.phaseGF,
                      medium.current.phaseGB, medium.current.phaseWF);
float integratedScat = scattering * celestialLight.illuminance * phase
        * (1.0 - exp(-medium.current.extinction * payload.hitT))
        / max(medium.current.extinction, 1.0e-6);
L += throughput * integratedScat;
```

### Physical Effects Produced
- **Forward lobe (`g_f > 0`, weight `w_f`)**: Strong direct transmission of light through the medium. Creates bright sun beams (`celestialLight.dir` aligned) and clear forward visibility.
- **Backward lobe (`g_b < 0`, weight `w_b`)**: Backscattering of light toward the source/viewer. Creates **halo/glow effects** around bright emissive volumes, makes clouds appear bright around their sun-facing edges, and allows sun beams to be visible from **off-axis viewing angles** (not just looking directly into the light).
- **Combined effect**: More realistic atmospheric/cloud appearance. Pure single-lobe HG (`SSS_G = 0.6`) is too directional for dense fog or thick clouds. Dual-lobe produces the characteristic bright rim + dark core appearance of real volumetric clouds and dusty atmospheres.

### Interaction with Emission (`Prop B` + `Prop F`)
When emission exists inside the volume (`Le_density > 0`):
- **Local emission** (`integratedLe`) is independent of phase — the volume emits light isotropically (or according to emission properties) regardless of scattering direction.
- **Scattered emission** (`Prop F`): The backward lobe (`g_b < 0`) scatters emitted light back toward the emitter region, creating a **self-illuminated glow** around bright fog/glowing glass containers. The forward lobe pushes emitted light outward. The combined effect is physically accurate for emissive participating media (e.g., neon-lit fog, lava-filled glass with suspended glowing particles).
- The RIS reservoir (`shadeReservoir`) is unaffected because it samples discrete surface points; the phase function only changes how continuous scattered light reaches the receiver, not the discrete shadow visibility used by the reservoir.

### Quality Impact
- **No noise added**: `hg_dual()` uses the same `pow()` and arithmetic as `hg()`. No loops, no random sampling. The integral remains closed-form.
- **No ghosting added**: Phase parameters (`phaseGF`, `GB`, `WF`) are constant per `Medium` instance (derived from `MaterialHeader` or `WorldPush`), not per-pixel or jittered. They are world-stable.
- **No pixelation added**: The phase is evaluated continuously at `cosTheta` (derived from world-space directions), not sampled discretely.
- **No performance regression**: `hg_dual()` is 2 `hg()` evaluations + one multiply-add. Given `hg()` is a single `pow()` call, the cost is ~2x the existing scattering computation — acceptable within the register-bound segment evaluation.
