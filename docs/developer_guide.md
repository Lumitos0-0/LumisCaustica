# Developer Guide

## Windows

1. Install the Vulkan SDK from <https://vulkan.lunarg.com/sdk/home>.
   The installer sets `VULKAN_SDK` automatically.
2. Download the DLSS SDK from <https://github.com/NVIDIA/DLSS/releases>.
   Extract it, then set `DLSS_SDK` to the folder you extracted.

   To set it permanently for your Windows user account, run PowerShell with:

   ```powershell
   [Environment]::SetEnvironmentVariable("DLSS_SDK", "C:\path\to\dlss-sdk", "User")
   ```

   Restart your terminal after setting it. To set it only for the current
   PowerShell session, use:

   ```powershell
   $env:DLSS_SDK = "C:\path\to\dlss-sdk"
   ```

3. Configure and build the native shim:

```powershell
cmake -S native/ngx_shim -B build/cmake/ngx_shim/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake/ngx_shim/release --config Release
```

4. Run the client:

```powershell
$env:JAVA_TOOL_OPTIONS = "-Xmx8G -XX:+UseCompactObjectHeaders -XX:+AlwaysPreTouch -XX:+UseStringDeduplication -XX:+UseZGC"
.\gradlew.bat runClient --args="--renderDebugLabels --graphicsBackend VULKAN"
```

## Linux

Set `DLSS_SDK` and `VULKAN_SDK` before configuring CMake:

```bash
export DLSS_SDK=/path/to/dlss-sdk
export VULKAN_SDK=/path/to/vulkan-sdk
```

`DLSS_SDK` must contain the NGX headers and static library. `VULKAN_SDK` must
contain Vulkan headers.

Then configure and build the native shim:

```bash
cmake -S native/ngx_shim -B build/cmake/ngx_shim/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake/ngx_shim/release
```

On NixOS, enter the development shell from `flake.nix` instead of setting up
the toolchain by hand:

```bash
nix develop
cmake -S native/ngx_shim -B build/cmake/ngx_shim/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake/ngx_shim/release
```

## Native Bundling

Gradle bundles NGX natives for the current host platform by default:

```bash
./gradlew build
```

Release builds that already have both platform shims available can request a
cross-platform native bundle:

```bash
./gradlew build -PngxPlatforms=windows-x64,linux-x64
```

Run the Vulkan RT/DLSS-RR client with:

```bash
JAVA_TOOL_OPTIONS='-Xmx8G -XX:+UseCompactObjectHeaders -XX:+AlwaysPreTouch -XX:+UseStringDeduplication -XX:+UseZGC' nvidia-offload ./gradlew runClient --args='--renderDebugLabels --graphicsBackend VULKAN'
```

## Froxel Volumetrics

The fog implementation is split by responsibility:

- `rt/volumetric/RtFroxelGrid` owns CPU-side grid and nonlinear depth math.
- `rt/volumetric/RtVolumetrics` owns the first-interface depth image, two temporal
  scattering volumes, the integrated volume, and pass scheduling.
- `world/froxel.slang` owns frustum mapping, density, reprojection, filtering,
  and scene-radiance composition.
- `world/volumetric_lighting.slang` injects sky and block-emitter lighting using
  the world pipeline's existing TLAS and power-weighted light hierarchy.
- `world/volumetric_inject.rgen.slang`, `world/volumetric_filter.rgen.slang`,
  and `world/volumetric_integrate.rgen.slang` are the GPU entry points.

With the default `grid-pixel-size = 16` and `depth-slices = 48`, Performance,
Balanced, High, Ultra, and Ultra+ use `/18 × 44`, `/14 × 56`, `/10 × 72`,
`/7 × 88`, and `/5 × 112` respectively. They average 1, 1, 2, 2, and 3
independently sampled and shadowed emitter reservoirs per froxel in both moving and
stationary views. This shifts the quality budget away from repeated light rays and
into real XYZ volume resolution. The advanced grid settings remain the scale baseline
for every preset. History weight `w` averages `(1 + w) / (1 - w)` frames, and the
default 0.75 averages about seven, so a new light or shadow settles in roughly ten
frames instead of smearing the volume through repeated trilinear resampling. High,
Ultra, and Ultra+ cap it at 0.65, 0.55, and 0.45 to retain more of that spatial detail.
Deterministic celestial visibility, temporal accumulation, and a centre-weighted
3×3 filter denoise the field without changing its continuous quadrilinear reconstruction.

Depth boundaries follow `distance = maxDistance × (slice / sliceCount)^exponent`,
which concentrates samples near the camera. Injection writes linear
`{scattering.rgb, extinction}`. A second pass analytically integrates each
constant-medium segment with Beer-Lambert transmittance and writes cumulative
`{inScattering.rgb, transmittance}`. The indirect ray-generation pass samples
that result at the primary ray's first interface before pre-exposure. Keeping a
separate first-interface depth prevents clear glass or water's behind-surface
DLSS guide depth from replacing the fog composition endpoint.

The participating-media field itself is independent of scene depth: every froxel is
populated, Gaussian-filtered, and quadrilinearly reconstructed at each pixel's exact
native-resolution depth. This continuity is what hides individual frustum cells.
Pass A runs before injection only so local block-emitter lighting can be suppressed
when a stochastic sample lies behind the first camera surface. That targeted gate
prevents a partially occupied slice from importing underground lava radiance into
its visible segment without carving empty geometry-shaped columns into the volume.

The grid uses the same atmosphere-derived dominant celestial light, TLAS
visibility routine, transparent-shadow handling, and emissive-block light
records as surface path tracing. While submerged, the froxel extinction coefficient
represents particulate out-scattering; the path tracer continues to own colored water
absorption, so the two are not double-counted. Directional light is attenuated from the
water surface to each froxel and modulated by the analytic wave-Jacobian caustic already
used on underwater receivers. This projects animated focusing bands through the water
volume while retaining TLAS shadows. Temporal reprojection ping-pongs the injection
volumes; camera cuts, skipped frames, medium changes, and optical-setting changes
invalidate history.

Runtime controls expose enable/disable, extinction, directional light-shaft strength,
shaft focus, and the five quality presets. Strength multiplies only sun/moon
in-scattering. Focus is a dedicated Henyey-Greenstein anisotropy that concentrates
that scattering toward the celestial direction. Neither changes extinction, ambient
fog, emissive blocks, or surface lighting. Underwater Fog controls particulate
scattering and Water Caustics controls wave focusing in the submerged volume. The
complete `[volumetrics]` TOML surface also includes the Balanced grid baseline, depth
distribution, maximum distance, single-scattering albedo, Henyey-Greenstein
anisotropy, height falloff, noise, temporal weight, and local-light candidate count.
Moonlight uses a luminance-preserving cool BT.709 tint shared by the atmosphere LUT,
surface NEE, and volumetric shafts; the look package's moon lux remains unchanged.

The design follows these references:

- Bart Wronski, *Volumetric Fog: Unified, Compute Shader Based Solution to
  Atmospheric Scattering* (SIGGRAPH 2014):
  <https://bartwronski.files.wordpress.com/2014/08/bwronski_volumetric_fog_siggraph2014.pdf>
- Sébastien Hillaire, *Physically Based and Unified Volumetric Rendering in
  Frostbite* (SIGGRAPH 2015):
  <https://advances.realtimerendering.com/s2015/>
- NVIDIA RTX Remix volumetric controls and defaults:
  <https://docs.omniverse.nvidia.com/kit/docs/rtx_remix/latest/docs/runtimeinterface/renderingtab/remix-runtimeinterface-rendering-volumetrics.html>
- NVIDIA RTX Remix's open-source froxel mapping and integration passes:
  <https://github.com/NVIDIAGameWorks/dxvk-remix/tree/main/src/dxvk/shaders/rtx/pass/volumetrics>
