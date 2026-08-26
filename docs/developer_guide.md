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
- `rt/volumetric/RtVolumetrics` owns the first-interface depth image, the scattering
  volume, the filtered volume, and pass scheduling.
- `world/froxel.slang` owns frustum mapping, density, stochastic raymarching integration,
  and scene-radiance composition.
- `world/volumetric_lighting.slang` injects sky and block-emitter lighting using
  the world pipeline's existing TLAS and power-weighted light hierarchy.
- `world/volumetric_inject.rgen.slang` and `world/volumetric_filter.rgen.slang`
  are the GPU froxel entry points.

The system evaluates raw, physically uniform participating media without procedural
noise blobs, arbitrary quality scaling, or un-shadowed ambient glow in dark interiors.
Deterministic celestial visibility and cell-centered injection keep the 3D media
field temporally stable.

Depth boundaries follow `distance = maxDistance × (slice / sliceCount)^exponent`,
which concentrates samples near the camera with a quadratic fast path when `exponent = 2.0`.
Injection writes linear `{scattering.rgb, extinction}`. The volume is integrated
along each primary ray via stochastic raymarching in the indirect ray-generation pass. For
each view ray, the march advances slice by slice, evaluating the analytic Beer-Lambert
segment integral and terminating early when reaching the first surface depth recorded in
`volumeDepth` or when transmittance drops below threshold. This eliminates temporal boiling
and motion shimmering while allowing DLSS Ray Reconstruction to reconstruct sharp, temporally
stable light shafts and volumetric shadows without temporal accumulation lag.

Pass A runs before injection so directional and local block-emitter lighting are suppressed
when a sample lies behind the first camera surface. That targeted gate prevents light from
leaking through solid walls, cave ceilings, or mountain geometry.

The grid uses the same atmosphere-derived dominant celestial light, TLAS
visibility routine, transparent-shadow handling, and emissive-block light
records as surface path tracing. Directional lighting uses a dual-lobe Henyey-Greenstein
phase function blending a sharp forward solar corona lobe with a broad back-scattering
diffuse lobe to simulate multiple scattering in single-bounce medium injection.
While submerged, the froxel extinction coefficient represents particulate out-scattering;
the path tracer continues to own colored water absorption, so the two are not double-counted.
Directional light is attenuated from the water surface to each froxel and modulated by
analytic wave-Jacobian caustics evaluated across wavelength-dependent refractive indices
(Red 1.331, Green 1.333, Blue 1.338) from a single shared wave gradient. This projects
shimmering chromatic dispersion bands through the water volume while retaining TLAS shadows.
Inter-pass synchronization uses targeted `VkImageMemoryBarrier` calls on the volume images
rather than global pipeline flushes.

Runtime controls expose enable/disable, extinction, directional light-shaft strength,
shaft focus, and the five quality presets. Strength multiplies only sun/moon
in-scattering. Focus is a dedicated Henyey-Greenstein anisotropy that concentrates
that scattering toward the celestial direction. Neither changes extinction, ambient
fog, emissive blocks, or surface lighting. Underwater Fog controls particulate
scattering and Water Caustics controls wave focusing in the submerged volume. The
complete `[volumetrics]` TOML surface also includes the Balanced grid baseline, depth
distribution, maximum distance, single-scattering albedo, Henyey-Greenstein
anisotropy, height falloff, noise, and local-light candidate count.
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
