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

## Volumetric frame graph

Atmospheric fog uses a camera-frustum froxel grid with fixed `64 x 32 x 64`,
`96 x 48 x 96`, and `128 x 64 x 128` quality presets. The default 128-block
range and linear 128-slice High preset provide one-block average longitudinal
resolution. The grid is rebuilt in full every frame and owns only three 3D
images: source/extinction, bilateral-filtered source/extinction, and cumulative
in-scattering/transmittance. There is no temporal volume history or
reprojection.

The world frame records these stages in order:

1. the primary trace writes the first-interface `volumeDepth` independently of
   the transmitted DLSS depth guide;
2. froxel injection evaluates static world-space fBm density, atmosphere-LUT
   ambient light, one deterministic TLAS visibility ray for the active sun or
   moon, and bounded proposals from the local coloured area-emitter hierarchy;
3. a deterministic 3x3x3 extinction-aware filter rejects geometry/density
   boundaries;
4. each XY column is prefix-integrated with the analytic constant-medium
   Beer-Lambert solution;
5. the indirect surface trace and DLSS Ray Reconstruction (or fallback upscale)
   produce the display-resolution scene;
6. a dedicated display-resolution raygen reconstructs the integrated volume
   trilinearly and composites into `rrOutput` before exposure and bloom.

The existing GPU light hierarchy is the technically available stable source for
block-local fog lighting. Minecraft's scalar 0-15 block-light field is not
currently uploaded as a GPU volume, so froxels do not perform an unavailable
scalar lookup. Coloured emitter use is local-only, finite-area, source-clamped,
first-interface-clipped, and spatially edge-filtered to avoid singular weights,
fireflies, and fog leaks. Atmospheric froxels are disabled while submerged.

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
