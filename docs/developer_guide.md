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
`96 x 48 x 96`, and capped `128 x 64 x 128` spatial presets. Ultra retains the
same cap and raises bounded emitter proposal/estimate counts. The default
128-block range and linear 128-slice High preset provide one-block average
longitudinal resolution. Injection is rebuilt from deterministic world-space
samples every frame. Resolved lighting and explicit first-interface depth use
two-volume ping-pong only for a subtle temporal blend (15% by default, hard
maximum 35%).

The world frame records these stages in order:

1. the primary trace writes the first-interface `volumeDepth` independently of
   the transmitted DLSS depth guide;
2. froxel injection evaluates static world-space fBm density, atmosphere-LUT
   ambient light, one deterministic TLAS visibility ray for the active sun or
   moon, bounded local coloured area emitters, and a 3D geometry-depth volume;
3. a deterministic 3x3x3 filter uses smooth first-interface depth differences,
   not extinction differences, so fBm does not outline block edges;
4. the current field is reprojected into the previous frustum, sampled
   trilinearly, neighborhood-clamped, and rejected where previous depth reports
   an occlusion or disocclusion;
5. each XY column is prefix-integrated with the analytic constant-medium
   Beer-Lambert solution;
6. the indirect surface trace and DLSS Ray Reconstruction (or fallback upscale)
   produce the display-resolution scene;
7. a dedicated display-resolution raygen reconstructs the integrated volume
   trilinearly and composites into `rrOutput` before exposure and bloom.

The existing GPU light hierarchy is the technically available stable source for
block-local fog lighting. Minecraft's scalar 0-15 block-light field is not
currently uploaded as a GPU volume, so froxels do not perform an unavailable
scalar lookup. Coloured emitter use is local-only, finite-area, source-clamped,
first-interface-clipped, and geometry-depth-filtered to avoid singular weights,
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
