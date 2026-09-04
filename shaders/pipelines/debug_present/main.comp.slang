// Computes and presents debugView content as a final inspection pass after ordinary path-traced
// radiance, RR/fallback upscale, exposure metering, and the ACES display transform. Keeping debug
// coloring out of the occupancy-bound primary raygen avoids register pressure on the normal render path.
//
// Reads the guide buffers directly at their real render resolution and nearest-samples them into the
// display-resolution output. Debug mode deliberately leaves DLSS-RR and its jitter enabled, so this
// pass shows the actual guide inputs used by RR rather than changing the renderer being inspected.

import display_common;
import bindings;

[[vk::push_constant]] DebugPresentPush pc;

static const float3 ACESCG_LUMA = float3(0.27222872, 0.67408177, 0.05368952);

// OCIO cg-config-v4.0.0 ACES 2.0: ACEScg/AP1/D60 to Linear Rec.709/D65. Debug guide colors cross
// back to the rgba8 target here; semantic debug colors (normal, depth, motion, weight) are already
// authored as literal sRGB code values and bypass this conversion.
static const float3 ACESCG_TO_BT709_R = float3( 1.70505095, -0.62179214, -0.08325887);
static const float3 ACESCG_TO_BT709_G = float3(-0.13025641,  1.14080477, -0.01054832);
static const float3 ACESCG_TO_BT709_B = float3(-0.02400336, -0.12896897,  1.15297234);

float linearToSrgbChannel(float value) {
    value = clamp(value, 0.0, 1.0);
    return value <= 0.0031308
            ? 12.92 * value
            : 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

float3 acesCgToSrgb(float3 color) {
    float3 linearBt709 = float3(
            dot(color, ACESCG_TO_BT709_R),
            dot(color, ACESCG_TO_BT709_G),
            dot(color, ACESCG_TO_BT709_B));
    return float3(
            linearToSrgbChannel(linearBt709.r),
            linearToSrgbChannel(linearBt709.g),
            linearToSrgbChannel(linearBt709.b));
}

// Display-referred, deliberately discrete one-stop bands. `ev == 0` means the value entering ACES
// is 18% grey; cool colors are below it, warm colors above it. These values are literal SDR/sRGB
// debug colors written after the display transform, not scene radiance.
float3 exposureFalseColor(float ev) {
    float band = floor(ev + 0.5);
    if (band <= -6.0) return float3(0.01, 0.00, 0.03);
    if (band == -5.0) return float3(0.12, 0.00, 0.25);
    if (band == -4.0) return float3(0.10, 0.15, 0.75);
    if (band == -3.0) return float3(0.00, 0.55, 1.00);
    if (band == -2.0) return float3(0.00, 0.75, 0.55);
    if (band == -1.0) return float3(0.10, 0.80, 0.15);
    if (band ==  0.0) return float3(0.46, 0.46, 0.46);
    if (band ==  1.0) return float3(0.78, 0.78, 0.05);
    if (band ==  2.0) return float3(1.00, 0.50, 0.00);
    if (band ==  3.0) return float3(1.00, 0.05, 0.02);
    if (band ==  4.0) return float3(1.00, 0.00, 0.70);
    if (band ==  5.0) return float3(1.00, 0.55, 0.80);
    return float3(1.00, 1.00, 1.00);
}

// Exactly the metering weight used by the histogram: local Gaussian centre weighting multiplied by
// the same frame-global sky/emissive scale computed by resolve from separate populations.
float meteringWeight(float2 uv, float depth, bool emissive) {
    float2 centered = uv * 2.0 - 1.0;
    float sigma = max(pc.centerWeightSigma, 0.01);
    float gaussian = exp(-0.5 * dot(centered, centered) / (sigma * sigma));
    float centerWeight = lerp(clamp(pc.centerWeightFloor, 0.0, 1.0), 1.0, gaussian);
    float populationWeight = depth <= 1.0e-6 ? exposureState[0].meteringSkyScale
            : (emissive ? exposureState[0].meteringEmissiveScale : 1.0);
    return centerWeight * populationWeight;
}

void fogDebugCoordinates(float2 uv, float slice, out uint x, out uint row0, out uint row1, out float blend) {
    float gx = max(pc.fogGrid.x, 1.0);
    float gy = max(pc.fogGrid.y, 1.0);
    float gz = max(pc.fogGrid.z, 1.0);
    uint atlasHeight = max(uint(max(pc.fogGrid.w, 1.0)), 1u);
    x = min(uint(clamp(uv.x, 0.0, 0.999999) * gx), uint(gx) - 1u);
    uint y = min(uint(clamp(uv.y, 0.0, 0.999999) * gy), uint(gy) - 1u);
    float z = clamp(slice, 0.0, 1.0) * (gz - 1.0);
    uint z0 = uint(floor(z));
    uint z1 = min(z0 + 1u, uint(gz - 1.0));
    blend = z - float(z0);
    row0 = min(y + z0 * uint(gy), atlasHeight - 1u);
    row1 = min(y + z1 * uint(gy), atlasHeight - 1u);
}

float4 fogDebugDirectSample(float2 uv, float slice) {
    uint x, row0, row1;
    float blend;
    fogDebugCoordinates(uv, slice, x, row0, row1, blend);
    return lerp(fogDirect[int2(x, row0)], fogDirect[int2(x, row1)], blend);
}

float4 fogDebugLocalSample(float2 uv, float slice) {
    uint x, row0, row1;
    float blend;
    fogDebugCoordinates(uv, slice, x, row0, row1, blend);
    return lerp(fogLocal[int2(x, row0)], fogLocal[int2(x, row1)], blend);
}

float4 fogDebugGiSample(float2 uv, float slice) {
    uint x, row0, row1;
    float blend;
    fogDebugCoordinates(uv, slice, x, row0, row1, blend);
    return lerp(fogGi[int2(x, row0)], fogGi[int2(x, row1)], blend);
}

float4 fogDebugCacheSample(float2 uv, float slice) {
    uint x, row0, row1;
    float blend;
    fogDebugCoordinates(uv, slice, x, row0, row1, blend);
    return lerp(fogCache[int2(x, row0)], fogCache[int2(x, row1)], blend);
}

float4 fogDebugAuxSample(float2 uv, float slice) {
    uint x, row0, row1;
    float blend;
    fogDebugCoordinates(uv, slice, x, row0, row1, blend);
    return lerp(fogAux[int2(x, row0)], fogAux[int2(x, row1)], blend);
}

float3 fogDebugRadiance(float3 stored) {
    return acesCgToSrgb(max(stored * 4.0, float3(0.0)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    int2 pix = int2(dispatchId.xy);
    uint w, h;
    outputImage.GetDimensions(w, h);
    if (pix.x >= int(w) || pix.y >= int(h)) {
        return;
    }

    uint guideW, guideH;
    gNormal.GetDimensions(guideW, guideH);
    float2 uv = (float2(pix) + 0.5) / float2(w, h);
    int2 guidePix = min(int2(uv * float2(guideW, guideH)), int2(guideW, guideH) - 1);
    float2 motionToDisplay = float2(w, h) / float2(guideW, guideH);

    float4 normalRough = gNormal[guidePix];
    float3 dbg;
    if (pc.debugView == 8u) {
        float luminance = dot(max(sceneImage[pix].rgb, float3(0.0)), ACESCG_LUMA);
        float exposedLuminance = luminance * max(exposureImage[int2(0, 0)], 0.0);
        float evFromMidGrey = log2(max(exposedLuminance, 1.0e-6) / 0.18);
        dbg = exposureFalseColor(evFromMidGrey);
    } else if (pc.debugView == 9u) {
        float weight = meteringWeight(uv, gDepth[guidePix], gAlbedo[guidePix].a > 0.5);
        dbg = float3(weight, weight, weight);
    } else if (pc.debugView == 1u) {
        dbg = normalRough.xyz * 0.5 + 0.5;
    } else if (pc.debugView == 2u) {
        dbg = acesCgToSrgb(gAlbedo[guidePix].rgb);
    } else if (pc.debugView == 3u) {
        float depth = gDepth[guidePix];
        dbg = float3(depth, depth, depth);
    } else if (pc.debugView == 4u) {
        dbg = float3(normalRough.w, normalRough.w, normalRough.w);
    } else if (pc.debugView == 6u) {
        dbg = acesCgToSrgb(gSpecAlbedo[guidePix].rgb);
    } else if (pc.debugView == 7u) {
        dbg = float3(clamp(0.5 + gSpecMotion[guidePix] * motionToDisplay * 0.05, 0.0, 1.0), 0.5);
    } else if (pc.debugView == 10u) {
        dbg = fogDebugRadiance(fogDebugDirectSample(uv, 0.5).rgb);
    } else if (pc.debugView == 11u) {
        dbg = fogDebugRadiance(fogDebugLocalSample(uv, 0.5).rgb);
    } else if (pc.debugView == 12u) {
        dbg = fogDebugRadiance(fogDebugGiSample(uv, 0.5).rgb);
    } else if (pc.debugView == 13u) {
        dbg = fogDebugRadiance(fogDebugCacheSample(uv, 0.5).rgb);
    } else if (pc.debugView == 14u) {
        float4 guide = fogDebugAuxSample(uv, 0.5);
        dbg = float3(guide.y, clamp(guide.z / 32.0, 0.0, 1.0), guide.w);
    } else if (pc.debugView == 15u) {
        float4 guide = fogDebugAuxSample(uv, 0.5);
        float4 directGuide = fogDebugDirectSample(uv, 0.5);
        // Red = normalized hit distance, green = surface occupancy, blue = reversed-Z primary depth.
        dbg = float3(clamp(guide.x * 2.0, 0.0, 1.0), guide.y, directGuide.a);
    } else {
        dbg = float3(clamp(0.5 + gMotion[guidePix] * motionToDisplay * 0.05, 0.0, 1.0), 0.5);
    }
    outputImage[pix] = float4(clamp(dbg, 0.0, 1.0), 1.0);
}
