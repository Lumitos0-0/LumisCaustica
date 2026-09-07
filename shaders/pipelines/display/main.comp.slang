// Maps the display-res scene-linear ACEScg RT image to sRGB SDR and, when enabled, PQ/BT.2020 HDR,
// via baked ACES 2.0 display-transform LUTs (see tools/bake_display_lut.py). Tonemap seam: the path tracer and DLSS-RR use scene-linear
// ACEScg; exposure is applied from the compositor-owned 1x1 image right here, followed by the selected
// scene-referred ACES look (LMT), then the shared ACES 2.0 output transform for each display.

import display_common;
import world_common;
import volumetric_common;
import bindings;

[[vk::push_constant]] DisplayPush pc;

// MUST exactly match SHAPER_LO_STOPS/SHAPER_HI_STOPS in tools/bake_display_lut.py -- these are the two
// halves of one shaper function, one run at bake time (Python), one at sample time (here); if they
// drift the LUT fetch reads the wrong shelf of the LUT everywhere, silently.
static const float LUT_SHAPER_LO_STOPS = -12.0;
static const float LUT_SHAPER_HI_STOPS = 12.0;

float3 shaperEncode(float3 linearColor) {
    float3 stops = log2(max(linearColor, float3(1.0e-6)));
    return clamp((stops - LUT_SHAPER_LO_STOPS) / (LUT_SHAPER_HI_STOPS - LUT_SHAPER_LO_STOPS), 0.0, 1.0);
}

// Edge-aligned LUT: texel 0's centre is uvw=0.0 and texel (lutSize-1)'s centre is uvw=1.0, but a
// sampler's normalized [0,1] domain spans texel *centres* at 0.5/N .. 1-0.5/N by default. Remap into
// that convention; RtToneLut's CLAMP_TO_EDGE then holds the boundary texel for any exposed value
// outside the shaper's +/-12 EV range instead of wrapping. toneLut/hdrToneLut share pc.lutSize;
// the independently sized lookLut passes pc.lookLutSize.
float3 lutTexCoord(float3 uvw, float size) {
    return uvw * ((size - 1.0) / size) + float3(0.5 / size);
}

float3 shaperDecode(float3 shaped) {
    return exp2(lerp(
            float3(LUT_SHAPER_LO_STOPS),
            float3(LUT_SHAPER_HI_STOPS),
            clamp(shaped, 0.0, 1.0)));
}

float3 applyLook(float3 exposedAcesCg) {
    if (pc.lookEnabled == 0) {
        return exposedAcesCg;
    }
    float3 uvw = shaperEncode(exposedAcesCg);
    float3 shaped = lookLut.SampleLevel(lutTexCoord(uvw, pc.lookLutSize), 0.0).rgb;
    return shaperDecode(shaped);
}

float3 sampleBloom(int2 outputPixel, uint outputWidth, uint outputHeight) {
    float2 uv = (float2(outputPixel) + 0.5) / float2(outputWidth, outputHeight);
    return bloomImage.SampleLevel(uv, 0.0).rgb;
}

// Artistic gamma after the view/display transform. Apply the curve to display luminance and scale
// RGB uniformly instead of exponentiating each channel: this lifts shadows/midtones without pulling
// chromaticities toward white. The uniform scale is limited at the display boundary so saturated
// colors preserve their channel ratios instead of clipping individual channels.
static const float3 BT709_LUMA = float3(0.2126, 0.7152, 0.0722);
static const float3 BT2020_LUMA = float3(0.2627, 0.6780, 0.0593);

float3 luminanceGamma(float3 linearDisplayColor, float3 lumaCoefficients) {
    float3 color = clamp(linearDisplayColor, 0.0, 1.0);
    float luminance = dot(color, lumaCoefficients);
    if (luminance <= 0.0) {
        return color;
    }

    float adjustedLuminance = pow(luminance, clamp(pc.gamma, 0.1, 5.0));
    float scale = adjustedLuminance / luminance;
    float maxChannel = max(color.r, max(color.g, color.b));
    float gamutLimitedScale = 1.0 / max(maxChannel, 1.0e-8);
    return color * min(scale, gamutLimitedScale);
}

float srgbDecodeChannel(float code) {
    return code <= 0.04045 ? code / 12.92 : pow((code + 0.055) / 1.055, 2.4);
}

float srgbEncodeChannel(float linearValue) {
    return linearValue <= 0.0031308
            ? 12.92 * linearValue
            : 1.055 * pow(linearValue, 1.0 / 2.4) - 0.055;
}

float3 srgbDecode(float3 code) {
    return float3(srgbDecodeChannel(code.r), srgbDecodeChannel(code.g), srgbDecodeChannel(code.b));
}

float3 srgbEncode(float3 linearColor) {
    return float3(
            srgbEncodeChannel(linearColor.r),
            srgbEncodeChannel(linearColor.g),
            srgbEncodeChannel(linearColor.b));
}

float3 displayGammaSdr(float3 displayColor) {
    float3 code = clamp(displayColor, 0.0, 1.0);
    if (abs(pc.gamma - 1.0) < 1.0e-6) {
        return code;
    }
    return clamp(srgbEncode(luminanceGamma(srgbDecode(code), BT709_LUMA)), 0.0, 1.0);
}

static const float PQ_M1 = 0.1593017578125;
static const float PQ_M2 = 78.84375;
static const float PQ_C1 = 0.8359375;
static const float PQ_C2 = 18.8515625;
static const float PQ_C3 = 18.6875;

float3 pqDecode(float3 code) {
    float3 p = pow(clamp(code, 0.0, 1.0), float3(1.0 / PQ_M2));
    float3 y = max((p - PQ_C1) / max(PQ_C2 - PQ_C3 * p, float3(1.0e-6)), float3(0.0));
    return pow(y, float3(1.0 / PQ_M1)) * 10000.0;
}

float3 pqEncode(float3 nits) {
    float3 y = pow(max(nits, float3(0.0)) / 10000.0, float3(PQ_M1));
    return pow((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y), float3(PQ_M2));
}

float3 displayGammaHdr(float3 pqColor) {
    if (abs(pc.gamma - 1.0) < 1.0e-6) {
        return clamp(pqColor, 0.0, 1.0);
    }
    float peak = max(pc.hdrPeakNits, 1.0);
    float3 normalizedNits = clamp(pqDecode(pqColor) / peak, 0.0, 1.0);
    float3 adjustedNits = peak * luminanceGamma(normalizedNits, BT2020_LUMA);
    return clamp(pqEncode(adjustedNits), 0.0, 1.0);
}

// Exposure -> log2 shaper -> trilinear LUT fetch. The LUT already bakes in ACES 2.0's gamut mapping to
// BT.709 and the sRGB OETF, so its output goes straight to outputImage. Single mip (RtToneLut), so an
// explicit level-0 SampleLevel is exact, not an approximation.
float3 tonemap(float3 lookedAcesCg) {
    float3 uvw = shaperEncode(lookedAcesCg);
    return displayGammaSdr(toneLut.SampleLevel(lutTexCoord(uvw, pc.lutSize), 0.0).rgb);
}

// Same shaper as tonemap() but sampling the HDR LUT, whose range is already PQ-encoded BT.2020 -- no
// separate encode or UI-brightness mapping needed.
float3 tonemapHdr(float3 lookedAcesCg) {
    float3 uvw = shaperEncode(lookedAcesCg);
    return displayGammaHdr(hdrToneLut.SampleLevel(lutTexCoord(uvw, pc.lutSize), 0.0).rgb);
}

float sampleDepthBilateral(float2 uv) {
    uint depthW, depthH;
    gDepth.GetDimensions(depthW, depthH);
    float2 texelPos = uv * float2(depthW, depthH) - 0.5;
    int2 base = int2(floor(texelPos));
    float2 f = frac(texelPos);

    int2 p00 = clamp(base + int2(0, 0), int2(0), int2(depthW - 1, depthH - 1));
    int2 p10 = clamp(base + int2(1, 0), int2(0), int2(depthW - 1, depthH - 1));
    int2 p01 = clamp(base + int2(0, 1), int2(0), int2(depthW - 1, depthH - 1));
    int2 p11 = clamp(base + int2(1, 1), int2(0), int2(depthW - 1, depthH - 1));

    float d00 = gDepth[p00];
    float d10 = gDepth[p10];
    float d01 = gDepth[p01];
    float d11 = gDepth[p11];

    // Detect silhouette edge discontinuities
    float minD = min(min(d00, d10), min(d01, d11));
    float maxD = max(max(d00, d10), max(d01, d11));
    if (maxD - minD > 0.01) {
        return (f.x < 0.5) ? ((f.y < 0.5) ? d00 : d01) : ((f.y < 0.5) ? d10 : d11);
    }
    return lerp(lerp(d00, d10, f.x), lerp(d01, d11, f.x), f.y);
}

float getLinearDepth(float depthHardware, float2 uv, float4x4 invViewProj) {
    if (depthHardware <= 1.0e-6) {
        return FROXEL_FAR;
    }
    float2 ndc = uv * 2.0 - 1.0;
    // Reversed-Z: depthHardware is in [0, 1] where 1 is near and 0 is far
    float4 clip = mul(invViewProj, float4(ndc.x, ndc.y, depthHardware, 1.0));
    if (clip.w <= 0.0) return FROXEL_FAR;
    float3 camRel = clip.xyz / clip.w;
    return clamp(length(camRel), FROXEL_NEAR, FROXEL_FAR);
}

float4 sampleVolumetricFog(int2 pix, float2 uv, float hardwareDepth, float4x4 invViewProj, float ditherStrength) {
    float linearDepth = getLinearDepth(hardwareDepth, uv, invViewProj);
    // Screen-space dither on slice depth breaks discrete concentric banding
    float dither = (interleavedGradientNoise(float2(pix), 0u) - 0.5) * max(ditherStrength, 0.0);
    float slice = depthToFroxelSlice(linearDepth, FROXEL_DIMS.z) + dither * 0.75;
    float3 uvw = float3(uv, (slice + 0.5) / float(FROXEL_DIMS.z));
    return volumetricLut.SampleLevel(clamp(uvw, float3(0.0), float3(1.0)), 0.0);
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

    float4 rt = rtImage[pix];
    float exposure = max(exposureImage[int2(0, 0)], 0.0);
    float3 exposedAcesCg = max(rt.rgb * exposure, float3(0.0));
    exposedAcesCg += sampleBloom(pix, w, h) * max(pc.bloomStrength, 0.0);

    if (pc.volumetricEnabled != 0 && pc.worldPushAddr != 0) {
        float2 uv = (float2(pix) + 0.5) / float2(w, h);
        float depth = sampleDepthBilateral(uv);
        WorldPush wp = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
        float4 fog = sampleVolumetricFog(pix, uv, depth, wp.invViewProj, pc.ditherStrength);
        exposedAcesCg = exposedAcesCg * clamp(fog.a, 0.0, 1.0) + max(fog.rgb, float3(0.0)) * exposure;
    }

    float3 lookedAcesCg = applyLook(exposedAcesCg);
    outputImage[pix] = float4(tonemap(lookedAcesCg), 1.0);

    if (pc.hdrEnabled != 0) {
        hdrImage[pix] = float4(tonemapHdr(lookedAcesCg), 1.0);
    }
}
