// Maps the display-res scene-linear ACEScg RT image to sRGB SDR and, when enabled, PQ/BT.2020 HDR,
// via baked ACES 2.0 display-transform LUTs (see tools/bake_display_lut.py). Tonemap seam: the path tracer and DLSS-RR use scene-linear
// ACEScg; exposure is applied from the compositor-owned 1x1 image right here, followed by the selected
// scene-referred ACES look (LMT), then the shared ACES 2.0 output transform for each display.

import display_common;
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

float fogDepthMismatch(float sceneDepth, float fogDepth) {
    bool sceneSky = sceneDepth <= 1.0e-6;
    bool fogSky = fogDepth <= 1.0e-6;
    if (sceneSky || fogSky) {
        return sceneSky == fogSky ? 0.0 : 1.0;
    }
    return abs(sceneDepth - fogDepth);
}

float4 chooseFogLayer(float sceneDepth, float2 fogDepths, float4 fogFront, float4 fogBack,
                      out float bestMismatch) {
    float frontMismatch = fogDepthMismatch(sceneDepth, fogDepths.x);
    float backMismatch = fogDepthMismatch(sceneDepth, fogDepths.y);
    bool useBack = backMismatch + 1.0e-6 < frontMismatch;
    bestMismatch = useBack ? backMismatch : frontMismatch;
    return useBack ? fogBack : fogFront;
}

float4 sampleFog(int2 pix, uint w, uint h) {
    if (pc.fogEnabled == 0) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    uint fogW, fogH;
    fogFrontImage.GetDimensions(fogW, fogH);
    float2 uv = (float2(pix) + 0.5) / float2(w, h);

    // A low-res fog texel can straddle two distinct depth layers at silhouettes (foreground surface plus
    // background/sky). The integration pass therefore stores both a front and back fog solution per texel;
    // the display pass picks whichever layer matches the full-res scene depth best before doing the usual
    // depth-aware bilateral upsample. This attacks the representative-depth failure directly instead of
    // trying to recover a lost layer from a single low-res fog sample.
    float sceneDepth = sceneDepthImage.SampleLevel(uv, 0.0).r;
    float2 fogTexelPos = uv * float2(fogW, fogH);
    float2 fogPos = fogTexelPos - 0.5;
    int2 base = int2(floor(fogPos));
    float2 fracPos = frac(fogPos);

    float4 accum = float4(0.0, 0.0, 0.0, 0.0);
    float totalWeight = 0.0;
    float bestMismatch = 1.0e9;
    float4 bestFog = float4(0.0, 0.0, 0.0, 1.0);

    [unroll]
    for (int oy = 0; oy < 2; ++oy) {
        [unroll]
        for (int ox = 0; ox < 2; ++ox) {
            int2 coord = clamp(base + int2(ox, oy), int2(0, 0), int2(int(fogW), int(fogH)) - 1);
            float2 fogUv = (float2(coord) + 0.5) / float2(fogW, fogH);
            float2 fogDepths = fogDepthImage.SampleLevel(fogUv, 0.0).rg;
            float4 fogFront = fogFrontImage.SampleLevel(fogUv, 0.0);
            float4 fogBack = fogBackImage.SampleLevel(fogUv, 0.0);
            float mismatch;
            float4 fog = chooseFogLayer(sceneDepth, fogDepths, fogFront, fogBack, mismatch);
            float bilinear = (ox != 0 ? fracPos.x : 1.0 - fracPos.x)
                    * (oy != 0 ? fracPos.y : 1.0 - fracPos.y);
            if (mismatch < bestMismatch) {
                bestMismatch = mismatch;
                bestFog = fog;
            }
            float depthWeight = exp2(-mismatch * 512.0);
            float weight = bilinear * depthWeight;
            accum += fog * weight;
            totalWeight += weight;
        }
    }

    if (totalWeight >= 0.35) {
        return accum / totalWeight;
    }

    float4 wideAccum = float4(0.0, 0.0, 0.0, 0.0);
    float wideWeight = 0.0;
    int2 center = int2(floor(fogPos + 0.5));
    [unroll]
    for (int oy = -1; oy <= 1; ++oy) {
        [unroll]
        for (int ox = -1; ox <= 1; ++ox) {
            int2 coord = clamp(center + int2(ox, oy), int2(0, 0), int2(int(fogW), int(fogH)) - 1);
            float2 fogUv = (float2(coord) + 0.5) / float2(fogW, fogH);
            float2 fogDepths = fogDepthImage.SampleLevel(fogUv, 0.0).rg;
            float4 fogFront = fogFrontImage.SampleLevel(fogUv, 0.0);
            float4 fogBack = fogBackImage.SampleLevel(fogUv, 0.0);
            float mismatch;
            float4 fog = chooseFogLayer(sceneDepth, fogDepths, fogFront, fogBack, mismatch);
            if (mismatch < bestMismatch) {
                bestMismatch = mismatch;
                bestFog = fog;
            }
            float2 delta = (float2(coord) + 0.5) - fogTexelPos;
            float spatialWeight = exp2(-dot(delta, delta));
            float depthWeight = 1.0 / (mismatch + 1.0e-5);
            float weight = spatialWeight * depthWeight;
            wideAccum += fog * weight;
            wideWeight += weight;
        }
    }

    return wideWeight > 1.0e-5 ? wideAccum / wideWeight : bestFog;
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
    float4 fog = sampleFog(pix, w, h);
    float3 fogScattering = fog.rgb;
    float fogTransmittance = fog.a;

    // Composite volumetric fog: scene * transmittance + scattering, both in scene-linear ACEScg
    float3 foggedScene = rt.rgb * fogTransmittance + fogScattering;

    float exposure = max(exposureImage[int2(0, 0)], 0.0);
    float3 exposedAcesCg = max(foggedScene * exposure, float3(0.0));
    exposedAcesCg += sampleBloom(pix, w, h) * max(pc.bloomStrength, 0.0);
    float3 lookedAcesCg = applyLook(exposedAcesCg);
    outputImage[pix] = float4(tonemap(lookedAcesCg), 1.0);

    if (pc.hdrEnabled != 0) {
        hdrImage[pix] = float4(tonemapHdr(lookedAcesCg), 1.0);
    }
}
