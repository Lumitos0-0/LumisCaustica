// Push-constant structs shared between the display-chain compute passes and
// layout/layout_probe.slang's build-time reflection probe (see GenerateShaderRecords.groovy).
// Each struct is imported into exactly one production shader AND probed here so its Java writer is
// generated from the compiler's reflected offsets, preventing Java/shader ABI drift when fields move.
//
// Each struct here needs a distinct name because reflection JSON parameters are named by struct type across the whole
// probe module, and per-entry-point SPIR-V allows only one push-constant block reachable from any
// single entry point, so the probe needs one small dedicated entry point per struct.

public struct ExposureHistPush {
    public uint stride;
    public float centerWeightSigma;
    public float centerWeightFloor;
};

public struct ExposureResolvePush {
    public float key;
    public float minEv;
    public float maxEv;
    // Adaptation time constants, in seconds, applied in EV space (see exposure_resolve's smoothing
    // step). Named for what the SCENE did, not for which way the exposure multiplier moved: a darkening
    // scene needs MORE exposure, so adaptDarken is the branch where target > prev.
    public float adaptDarken;
    public float adaptBrighten;
    public float frameTimeSeconds;
    public float evBias;
    public float lowPercentile;
    public float highPercentile;
    public float skyWeightCap;
    public float curveScene0;
    public float curveCompensation0;
    public float curveScene1;
    public float curveCompensation1;
    public float curveScene2;
    public float curveCompensation2;
    public float curveScene3;
    public float curveCompensation3;
    public float emissiveWeightCap;
    // evOffset takes log2(metered stored luminance) to EV100:
    // RtSceneUnits.EV100_OFFSET - log2(preExposure). preExposure is the scalar raygen already
    // multiplied into the stored radiance (1.0 when pre-exposure is disabled), so the metered
    // buffer holds L*preExposure and the resolve divides it back out to recover absolute scene EV --
    // and to keep minEv/maxEv bounding ABSOLUTE exposure rather than the residual.
    public float evOffset;
    public float preExposure;
    // resetSeq invalidates GPU exposure history without a host write or fence.
    public uint resetSeq;
};

// Persistent auto-exposure controller state. This shared definition is also reflected into
// ExposureStateData.java so host reads and initialization follow Slang's Std430 layout.
public struct ExposureState {
    public float previous;
    public uint initialized;
    public float evScene;
    public float evTarget;
    public float evApplied;
    public float clipLowFrac;
    public float clipHighFrac;
    public uint resetSeq;
    public float meteringSkyScale;
    public float meteringSkyFrac;
    public float curveCompensation;
    public float effectiveSlope;
    public float meteringEmissiveScale;
    public float meteringEmissiveFrac;
};

public struct DisplayPush {
    public int hdrEnabled;      // 0 = SDR only, 1 = also write the PQ HDR image
    public float lutSize;       // toneLut/hdrToneLut texels per axis; both LUTs share one size (asserted at load)
    public float gamma;         // post-display artistic gamma; below 1 brightens midtones
    public float hdrPeakNits;   // mastering peak baked into the currently bound HDR ACES LUT
    public int lookEnabled;     // 0 = identity, 1 = apply the scene-referred ACES look LUT
    public float lookLutSize;   // lookLut texels per axis (independent of the output-transform LUT size)
    // Scene-referred blurred highlight signal added before the LMT. The bloom pyramid's level 0 holds the
    // SUM of every band, so RtComposite folds the 1/levelCount normalisation into this value: the authored
    // look-package strength then means the same thing whichever level count the resolution supports.
    public float bloomStrength;
};

public struct BloomPush {
    public int mode;            // 0 prefilter+downsample, 1 downsample, 2 tent upsample (accumulate)
    public float threshold;     // exposed scene-linear ACEScg luminance
    public float softKnee;      // absolute exposed scene-linear knee width
    public float radius;        // upsample tent radius in SOURCE texels; resolution-independent
};

public struct DebugPresentPush {
    public uint debugView;     // 1..11; this pass is only dispatched for debugView != 0
    public float centerWeightSigma;
    public float centerWeightFloor;
};

// Froxel temporal/filter pass. parity selects which half of the resolved cache's ping-pong pair is read
// as history and which is written, so a frame never samples the cache it is overwriting.
public struct FogFilterPush {
    public uint64_t worldPushAddr;  // the frame's WorldPush slot, same BDA pattern as the sky LUT passes
    public uint parity;         // 0 reads cacheA / writes cacheB, 1 the reverse
    public uint2 grid;          // froxel width, height in texels (clamps the spatial filter's taps)
    public float maxHistory;    // temporal sample-count clamp; bounds motion trail length
    public float fireflyClamp;  // per-froxel luminance ceiling applied after the spatial filter
};

// Display-res fog march. depthSize is the render resolution of the reversed-Z depth guide, which the
// march nearest-samples at display resolution to find each ray's scene distance.
public struct FogMarchPush {
    public uint64_t worldPushAddr;
    public uint enabled;        // 0 = copy the scene through untouched
    public uint parity;         // selects the resolved-cache half the filter pass just wrote
    public uint frameIndex;     // salts the per-slice jitter so banding integrates to noise
    public uint2 size;          // march (display) resolution
    public uint2 depthSize;     // depth guide (render) resolution
};
