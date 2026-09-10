package dev.comfyfluffy.caustica;

import com.electronwill.nightconfig.core.CommentedConfig;
import com.electronwill.nightconfig.core.file.CommentedFileConfig;
import com.electronwill.nightconfig.core.file.FileNotFoundAction;
import com.electronwill.nightconfig.toml.TomlFormat;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntUnaryOperator;
import java.util.function.UnaryOperator;
import net.fabricmc.loader.api.FabricLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Central mutable runtime configuration. Each setting resolves its value, in order of precedence, from a
 * {@code -Dcaustica.*} system property, then the {@code config/caustica.toml} file, then a hardcoded
 * default. The settings UI and any other code call the same {@code set(...)} methods, and {@link #save()}
 * writes the current values back to the TOML file.
 *
 * <p>The system property namespace ({@code caustica.rt.foo}) and the TOML layout are independent: the file
 * uses real nested tables (e.g. {@code [omm]} with a {@code subdivision} key) grouped for readability, while
 * the property namespace stays flat and dotted for convenient one-off {@code -D} overrides.
 */
public final class CausticaConfig {
    private static final Logger LOGGER = LoggerFactory.getLogger("Caustica");
    private static final List<RuntimeSetting<?>> SETTINGS = new CopyOnWriteArrayList<>();

    private static final Path CONFIG_PATH = resolveConfigPath();
    private static final CommentedFileConfig FILE = loadFile(CONFIG_PATH);

    private CausticaConfig() {
    }

    public static List<RuntimeSetting<?>> settings() {
        return List.copyOf(SETTINGS);
    }

    public static Path configPath() {
        return CONFIG_PATH;
    }

    public static void reloadFromSystemProperties() {
        for (RuntimeSetting<?> setting : SETTINGS) {
            setting.reloadFromSystemProperties();
        }
    }

    /**
     * Forces every settings holder to class-initialize so all settings are registered (and have applied
     * their file values). Call before {@link #save()} to write a complete file, and once at startup so the
     * file round-trips the full surface even for settings the renderer has not touched yet.
     */
    public static void ensureRegistered() {
        @SuppressWarnings("unused")
        Object[] touch = {
            Rt.ENABLED, Rt.Composite.SPP, Rt.Composite.MAX_BOUNCES, Rt.Terrain.ASYNC_DISPATCH_PER_PASS, Rt.Omm.ENABLED,
            Rt.Entities.ENABLED, Rt.Entities.GLOW_ENABLED, Rt.EntityTextures.MAX_TEXTURES, Rt.DlssRr.ENABLED, Rt.Fg.ENABLED,
            Rt.Reflex.ENABLED, Rt.Exposure.MODE, Rt.Tonemap.GAMMA, Rt.FrameStats.ENABLED,
            Rt.Screenshots.EXR_ENABLED, Rt.Hdr.ENABLED, Ngx.PATH,
        };
    }

    /** Writes the default config file if it does not exist yet. */
    public static void saveIfMissing() {
        ensureRegistered();
        if (FILE.valueMap().isEmpty()) {
            save();
        }
    }

    /** Serializes all registered settings to the TOML config file. */
    public static synchronized void save() {
        ensureRegistered();
        writeComments();
        for (RuntimeSetting<?> setting : SETTINGS) {
            setting.writeToFile(FILE);
        }
        FILE.save();
    }

    private static void writeComments() {
        FILE.setComment("enabled",
                " Caustica ray-tracing settings. A matching -Dcaustica.* system property overrides a value here.");
        FILE.setComment("terrain",
                " Controls terrain loading. Higher limits can load terrain faster but use more CPU and GPU time.");
        FILE.setComment("frame-generation",
                " DLSS Frame Generation. Requires supported NVIDIA hardware and drivers.\n"
                        + " multi-frame-count sets generated frames per rendered frame (1 = 2x, 2 = 3x, ...).");
        FILE.setComment("reflex",
                " NVIDIA Reflex. Requires supported NVIDIA hardware and drivers.\n"
                        + " minimum-interval-us controls frame limiting; 0 disables the limit.");
        FILE.setComment("lights",
                " Controls direct lighting from glowing blocks such as torches, glowstone, and lava.\n"
                        + " Set ris-candidates to 0 to disable it. stats, dump, and dump-radius are debugging options.");
        FILE.setComment("tonemap",
                " Controls the final image. gamma: 1 is neutral; lower values brighten midtones.");
        FILE.setComment("exposure",
                " Controls automatic exposure. manual-ev sets exposure in manual mode and adjusts it in auto mode.\n"
                        + " adapt-darken and adapt-brighten control adjustment speed in seconds.\n"
                        + " sky-weight-cap and emissive-weight-cap limit how much bright areas affect exposure.");
        FILE.setComment("hdr",
                " HDR display output. Requires operating system and display support.\n"
                        + " ui-nits controls UI brightness; peak-nits must be 500, 1000, 2000, or 4000.");
        FILE.setComment("screenshots",
                " exr-enabled saves an ACEScg EXR beside the normal F2 PNG while ray tracing is active.");
    }

    private static Path resolveConfigPath() {
        try {
            return FabricLoader.getInstance().getConfigDir().resolve("caustica.toml");
        } catch (Throwable t) {
            return Path.of("config", "caustica.toml");
        }
    }

    private static CommentedFileConfig loadFile(Path path) {
        CommentedFileConfig config = CommentedFileConfig.builder(path, TomlFormat.instance())
                .onFileNotFound(FileNotFoundAction.CREATE_EMPTY)
                .preserveInsertionOrder()
                .sync()
                .build();
        try {
            config.load();
        } catch (Exception e) {
            LOGGER.warn("Failed to read Caustica config {}: {}", path, e.toString());
        }
        return config;
    }

    private static Boolean fileBoolean(String tomlPath) {
        return FILE.contains(tomlPath) ? FILE.<Boolean>get(tomlPath) : null;
    }

    private static Number fileNumber(String tomlPath) {
        return FILE.contains(tomlPath) ? FILE.<Number>get(tomlPath) : null;
    }

    private static String fileString(String tomlPath) {
        return FILE.contains(tomlPath) ? FILE.<String>get(tomlPath) : null;
    }

    public interface RuntimeSetting<T> {
        /** The {@code -Dcaustica.*} system property name that overrides this setting. */
        String key();

        /** The dotted path of this setting inside the nested {@code config/caustica.toml} tables. */
        String tomlPath();

        T defaultValue();

        T get();

        void set(T value);

        void reloadFromSystemProperties();

        /** Writes this setting's current value into the given config at {@link #tomlPath()}. */
        void writeToFile(CommentedConfig config);
    }

    public static final class BooleanSetting implements RuntimeSetting<Boolean> {
        private final String key;
        private final String tomlPath;
        private final boolean defaultValue;
        private volatile boolean value;

        private BooleanSetting(String key, String tomlPath, boolean defaultValue) {
            this.key = key;
            this.tomlPath = tomlPath;
            this.defaultValue = defaultValue;
            this.value = resolveInitial();
            SETTINGS.add(this);
        }

        @Override
        public String key() {
            return key;
        }

        @Override
        public String tomlPath() {
            return tomlPath;
        }

        @Override
        public Boolean defaultValue() {
            return defaultValue;
        }

        @Override
        public Boolean get() {
            return value;
        }

        public boolean value() {
            return value;
        }

        @Override
        public void set(Boolean value) {
            this.value = value != null ? value : defaultValue;
        }

        @Override
        public void reloadFromSystemProperties() {
            set(Boolean.parseBoolean(System.getProperty(key, Boolean.toString(defaultValue))));
        }

        @Override
        public void writeToFile(CommentedConfig config) {
            config.set(tomlPath, value);
        }

        private boolean resolveInitial() {
            String prop = System.getProperty(key);
            if (prop != null) {
                return Boolean.parseBoolean(prop.trim());
            }
            Boolean fromFile = fileBoolean(tomlPath);
            return fromFile != null ? fromFile : defaultValue;
        }
    }

    public static final class IntSetting implements RuntimeSetting<Integer> {
        private final String key;
        private final String tomlPath;
        private final int defaultValue;
        private final IntUnaryOperator sanitize;
        private volatile int value;

        private IntSetting(String key, String tomlPath, int defaultValue, IntUnaryOperator sanitize) {
            this.key = key;
            this.tomlPath = tomlPath;
            this.defaultValue = sanitize.applyAsInt(defaultValue);
            this.sanitize = sanitize;
            this.value = resolveInitial();
            SETTINGS.add(this);
        }

        @Override
        public String key() {
            return key;
        }

        @Override
        public String tomlPath() {
            return tomlPath;
        }

        @Override
        public Integer defaultValue() {
            return defaultValue;
        }

        @Override
        public Integer get() {
            return value;
        }

        public int value() {
            return value;
        }

        @Override
        public void set(Integer value) {
            this.value = sanitize.applyAsInt(value != null ? value : defaultValue);
        }

        @Override
        public void reloadFromSystemProperties() {
            String prop = System.getProperty(key);
            if (prop == null) {
                this.value = defaultValue;
                return;
            }
            try {
                this.value = sanitize.applyAsInt(Integer.parseInt(prop.trim()));
            } catch (NumberFormatException e) {
                this.value = defaultValue;
            }
        }

        @Override
        public void writeToFile(CommentedConfig config) {
            config.set(tomlPath, value);
        }

        private int resolveInitial() {
            String prop = System.getProperty(key);
            if (prop != null) {
                try {
                    return sanitize.applyAsInt(Integer.parseInt(prop.trim()));
                } catch (NumberFormatException e) {
                    return defaultValue;
                }
            }
            Number fromFile = fileNumber(tomlPath);
            return fromFile != null ? sanitize.applyAsInt(fromFile.intValue()) : defaultValue;
        }
    }

    public static final class FloatSetting implements RuntimeSetting<Float> {
        private final String key;
        private final String tomlPath;
        private final float defaultValue;
        // Maps a raw external number (system property, file, or the constructor's raw default) into the
        // stored value domain, e.g. degrees -> radians.
        private final DoubleUnaryOperator inputTransform;
        // Inverse of inputTransform: maps the stored value domain back to the raw external domain (e.g.
        // radians -> degrees) for writeToFile, so a value round-trips through the file unchanged instead
        // of having inputTransform re-applied to an already-transformed number on the next load.
        private final DoubleUnaryOperator outputTransform;
        // Idempotent guard on a value-domain number (clamp / finite check); safe to apply to any source.
        private final DoubleUnaryOperator valueClamp;
        private volatile float value;

        private FloatSetting(String key, String tomlPath, float rawDefault, DoubleUnaryOperator inputTransform,
                             DoubleUnaryOperator outputTransform, DoubleUnaryOperator valueClamp) {
            this.key = key;
            this.tomlPath = tomlPath;
            this.inputTransform = inputTransform;
            this.outputTransform = outputTransform;
            this.valueClamp = valueClamp;
            this.defaultValue = (float) valueClamp.applyAsDouble(inputTransform.applyAsDouble(rawDefault));
            this.value = resolveInitial();
            SETTINGS.add(this);
        }

        @Override
        public String key() {
            return key;
        }

        @Override
        public String tomlPath() {
            return tomlPath;
        }

        @Override
        public Float defaultValue() {
            return defaultValue;
        }

        @Override
        public Float get() {
            return value;
        }

        public float value() {
            return value;
        }

        @Override
        public void set(Float value) {
            if (value == null) {
                this.value = defaultValue;
            } else {
                this.value = (float) valueClamp.applyAsDouble(inputTransform.applyAsDouble(value));
            }
        }

        @Override
        public void reloadFromSystemProperties() {
            String prop = System.getProperty(key);
            if (prop == null) {
                this.value = defaultValue;
                return;
            }
            try {
                this.value = (float) valueClamp.applyAsDouble(inputTransform.applyAsDouble(Double.parseDouble(prop.trim())));
            } catch (NumberFormatException e) {
                this.value = defaultValue;
            }
        }

        @Override
        public void writeToFile(CommentedConfig config) {
            // Round-trip through Float.toString() so the file gets the shortest decimal that reproduces
            // this float (e.g. "0.6"), not outputTransform's raw double with float's binary noise spelled
            // out to 17 digits (e.g. 0.6000000487130328).
            float raw = (float) outputTransform.applyAsDouble(value);
            config.set(tomlPath, Double.parseDouble(Float.toString(raw)));
        }

        private float resolveInitial() {
            String prop = System.getProperty(key);
            if (prop != null) {
                try {
                    return (float) valueClamp.applyAsDouble(inputTransform.applyAsDouble(Double.parseDouble(prop.trim())));
                } catch (NumberFormatException e) {
                    return defaultValue;
                }
            }
            Number fromFile = fileNumber(tomlPath);
            if (fromFile == null) {
                return defaultValue;
            }
            return (float) valueClamp.applyAsDouble(inputTransform.applyAsDouble(fromFile.doubleValue()));
        }
    }

    public static final class StringSetting implements RuntimeSetting<String> {
        private final String key;
        private final String tomlPath;
        private final String defaultValue;
        private final UnaryOperator<String> sanitize;
        private volatile String value;

        private StringSetting(String key, String tomlPath, String defaultValue, UnaryOperator<String> sanitize) {
            this.key = key;
            this.tomlPath = tomlPath;
            this.defaultValue = sanitize.apply(defaultValue);
            this.sanitize = sanitize;
            this.value = resolveInitial();
            SETTINGS.add(this);
        }

        @Override
        public String key() {
            return key;
        }

        @Override
        public String tomlPath() {
            return tomlPath;
        }

        @Override
        public String defaultValue() {
            return defaultValue;
        }

        @Override
        public String get() {
            return value;
        }

        @Override
        public void set(String value) {
            this.value = sanitize.apply(value != null ? value : defaultValue);
        }

        @Override
        public void reloadFromSystemProperties() {
            set(System.getProperty(key, defaultValue));
        }

        @Override
        public void writeToFile(CommentedConfig config) {
            config.set(tomlPath, value);
        }

        private String resolveInitial() {
            String prop = System.getProperty(key);
            if (prop != null) {
                return sanitize.apply(prop);
            }
            String fromFile = fileString(tomlPath);
            return sanitize.apply(fromFile != null ? fromFile : defaultValue);
        }
    }

    public static final class OptionalStringSetting implements RuntimeSetting<String> {
        private final String key;
        private final String tomlPath;
        private volatile String value;

        private OptionalStringSetting(String key, String tomlPath) {
            this.key = key;
            this.tomlPath = tomlPath;
            this.value = resolveInitial();
            SETTINGS.add(this);
        }

        @Override
        public String key() {
            return key;
        }

        @Override
        public String tomlPath() {
            return tomlPath;
        }

        @Override
        public String defaultValue() {
            return null;
        }

        @Override
        public String get() {
            return value;
        }

        @Override
        public void set(String value) {
            this.value = value;
        }

        @Override
        public void reloadFromSystemProperties() {
            this.value = System.getProperty(key);
        }

        @Override
        public void writeToFile(CommentedConfig config) {
            if (value != null) {
                config.set(tomlPath, value);
            } else {
                config.remove(tomlPath);
            }
        }

        private String resolveInitial() {
            String prop = System.getProperty(key);
            return prop != null ? prop : fileString(tomlPath);
        }
    }

    public static final class Rt {
        public static final BooleanSetting ENABLED = bool("caustica.rt", "enabled", true);
        public static final IntSetting WORKER_THREADS =
                intAtLeast("caustica.rt.workerThreads", "worker-threads", defaultWorkerThreads(), 1);

        private Rt() {
        }

        public static final class Composite {
            public static final IntSetting DEBUG_VIEW = intValue("caustica.rt.debugView", "composite.debug-view", 0);
            public static final IntSetting SPP = intAtLeast("caustica.rt.spp", "composite.spp", 1, 1);
            public static final IntSetting MAX_BOUNCES =
                    clampedInt("caustica.rt.maxBounces", "composite.max-bounces", 4, 2, 8);
            public static final BooleanSetting WATER_WAVES =
                    bool("caustica.rt.waterWaves", "composite.water-waves", true);
            // Sun/moon angular radii and the noon south tilt moved into the versioned look package
            // (look.json "sky"): they shape the sky alongside the exposure curve, the LMT and the
            // photometric anchors that were already authored there, and splitting them across two
            // sources meant a package could not fully describe its own look.
            public static final FloatSetting JITTER_SIGN_X =
                    finiteFloat("caustica.rt.jitterSignX", "composite.jitter-sign-x", 1.0f);
            public static final FloatSetting JITTER_SIGN_Y =
                    finiteFloat("caustica.rt.jitterSignY", "composite.jitter-sign-y", -1.0f);

            private Composite() {
            }
        }

        public static final class Terrain {
            // External keys retain their historical "per-tick" names for config compatibility; terrain
            // streaming is render-pass driven and these Java names reflect the actual scheduling unit.
            public static final IntSetting ASYNC_DISPATCH_PER_PASS =
                    intAtLeast("caustica.rt.asyncDispatchPerTick", "terrain.async-dispatch-per-tick", 32, 0);
            public static final IntSetting COMPLETION_RESULTS_PER_PASS =
                    intAtLeast("caustica.rt.sectionResultsPerTick", "terrain.section-results-per-tick", 32, 0);
            public static final IntSetting MAX_INFLIGHT_SECTIONS =
                    intAtLeast("caustica.rt.maxInflightSections", "terrain.max-inflight-sections", 32, 0);
            public static final IntSetting SECTION_TABLE_INITIAL_CAPACITY =
                    intAtLeast("caustica.rt.sectionTableInitialCapacity", "terrain.section-table-initial-capacity", 512, 1);
            public static final IntSetting REBASE_DISTANCE_BLOCKS =
                    intAtLeast("caustica.rt.rebaseDistanceBlocks", "terrain.rebase-distance-blocks", 128, 0);
            public static final BooleanSetting BLAS_COMPACTION =
                    bool("caustica.rt.blasCompaction", "terrain.blas-compaction", true);

            private Terrain() {
            }
        }

        /** RIS block-emitter lights. {@code ris-candidates = 0} disables everything. */
        public static final class Lights {
            public static final IntSetting RIS_CANDIDATES =
                    intAtLeast("caustica.rt.risCandidates", "lights.ris-candidates", 8, 0);
            public static final FloatSetting MIN_FILL_RATIO =
                    finiteFloat("caustica.rt.lightMinFillRatio", "lights.min-fill-ratio", 0.25f);
            public static final BooleanSetting STATS = bool("caustica.rt.lightStats", "lights.stats", false);
            public static final BooleanSetting DUMP = bool("caustica.rt.lightDump", "lights.dump", false);
            public static final IntSetting DUMP_RADIUS =
                    intAtLeast("caustica.rt.lightDumpRadius", "lights.dump-radius", 12, 1);

            private Lights() {
            }
        }

        public static final class Omm {
            public static final BooleanSetting ENABLED = bool("caustica.rt.omm", "omm.enabled", true);
            public static final IntSetting SUBDIVISION =
                    clampedInt("caustica.rt.ommSubdivision", "omm.subdivision", 4, 0, 6);
            public static final BooleanSetting STATS = bool("caustica.rt.ommStats", "omm.stats", false);

            private Omm() {
            }
        }

        /**
         * Aerial-perspective medium (fog), integrated along each traced path segment inside the transport
         * rather than composited in screen space. Its density falls exponentially with ABSOLUTE altitude
         * above the level's sea level, so a distant horizon hazes over while caves and ocean trenches sit
         * at the flat profile floor instead of thickening without bound. What keeps a sealed room from
         * glowing is the light ray each fog region casts, not the profile: a roof occludes the sun, and
         * the same ray is what carves visible shafts out of the haze.
         *
         * <p>DENSITY is per-block extinction, so optical depth grows with path length: a value tuned so
         * that one e-fold sits past a typical render distance keeps short views crisp and long views hazy
         * instead of needing a re-tune per display setting.
         */
        public static final class Fog {
            public static final BooleanSetting ENABLED = bool("caustica.rt.fog.enabled", "fog.enabled", false);
            // Per-block extinction at the reference altitude. Vanilla's temperate biomes author this as
            // max_density 0.05 times a scattering coefficient of 0.04, i.e. 0.002, which is what a
            // "reasonable haze over a valley" turns out to mean in Bedrock's own numbers; jungles use 0.06.
            public static final FloatSetting DENSITY =
                    clampedFloat("caustica.rt.fog.density", "fog.density", 0.002f, 0.0f, 0.05f);
            public static final FloatSetting SCALE_HEIGHT =
                    clampedFloat("caustica.rt.fog.scaleHeight", "fog.scale-height", 96.0f, 4.0f, 4096.0f);
            // Fraction of extinction that scatters instead of being absorbed: at 1.0 the medium is pure
            // haze and stays bright at the horizon, at 0.5 it turns smoky and darkens what it covers.
            // Vanilla's air entries set absorption to zero, i.e. an albedo of exactly 1.0 -- a pure
            // redistributor that never loses energy. It stops short here because our skylight is a per-voxel
            // dome integral rather than their one global light meter, and a lossless medium makes that term
            // the only thing left in a sealed room.
            public static final FloatSetting SCATTER_ALBEDO =
                    clampedFloat("caustica.rt.fog.scatterAlbedo", "fog.scatter-albedo", 0.95f, 0.0f, 1.0f);
            // Forward lobe of the phase function, at the value every temperate vanilla biome uses. Kept off
            // 1.0 by the clamp because that lobe's peak grows as 1/(1-g)^2: past ~0.9 the pixels around the
            // sun stop being bright haze and become a saturated blob, which no exposure or denoiser recovers.
            public static final FloatSetting ANISOTROPY =
                    clampedFloat("caustica.rt.fog.anisotropy", "fog.anisotropy", 0.6f, -0.9f, 0.9f);
            // Distance a segment is resolved into STEPS taps, the vertical extent of the sun column, and
            // how far the light volume's rays look for occluders. Not a "how far fog reaches": a path's whole
            // length is always integrated, only its near field is finely sampled, so raising REACH buys
            // detail rather than adding a visible wall. It also stretches the volume's slices and the rays
            // that fill them, which are a fixed count per frame rather than one per fog tap -- so this is
            // cheap to raise, and it is the knob that decides how far away a shaft can still be lit.
            public static final FloatSetting REACH =
                    clampedFloat("caustica.rt.fog.reach", "fog.reach", 512.0f, 16.0f, 4096.0f);
            // 16, which is what a march of this shape wants: the taps are placed on a square law, so the
            // extra budget goes into the first stretch of the segment rather than spreading the error out.
            public static final IntSetting STEPS =
                    clampedInt("caustica.rt.fog.steps", "fog.steps", 16, 1, 32);
            // Screen pixels per edge of a light-volume voxel. The volume is where the medium learns what light
            // arrives at a particular bit of air, so this is the resolution of every shaft edge: a beam that is
            // narrower than a voxel is smoothed into a ramp rather than resolved. Lower costs rays -- the gather
            // is a fixed set per voxel, once per frame -- and 8 already quadruples the voxel count over 16. The
            // "frame.traceFroxels" line in the F3 stats is that cost, measured.
            public static final IntSetting GRID_DIVISOR =
                    clampedInt("caustica.rt.fog.gridDivisor", "fog.grid-divisor", 16, 4, 32);
            // Depth slices of the same volume, log-spaced between a block and `reach`. This is the resolution
            // of a shaft ALONG the view ray: two taps that land in the same slice share a bilinear layer, so
            // a beam that ends or starts between slices is positioned by the interpolation between them. 64
            // doubles the gather's voxel count exactly like halving the divisor does.
            public static final IntSetting GRID_SLICES =
                    clampedInt("caustica.rt.fog.gridSlices", "fog.grid-slices", 32, 8, 64);
            // Rays per voxel toward the sun. MCRTX exposes one `rayCountMultiplier` for the whole march; this
            // is split in two because the sun's rays are the ones whose count shows up as shaft detail, while
            // the sky's are a veil that averages down cheaply. Zero sun rays means "no direct term at all",
            // which is the fastest way to see what the volume is contributing.
            public static final IntSetting SUN_RAYS = clampedInt("caustica.rt.fog.sunRays", "fog.sun-rays", 4, 0, 16);
            // Sky directions over the whole sphere, each with a ray of its own: this is the ambient term, and
            // halving it is the cheapest thing to try when the gather's cost line is the complaint.
            public static final IntSetting SKY_RAYS = clampedInt("caustica.rt.fog.skyRays", "fog.sky-rays", 8, 0, 32);
            // How many frames the light volume averages over, named after MCRTX's `maxHistoryLength` because
            // it means the same thing. 1 is off -- every voxel reports only this frame's rays, which is where
            // the puffs between leaves come from; the blend weight it implies is (N-1)/(N+1), the standard
            // moving average over N frames, so 8 is the 78% history the volume shipped with.
            public static final IntSetting HISTORY_FRAMES =
                    clampedInt("caustica.rt.fog.history", "fog.history-frames", 8, 1, 32);
            // Spatial filter over the volume: 0 off, 1 a 3x3x3 box, above 1 a wider 5x5x5 rather than a
            // stronger blend, which is MCRTX's rule for the GI blur ("kernel size depending on history
            // length"): volume that has already been averaged in time can afford to be averaged in space.
            public static final FloatSetting FILTER =
                    clampedFloat("caustica.rt.fog.filter", "fog.filter", 1.0f, 0.0f, 2.0f);
            // Rain deepens the medium, the participating-medium version of what vanilla does by swapping to
            // its `weather` fog entry: this is the multiplier at a full downpour, ramped by the rain level, so
            // 2.0 means a dry day is unchanged and a storm doubles the extinction.
            public static final FloatSetting RAIN_DENSITY_FACTOR =
                    clampedFloat("caustica.rt.fog.rainDensity", "fog.rain-density-factor", 2.0f, 1.0f, 8.0f);

            private Fog() {
            }
        }

        public static final class Entities {
            public static final BooleanSetting ENABLED = bool("caustica.rt.entities", "entities.enabled", true);
            public static final BooleanSetting PARTICLES_ENABLED =
                    bool("caustica.rt.particles", "particles.enabled", true);
            public static final BooleanSetting GLOW_ENABLED =
                    bool("caustica.rt.glow", "entities.glow.enabled", true);
            public static final BooleanSetting NAME_TAGS_ENABLED =
                    bool("caustica.rt.nameTags", "entities.name-tags.enabled", true);
            /** Debug-only: render each model submission twice and require bitwise-identical CPU captures. */
            public static final BooleanSetting CAPTURE_PARITY =
                    bool("caustica.rt.entityCaptureParity", "entities.debug.capture-parity", false);
            public static final IntSetting MAX_ORDINARY_ENTITIES =
                    intAtLeast("caustica.rt.maxOrdinaryEntities", "entities.max-ordinary-entities", 1024, 0);
            public static final IntSetting MAX_BLOCK_ENTITIES =
                    intAtLeast("caustica.rt.maxBlockEntities", "entities.block-entities.max-entities", 1024, 0);
            public static final IntSetting MAX_PARTICLES =
                    intAtLeast("caustica.rt.maxParticles", "particles.max-particles", 1024, 0);
            public static final IntSetting BE_VIEW_CHUNKS =
                    intAtLeast("caustica.rt.beViewChunks", "entities.block-entities.view-chunks", 8, 0);
            public static final IntSetting BE_BUILDS_PER_FRAME =
                    intAtLeast("caustica.rt.beBuildsPerFrame", "entities.block-entities.builds-per-frame", 64, 0);
            public static final BooleanSetting REFIT_ENABLED =
                    bool("caustica.rt.entityRefit", "entities.refit.enabled", true);

            private Entities() {
            }

            public static int maxEntities() {
                return Math.addExact(Math.addExact(
                        MAX_ORDINARY_ENTITIES.value(), MAX_BLOCK_ENTITIES.value()), MAX_PARTICLES.value());
            }

            public static int entityListCapacity() {
                return Math.max(16, maxEntities());
            }

            public static int entityMapCapacity() {
                // Fastutil expected-size constructors apply their own load-factor headroom.
                return Math.max(16, MAX_ORDINARY_ENTITIES.value());
            }
        }

        public static final class EntityTextures {
            public static final IntSetting MAX_TEXTURES =
                    intAtLeast("caustica.rt.maxEntityTextures", "entities.textures.max-textures", 256, 1);
            public static final BooleanSetting PBR = bool("caustica.rt.entityPbr", "entities.textures.pbr", true);

            private EntityTextures() {
            }
        }

        public static final class Overlay {
            public static final BooleanSetting BLOCK_OUTLINE_ENABLED =
                    bool("caustica.rt.blockOutline", "overlay.block-outline.enabled", true);

            private Overlay() {
            }
        }

        public static final class DlssRr {
            public static final BooleanSetting ENABLED = bool("caustica.rt.dlssRr", "dlss-rr.enabled", true);
            public static final IntSetting PRESET = intValue("caustica.rt.dlssRr.preset", "dlss-rr.preset", 0);

            // NVSDK_NGX_PerfQuality_Value. Per NVIDIA's DLSS-RR programming guide, Ray Reconstruction only
            // supports Performance(0), Balanced(1), Quality(2), Ultra-Performance(3), and DLAA(5) —
            // Ultra Quality(4) is not a valid PerfQualityValue for RR (its optimal-settings query returns a
            // zeroed render size for it) and is deliberately excluded here.
            public static final List<Integer> QUALITY_STEPS = List.of(3, 0, 1, 2, 5);
            public static final IntSetting QUALITY =
                    intChoice("caustica.rt.dlssRr.quality", "dlss-rr.quality", 0, QUALITY_STEPS);

            private DlssRr() {
            }
        }

        /** DLSS Frame Generation. Default off; gated additionally by hardware/driver availability. */
        public static final class Fg {
            public static final BooleanSetting ENABLED = bool("caustica.rt.fg", "frame-generation.enabled", false);
            public static final IntSetting MULTI_FRAME_COUNT =
                    intAtLeast("caustica.rt.fg.multiFrameCount", "frame-generation.multi-frame-count", 1, 1);

            private Fg() {
            }
        }

        /**
         * NVIDIA Reflex ({@code VK_NV_low_latency2}). Default off; gated additionally by device support.
         * The renderer configures the swapchain latency mode, paces frames with {@code vkLatencySleepNV},
         * and emits simulation, render-submit, and present latency markers.
         */
        public static final class Reflex {
            public static final BooleanSetting ENABLED = bool("caustica.rt.reflex", "reflex.enabled", false);
            public static final BooleanSetting LOW_LATENCY_BOOST =
                    bool("caustica.rt.reflex.boost", "reflex.low-latency-boost", false);
            public static final IntSetting MINIMUM_INTERVAL_US =
                    intAtLeast("caustica.rt.reflex.minIntervalUs", "reflex.minimum-interval-us", 0, 0);

            private Reflex() {
            }
        }

        public static final class Exposure {
            // Control points are measured-EV100 : compensation-EV.
            // Rendered median (log) = log2(key) + comp(evScene), so comp IS the rendered offset in EV
            // from the noon reference.
            //
            // Fitted to measured in-game EV100 and the current emissive baseline:
            //   noon sand       +17.45 -> -0.01   renders at key, the reference
            //   noon blue sky   +16.50 -> -0.17
            //   daylight shade   +7.00 -> -1.82
            //   lit night room   +7.00 -> -1.82   (same measured luminance as daylight shade)
            //   night street     +1.50 -> -3.01
            //   starlit sky      -8.00 -> -5.00   (floor)
            // Effective slope is 0.79 / 0.78 / 0.83 across the three segments, compressing 25 EV of
            // scene range to 5.0 EV of rendered difference.
            //
            // Daylight shade and a lit interior at night measure the SAME (~EV 7), so no luminance-only
            // curve can separate them -- what does is the asymmetric temporal adaptation above, which
            // holds a low exposure when you step from noon sun into shade. That is a real limit of this
            // controller, not a tuning miss.
            public static final StringSetting MODE =
                    string("caustica.rt.exposure.mode", "exposure.mode", "auto", Exposure::sanitizeMode);
            public static final FloatSetting MANUAL_EV =
                    clampedFloat("caustica.rt.exposure.manualEv", "exposure.manual-ev",
                            0.0f, -15.0f, 15.0f);
            public static final FloatSetting KEY = exposureScale("caustica.rt.exposure.key", "exposure.key", 0.18f);
            // Bounds on the ABSOLUTE exposure multiplier. Sized from what the curve above actually asks
            // for at the measured scene extremes: -16.9 EV at noon sand, +3.5 EV at the starlit-sky
            // floor. A clamp should be a guard rail, not the controller, so these sit just outside that.
            //
            // max-ev was +10 and blew out the frame: with 13 EV of headroom above what the curve wants,
            // exposure ran away whenever the camera held something very dark, and anything bright
            // entering the frame then arrived pre-blown. +5 keeps 1.5 EV over the curve's own demand.
            //
            // min-ev deliberately does NOT cover a zoomed-in sun (which asks for about -20.8): letting
            // the whole frame go black because the sun is in shot is worse than clamping it. The sky
            // metering cap already bounds the sun's share, so in practice this only engages on a
            // near-full-screen sun.
            /**
             * Adaptation time constants in seconds, applied in EV space by the resolve. Named for what
             * the SCENE did: walking into a dark cave is "darken" (exposure has to rise), stepping back
             * out is "brighten".
             *
             * <p>Asymmetric on purpose, and in the direction human vision actually works — light
             * adaptation takes seconds, dark adaptation takes minutes. Every shipping game compresses
             * that, but keeping the sign right is what makes a sunrise read as a sunrise instead of as a
             * lens. The names describe the scene change, not the inverse movement of the exposure multiplier.
             */
            public static final FloatSetting ADAPT_DARKEN =
                    exposureScale("caustica.rt.exposure.adaptDarken", "exposure.adapt-darken", 2.0f);
            public static final FloatSetting ADAPT_BRIGHTEN =
                    exposureScale("caustica.rt.exposure.adaptBrighten", "exposure.adapt-brighten", 0.4f);
            public static final FloatSetting LOW_PERCENTILE =
                    clampedFloat("caustica.rt.exposure.lowPercentile", "exposure.low-percentile", 0.50f, 0.0f, 1.0f);
            public static final FloatSetting HIGH_PERCENTILE =
                    clampedFloat("caustica.rt.exposure.highPercentile", "exposure.high-percentile", 0.95f, 0.0f, 1.0f);
            public static final IntSetting STRIDE =
                    clampedInt("caustica.rt.exposure.stride", "exposure.stride", 2, 1, 8);
            public static final FloatSetting CENTER_WEIGHT_SIGMA =
                    clampedFloat("caustica.rt.exposure.centerWeightSigma",
                            "exposure.center-weight-sigma", 0.35f, 0.01f, 2.0f);
            public static final FloatSetting CENTER_WEIGHT_FLOOR =
                    clampedFloat("caustica.rt.exposure.centerWeightFloor",
                            "exposure.center-weight-floor", 0.15f, 0.0f, 1.0f);
            public static final FloatSetting SKY_WEIGHT_CAP =
                    clampedFloat("caustica.rt.exposure.skyWeightCap",
                            "exposure.sky-weight-cap", 0.25f, 0.0f, 1.0f);
            public static final FloatSetting EMISSIVE_WEIGHT_CAP =
                    clampedFloat("caustica.rt.exposure.emissiveWeightCap",
                            "exposure.emissive-weight-cap", 0.10f, 0.0f, 1.0f);
            /**
             * Pre-exposure: raygen multiplies scene radiance by the previous frame's exposure before
             * the fp16 write, and the display pass divides it back out, so stored values sit near
             * {@code key} instead of spanning the ~26 EV physical photometric units require. The two
             * cancel algebraically, so <b>toggling this must not change the
             * image</b>; it exists as an A/B switch for exactly that check, and as an escape hatch
             * if DLSS-RR ever proves sensitive to its history being at the previous frame's scale.
             */
            public static final BooleanSetting PRE_EXPOSURE =
                    bool("caustica.rt.exposure.preExposure", "exposure.pre-exposure", true);

            private Exposure() {
            }

            public static float minEv() {
                return dev.comfyfluffy.caustica.rt.RtLookPackage.current().exposure().minEv();
            }

            public static float maxEv() {
                return dev.comfyfluffy.caustica.rt.RtLookPackage.current().exposure().maxEv();
            }

            public static String curve() {
                return dev.comfyfluffy.caustica.rt.RtLookPackage.current().exposure().curve();
            }

            /**
             * Sanity bound on an exposure multiplier, not an artistic one. It must remain below the
             * 3.8e-6 multiplier requested by {@code -18 EV}; min-ev/max-ev provides the artistic bound.
             */
            public static float clampScale(float value) {
                return Math.clamp(value, 1.0e-8f, 1.0e8f);
            }

            private static String sanitizeMode(String value) {
                if ("auto".equalsIgnoreCase(value)) {
                    return "auto";
                }
                if ("manual".equalsIgnoreCase(value)) {
                    return "manual";
                }
                return "auto";
            }

        }

        /** Scene-referred look transform and baked SDR/HDR ACES display transforms. */
        public static final class Tonemap {
            public static final FloatSetting GAMMA =
                    clampedFloat("caustica.rt.tonemap.gamma", "tonemap.gamma", 1.0f, 0.1f, 5.0f);

            private Tonemap() {
            }
        }

        /** Render-frame timing + hitch logging. See {@code RtFrameStats}. */
        public static final class FrameStats {
            public static final BooleanSetting ENABLED = bool("caustica.rt.frameStats", "frame-stats.enabled", false);

            private FrameStats() {
            }
        }

        /** Optional high-dynamic-range screenshot output paired with vanilla's F2 PNG. */
        public static final class Screenshots {
            public static final BooleanSetting EXR_ENABLED =
                    bool("caustica.rt.screenshots.exr", "screenshots.exr-enabled", false);

            private Screenshots() {
            }
        }

        /** Startup Vulkan inventory + {@code VK_EXT_device_fault} reporting on device loss. See {@code VulkanDiagnostics}. */
        public static final class Diagnostics {
            /** Heavy driver-side crash diagnostics: vendor diagnostics-config extensions (shader debug
             * info, resource tracking, automatic checkpoints, shader error reporting) and the
             * {@code deviceFaultVendorBinary} feature (vendor-format crash dump on device loss). Off by
             * default: measured ~10x BLAS build time / -20% fps when enabled. Plain {@code deviceFault}
             * reporting (fault addresses + vendor records) is always on and unaffected. Turn on only
             * while chasing a live device-loss crash. */
            public static final BooleanSetting HEAVY_CRASH_DIAGNOSTICS =
                    bool("caustica.rt.heavyCrashDiagnostics", "diagnostics.heavy-crash-diagnostics", false);

            private Diagnostics() {
            }
        }

        /**
         * HDR display output. When enabled the swapchain is created in PQ (ST.2084/HDR10 — the display-ready
         * encoding both HDR10 swapchains and DLSS Frame Generation require; whatever pixel format the surface
         * pairs with that color space, commonly a 10-bit UNORM), falling back to SDR if the surface doesn't
         * advertise it. The ACES LUT owns scene-to-display mapping; {@code uiNits} places SDR-authored UI
         * in that PQ output, while {@code peakNits} selects the LUT's mastering target.
         */
        public static final class Hdr {
            public static final BooleanSetting ENABLED = bool("caustica.rt.hdr", "hdr.enabled", false);
            public static final FloatSetting UI_NITS =
                    clampedFloat("caustica.rt.hdr.uiNits", "hdr.ui-nits", 200.0f, 80.0f, 500.0f);

            // ACES HDR LUTs are available only for these mastering targets.
            public static final List<Integer> PEAK_NITS_STEPS = List.of(500, 1000, 2000, 4000);
            public static final IntSetting PEAK_NITS =
                    intChoice("caustica.rt.hdr.peakNits", "hdr.peak-nits", 1000, PEAK_NITS_STEPS);

            // Surface capability and current swapchain state are separate: HDR controls remain available
            // while the swapchain is native SDR, so enabling HDR can recreate it in PQ.
            private static volatile boolean SWAPCHAIN_PQ_AVAILABLE = false;
            private static volatile boolean SWAPCHAIN_PQ_ACTIVE = false;

            private Hdr() {
            }

            public static void setSwapchainPqAvailable(boolean available) {
                SWAPCHAIN_PQ_AVAILABLE = available;
            }

            public static void setSwapchainPqActive(boolean active) {
                SWAPCHAIN_PQ_ACTIVE = active;
            }

            /**
             * Whether this session's surface can create a PQ swapchain, independent of which format the
             * current swapchain uses.
             */
            public static boolean swapchainPqAvailable() {
                return SWAPCHAIN_PQ_AVAILABLE;
            }

            /** Whether the currently configured swapchain is HDR10/PQ rather than native SDR. */
            public static boolean swapchainPqActive() {
                return SWAPCHAIN_PQ_ACTIVE;
            }

            /**
             * Whether the HDR display path (world HDR + PQ swapchain + UI overlay) should be active this
             * frame. The option invalidates the surface configuration after changing {@link #ENABLED};
             * the ordinary resize/configure path recreates the swapchain in SDR or PQ.
             */
            public static boolean enabled() {
                return SWAPCHAIN_PQ_ACTIVE && ENABLED.value();
            }

            /** Absolute brightness assigned to SDR-authored UI in the PQ output. */
            public static float uiNits() {
                return UI_NITS.value();
            }

        }
    }

    public static final class Ngx {
        public static final OptionalStringSetting PATH = optionalString("caustica.ngx.path", "ngx.path");

        private Ngx() {
        }
    }

    private static BooleanSetting bool(String key, String tomlPath, boolean fallback) {
        return new BooleanSetting(key, tomlPath, fallback);
    }

    private static StringSetting string(String key, String tomlPath, String fallback, UnaryOperator<String> sanitize) {
        return new StringSetting(key, tomlPath, fallback, sanitize);
    }

    private static OptionalStringSetting optionalString(String key, String tomlPath) {
        return new OptionalStringSetting(key, tomlPath);
    }

    private static IntSetting intValue(String key, String tomlPath, int fallback) {
        return new IntSetting(key, tomlPath, fallback, v -> v);
    }

    private static IntSetting intAtLeast(String key, String tomlPath, int fallback, int min) {
        return new IntSetting(key, tomlPath, fallback, v -> Math.max(min, v));
    }

    private static IntSetting intChoice(String key, String tomlPath, int fallback, List<Integer> choices) {
        return new IntSetting(key, tomlPath, fallback, v -> choices.contains(v) ? v : fallback);
    }

    private static IntSetting clampedInt(String key, String tomlPath, int fallback, int min, int max) {
        return new IntSetting(key, tomlPath, fallback, v -> Math.clamp(v, min, max));
    }

    private static FloatSetting finiteFloat(String key, String tomlPath, float fallback) {
        return new FloatSetting(key, tomlPath, fallback, v -> v, v -> v, v -> Double.isFinite(v) ? v : fallback);
    }

    private static FloatSetting exposureScale(String key, String tomlPath, float fallback) {
        return new FloatSetting(key, tomlPath, fallback, v -> v, v -> v, v -> Math.clamp(v, 1.0e-4, 1.0e4));
    }

    private static FloatSetting clampedFloat(String key, String tomlPath, float fallback, float min, float max) {
        return new FloatSetting(key, tomlPath, fallback, v -> v, v -> v, v -> Math.clamp(v, min, max));
    }

    private static FloatSetting radians(String key, String tomlPath, float fallbackDegrees) {
        return new FloatSetting(key, tomlPath, fallbackDegrees, Math::toRadians, Math::toDegrees, v -> Double.isFinite(v) ? v : 0.0);
    }

    private static int defaultWorkerThreads() {
        return Math.clamp(Runtime.getRuntime().availableProcessors() / 2, 1, 4);
    }
}
