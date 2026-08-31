// Manual push data for FogIntegrationPush — layout matches volumetric_fog_common.slang
package dev.comfyfluffy.caustica.rt.gen;

import java.nio.ByteBuffer;
import java.util.Objects;

public record FogIntegrationPushData(
        long worldPushAddr,
        long tableAddr,
        long entityTableAddr,
        long materialTableAddr,
        Int2 displaySize,
        Int2 froxelDim,
        int froxelDepth,
        float nearPlane,
        float farPlane,
        float density,
        float anisotropy,
        float scattering,
        float extinction,
        float heightFalloff,
        float sunIntensity,
        float moonIntensity,
        float jitterStrength,
        float maxDistance,
        int frameIndex,
        int sampleCount,
        int temporalEnabled,
        int colorTransmissionEnabled,
        float exposure,
        Float2 jitterOffset,
        Float2 invDisplaySize,
        Float3 terrainOrigin,
        Float3 camWorldPos,
        Float3 sunDir,
        Float3 sunIlluminance,
        Float3 moonDir,
        Float3 moonIlluminance
) {
    public static final int BYTE_SIZE = 240;

    public FogIntegrationPushData {
        Objects.requireNonNull(displaySize, "displaySize");
        Objects.requireNonNull(froxelDim, "froxelDim");
        Objects.requireNonNull(jitterOffset, "jitterOffset");
        Objects.requireNonNull(invDisplaySize, "invDisplaySize");
        Objects.requireNonNull(terrainOrigin, "terrainOrigin");
        Objects.requireNonNull(camWorldPos, "camWorldPos");
        Objects.requireNonNull(sunDir, "sunDir");
        Objects.requireNonNull(sunIlluminance, "sunIlluminance");
        Objects.requireNonNull(moonDir, "moonDir");
        Objects.requireNonNull(moonIlluminance, "moonIlluminance");
    }

    public void write(ByteBuffer dst) {
        Objects.requireNonNull(dst, "dst");
        if (dst.capacity() < BYTE_SIZE) throw new IllegalArgumentException("FogIntegrationPushData buffer too small");
        for (int i = 0; i < BYTE_SIZE; i++) dst.put(i, (byte) 0);
        dst.putLong(0, worldPushAddr());
        dst.putLong(8, tableAddr());
        dst.putLong(16, entityTableAddr());
        dst.putLong(24, materialTableAddr());
        dst.putInt(32, displaySize().x());
        dst.putInt(36, displaySize().y());
        dst.putInt(40, froxelDim().x());
        dst.putInt(44, froxelDim().y());
        dst.putInt(48, froxelDepth());
        dst.putFloat(52, nearPlane());
        dst.putFloat(56, farPlane());
        dst.putFloat(60, density());
        dst.putFloat(64, anisotropy());
        dst.putFloat(68, scattering());
        dst.putFloat(72, extinction());
        dst.putFloat(76, heightFalloff());
        dst.putFloat(80, sunIntensity());
        dst.putFloat(84, moonIntensity());
        dst.putFloat(88, jitterStrength());
        dst.putFloat(92, maxDistance());
        dst.putInt(96, frameIndex());
        dst.putInt(100, sampleCount());
        dst.putInt(104, temporalEnabled());
        dst.putInt(108, colorTransmissionEnabled());
        dst.putFloat(112, exposure());
        // padding 4 bytes to 120
        dst.putFloat(120, jitterOffset().x());
        dst.putFloat(124, jitterOffset().y());
        dst.putFloat(128, invDisplaySize().x());
        dst.putFloat(132, invDisplaySize().y());
        // padding to 144
        dst.putFloat(144, terrainOrigin().x());
        dst.putFloat(148, terrainOrigin().y());
        dst.putFloat(152, terrainOrigin().z());
        dst.putFloat(160, camWorldPos().x());
        dst.putFloat(164, camWorldPos().y());
        dst.putFloat(168, camWorldPos().z());
        dst.putFloat(176, sunDir().x());
        dst.putFloat(180, sunDir().y());
        dst.putFloat(184, sunDir().z());
        dst.putFloat(192, sunIlluminance().x());
        dst.putFloat(196, sunIlluminance().y());
        dst.putFloat(200, sunIlluminance().z());
        dst.putFloat(208, moonDir().x());
        dst.putFloat(212, moonDir().y());
        dst.putFloat(216, moonDir().z());
        dst.putFloat(224, moonIlluminance().x());
        dst.putFloat(228, moonIlluminance().y());
        dst.putFloat(232, moonIlluminance().z());
    }

    public record Int2(int x, int y) {}
    public record Float2(float x, float y) {}
    public record Float3(float x, float y, float z) {}
    public record Int3(int x, int y, int z) {}
}
