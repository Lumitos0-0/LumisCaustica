// Manual push data for FogInjectionPush — layout matches volumetric_fog_common.slang
package dev.comfyfluffy.caustica.rt.gen;

import java.nio.ByteBuffer;
import java.util.Objects;

public record FogInjectionPushData(
        long worldPushAddr,
        Int3 froxelDim,
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
        int sampleSeed,
        int temporalEnabled,
        int colorTransmissionEnabled,
        Float2 jitterOffset,
        Float3 terrainOrigin,
        Float3 camWorldPos
) {
    public static final int BYTE_SIZE = 128;

    public FogInjectionPushData {
        Objects.requireNonNull(froxelDim, "froxelDim");
        Objects.requireNonNull(jitterOffset, "jitterOffset");
        Objects.requireNonNull(terrainOrigin, "terrainOrigin");
        Objects.requireNonNull(camWorldPos, "camWorldPos");
    }

    public void write(ByteBuffer dst) {
        Objects.requireNonNull(dst, "dst");
        if (dst.capacity() < BYTE_SIZE) throw new IllegalArgumentException("FogInjectionPushData buffer too small");
        for (int i = 0; i < BYTE_SIZE; i++) dst.put(i, (byte) 0);
        dst.putLong(0, worldPushAddr());
        // froxelDim at 16
        dst.putInt(16, froxelDim().x());
        dst.putInt(20, froxelDim().y());
        dst.putInt(24, froxelDim().z());
        dst.putFloat(28, nearPlane());
        dst.putFloat(32, farPlane());
        dst.putFloat(36, density());
        dst.putFloat(40, anisotropy());
        dst.putFloat(44, scattering());
        dst.putFloat(48, extinction());
        dst.putFloat(52, heightFalloff());
        dst.putFloat(56, sunIntensity());
        dst.putFloat(60, moonIntensity());
        dst.putFloat(64, jitterStrength());
        dst.putFloat(68, maxDistance());
        dst.putInt(72, frameIndex());
        dst.putInt(76, sampleSeed());
        dst.putInt(80, temporalEnabled());
        dst.putInt(84, colorTransmissionEnabled());
        dst.putFloat(88, jitterOffset().x());
        dst.putFloat(92, jitterOffset().y());
        dst.putFloat(96, terrainOrigin().x());
        dst.putFloat(100, terrainOrigin().y());
        dst.putFloat(104, terrainOrigin().z());
        dst.putFloat(112, camWorldPos().x());
        dst.putFloat(116, camWorldPos().y());
        dst.putFloat(120, camWorldPos().z());
    }

    public record Int3(int x, int y, int z) {}
    public record Float2(float x, float y) {}
    public record Float3(float x, float y, float z) {}
}
