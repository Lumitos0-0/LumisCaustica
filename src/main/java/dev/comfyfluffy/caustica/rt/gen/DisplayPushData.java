// GENERATED manually for fog support — mirrors display_common.slang DisplayPush
package dev.comfyfluffy.caustica.rt.gen;

import java.nio.ByteBuffer;
import java.util.Objects;

public record DisplayPushData(
        int hdrEnabled,
        float lutSize,
        float gamma,
        float hdrPeakNits,
        int lookEnabled,
        float lookLutSize,
        float bloomStrength,
        int fogEnabled,
        float fogReserved0,
        float fogReserved1,
        float fogReserved2
) {
    public static final int BYTE_SIZE = 44;

    public DisplayPushData {
    }

    public DisplayPushData(int hdrEnabled, float lutSize, float gamma, float hdrPeakNits,
                           int lookEnabled, float lookLutSize, float bloomStrength) {
        this(hdrEnabled, lutSize, gamma, hdrPeakNits, lookEnabled, lookLutSize, bloomStrength, 0, 0f, 0f, 0f);
    }

    public void write(ByteBuffer dst) {
        Objects.requireNonNull(dst, "dst");
        if (dst.capacity() < BYTE_SIZE) throw new IllegalArgumentException("DisplayPushData buffer too small");
        for (int i = 0; i < BYTE_SIZE; i++) dst.put(i, (byte) 0);
        dst.putInt(0, hdrEnabled());
        dst.putFloat(4, lutSize());
        dst.putFloat(8, gamma());
        dst.putFloat(12, hdrPeakNits());
        dst.putInt(16, lookEnabled());
        dst.putFloat(20, lookLutSize());
        dst.putFloat(24, bloomStrength());
        dst.putInt(28, fogEnabled());
        dst.putFloat(32, fogReserved0());
        dst.putFloat(36, fogReserved1());
        dst.putFloat(40, fogReserved2());
    }

    public record Float2(float x, float y) {}
    public record Float3(float x, float y, float z) {}
    public record Int2(int x, int y) {}
    public record Int3(int x, int y, int z) {}
}
