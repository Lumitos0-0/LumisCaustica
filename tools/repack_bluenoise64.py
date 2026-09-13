#!/usr/bin/env python3
"""Repack Christoph Peters' CC0 3D blue noise into the fog jitter volume.

Input: 64_64_64/HDR_LA.raw from https://github.com/Calinou/free-blue-noise-textures
(mirror of momentsingraphics.de; Copyright 2016-2017 Christoph Peters, CC0 1.0).
The .raw holds a 6xint32 header [version, channels, dims, sx, sy, sz] followed by
uint32 rank values in [0, 64^3), x fastest.

Output: RGBA8 64x64x64, one texel per (pixel.xy, frame):
  R = L(x, y, f)   G = A(x, y, f)   B = L(x, y, f+32)   A = A(x, y, f+32)   (mod 64)
L/A is Peters' decorrelated channel pair; the +32-slice pair is ~independent, so
all four channels are usable as independent spatiotemporal jitter sources. The fog
bake reads one texel per froxel column per frame (nearest, repeat wrap).

Usage: tools/repack_bluenoise64.py <HDR_LA.raw> <output.bin>
Verifies size, value range, permutation integrity, and blue (negative)
neighbor correlation before writing.
"""
import struct
import sys

SIZE = 64
N = SIZE * SIZE * SIZE
OFFSET = 32


def main() -> None:
    src_path, dst_path = sys.argv[1], sys.argv[2]
    with open(src_path, "rb") as f:
        blob = f.read()
    header = struct.unpack("<6I", blob[:24])
    assert header == (1, 2, 3, SIZE, SIZE, SIZE), f"unexpected header {header}"
    assert len(blob) == 24 + N * 2 * 4, f"unexpected size {len(blob)}"
    # ranks[channel][texel], x fastest: idx = x + 64*y + 64*64*z
    words = struct.unpack(f"<{N * 2}I", blob[24:])
    chans = (words[0::2], words[1::2])
    for c, ranks in enumerate(chans):
        assert min(ranks) == 0 and max(ranks) == N - 1, f"ch{c} not a full rank range"
        assert len(set(ranks)) == N, f"ch{c} ranks are not a permutation"
    print(f"header ok, both channels are full 0..{N - 1} permutations")

    # Normalize to float for statistics (kept as ranks for quantization).
    def val(rank: int) -> float:
        return rank / N

    means = [sum(val(r) for r in ranks) / N for ranks in chans]
    print(f"channel means: {means[0]:.4f} {means[1]:.4f} (want ~0.5)")

    def neighbor_corr(ranks, stride: int) -> float:
        m = 0.5
        n = 0
        acc = 0.0
        for i in range(N):
            j = i + stride
            # Stay within the axis (x/y/z neighbor, no wraparound pairing).
            ax = stride == 1 and (i % SIZE) == SIZE - 1
            ay = stride == SIZE and ((i // SIZE) % SIZE) == SIZE - 1
            az = stride == SIZE * SIZE and (i // (SIZE * SIZE)) == SIZE - 1
            if ax or ay or az or j >= N:
                continue
            acc += (val(ranks[i]) - m) * (val(ranks[j]) - m)
            n += 1
        return acc / n / (1.0 / 12.0)  # uniform variance 1/12

    for axis, stride in (("x", 1), ("y", SIZE), ("z", SIZE * SIZE)):
        corrs = [neighbor_corr(ranks, stride) for ranks in chans]
        print(f"{axis}-neighbor correlation: L={corrs[0]:+.3f} A={corrs[1]:+.3f} (want < 0, blue)")
        assert all(c < 0.0 for c in corrs), "lost blue property?"

    out = bytearray(N * 4)
    for f in range(SIZE):
        f2 = (f + OFFSET) % SIZE
        for y in range(SIZE):
            for x in range(SIZE):
                i0 = x + SIZE * y + SIZE * SIZE * f
                i1 = x + SIZE * y + SIZE * SIZE * f2
                o = (x + SIZE * y + SIZE * SIZE * f) * 4
                out[o] = min(255, int(chans[0][i0] / N * 255 + 0.5))
                out[o + 1] = min(255, int(chans[1][i0] / N * 255 + 0.5))
                out[o + 2] = min(255, int(chans[0][i1] / N * 255 + 0.5))
                out[o + 3] = min(255, int(chans[1][i1] / N * 255 + 0.5))
    with open(dst_path, "wb") as f:
        f.write(out)
    print(f"wrote {dst_path} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
