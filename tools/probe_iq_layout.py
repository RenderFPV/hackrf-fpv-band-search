"""Decide int8-interleaved vs complex64 for a capture, by stride signatures.

Guessing wrong here is the worst outcome available: it would mean "fixing" the
signed-IQ bug into something still wrong. So this separates the two encodings
structurally rather than by looking at the values.

float32 complex64: every 4th byte (the high byte of the float) encodes the
exponent, and for signal-sized values it can only be a handful of values --
0x3D..0x3F positive, 0xBD..0xBF negative. The other three bytes are mantissa
noise and look uniform. So the high-byte stride has few distinct values and sits
in a narrow band.

int8 interleaved: no such structure. All four stride positions look alike,
because I and Q samples are just bytes with no alignment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SAMPLES = Path.home() / "Documents" / "hackrf-fpv-still-decoder" / "Samples"


def report(path: Path) -> str:
    raw = np.frombuffer(path.read_bytes()[:8_000_000], dtype=np.uint8)
    out = [f"{path.name}  ({path.stat().st_size:,} bytes)"]

    for stride in range(4):
        col = raw[stride::4]
        uniq = np.unique(col)
        # share of the column sitting in the middle third of the byte range
        mid = float(np.mean((col >= 85) & (col < 171)))
        out.append(
            f"  stride {stride}: {len(uniq):3d} distinct values, "
            f"mean {col.mean():6.1f}, {mid*100:5.1f}% in middle third"
        )

    f32 = raw.view(np.float32)
    i8 = raw.view(np.int8).reshape(-1, 2).astype(np.float32)

    # As float32: how many samples are finite and in a sane signal range?
    finite = np.isfinite(f32)
    sane_f32 = float(np.mean(finite & (np.abs(f32) < 8.0)))
    out.append(f"  as complex64: {finite.mean()*100:5.1f}% finite, "
               f"{sane_f32*100:5.1f}% within +/-8")

    # As int8 IQ: amplitude distribution
    i = i8[:, 0]
    q = i8[:, 1]
    amp = np.hypot(i, q)
    out.append(f"  as int8 IQ  : I mean {i.mean():+7.2f} sd {i.std():6.2f} | "
               f"Q mean {q.mean():+7.2f} sd {q.std():6.2f} | "
               f"amp median {np.median(amp):6.1f} p99 {np.percentile(amp, 99):6.1f}")

    # A real FPV FM carrier has strong sample-to-sample amplitude variation.
    # A float misread as bytes, or bytes misread as floats, does not.
    swing = float(np.mean(np.abs(np.diff(amp)) > 1.0))
    out.append(f"  int8 IQ sample-to-sample amplitude swings: {swing*100:5.1f}%")

    return "\n".join(out)


def main() -> int:
    args = sys.argv[1:]
    files = [Path(a) for a in args] if args else [
        SAMPLES / "r5_5802_g16.u8",
        SAMPLES / "fpv_r5_5802_s10m_l32_g16_long_001.u8",
    ]
    for f in files:
        if f.exists():
            print(report(f))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
