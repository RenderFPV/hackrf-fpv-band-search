"""Determine the byte format of a real hackrf_transfer capture.

Signed int8 and unsigned-offset uint8 are indistinguishable by eye but not by
histogram, because the two encodings place a zero-mean signal in different
places on the 0..255 axis:

  signed int8   s in [-128, 127]  -> byte = s & 0xFF
                  small amplitude: bytes pile up near 0 and near 255
                  full amplitude:  spread evenly over 0..255
  unsigned      s in [-1, +1]     -> byte = round(s*127.5) + 128
                  always centred on 128, never touching the extremes

So the share of bytes in the outer quarters settles it:

  >40% in [0,64) or [192,256)   -> signed
  <5%  there, concentrated on 128 -> unsigned

Run:  python tools/probe_iq_format.py [file.u8 ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DEFAULT = Path.home() / "Documents" / "hackrf-fpv-still-decoder" / "Samples"


def describe(path: Path, limit: int = 8_000_000) -> None:
    raw = path.read_bytes()[:limit]
    b = np.frombuffer(raw, dtype=np.uint8)

    outer = float(np.mean((b < 64) | (b >= 192)))
    mid = float(np.mean((b >= 108) & (b < 148)))
    signed_mean = float(b.astype(np.float32).mean() - 128.0)   # ~0 if signed
    unsig_mean = float(np.abs(b.astype(np.float32).mean() - 128.0))

    verdict = "SIGNED int8" if outer > 0.40 else (
        "UNSIGNED offset" if mid > 0.80 else "inconclusive"
    )

    print(f"{path.name}")
    print(f"  bytes            {len(b):,}")
    print(f"  outer quarters   {outer*100:5.1f}%   (signed => high, unsigned => low)")
    print(f"  within 108..147  {mid*100:5.1f}%   (unsigned => high)")
    print(f"  mean byte        {b.mean():6.1f}   deviation from 128: {signed_mean:+.1f}")
    print(f"  p01/p50/p99      "
          f"{np.percentile(b, 1):.0f} / {np.percentile(b, 50):.0f} / "
          f"{np.percentile(b, 99):.0f}")
    print(f"  => {verdict}")
    print()


def main() -> int:
    args = sys.argv[1:]
    if args:
        files = [Path(a) for a in args]
    else:
        files = sorted(DEFAULT.glob("*_g16.u8"))[:6]
    if not files:
        print("no captures found")
        return 1
    for f in files:
        if f.exists():
            describe(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
