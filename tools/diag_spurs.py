"""Is the carrier at +0.00 MHz real RF, or the HackRF hearing itself?

Tuning to several frequencies tells us. A real signal stays at a fixed
frequency while the tuner moves, so it appears at a *different* offset inside
each 10 MHz window. The receiver's own LO leakage, by contrast, sits at the
centre of every window and never moves -- and a band search would then find
one "transmitter" per hop, which is worth knowing about before trusting a
sweep on hardware.

Run with --steps N to keep the retune count down while debugging a hang.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from fpv_rf import dsp, sdr  # noqa: E402

# A spread, not a line: LO leakage and a real signal are indistinguishable if
# every step is near the frequency we already suspect.
STEPS = (
    5_660_000_000,
    5_700_000_000,
    5_802_000_000,
    5_850_000_000,
    5_900_000_000,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=len(STEPS))
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--timeout", type=float, default=6.0)
    args = ap.parse_args()
    steps = STEPS[: max(0, args.steps)]

    src = sdr.make_source("hackrf", steps[0] if steps else 5_802_000_000)
    print(f"source: {src.describe()}", flush=True)
    src.start()
    time.sleep(0.4)
    print(f"{'tuned MHz':>10}  {'peak-med':>9}  {'offset':>9}  "
          f"{'noise dB':>9}  verdict", flush=True)
    try:
        for f in steps:
            t0 = time.monotonic()
            iq = src.retune_settled(f, args.samples, timeout=args.timeout)
            waited = time.monotonic() - t0
            if iq.size < args.samples // 2:
                print(f"{f/1e6:10.1f}  retune returned only {iq.size} samples "
                      f"after {waited:.1f}s  "
                      f"bytes={src.stats.bytes_total} "
                      f"err={src.stats.last_error!r}", flush=True)
                continue
            r = dsp.assess_channel(iq, src.sample_rate)
            print(f"{f/1e6:10.1f}  {r.peak_to_median_db:9.1f}  "
                  f"{r.peak_offset_hz/1e6:+9.2f}  {r.noise_db:9.1f}  "
                  f"[{waited:4.1f}s]  {r.describe()[:56]}", flush=True)
    finally:
        src.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
