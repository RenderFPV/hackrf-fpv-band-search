"""Name the exact bin that assess_channel is calling a 54 dB carrier.

`describe()` prints the offset to two decimals, so "+0.00 MHz" only says the peak
is within 5 kHz of the centre of the window -- which is the receiver measuring
itself, or a real signal, and those need opposite handling. This uses dsp.psd()
exactly as assess_channel does, on a fresh capture, and lists the top bins.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from fpv_rf import dsp, sdr  # noqa: E402


def report(tag: str, iq: np.ndarray, rate: int) -> None:
    freqs, psd_db = dsp.psd(iq, rate)
    order = np.argsort(psd_db)[::-1][:12]
    r = dsp.assess_channel(iq, rate)
    print(f"\n{tag}: {iq.size} samples, psd {freqs.size} bins "
          f"({rate/freqs.size/1e3:.1f} kHz each)")
    print(f"  assess_channel: {r.describe()}")
    print(f"  peak_to_median {r.peak_to_median_db:.1f} dB, "
          f"snr {r.snr_db:.1f} dB, median {np.median(psd_db):.1f} dB")
    print(f"  {'rank':>4}  {'bin':>6}  {'MHz':>9}  {'dB':>7}  {'over median':>11}")
    for k, b in enumerate(order):
        print(f"  {k:>4}  {int(b):>6}  {freqs[b]/1e6:+9.4f}  {psd_db[b]:7.1f}  "
              f"{psd_db[b]-np.median(psd_db):11.1f}")


def main() -> int:
    src = sdr.make_source("hackrf", 5_800_000_000)
    src.start()
    time.sleep(0.4)
    for tag, n in (("200k (as the scanner reads it)", 200_000),
                   ("1M", 1 << 20)):
        iq = src.retune_settled(5_800_000_000, n, timeout=8.0)
        if iq.size < n // 2:
            print(f"{tag}: capture too small ({iq.size})")
            continue
        report(tag, iq, src.sample_rate)
    src.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
