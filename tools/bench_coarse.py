"""Pick the coarse-pass "is there energy here" statistic from measurements.

The gate decides whether a hop is worth a second look, so it has to separate a
VTX from thermal noise. The obvious statistic -- peak bin minus median bin -- is
not good enough at a short dwell, and it is worth showing why rather than
guessing a threshold.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv_rf import dsp, sdr  # noqa: E402

from _paths import CAPTURE  # noqa: E402  (resolved, never hardcoded)


def stats(iq: np.ndarray, fs: float, nfft: int, averages: int) -> tuple[float, float, int]:
    """(peak-median dB, bin spacing kHz, count of bins > median+6 dB)."""
    freqs, psd_db = dsp.psd(iq, fs, nfft=nfft, averages=averages)
    med = float(np.median(psd_db))
    peak = float(np.max(psd_db))
    above = int(np.count_nonzero(psd_db > med + 6.0))
    bin_khz = float(abs(freqs[1] - freqs[0])) / 1e3 if freqs.size > 1 else 0.0
    return peak - med, bin_khz, above


def take(src: sdr.IQSource, n: int) -> np.ndarray:
    src.start()
    time.sleep(0.3)
    iq = src.take_iq(n)
    src.stop()
    return iq


def main() -> int:
    fs = sdr.DEFAULT_SAMPLE_RATE
    print("Collecting one block per case at the coarse dwell size.\n")

    noise = take(sdr.SimSource(video=False), 16_384)
    video = take(sdr.SimSource(video=True), 16_384)
    real_iq = np.zeros(0, dtype=np.complex64)
    if CAPTURE.exists():
        src = sdr.FileSource(str(CAPTURE), frequency_hz=5_802_000_000)
        src.start()
        time.sleep(0.3)
        real_iq = src.take_iq(16_384)
        src.stop()

    cases = [("sim noise", noise), ("sim video", video), ("real capture", real_iq)]
    for nfft, averages in ((1024, 8), (2048, 8), (2048, 16), (2048, 48), (2048, 128)):
        need = nfft * 2 * averages
        print(f"nfft={nfft} averages={averages} "
              f"(needs {need/1000:.0f}k samples, {need/fs*1000:.1f} ms)")
        print(f"  {'case':12s} {'peak-med':>9s} {'bins>med+6':>11s} {'bin kHz':>8s}")
        for name, iq in cases:
            if iq.size < need:
                # emulate the shorter dwell by using as many segments as fit
                av = max(2, int(iq.size // (2 * nfft)))
                pm, bk, ab = stats(iq, fs, nfft, av)
                print(f"  {name:12s} {pm:9.1f} {ab:11d} {bk:8.2f}   "
                      f"(only {av} averages fit in 16k)")
            else:
                pm, bk, ab = stats(iq, fs, nfft, averages)
                print(f"  {name:12s} {pm:9.1f} {ab:11d} {bk:8.2f}")
        print()

    print("Repeat the chosen config on many independent noise blocks, to see")
    print("the spread rather than a single sample:\n")
    src = sdr.SimSource(video=False)
    src.start()
    time.sleep(0.3)
    pms, abs_, floors, flatness = [], [], [], []
    for _ in range(200):
        iq = src.take_iq(16_384)
        if iq.size < 16_384:
            time.sleep(0.005)
            continue
        pm, bk, ab = stats(iq, fs, 1024, 8)
        pms.append(pm)
        abs_.append(ab)
        _, psd_db = dsp.psd(iq, fs, nfft=1024, averages=8)
        med = float(np.median(psd_db))
        floors.append(med)
        # flatness: spread of the spectrum away from the extreme edges
        flatness.append(float(np.std(psd_db[8:-8])))
    src.stop()
    pms = np.array(pms)
    abs_ = np.array(abs_)
    flatness = np.array(flatness)
    print(f"  {len(pms)} independent noise blocks at the coarse dwell")
    print(f"  peak-median      : min {pms.min():.1f}  median {np.median(pms):.1f}  "
          f"max {pms.max():.1f} dB")
    print(f"  bins > median+6dB: min {abs_.min()}  median {np.median(abs_):.0f}  "
          f"max {abs_.max()}  (false positives: "
          f"{int(np.count_nonzero(abs_ >= 3))}/{len(abs_)})")
    print(f"  spectrum floor   : {np.median(floors):.1f} dB, in-band flatness "
          f"std {np.median(flatness):.1f} dB (a 1/f^2 hump would be far larger)")
    print(f"\n  -> a gate on peak-median must exceed {pms.max():.1f} dB to be safe;")
    print(f"     a gate on bin count must exceed {abs_.max()}.")
    return 0 if abs_.max() < 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
