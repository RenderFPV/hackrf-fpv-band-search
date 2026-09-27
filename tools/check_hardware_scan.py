"""Does a sweep of a band with nothing on it come back empty, on real hardware?

This is the check that matters and the one that could not be run before. The
simulator has no DC leakage, so its 0-false-positive result said nothing about
the HackRF, whose local oscillator puts a 52 dB carrier at the centre of every
window. If the guard band is not doing its job, this reports 34 candidates for
one receiver and the answer is obvious.

Two things are checked, because they fail differently:
  * the coarse pass, hop by hop -- does any hop clear the gate?
  * a full sweep -- does the whole band come back empty?

The occupied cases are the control. If nothing is transmitting, the controls
cannot run, and the script says so instead of quietly passing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpv_rf import bands, dsp, scan, sdr  # noqa: E402


def main() -> int:
    lo, hi = bands.default_scan_span(bands.Region.US)
    src = sdr.make_source("hackrf", 5_802_000_000)
    src.start()
    time.sleep(0.4)
    print(f"source: {src.describe()}")
    print(f"sweeping {bands.format_mhz(lo)} to {bands.format_mhz(hi)}\n")

    sc = scan.BandScanner(src, hop_frac=0.9)
    sc.hop_hz = int((hi - lo) / ((hi - lo) / 9_000_000)) or 9_000_000

    print(f"{'hop MHz':>9}  {'bins>gate':>9}  {'peak dB':>8}  {'offset':>9}  verdict")
    bad = 0
    try:
        n_hops = 0
        h = lo
        while h < hi:
            iq = src.retune_settled(h, 16_384, timeout=6.0)
            if iq.size < 8192:
                print(f"{h/1e6:9.1f}  read failed ({iq.size} samples)")
                bad += 1
                n_hops += 1
                h += sc.hop_hz
                continue
            hit = sc._coarse_hit(h, iq)
            occupied = (
                hit.bins_above >= scan.COARSE_BIN_COUNT
                or hit.peak_db >= scan.COARSE_PEAK_GATE_DB
            )
            print(f"{h/1e6:9.1f}  {hit.bins_above:9d}  {hit.peak_db:8.1f}  "
                  f"{hit.peak_offset_hz/1e6:+9.2f}  "
                  f"{'OCCUPIED' if occupied else 'quiet'}")
            if occupied:
                bad += 1
            n_hops += 1
            h += sc.hop_hz
        print(f"\n{n_hops} hops, {bad} read as occupied")

        print("\nfull sweep through the scanner:", flush=True)
        res = sc.scan(lo, hi, bands.Region.US)
        print(f"  {res.describe()}")
        print(f"  candidates: {len(res.candidates)}, "
              f"false positives: {res.false_positives}")
        for c in res.candidates:
            print(f"    {c.label()}")
        empty = not res.candidates and res.false_positives == 0
        print(f"\nRESULT: {'PASS' if (bad == 0 and empty) else 'FAIL'} "
              f"-- a band with nothing on it "
              f"{'came back empty' if empty else 'did NOT come back empty'}")
        print(f"        {src.stats.open_retries} device-open retries, "
              f"{src.stats.restarts} restarts, "
              f"{src.stats.bytes_total/2e6:.1f} Msamples")
        return 0 if (bad == 0 and empty) else 1
    finally:
        src.stop()


if __name__ == "__main__":
    raise SystemExit(main())
