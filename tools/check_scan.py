"""Band search and auto channel select, against a sim with one transmitter."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv_rf import bands, scan, sdr  # noqa: E402
from _paths import CAPTURE  # noqa: E402

from _paths import CAPTURE  # noqa: E402  (resolved, never hardcoded)


def show(res: scan.ScanResult) -> None:
    print(f"  span      {bands.format_mhz(res.lo_hz)} .. {bands.format_mhz(res.hi_hz)}")
    print(f"  hops      {res.hops}   gate >= {res.gate_bins} bins at "
          f"+{scan.COARSE_BIN_GATE_DB:.0f} dB or {res.gate_peak_db:.0f} dB peak   "
          f"floor {res.noise_floor_db:.1f} dB")
    print(f"  elapsed   {res.elapsed_s:.2f} s   "
          f"loudest quiet hop {res.peak_noise_db:.1f} dB over floor   "
          f"gate false positives {res.false_positives}")
    if res.error:
        print(f"  note      {res.error}")
    print(f"  candidates {len(res.candidates)}")
    for c in res.candidates:
        print(f"    {c.describe()}")
    best = res.best()
    print(f"  best      {best.describe() if best else 'none'}")


def case_sim_present() -> bool:
    print("--- sim: one transmitter at 5802 MHz (2 MHz above F4) ---")
    src = sdr.SimSource(signal_hz=5_802_000_000)
    src.start()
    time.sleep(0.2)
    try:
        sc = scan.BandScanner(src)
        res = sc.scan(region=bands.Region.US)
        show(res)
        best = res.best()
        if best is None:
            print("  FAIL: nothing found")
            return False
        ok = (
            best.frequency_hz == 5_800_000_000
            and best.band == "F"
            and best.channel == 4
            and best.decodable
        )
        print(f"  snapped to {best.label()}"
              f"  -> {'OK' if ok else 'FAIL: expected decodable F4'}")
        # A transmitter 2 MHz off the chart must not be invented into one, but
        # the honest answer is the nearest channel plus the real frequency.
        print(f"  transmitter is at 5802.0 MHz, chart F4 is 5800.0 MHz: "
              f"reported {bands.format_mhz(best.frequency_hz)} "
              f"+-{best.half_width_hz/1e6:.1f} MHz")
        return ok
    finally:
        src.stop()


def case_offchart_fine_grid() -> bool:
    print("\n--- sim: 5812 MHz, scanned on a 2 MHz grid ---")
    # Nearest chart channels to 5812 are L4 at 5807, B5 at 5809 and F5 at 5820.
    # On a 2 MHz grid the hop centres 5811 and 5813 are further from every
    # channel than the grid's own resolution, so the honest answer there is
    # "off-chart" -- and one 15 MHz transmitter seen across eight hops has to
    # come back as one finding, not eight.
    src = sdr.SimSource(signal_hz=5_812_000_000)
    src.start()
    time.sleep(0.2)
    try:
        sc = scan.BandScanner(src, hop_frac=0.2)   # 2 MHz hops
        print(f"  hop grid {sc.hop_hz/1e6:.1f} MHz -> "
              f"+-{sc.hop_hz/2e6:.1f} MHz resolution")
        res = sc.scan(region=bands.Region.US)
        show(res)
        best = res.best()
        # A tuner cannot see further than its own span, so the measured width of
        # a transmitter is capped at the sample rate however wide the signal
        # really is. Anything at or near 10 MHz means the whole span was lit.
        span_cap = src.sample_rate
        ok = (
            best is not None
            and best.decodable
            and len(res.candidates) == 1
            and best.merged > 1
            and 0.7 * span_cap < best.span_hz <= span_cap
        )
        print(f"  one transmitter -> {len(res.candidates)} finding(s) from "
              f"{best.merged} hops, span {best.span_hz/1e6:.0f} MHz "
              f"(capped by the {span_cap/1e6:.0f} MHz span)"
              f"  -> {'OK' if ok else 'FAIL'}")
        return ok
    finally:
        src.stop()


def case_sim_empty() -> bool:
    print("\n--- sim: empty band (noise only) ---")
    src = sdr.SimSource(video=False)
    src.start()
    time.sleep(0.2)
    try:
        sc = scan.BandScanner(src)
        res = sc.scan(region=bands.Region.US)
        show(res)
        ok = res.best() is None
        print(f"  auto select on a quiet band: {'None (correct)' if ok else 'FAIL'}")
        return ok
    finally:
        src.stop()


def case_file() -> bool:
    print("\n--- file: the known-good 5802 MHz capture ---")
    if not CAPTURE.exists():
        print(f"  SKIP: {CAPTURE} not present")
        return True
    src = sdr.FileSource(str(CAPTURE), frequency_hz=5_802_000_000)
    src.start()
    time.sleep(0.3)
    try:
        sc = scan.BandScanner(src)
        res = sc.scan(region=bands.Region.US)
        show(res)
        best = res.best()
        ok = (
            best is not None
            and best.decodable
            and best.band == "F"
            and best.channel == 4
            and best.frequency_hz == 5_802_000_000
        )
        print(f"  a file must not be swept: retunable={src.retunable}")
        print(f"  identified {best.label() if best else 'nothing'}"
              f"  -> {'OK' if ok else 'FAIL'}")
        return ok
    finally:
        src.stop()


def case_band_span() -> bool:
    print("\n--- one band plan instead of the whole region ---")
    src = sdr.SimSource(signal_hz=5_806_000_000)   # R5
    src.start()
    time.sleep(0.2)
    try:
        sc = scan.BandScanner(src)
        res = sc.scan_band(bands.band_of(5_806_000_000))
        show(res)
        best = res.best()
        # 5806 MHz is R5, O5, L4, B5, A4, U4, H4 and D4 all within 3 MHz of each
        # other, so *which* chart channel the scan names is genuinely ambiguous
        # from a 9 MHz grid. What must hold is that it lands within one hop of
        # the truth and reads as decodable video.
        near = best is not None and abs(best.frequency_hz - 5_806_000_000) <= 4_500_000
        # 5806 MHz carries eight chart channels inside a 9 MHz window, so the
        # only honest claim is "within one hop of the truth, and it decodes".
        ok = near and best.decodable
        print(f"  a band-plan sweep of 5725-5865 found {best.label() if best else 'none'}"
              f"  -> {'OK' if ok else 'FAIL'}")
        return ok
    finally:
        src.stop()


def case_legality() -> bool:
    print("\n--- region legality ---")
    # 5905 MHz is inside the US 5.65-5.95 amateur span and outside every
    # 5725-5875 ISM band, so it has to differ between the tables.
    f = 5_905_000_000
    rows = [(reg.value, bands.is_legal(f, reg)) for reg in bands.REGIONS]
    for name, v in rows:
        plan = bands.REGIONS[bands.Region(name)]
        print(f"  {name:3s} {plan.low_hz/1e6:.0f}-{plan.high_hz/1e6:.0f} MHz: "
              f"5905 MHz legal = {v}")
    ok = rows[0][1] and not any(v for _, v in rows[1:])
    print(f"  -> {'OK' if ok else 'FAIL'}")
    m = bands.match_frequency(5_665_000_000)
    print(f"  5665 MHz -> {m.label if m else None} "
          f"{m.frequency_hz/1e6 if m else 0:.0f} MHz")
    return ok and m is not None


def main() -> int:
    results = [
        case_sim_present(),
        case_offchart_fine_grid(),
        case_sim_empty(),
        case_file(),
        case_band_span(),
        case_legality(),
    ]
    print("\nRESULT:", "PASS" if all(results) else "FAIL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
