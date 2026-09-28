"""Hardware check for the fast sweep path.

Everything else in this project's test history has been a *negative* -- a
transmitter correctly not appearing. Nothing confirmed the instrument sees one
when there is one, so the detection path had never been exercised against a
signal.

This closes that gap without a drone and without transmitting, by using the
2.4 GHz band as a known-answer test. It reliably carries real WiFi: repeated
scans here found a signal at +42 dB over floor around 2430-2433 MHz, and a
floor-to-peak spread of 74 dB across the band.

That signal is *intermittent*, though, and the test has to admit it. Three
consecutive single-pass scans gave spreads of 13, 42 and 15 dB -- one pass
caught the transmitter and two missed it entirely. A WiFi access point beacons
rather than transmitting continuously, and a 130 ms sweep samples it only
sometimes. So the test retries and reports how many attempts it needed, which
is itself the honest description of the environment.

This matters for the real use case in the other direction: FPV video transmits
continuously, so it is caught on every pass, and the default single sweep is
the right setting for it.

It also checks the three ways a spectrum sweep can mislead:

* a centre-frequency spur, which would appear as a regular comb of
  "transmitters" across the band;
* a gate that drifts, so that sweeping repeatedly manufactures findings out of
  an empty band -- the failure mode that would make a scan lie about being
  quiet;
* coverage, since the tool rounds the requested span to a 20 MHz grid and
  reporting the wrong span would be worse than reporting none.

    python tools/check_sweep.py           # 2.4 GHz known-answer, spur, coverage
    python tools/check_sweep.py --band    # also the real 5.8 GHz band
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpv_rf import bands, sweep  # noqa: E402

#: 2.4 GHz, where this machine can see real traffic.
BUSY_LO, BUSY_HI = 2_400_000_000, 2_500_000_000

#: The 5.8 GHz band is empty here, which is the other half of the test: a
#: scanner that reports transmitters in an empty band is worse than useless.
QUIET_LO, QUIET_HI = bands.default_scan_span(bands.Region.US)

fails: list[str] = []
skipped: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}"
          + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def skip(name: str, why: str) -> None:
    print(f"  --   {name}   {why}")
    skipped.append(name)


def spur_report(trace: sweep.SweepTrace) -> None:
    """Look for a comb: energy at a constant offset inside every row.

    A centre-frequency artefact is present in every capture the tool makes, so
    one would appear at a constant frequency *relative to each row* -- a
    periodic pattern in frequency order. Real transmitters do not do that.

    The subtlety is that the periodicity must be measured between *separate
    groups*, not between adjacent bins. A real 15 MHz-wide signal lights up 15
    consecutive bins, whose gaps are all one bin wide, and that is
    indistinguishable from a comb if the gaps are taken between neighbouring
    hits. An earlier version of this check did exactly that and failed a
    perfectly good detection: 16 contiguous bins, 15 one-MHz gaps, and a
    "periodic" verdict on a real WiFi transmitter.

    So the hits are grouped into runs first. A comb is regular spacing between
    runs; a single run cannot be a comb at all.
    """
    floor = trace.floor_db()
    wide = sweep.wideband_db(trace.bins, floor)
    sigma = sweep.robust_sigma([b.db for b in wide])
    gate = sweep.robust_gate(sigma, trace.sweeps)
    over = sorted(b.freq_hz for b in trace.bins if b.db >= floor + gate)
    print(f"\n  raw bins over the {gate:.1f} dB gate: {len(over)} of "
          f"{len(trace.bins)}  (wideband sigma {sigma:.2f} dB, "
          f"{trace.sweeps} sweep(s), {trace.stat})")
    if not over:
        skip("comb / spur check", "nothing over the gate to examine")
        return

    step = max(1, trace.bin_width_hz)
    runs: list[list[int]] = [[over[0]]]
    for f in over[1:]:
        if f - runs[-1][-1] <= step * 1.5:
            runs[-1].append(f)
        else:
            runs.append([f])
    centres = [(r[0] + r[-1]) // 2 for r in runs]
    print(f"  {len(over)} hits in {len(runs)} group(s), bin counts "
          f"{[len(r) for r in runs][:8]}")
    if len(runs) < 4:
        skip("comb / spur check",
             f"{len(runs)} group(s) is too few to be periodic")
        return
    gaps = [centres[i + 1] - centres[i] for i in range(len(centres) - 1)]
    med = statistics.median(gaps)
    same = sum(1 for g in gaps if med and abs(g - med) <= med * 0.05)
    print(f"  spacing between group centres: median {med/1e6:.2f} MHz; "
          f"{same} of {len(gaps)} within 5% of it")
    check("hits are not a regular comb (no centre-frequency spur)",
          same < len(gaps) * 0.4,
          f"{same}/{len(gaps)} identical -> periodic")


def find_signal(attempts: int, sweeps: int) -> tuple[sweep.SweepTrace | None,
                                                   sweep.ScanResult | None,
                                                   int]:
    """Retry until the intermittent 2.4 GHz signal shows up.

    Reports the attempt count rather than hiding it. A test that quietly
    retried forty times would look identical to one that found it first try,
    and the difference matters: it is the difference between "the detector
    works" and "the detector works on a transmitter that is on all the time".
    """
    for i in range(1, attempts + 1):
        tr = sweep.run_sweep(BUSY_LO, BUSY_HI, sweeps=sweeps)
        if not tr.ok:
            continue
        res = sweep.result_from_trace(tr)
        if res.candidates:
            return tr, res, i
    return None, None, attempts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", action="store_true",
                    help="also sweep the real 5.8 GHz band")
    ap.add_argument("--sweeps", type=int, default=1)
    ap.add_argument("--attempts", type=int, default=12)
    args = ap.parse_args()

    print("== can the fast path run at all? ==")
    tool = sweep.runnable_tool()
    check("sweep tool found and made runnable", tool is not None,
          str(tool) if tool else sweep.unavailable_reason())
    if tool is None:
        print("\nRESULT: FAIL (no fast path on this machine)")
        return 1
    check("tool starts and produces output", sweep.available())
    print(f"  fftw source: {sweep.find_fftw()}")

    print(f"\n== 2.4 GHz known-answer test ({BUSY_LO/1e6:.0f}-"
          f"{BUSY_HI/1e6:.0f} MHz, {args.sweeps} sweep(s) each) ==")
    print("   a correct answer is 'found something'; the signal is"
          " intermittent, so attempts are counted")
    tr, res, tries = find_signal(args.attempts, args.sweeps)
    if tr is None or res is None:
        print(f"  no candidates in {args.attempts} attempts")
        check("found real signals in a crowded band", False,
              "detection path would be untested if this is zero")
        return 1
    print(f"  found after {tries} attempt(s) of {args.attempts}")
    print(f"  {tr.describe()}")
    print(f"  {len(res.candidates)} candidate(s) from {len(tr.bins)} bins")
    for c in res.candidates[:8]:
        print(f"    {c.label()}   excess {c.reading.snr_db:+.1f} dB"
              f"   over {c.span_hz/1e6:.1f} MHz")
    check("found real signals in a crowded band", True)
    check("every candidate carries a frequency",
          all(c.frequency_hz for c in res.candidates))
    check("candidates are sorted strongest first",
          all(res.candidates[i].strength >= res.candidates[i + 1].strength
              for i in range(len(res.candidates) - 1)))
    check("candidates resolve to a real frequency, not a bin edge",
          all(tr.coverage_lo_hz <= c.frequency_hz <= tr.coverage_hi_hz
              for c in res.candidates))
    check("best() returns something when candidates exist", res.best() is not None)
    spur_report(tr)

    print("\n== coverage honesty ==")
    want_lo, want_hi = sweep.round_to_grid(BUSY_LO, BUSY_HI)
    check("grid rounding covers the request",
          want_lo <= BUSY_LO and want_hi >= BUSY_HI,
          f"{want_lo/1e6:.0f}-{want_hi/1e6:.0f} MHz")
    # Against the coverage edges, not the outermost bin centres: the centres sit
    # half a bin inside the span, so comparing them to the request fails on a
    # trace that in fact covered it exactly.
    check("reported coverage matches the request",
          (tr.coverage_lo_hz, tr.coverage_hi_hz) == (want_lo, want_hi),
          f"asked {want_lo/1e6:.0f}-{want_hi/1e6:.0f}, got "
          f"{tr.coverage_lo_hz/1e6:.1f}-{tr.coverage_hi_hz/1e6:.1f}")
    check("coverage edges sit outside the outermost bin centres",
          tr.coverage_lo_hz < tr.actual_lo_hz
          <= tr.actual_hi_hz < tr.coverage_hi_hz,
          f"cov {tr.coverage_lo_hz/1e6:.1f}-{tr.coverage_hi_hz/1e6:.1f}, "
          f"centres {tr.actual_lo_hz/1e6:.1f}-{tr.actual_hi_hz/1e6:.1f}")
    check("bins are contiguous and evenly spaced",
          all(b.freq_hz > a.freq_hz for a, b in zip(tr.bins, tr.bins[1:]))
          and len({b.freq_hz - a.freq_hz
                   for a, b in zip(tr.bins, tr.bins[1:])}) <= 2)

    print("\n== the quiet band must stay quiet ==")
    print("   5.8 GHz has nothing on air here. A scanner that reports")
    print("   transmitters in an empty band is worse than no scanner, so this")
    print("   is checked at several sweep counts -- the loudest-of-N statistic")
    print("   runs high on noise, and a gate that did not price that in would")
    print("   drift upward with every extra pass.")
    for n in (1, 5, 20):
        t0 = time.monotonic()
        q = sweep.run_sweep(QUIET_LO, QUIET_HI, sweeps=n)
        el = time.monotonic() - t0
        if not q.ok:
            check(f"{n} sweep(s) of the empty band succeeds", False, q.error)
            continue
        qr = sweep.result_from_trace(q)
        wide = sweep.wideband_db(q.bins, q.floor_db())
        sigma = sweep.robust_sigma([b.db for b in wide])
        gate = sweep.robust_gate(sigma, n)
        peak = max((b.db for b in wide), default=0.0)
        print(f"  {n:2d} sweep(s): {el:5.2f} s  {len(qr.candidates)} candidate(s)"
              f"  floor {q.floor_db():.1f} dB  sigma {sigma:.2f}  gate {gate:.1f}"
              f"  peak {peak:+.1f}  margin {gate - peak:+.1f} dB")
        check(f"{n} sweep(s) of the empty band reports nothing", not qr.candidates,
              f"{len(qr.candidates)} candidate(s): "
              f"{[c.label() for c in qr.candidates[:3]]}")
        check(f"{n} sweep(s) keeps real margin below the gate", gate - peak > 3.0,
              f"only {gate - peak:.1f} dB of headroom")

    if args.band:
        print("\n== the real 5.8 GHz band, timed ==")
        # Kept per sweep count rather than in one `el`, so the timing check
        # below reports the single-sweep figure it claims to. Reading the last
        # value out of the loop instead silently checks the ten-sweep time
        # against a one-sweep target.
        one_sweep_s: float | None = None
        for n in (1, 10):
            t0 = time.monotonic()
            q = sweep.run_sweep(QUIET_LO, QUIET_HI, sweeps=n)
            el = time.monotonic() - t0
            if n == 1:
                one_sweep_s = el
            print(f"  {n:2d} sweep(s): {el:5.2f} s  {q.describe()}")
            for c in sweep.result_from_trace(q).candidates[:5]:
                print(f"      {c.label()}  {c.reading.snr_db:+.1f} dB")
        check("a single sweep of the US band is under 1 s",
              one_sweep_s is not None and one_sweep_s <= 1.0,
              f"{one_sweep_s:.2f} s" if one_sweep_s is not None else "not run")

    print()
    if fails:
        print(f"RESULT: FAIL ({len(fails)})")
        for f in fails:
            print(f"   - {f}")
        return 1
    if skipped:
        print(f"RESULT: PASS with {len(skipped)} check(s) inconclusive: "
              + ", ".join(skipped))
        return 0
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
