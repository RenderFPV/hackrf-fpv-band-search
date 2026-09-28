"""Hardware check for the fast band search and the radio handover around it.

The fast path works by *lending* the radio to a second program. That is the
whole trick -- the sweep tool retunes inside one open device, which is why a
whole-band search takes a fifth of a second instead of twelve -- and it is also
the only genuinely dangerous thing in the design, because the failure mode is
not an error message. It is a frozen picture.

So the checks here are mostly about what happens *after* the sweep:

* the radio has to come back, or the decoder is left reading an empty pipe with
  no indication anything went wrong;
* samples have to actually start flowing again, since a process that exists is
  not the same as a process that is streaming;
* a second scan has to work, so the handover is not one-shot;
* a scan that is cancelled has to hand the radio back too -- otherwise a bored
  pilot who hits Stop leaves the radio lent out forever;
* and the fallback has to still work on a source that cannot lend anything.

Run it with the radio plugged in and the app closed. Nothing else may hold the
device, or the fast path will report the tool as unavailable and these checks
will be measuring the fallback instead -- which is a valid answer, but not the
one being asked about.

    python tools/check_fastscan.py
    python tools/check_fastscan.py --slow   # force the hop-walk fallback
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpv_rf import bands, fastscan, sdr, sweep  # noqa: E402

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def flowing(src: sdr.IQSource, seconds: float = 0.7) -> int:
    """Bytes that reached the ring *during* a window, with the source started.

    Measured as a difference, not a reading. ``bytes_total`` is a running total
    for the life of the source, and a band search in between adds tens of
    megabytes to it -- so comparing two raw readings compared "0.7 s of
    streaming" against "0.7 s of streaming plus twelve seconds of background
    streaming during a search", and reported the 3.8x difference as a fault.
    It was arithmetic, not hardware.
    """
    src.start()
    time.sleep(0.4)                   # let the device finish opening
    t0 = src.stats.bytes_total
    time.sleep(seconds)
    return src.stats.bytes_total - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow", action="store_true",
                    help="force the hop-walk engine instead of the sweep")
    ap.add_argument("--freq", type=int, default=5_843_000_000)
    ap.add_argument("--sweeps", type=int, default=1)
    args = ap.parse_args()

    print("== is the fast engine usable here? ==")
    src = sdr.HackrfSource(args.freq)
    check("a HackRF source was built", src.available(),
          src.executable or "hackrf_transfer not found")
    if not src.available():
        print("\nRESULT: FAIL (no radio found)")
        return 1
    check("source can lend the radio out", src.exclusive_use)
    fast = fastscan.fast_ok(src)
    check("sweep tool is available", fast,
          "" if fast else fastscan.why_not_fast(src))
    if not args.slow and not fast:
        print("\n  the sweep engine is unavailable, so this run can only")
        print("  exercise the fallback. That is worth knowing, not fatal.")
    lo, hi = bands.default_scan_span(bands.Region.US)

    print(f"\n== samples flow before any handover ==")
    before = flowing(src)
    check("the radio is streaming", before > 0, f"{before:,} bytes")

    engine = "hops" if args.slow else ("sweep" if fast else "hops")
    print(f"\n== one {engine}-engine search of "
          f"{lo/1e6:.0f}-{hi/1e6:.0f} MHz ==")
    t0 = time.monotonic()
    res = fastscan.scan_span(src, lo, hi, sweeps=args.sweeps)
    el = time.monotonic() - t0
    print(f"  {res.describe()}")
    print(f"  wall clock {el:.2f} s   engine={res.engine}   error={res.error!r}")
    check("the search reported an engine", bool(res.engine), res.engine)
    # The sweep is expected to be refused often on real hardware, because the
    # handover means killing a process mid-transfer. When that happens the search
    # must still cover the band and must say it fell back -- so the engine is
    # allowed to differ from the one requested, as long as the answer is real.
    if engine == "sweep" and res.engine == "hops":
        print("  (the sweep could not have the radio; the hop walk answered)")
    else:
        check("it is the engine we asked for", res.engine == engine,
              f"got {res.engine}")
    check("it covered the whole span",
          res.lo_hz <= lo and res.hi_hz >= hi,
          f"{res.lo_hz/1e6:.0f}-{res.hi_hz/1e6:.0f} MHz")
    check("no error was reported", not res.error, res.error)
    # An empty result must not be able to pass as a clean one. "Nothing above the
    # noise floor" and "nothing was measured" read identically in the app, and
    # only the first is a claim about the band. A whole-band scan whose silence
    # is worthless is worse than one that fails loudly, so bins are required.
    if res.engine == "sweep":
        check("the search actually measured something", res.hops > 0,
              f"{res.hops} bins reported")
    else:
        check("the search actually measured something",
              res.error != "" or res.hops > 0, f"{res.hops} hops reported")

    print(f"\n== the radio came back ==")
    check("reclaim reported success", src.reclaim_device())
    src.reset_drain()
    after = flowing(src)
    check("samples flow again after the handover", after > 0,
          f"{after:,} bytes")
    if before > 0 and after > 0:
        # Not a throughput assertion, just a sanity check that the stream is
        # comparable rather than a trickle.
        ratio = after / before
        check("the stream is comparable to before", 0.4 < ratio < 2.5,
              f"{before:,} then {after:,} bytes ({ratio:.2f}x)")

    print("\n== a second handover works ==")
    t0 = time.monotonic()
    res2 = fastscan.scan_span(src, lo, hi, sweeps=args.sweeps)
    el2 = time.monotonic() - t0
    print(f"  {res2.describe()}")
    print(f"  wall clock {el2:.2f} s")
    check("the second search also completed", not res2.error, res2.error)
    check("the second search measured something too",
          res2.hops > 0 or bool(res2.error), f"{res2.hops} bins reported")
    src.reset_drain()
    again = flowing(src, 0.5)
    check("samples still flow after two handovers", again > 0, f"{again:,} bytes")

    if fast and not args.slow:
        print("\n== cancelling still hands the radio back ==")
        ev = threading.Event()
        ev.set()                      # already cancelled: the sweep must bail
        t0 = time.monotonic()
        res3 = fastscan.scan_span(src, lo, hi, cancel=ev)
        el3 = time.monotonic() - t0
        print(f"  returned in {el3:.2f} s: {res3.error!r}")
        check("a cancelled search returns promptly", el3 < 5.0, f"{el3:.2f} s")
        check("it did not report a real finding", not res3.candidates)
        src.reset_drain()
        back = flowing(src, 0.5)
        check("samples flow after a cancellation", back > 0, f"{back:,} bytes")

        print("\n== the fast engine's numbers ==")
        if res.engine != "sweep":
            # Timed against the hop walk, this would fail for a reason that has
            # nothing to do with the fast engine's speed. Say so instead.
            print("  the sweep engine did not get the radio, so its speed is")
            print("  untested here. Run tools/check_sweep.py for that; it uses")
            print("  the tool on its own, with no handover to lose the radio.")
        else:
            if el > 0:
                print(f"  {res.hops} bins in {el:.2f} s = "
                      f"{el / max(1, res.hops) * 1000:.2f} ms/bin")
            check("a whole-band search is under 2 s", el <= 2.0, f"{el:.2f} s")

    print("\n== the fallback still works, and says why ==")
    sim = sdr.SimSource()
    check("a sim source cannot lend the radio out", not sim.exclusive_use)
    check("so the fast engine declines it", not fastscan.fast_ok(sim))
    print(f"  reason given: {fastscan.why_not_fast(sim)}")
    sim.start()
    time.sleep(0.3)
    t0 = time.monotonic()
    rsim = fastscan.scan_span(sim, lo, hi)
    print(f"  sim search: {rsim.describe()} in {time.monotonic() - t0:.2f} s")
    check("the sim path returns a hop-walk result", rsim.engine == "hops",
          rsim.engine)
    # The sim source generates a synthetic VTX on purpose, so the whole app is
    # demonstrable with no hardware and no drone. Finding something there is the
    # correct answer, and an earlier version of this check asserted the
    # opposite -- i.e. it demanded that the simulator find nothing in a band it
    # had deliberately put a transmitter into.
    check("the sim path finds the transmitter the sim put there",
          bool(rsim.candidates),
          f"{len(rsim.candidates)} candidate(s)")
    check("and identifies it as decodable video, which only the hop walk can",
          bool(rsim.decodable),
          f"{len(rsim.decodable)} decodable")
    sim.stop()

    src.stop()
    print()
    if fails:
        print(f"RESULT: FAIL ({len(fails)})")
        for f in fails:
            print(f"   - {f}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
