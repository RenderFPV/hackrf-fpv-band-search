"""Alerting: dedupe, cooldown, lost signals, and the band+channel+frequency text.

Driven by a fake clock, because the interesting behaviour all lives on the other
side of a time threshold and a test that sleeps its way there is slow and flaky.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv_rf import alerts, bands, dsp, scan  # noqa: E402


class Clock:
    def __init__(self) -> None:
        self.t = 1_700_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def cand(
    freq_hz: int = 5_800_000_000,
    kind: dsp.SignalKind = dsp.SignalKind.VIDEO,
    qual: float = 0.98,
    snr: float = 9.0,
) -> scan.Candidate:
    return scan.Candidate(
        frequency_hz=freq_hz,
        reading=dsp.ChannelReading(
            kind=kind,
            snr_db=snr,
            peak_to_median_db=18.0 if kind is not dsp.SignalKind.NOISE else 4.0,
            noise_db=-30.0,
            sync_pulses=1200,
            sync_density=0.95,
            sync_quality=qual,
            line_rate_hz=dsp.NTSC.line_rate_hz,
            spec=dsp.NTSC,
            peak_offset_hz=0.0,
        ),
        match=bands.match_frequency(freq_hz),
        uncertainty_hz=9_000_000,
    )


def result(cands: list[scan.Candidate], **kw) -> scan.ScanResult:
    r = scan.ScanResult(lo_hz=5_650_000_000, hi_hz=5_950_000_000, **kw)
    r.candidates = cands
    return r


def case_repeat_is_not_an_alert() -> bool:
    print("--- the same transmitter every sweep ---")
    clock = Clock()
    eng = alerts.AlertEngine(clock=clock)
    got = [eng.update(result([cand()])) for _ in range(10)]
    n = sum(len(g) for g in got)
    ev = eng.log[-1]
    live = eng.repeat_count(cand())
    print(f"  10 sweeps of one steady signal -> {n} alert(s)")
    print(f"  alert text: {ev.headline()}")
    print(f"  seen at raise time {ev.seen}, live repeat count {live}")
    ok = n == 1 and ev.seen == 1 and live == 10
    print(f"  -> {'OK' if ok else 'FAIL'}")
    return ok


def case_cooldown_expires() -> bool:
    print("\n--- still there after the cooldown ---")
    clock = Clock()
    eng = alerts.AlertEngine(clock=clock)
    first = eng.update(result([cand()]))
    clock.advance(eng.cooldown_s + 0.1)
    again = eng.update(result([cand()]))
    print(f"  immediately: {len(first)} alert")
    print(f"  after {eng.cooldown_s:.0f} s: {len(again)} alert")
    print(f"  {eng.log[-1].headline()}")
    ok = len(first) == 1 and len(again) == 1
    print(f"  -> {'OK' if ok else 'FAIL'}")
    return ok


def case_lost_and_returned() -> bool:
    print("\n--- signal goes away, then comes back ---")
    clock = Clock()
    eng = alerts.AlertEngine(clock=clock)
    eng.update(result([cand()]))
    clock.advance(eng.forget_s + 1.0)
    gone = eng.update(result([]))
    print(f"  after {eng.forget_s:.0f} s of silence: "
          f"{[e.kind.value for e in gone]}")
    back = eng.update(result([cand()]))
    print(f"  it returns: {back[0].headline() if back else 'no alert'}")
    ok = (
        len(gone) == 1
        and gone[0].kind is alerts.AlertKind.LOST
        and len(back) == 1
        and back[0].kind is alerts.AlertKind.FOUND
    )
    print(f"  -> {'OK' if ok else 'FAIL'}")
    return ok


def case_incomplete_scan_is_not_evidence() -> bool:
    print("\n--- a cancelled scan must not report everything as lost ---")
    clock = Clock()
    eng = alerts.AlertEngine(clock=clock)
    eng.update(result([cand()]))
    clock.advance(eng.forget_s + 1.0)
    got = eng.update(None)
    print(f"  update(None) -> {len(got)} alert(s)")
    print(f"  summary: {eng.summary()}")
    ok = len(got) == 0
    print(f"  -> {'OK' if ok else 'FAIL'}")
    return ok


def case_fine_frequency_shift_is_same() -> bool:
    print("\n--- a sweep whose grid moved is still the same drone ---")
    clock = Clock()
    eng = alerts.AlertEngine(clock=clock)
    eng.update(result([cand(5_800_000_000)]))
    # A different hop grid lands the same VTX 3 MHz higher and, because 5803 is
    # nearer A4/U4/H4/D4 at 5805 than F4 at 5800, reports a different channel.
    # A key built from the label or from a frequency grid would call this new.
    for f in (5_803_000_000, 5_798_000_000, 5_801_000_000):
        moved = cand(f)
        moved.uncertainty_hz = 4_000_000
        got = eng.update(result([moved]))
        print(f"  {bands.format_mhz(f)} -> {len(got)} new alert(s)")
    print(f"  {eng.summary()}")
    ok = eng.counts[alerts.AlertKind.FOUND] == 1
    print(f"  -> {'OK (one drone, one alert)' if ok else 'FAIL'}")
    return ok


def case_different_kinds_are_different() -> bool:
    print("\n--- a carrier where video was is a new alert ---")
    clock = Clock()
    eng = alerts.AlertEngine(clock=clock)
    eng.update(result([cand()]))
    got = eng.update(result([cand(kind=dsp.SignalKind.CARRIER)]))
    print(f"  video then carrier -> {len(got)} alert(s)")
    print(f"  {got[0].headline() if got else ''}")
    ok = len(got) == 1
    print(f"  -> {'OK' if ok else 'FAIL'}")
    return ok


def case_two_transmitters() -> bool:
    print("\n--- two different bands at once ---")
    clock = Clock()
    eng = alerts.AlertEngine(clock=clock)
    got = eng.update(result([cand(5_800_000_000), cand(5_658_000_000)]))
    print(f"  F4 and R1 in one sweep -> {len(got)} alert(s)")
    for e in got:
        print(f"    {e.headline()}")
    again = eng.update(result([cand(5_800_000_000), cand(5_658_000_000)]))
    print(f"  next sweep -> {len(again)} (both repeats)")
    ok = len(got) == 2 and len(again) == 0
    print(f"  -> {'OK' if ok else 'FAIL'}")
    return ok


def case_carrier_wording() -> bool:
    print("\n--- wording, since the operator reads this and acts on it ---")
    for kind in (dsp.SignalKind.VIDEO, dsp.SignalKind.CARRIER):
        eng = alerts.AlertEngine(clock=Clock())
        ev = eng.update(result([cand(kind=kind)]))[0]
        print(f"  {ev}")
    off = cand(5_802_000_000)
    off.off_chart = True
    eng = alerts.AlertEngine(clock=Clock())
    print(f"  off-chart: {eng.update(result([off]))[0]}")
    off2 = cand(5_747_000_000)
    off2.off_chart = True
    off2.match = None
    eng = alerts.AlertEngine(clock=Clock())
    print(f"  no chart match: {eng.update(result([off2]))[0]}")
    ok = True
    return ok


def case_beeper() -> bool:
    print("\n--- beeper rate limiting ---")
    b = alerts.Beeper(enabled=True, minimum_gap_s=10.0)
    made = b.beep()
    again = b.beep()
    print(f"  platform {sys.platform}: available={b.available}")
    print(f"  first beep -> {made}, immediate second -> {again} "
          f"(suppressed {b.suppressed})")
    ok = made is True and again is False and b.suppressed == 1
    print(f"  -> {'OK' if ok else 'FAIL'}")
    return ok


def case_listener_failure_is_contained() -> bool:
    print("\n--- a broken listener must not stop the others ---")
    eng = alerts.AlertEngine(clock=Clock())
    seen: list[str] = []

    def bad(_):
        raise RuntimeError("boom")

    eng.add_listener(bad)
    eng.add_listener(lambda e: seen.append(e.label))
    got = eng.update(result([cand()]))
    print(f"  alert still raised: {len(got)}; good listener saw "
          f"{len(seen)} event(s)")
    ok = len(got) == 1 and len(seen) == 1
    print(f"  -> {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    results = [
        case_repeat_is_not_an_alert(),
        case_cooldown_expires(),
        case_lost_and_returned(),
        case_incomplete_scan_is_not_evidence(),
        case_fine_frequency_shift_is_same(),
        case_different_kinds_are_different(),
        case_two_transmitters(),
        case_carrier_wording(),
        case_beeper(),
        case_listener_failure_is_contained(),
    ]
    print("\nRESULT:", "PASS" if all(results) else "FAIL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
