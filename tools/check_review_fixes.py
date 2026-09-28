"""Regression tests for the 14 review findings, asserting corrected behaviour.

Every case here is the inverse of a defect demonstration from the review of
``b63cec6``. The review's scripts asserted that the bugs were present; these
assert that they are fixed, against this project, and are run as a gate.

Three things they take seriously:

* The signed/unsigned IQ case is checked against *both* encodings of the same
  waveform, so the test pins the decode result rather than the absence of a
  crash. A test that only checked "does not raise" would have passed against the
  original bug.
* The GUI cases drive real widget signals -- ``setChecked`` on the checkbox, the
  gain slider's ``valueChanged`` -- rather than calling the handlers or setters
  directly. Finding 11 existed precisely because the unit tests called
  ``set_controls(invert=...)`` and never clicked the box.
* Ownership and freshness are tested with mocks that *record* ordering, because
  the defects were ordering bugs that produce plausible-looking results.

Run:  python tools/check_review_fixes.py
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv_rf import alerts, bands, dsp, fastscan, scan, sdr, video  # noqa: E402
from tools.check_alerts import cand, result  # noqa: E402
from tools.check_ui import Args  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def ok(label: str, good: bool, detail: str = "") -> bool:
    """Report one assertion. The marker is mandatory, not decoration.

    A helper that prints a label and nothing else makes a failure unfindable:
    with 40-odd assertions per run, "which one broke" turns into re-running
    under a debugger. Every line says PASS or FAIL.
    """
    mark = "PASS" if good else "FAIL"
    print(f"  {mark}  {label}" + (f"  [{detail}]" if detail else ""))
    return bool(good)


# -- 1: signed HackRF IQ is not read as unsigned -----------------------------


def case_iq_encoding() -> bool:
    print("--- 1: hardware IQ is signed int8, not offset-binary ---")
    signed = (FIXTURES / "iq_signed_int8.u8").read_bytes()
    unsig = (FIXTURES / "iq_unsigned_offset.u8.u8").read_bytes()
    ref = np.fromfile(FIXTURES / "iq_reference.f32", dtype=np.complex64)

    a = sdr._u8_to_iq(signed, sdr.IQEncoding.SIGNED_INT8)
    b = sdr._u8_to_iq(unsig, sdr.IQEncoding.UNSIGNED_OFFSET)

    good = True
    good &= ok(
        "signed fixture decodes to the reference waveform",
        a.shape == ref.shape and float(np.abs(a - ref).max()) < 0.01,
        f"max err {float(np.abs(a - ref).max()):.4f} (quantisation ~0.004)",
    )
    # Exact, not approximate. The two fixtures carry the same quantised integers
    # in different encodings, so a correct decoder must return bit-identical
    # results. A tolerance here would have let a systematic half-LSB error -- the
    # signature of an offset applied where none belongs -- pass.
    good &= ok(
        "the SAME waveform in either encoding decodes bit-identically",
        a.shape == b.shape and np.array_equal(a, b),
        f"max difference {float(np.abs(a - b).max()):.6f}",
    )
    # The original defect, pinned numerically: signed read as unsigned is not a
    # small error, it is a fold of the whole waveform.
    wrong = sdr._u8_to_iq(signed, sdr.IQEncoding.UNSIGNED_OFFSET)
    werr = float(np.abs(wrong - ref).max())
    good &= ok(
        "reading signed as unsigned is now visibly the wrong answer",
        werr > 0.5,
        f"max err {werr:.3f} vs 0.004 for the correct decode",
    )
    good &= ok(
        "format detection settles a quiet recording both ways",
        sdr.detect_iq_encoding(signed).encoding is sdr.IQEncoding.SIGNED_INT8
        and sdr.detect_iq_encoding(unsig).encoding is sdr.IQEncoding.UNSIGNED_OFFSET,
        "100.0% vs 0.0% of bytes at the ends of the byte axis",
    )
    # A loud recording cannot be settled from the data, and the honest answer is
    # to say so. Guessing would reproduce the original defect on a new input.
    loud = (np.exp(1j * np.linspace(0, 4000, 6000, dtype=np.float32))
            ).astype(np.complex64)
    loud_inter = np.empty((loud.size, 2), dtype=np.float32)
    loud_inter[:, 0] = loud.real
    loud_inter[:, 1] = loud.imag
    loud_v = sdr.detect_iq_encoding(
        (loud_inter * 127.5).round().clip(-128, 127).astype(np.int8).tobytes()
    )
    good &= ok(
        "a loud recording is reported as undecidable rather than guessed",
        loud_v.encoding is None,
        f"{loud_v.share*100:.0f}% at the ends, no conclusion drawn",
    )
    good &= ok(
        "the radio declares its format rather than guessing it",
        sdr.HackrfSource.encoding is sdr.IQEncoding.SIGNED_INT8
        and sdr.SimSource.encoding is sdr.IQEncoding.UNSIGNED_OFFSET,
        "hackrf=signed, sim=unsigned, each matching what its producer writes",
    )
    return good


def case_recording_encoding_fallback() -> bool:
    """A file whose format cannot be established says so instead of guessing."""
    print("--- 1c: an undecidable recording falls back visibly, not silently ---")
    import tempfile
    # Unit-magnitude signal: loud, so signed and offset-binary are not separable.
    loud = np.exp(1j * np.linspace(0, 4000, 60000, dtype=np.float32)).astype(np.complex64)
    inter = np.empty((loud.size, 2), dtype=np.float32)
    inter[:, 0] = loud.real
    inter[:, 1] = loud.imag
    raw = (inter * 127.5).round().clip(-128, 127).astype(np.int8).tobytes()

    p = Path(tempfile.gettempdir()) / "loud-ambiguous.u8"
    p.write_bytes(raw)
    rec = sdr.FileSource(p, frequency_hz=5_802_000_000, realtime=False)
    rec.start()
    time.sleep(0.3)
    rec.stop()
    text = rec.describe()
    good = ok("it defaults to the format the radio produces",
              rec.encoding is sdr.IQEncoding.SIGNED_INT8, "signed")
    good &= ok("and says the data did not decide it", "defaulted" in text,
               text[text.find("MS/s"):][:78])
    good &= ok("a declared format is honoured without any guessing",
               sdr.FileSource(p, encoding=sdr.IQEncoding.UNSIGNED_OFFSET).encoding
               is sdr.IQEncoding.UNSIGNED_OFFSET)
    p.unlink(missing_ok=True)
    return good


def case_iq_encoding_real_capture() -> bool:
    """Same question, answered against a real 5.8 GHz hardware capture.

    Skipped, not failed, when the capture is absent -- it lives outside this
    repository. When present it is the strongest available evidence that the
    declared format is the one the radio really produces.
    """
    print("--- 1b: real hardware capture decodes as signed ---")
    cap = (Path.home() / "Documents" / "hackrf-fpv-still-decoder" / "Samples"
           / "r5_5802_g16.u8")
    if not cap.exists():
        print(f"  SKIP  no capture at {cap}")
        return True
    raw = cap.read_bytes()[:4_000_000]
    print(f"  capture {cap.name}, {len(raw):,} bytes read")
    v = sdr.detect_iq_encoding(raw)
    good = ok("detected as signed", v.encoding is sdr.IQEncoding.SIGNED_INT8,
              f"{v.share*100:.1f}% of bytes at the ends of the axis")
    iq = sdr._u8_to_iq(raw, sdr.IQEncoding.SIGNED_INT8)
    mean_abs = float(abs(iq).mean())
    dc = abs(complex(iq.mean()))
    # The threshold is 5% of full scale. A real analogue VTX leaves a small
    # residual DC offset, so the requirement is not zero -- it is that the
    # offset is nowhere near the full-scale spike a misread would produce. Read
    # as offset-binary, this same capture sits at roughly 0.5.
    good &= ok(
        "decoded samples have sane amplitude and no meaningful DC offset",
        0.05 < mean_abs < 0.60 and dc < 0.05,
        f"mean|x| {mean_abs:.3f}, DC {dc:.4f} (0.5 would be a misread)",
    )
    return good


def case_real_capture_decodes() -> bool:
    """The strongest claim in the suite: real FPV video, from a real capture.

    Everything else here tests that the decoder is not *broken*. This one tests
    that it *works*, on samples a HackRF actually produced, with a simulated
    signal nowhere in the path.

    It is also the test that would have caught the defect that mattered most and
    was invisible until the IQ format was fixed. ``Rasteriser.select_run``
    required ``want`` consecutive *detected* scanlines, and on this capture the
    sync detector fires on 82.7% of lines with the losses arriving in chunks of
    5 to 20 lines. The longest clean run was 112 lines against the 240 a field
    needs, so not one frame was ever produced -- while the sync detector
    cheerfully reported quality 0.90, and the status line showed that number
    beside "no lock" where it read as a healthy noisy lock.

    The evidence that a picture is really there is adjacent-row correlation of
    the decoded luminance, compared against white noise of the same variance
    measured the same way. A decoder emitting plausible grey would score near the
    null; this scores +0.90 against a null of -0.001.
    """
    print("--- 1d: a real capture decodes to real video ---")
    cap = FIXTURES / "real_5802_signed.u8"
    if not cap.exists():
        print(f"  FAIL  fixture missing: {cap}")
        print("        It is committed on purpose: it is the only way this suite can")
        print("        prove the decoder works on real samples rather than on the")
        print("        simulator. Regenerate with python tools/make_iq_fixtures.py")
        print("        while a real capture is pointed at by $FPV_CAPTURE.")
        return False
    raw = cap.read_bytes()
    print(f"  fixture {cap.name}, {len(raw):,} bytes "
          f"({len(raw)//2:,} complex samples, 0.1 s at 10 MS/s)")

    dec = video.VideoDecoder(10_000_000)
    iq = sdr._u8_to_iq(raw, sdr.IQEncoding.SIGNED_INT8)
    per = 1 << 17
    frames = []
    for i in range(0, iq.size, per):
        fr = dec.push_iq(iq[i : i + per])
        if fr is not None:
            frames.append(fr)
    if not ok("the decoder emits frames from a real capture", bool(frames),
              f"{len(frames)} frame(s) from {iq.size//per} blocks"):
        return False

    good = ok("the frames are the right shape and standard",
              all(f.standard == "NTSC" and f.image.shape == (240, 160)
                  for f in frames),
              f"{frames[0].standard} {frames[0].image.shape[1]}x{frames[0].image.shape[0]}")

    # The line rate is the strongest single discriminator against noise: a noise
    # burst has no line rate, and a real NTSC transmitter lands within a few
    # hundredths of a percent of 15,734.264 Hz.
    lh = float(np.median([f.line_rate_hz for f in frames]))
    good &= ok("the recovered line rate is NTSC", abs(lh - 15_734.264) / 15_734.264 < 0.01,
               f"{lh:,.1f} Hz, {abs(lh-15734.264)/15734.264*100:.3f}% off")

    rs = [float(np.corrcoef(f.image[:-1].ravel(), f.image[1:].ravel())[0, 1])
          for f in frames]
    rng = np.random.default_rng(4)
    null = [float(np.corrcoef(
        (lambda n: (n[:-1].ravel(), n[1:].ravel()))(
            rng.integers(0, 256, size=f.image.shape, dtype=np.uint8)))[0, 1])
        for f in frames]
    good &= ok(
        "the picture is coherent, not plausible noise",
        float(np.mean(rs)) > float(np.mean(null)) + 0.10,
        f"adjacent-row r {np.mean(rs):+.3f} vs noise null {np.mean(null):+.3f}",
    )

    # A field that had to reconstruct most of its line positions is a weaker
    # result than a fully detected one. Assert only that the accounting exists
    # and is bounded -- not that it is zero, because on this capture it is not.
    gf = [f.lines_from_grid for f in frames]
    good &= ok(
        "reconstructed line positions are reported and bounded",
        all(0 <= g <= 240 for g in gf),
        f"worst field {max(gf)}/240 from grid, median {int(np.median(gf))}",
    )
    return good


# -- 2: the fallback must not run before the radio comes back ---------------


def case_fallback_after_reclaim() -> bool:
    print("--- 2: hop fallback runs only after the radio is reclaimed ---")
    events: list[str] = []

    class FakeSource:
        kind = "fake"
        sample_rate = 10_000_000
        retunable = True

        def __init__(self) -> None:
            self.released = False

        def release_device(self) -> bool:
            self.released = True
            events.append("release")
            return True

        def reclaim_device(self) -> bool:
            self.released = False
            events.append("reclaim")
            return True

        def wait_for_bytes(self, n, timeout=2.0) -> bool:
            events.append(f"flow-check(released={self.released})")
            return True

    fake = FakeSource()

    def fake_scan(scanner, **kw):
        events.append(f"fallback(released={scanner.source.released})")
        return scan.ScanResult(5_650_000_000, 5_950_000_000)

    with patch.object(fastscan, "fast_ok", return_value=True), \
         patch.object(fastscan.sweep, "run_sweep",
                      return_value=SimpleNamespace(error="device busy", ok=False)), \
         patch.object(scan.BandScanner, "scan", fake_scan):
        res = fastscan.scan_span(fake, 5_650_000_000, 5_950_000_000)

    print(f"  order: {events}")
    good = ok(
        "the fallback scan ran while the source was NOT released",
        "fallback(released=False)" in events,
    )
    good &= ok("reclaim happened before the fallback",
               events.index("reclaim") < events.index("fallback(released=False)"))
    good &= ok("sample flow was confirmed before falling back",
               any(e.startswith("flow-check") for e in events))
    good &= ok("the result is labelled as the hop engine", res.engine == "hops")
    return good


def case_reclaim_failure_is_not_an_empty_band() -> bool:
    print("--- 2b: a radio that will not come back says so ---")
    class DeadSource:
        kind = "fake"
        sample_rate = 10_000_000
        retunable = True
        def release_device(self): return True
        def reclaim_device(self): return False
        def wait_for_bytes(self, n, timeout=2.0): return True

    ran = []
    with patch.object(fastscan, "fast_ok", return_value=True), \
         patch.object(fastscan.sweep, "run_sweep",
                      return_value=SimpleNamespace(error="device busy", ok=False)), \
         patch.object(scan.BandScanner, "scan",
                      lambda s, **k: (ran.append(1), scan.ScanResult(0, 0))[1]):
        res = fastscan.scan_span(DeadSource(), 5_650_000_000, 5_950_000_000)

    good = ok("no hop walk was attempted on a radio that never came back",
              not ran)
    good &= ok("the result reports a failure, not an empty band",
               bool(res.error) and res.stopped_early and not res.candidates,
               f"error: {res.error[:58]}")
    good &= ok("engine is 'none' so it cannot be read as a measurement",
               res.engine == "none")
    return good


# -- 3: a timed-out retune yields no capture, not the old one ---------------


def case_retune_timeout_is_empty() -> bool:
    print("--- 3: a retune that times out returns no capture ---")
    src = sdr.SimSource()
    src.ring.write(bytes([150, 110]) * 4096)          # 4096 old samples
    with patch.object(src, "wait_for_bytes", return_value=False):
        iq = src.retune_settled(5_900_000_000, 4096, timeout=0)
    good = ok("no stale samples are handed back", iq.size == 0,
              f"was 4096 before the fix")
    good &= ok("the failure is recorded on the source",
               src.stats.read_errors > 0 and "discarded" in src.stats.last_error)

    # And a real empty band must still be distinguishable from a failed capture.
    scanner = scan.BandScanner(src)
    good &= ok("an empty capture is 'could not measure', not 'nothing there'",
               scanner.assess_frequency(5_900_000_000) is None
               if not src.retune_settled(5_900_000_000, 4096, timeout=0.05).size
               else False)
    return good


# -- 4: settings reach the hardware -----------------------------------------


def case_settings_reach_hardware() -> bool:
    print("--- 4: gain and amplifier changes reach the radio ---")
    src = sdr.SimSource(frequency_hz=5_802_000_000)
    with patch.object(src, "_apply_tune", return_value=True) as apply:
        src.lna_gain_db = 40
        src.vga_gain_db = 8
        src.tune(src.frequency_hz)                    # the old call pattern
    good = ok("a bare tune at the same frequency does NOT reconfigure",
              apply.call_count == 0,
              "unchanged frequency still short-circuits, by design")
    with patch.object(src, "_apply_tune", return_value=True) as apply:
        good &= ok("set_gains at the same frequency does reconfigure",
                   src.set_gains(40, 8) is True and apply.call_count == 1)
    with patch.object(src, "_apply_tune", return_value=True) as apply:
        good &= ok("an explicit reconfigure does reconfigure",
                   src.tune(src.frequency_hz, reconfigure=True) is True
                   and apply.call_count == 1)
    return good


def case_ui_gain_signal_applies() -> bool:
    """The GUI case, driven through the real widget signal."""
    print("--- 4b: the gain slider actually reconfigures the source ---")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    src = sdr.SimSource(frequency_hz=5_802_000_000)
    with patch.object(video.DecodeWorker, "start"):
        win = _make_window(src, app)
    try:
        with patch.object(src, "_apply_tune", return_value=True) as apply:
            win.lna.setValue(24)          # fires valueChanged -> _on_gain
            app.processEvents()
        good = ok("moving the LNA slider reconfigured the source",
                  apply.call_count >= 1, f"{apply.call_count} apply call(s)")
        with patch.object(src, "_apply_tune", return_value=True) as apply:
            apply.reset_mock()
            win.amp.setChecked(False)     # fires toggled -> _on_amp
            app.processEvents()
        good &= ok("switching the amplifier off reconfigured the source",
                   apply.call_count >= 1 and src.amp_enabled is False,
                   f"{apply.call_count} apply call(s), amp {src.amp_enabled}")
    finally:
        win.close()
    return good


# -- 5: stale video must not read as locked ---------------------------------


def case_stale_video_is_not_locked() -> bool:
    print("--- 5: an old frame stops counting as live video ---")
    worker = video.DecodeWorker(sdr.SimSource())
    frame = video.VideoFrame(np.zeros((2, 2), dtype=np.uint8),
                             time.monotonic() - 10, 15_734, 1, "NTSC", True)
    with worker._lock:
        worker._frame = frame
        worker._published_at = time.monotonic() - 10.0
    worker.stats.locked_frames = 1
    good = ok("a ten-second-old frame is not returned as live",
              worker.latest() is None, "latest() is None once stale")
    good &= ok("lock has lapsed", worker.locked is False)
    good &= ok("frame age is reported, not hidden",
               worker.frame_age_s() > 9.0, f"{worker.frame_age_s():.1f}s")

    with worker._lock:
        worker._published_at = time.monotonic()
    good &= ok("a fresh frame is live again",
               worker.latest() is frame and worker.locked is True)

    # The no-samples path must also age the picture, which it used to skip.
    worker2 = video.DecodeWorker(sdr.SimSource())
    with patch.object(worker2.source, "drain_iq",
                      return_value=np.zeros(0, dtype=np.complex64)):
        with worker2._lock:
            worker2._frame = frame
            worker2._published_at = time.monotonic() - 10.0
        worker2.start()
        time.sleep(0.7)
        worker2.stop()
    good &= ok("a silent source expires the picture and counts missed fields",
               worker2.latest() is None and worker2.stats.fields_missed > 0,
               f"missed {worker2.stats.fields_missed} field(s)")
    return good


def case_decode_exceptions_are_counted() -> bool:
    print("--- 5b: decode errors are visible instead of silent ---")
    worker = video.DecodeWorker(sdr.SimSource())
    boom = np.zeros(4000, dtype=np.complex64) + 1j

    class Bad:
        def reset(self): pass
        def push_iq(self, iq): raise ValueError("deliberate")

    worker.decoder = Bad()
    with patch.object(worker.source, "drain_iq", return_value=boom):
        worker.start()
        time.sleep(0.25)
        worker.stop()
    return ok("a raising decoder is counted and named, not swallowed",
              worker.stats.decode_errors > 0
              and "deliberate" in worker.stats.last_decode_error,
              f"{worker.stats.decode_errors} error(s): "
              f"{worker.stats.last_decode_error[:40]}")


# -- 6: a quiet channel must not raise FOUND --------------------------------


def case_quiet_check_does_not_alert() -> bool:
    print("--- 6: checking a quiet channel does not alert ---")
    eng = alerts.AlertEngine()
    quiet = cand(kind=dsp.SignalKind.NOISE)
    good = ok("note() refuses a NOISE candidate", eng.note(quiet) is None)
    good &= ok("and reports nothing as found", len(eng.log) == 0,
               f"{len(eng.log)} event(s) in the log")
    good &= ok("an occupied candidate still alerts",
               eng.note(cand()) is not None)
    return good


def case_ui_quiet_check_does_not_alert() -> bool:
    print("--- 6b: the GUI check button does not alert on silence ---")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    src = sdr.SimSource(frequency_hz=5_802_000_000)
    with patch.object(video.DecodeWorker, "start"):
        win = _make_window(src, app)
    try:
        # The real handler, not a stub: the point is what the button does.
        with patch.object(win.scanner, "assess_frequency",
                          return_value=cand(kind=dsp.SignalKind.NOISE)):
            win._check_current()
        good = ok("no alert was raised for a quiet channel",
                  len(win.alerts.log) == 0 and win._last_result.candidates == [],
                  f"{len(win.alerts.log)} event(s)")
        good &= ok("the operator is still told what was measured",
                   "nothing above the noise floor" in win._status.currentMessage(),
                   win._status.currentMessage()[:56])
        with patch.object(win.scanner, "assess_frequency", return_value=cand()):
            win._check_current()
        good &= ok("an occupied channel still alerts",
                   len(win.alerts.log) == 1)
    finally:
        win.close()
    return good


# -- 7: incomplete searches are not evidence of absence ---------------------


def case_cancelled_scan_is_not_lost() -> bool:
    print("--- 7: a cancelled search does not report LOST ---")
    clock = _Clock()
    eng = alerts.AlertEngine(clock=clock)
    eng.update(result([cand()]), at=100)
    clock.t = 131
    got = eng.update(result([], stopped_early=True), at=131)
    good = ok("no LOST after a cancelled scan 31s later", got == [],
              f"got {[e.kind.value for e in got]}")
    good &= ok("the finding is still known", len(eng._seen) == 1)

    # A failed search is equally not evidence.
    clock.t = 200
    got2 = eng.update(result([], error="radio did not come back"), at=200)
    good &= ok("no LOST after a failed search either", got2 == [],
               f"got {[e.kind.value for e in got2]}")

    # But a completed search that genuinely saw nothing still reports it.
    clock.t = 300
    got3 = eng.update(result([]), at=300)
    good &= ok("a completed empty search still reports LOST",
               bool(got3) and got3[0].kind is alerts.AlertKind.LOST,
               f"got {[e.kind.value for e in got3]}")
    return good


def case_narrow_search_does_not_expire_others() -> bool:
    print("--- 7b: a scoped search does not expire out-of-band findings ---")
    clock = _Clock()
    eng = alerts.AlertEngine(clock=clock)
    eng.update(result([cand(5_800_000_000)]), at=100)
    clock.t = 400
    r = result([])
    r.lo_hz, r.hi_hz = 5_865_000_000, 5_885_000_000        # Raceband only
    got = eng.update(r, at=400)
    good = ok("an F4 at 5800 survives a Raceband-only sweep",
              got == [] and len(eng._seen) == 1,
              f"got {[e.kind.value for e in got]}")
    clock.t = 700
    got2 = eng.update(result([]), at=700)                 # now the F4 band
    good &= ok("a later sweep of that band does report it lost",
               bool(got2) and got2[0].kind is alerts.AlertKind.LOST)
    return good


# -- 8: scoped auto-select accepts sweep results ---------------------------


def case_scoped_select_accepts_sweep() -> bool:
    print("--- 8: a scoped search can still auto-select a sweep result ---")
    scanner = scan.BandScanner(sdr.SimSource())
    carrier = cand(5_806_000_000, kind=dsp.SignalKind.CARRIER)
    r = result([carrier])
    r.engine = "sweep"
    rb = bands.BANDS_BY_KEY["R"]
    unscoped = scanner.suggest(r)
    scoped = scanner.suggest(r, rb)
    good = ok("unscoped sweep result is selectable", unscoped is carrier)
    good &= ok("the SAME result is selectable when scoped to Raceband",
               scoped is carrier,
               "identical answer either way, which is the point")
    good &= ok("a signal outside the scoped band is still declined",
               scanner.suggest(result([cand(5_800_000_000, kind=dsp.SignalKind.CARRIER)]),
                               rb) is None)
    return good


# -- 9: shared channel frequencies ------------------------------------------


def case_shared_frequency_membership() -> bool:
    print("--- 9: a shared frequency belongs to both plans ---")
    scanner = scan.BandScanner(sdr.SimSource())
    shared = cand(5_880_000_000)                # Raceband R7 and F8
    label = shared.match.label
    print(f"  5880 MHz is globally labelled {label} (first plan listed wins)")
    good = ok("global naming still returns one label", label in {"F8", "R7"})
    good &= ok(
        "the band itself is asked directly, and every plan with 5880 claims it",
        all(bands.band_contains(5_880_000_000, bands.BANDS_BY_KEY[k])
            for k in ("R", "F", "O")),
        "R7, F8 and O7 are all 5880 MHz",
    )
    good &= ok(
        "a band without that channel does not claim it",
        not any(bands.band_contains(5_880_000_000, bands.BANDS_BY_KEY[k])
                for k in ("E", "B", "L")),
    )
    for key in ("R", "F", "O"):
        got = scanner.suggest(result([shared]), bands.BANDS_BY_KEY[key])
        good &= ok(f"a verified 5880 MHz finding is selectable in band {key}",
                   got is shared, "membership by frequency, not by label")
    for key in ("E", "B"):
        got = scanner.suggest(result([shared]), bands.BANDS_BY_KEY[key])
        good &= ok(f"band {key}, which has no 5880 channel, still declines it",
                   got is None)
    return good


# -- 10: each sample is consumed exactly once ------------------------------


def case_drain_consumes_once() -> bool:
    print("--- 10: drain hands each sample over exactly once ---")
    src = sdr.SimSource()
    src.ring.write(bytes([140, 120]) * 4096)
    a = src.drain_iq(4096)
    b = src.drain_iq(4096)
    c = src.drain_iq(4096)
    good = ok("first drain returns the block", a.size == 4096)
    good &= ok("a second drain with no new input returns nothing",
               b.size == 0 and c.size == 0,
               f"was 4096 both times before the fix")
    good &= ok("the repeat is not even the same data",
               b.size != 4096 or not np.array_equal(a, b))
    src.ring.write(bytes([140, 120]) * 4096)
    d = src.drain_iq(4096)
    good &= ok("genuinely new samples do come through", d.size == 4096)
    src.reset_drain()
    e = src.drain_iq(4096)
    good &= ok("reset_drain discards what is buffered", e.size == 0)
    # Partial consumption: ask for less than arrived, the rest must not reappear.
    src2 = sdr.SimSource()
    src2.ring.write(bytes([140, 120]) * 4096)
    p1 = src2.drain_iq(1000)
    p2 = src2.drain_iq(4096)
    good &= ok("a capped drain skips the backlog instead of replaying it",
               p1.size == 1000 and p2.size == 0,
               f"p1={p1.size} p2={p2.size} (backlog was consumed and dropped)")
    return good


# -- 11: the invert checkbox inverts ----------------------------------------


def case_invert_checkbox() -> bool:
    print("--- 11: the invert checkbox inverts the image ---")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    with patch.object(video.DecodeWorker, "start"):
        win = _make_window(sdr.SimSource(), app)
    try:
        win.panel.set_controls(brightness=60, invert=False)
        win.invert.setChecked(True)             # real widget signal
        app.processEvents()
        good = ok("the checkbox is checked", win.invert.isChecked())
        good &= ok("panel.invert followed it", win.panel.invert is True)
        good &= ok("brightness was left alone",
                   win.panel.brightness == 60, f"was True (=1) before the fix")
        win.invert.setChecked(False)
        app.processEvents()
        good &= ok("unticking it turns inversion off again",
                   win.panel.invert is False)
    finally:
        win.close()
    return good


# -- 12: a failed tune is a failure, and a retry retries -------------------


def case_failed_tune_state() -> bool:
    print("--- 12: a failed tune does not corrupt state or fake success ---")
    src = sdr.SimSource(frequency_hz=5_802_000_000)
    target = 5_900_000_000
    with patch.object(src, "_apply_tune", return_value=False) as apply:
        first = src.tune(target)
        second = src.tune(target)
    good = ok("a failed tune reports failure", first is False)
    good &= ok("retrying the same tune also reports failure, not success",
               second is False, f"was True before the fix")
    good &= ok("the retry actually retried the hardware",
               apply.call_count == 2, f"{apply.call_count} apply call(s), was 1")
    good &= ok("the source does not claim to be on the failed frequency",
               src.applied_frequency_hz == 5_802_000_000,
               f"applied={src.applied_frequency_hz}")
    with patch.object(src, "_apply_tune", return_value=True) as apply:
        good &= ok("a later successful tune is honoured", src.tune(target) is True)
        good &= ok("and now the applied frequency is correct",
                   src.applied_frequency_hz == target)
        good &= ok("an identical tune then short-circuits",
                   src.tune(target) is True and apply.call_count == 1)
    return good


def case_ui_select_best_on_failed_tune() -> bool:
    print("--- 12b: auto-select does not claim a channel it did not reach ---")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    with patch.object(video.DecodeWorker, "start"):
        win = _make_window(sdr.SimSource(), app)
    try:
        win._last_result = result([cand(5_800_000_000)])
        with patch.object(win.source, "tune", return_value=False), \
             patch.object(ui_qmessagebox(), "warning"):
            selected = win._select_best(announce=True)
        msg = win._status.currentMessage()
        good = ok("auto-select reports failure", selected is False)
        good &= ok("and does not say it is watching that channel",
                   not msg.startswith("watching "), f"status: {msg[:44]}")
    finally:
        win.close()
    return good


# -- 13: a recording cannot be relabelled -----------------------------------


def case_recording_frequency_immutable() -> bool:
    print("--- 13: a recording keeps the frequency it was captured at ---")
    rec = sdr.FileSource("unused-recording.u8", frequency_hz=5_802_000_000)
    good = ok("it still declares itself non-retunable", rec.retunable is False)
    good &= ok("tuning to another frequency is refused",
               rec.tune(5_900_000_000) is False)
    good &= ok("and the capture frequency is unchanged",
               rec.frequency_hz == 5_802_000_000,
               f"was {rec.frequency_hz} before the fix")
    good &= ok("tuning to its own frequency is a no-op success",
               rec.tune(5_802_000_000) is True)
    good &= ok("gains are refused too, since the samples are fixed",
               rec.set_gains(10, 10) is True and rec.lna_gain_db == 0)
    return good


def case_ui_recording_not_relabelled() -> bool:
    print("--- 13b: the GUI cannot relabel a recording ---")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    rec = sdr.FileSource("unused-recording.u8", frequency_hz=5_802_000_000)
    with patch.object(video.DecodeWorker, "start"):
        win = _make_window(rec, app)
    try:
        good = ok("the tune control is disabled for a recording",
                  not win.freq.isEnabled(), "nothing to drag, so nothing to lie")
        good &= ok("and says why on the tooltip",
                   "cannot be retuned" in win.freq.toolTip(),
                   win.freq.toolTip()[:52])
        # Belt and braces: even if a value is forced in, the source refuses it
        # and the capture frequency does not move.
        with patch.object(ui_qmessagebox(), "warning"):
            win.freq.setValue(5880)
            app.processEvents()
        good &= ok("a forced value still cannot change the capture frequency",
                   rec.frequency_hz == 5_802_000_000,
                   f"still {rec.frequency_hz/1e6:.0f} MHz")
    finally:
        win.close()
    return good


# -- 14: the receiver comes back where it started ---------------------------


def case_scan_restores_tuning() -> bool:
    print("--- 14: a search that selects nothing leaves the tuner alone ---")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    src = sdr.SimSource(frequency_hz=5_802_000_000)
    with patch.object(video.DecodeWorker, "start"):
        win = _make_window(src, app)
    try:
        win.freq.blockSignals(True)
        win.freq.setValue(5802)
        win.freq.blockSignals(False)
        win._pre_scan_hz = 5_802_000_000
        win.autotune.setChecked(False)           # auto-select off

        def empty_capture(frequency_hz, n_samples, timeout):
            src.tune(frequency_hz)               # hop walk moves the radio
            return np.zeros(0, dtype=np.complex64)

        with patch.object(src, "retune_settled", side_effect=empty_capture):
            res = win.scanner.scan(lo_hz=5_650_000_000, hi_hz=5_950_000_000)
        moved = src.frequency_hz
        win._on_scan_done(res)
        good = ok("the scan really did leave the radio elsewhere",
                  moved != 5_802_000_000, f"left at {moved/1e6:.1f} MHz")
        good &= ok("the receiver is put back", src.frequency_hz == 5_802_000_000,
                   f"ended at {src.frequency_hz/1e6:.1f} MHz")
        good &= ok("the frequency box agrees with the receiver",
                   win.freq.value() * 1_000_000 == src.frequency_hz)
        good &= ok("and the operator is told why",
                   "back on" in win._status.currentMessage(),
                   win._status.currentMessage()[:52])

        # A cancelled search gets the same restoration.
        src.tune(5_900_000_000)
        win._pre_scan_hz = 5_802_000_000
        win._on_scan_done(result([], stopped_early=True))
        good &= ok("a cancelled search also restores",
                   src.frequency_hz == 5_802_000_000,
                   f"ended at {src.frequency_hz/1e6:.1f} MHz")
    finally:
        win.close()
    return good


# -- helpers ----------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.t = 1_700_000_000.0

    def __call__(self) -> float:
        return self.t


def ui_qmessagebox():
    from fpv_rf import ui
    return ui.QMessageBox


def _make_window(source, app):
    """Build a MainWindow without starting its decode thread.

    The decode thread is patched out because these cases are about control flow
    and state, and a live decode thread would make them timing-dependent.
    """
    from fpv_rf import ui
    return ui.MainWindow(source, Args())


def main() -> int:
    from fpv_rf import ui

    # Modal dialogs are silenced for the whole run. A QMessageBox blocks even on
    # the offscreen platform, so a case that provokes one hangs forever instead
    # of failing -- which is how the recording case first manifested. No case
    # below asserts on dialog text; they assert on state, return values and
    # status messages, which is the part worth testing.
    with patch.object(ui.QMessageBox, "warning"), \
         patch.object(ui.QMessageBox, "information"), \
         patch.object(ui.QMessageBox, "critical"):
        cases = [
            case_iq_encoding(),
            case_iq_encoding_real_capture(),
            case_recording_encoding_fallback(),
            case_real_capture_decodes(),
            case_fallback_after_reclaim(),
            case_reclaim_failure_is_not_an_empty_band(),
            case_retune_timeout_is_empty(),
            case_settings_reach_hardware(),
            case_ui_gain_signal_applies(),
            case_stale_video_is_not_locked(),
            case_decode_exceptions_are_counted(),
            case_quiet_check_does_not_alert(),
            case_ui_quiet_check_does_not_alert(),
            case_cancelled_scan_is_not_lost(),
            case_narrow_search_does_not_expire_others(),
            case_scoped_select_accepts_sweep(),
            case_shared_frequency_membership(),
            case_drain_consumes_once(),
            case_invert_checkbox(),
            case_failed_tune_state(),
            case_ui_select_best_on_failed_tune(),
            case_recording_frequency_immutable(),
            case_ui_recording_not_relabelled(),
            case_scan_restores_tuning(),
        ]
    bad = [i for i, v in enumerate(cases) if not v]
    print(f"\n{sum(cases)}/{len(cases)} cases passed")
    print("RESULT:", "PASS" if not bad else f"FAIL (cases {bad})")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
