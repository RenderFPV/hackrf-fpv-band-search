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

Cases 16 and 17 are from a later finding rather than that review: the frozen
build. A capability probe launched a console, which cost 0.66 s, and it ran
ahead of the first reception, so ``--check``'s fixed half-second sleep read a
ring nothing had written to yet and reported a working HackRF as one producing
nothing. Both cases pin the fix and everything around it that a live run would
not notice if it regressed.

Case 18 is the same finding read a second time, after 17 had already replaced the
fixed half-second sleep with the bounded wait: the new wait widened a window
that had been wrongly shaped all along. ``start()`` and the wait sat outside the
``try`` whose ``finally`` stops the source, so Ctrl-C during the wait leaked the
receiver -- which is a worse outcome than the bug 17 fixed, because the operator
is told nothing at all.

Run:  python tools/check_review_fixes.py
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import threading
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


# -- 15: a retune must not starve its own wait ------------------------------


class _FakePipe:
    """A stdout stand-in for a transfer process, with no radio behind it.

    Hands over its payload in ``readinto`` calls, then EOF. A silent pipe --
    one given a ``silence`` event -- blocks in ``readinto`` on that event, the
    way a live process with nothing to say does, and only comes back when the
    event is set.

    It must block rather than give up on a short read timeout, because
    :meth:`HackrfSource._run` reads *any* zero-length read as EOF: it closes
    the pipe, kills the process it was reading and clears ``_proc``. A pipe
    that timed out and answered 0 would therefore have the reader tear down its
    own fake receiver, and a test written against that arrangement would be
    measuring a *missing* receiver while claiming to measure a silent one --
    passing on the EOF path and never reaching the behaviour it is about. The
    flip side is that whoever sets up a silence owns releasing it: the event
    has to be set, as the cleanup of the case that installed the pipe does,
    or the reader stays parked on it.
    """

    def __init__(self, payload: bytes = b"", *, silence: threading.Event | None = None) -> None:
        self._payload = payload
        self._silence = silence
        self.closed = False
        self.reads = 0

    def readinto(self, view) -> int:
        self.reads += 1
        if self._silence is not None:
            self._silence.wait()            # alive, and saying nothing
            return 0                        # ...until told to shut up
        if self.closed or not self._payload:
            return 0
        n = min(len(view), len(self._payload))
        view[:n] = self._payload[:n]
        self._payload = self._payload[n:]
        return n

    def close(self) -> None:
        self.closed = True


class _FakeProc:
    """Just enough ``subprocess.Popen`` for :meth:`HackrfSource._run`."""

    def __init__(self, stdout, stderr=None) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


class _LockHolder:
    """Stands in for ``retune_settled``, holding ``_tune_lock`` from another thread.

    ``retune_settled`` holds that lock across its wait for samples, and it is
    the reader that has to produce those samples -- so a reader that queues up
    behind the lock starves the very wait that is holding it. That is the
    arrangement reproduced here, with a watchdog release so a regression fails
    its assertions instead of hanging the gate.
    """

    def __init__(self, lock, hold_s: float) -> None:
        self._lock = lock
        self._hold_s = hold_s
        self._go = threading.Event()
        self._held = threading.Event()
        self._thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self) -> None:
        self._lock.acquire()
        self._held.set()
        self._go.wait(self._hold_s)
        self._lock.release()

    def start(self) -> None:
        self._thread.start()
        self._held.wait(1.0)

    def free(self) -> None:
        self._go.set()
        self._thread.join(timeout=2.0)


def _reader_source() -> "sdr.HackrfSource":
    """A HackrfSource wired to run the real ``_run``, minus the radio."""
    src = sdr.HackrfSource(frequency_hz=5_802_000_000, executable="hackrf_transfer")
    src._stop.clear()
    return src


def _start_reader(src) -> threading.Thread:
    """Run the real sampling loop against fake pipes."""
    th = threading.Thread(target=src._run, name="iq-hackrf-test", daemon=True)
    th.start()
    return th


def _stop_reader(src, th) -> None:
    src._stop.set()
    th.join(timeout=2.0)


def _wait_ring(src, n_bytes: int, timeout: float) -> float:
    """Seconds until the ring holds ``n_bytes``, or -1.0 if it never did."""
    t0 = time.monotonic()
    deadline = t0 + timeout
    while time.monotonic() < deadline:
        if src.ring.total_written >= n_bytes:
            return time.monotonic() - t0
        time.sleep(0.002)
    return -1.0 if src.ring.total_written < n_bytes else time.monotonic() - t0


def case_respawn_does_not_starve_retune() -> bool:
    print("--- 15: the reader's respawn does not starve the retune that caused it ---")
    good = True

    # 15a: the real defect. _apply_tune() detaches the old process and publishes
    # None before spawning its replacement, so the reader sees exactly this state
    # on every retune. It used to block on _tune_lock here -- held by the very
    # retune now waiting, under that lock, for samples only this thread can
    # produce.
    src = _reader_source()
    spawns = []

    def counted_spawn() -> bool:
        spawns.append(time.monotonic())
        return True

    src._spawn_settled = counted_spawn
    holder = _LockHolder(src._tune_lock, hold_s=4.0)
    holder.start()
    th = _start_reader(src)
    try:
        time.sleep(0.05)                       # let it reach the lock attempt
        payload = bytes([100, 120]) * 4096     # 8192 bytes: one complete block
        src._proc = _FakeProc(_FakePipe(payload))
        dt = _wait_ring(src, len(payload), timeout=1.0)
        good &= ok(
            "a fresh capture completes while the retune still holds the lock",
            dt >= 0.0,
            f"{len(payload)} bytes in {dt*1000:.0f} ms" if dt >= 0
            else f"only {src.ring.total_written} of {len(payload)} bytes",
        )
        good &= ok("the wait was not the retune's 2 s timeout",
                   0.0 <= dt < 1.0, f"{dt:.3f} s")
        good &= ok("and it happened before the lock came free",
                   src._tune_lock.locked(), "retune still inside its transaction")
        good &= ok("no second transfer process was started behind the retune's",
                   not spawns, f"{len(spawns)} spawn attempt(s)")
    finally:
        _stop_reader(src, th)
        holder.free()

    # The reader exited the loop rather than spinning on the contended lock.
    good &= ok("stopping ends the reader promptly even under contention",
               not th.is_alive(), "thread joined inside 2 s")

    # 15b: the replacement is alive but saying nothing. The lock is free, so this
    # is the plain honest path -- but it must not spawn over the top of a
    # process that is already there, and the capture must time out rather than
    # hand back whatever the ring happens to hold.
    src2 = _reader_source()
    spawns2 = []
    src2._spawn_settled = lambda: (spawns2.append(1), True)[1]
    silence = threading.Event()
    pipe2 = _FakePipe(silence=silence)
    proc2 = _FakeProc(pipe2)
    src2._proc = proc2
    th2 = _start_reader(src2)
    try:
        block = bytes([130, 110]) * 2048      # 4096 bytes from the *old* tuning
        src2.ring.write(block, 0)
        iq = src2.retune_settled(5_900_000_000, 16384, timeout=0.5)
        good &= ok("a replacement that is alive but silent yields an empty capture",
                   iq.size == 0, f"{iq.size} samples")
        good &= ok("and the failure is recorded as a discarded measurement",
                   "discarded" in src2.stats.last_error,
                   src2.stats.last_error[-52:])
        good &= ok("the reader did not spawn a second receiver over it",
                   not spawns2, f"{len(spawns2)} spawn(s)")
        # The empty capture above is only evidence of silence if there was
        # something there to be silent. _run treats a zero-length read as EOF:
        # it closes the pipe, kills the process and clears _proc, so a receiver
        # that had gone away would produce the same three answers. Asserted
        # after the capture timeout, while the pipe is still blocked, that the
        # same fake process is the current one and is still running.
        good &= ok("the silent receiver is still the process the reader has",
                   src2._proc is proc2,
                   "same fake process" if src2._proc is proc2
                   else f"replaced by {src2._proc!r}")
        good &= ok("it was never killed for its silence",
                   proc2.poll() is None and not proc2.terminated,
                   f"poll={proc2.poll()} terminated={proc2.terminated}")
        good &= ok("and its pipe is still open, with the read in flight",
                   not pipe2.closed and pipe2.reads >= 1,
                   f"closed={pipe2.closed} reads={pipe2.reads}")
    finally:
        silence.set()                        # release the blocked reader first
        _stop_reader(src2, th2)

    # 15c: _released set before the lock could be taken. The check and the spawn
    # have to stay atomic -- so the reader must wait, take the lock, and then
    # decline, rather than opening the radio back up in the gap.
    src3 = _reader_source()
    spawns3 = []
    src3._spawn_settled = lambda: (spawns3.append(1), True)[1]
    holder3 = _LockHolder(src3._tune_lock, hold_s=4.0)
    holder3.start()
    th3 = _start_reader(src3)
    try:
        time.sleep(0.05)
        src3._released = True                 # before the lock can be acquired
        holder3.free()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline and th3.is_alive() and not spawns3:
            time.sleep(0.01)
        good &= ok("a released radio is not reopened when the lock frees up",
                   not spawns3 and src3._proc is None,
                   f"{len(spawns3)} spawn(s)")
    finally:
        _stop_reader(src3, th3)
        holder3.free()

    # 15d: stop while the lock is held. The contended path must not turn that
    # into a wait for a lock nobody is going to release.
    src4 = _reader_source()
    holder4 = _LockHolder(src4._tune_lock, hold_s=4.0)
    holder4.start()
    th4 = _start_reader(src4)
    time.sleep(0.05)
    t0 = time.monotonic()
    _stop_reader(src4, th4)
    dt_stop = time.monotonic() - t0
    good &= ok("stop is bounded while the lock stays contended",
               not th4.is_alive() and dt_stop < 1.0,
               f"exited in {dt_stop*1000:.0f} ms")
    holder4.free()
    return good


# -- 16: the capability probe must not open a console ------------------------


def case_probe_launches_without_a_console() -> bool:
    """``hackrf_transfer -h`` runs before the first sample, so its cost *is* startup.

    The probe was a plain ``subprocess.run``. Under console python that is
    cheap, but the packaged app is a *windowed* exe, and a console-subsystem
    child launched from one has to bring a console up with it: measured here at
    0.66 s, against 0.024 s under console python and 0.030 s with
    ``CREATE_NO_WINDOW``. ``_cmdline`` asks for the flag set while it is
    building the *receive* command line, so that 0.66 s lands squarely in front
    of the first reception -- which is how a healthy HackRF came to be reported
    as a radio that produced nothing.

    Everything else about the probe is pinned here too, because the fix is a
    single keyword in a call that also carries the arguments, the output
    capture, the text mode, the timeout and the cache. A probe that quietly
    stopped caching, or stopped reading stderr, or lost its fail-safe, would be
    a worse defect than the slow one and no live run would reveal it.
    """
    import subprocess

    from fpv_rf import sdr

    HELP = (
        "usage: hackrf_transfer\n"
        "  [-r <filename>]        receive IQ to a file\n"
        "  [-f <frequency>]       frequency\n"
        "  [-a <amp_enable>]      RF amplifier\n"
        "  use '-' for stdout\n"
    )
    calls: list[dict] = []

    def help_run(args, **kwargs):
        calls.append({"args": args, **kwargs})
        return SimpleNamespace(stdout=HELP, stderr="")

    def stderr_run(args, **kwargs):
        calls.append({"args": args, **kwargs})
        return SimpleNamespace(stdout="", stderr="  [-i]\n")

    sdr.supported_flags.cache_clear()
    try:
        with patch.object(sdr.subprocess, "run", help_run):
            flags = sdr.supported_flags("fake-hackrf.exe")
        good = ok("the option set is still parsed out of the help text",
                  {"r", "f", "a"} <= set(flags),
                  f"parsed {''.join(sorted(flags))}")
        good &= ok("the helper is probed exactly once", len(calls) == 1,
                   f"{len(calls)} call(s)")
        if calls:
            kw = calls[0]
            good &= ok("it still asks the helper for its help",
                       kw["args"] == ["fake-hackrf.exe", "-h"],
                       f"args {kw['args']}")
            good &= ok("and still captures its output as text",
                       kw.get("capture_output") is True and kw.get("text") is True,
                       f"capture_output={kw.get('capture_output')} "
                       f"text={kw.get('text')}")
            good &= ok("and is still bounded by a timeout",
                       kw.get("timeout") == 10, f"timeout={kw.get('timeout')}")
            # The fix itself. Asserted against the real constant rather than a
            # literal, so this cannot quietly pass on a platform where the
            # flag does not exist and the fallback is what is being checked.
            good &= ok("the launch does not bring a console window with it",
                       kw.get("creationflags")
                       == getattr(subprocess, "CREATE_NO_WINDOW", 0),
                       f"creationflags={kw.get('creationflags')!r} vs "
                       f"CREATE_NO_WINDOW="
                       f"{getattr(subprocess, 'CREATE_NO_WINDOW', 0)}")

        # Cached per executable: this is paid once per process, and the answer
        # is what makes a retune cheap.
        n_before = len(calls)
        with patch.object(sdr.subprocess, "run", help_run):
            again = sdr.supported_flags("fake-hackrf.exe")
        good &= ok("the answer is cached, so a retune pays nothing",
                   again == flags and len(calls) == n_before,
                   f"{len(calls) - n_before} extra call(s)")

        # This build prints its options on stderr instead of stdout.
        sdr.supported_flags.cache_clear()
        with patch.object(sdr.subprocess, "run", stderr_run):
            good &= ok("options printed on stderr are read too",
                       "i" in sdr.supported_flags("stderr-hackrf.exe"))

        # And a probe that cannot answer still means "pass nothing optional"
        # rather than raising: every flag guarded here is a refinement, and the
        # direction that loses the radio entirely is passing one.
        sdr.supported_flags.cache_clear()
        with patch.object(sdr.subprocess, "run",
                          side_effect=subprocess.TimeoutExpired("cmd", 10)):
            good &= ok("a probe that hangs yields the safe empty set",
                       sdr.supported_flags("hanging-hackrf.exe") == frozenset(),
                       "no optional flags passed")
    finally:
        sdr.supported_flags.cache_clear()
    return good


# -- 17: --check waits for the radio rather than guessing at its timing ------


class _DelayedSource(sdr.IQSource):
    """A source that takes its time, which is what a real radio opening does.

    ``start()`` returns at once and the thread then does its work, so a check
    that reads the ring at a fixed moment after ``start`` sees nothing. Three
    shapes, because the check has to tell them apart:

    * ``write=True`` -- samples arrive, late. A working radio.
    * ``alive=True, write=False`` -- the transfer process is up and saying
      nothing. Alive is *not* proof of IQ, and this is the state the old fixed
      sleep used to mistake for a dead one.
    * ``error=...`` -- the source gave up at once and said why.
    """

    kind = "delayed"

    def __init__(
        self,
        delay: float = 0.0,
        *,
        write: bool = True,
        n_bytes: int = 2 << 20,
        alive: bool = True,
        error: str = "",
    ) -> None:
        super().__init__(10_000_000)
        self.frequency_hz = 5_802_000_000
        self._applied_hz = self.frequency_hz
        self.delay = float(delay)
        self.write = write
        self.n_bytes = int(n_bytes)
        self.alive = alive
        self.error = error
        self.stopped = 0
        self._rng = np.random.default_rng(7)

    # start/stop are overridden rather than inherited only to count them: the
    # claim being tested is that the check stops the source it started, on every
    # path including the failures. Inheriting the real ones would also acquire
    # the 1 ms system timer, which is not what these cases are about.
    def start(self) -> None:
        self._stop.clear()
        self.stats.running = self.alive
        self._thread = threading.Thread(target=self._run, name="iq-delayed",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stopped += 1
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.stats.running = False

    def _run(self) -> None:
        if self.error:
            self.stats.last_error = self.error
            self.stats.running = False
            return
        if self.delay and self._stop.wait(self.delay):
            return
        if not self.write:
            # Alive and silent, until told to stop.
            self._stop.wait(5.0)
            self.stats.running = False
            return
        raw = self._rng.integers(0, 256, size=self.n_bytes,
                                 dtype=np.uint8).tobytes()
        self.ring.write(raw)
        self.stats.bytes_total += len(raw)


def case_check_waits_for_samples() -> bool:
    print("--- 17: --check waits for the radio instead of guessing its timing ---")
    import io
    from contextlib import redirect_stdout

    import main

    good = ok("the readiness wait is bounded, and generously so",
              main.CHECK_READY_TIMEOUT_S == 3.0,
              f"{main.CHECK_READY_TIMEOUT_S:.1f}s (was a flat 0.5 s sleep)")
    good &= ok("and the minimum verdict is the one it was",
               main.CHECK_MIN_SAMPLES == 100_000
               and main.CHECK_SNAPSHOT_SAMPLES == 1_000_000,
               f"need {main.CHECK_MIN_SAMPLES:,} of "
               f"{main.CHECK_SNAPSHOT_SAMPLES:,} samples")

    # 17a: the radio is slower than the sleep the check used to take. It does
    # produce IQ, so the check must pass and must say how many samples it read.
    src = _DelayedSource(delay=0.7)
    buf = io.StringIO()
    t0 = time.monotonic()
    with patch.object(main, "_alert_box"), redirect_stdout(buf):
        code = main._check(src)
    dt = time.monotonic() - t0
    out = buf.getvalue()
    good &= ok("a slow radio that does produce IQ passes", code == 0,
               f"exit {code} after {dt:.2f}s")
    good &= ok("it waited for the samples rather than reading an empty ring",
               dt >= 0.7 and src.ring.total_written > 0,
               f"waited {dt:.2f}s, {src.ring.total_written // 2:,} samples in")
    good &= ok("and it reported the real count, not a placeholder",
               "ok: 1000000 samples" in out,
               next((l for l in out.splitlines() if l.startswith("ok:")), "")[:46])
    good &= ok("the wait finished inside the deadline",
               dt < main.CHECK_READY_TIMEOUT_S,
               f"{dt:.2f}s < {main.CHECK_READY_TIMEOUT_S:.1f}s")
    good &= ok("and it stopped the source it started", src.stopped == 1)

    # 17b: a transfer process that is alive and silent. Liveness is not IQ, and
    # this is the state a half-second sleep used to score as a radio producing
    # nothing. The deadline is shortened here so the gate does not pay 3 s of
    # wall clock for a case whose point is the verdict, not the patience.
    silent = _DelayedSource(write=False, alive=True)
    buf2 = io.StringIO()
    t0 = time.monotonic()
    with patch.object(main, "CHECK_READY_TIMEOUT_S", 0.4), \
         patch.object(main, "_alert_box"), redirect_stdout(buf2):
        code2 = main._check(silent)
    dt2 = time.monotonic() - t0
    out2 = buf2.getvalue()
    good &= ok("a live process with no samples is not taken as working",
               code2 == 1, f"exit {code2} after {dt2:.2f}s (transfer was alive)")
    good &= ok("and it fails within the deadline, not after it",
               dt2 < 1.5, f"{dt2:.2f}s")
    good &= ok("the failure reports the startup it waited for",
               "FAIL:" in out2 and "startup:" in out2 and "radio:" in out2,
               next((l for l in out2.splitlines() if l.startswith("startup:")),
                    "")[:52])
    good &= ok("the silent source was stopped too", silent.stopped == 1)

    # 17c: a source that gave up immediately. Still a nonzero exit, still the
    # source's own error in the report -- the point of the extra diagnostics is
    # that they say what the radio said rather than only how few bytes landed.
    dead = _DelayedSource(alive=False, write=False,
                          error="HackRF not found (-5)")
    buf3 = io.StringIO()
    t0 = time.monotonic()
    with patch.object(main, "CHECK_READY_TIMEOUT_S", 0.4), \
         patch.object(main, "_alert_box"), redirect_stdout(buf3):
        code3 = main._check(dead)
    dt3 = time.monotonic() - t0
    out3 = buf3.getvalue()
    good &= ok("a source that dies at once still fails, within the deadline",
               code3 == 1 and dt3 < 1.5, f"exit {code3} after {dt3:.2f}s")
    good &= ok("and the radio's own error is in the report",
               "HackRF not found (-5)" in out3,
               next((l for l in out3.splitlines() if l.startswith("last error:")),
                    "")[:52])
    good &= ok("and it is stopped on the failure path as well", dead.stopped == 1)
    return good


# -- 18: an interrupted check still stops the source it started -------------


def case_interrupted_check_stops_the_source() -> bool:
    """A Ctrl-C during the readiness wait must not leak the receiver.

    ``start()`` and the bounded wait sat above the ``try`` whose ``finally``
    stops the source, so an interrupt inside that wait -- up to three seconds of
    it, on a radio that has not answered yet -- unwound straight past the only
    ``stop()`` in the function. What leaked was not a handle on paper: the
    sampling thread, the transfer child holding the USB handle, and the 1 ms
    system timer held for the whole streaming window. The operator is told
    nothing either, because the interrupt lands before the first line of the
    report exists.

    ``_check`` is driven directly, as 17 drives it: it takes a source, prints,
    raises no dialog of its own that cannot be patched, and returns an int
    rather than exiting, so no part of it has to be reimplemented here. The
    seam is the wait, and the fake underneath it is a real shape rather than a
    stub -- a live-but-silent receiver, the same one 17b uses, which really does
    sit in the polling loop. The interrupt is raised from inside the sleep that
    loop spends polling in, which is where a real one lands: on a signal during
    ``time.sleep``. It raises on the first such sleep, so the case proves the
    wait was entered as well as proving what happens when it is left.
    """
    print("--- 18: a check interrupted while waiting still stops its source ---")
    import io
    from contextlib import redirect_stdout

    import main

    src = _DelayedSource(write=False, alive=True)
    buf = io.StringIO()
    sleeps: list[float] = []

    def interrupted_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        raise KeyboardInterrupt

    raised = ""
    # The deadline is shortened as in 17b so that a regression which somehow
    # skips the sleep fails its assertions quickly instead of costing the gate
    # the full three seconds on the way to the same failure.
    with patch.object(main, "CHECK_READY_TIMEOUT_S", 0.4), \
         patch.object(main, "_alert_box") as alert, \
         patch("time.sleep", interrupted_sleep), redirect_stdout(buf):
        try:
            main._check(src)
        except KeyboardInterrupt:
            raised = "KeyboardInterrupt"
        except BaseException as exc:           # any other escape is a new path
            raised = repr(exc)

    out = buf.getvalue()
    good = ok("the interrupt landed inside the readiness wait, not before it",
              raised == "KeyboardInterrupt" and bool(sleeps),
              f"{len(sleeps)} poll sleep(s); _check raised {raised or 'nothing'}")
    good &= ok("the source the check started was stopped, exactly once",
               src.stopped == 1, f"stop() ran {src.stopped} time(s)")
    good &= ok("its sampling thread is gone, not merely told to stop",
               src._thread is None and not src.running,
               "thread detached, _stop set")
    good &= ok("nothing was reported: an interrupted check reaches no verdict",
               not any(k in out for k in ("ok:", "FAIL:", "startup:", "radio:",
                                          "last error:")),
               out.strip()[:46] or "(no output at all)")
    good &= ok("and no alert box was raised on the way out either",
               alert.call_count == 0, f"{alert.call_count} alert(s)")

    # The fix is only allowed to work because ``stop`` survives being reached
    # from more than one place -- which is why there is exactly one ``finally``
    # rather than a defensive second call. Pinned against the real IQSource,
    # since the fake above only counts calls and proves nothing about it.
    real = sdr.SimSource(video=False)
    real.start()
    time.sleep(0.05)
    real.stop()
    second = ""
    try:
        real.stop()
    except Exception as exc:
        second = repr(exc)
    good &= ok("IQSource.stop is safe to call twice, which is what this relies on",
               second == "" and real._thread is None and not real.running,
               second or "the second stop() returned cleanly")
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
            case_respawn_does_not_starve_retune(),
            case_probe_launches_without_a_console(),
            case_check_waits_for_samples(),
            case_interrupted_check_stops_the_source(),
        ]
    bad = [i for i, v in enumerate(cases) if not v]
    print(f"\n{sum(cases)}/{len(cases)} cases passed")
    print("RESULT:", "PASS" if not bad else f"FAIL (cases {bad})")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
