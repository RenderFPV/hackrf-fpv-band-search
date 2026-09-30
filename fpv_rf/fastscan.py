"""Pick the fastest band search this machine can actually run, and hand back the
result in the one shape the rest of the app understands.

Two engines produce a :class:`fpv_rf.scan.ScanResult`, and the choice between
them is not a preference -- it is forced by the hardware.

*The sweep* (:mod:`fpv_rf.sweep`) is about a hundred times faster. Measured on a
HackRF One, the whole US band goes from 12.5 s to 0.15 s, which is the
difference between a search you wait for and one you keep an eye on. It works
by never letting go of the radio.

*The hop walk* (:class:`fpv_rf.scan.BandScanner`) is the fallback. It reopens the
USB device once per hop, and that costs about 360 ms every time, so it is slow
by arithmetic rather than by implementation. It is still needed, because it reads
a full video field and can therefore say whether a signal is decodable analogue
video, which a spectrum sweep cannot. It also needs nothing but
``hackrf_transfer``, so it is what runs on a machine with no sweep tool, or one
whose FFTW is missing.

The engines are not interchangeable in what they can *claim*. A sweep resolves
to 1 MHz bins and reports energy, not video. So the result carries the engine
that produced it, and :meth:`fpv_rf.scan.ScanResult.describe` says so, rather
than letting a coarse answer be read as a precise one.

The handover itself is the delicate part, and it is the reason the fallback is
not a formality. The sweep tool opens the radio itself, so nothing else may hold
it, which means stopping a ``hackrf_transfer`` that is mid-transfer. On a
HackRF One that leaves the device refusing the next open for a period nobody can
predict -- measured on this machine from 0 s to over 30 s, with no reliable way
to wait it out and no way to ask whether the wait is over. A polite stop does
not help: the tool ignores Ctrl-C entirely.

So the fast path is *usually* fast and *always* answered. When the sweep cannot
have the radio, the search falls back to the hop walk, which drives the source it
is already holding and is therefore unaffected. The radio is taken back in a
``finally`` either way, because leaving it lent out would leave the operator
watching a frozen picture with nothing in the log to say why.
"""

from __future__ import annotations

import threading
from typing import Callable

from . import scan, sdr, sweep

#: Reported when the fast engine is wanted but cannot run, so the status line
#: can say why rather than just how slow it was.
NO_SWEEP = "sweep unavailable"

#: How long a caller off the GUI thread will wait for the shared probe before
#: giving up and answering "no". Generous enough to cover the probe's own 20 s
#: subprocess timeout, plus slack for a loaded machine.
PROBE_WAIT_S = 30.0

#: The session's one sweep-availability answer, and the machinery to obtain it
#: without anyone having to wait for it.
#:
#: ``sweep.available()`` starts the helper with a 20 s timeout to find out
#: whether it can run at all. That is a once-per-process question, and it used
#: to be asked on the GUI thread the first time a band search was started -- so
#: the window froze for up to twenty seconds at exactly the moment the operator
#: clicked the one button whose whole job is to be quick. The probe is the same
#: question whoever asks it and the same answer every time, so it is asked once
#: on a worker thread and read from here by everyone who needs it.
_probe_lock = threading.Lock()
_probe_done = threading.Event()
_probe_result: bool | None = None


def _run_probe() -> bool:
    try:
        return bool(sweep.available())
    except Exception:
        # A probe that raised has not proved the tool is unusable, but it has
        # certainly not proved it works, and the fallback is a working search.
        return False


def _probe_worker() -> None:
    global _probe_result
    try:
        _probe_result = _run_probe()
    finally:
        # Set even on an unexpected failure, so no caller waits forever on a
        # probe that has already given up.
        _probe_done.set()


def start_probe() -> bool | None:
    """Begin the sweep-availability probe on a worker thread.

    Idempotent, and never blocks. Returns the answer if it is already known and
    None if it is still being established -- so a caller can report what it
    knows and let the answer arrive later rather than wait for it.
    """
    global _probe_result
    if _probe_done.is_set():
        return _probe_result
    with _probe_lock:
        if not _probe_done.is_set():
            threading.Thread(
                target=_probe_worker, name="sweep-probe", daemon=True
            ).start()
    return _probe_result if _probe_done.is_set() else None


def probe_result() -> bool | None:
    """The session's answer, or None while it is still being established."""
    return _probe_result if _probe_done.is_set() else None


def wait_for_probe(timeout: float = PROBE_WAIT_S) -> bool:
    """The answer, waiting for it if necessary. For callers off the GUI thread.

    Waiting on the shared probe rather than running a second one matters: two
    concurrent probes would spawn the helper twice, and the two could disagree
    about a machine that is slow to start.
    """
    if not _probe_done.is_set():
        start_probe()
        _probe_done.wait(timeout)
    return bool(_probe_result)


def fast_ok(source: sdr.IQSource) -> bool:
    """Whether the sweep engine can serve this source right now.

    Two separate things have to hold, and they fail for unrelated reasons worth
    telling apart. The source has to be a radio that can be lent out, and the
    sweep tool has to exist and start. A simulator satisfies neither, and a file
    source is not retunable at all.

    This blocks on the shared probe, so it belongs on a worker thread. The UI
    reads :func:`probe_result` instead and reports "still being checked" rather
    than making the window wait for it.
    """
    if not getattr(source, "exclusive_use", False):
        return False
    return wait_for_probe()


def why_not_fast(source: sdr.IQSource) -> str:
    """Why the slow engine is in use, in a sentence.

    Non-blocking. If the shared probe has not answered yet this says so rather
    than asking the question itself, because the caller here is usually the
    status line and the answer it wants is "we do not know yet", not a twenty
    second wait followed by the same sentence.
    """
    if not getattr(source, "exclusive_use", False):
        return f"{NO_SWEEP}: {source.kind} holds no radio to lend out"
    if probe_result() is None:
        start_probe()
        return f"{NO_SWEEP}: still checking whether the sweep tool runs here"
    # Empty would mean nothing is wrong, which cannot be true on this path --
    # fast_ok() already said no. Kept as a belt-and-braces default so this
    # cannot return an empty reason and leave the status line blank.
    return sweep.unavailable_reason() or f"{NO_SWEEP} for an unknown reason"


def scan_span(
    source: sdr.IQSource,
    lo_hz: int,
    hi_hz: int,
    hop_frac: float = 0.9,
    sweeps: int = 1,
    lna_db: int = 32,
    vga_db: int = 16,
    amp: bool = True,
    cancel: threading.Event | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> scan.ScanResult:
    """Search a span with the fastest engine available, and return a ScanResult.

    Falls back to the hop walk on any condition that makes the sweep unusable --
    a missing tool, a radio that will not lend itself, a sweep the tool refused
    to run. The fallback is silent to the caller but not to the operator:
    ``result.engine`` says which one answered, and ``result.error`` carries the
    reason when something went wrong.
    """
    def say(msg: str, frac: float) -> None:
        if progress is not None:
            progress(msg, frac)

    def by_hops(why: str) -> scan.ScanResult:
        """Run the slow engine, saying why on the way in."""
        say(f"{why}; sweeping the band one hop at a time", 0.0)
        scanner = scan.BandScanner(source, hop_frac=hop_frac)
        watcher = None
        if cancel is not None:
            # The scanner is cancelled from the GUI thread in the normal flow;
            # here it is driven from an event, so something has to watch the
            # event. A daemon thread that goes away with the scan is enough --
            # it holds no state the scanner needs back.
            def _watch() -> None:
                cancel.wait()
                scanner.cancel()
            watcher = threading.Thread(target=_watch, name="scan-cancel",
                                       daemon=True)
            watcher.start()
        try:
            res = scanner.scan(lo_hz=lo_hz, hi_hz=hi_hz, progress=progress)
        finally:
            if watcher is not None:
                # Unblock the watcher so the thread can end rather than linger
                # for as long as the process lives.
                cancel.set()
        res.engine = "hops"
        return res

    if not fast_ok(source):
        return by_hops(why_not_fast(source))

    if not fast_ok(source):
        return by_hops(why_not_fast(source))

    say("lending the radio to the sweep tool", 0.0)
    if not source.release_device():
        # A radio that will not let go is not a dead end, because the hop walk
        # never asks to borrow it -- it drives the source it is already holding.
        # Falling back is both the fast thing and the honest one; reporting an
        # error here would be wrong twice over, since the answer the operator
        # wants is still obtainable.
        #
        # Take the radio back before handing over. A release that returns False
        # may still have torn down the reader, and leaving the source released
        # would suppress process spawning for the fallback that follows -- which
        # is the same defect this function used to have on the *successful*
        # release path.
        source.reclaim_device()
        return by_hops("the radio would not let go for a sweep")

    # -- the sweep, with no control flow that returns from inside ------------
    #
    # Everything below used to be wrapped in try/finally with the fallback
    # written as ``return by_hops(...)``. That was the bug, and it was a quiet
    # one: a return expression is evaluated *before* its finally block runs, so
    # the entire hop-walk fallback executed while the source was still released.
    # A released source refuses to spawn, so every hop waited out its settle
    # timeout and then read whatever was left in the ring -- one pre-scan
    # capture, re-reported at 34 different frequencies. The operator saw a
    # complete, confident, entirely fabricated result.
    #
    # So the sweep is attempted into a value, the radio is handed back, sample
    # flow is confirmed, and only then is any other engine allowed to run.
    sweep_res: scan.ScanResult | None = None
    why = ""
    say("sweeping", 0.1)
    try:
        tr = sweep.run_sweep(
            lo_hz, hi_hz, sweeps=sweeps,
            # Snap to what the tool can actually ask for. The sweep tool takes
            # LNA gain in 8 dB steps and VGA in 2 dB steps, so a value the UI
            # allows but the tool rejects would be rejected as a usage error
            # and the sweep would return nothing at all.
            lna_db=_snap_lna(lna_db), vga_db=_snap_vga(vga_db), amp=amp,
            cancel=cancel,
        )
        if tr.error == "cancelled" or (cancel is not None and cancel.is_set()):
            sweep_res = sweep.result_from_trace(tr)
            sweep_res.engine = "sweep"
        elif tr.ok:
            say("reading the trace", 0.8)
            sweep_res = sweep.result_from_trace(tr)
            sweep_res.engine = "sweep"
        else:
            # The sweep could not have the radio. This is common rather than
            # exceptional on a HackRF: the handover means killing a process
            # that is mid-transfer, and the device then refuses the next open
            # for a period nobody can predict -- measured anywhere from 0 s to
            # well over 30 s, with no clean way to wait it out. So the answer is
            # not to report a failure the operator cannot act on, but to go and
            # get the answer by the route that still works. The hop walk drives
            # the radio we already hold, so it is unaffected.
            why = f"the sweep could not use the radio ({tr.error})"
    except Exception as exc:                    # noqa: BLE001 - never abandon
        why = f"the sweep failed ({exc})"

    # -- the radio comes back, and must actually deliver -------------------
    if not source.reclaim_device():
        # Surfaced rather than swallowed, and not papered over with a fallback:
        # with no radio there is no other engine to fall back to, and running
        # one would manufacture a result out of nothing. "I could not receive
        # anything" and "the band is empty" are different facts, and only one of
        # them is true.
        say("the radio did not come back after the sweep", 1.0)
        return _acquisition_failure(lo_hz, hi_hz, why or
                                    "the radio did not come back after the sweep")

    if sweep_res is not None:
        return sweep_res

    # Confirm the source is live before asking it to measure 34 hops. Normally
    # this is satisfied in a few milliseconds, because reclaim respawns the
    # transfer process; the wait is here so a radio that reopens but never
    # delivers is reported as such instead of as a quiet band.
    if not source.wait_for_bytes(4 * RECLAIM_FLOW_BYTES, timeout=RECLAIM_FLOW_TIMEOUT):
        return _acquisition_failure(
            lo_hz, hi_hz,
            f"{why}; the radio reopened but delivered no samples, so the band "
            f"was not measured",
        )

    return by_hops(why)


#: Bytes asked for when checking that a reclaimed radio is really delivering.
#: Small on purpose: this is a liveness check, not a measurement, and it runs on
#: the fallback path where the operator is already waiting.
RECLAIM_FLOW_BYTES = 2_000

#: Bound on that check. A live radio at 10 MS/s delivers 20 MB/s, so this is
#: roughly 0.4 ms of signal; anything slower is a dead pipe, not slow data.
RECLAIM_FLOW_TIMEOUT = 1.5


def _acquisition_failure(lo_hz: int, hi_hz: int, why: str) -> scan.ScanResult:
    """A result that says "not measured", never "nothing there".

    Without this the app has two ways to report an empty band, and it was
    reaching for the wrong one: a receiver that was never delivering produced a
    confident "nothing above the noise floor", which is indistinguishable from a
    genuinely quiet band and actively misleading. The distinction is carried in
    ``error``, which the UI already surfaces, and in the absence of candidates.
    """
    res = scan.ScanResult(lo_hz=int(lo_hz), hi_hz=int(hi_hz))
    res.error = f"band search could not run: {why}"
    res.stopped_early = True
    res.engine = "none"
    return res


def _snap_lna(db: int) -> int:
    """Round a gain to a step the sweep tool accepts."""
    return max(0, min(40, int(round(int(db) / 8.0)) * 8))


def _snap_vga(db: int) -> int:
    """Round a gain to a step the sweep tool accepts."""
    return max(0, min(62, int(round(int(db) / 2.0)) * 2))
