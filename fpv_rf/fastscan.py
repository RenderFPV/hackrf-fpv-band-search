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


def fast_ok(source: sdr.IQSource) -> bool:
    """Whether the sweep engine can serve this source right now.

    Two separate things have to hold, and they fail for unrelated reasons worth
    telling apart. The source has to be a radio that can be lent out, and the
    sweep tool has to exist and start. A simulator satisfies neither, and a file
    source is not retunable at all.
    """
    return bool(getattr(source, "exclusive_use", False)) and sweep.available()


def why_not_fast(source: sdr.IQSource) -> str:
    """Why the slow engine is in use, in a sentence."""
    if not getattr(source, "exclusive_use", False):
        return f"{NO_SWEEP}: {source.kind} holds no radio to lend out"
    return sweep.unavailable_reason()


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

    say("lending the radio to the sweep tool", 0.0)
    if not source.release_device():
        # A radio that will not let go is not a dead end, because the hop walk
        # never asks to borrow it -- it drives the source it is already holding.
        # Falling back is both the fast thing and the honest one; reporting an
        # error here would be wrong twice over, since the answer the operator
        # wants is still obtainable.
        return by_hops("the radio would not let go for a sweep")
    try:
        say("sweeping", 0.1)
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
            res = sweep.result_from_trace(tr)
            res.engine = "sweep"
            return res
        if not tr.ok:
            # The sweep could not have the radio. This is common rather than
            # exceptional on a HackRF: the handover means killing a process
            # that is mid-transfer, and the device then refuses the next open
            # for a period nobody can predict -- measured anywhere from 0 s to
            # well over 30 s, with no clean way to wait it out. So the answer is
            # not to report a failure the operator cannot act on, but to go and
            # get the answer by the route that still works. The hop walk drives
            # the radio we already hold, so it is unaffected.
            #
            # Reported through the status line rather than left in ``error``:
            # the result is a good one, and marking it failed would be a lie in
            # the other direction -- it would train the operator to ignore
            # errors that matter.
            return by_hops(f"the sweep could not use the radio ({tr.error})")
        say("reading the trace", 0.8)
        res = sweep.result_from_trace(tr)
        res.engine = "sweep"
        return res
    finally:
        if not source.reclaim_device():
            # Surfaced rather than swallowed: the picture is now frozen, and
            # the operator deserves to know the radio did not come back.
            say("the radio did not come back after the sweep", 1.0)


def _snap_lna(db: int) -> int:
    """Round a gain to a step the sweep tool accepts."""
    return max(0, min(40, int(round(int(db) / 8.0)) * 8))


def _snap_vga(db: int) -> int:
    """Round a gain to a step the sweep tool accepts."""
    return max(0, min(62, int(round(int(db) / 2.0)) * 2))
