"""Raising an alert when a band search finds energy above the noise floor.

The condition this exists to serve is the operator's: *tell me when the HackRF
sees RF energy above the noise floor, and tell me which band and channel it is.*
Everything here is arranged around one uncomfortable fact -- a band search is run
repeatedly, on a 9 MHz grid, over a band where channels sit 2-3 MHz apart. Left
alone that produces the same alert every sweep, which is not an alert, it is
noise with a siren. So:

* a finding is keyed by *what was found*, not by when it was found, so the same
  transmitter is recognised across sweeps;
* the first sighting alerts, and repeats inside the cooldown are counted instead
  of raised, so a banner that flashes every 2 seconds stops being readable;
* a finding that goes away and comes back is a new alert, because an operator who
  watched it drop out needs to know it is back;
* a finding that goes away and stays away raises a "lost" event, because silence
  from a drone that was transmitting two minutes ago is itself information.

The alert carries band, channel and frequency because that is what was asked for
and because it is what makes the alert actionable: "something is on 5.8 GHz" is
not a thing an operator can act on, and "F4, 5800 MHz, analogue video" is.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable

from . import bands, dsp
from .scan import Candidate, ScanResult

#: How long a finding is remembered after it was last seen. A sweep takes
#: seconds, so this has to be comfortably longer than one -- but not so long that
#: a pilot who landed, took off again and landed is treated as the same event.
FORGET_S = 30.0

#: Minimum gap between two *raised* alerts for the same finding. Repeats inside
#: this window are counted, not raised.
COOLDOWN_S = 15.0

#: How wide a finding is, in Hz, for the purpose of deciding whether two sweeps
#: are looking at the same transmitter. Matches the scanner's own merge width:
#: a 9 MHz hop grid can legitimately report the same VTX 3 MHz apart, and 5802
#: and 5805 are 3 MHz apart while every chart channel is at least 19 MHz. Two
#: real transmitters closer than this are not separable by a 10 MS/s sweep anyway,
#: and the scanner already reports them as one finding -- keying them apart here
#: would make the alert list disagree with the scan it came from.
SAME_TRANSMITTER_HZ = 15_000_000

#: How many alerts to keep. Enough to cover a session's interesting moments
#: without the log becoming the thing that eats memory.
LOG_LIMIT = 500


class AlertKind(str, Enum):
    FOUND = "found"
    LOST = "lost"


@dataclass(frozen=True)
class AlertEvent:
    """One thing worth telling the operator about."""

    kind: AlertKind
    at: float
    key: str
    band: str
    channel: int | None
    frequency_hz: int
    signal: dsp.SignalKind
    detail: str
    #: How many times this finding has been seen, including this one.
    seen: int = 1
    #: The channel label as the operator sees it, e.g. ``F4``.
    label: str = ""

    def headline(self) -> str:
        """The one-line form: band, channel, frequency."""
        if self.kind is AlertKind.LOST:
            return f"{self.label or 'signal'} gone -- {self.detail}"
        return f"{self.label} @ {bands.format_mhz(self.frequency_hz)} -- {self.detail}"

    def __str__(self) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.at))
        return f"{stamp}  {self.headline()}"


def finding_key(freq_hz: int, kind: dsp.SignalKind) -> str:
    """A canonical name for a finding, for logs and de-duplication."""
    return f"{int(freq_hz)} {kind.value}"


def _describe(cand: Candidate) -> str:
    """One-line summary: classification first, then the numbers behind it."""
    r = cand.reading
    if r.kind is dsp.SignalKind.VIDEO:
        std = r.spec.name if r.spec else "?"
        quality = (
            f"decodable {std} video, sync {r.sync_quality*100:.0f}%, "
            f"SNR {r.snr_db:.1f} dB"
        )
    elif r.kind is dsp.SignalKind.CARRIER:
        quality = f"RF present but no video structure ({r.peak_to_median_db:.0f} dB)"
    else:
        quality = "RF present, noise-like"
    if cand.off_chart:
        quality += ", off chart"
    if cand.merged > 1:
        quality += f", {cand.merged} finds over {cand.span_hz/1e6:.0f} MHz"
    return quality


class AlertEngine:
    """Turns a stream of scan results into a stream of alerts worth reading.

    Not thread-safe by itself in the sense that matters here: :meth:`update` is
    meant to be called from one thread (the scanner's), and listeners are
    invoked on that same thread. A Qt front end must therefore marshal to its own
    thread -- see the note in :meth:`_emit`.
    """

    def __init__(
        self,
        cooldown_s: float = COOLDOWN_S,
        forget_s: float = FORGET_S,
        log_limit: int = LOG_LIMIT,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cooldown_s = float(cooldown_s)
        self.forget_s = float(forget_s)
        self.clock = clock
        self.log: deque[AlertEvent] = deque(maxlen=int(log_limit))
        #: key -> (first seen, last seen, last raised, count, candidate)
        self._seen: dict[str, tuple[float, float, float, int, Candidate]] = {}
        self._listeners: list[Callable[[AlertEvent], None]] = []
        self._lock = threading.Lock()
        self.counts: dict[AlertKind, int] = {AlertKind.FOUND: 0, AlertKind.LOST: 0}
        #: Findings seen since the last :meth:`update`, for the UI's live panel.
        self.active: list[Candidate] = []
        self.last_result: ScanResult | None = None

    # -- listeners ---------------------------------------------------------

    def add_listener(self, fn: Callable[[AlertEvent], None]) -> None:
        """Register a callback. It is called on whichever thread updated."""
        self._listeners.append(fn)

    def _emit(self, events: list[AlertEvent]) -> None:
        for ev in events:
            self.log.append(ev)
            self.counts[ev.kind] += 1
            for fn in list(self._listeners):
                try:
                    fn(ev)
                except Exception:
                    # A broken listener must not take the scanner down with it,
                    # and must not stop the remaining listeners from being told.
                    print(f"alert listener failed: {fn!r}", file=sys.stderr)

    # -- main entry point --------------------------------------------------

    def update(
        self, result: ScanResult | None, at: float | None = None
    ) -> list[AlertEvent]:
        """Fold one scan result in and return whatever is worth alerting on.

        ``None`` means "the search did not complete or found nothing" and is
        treated as an empty observation -- *not* as "everything is gone". A scan
        that was cancelled halfway through has not proved anything, and reporting
        a dozen "lost" events for channels it never got to would be a lie.
        """
        now = self.clock() if at is None else at
        with self._lock:
            self.last_result = result
            if result is None:
                return []
            self.active = list(result.candidates)
            return self._fold(now, result.candidates)

    def note(self, cand: Candidate, at: float | None = None) -> AlertEvent | None:
        """Alert on a single finding, e.g. a channel the operator picked.

        Goes through the same key and cooldown machinery as a scan, so watching
        a fixed channel produces one alert and a repeat count rather than a
        banner that never goes away.
        """
        now = self.clock() if at is None else at
        with self._lock:
            self.active = [cand]
            events = self._fold(now, [cand])
        return events[0] if events else None

    def _fold(self, now: float, cands: Iterable[Candidate]) -> list[AlertEvent]:
        events: list[AlertEvent] = []
        matched: set[str] = set()

        for cand in cands:
            key = self._identify(cand, matched)
            matched.add(key)
            first, last, raised, count, _ = self._seen.get(
                key, (now, now, -1e18, 0, cand)
            )
            count += 1
            if count == 1 or now - raised >= self.cooldown_s:
                events.append(self._make(AlertKind.FOUND, now, key, cand, count))
                self._seen[key] = (first, now, now, count, cand)
            else:
                # A repeat. Keep the sighting time fresh, so a signal that drops
                # out and returns inside the cooldown is still a repeat rather
                # than a new alert -- and so it does not trip the "lost" rule
                # below, which fires once a finding has been absent long enough.
                self._seen[key] = (first, now, raised, count, cand)

        # Anything we are not seeing this round, long enough gone, is lost.
        # One loop, not two: an earlier version forgot stale keys first and then
        # looked for them, so the lost event could never fire.
        for key, (first, last, raised, count, cand) in list(self._seen.items()):
            if key in matched or now - last < self.forget_s:
                continue
            del self._seen[key]
            if raised >= 0:
                events.append(self._make(AlertKind.LOST, now, key, cand, count))

        self._emit(events)
        return events

    def _identify(self, cand: Candidate, taken: set[str]) -> str:
        """Which known finding, if any, does this one belong to?

        Matched by frequency proximity rather than by rounding to a grid. A grid
        cannot do this job: two frequencies within the sweep's own resolution can
        still fall either side of a bucket edge, so a 9 MHz grid would split a
        single VTX into two alerts and then report the operator's drone as two
        separate aircraft. Proximity is also the right question -- "is this the
        thing I already told you about", not "does this hash match".

        ``taken`` keeps two findings in the *same* sweep from collapsing into one
        key, which would silently lose the second.
        """
        for key, rec in self._seen.items():
            if key in taken:
                continue
            prior = rec[4]
            if prior.reading.kind is not cand.reading.kind:
                continue
            if abs(prior.frequency_hz - cand.frequency_hz) <= SAME_TRANSMITTER_HZ:
                # Trust the newer, tighter estimate for the stored position.
                self._seen[key] = (rec[0], rec[1], rec[2], rec[3], cand)
                return key
        return finding_key(cand.frequency_hz, cand.reading.kind)

    def _make(
        self,
        kind: AlertKind,
        now: float,
        key: str,
        cand: Candidate,
        count: int,
    ) -> AlertEvent:
        # An off-chart finding must not present itself as a confident channel
        # number. "F4" when the nearest channel is 2 MHz away is a claim the
        # sweep has not earned.
        if cand.match is None:
            label = "off-chart"
        elif cand.off_chart:
            label = f"off-chart (near {cand.match.label})"
        else:
            label = cand.match.label
        return AlertEvent(
            kind=kind,
            at=now,
            key=key,
            band=cand.band,
            channel=cand.channel,
            frequency_hz=cand.frequency_hz,
            signal=cand.reading.kind,
            detail=_describe(cand),
            seen=count,
            label=label,
        )

    # -- state -------------------------------------------------------------

    def repeat_count(self, cand: Candidate) -> int:
        """How many sweeps in a row this finding has appeared.

        ``AlertEvent.seen`` is fixed at the moment the alert was raised, which is
        right for a log entry and useless for a live "seen 47 times" readout.
        This is the live number.
        """
        with self._lock:
            key = self._identify(cand, set())
            rec = self._seen.get(key)
            return rec[3] if rec else 0

    def clear(self) -> None:
        """Forget every finding. Used when the operator changes band or source."""
        with self._lock:
            self._seen.clear()
            self.active = []

    @property
    def muted(self) -> bool:
        """True when everything currently held has already been alerted on.

        The UI uses this to show a steady "known: F4" indicator rather than a
        flashing banner, which is the difference between a status light and an
        alarm.
        """
        if not self._seen:
            return True
        now = self.clock()
        return all(v[2] >= 0 and now - v[2] < self.cooldown_s for v in self._seen.values())

    def recent(self, n: int = 20) -> list[AlertEvent]:
        return list(self.log)[-n:]

    def summary(self) -> str:
        return (
            f"{self.counts[AlertKind.FOUND]} found, "
            f"{self.counts[AlertKind.LOST]} lost, "
            f"{len(self.active)} active, {len(self.log)} in the log"
        )


# --------------------------------------------------------------------------
# Audible notification
# --------------------------------------------------------------------------


class Beeper:
    """Optional audible alert, on a short-delay tone.

    Two things this deliberately does not do. It does not beep synchronously:
    ``winsound.Beep`` blocks for its whole duration, and doing that on the
    scanner's thread would stall the next hop. And it does not queue: if three
    findings appear in one sweep, that is one pattern, not three overlapping
    tones, because three simultaneous beeps are just noise.

    Falls back to printing to stderr where there is no console beep, so a
    headless run still says something.
    """

    #: (frequency Hz, duration ms) by severity. Distinct pitches so the two
    #: events are tellable apart without looking at the screen.
    FOUND_TONE = (1320, 180)
    LOST_TONE = (660, 90)
    REPEAT_TONE = (990, 60)

    def __init__(self, enabled: bool = True, minimum_gap_s: float = 0.35) -> None:
        self.enabled = bool(enabled)
        self.minimum_gap_s = float(minimum_gap_s)
        self._last = 0.0
        self._lock = threading.Lock()
        self.suppressed = 0

    @property
    def available(self) -> bool:
        return sys.platform == "win32"

    def _tone(self, kind: AlertKind) -> tuple[int, int]:
        if kind is AlertKind.LOST:
            return self.LOST_TONE
        return self.FOUND_TONE

    def beep(self, kind: AlertKind = AlertKind.FOUND) -> bool:
        """Sound once. Returns whether a tone was actually produced."""
        if not self.enabled:
            return False
        with self._lock:
            now = time.monotonic()
            if now - self._last < self.minimum_gap_s:
                self.suppressed += 1
                return False
            self._last = now
        freq, ms = self._tone(kind)
        if self.available:
            try:
                import winsound

                winsound.Beep(freq, ms)   # a short tone cannot outlast a hop
                return True
            except Exception:
                pass
        print(f"\a[{freq} Hz {ms} ms] {kind.value}", file=sys.stderr)
        return True

    def pattern(self, events: list[AlertEvent]) -> int:
        """Sound for a batch of new alerts. Returns how many tones were made."""
        if not events:
            return 0
        n = 0
        for i, ev in enumerate(events):
            # One tone per event, but only if they are far enough apart in time
            # to be heard as separate events.
            if i == 0 or self.beep(ev.kind):
                n += 1
        return n
