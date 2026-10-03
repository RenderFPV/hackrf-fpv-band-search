"""Video acquisition and monitoring policy; independent of Qt and audio."""
from __future__ import annotations
from dataclasses import dataclass
import threading
import time
import numpy as np
from . import bands, dsp, scan


def video_present(reading: dsp.ChannelReading) -> bool:
    """Use the same evidence bar for acquisition and subsequent observations."""
    return (reading.kind is dsp.SignalKind.VIDEO and reading.sync_quality >= .60
            and .55 <= reading.sync_density <= 1.25)


def confirm(source, frequency_hz: int, cancel=None, first=None):
    """Require video on two independent buffers; never promote a bare carrier."""
    good = []
    readings = []
    for i in range(3):
        if cancel is not None and cancel.is_set():
            return None
        if i == 0 and first is not None:
            reading = first
        else:
            iq = source.retune_settled(frequency_hz, int(source.sample_rate * .042), timeout=2)
            if iq.size < int(source.sample_rate * .02):
                return None
            reading = dsp.assess_channel(iq, source.sample_rate)
        readings.append(reading)
        if video_present(reading):
            good.append(reading)
        if len(good) >= 2:
            rates = [r.line_rate_hz for r in good]
            if max(rates) - min(rates) <= np.mean(rates) * .005:
                return scan.Candidate(frequency_hz=frequency_hz,
                    reading=min(good, key=lambda r: r.sync_quality),
                    match=bands.match_frequency(frequency_hz), uncertainty_hz=0)
        if i == 1 and not good:
            break
    return None


def acquire(source, lo_hz, hi_hz, *, cancel=None, progress=None, frequencies=None,
            auto_gain=True):
    """Walk channel presets directly; avoids a second process owning the radio.

    Current tuning is tried first. Every preset gets IQ verification even if
    its broad FM spectrum has no peak above its own median noise estimate.
    A hit is returned immediately instead of waiting for the entire band.
    """
    start = time.monotonic()
    result = scan.ScanResult(lo_hz=lo_hz, hi_hz=hi_hz, engine="video")
    candidates = (list(frequencies) if frequencies is not None else
                  [f for b in bands.ALL_BANDS for f in b.channels])
    candidates = sorted(set(f for f in candidates if lo_hz <= f <= hi_hz))
    current = int(source.frequency_hz)
    if lo_hz <= current <= hi_hz and (frequencies is None or current in candidates):
        candidates = [current] + [f for f in candidates if f != current]
    if not source.retunable:
        candidates = [current]
    n = int(source.sample_rate * .042)
    consecutive_failures = 0
    for index, freq in enumerate(candidates):
        if cancel is not None and cancel.is_set():
            result.stopped_early = True
            break
        if progress:
            progress(f"checking video at {freq / 1e6:.3f} MHz", index / max(1, len(candidates)))
        iq = source.retune_settled(freq, n, timeout=2)
        result.hops += 1
        if iq.size < int(source.sample_rate * .02):
            result.failed_hops += 1
            consecutive_failures += 1
            result.error = "Receiver did not deliver fresh samples; search incomplete"
            if consecutive_failures >= 3:
                break
            continue
        consecutive_failures = 0
        result.measured += 1
        # ADC occupancy is evaluated after removing DC. Changing gain needs a
        # fresh buffer, and is bounded to one correction per frequency.
        if auto_gain and source.kind == "hackrf":
            peak = float(np.percentile(np.maximum(abs(iq.real), abs(iq.imag)), 99.9))
            rms = float(np.sqrt(np.mean(abs(iq - iq.mean()) ** 2)))
            lna, vga = source.lna_gain_db, source.vga_gain_db
            new = ((lna, max(0, vga - 12)) if peak > .95 else
                   (min(40, lna + 8), min(40, vga + 8)) if rms < .05 else (lna, vga))
            if new != (lna, vga) and source.set_gains(*new):
                iq = source.retune_settled(freq, n, timeout=2)
                if iq.size < int(source.sample_rate * .02):
                    result.failed_hops += 1
                    result.error = "No fresh samples after gain adjustment"
                    continue
        reading = dsp.assess_channel(iq, source.sample_rate)
        found = confirm(source, freq, cancel, first=reading) if video_present(reading) else None
        # A nominal channel may be offset. Only spend extra retunes where
        # horizontal structure supports it, never chase an isolated RF peak.
        if (found is None and reading.sync_quality >= .35
                and reading.sync_density >= .4 and source.retunable):
            for offset in (-1_000_000, 1_000_000, -2_000_000, 2_000_000):
                if cancel is not None and cancel.is_set():
                    break
                fine = freq + offset
                if not lo_hz <= fine <= hi_hz:
                    continue
                if progress:
                    progress(f"verifying video near {fine / 1e6:.3f} MHz",
                             index / max(1, len(candidates)))
                found = confirm(source, fine, cancel)
                if found is not None:
                    break
        if found is not None:
            result.candidates = [found]
            # Successful early acquisition says nothing about the rest of
            # the band. Scope loss accounting only to the measured channel.
            result.lo_hz = result.hi_hz = found.frequency_hz
            break
    result.stopped_early |= bool(cancel is not None and cancel.is_set())
    if result.stopped_early:
        result.candidates = []
    result.elapsed_s = time.monotonic() - start
    if progress:
        progress("video acquired" if result.candidates else "no confirmed video", 1.0)
    return result


@dataclass
class Monitor:
    """Monotonic deadlines for retry/loss, with explicit pause and cancellation."""
    enabled: bool = True
    retry_s: float = 2.0
    loss_s: float = 3.0
    next_scan: float = 0.0
    last_lock: float = 0.0
    tracking: bool = False

    def acquired(self, now):
        self.tracking = True
        self.last_lock = now
        self.next_scan = now + self.loss_s

    def completed(self, now, success=False):
        if success:
            self.acquired(now)
        else:
            self.tracking = False
            self.next_scan = now + self.retry_s

    def due(self, now, locked=False, busy=False):
        if not self.enabled or busy:
            return False
        if self.tracking and locked:
            self.last_lock = now
            return False
        if self.tracking and now - self.last_lock < self.loss_s:
            return False
        return now >= self.next_scan
