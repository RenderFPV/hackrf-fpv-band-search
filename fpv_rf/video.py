"""Live decode pipeline: IQ in, 60 fps luminance frames out.

Threading model
---------------
One background thread (:class:`DecodeWorker`) owns all decoding. The GUI never
touches the sample path -- it only ever reads the most recently published frame.
That keeps the numpy work off Qt's thread (which would stall painting) and means
a slow field simply means the *previous* frame stays on screen, which is the
right behaviour for a live view.

Why the decoder consumes a stream rather than a snapshot
--------------------------------------------------------
The FM discriminator is a phase difference, so it carries state from one sample
to the next. Feeding it a fresh overlapping snapshot each time re-does the same
arithmetic and puts a false edge at the seam. :meth:`IQSource.drain_iq` hands
over only newly produced samples, so every sample is demodulated exactly once
and a rolling demodulator buffer is continuous.

Cost, honestly
--------------
At 10 MS/s a field is ~166k samples. Demodulating only the new samples makes a
field cost roughly 4 ms of numpy, plus ~3 ms to detect sync over the rolling
buffer and ~2 ms to rasterise -- comfortably inside the 16.7 ms field budget, so
the decode side can sustain the native field rate on this machine. The display
is then painted by Qt at up to 60 fps, repainting the newest available frame.
Measured rates are exposed through :class:`DecodeStats` rather than assumed, so
the UI can show what is actually happening.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import dsp
from .sdr import IQSource

#: Rolling demodulator window. Must comfortably exceed one field so the
#: rasteriser can always find a complete contiguous run of 240 lines even when
#: the newest line was dropped: 262 lines * ~640 samples = ~168k, so 320k gives
#: roughly a field and a half of slack.
DEFAULT_WINDOW = 320_000


@dataclass
class DecodeStats:
    frames: int = 0
    fields_missed: int = 0
    locked_frames: int = 0
    dropped_iq_samples: int = 0
    drain_ms: float = 0.0
    decode_ms: float = 0.0
    cycle_ms: float = 0.0
    last_frame_at: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    line_rate_hz: float = 0.0
    sync_quality: float = 0.0
    standard: str = ""
    chunk_samples: int = 0

    def frame_rate(self) -> float:
        el = max(1e-6, time.monotonic() - self.started_at)
        return self.frames / el

    def miss_rate(self) -> float:
        total = self.locked_frames + self.fields_missed
        return (self.fields_missed / total) if total else 0.0


@dataclass
class VideoFrame:
    """One published field, ready to paint."""

    image: np.ndarray            # uint8 (H, W)
    timestamp: float
    line_rate_hz: float
    sync_quality: float
    standard: str
    locked: bool

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


class VideoDecoder:
    """Stateful, incremental composite-video decoder.

    Holds the rolling demodulator buffer and the rasteriser. Not thread-safe by
    itself; :class:`DecodeWorker` owns one instance on its own thread.
    """

    def __init__(
        self,
        sample_rate: int,
        width: int = 160,
        out_h: int | None = None,
        window: int = DEFAULT_WINDOW,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.window = int(window)
        self.raster = dsp.Rasteriser(width=width, out_h=out_h)
        self._demod = np.zeros(0, dtype=np.float32)
        self.brightness = 1.0
        self.contrast = 1.0
        self.invert = False
        self.line_filter_hz: float | None = None

    def reset(self) -> None:
        self._demod = np.zeros(0, dtype=np.float32)
        self.raster = dsp.Rasteriser(
            width=self.raster.width, out_h=self.raster.out_h
        )

    def push_iq(self, iq: np.ndarray) -> VideoFrame | None:
        """Demodulate newly arrived samples and emit a frame if one is ready."""
        if iq.size < 2:
            return None
        new = dsp.fm_demodulate(iq)
        if self._demod.size:
            self._demod = np.concatenate((self._demod, new))
        else:
            self._demod = new
        if self._demod.size > self.window:
            self._demod = self._demod[-self.window :]

        if self._demod.size < 32_000:
            return None

        sig = self._demod
        if self.line_filter_hz:
            sig = dsp.limit_video_bandwidth(self._demod, self.sample_rate, self.line_filter_hz)

        sync = dsp.detect_sync(sig, self.sample_rate, polarity="auto")
        if sync.pulse_starts.size < 8 or sync.quality < 0.35:
            return None
        fr = self.raster.push(
            sig, sync, brightness=self.brightness, contrast=self.contrast, invert=self.invert
        )
        if fr is None:
            return None
        return VideoFrame(
            image=fr.image,
            timestamp=time.monotonic(),
            line_rate_hz=fr.line_samples and self.sample_rate / fr.line_samples or 0.0,
            sync_quality=sync.quality,
            standard=fr.spec.name,
            locked=True,
        )


class DecodeWorker:
    """Background thread: drain IQ, decode, publish the newest frame."""

    def __init__(self, source: IQSource, width: int = 160, out_h: int | None = None) -> None:
        self.source = source
        self.decoder = VideoDecoder(source.sample_rate, width=width, out_h=out_h)
        self.stats = DecodeStats()
        self._lock = threading.Lock()
        self._frame: VideoFrame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_decode = 0.0
        self._field_interval = 1.0 / 60.0
        # How much IQ one iteration is allowed to take: one field, plus 5% for
        # jitter. This is the knob that sets the frame rate, and getting it wrong
        # is subtle. The decoder publishes at most *one* frame per iteration but
        # would happily consume several fields' worth, so a cap above one field
        # makes the two rates disagree: at 1.5 fields the loop settled at 40
        # iterations per second, because 40 x 1.5 fields is exactly the 60
        # fields per second the source produces. The backlog drains, but a third
        # of every field is thrown away and never becomes a frame. One field per
        # iteration makes the rates match and the loop runs at the field rate.
        self._max_chunk = int(source.sample_rate * self._field_interval * 1.05)

    # -- frame access (safe from any thread) --------------------------------

    def latest(self) -> VideoFrame | None:
        with self._lock:
            return self._frame

    def reset_stats(self) -> None:
        self.stats = DecodeStats()
        self.decoder.reset()

    def reset(self) -> None:
        """Forget the current frequency's picture.

        Called after a retune. Without it the panel keeps showing the previous
        channel until the new one locks, and a half-decoded frame straddling the
        two frequencies is published in the meantime -- a picture made of two
        different transmitters, which looks like a decoder fault rather than a
        channel change.
        """
        self.decoder.reset()
        with self._lock:
            self._frame = None
        self.stats.locked_frames = 0
        self.stats.sync_quality = 0.0

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="decode", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- main loop ----------------------------------------------------------

    def _run(self) -> None:
        # ``t0`` is the start of the *whole* iteration, not of the decode alone.
        # Pacing against the decode time alone quietly adds the drain on top, so
        # the loop ran at 20 ms/field instead of 16.7 and published 50 fps from
        # a 16.7 ms budget it was already meeting.
        t0 = time.perf_counter()
        deadline = t0
        while not self._stop.is_set():
            iq = self.source.drain_iq(self._max_chunk)
            self.stats.chunk_samples = iq.size
            t_dec = time.perf_counter()
            self.stats.drain_ms = self.stats.drain_ms * 0.9 + (t_dec - t0) * 1000.0 * 0.1
            if iq.size < 64:
                self._stop.wait(0.004)
                t0 = time.perf_counter()
                continue
            try:
                frame = self.decoder.push_iq(iq)
            except Exception:  # keep the pipeline alive on a bad chunk
                frame = None
            self.stats.decode_ms = (
                self.stats.decode_ms * 0.9 + (time.perf_counter() - t_dec) * 1000.0 * 0.1
            )
            self.stats.dropped_iq_samples = self.source.ring.dropped_bytes // 2

            if frame is not None:
                with self._lock:
                    self._frame = frame
                self.stats.frames += 1
                self.stats.locked_frames += 1
                self.stats.last_frame_at = time.monotonic()
                self.stats.line_rate_hz = frame.line_rate_hz
                self.stats.sync_quality = frame.sync_quality
                self.stats.standard = frame.standard
                self._last_decode = time.monotonic()
            elif time.monotonic() - self._last_decode > 0.5:
                # no lock for half a second: count the fields we could not use
                elapsed = time.monotonic() - max(self._last_decode, self.stats.started_at)
                self.stats.fields_missed += max(0, int(elapsed / self._field_interval) - self.stats.locked_frames)
                self._last_decode = time.monotonic()

            # Pace the whole iteration to the field rate; decoding faster than
            # real time is pointless, and the wait is what keeps the ring from
            # growing a backlog.
            #
            # Paced against an *absolute* schedule rather than "sleep the
            # remainder after each pass". A wait overshoots by about a
            # millisecond even at 1 ms timer resolution, and paid on every field
            # that error never washes out: the loop settled at 18.3 ms per field,
            # 55 fps, while actually finishing its work in 15.3. Advancing a
            # fixed deadline and simply not sleeping when it has already passed
            # makes the long-run average right, so a late field is followed by a
            # slightly early one instead of a permanently late stream.
            deadline += self._field_interval
            now = time.perf_counter()
            if deadline < now - self._field_interval:
                deadline = now                    # fell a field behind: resync
            slack = deadline - now
            if slack > 0.0005:
                self._stop.wait(slack)
            now = time.perf_counter()
            self.stats.cycle_ms = self.stats.cycle_ms * 0.9 + (now - t0) * 1000.0 * 0.1
            t0 = now
