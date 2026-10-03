"""TEMP DIAG -- one-second aggregates for the FPV decode path.

**This module is temporary instrumentation and is meant to be deleted.** It
exists to answer one question -- why does a real 5.8 GHz downlink produce no
picture -- and it does that by counting, not by deciding. Nothing here changes
what the decoder computes: the counters are dictionary updates, and the only
work that costs real time (the array reductions in :func:`signal_stats` and
:func:`iq_stats`, and the extra sync statistics in :func:`fpv_rf.dsp.detect_sync`)
runs only when ``FPV_RF_DIAG`` is set. Every call site in ``sdr.py``,
``video.py`` and ``dsp.py`` is marked ``TEMP DIAG``, so removing this file plus
those blocks restores the previous behaviour exactly.

How to read it
--------------
Two independent switches, both off by default, both environment variables so
that no flag has to be threaded through the UI:

``FPV_RF_DIAG=1``
    Print one aggregate line per second on stdout. ``main.py`` tees stdout to
    ``fpv-rf.log`` (deleted at each start), so a live session needs nothing more
    than the variable set and a look at that file. The line is built from four
    sections, in the order the data flows:

    ``iq``
        the drain hand-over: absolute delivered ranges, pending/delivered/skipped
        samples, gaps between hand-overs, ring overrun and overwrite deltas,
        read-size histogram, tuning and stream ids at both ends, spawn epoch,
        reclaim count, recorded ring seams, drain resets and refused tunes.
    ``dec``
        the decoder: samples in, demod/IQ statistics, the sync gate's decision
        *including the frames that never happened*, frames the rasteriser
        declined to draw, frames decoded, boundary steps, decoder resets. The
        sync detail lands here as ``sync_*`` -- threshold, pulse count, quality,
        line rate and polarity, plus the per-polarity statistics that
        :func:`fpv_rf.dsp.detect_sync` gathers when asked.
    ``raster``
        line-run selection: which stage rejected and why, usable count against
        the pigeonhole floor, longest run, inferred period, rows placed on the
        grid instead of at a detection, frames drawn.
    ``worker``
        the decode loop: iterations, buffers and samples, chunk sizes, capped
        skips, gap and overlap samples, dropped seam blocks, blocks refused and
        why, segments started, decode errors, frames published or withheld as
        duplicate or stale, frame age and generation.

``FPV_RF_DIAG_DUMP=<path-prefix>``
    Also write a bounded rolling capture to disk: ``<prefix>_NN_iq.u8`` (the raw
    interleaved bytes exactly as received, *before* scaling),
    ``<prefix>_NN_demod.f32`` (the demodulator's float32 output) and
    ``<prefix>_NN.json`` (sample rate, encoding, frequency, tuning ids at both
    ends of the range, spawn epoch, absolute write range and the counters at
    that moment). The raw half is a snapshot of the IQ ring taken at the flush
    deadline, not a concatenation of drain hand-overs: a hand-over is capped,
    skips a backlog and starts wherever the cursor was, so several of them make a
    file of non-adjacent stretches of the stream that reads as one recording.
    Tuned by ``FPV_RF_DIAG_DUMP_SAMPLES`` (window, default 1M complex samples =
    2 MB, half the ring), ``FPV_RF_DIAG_DUMP_FILES`` (default 4) and
    ``FPV_RF_DIAG_DUMP_FLUSH_S`` (default 5 s). The window is bounded in memory
    and the file count is bounded on disk, so an enabled dump cannot fill a disk
    on a long session.

``FPV_RF_DIAG_PUBCAP=<n>`` (with ``FPV_RF_DIAG_DUMP`` set for the prefix)
    Capture what a *published* picture was made of, instead of what the stream
    happened to contain. The periodic dump above is a snapshot of the ring at a
    deadline: it is contiguous, but it is aligned to nothing -- not to the
    decoder's window, not to the block the discriminator used, not to the frame
    that came out. So a capture and a replayed decode of it are two different
    experiments, and a disagreement between them cannot be attributed to
    anything. This one is taken at the publication decision itself and contains,
    per publication:

    * ``<prefix>_pubNN_iq.u8`` -- the exact accepted IQ supporting the decoder's
      window, *including the discriminator's predecessor sample*. The FM
      discriminator is a difference between consecutive samples, so output ``j``
      is the step between IQ ``j`` and ``j+1``: reproducing a window of ``n``
      outputs needs ``n+1`` samples, and the extra one is the sample *before* the
      block, held by the decoder specifically because it cannot be recomputed.
      Without it the boundary step is unrecoverable and a replay of the capture
      differs from the live decode by one sample per block boundary.
    * ``<prefix>_pubNN_demod.f32`` -- the demodulator's actual float32 window,
      the one sync detection and the rasteriser both read. Band-limiting, if any
      was applied, is recorded rather than folded in, so the same bytes can be
      re-filtered.
    * ``<prefix>_pubNN.png`` -- the published picture, uint8, written with
      ``zlib`` alone. ``fpv_rf`` never imports Pillow and the packaged build
      excludes it, so a capture that depended on it would not exist in the
      application it exists to diagnose.
    * ``<prefix>_pubNN.json`` -- absolute sample ranges, the segment, the stream
      and tuning identity at each delivery boundary, every reset and dropped
      backlog inside the window, **both** sync polarities (threshold, raw and
      clustered counts, density, spacing min/max/mean/median, quality, which one
      was selected), the selected line positions and how many of them were
      reconstructed on the grid rather than detected, and the published frame's
      own provenance (its IQ interval, segment and generation).

    The raw half is **not** a concatenation. Blocks that do not abut inside the
    window are written in stream order with every discontinuity listed in the
    JSON -- the absolute sample each span starts at, and every hole or overlap
    between two of them -- so a reader can place every byte and can tell at once
    whether the file is one continuous stretch. The decoder resets its window at
    any discontinuity, so the honest answer is normally "no gaps"; the check is
    there because that is the claim being made.

    Bounded three ways, all of them necessary: ``FPV_RF_DIAG_PUBCAP`` is the
    number of publications kept on disk (default 3, written as a rolling
    ``pub00``/``pub01``/``pub02`` set), the decoder keeps at most one window's
    worth of IQ references in memory while the capture is on, and captures stop
    for good once ``FPV_RF_DIAG_DUMP_FILES`` has been spent -- *shared* with the
    periodic dump, so the two together cannot exceed the budget that was
    documented. Off unless set, and the decode is bit-identical with it off: the
    only difference is that ``detect_sync`` is asked to fill in a record it would
    otherwise skip, which it does with the live early-break honoured.

    Cost, because it is not small: one capture writes about 2.5 MB and takes
    tens of milliseconds, on the thread that publishes frames. That is why it is
    bounded by a *count* as well as by a file set.

Why counters and not logging
----------------------------
Per-sample logging of a 10 MS/s stream is useless -- it cannot be read, and it
changes the timing it is measuring. Everything here is a total or a bounded
histogram, sampled once per field at most, and emitted once per second.
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Set to anything truthy to enable the per-second aggregate line.
ENV_DIAG = "FPV_RF_DIAG"
#: Path prefix for the bounded raw/demod capture. Off unless set.
ENV_DUMP = "FPV_RF_DIAG_DUMP"
ENV_DUMP_SAMPLES = "FPV_RF_DIAG_DUMP_SAMPLES"
ENV_DUMP_FILES = "FPV_RF_DIAG_DUMP_FILES"
ENV_DUMP_FLUSH_S = "FPV_RF_DIAG_DUMP_FLUSH_S"
#: Publications to keep on disk, and the switch that turns the
#: publication-matched capture on. A bare "1"/"on" means the default.
ENV_PUBCAP = "FPV_RF_DIAG_PUBCAP"

#: One line per second: fine enough to see a fault develop, coarse enough that
#: the log stays readable and the decode loop pays nothing for it.
LOG_PERIOD_S = 1.0

#: Rolling capture bounds. 1M complex samples is 2 MB of raw IQ -- half of the
#: 4 MiB ring, so the raw half of a segment can be snapshotted out of the ring
#: whole and in one contiguous run -- and 100 ms at 10 MS/s, which covers the
#: ~24 ms field a rasteriser works on several times over. The demod half of the
#: same window is 1M float32, i.e. 4 MB.
DUMP_SAMPLES = 1_000_000
DUMP_FILES = 4
DUMP_FLUSH_S = 5.0

#: Publications kept on disk by ``FPV_RF_DIAG_PUBCAP``. Three is enough to see
#: that a frame which repeats has the same geometry as the one before it and one
#: that does not have changed, and small enough that the whole feature is bounded
#: at 4 files x 3 = 12 regardless of how long the session runs.
PUBCAP_FILES = 3

#: A demodulator step this close to pi is where the phase difference wraps: a
#: wideband signal sampled well below twice its deviation cannot be recovered,
#: and the symptom is a demod output that looks plausible and is not.
NEAR_PI = 0.95
#: |I| or |Q| at or above this is a full-scale sample, i.e. a clipped one.
CLIP_ABS = 0.995
#: |x| below this is effectively a dead sample: DC, weak signal or a dropout.
LOW_MAG = 0.01
#: Percentiles are estimated from a strided subsample at most this large. An
#: exact partition of a 320k window costs more than it is worth for a number
#: that only has to resolve the top of the distribution.
STAT_MAX = 1 << 16

_T0 = time.monotonic()


def _truthy(name: str) -> bool:
    """Whether ``name`` is set to something other than an explicit "off"."""
    return os.environ.get(name, "").strip().lower() not in ("", "0", "no", "off", "false")


def enabled() -> bool:
    """Whether the per-block array statistics are wanted.

    Off by default: unlike the counters, these are real reductions over a
    320k-sample window, and they must not be able to push the field over its
    budget on a machine that is already close to it.
    """
    return _truthy(ENV_DIAG)


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(str(os.environ.get(name, "")).strip()))
    except (TypeError, ValueError):
        return int(default)


# --------------------------------------------------------------------------
# Array statistics (only called when FPV_RF_DIAG is set)
# --------------------------------------------------------------------------


def signal_stats(x: np.ndarray) -> dict[str, float]:
    """Distribution of a demodulated block, and how close it gets to wrapping.

    ``nearpi`` is the share of samples whose phase step is within
    ``1 - NEAR_PI`` of pi. It is the number that distinguishes "the deviation
    exceeds what the sample rate can resolve" from "the threshold is in the
    wrong place", and the two look identical from the outside -- both produce
    noise and a plausible-looking sync score.
    """
    a = np.asarray(x)
    if a.size == 0:
        return {}
    f = a.astype(np.float32, copy=False)
    sub = f if f.size <= STAT_MAX else f[:: f.size // STAT_MAX + 1]
    p50, p99, p999 = (float(v) for v in np.percentile(sub, [50.0, 99.0, 99.9]))
    return {
        "n": int(f.size),
        "min": float(f.min()),
        "max": float(f.max()),
        "mean": float(f.mean()),
        "std": float(f.std()),
        "p50": p50,
        "p99": p99,
        "p999": p999,
        "nearpi": float(np.count_nonzero(np.abs(sub) >= NEAR_PI * np.pi)) / float(sub.size),
    }


def iq_stats(iq: np.ndarray) -> dict[str, float]:
    """DC offset, level and pathology of a block of complex samples.

    ``clip`` and ``lowmag`` are the two ends of the same question -- is the
    front end saturated, or is there barely anything there. ``dc_i``/``dc_q``
    catch the other classic: a recording read in the wrong byte convention
    looks like a large offset on one axis rather than like an inversion.
    """
    z = np.asarray(iq)
    if z.size == 0:
        return {}
    re = z.real
    im = z.imag
    mag = np.abs(z)
    clipped = (np.abs(re) >= CLIP_ABS) | (np.abs(im) >= CLIP_ABS)
    return {
        "n": int(z.size),
        "dc_i": float(re.mean()),
        "dc_q": float(im.mean()),
        "mag": float(mag.mean()),
        "mag_max": float(mag.max()),
        "clip": float(np.count_nonzero(clipped)) / float(z.size),
        "lowmag": float(np.count_nonzero(mag < LOW_MAG)) / float(z.size),
    }


# --------------------------------------------------------------------------
# Counters
# --------------------------------------------------------------------------


def _fmt(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, np.integer)):
        return str(int(value))
    f = float(value)
    if not np.isfinite(f):
        return str(f)
    a = abs(f)
    if a >= 1000.0:
        return f"{f:.0f}"
    if a >= 1.0:
        return f"{f:.3f}"
    return f"{f:.4g}"


class Diag:
    """Totals, last-known values and bounded histograms for one stage.

    Sections are process-wide and shared by name (see :func:`section`), so a
    counter survives the object rebuilds a reset causes -- a new
    :class:`~fpv_rf.sdr.IQRing` or rasteriser on the same channel keeps adding
    to the same totals instead of restarting them, which is what makes a
    one-second line comparable across a retune.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._c: dict[str, float] = {}
        self._g: dict[str, object] = {}
        self._h: dict[str, Counter] = {}

    # -- writers -----------------------------------------------------------

    def bump(self, key: str, n: float = 1) -> None:
        """Add ``n`` to a running total."""
        if not n:
            return
        with self._lock:
            self._c[key] = self._c.get(key, 0) + n

    def gauge(self, key: str, value: object) -> None:
        """Record a last-known value. Strings are allowed, for reasons."""
        with self._lock:
            self._g[key] = value

    def put(self, prefix: str, values: dict[str, float]) -> None:
        """Merge a dictionary of statistics in as gauges named ``prefix_*``."""
        if not values:
            return
        with self._lock:
            for k, v in values.items():
                self._g[f"{prefix}_{k}"] = v

    def hist(self, key: str, bucket: float) -> None:
        """Count one occurrence in a fixed bucket."""
        with self._lock:
            self._h.setdefault(key, Counter())[int(bucket)] += 1

    def observe(self, key: str, value: float) -> None:
        """Track count/sum/min/max of a quantity, for its mean and spread."""
        value = float(value)
        with self._lock:
            c = self._c
            c[f"{key}_n"] = c.get(f"{key}_n", 0) + 1
            c[f"{key}_sum"] = c.get(f"{key}_sum", 0.0) + value
            lo = c.get(f"{key}_min")
            hi = c.get(f"{key}_max")
            if lo is None or value < lo:
                c[f"{key}_min"] = value
            if hi is None or value > hi:
                c[f"{key}_max"] = value

    # -- readers -----------------------------------------------------------

    def value(self, key: str) -> float:
        with self._lock:
            return float(self._c.get(key, 0))

    def mean(self, key: str) -> float:
        with self._lock:
            n = float(self._c.get(f"{key}_n", 0.0))
            return self._c.get(f"{key}_sum", 0.0) / n if n else 0.0

    def gauge_value(self, key: str) -> object:
        with self._lock:
            return self._g.get(key)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": dict(self._c),
                "gauges": dict(self._g),
                "histograms": {k: dict(v) for k, v in self._h.items()},
            }

    def clear(self) -> None:
        with self._lock:
            self._c.clear()
            self._g.clear()
            self._h.clear()

    def line(self, max_buckets: int = 12) -> str:
        with self._lock:
            c = dict(self._c)
            g = dict(self._g)
            h = {k: dict(v) for k, v in self._h.items()}
        parts = [f"{k}={_fmt(v)}" for k, v in sorted(c.items())]
        parts += [f"{k}={_fmt(v)}" for k, v in sorted(g.items())]
        for k, v in sorted(h.items()):
            top = sorted(v.items())[:max_buckets]
            parts.append(k + "={" + ",".join(f"{kk}:{vv}" for kk, vv in top) + "}")
        return f"{self.name}: " + " ".join(parts)


_SECTIONS: dict[str, Diag] = {}
_REGISTRY_LOCK = threading.Lock()


def section(name: str) -> Diag:
    """The process-wide :class:`Diag` for ``name``, created on first use."""
    with _REGISTRY_LOCK:
        d = _SECTIONS.get(name)
        if d is None:
            d = Diag(name)
            _SECTIONS[name] = d
        return d


def report() -> str:
    """Every section as one string. Cheap enough to call once per second."""
    with _REGISTRY_LOCK:
        secs = list(_SECTIONS.values())
    return " | ".join(s.line() for s in secs)


def reset() -> None:
    """Clear every section. For tools, between cases."""
    with _REGISTRY_LOCK:
        secs = list(_SECTIONS.values())
    for s in secs:
        s.clear()


def _echo(text: str) -> None:
    """Print to the app's log, and never raise into the caller.

    A diagnostic that can take the decoder down is worse than no diagnostic,
    so every failure here is swallowed.
    """
    try:
        print(text, flush=True)
    except Exception:
        pass


_log_lock = threading.Lock()
_last_log = 0.0
_announced = False


def maybe_log(force: bool = False) -> str | None:
    """Emit the aggregate line if ``LOG_PERIOD_S`` has passed.

    Returns the line, or None if nothing was due. Called from the decode loop,
    so the rate limit is what keeps this off the critical path.
    """
    if not enabled():
        return None
    global _last_log, _announced
    now = time.monotonic()
    with _log_lock:
        if not force and now - _last_log < LOG_PERIOD_S:
            return None
        _last_log = now
        first = not _announced
        _announced = True
        line = f"DIAG t={now - _T0:.1f}s {report()}"
    if first:
        _echo(
            f"DIAG enabled: {ENV_DIAG}=1 every {LOG_PERIOD_S:g}s "
            f"(dump: {dump_prefix() or 'off'}) -- stdout is teed to fpv-rf.log"
        )
    _echo(line)
    return line


# --------------------------------------------------------------------------
# Bounded raw/demod capture
# --------------------------------------------------------------------------


def dump_enabled() -> bool:
    return _truthy(ENV_DUMP)


def dump_prefix() -> str:
    return os.environ.get(ENV_DUMP, "").strip()


class _Dump:
    """Rolling window of demodulated samples plus raw IQ from the ring.

    Two halves, deliberately separate. The demodulator's output is offered block
    by block (:meth:`add`) and accumulated into a rolling window, because it is
    produced downstream of the ring and exists nowhere else. The raw IQ is *not*
    accumulated: at the flush deadline it is snapshotted out of the ring itself
    (:meth:`tick`), newest first, in one contiguous run.

    That used to be the other way round, and the difference is the whole point.
    The old capture was assembled from the drain hand-overs, each of which is
    capped at the caller's ``max_samples``, begins wherever the drain cursor
    happened to be, and silently skips a backlog when the decoder falls behind.
    Concatenating those yields a file whose bytes are in stream order but are
    not adjacent in the stream -- gaps the metadata never mentions, presented as
    one continuous recording. A ring snapshot cannot have that defect: the newest
    ``n`` bytes of a ring are one run by construction, and the absolute range and
    tuning at each end of it are read under the same lock as the bytes.

    Memory is bounded by ``FPV_RF_DIAG_DUMP_SAMPLES`` and disk by
    ``FPV_RF_DIAG_DUMP_FILES``. The two buffers are not sample-aligned -- a
    block of IQ demodulates to one fewer sample, and a block too short to
    demodulate produces none at all -- so the exact count of each is written
    into the JSON beside them rather than assumed equal.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_state()

    def _reset_state(self) -> None:
        self._demod: list[np.ndarray] = []
        self._meta: dict = {}
        self._demod_n = 0
        self._file_no = 0
        #: TEMP DIAG: publications captured, and the file sets spent between the
        #: periodic dump and the publication capture. One counter, because the
        #: documented budget is one budget: two writers each holding their own
        #: ``DUMP_FILES`` would write twice the files the README promises cannot
        #: happen.
        self._pub_no = 0
        self._spent = 0
        #: Metadata of the last publications, so a tool can compare them without
        #: reading the files back. The arrays stay out of it on purpose -- the
        #: bound has to be about bytes, not about how convenient it is to keep
        #: them.
        self._pub_recent: list[dict] = []
        self._next = time.monotonic() + _env_int(ENV_DUMP_FLUSH_S, DUMP_FLUSH_S)
        self._done = False
        self._warned = False
        self._ignored_warned = False
        #: Transfer-process generation of the last segment written, so a segment
        #: that follows a restart can be labelled rather than passed off as a
        #: continuation of the one before it.
        self._epoch: int | None = None

    def add(self, kind: str, data, meta: dict | None = None) -> None:
        """Offer one demodulator block. The raw-IQ half is not taken from here."""
        try:
            with self._lock:
                if self._done:
                    return
                if kind != "demod":
                    # The ring is the only source of raw IQ now, and stitching
                    # blocks here is the defect this replaced. Say so once
                    # rather than silently writing a file nobody can trust.
                    if not self._ignored_warned:
                        self._ignored_warned = True
                        _echo(f"DIAG dump: ignoring a {kind!r} block; raw IQ is "
                              "snapshotted from the ring at the flush deadline")
                    return
                limit = _env_int(ENV_DUMP_SAMPLES, DUMP_SAMPLES)
                arr = np.asarray(data, dtype=np.float32).ravel()
                self._demod.append(arr)
                excess = (self._demod_n + arr.size) - limit
                while excess > 0 and self._demod:
                    head = self._demod[0]
                    if head.size <= excess:
                        excess -= head.size
                        self._demod.pop(0)
                    else:
                        self._demod[0] = head[excess:]
                        excess = 0
                self._demod_n = sum(a.size for a in self._demod)
                if meta:
                    # Facts about the demodulator accumulate across the adds that
                    # make up one segment. The ring snapshot supplies the rest --
                    # encoding, frequency, absolute range, tuning -- from the
                    # instant the segment is written, which is the only place
                    # they are all true of the same bytes.
                    self._meta.update(meta)
        except Exception as exc:                       # never into the caller
            self._fail(exc)

    def tick(self, snapshot) -> None:
        """Called on every read; writes a segment once the deadline has passed.

        ``snapshot`` is called only when a segment is actually due, and it is
        the *only* thing that writes one, so the raw IQ and the demodulated
        samples in a segment come from one instant rather than from whichever
        call happened to cross the deadline first.
        """
        try:
            with self._lock:
                if self._done or time.monotonic() < self._next:
                    return
                self._flush(*snapshot(_env_int(ENV_DUMP_SAMPLES, DUMP_SAMPLES)))
        except Exception as exc:                       # never into the caller
            self._fail(exc)

    def _fail(self, exc: BaseException) -> None:
        self._done = True
        if not self._warned:
            self._warned = True
            _echo(f"DIAG dump disabled after an error: {type(exc).__name__}: {exc}")

    def _flush(self, iq: bytes = b"", iq_meta: dict | None = None) -> None:
        if not iq and not self._demod:
            self._next = time.monotonic() + _env_int(ENV_DUMP_FLUSH_S, DUMP_FLUSH_S)
            return
        stem = f"{dump_prefix()}_{self._file_no:02d}"
        demod = (
            np.concatenate(self._demod).astype(np.float32, copy=False)
            if self._demod
            else np.zeros(0, dtype=np.float32)
        )
        # Caller-supplied metadata first, then this file's own facts about what
        # it actually holds, which are the ones an offline reader must trust.
        # The ring snapshot's facts go in between, so a demod-side key of the
        # same name cannot overwrite the range the bytes really came from.
        info = dict(self._meta)
        info.update(iq_meta or {})
        info.update({
            "iq_samples": len(iq) // 2,
            "demod_samples": int(demod.size),
            "iq_source": "ring_snapshot" if iq else "absent",
            "file": int(self._file_no),
            "counts": report(),
        })
        # Anything that stops the file being one continuous capture of one
        # tuning from one transfer process has to be in the file, because
        # nothing in the samples can say so.
        notes: list[str] = []
        if iq and info.get("spans_tune"):
            info["continuous"] = False
            notes.append(f"SPANS A RETUNE ({info.get('first_tune_id')} -> "
                         f"{info.get('last_tune_id')})")
        epoch = info.get("epoch")
        if iq and epoch is not None and self._epoch is not None and epoch != self._epoch:
            # The ring outlives the transfer process, so a snapshot can hold
            # samples from two of them under one tuning id. Nothing else in the
            # file distinguishes that case, so it is recorded here.
            info["continuous"] = False
            info["after_process_restart"] = {"previous_epoch": self._epoch,
                                             "epoch": epoch}
            notes.append(f"AFTER A TRANSFER-PROCESS RESTART ({self._epoch} -> {epoch})")
        if epoch is not None:
            self._epoch = epoch
        # Written outside any ring lock: the snapshot already holds its own copy.
        if iq:
            Path(f"{stem}_iq.u8").write_bytes(iq)
        if demod.size:
            demod.tofile(f"{stem}_demod.f32")
        Path(f"{stem}.json").write_text(json.dumps(info, indent=1, default=str))
        seam = f"  -- {', '.join(notes)}: not one continuous capture" if notes else ""
        _echo(
            f"DIAG dump {stem}: {info['iq_samples']} IQ samples, "
            f"{demod.size} demod samples{seam}"
        )
        self._demod = []
        self._meta = {}
        self._demod_n = 0
        self._file_no += 1
        self._next = time.monotonic() + _env_int(ENV_DUMP_FLUSH_S, DUMP_FLUSH_S)
        self._spend()

    # TEMP DIAG: one file budget, shared with the publication capture. The
    # periodic dump existed first and had the check inline; the publication
    # capture needed the same limit to be meaningful, and a second counter would
    # have meant the two together wrote twice the files the documentation says
    # they cannot. Spending is counted in *file sets*, as before, so the default
    # of four still means four whatever is producing them.
    def _spend(self) -> None:
        self._spent += 1
        if self._spent >= _env_int(ENV_DUMP_FILES, DUMP_FILES):
            self._done = True

    def reset(self) -> None:
        with self._lock:
            self._reset_state()


_DUMP = _Dump()


def dump_add(kind: str, data, meta: dict | None = None) -> None:
    """Offer one demodulator block to the bounded capture. No-op unless enabled."""
    if not dump_enabled():
        return
    _DUMP.add(kind, data, meta)


def dump_tick(snapshot) -> None:
    """Give the bounded capture its chance to write a segment.

    ``snapshot(samples)`` must return ``(bytes, metadata)`` for the newest
    ``samples`` complex samples in the ring, read without consuming them. It is
    called only when the flush deadline has passed, and only while the dump is
    enabled -- with the environment variable unset this is one dictionary lookup
    and a return.
    """
    if not dump_enabled():
        return
    _DUMP.tick(snapshot)


# --------------------------------------------------------------------------
# TEMP DIAG: publication-matched capture
# --------------------------------------------------------------------------
# The switch and the on-disk count live in the constants block at the top, with
# the periodic dump's, so that the two budgets can be read side by side.


def pubcap_enabled() -> bool:
    """Whether the publication-matched capture is wanted.

    Off unless ``FPV_RF_DIAG_PUBCAP`` is set to something other than an explicit
    "off", and needs ``FPV_RF_DIAG_DUMP`` as well, since that is where the file
    prefix comes from.

    The switch is read **first**, so with the feature off this is one
    ``os.environ`` lookup and a return -- the same cost as the periodic dump's own
    test, and it is called per field. The prefix is only consulted when the
    capture is actually on, because looking it up first would make the common
    case pay for a switch it is not using.
    """
    if not _truthy(ENV_PUBCAP):
        return False
    return bool(dump_prefix())


def pubcap_files() -> int:
    """How many publications are kept on disk.

    A bare ``1`` -- the spelling the switch is turned *on* with, and the one a
    user reads as "yes" -- means the default count rather than one publication.
    Reading it literally would quietly turn "keep the last few" into "keep the
    last one", and one publication cannot answer the question the capture exists
    for: whether a frame that repeats is the same field drawn again or a new one.
    An explicit ``N`` above 1 is read as ``N``.
    """
    raw = os.environ.get(ENV_PUBCAP, "").strip().lower()
    if raw in ("", "1", "on", "yes", "true"):
        return PUBCAP_FILES
    return max(1, _env_int(ENV_PUBCAP, PUBCAP_FILES))


def iq_to_u8(iq: np.ndarray, encoding: str = "signed-int8") -> bytes:
    """Interleaved int8/uint8 bytes for a block of complex samples.

    The inverse of :func:`fpv_rf.sdr.u8_to_iq`, and exact for both encodings:
    that function multiplies an integer byte by ``1/127.5``, so multiplying by
    ``127.5`` and rounding recovers the integer that was multiplied.
    ``sdr.u8_to_iq`` on this file therefore returns the samples the decoder held.
    Not a claim about the *original* bytes -- those no longer exist by the time
    the decoder has samples -- but about the samples, which is what the capture
    has to be matched against.

    Deliberately re-encodes rather than capturing the ring's bytes a second
    time. The ring snapshot is a *different* stream position from the one the
    picture was drawn from, and a capture whose IQ does not correspond to the
    picture beside it cannot answer the question it was taken to answer.
    """
    a = np.asarray(iq)
    if a.size == 0:
        return b""
    if encoding not in ("signed-int8", "unsigned-offset"):
        raise ValueError(f"unknown IQ encoding {encoding!r}")
    # Round first, clip after: clipping first would turn an out-of-range sample
    # into a different out-of-range sample rather than the nearest representable
    # one, and the difference is the whole point of doing this in float64.
    re = np.rint(np.asarray(a.real, dtype=np.float64) * 127.5)
    im = np.rint(np.asarray(a.imag, dtype=np.float64) * 127.5)
    if encoding == "unsigned-offset":
        # Clip to the range the *stored* byte lives in, which is the shifted one:
        # ``u8_to_iq`` subtracts 128 before scaling, so a sample at the top of the
        # signed range has to be allowed to reach 255. Clipping at 127 here would
        # fold every sample above -1/127.5 -- that is, almost all of them -- onto
        # one value, and the capture would decode to a constant.
        lo, hi, dtype = 0.0, 255.0, np.uint8
        re, im = re + 128.0, im + 128.0
    else:
        lo, hi, dtype = -128.0, 127.0, np.int8
    out = np.empty((2, a.size), dtype=dtype)
    out[0] = np.clip(re, lo, hi)
    out[1] = np.clip(im, lo, hi)
    return out.T.tobytes()


def png_bytes(image: np.ndarray) -> bytes:
    """An 8-bit greyscale PNG, from ``zlib`` and nothing else.

    The published picture is uint8 one byte per pixel, which is exactly greyscale
    PNG with no filtering -- so this is a few lines rather than a dependency.
    ``fpv_rf`` never imports Pillow and the packaged build excludes it, so using
    it here would mean the capture does not exist in the application the capture
    is diagnosing.
    """
    a = np.ascontiguousarray(image, dtype=np.uint8)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D image, got {a.ndim}-D")
    height, width = int(a.shape[0]), int(a.shape[1])
    # One filter byte (0 = None) in front of every row.
    raw = np.empty((height, width + 1), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = a
    return b"".join((
        b"\x89PNG\r\n\x1a\n",
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)),
        _png_chunk(b"IEND", b""),
    ))


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return b"".join((
        struct.pack(">I", len(payload)),
        tag,
        payload,
        struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF),
    ))


@dataclass
class IqSpan:
    """One block of IQ the decoder accepted, and where it sat in the stream.

    The ledger is a list of these, not a buffer, because the decoder's window is
    a *sequence* of blocks and the only honest way to reproduce what it saw is to
    keep them individually labelled. A flat buffer would present a window that
    spans a gap, a retune or a process restart as one continuous stretch of
    samples, which is the specific defect the previous dump was fixed for and the
    specific thing this capture must not reintroduce.
    """

    iq: np.ndarray
    first_sample: int              # absolute index of iq[0]; -1 when unplaced
    stream_id: int = -1
    tune_id: int = -1
    segment: int = 0
    delivery: int = -1
    #: The delivery that produced this block refused to join the one before it,
    #: so its samples begin a new segment. Named, because it is the boundary a
    #: reader has to break the file at.
    starts_segment: bool = False
    #: The discriminator's predecessor sample, kept with the block rather than
    #: with the window. Output ``j`` is the step from IQ ``j`` to IQ ``j+1``, so
    #: reproducing a window of ``n`` outputs needs ``n+1`` samples and the extra
    #: one is always the sample *before* the block. Without it the boundary step
    #: is unrecoverable and a replay of the capture differs from the live decode
    #: by one wrong sample per block boundary -- which is exactly the kind of
    #: small, plausible, wrong difference this capture exists to rule out.
    predecessor: "np.complex64 | None" = None
    reset_reason: str = ""
    skipped_samples: int = 0

    @property
    def samples(self) -> int:
        return int(self.iq.size)

    @property
    def last_sample(self) -> int:
        return int(self.first_sample + self.samples) if self.first_sample >= 0 else -1

    def as_dict(self) -> dict:
        return {
            "first_sample": int(self.first_sample),
            "last_sample": int(self.last_sample),
            "samples": self.samples,
            "stream_id": int(self.stream_id),
            "tune_id": int(self.tune_id),
            "segment": int(self.segment),
            "delivery": int(self.delivery),
            "starts_segment": bool(self.starts_segment),
            "has_predecessor": self.predecessor is not None,
            "reset_reason": self.reset_reason,
            "skipped_samples": int(self.skipped_samples),
        }


class IqLedger:
    """Bounded, ordered record of the blocks behind one decoder window.

    Append-only within a window and trimmed from the front, so the oldest sample
    is the first to go: the window the rasteriser draws from is the *newest*
    part of the stream, and a capture that dropped its head would be missing the
    lines at the top of the picture.

    Every span knows its own absolute range and identity, so the window can be
    described exactly -- including the fact that a window spanning a
    discontinuity is *several* streams' worth of bytes and not one recording.
    """

    def __init__(self, limit: int = 1 << 20) -> None:
        self._spans: list[IqSpan] = []
        self._samples = 0
        self.limit = max(2, int(limit))

    def add(self, span: IqSpan) -> None:
        self._spans.append(span)
        self._samples += span.samples
        self._trim()

    def _trim(self) -> None:
        while self._spans and self._samples > self.limit:
            oldest = self._spans.pop(0)
            self._samples -= oldest.samples
            #: Never drop the predecessor along with the block it belongs to: it
            #: is the sample the *next* span's boundary step was computed from.
            if oldest.predecessor is not None and self._spans:
                self._spans[0].predecessor = oldest.predecessor

    def clear(self) -> None:
        self._spans.clear()
        self._samples = 0

    def __len__(self) -> int:
        return len(self._spans)

    @property
    def samples(self) -> int:
        return self._samples

    def spans(self) -> tuple[IqSpan, ...]:
        return tuple(self._spans)

    def slice(self, first_sample: int, count: int) -> list[IqSpan]:
        """The spans covering ``[first_sample, first_sample + count)``.

        Clipped at both ends, so the result can start or end part-way through a
        block. Returns every span that overlaps the range, in stream order.
        """
        out: list[IqSpan] = []
        last = int(first_sample) + int(count)
        for span in self._spans:
            if span.first_sample < 0:
                out.append(span)          # unplaced: cannot be clipped, take it whole
                continue
            if span.last_sample <= first_sample or span.first_sample >= last:
                continue
            out.append(span)
        return out


class PubCap:
    """One publication's worth of everything a replay would need.

    Built on the decode thread from values the decoder already holds, then written
    outside every lock. Holding the decoder lock across a 2.5 MB write would stall
    the next field for longer than a field takes; taking a snapshot and copying it
    afterwards costs the same bytes and nothing else.
    """

    def __init__(self) -> None:
        self.frame: object | None = None        # video.VideoFrame
        self.image: np.ndarray | None = None
        self.demod: np.ndarray | None = None
        self.demod_origin: int = -1
        self.spans: tuple[IqSpan, ...] = ()
        self.sync: dict | None = None
        self.positions: np.ndarray | None = None
        self.lines_from_grid: int = 0
        self.lines_used: int = 0
        self.lines_detected: int = 0
        self.generation: int = 0
        self.segment: int = 0
        self.sample_rate: int = 0
        self.encoding: str = "signed-int8"
        self.source_kind: str = ""
        self.frequency_hz: int = 0
        self.line_filter_hz: float | None = None
        self.stats: dict = field(default_factory=dict)

    # -- description -------------------------------------------------------

    def ranges(self) -> tuple[list[dict], list[dict]]:
        """``(spans, gaps)`` for the window, in stream order.

        The gaps are the point. Two spans that abut are one run; two that do not
        are two, and the bytes between them are missing -- not zero, not
        repeated, missing. Listing every discontinuity with its size is what stops
        a reader treating the file as one continuous recording, which is the
        defect that made the first version of this capture unreadable.
        """
        described = [s.as_dict() for s in self.spans]
        gaps: list[dict] = []
        for a, b in zip(self.spans, self.spans[1:]):
            if a.first_sample < 0 or b.first_sample < 0:
                gaps.append({"after_span": a.as_dict()["first_sample"],
                             "reason": "position_unknown"})
                continue
            delta = int(b.first_sample) - int(a.last_sample)
            if delta > 0:
                gaps.append({"after_sample": int(a.last_sample),
                             "before_sample": int(b.first_sample),
                             "missing_samples": int(delta),
                             "reason": "gap"})
            elif delta < 0:
                gaps.append({"after_sample": int(a.last_sample),
                             "before_sample": int(b.first_sample),
                             "overlap_samples": int(-delta),
                             "reason": "overlap"})
            elif a.tune_id != b.tune_id:
                gaps.append({"after_sample": int(a.last_sample),
                             "reason": "retune",
                             "tune_id": [int(a.tune_id), int(b.tune_id)]})
            elif a.stream_id != b.stream_id:
                gaps.append({"after_sample": int(a.last_sample),
                             "reason": "stream_restart",
                             "stream_id": [int(a.stream_id), int(b.stream_id)]})
        return described, gaps

    def iq_array(self) -> np.ndarray:
        """The window's IQ as one array, in stream order.

        Concatenated **only** to write a file whose per-span offsets are recorded
        beside it. The array is not a continuous recording and the JSON says so;
        what it is for is being fed back through the same decoder, where the
        discontinuity lands as a reset rather than as a phase step across samples
        that never existed.
        """
        if not self.spans:
            return np.zeros(0, dtype=np.complex64)
        return np.concatenate([np.asarray(s.iq, dtype=np.complex64) for s in self.spans])

    def as_dict(self) -> dict:
        """Everything in the JSON except the counters and the identity fields."""
        fr = self.frame
        spans, gaps = self.ranges()
        total = sum(int(s["samples"]) for s in spans)
        unplaced = any(s["first_sample"] < 0 for s in spans)
        unmatched = not spans and bool(self.demod is not None and self.demod.size)
        return {
            "demod_samples": int(self.demod.size) if self.demod is not None else 0,
            "demod_first_step": int(self.demod_origin),
            "iq_samples": int(total),
            "iq_spans": spans,
            "iq_gaps": gaps,
            # The claim a reader needs in order to trust the file as a recording:
            # true only when every span abuts its neighbour, carries one identity
            # and is placed in the stream. Derived here rather than in the writer,
            # so a caller reading :class:`PubCap` directly and a reader of the JSON
            # cannot be told two different things about the same window.
            "continuous": not gaps and not unplaced and not unmatched,
            "iq_unplaced": unplaced,
            "iq_unmatched": unmatched,
            "line_filter_hz": self.line_filter_hz,
            "sync": self.sync or {},
            "lines_used": int(self.lines_used),
            "lines_detected": int(self.lines_detected),
            "lines_from_grid": int(self.lines_from_grid),
            "line_positions": ([float(p) for p in self.positions]
                               if self.positions is not None else []),
            "image": ({"width": int(self.image.shape[1]),
                       "height": int(self.image.shape[0]),
                       "dtype": str(self.image.dtype)} if self.image is not None
                      else {}),
            "frame": {
                "iq_first": int(getattr(fr, "iq_first", -1)),
                "iq_last": int(getattr(fr, "iq_last", -1)),
                "segment": int(getattr(fr, "segment", -1)),
                "generation": int(getattr(fr, "generation", 0)),
                "line_rate_hz": float(getattr(fr, "line_rate_hz", 0.0)),
                "sync_quality": float(getattr(fr, "sync_quality", 0.0)),
                "standard": str(getattr(fr, "standard", "")),
                "timestamp": float(getattr(fr, "timestamp", 0.0)),
            },
        }


class _Pub:
    """Rolling set of the last N publications on disk, sharing the file budget.

    Files are named by slot, not by a monotonically growing counter: publishing
    overwrites slot ``n mod N``. That is what makes the bound a *file* bound --
    four files per publication, three publications, twelve files, forever -- with
    no separate deletion logic to get wrong and no growth between runs. The
    counter that grows is the number of publications captured, and it is bounded
    by ``FPV_RF_DIAG_DUMP_FILES`` shared with the periodic dump, because two
    budgets would be one more way to write more than documented.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._n = 0
        self._done = False
        self._warned = False

    def reset(self) -> None:
        with self._lock:
            self._n = 0
            self._done = False
            self._warned = False

    def _fail(self, exc: BaseException) -> None:
        self._done = True
        if not self._warned:
            self._warned = True
            _echo(f"DIAG pubcap disabled after an error: "
                  f"{type(exc).__name__}: {exc}")

    def write(self, cap: PubCap) -> bool:
        """Write one publication. Returns whether anything was written."""
        try:
            with self._lock:
                if self._done:
                    return False
                return self._write_locked(cap)
        except Exception as exc:                # never into the decode thread
            self._fail(exc)
            return False

    def _write_locked(self, cap: PubCap) -> bool:
        prefix = dump_prefix()
        if not prefix:
            if not self._warned:
                self._warned = True
                _echo("DIAG pubcap: FPV_RF_DIAG_DUMP is unset, so there is "
                      "nowhere to write; nothing captured")
            return False
        slots = max(1, pubcap_files())
        budget = _env_int(ENV_DUMP_FILES, DUMP_FILES)
        if self._n >= budget:
            # The shared budget is spent. Said once, because a capture that
            # silently stops answering questions is worse than one that says why.
            self._done = True
            _echo(f"DIAG pubcap: budget of {budget} file sets spent; no more "
                  "captures (raise FPV_RF_DIAG_DUMP_FILES to keep going)")
            return False

        index = self._n
        slot = index % slots
        stem = f"{prefix}_pub{slot:02d}"
        info: dict = {
            "capture": "publication",
            "publication_index": int(index),
            "slot": int(slot),
            "slots": int(slots),
            "sample_rate": int(cap.sample_rate),
            "encoding": cap.encoding,
            "source_kind": cap.source_kind,
            "frequency_hz": int(cap.frequency_hz),
            "segment": int(cap.segment),
            "generation": int(cap.generation),
            "stats": dict(cap.stats),
        }
        info.update(cap.as_dict())
        info["counts"] = report()
        info["files"] = {
            "iq": f"{stem}_iq.u8",
            "demod": f"{stem}_demod.f32",
            "png": f"{stem}.png",
            "json": f"{stem}.json",
        }
        notes: list[str] = []
        if not info["continuous"]:
            if info["iq_unmatched"]:
                notes.append("NO IQ SPANS: the window could not be matched to "
                             "stream positions, so these samples are not a "
                             "capture of anything")
            elif info["iq_unplaced"]:
                notes.append("STREAM POSITION UNKNOWN: the source cannot place "
                             "these samples, so the ranges here are ordinal only")
            else:
                why = ", ".join(str(g.get("reason")) for g in info["iq_gaps"])
                notes.append(f"NOT ONE CONTINUOUS CAPTURE ({why})")

        # Written outside the decoder's lock: the snapshot already holds its own
        # copy of everything it needs, and these are megabytes of I/O.
        iq = iq_to_u8(cap.iq_array(), cap.encoding)
        if iq:
            Path(f"{stem}_iq.u8").write_bytes(iq)
        if cap.demod is not None and cap.demod.size:
            np.asarray(cap.demod, dtype=np.float32).tofile(f"{stem}_demod.f32")
        if cap.image is not None and cap.image.size:
            Path(f"{stem}.png").write_bytes(png_bytes(cap.image))
        Path(f"{stem}.json").write_text(json.dumps(info, indent=1, default=str))
        _echo(
            f"DIAG pubcap {stem}: pub {index} (slot {slot}), "
            f"{info['iq_samples']} IQ samples, {info['demod_samples']} demod, "
            f"{info['lines_used']} lines ({info['lines_from_grid']} from grid)"
            + (f" -- {notes[0]}" if notes else "")
        )
        # The metadata only: a tool can compare the last three publications
        # without reading the files back, and holding the arrays here would
        # quietly triple the memory bound the files already impose.
        summary = {
            "publication_index": int(index),
            "slot": int(slot),
            "iq_samples": info["iq_samples"],
            "iq_first": int(info["iq_spans"][0]["first_sample"]) if info["iq_spans"] else -1,
            "iq_last": int(info["frame"]["iq_last"]),
            "segment": int(cap.segment),
            "generation": int(cap.generation),
            "line_positions": int(len(info["line_positions"])),
            "lines_from_grid": int(info["lines_from_grid"]),
            "continuous": bool(info["continuous"]),
        }
        _DUMP._pub_recent.append(summary)
        del _DUMP._pub_recent[:-slots]
        self._n += 1
        _DUMP._spend()
        if _DUMP._done:
            self._done = True
        return True


_PUB = _Pub()


def pubcap_recent() -> list[dict]:
    """Metadata of the last few publications, newest last. For tools."""
    return list(_DUMP._pub_recent)


def pubcap_reset() -> None:
    """Zero the publication counter and re-arm the budget.

    Separate from :func:`reset` on purpose: that one clears the counters and is
    called between cases in tools, while this holds state about *files on disk*
    and has the same standing as :meth:`_Dump.reset`, which tools also call
    explicitly. Folding it in would silently re-arm a capture that had stopped
    for its budget being spent, which is a thing the caller asked for.
    """
    _PUB.reset()


def pubcap_write(cap: PubCap) -> bool:
    """Write one publication's capture. Returns whether anything was written.

    The one public entry point to the writer, so the decode thread has no reason
    to know that ``_PUB`` exists -- and so an error in the capture can never
    reach the caller as an exception: :meth:`_Pub.write` swallows it, counts it
    in the log once, and disables the capture, because a diagnostic that can
    take the decoder down with it is worse than no diagnostic at all.
    """
    return _PUB.write(cap)


def pubcap_ledger(window: int, block: int) -> IqLedger:
    """A ledger big enough to cover a whole decoder window.

    ``window`` is the decoder's rolling demodulator buffer in samples and ``block``
    the largest hand-over it can be fed. The extra ``block + 1`` is what makes the
    ledger's trimming safe, and it is not padding: a window of ``window``
    demodulator outputs is the work of ``window + 1`` IQ samples, and trimming
    drops whole spans from the front, so the *smallest* total it can settle at is
    ``limit - block + 1``. Without the margin a ledger would occasionally settle
    a block short of the window it exists to describe -- and the lines at the top
    of the picture are the oldest ones in it, so that is exactly the part a
    capture would be missing.
    """
    return IqLedger(int(window) + int(block) + 1)