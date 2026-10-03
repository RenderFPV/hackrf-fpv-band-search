"""IQ acquisition: HackRF over USB, plus file replay and a signal generator.

The hardware path shells out to the Mayhem ``hackrf_transfer`` build and reads
raw IQ from its stdout. That is a deliberate choice:

* ``pyhackrf2`` is unusable here -- it hardcodes ``libhackrf.so.0``, and this
  machine has no ``hackrf.dll`` for it to load. The Mayhem
  ``hackrf_transfer.exe`` is statically linked, so it needs nothing installed.
* Streaming the receiver's own transfer loop avoids libhackrf's transfer
  callbacks entirely, which is where the Mayhem build and the stock library
  disagree.

Two command-line details are load-bearing:

* ``-r`` must precede the output argument. A bare ``-`` is a non-option
  argument, and the arg parser stops at the first non-option, so
  ``hackrf_transfer - -r`` silently stops parsing and never enters receive.
* The capture must sustain 20 MB/s at 10 MS/s. ``proc.stdout.read()`` on this
  machine managed only 17.2 MB/s and would overrun, so we use ``readinto`` into
  a single preallocated 4 MiB buffer, which sustains 19.8 MB/s.

Everything above the source is source-agnostic, which is what lets the app run
against a recording or a synthetic signal when no VTX is on the air.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path

import numpy as np
import re

# TEMP DIAG: counters for the decode-failure investigation. Off unless
# FPV_RF_DIAG is set; delete this import and every TEMP DIAG block below
# together with fpv_rf/diag.py. The fallback keeps this file runnable on its own
# (``python fpv_rf\sdr.py``), which the relative import would otherwise break.
try:
    from . import diag
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fpv_rf import diag

# 10 MS/s sustains a native NTSC/PAL field rate: 60 fields/s needs
# 10.014 MS/s and 50 fields/s needs 9.98 MS/s, so this is the one rate that
# serves both standards without resampling. 20 MS/s is not an option because
# the HackRF shares a USB 2.0 bus with other devices on this machine.
DEFAULT_SAMPLE_RATE = 10_000_000

#: Big enough to hold several fields, small enough to keep cache pressure low.
RING_BYTES = 4 << 20

#: A retune kills a process that is holding the USB device, and the firmware
#: refuses a second opener until the first one's handles are gone. So the spawn
#: after a retune is *expected* to fail once, briefly. These bound the retry.
OPEN_RETRIES = 14

#: How long to wait for samples from a process that is *alive*. This is a
#: deadline, not a sleep: the loop returns the instant the first byte lands, so
#: a generous ceiling costs nothing on a fast open.
OPEN_ARRIVE_WAIT = 0.35

#: Backoff between a failed open and the next attempt. This was a flat 350 ms,
#: and that single number was the largest cost in a band sweep: every hop pays
#: it, because the first spawn after a retune fails *by design*, and at 9 MHz
#: hops across 300 MHz that is 30 hops x 350 ms = 10.5 s of dead time before a
#: single sample was analysed. The device is nearly always free within a few
#: milliseconds of the outgoing process exiting -- ``_kill`` has already waited
#: for that exit -- so the first retry starts almost immediately and only backs
#: off if the device is genuinely still busy.
OPEN_RETRY_MIN = 0.015
OPEN_RETRY_MAX = 0.35
OPEN_RETRY_GROWTH = 1.6

#: Where ``hackrf_transfer`` is looked for last, after ``$HACKRF_TRANSFER``, the
#: copy beside this program and the one in ``tools/``. These are globs rather
#: than fixed paths because
#: the official Windows builds and the Mayhem firmware releases both unpack into a
#: version-stamped folder whose name nobody can predict -- and a hardcoded
#: ``C:\Users\<you>\Desktop\...`` in a public repository is nobody's path but
#: yours.
_TRANSFER_GLOBS = (
    "Desktop/*/utils/hackrf_transfer.exe",
    "Desktop/*/*/utils/hackrf_transfer.exe",
    "Downloads/*/utils/hackrf_transfer.exe",
    "tools/hackrf_transfer.exe",
)


def _candidate_transfer() -> tuple[str, ...]:
    """Expand the globs now, so a folder unpacked after startup is still found."""
    home = Path.home()
    found: list[str] = []
    for pattern in _TRANSFER_GLOBS:
        try:
            found += [str(p) for p in home.glob(pattern) if p.is_file()]
        except OSError:
            continue
    return tuple(found)


def app_dir() -> Path:
    """The directory the running program lives in.

    Frozen, that is the folder holding the executable, which is where a bundled
    ``hackrf_transfer.exe`` sits. Unfrozen, it is the source tree. Everything
    that has to be found at runtime -- the radio helper, the log file -- is
    looked up relative to this, because in a frozen build there is no
    ``__file__`` that points at a real directory on disk.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=4)
def supported_flags(executable: str) -> frozenset[str]:
    """Which single-letter options this ``hackrf_transfer`` build accepts.

    Builds differ. The Mayhem build has ``-a`` (RF amplifier), ``-p`` (antenna
    port power), ``-o`` (front-end LO) and ``-i``; the stock Great Scott build
    has none of them and treats an unknown option as a usage error, so a flag
    that is correct on one machine is fatal on another. Rather than guess from
    the file name, ask the binary once and cache the answer for the process.

    A probe that fails yields the empty set, which means "pass nothing
    optional" -- the safe direction, since every optional flag here is a
    refinement rather than a requirement.

    ``CREATE_NO_WINDOW`` is not tidiness, it is the reason this probe is not
    free. It runs on the startup path -- :meth:`HackrfSource._cmdline` asks for
    the flag set while building the *receive* command line -- so it happens
    before the first sample can exist. A console-subsystem child launched from
    the windowed frozen exe cost ~0.66 s here, against ~0.024 s under console
    python and ~0.030 s with this flag set: that is the whole of the "the
    radio produced nothing" delay, and it was charged to the radio rather than
    to the probe. The receiver itself is spawned with the same flag for the
    same reason; see :meth:`HackrfSource._spawn`.
    """
    try:
        r = subprocess.run(
            [executable, "-h"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return frozenset()
    # The help text is inconsistent about brackets: most options are spelled
    # "[-a amp_enable]" but the receive and write options are bare "-r <file>".
    # Matching only the bracketed form silently under-reports what the build
    # supports, which is the same class of bug as assuming a flag is absent.
    # Anchoring to the start of a line skips prose like "use '-' for stdout",
    # where the dash is not followed by a letter.
    text = f"{r.stdout or ''}\n{r.stderr or ''}"
    return frozenset(re.findall(r"(?m)^\s*\[?-([A-Za-z])(?=[\s\]>])", text))


def find_hackrf_transfer() -> str | None:
    """Locate ``hackrf_transfer``: env override, beside the app, ``tools/``, PATH,
    then known installs.

    A copy sitting next to the executable -- or in this checkout's ``tools/``
    folder, for an unfrozen run -- is preferred over ``PATH`` on purpose. A
    self-contained build that silently picked up a *different* version of the
    radio helper from somewhere else on the machine would be a miserable bug to
    chase, and the whole point of shipping one beside the exe is that it is the
    one being used.
    """
    env = os.environ.get("HACKRF_TRANSFER")
    if env and Path(env).exists():
        return env
    for name in ("hackrf_transfer.exe", "hackrf_transfer"):
        beside = app_dir() / name
        if beside.exists():
            return str(beside)
    # ``tools/`` is where this repository keeps the helper, and ``_TRANSFER_GLOBS``
    # did not cover it: those patterns are anchored at ``Path.home()``, so its
    # "tools/hackrf_transfer.exe" entry only ever matched ``~/tools/``, never the
    # checkout this module ships in. Unfrozen, that left the copy sitting in the
    # tree invisible and resolution fell through to whatever was on the Desktop
    # -- the opposite of the preference stated above. Frozen, this is simply a
    # miss, which is right: beside-the-exe has already had its turn by then.
    for name in ("hackrf_transfer.exe", "hackrf_transfer"):
        in_tools = app_dir() / "tools" / name
        if in_tools.exists():
            return str(in_tools)
    found = shutil.which("hackrf_transfer") or shutil.which("hackrf_transfer.exe")
    if found:
        return found
    for c in _candidate_transfer():
        if Path(c).exists():
            return c
    return None


# --------------------------------------------------------------------------
# Ring buffer
# --------------------------------------------------------------------------


class TimerResolution:
    """Hold the Windows system timer at 1 ms while a source is streaming.

    ``Event.wait`` on Windows is quantised by the system clock tick, so asking it
    to sleep for 1.8 ms can return five milliseconds later. A real-time source
    paced with one sleep per block then loses a large fraction of its sample
    rate -- measured here as 0.69x real time when the generator itself had
    1.22x of headroom to spare. The deficit is entirely in the wait, so making
    the generator faster cannot fix it.

    ``timeBeginPeriod(1)`` lowers the tick from 15.6 ms to 1 ms. It raises idle
    power draw for the whole system, so it is scoped to the streaming window and
    always released in ``stop``, including on error paths.
    """

    def __init__(self) -> None:
        self._winmm = None
        self._held = False

    def acquire(self) -> bool:
        if os.name != "nt" or self._held:
            return self._held
        try:
            import ctypes

            self._winmm = ctypes.WinDLL("winmm")
            self._held = self._winmm.timeBeginPeriod(1) == 0
        except Exception:                      # not fatal, just slower pacing
            self._winmm = None
            self._held = False
        return self._held

    def release(self) -> None:
        if self._winmm is not None and self._held:
            try:
                self._winmm.timeEndPeriod(1)
            except Exception:
                pass
        self._held = False

    def __enter__(self) -> "TimerResolution":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


@dataclass(frozen=True)
class RingSnapshot:
    """One contiguous run of ring bytes, with the absolute range it covers.

    ``first_byte``/``last_byte`` are absolute write-counter positions, so they
    name the same stretch of stream wherever they are read; ``first_tune_id``
    and ``last_tune_id`` say which tuning was in effect at each end, and the
    ``*_stream_id`` pair says which producing process wrote each end. Those
    differing is the only way a snapshot can be non-contiguous in *meaning*
    rather than in bytes, and it has to travel with the data: samples from two
    frequencies look like two frequencies, not like a splice -- and neither do
    samples either side of a transfer-process restart.

    The stream ids default to 0 so a caller constructing one positionally keeps
    working; every path that goes through :class:`IQRing` fills them in.
    """

    data: bytes
    first_byte: int
    last_byte: int
    first_tune_id: int
    last_tune_id: int
    first_stream_id: int = 0
    last_stream_id: int = 0

    @property
    def samples(self) -> int:
        return len(self.data) // 2

    @property
    def spans_tune(self) -> bool:
        """True when the snapshot holds the tail of one tuning and the head of the next."""
        return self.first_tune_id != self.last_tune_id

    @property
    def spans_stream(self) -> bool:
        """True when the snapshot holds the tail of one process and the head of the next."""
        return self.first_stream_id != self.last_stream_id


@dataclass(frozen=True)
class RingRead:
    """One hand-over out of the ring, with everything a consumer needs.

    All of it comes out of *one* acquisition of the ring's lock. That is the
    whole reason this is a record rather than four return values: reading the
    write counter and then the bytes lets a write landing in between be counted
    as consumed and never handed over, which cannot be made correct, only less
    likely.

    Ranges are half-open and absolute -- ``[first_byte, last_byte)`` of the
    ring's monotonic write counter -- and byte positions are always even, so
    ``// 2`` gives the absolute *sample* index of the same span with no
    rounding. ``cursor`` is the new read position: the write counter as observed
    under that same lock, so everything up to it has either been delivered in
    ``data`` or deliberately skipped.
    """

    data: bytes | bytearray
    cursor: int
    first_byte: int
    last_byte: int
    first_tune_id: int
    last_tune_id: int
    first_stream_id: int
    last_stream_id: int
    #: Bytes the cursor moved past without ever being handed over: a backlog
    #: longer than the cap, consumed and dropped on purpose. Deliberately kept
    #: apart from a ring overrun, which is loss rather than policy.
    skipped_bytes: int

    @property
    def samples(self) -> int:
        return len(self.data) // 2

    @property
    def skipped_samples(self) -> int:
        return int(self.skipped_bytes) // 2


class IQRing:
    """Byte ring buffer handing contiguous IQ blocks to the decoder.

    The reader thread appends; the decoder takes the newest ``n`` bytes as one
    contiguous copy. On overflow the *oldest* data is discarded, because for a
    live view stale video is worse than a dropped frame.

    Every write is stamped with the id of the tuning that produced it *and* the
    id of the producing process, and the position at which either changed is
    remembered. That is what lets a consumer be told whether the block it was
    handed straddles a seam: an FM discriminator carries state from one sample
    to the next, so a block holding the tail of one frequency and the head of
    another cannot be demodulated as one continuous stream, and no consumer can
    tell from the bytes alone. The same is true of a block that spans a transfer
    -process restart or a replayed file's loop point: both are real
    discontinuities and both are invisible in the samples.

    The identities are recorded **at write time, from the writer**, not read
    back later. On a HackRF that matters: the sampling thread can be blocked in
    ``readinto`` on a process a retune has already replaced, so reading the
    current process after the read would stamp the *old* process's samples with
    the *new* process's identity and the seam would be recorded in the wrong
    place -- which is to say, not recorded at all.

    **Two different things are counted, because they mean different things.**
    The ring is a 4 MiB history buffer and a decoder that has been running for
    a minute has, necessarily, pushed most of that out of the back -- the ring
    holds only the last couple of seconds, and every byte after it was read by
    the decoder and is not lost at all. That is :attr:`overwritten_bytes`, and
    it is the expected steady state of a working stream. :attr:`dropped_bytes`
    counts only the bytes that went out of the back *without any consumer ever
    having been offered them* -- real loss, the thing an operator is reading the
    counter to find out about. They used to be one number, and a healthy stream
    reported tens of megabytes of "loss" every second, which is both useless as
    a fault signal and a standing invitation to ignore the field.
    """

    #: How many tuning transitions are remembered. Only one ring's worth of
    #: history can ever be read back, so this is a bound on how many retunes
    #: can happen inside a single decode block, not a growing log. One entry
    #: more than this may be held: the *anchor* (see :meth:`_prune_tags_locked`),
    #: which is the identity in force at the oldest byte still in the ring and
    #: cannot be replaced by a later entry.
    TAG_HISTORY = 64

    def __init__(self, size: int = RING_BYTES) -> None:
        self._buf = bytearray(size)
        self._size = size
        self._wpos = 0
        self._count = 0
        self._lock = threading.Lock()
        #: Bytes destroyed by an overrun that no consumer had read. What an
        #: operator means by "dropped samples".
        self.dropped_bytes = 0
        #: Bytes pushed out of the ring's history, read or not. The ring is a
        #: rolling buffer; this grows on a perfectly healthy stream and only
        #: means "the buffer turned over", which is what it now says.
        self.overwritten_bytes = 0
        #: Absolute position up to which some consumer has taken delivery. Not
        #: advanced by writes: it is the reader's promise, so the ring can tell
        #: which of the bytes it is discarding were never offered to anybody.
        self._read_upto = 0
        self.total_written = 0
        #: ``(byte position, tuning id, stream id)``, ascending, recording where
        #: each tuning started writing and which process wrote it. Positions are
        #: absolute, so they mean the same thing as ``total_written``. The first
        #: entry always exists once anything has been written, which is what
        #: lets ``_tag_at`` answer for a position older than the list.
        self._tags: list[tuple[int, int, int]] = []

    def _mark_read_locked(self, upto: int) -> None:
        """Record that a consumer has taken delivery up to ``upto``. Lock held."""
        if upto > self._read_upto:
            self._read_upto = int(upto)

    def _loss_locked(self, oldest: int, lost: int, end: int) -> int:
        """Of ``lost`` bytes leaving the ring at ``oldest``, how many were unread.

        Caller holds the lock. The bytes discarded are the oldest ones, and the
        unread ones are the newest, so the two overlap only when the overrun
        reaches past everything a consumer has taken -- which is exactly the
        case that is genuine loss.

        ``end`` is the absolute position just past everything written so far,
        passed in rather than read from ``total_written`` because the counter is
        updated after this runs, and the ceiling has to be the *new* end when an
        oversized write has discarded a head.
        """
        if lost <= 0:
            return 0
        return max(
            0,
            min(oldest + lost, int(end)) - max(oldest, self._read_upto),
        )

    def write(
        self, data: bytes | memoryview, tune_id: int = 0, stream_id: int = 0
    ) -> int:
        """Append bytes, stamped with the tuning and process that produced them.

        Returns the number of bytes *stored*. A write larger than the ring keeps
        only its tail; the discarded head is counted in
        :attr:`dropped_bytes`/``overwritten_bytes`` and, importantly, in the
        absolute counters as well.

        That last part used to be wrong, and it mattered. ``total_written``
        advanced by the *retained* length, so a write too large to store left a
        hole in the ring's own address space: the positions of everything after
        it silently disagreed with the real length of the stream, and no consumer
        could tell the difference between "the producer wrote nothing for a
        while" and "the producer wrote a great deal that the ring could not
        hold". Both the loss accounting and the drain's absolute ranges are
        expressed in those positions, so the hole made a real, unnameable gap
        look like a continuous stream. The counters now span the whole write,
        discarded head included, and ``base`` below is the absolute position of
        the first *retained* byte.
        """
        n = len(data)
        with self._lock:
            head = 0
            if n > self._size:
                # Far more than we can hold; keep only the tail. The head is
                # gone before any consumer could have seen any of this write.
                head = n - (self._size // 2)
                self.dropped_bytes += head
                self.overwritten_bytes += head
                # Everything the ring already held goes with it. Keeping the
                # tail of this write and the head of the old contents in one
                # buffer left the ring answering "the newest n bytes" with two
                # different stretches of stream: the new tail, and the *old*
                # bytes that happened to sit behind it in the buffer, presented
                # as positions contiguous with it. Since only half a ring is
                # stored, those old bytes survived the write untouched -- so the
                # ring claimed a history it had just invalidated, and a decoder
                # fed from it would have demodulated a jump as a continuous
                # stream. Counted as loss, since the unread ones really are:
                # they were readable until this call destroyed them.
                if self._count:
                    self.dropped_bytes += self._loss_locked(
                        self.total_written - self._count, self._count, self.total_written
                    )
                    self.overwritten_bytes += self._count
                    self._count = 0
                data = memoryview(data)[head:]
                n = len(data)
            # Absolute position of the first byte this call actually stores.
            base = self.total_written + head
            pos = self._wpos
            end = pos + n
            if end <= self._size:
                self._buf[pos:end] = data
            else:
                first = self._size - pos
                self._buf[pos:] = data[:first]
                self._buf[: end - self._size] = data[first:]
            self._wpos = end % self._size
            if self._count + n > self._size:
                over = self._count + n - self._size
                # Absolute position of the oldest byte still in the ring,
                # which is the start of the run about to be overwritten.
                oldest = base - self._count
                self.dropped_bytes += self._loss_locked(oldest, over, base + n)
                self.overwritten_bytes += over
            self._count = min(self._count + n, self._size)
            # Recorded before the counter moves, because the seam belongs to the
            # first byte of *this* write, not the one after it.
            if (not self._tags
                    or self._tags[-1][1] != tune_id
                    or self._tags[-1][2] != stream_id):
                self._tags.append((base, int(tune_id), int(stream_id)))
                self._prune_tags_locked(base)
            self.total_written = base + n
        return n

    def _prune_tags_locked(self, base: int) -> None:
        """Drop the seams the ring can no longer answer for. Lock held.

        A tag may be dropped only once every byte it describes has left the ring:
        below that point it is still the identity in force at the head of the
        buffer, and :meth:`_tag_at` needs it to answer for those bytes.

        The *anchor* -- the newest tag older than the oldest byte still held --
        is therefore kept, deliberately, and the rest is bounded by
        :attr:`TAG_HISTORY`. Without it the tag list becomes empty on a long
        unchanged stream (which is every healthy stream: the first tag falls
        below the floor one ring-length in), and the *next* retune then becomes
        the oldest tag. At that point a read of the retained tail is answered
        with the new tuning's identity -- the old bytes are labelled as samples
        of a frequency they were not recorded at -- and a snapshot that
        straddles the retune reports one identity at both ends and no seam at
        all, which is the one answer that would have a decoder demodulate two
        frequencies as one stream.
        """
        floor = base - self._size
        anchor: tuple[int, int, int] | None = None
        kept: list[tuple[int, int, int]] = []
        for tag in self._tags:
            if tag[0] < floor:
                anchor = tag                     # the newest one below the floor
            else:
                kept.append(tag)
        kept = kept[-self.TAG_HISTORY:]
        if anchor is not None and (not kept or kept[0][0] != anchor[0]):
            kept.insert(0, anchor)
        self._tags = kept

    def available(self) -> int:
        with self._lock:
            return self._count

    def write_position(self) -> int:
        """The absolute write counter, read under the ring's own lock."""
        with self._lock:
            return self.total_written

    def _newest_locked(self, n: int) -> bytearray:
        """Newest ``n`` bytes, oldest-first. Caller holds the lock."""
        n = min(n, self._count)
        if n <= 0:
            return bytearray()
        end = self._wpos
        start = (end - n) % self._size
        if start + n <= self._size:
            return bytearray(self._buf[start : start + n])
        first = self._size - start
        return bytearray(self._buf[start:] + self._buf[: n - first])

    def _tag_at(self, pos: int) -> tuple[int, int]:
        """``(tuning, producing process)`` in effect at byte position ``pos``.

        Lock held. Positions older than the oldest recorded seam belong to that
        seam's entry, which is what the floor in :meth:`write` preserves.
        """
        if not self._tags:
            return (0, 0)
        tune = self._tags[0][1]
        stream = self._tags[0][2]
        for p, t, s in self._tags:
            if p > pos:
                break
            tune, stream = t, s
        return (tune, stream)

    def diag_tags(self) -> tuple[tuple[int, int], ...]:
        """TEMP DIAG: a copy of the recorded ``(position, tuning id)`` seams.

        This is the evidence for "was anything spliced across a retune". A
        consumer cannot infer it from the bytes -- two frequencies look like two
        frequencies -- and the hand-over ids only say whether *this* block
        straddled a seam, not whether one happened just before it.

        Pairs only, for the callers that read this; :meth:`diag_seams` has the
        stream identity as well.
        """
        with self._lock:
            return tuple((p, t) for p, t, _s in self._tags)

    def diag_seams(self) -> tuple[tuple[int, int, int], ...]:
        """TEMP DIAG: the seams as ``(position, tuning id, stream id)``."""
        with self._lock:
            return tuple(self._tags)

    def read_newest(self, n: int) -> bytearray:
        """Newest ``n`` bytes, oldest-first. Returns fewer if that is all there is.

        A snapshot read, so it counts as consumption for the loss accounting:
        whatever the caller looked at, it is not going to be asked for again,
        and an overrun that destroys it has not destroyed anything anybody was
        waiting on.
        """
        with self._lock:
            data = self._newest_locked(n)
            if data:
                self._mark_read_locked(self.total_written)
            return data

    def snapshot_newest(self, max_bytes: int) -> RingSnapshot:
        """Newest ``max_bytes`` bytes plus the absolute range and tuning they cover.

        One acquisition of the lock returns the bytes *and* the range they came
        from, so the two cannot disagree: there is no window in which a write
        lands between reading the counter and reading the data, which is the loss
        :meth:`read_since` exists to make impossible for the streaming path.

        It deliberately does **not** move the loss accounting. That is the whole
        difference from :meth:`read_newest`, which counts as consumption because
        whatever the caller looked at it will not be asked for again. This one is
        a look: the diagnostic dump reads the ring and the decoder still gets
        every sample it would have got, and an overrun that destroys bytes the
        dump has already written must still count as loss against the decoder
        that had not read them yet.

        Bytes come out oldest-first and are contiguous -- the newest ``n`` bytes
        of a ring are one run, never two -- so ``last_byte - first_byte`` is the
        length of the file that gets written, with no stitching.
        """
        with self._lock:
            data = self._newest_locked(int(max_bytes))
            last = self.total_written
            first = last - len(data)
            first_tune, first_stream = self._tag_at(first)
            last_tune, last_stream = self._tag_at(last)
            return RingSnapshot(
                bytes(data),
                first,
                last,
                first_tune,
                last_tune,
                first_stream,
                last_stream,
            )

    def read_since(self, cursor: int, max_bytes: int) -> RingRead:
        """Unconsumed bytes since ``cursor``, with everything a consumer needs.

        :meth:`IQSource.drain_block` documents why this returns a record rather
        than a tuple: the data, the new cursor, the absolute range and the two
        identities at each end of it all come out of *one* acquisition of the
        lock, and cannot be made to disagree by a write landing in between.

        The new cursor is the write position observed under that same lock, so
        whatever was written while this block was being copied is *not* counted
        as consumed and comes round on the next call. A backlog longer than
        ``max_bytes`` is consumed and dropped, as documented on
        :meth:`IQSource.drain_block`, and the size of that drop travels with the
        record rather than only being counted.
        """
        with self._lock:
            written = self.total_written
            pending = max(0, written - int(cursor))
            n = min(pending, max(0, int(max_bytes)))
            data = self._newest_locked(n)
            # The consumer's new cursor is the write position observed under
            # this same lock, so everything up to it is either in ``data`` or
            # deliberately skipped -- either way it has been dealt with and an
            # overrun must not count it again as data nobody read.
            self._mark_read_locked(written)
            first_pos = written - len(data)
            first_tune, first_stream = self._tag_at(first_pos)
            last_tune, last_stream = self._tag_at(written)
            return RingRead(
                data=data,
                cursor=written,
                first_byte=first_pos,
                last_byte=written,
                first_tune_id=first_tune,
                last_tune_id=last_tune,
                first_stream_id=first_stream,
                last_stream_id=last_stream,
                skipped_bytes=max(0, pending - n),
            )


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


@dataclass
class SourceStats:
    bytes_total: int = 0
    read_errors: int = 0
    restarts: int = 0
    #: Times a spawn was retried because the device was still held by the
    #: outgoing process. Non-zero is normal on hardware and worth showing, since
    #: it is the difference between a sweep that is merely slow and one that is
    #: quietly measuring the wrong hop.
    open_retries: int = 0
    last_error: str = ""
    running: bool = False
    started_at: float = field(default_factory=time.monotonic)
    #: Wall clock spent inside retunes, and how many. On hardware this is the
    #: entire cost of a band sweep -- the samples themselves are 1.6 ms a hop --
    #: so it is the one number that says whether a sweep is fast, and it is
    #: reported rather than inferred.
    tune_total_s: float = 0.0
    tune_count: int = 0
    tune_last_ms: float = 0.0

    def rate(self, sample_rate: int) -> float:
        el = max(1e-6, time.monotonic() - self.started_at)
        return self.bytes_total / 2.0 / el / sample_rate

    @property
    def tune_mean_ms(self) -> float:
        if not self.tune_count:
            return 0.0
        return self.tune_total_s * 1000.0 / self.tune_count

    def snapshot(self) -> dict:
        return {
            "bytes_total": self.bytes_total,
            "read_errors": self.read_errors,
            "restarts": self.restarts,
            "open_retries": self.open_retries,
            "last_error": self.last_error,
            "running": self.running,
            "tune_total_s": self.tune_total_s,
            "tune_count": self.tune_count,
            "tune_last_ms": self.tune_last_ms,
            "tune_mean_ms": self.tune_mean_ms,
        }


class IQEncoding(Enum):
    """How the raw interleaved I/Q bytes are scaled into the +/-1 complex domain.

    This has to be explicit. The two encodings fold a zero-mean signal into
    different parts of the same byte axis, so reading one as the other is not a
    small error that a threshold can absorb -- it is a non-linear distortion of
    every sample, plus a sign inversion. A decoder handed that produces noise,
    while a signal-strength meter still reports a plausible number, because the
    corruption is symmetric and its power is preserved. That is why this went
    unnoticed.

    ``SIGNED_INT8`` is what ``hackrf_transfer`` emits, both to stdout and to the
    ``-r`` file. Confirmed against real 5802 MHz captures from the Mayhem
    toolchain this project drives: 100% of bytes fall outside [64,192) and none
    inside [108,147), which is the signature of two's-complement samples and the
    exact opposite of offset-binary data. See ``tools/probe_iq_format.py``.

    ``UNSIGNED_OFFSET`` is ``round(s * 127.5) + 128``, written by
    :class:`SimSource` and by some third-party recorders. Kept because existing
    unsigned recordings must keep working, and because the simulator is
    validated against it.
    """

    SIGNED_INT8 = "signed-int8"
    UNSIGNED_OFFSET = "unsigned-offset"


#: Both encodings are scaled by 127.5 so that a sample decoded from a file and
#: the same sample from the radio land on identical amplitudes. 127.5 is half the
#: span of the 0..255 byte range, which is what the unsigned mapping was built
#: around; reusing it keeps the two paths amplitude-comparable instead of
#: introducing a 0.4% discrepancy between hardware and simulator.
_IQ_SCALE = np.float32(1.0 / 127.5)


def u8_to_iq(
    raw: bytes | bytearray | memoryview,
    encoding: "IQEncoding" = IQEncoding.SIGNED_INT8,
) -> np.ndarray:
    """Interleaved raw I/Q bytes to normalised complex64. Public entry point.

    The conversion itself is :func:`_u8_to_iq`; this is the same function under
    a name tools are allowed to call. Offline analysis of a capture has to
    decode it *somehow*, and the version it used to have was an open-coded copy
    of the wrong branch of :class:`IQEncoding` -- reading a HackRF recording as
    offset-binary, which inverts the waveform and leaves every power figure
    looking fine. Reaching for the real one here is what keeps a diagnostic and
    the app it is diagnosing from disagreeing about the same file.
    """
    return _u8_to_iq(raw, encoding)


def _u8_to_iq(
    raw: bytes | bytearray | memoryview,
    encoding: "IQEncoding" = IQEncoding.SIGNED_INT8,
) -> np.ndarray:
    """Interleaved raw I/Q bytes to normalised complex64.

    This sits directly in the streaming path -- it runs on every field, so its
    cost is a fixed slice of the 16.7 ms budget. The obvious spelling allocates
    five full-size temporaries:

        frombuffer -> astype(float32) -> subtract -> divide -> reshape
                  -> two strided reads -> complex128 temp -> astype(complex64)

    which measured ~7 ms for a 250k-sample chunk. The version below does two
    conversions into the real and imaginary views and nothing else, so there is
    no complex128 intermediate and no separate scale pass; the de-interleave is
    the single strided copy.

    ``SIGNED_INT8`` needs no bias pass at all: the samples are already centred on
    zero, so one multiply per component is the whole conversion. Offset-binary
    data needs its 128 removed first; see the note on the two-pass form below for
    why that ordering is not optional.

    Both branches are exact, so the same sample written either way decodes to the
    same complex value bit for bit. That is what lets a recording and a live
    capture of one signal be compared directly.
    """
    n = len(raw) // 2
    if n == 0:
        return np.zeros(0, dtype=np.complex64)
    iq = np.empty(n, dtype=np.complex64)
    re, im = iq.real, iq.imag

    if encoding is IQEncoding.SIGNED_INT8:
        arr = np.frombuffer(raw, dtype=np.int8, count=2 * n)
        np.multiply(arr[0 : 2 * n : 2], _IQ_SCALE, out=re, casting="unsafe")
        np.multiply(arr[1 : 2 * n : 2], _IQ_SCALE, out=im, casting="unsafe")
        return iq

    arr = np.frombuffer(raw, dtype=np.uint8, count=2 * n)
    # The bias is removed *before* scaling, not after, which is the whole point of
    # the two-pass form. The previous version scaled by 1/127.5 and then
    # subtracted 1.0, on the assumption that 128/127.5 is 1. It is not: it is
    # 1.0039. So byte 128 -- a zero sample, the exact centre of the encoding --
    # decoded to +0.0039, and every sample carried a 0.4%-of-full-scale DC
    # offset. Small enough to look like nothing and large enough to be wrong: it
    # meant the same signal recorded two ways did not decode to the same thing.
    # Subtracting in the float domain and then scaling once makes this path
    # bit-identical to the signed path for the same underlying sample.
    #
    # The extra pass costs nothing that matters. The live radio path is signed
    # and takes the single-multiply branch above; this one is only reached by the
    # simulator and by offset-binary recordings.
    np.subtract(arr[0 : 2 * n : 2], 128.0, out=re, casting="unsafe")
    np.multiply(re, _IQ_SCALE, out=re)
    np.subtract(arr[1 : 2 * n : 2], 128.0, out=im, casting="unsafe")
    np.multiply(im, _IQ_SCALE, out=im)
    return iq


def _whole_pairs(view) -> tuple[memoryview, bytes]:
    """Split a buffer into whole I/Q pairs plus at most one dangling byte.

    ``view`` is the bytes just received, *including* any byte carried over from
    the previous read. Returns ``(even-length prefix, remainder)``, the
    remainder being ``b""`` or the single byte that has no partner yet.

    The ring is a byte stream and :func:`_u8_to_iq` pairs it from the front, so
    it only ever holds whole samples if every producer hands it an even number
    of bytes. A pipe read can end anywhere -- an odd count is a routine, legal
    answer from ``readinto`` -- and dropping its last byte lost half a sample:
    the next read began with a Q where an I belonged, so every pair after that
    was built from two halves of different samples. Nothing downstream could
    say so. The ranges stayed adjacent, the byte counts still added up, and the
    decode quietly lost a little of its own frequency every time it happened.

    The odd byte is therefore carried forward rather than dropped, and only ever
    to a read *of the same producer*: a replacement process starts a new
    recording whose first byte is an I, and gluing the outgoing process's
    dangling half to it would fabricate exactly the pair this exists to prevent.
    :meth:`IQSource._take_carry` and :meth:`IQSource._keep_pairs` hold that half
    of the rule.
    """
    if len(view) & 1:
        return view[:-1], bytes(view[-1:])
    return view, b""


@dataclass(frozen=True)
class EncodingVerdict:
    """What the histogram could and could not establish about a recording.

    ``encoding`` is ``None`` when the evidence is not decisive. That is a real
    outcome, not a failure to be papered over, and it is the common case for any
    loud signal -- see :func:`detect_iq_encoding` for why.
    """

    encoding: IQEncoding | None
    #: Share of bytes falling outside [64,192). The single number the verdict is
    #: based on, kept so the reason can be shown and the test can be re-tuned.
    share: float
    reason: str


def detect_iq_encoding(raw: bytes | bytearray | memoryview) -> EncodingVerdict:
    """Infer whether ``raw`` is signed int8 or unsigned-offset -- if it can be.

    Only used for recordings, where the format is a property of the file rather
    than a known property of the tool that produced it. Hardware has a known
    format, so it is declared, never guessed: a detector that silently
    mis-classifies live samples would be far worse than the bug it replaces.

    **The histogram cannot always tell them apart, and this says so.** The two
    encodings differ by a fold about 128, so what distinguishes them is where
    the samples sit relative to the middle of the byte axis:

    * a *quiet* signal in either encoding sits near the middle, but the two
      middle regions are different bytes -- offset-binary puts it at 128, signed
      wraps it to 0 and 255. Easy to separate.
    * a *loud* signal reaches the ends of the axis in either encoding, and a
      wrapped loud signal is just a loud signal again. Indistinguishable.

    So this is decisive for quiet recordings and genuinely undecidable for loud
    ones. Quantified on real 5802 MHz captures (mean |x| 0.22): 100.0% of bytes
    fall outside [64,192) and 0.0% inside [108,147). A synthetic unit-magnitude
    fixture, by contrast, lands at 33.5% and is correctly reported as ambiguous.

    Returning ``None`` rather than guessing is the point. Guessing here would
    reproduce the original defect on a different input: a wrong decode that
    produces noise and plausible power readings, with nothing to indicate it.
    Callers fall back to a declared default and say so.
    """
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size < 1024:
        return EncodingVerdict(None, 0.0, "too few bytes to judge")
    share = float(np.mean((arr < 64) | (arr >= 192)))
    if share >= 0.90:
        return EncodingVerdict(
            IQEncoding.SIGNED_INT8, share,
            "almost every byte is at one end of the axis, which only a "
            "two's-complement wrap produces at this amplitude",
        )
    if share <= 0.10:
        return EncodingVerdict(
            IQEncoding.UNSIGNED_OFFSET, share,
            "almost every byte is near the middle of the axis, which is "
            "offset-binary",
        )
    return EncodingVerdict(
        None, share,
        f"{share*100:.0f}% of bytes are at the ends of the axis -- "
        f"consistent with either encoding, so the format cannot be "
        f"established from the data",
    )


@dataclass(frozen=True)
class IQBlock:
    """One hand-over of IQ, with the position and identity of the samples.

    The decoder cannot decide whether it may carry state across from the previous
    block out of the samples. Two frequencies look like two frequencies; a block
    either side of a transfer-process restart looks like one continuous stream.
    So the two facts that decide it travel with the data:

    * ``[first_sample, last_sample)`` -- absolute, half-open, on the ring's
      monotonic write counter. Adjacent means ``first_sample == previous
      last_sample``, and a sample *missing* between two blocks is a gap of
      exactly the right size rather than an unfalsifiable assumption that there
      was none.
    * ``first_stream_id``/``last_stream_id`` and ``first_tune_id``/
      ``last_tune_id`` -- who produced the samples at each end. Differing ids at
      *either* pair of ends are an internal seam (:attr:`seam`), which is not a
      gap but a mixture: it must be refused outright rather than counted.

    An **empty** delivery is a real answer, not an error: nothing arrived. It
    reports a zero-length range at the read cursor and must not be mistaken for
    a boundary, which is why consumers are told to leave their bookkeeping alone
    when :attr:`empty` is true.

    ``first_sample < 0`` means the position is *unknown* -- a source that offers
    samples without a position (a bare array, a test double). That is
    distinguished from a known range of zero length, because "I cannot place
    these samples in the stream" and "these samples are an empty span" must not
    be allowed to look alike.
    """

    iq: np.ndarray
    first_sample: int
    last_sample: int
    first_stream_id: int
    last_stream_id: int
    first_tune_id: int
    last_tune_id: int
    sample_rate: int = 0
    encoding: "IQEncoding | None" = None
    #: Samples the read cursor moved past without delivering: a backlog longer
    #: than the cap. Policy, not loss.
    skipped_samples: int = 0
    #: Why the previous span ended, when a reset said so ("retune", "seam",
    #: "overload", ...). Empty otherwise.
    reset_reason: str = ""
    #: Monotonic per-source hand-over number, so a consumer can tell one
    #: delivery from another even when both are empty.
    delivery: int = 0

    @property
    def samples(self) -> int:
        return int(self.iq.size)

    @property
    def known(self) -> bool:
        """Whether the absolute position of these samples is known."""
        return self.first_sample >= 0

    @property
    def empty(self) -> bool:
        return self.iq.size == 0

    def drop_head(self, n: int) -> "IQBlock":
        """This block without its first ``n`` samples, at the matching position.

        Everything that describes the *span* travels with it -- the absolute
        range moves up by ``n`` and the identities of the two ends are the same
        identities, because dropping samples cannot change who produced them.
        The block's provenance is the point of the type, so a shortened copy
        that kept the old ``first_sample`` would place the remaining samples
        ``n`` positions too early.

        Returns the same block when ``n`` is zero or negative, so a caller can
        subtract a repeat of unknown size without branching.
        """
        cut = max(0, min(int(n), self.samples))
        if not cut:
            return self
        return IQBlock(
            iq=self.iq[cut:],
            first_sample=self.first_sample + cut if self.known else -1,
            last_sample=self.last_sample,
            first_stream_id=self.first_stream_id,
            last_stream_id=self.last_stream_id,
            first_tune_id=self.first_tune_id,
            last_tune_id=self.last_tune_id,
            sample_rate=self.sample_rate,
            encoding=self.encoding,
            skipped_samples=self.skipped_samples,
            reset_reason=self.reset_reason,
            delivery=self.delivery,
        )

    @property
    def spans_stream(self) -> bool:
        return self.first_stream_id != self.last_stream_id

    @property
    def spans_tune(self) -> bool:
        return self.first_tune_id != self.last_tune_id

    @property
    def seam(self) -> str:
        """Why this block cannot be decoded as one stream, or ``""``.

        Both ids are checked at both ends. A block that holds the tail of one
        producer and the head of another is a mixture: the samples on either
        side of the boundary are each fine, and together they are a picture
        assembled out of two different pictures. Refusing the whole block is the
        only honest option.
        """
        if self.spans_stream:
            return "stream"
        if self.spans_tune:
            return "tune"
        return ""

    @property
    def out_of_range(self) -> bool:
        """A range that cannot be true: backwards, negative, or unknown."""
        return self.first_sample < 0 or self.last_sample < self.first_sample

    @classmethod
    def orphan(
        cls,
        iq: np.ndarray,
        sample_rate: int = 0,
        encoding: "IQEncoding | None" = None,
    ) -> "IQBlock":
        """A block from a source that offers samples but no position or identity.

        Used for the duck-typed source that only implements :meth:`drain_iq`.
        Nothing about it can be proven, so ``first_sample < 0`` and a consumer
        has to treat it as the start of its own segment rather than as a
        continuation of whatever came before.
        """
        return cls(
            iq=iq,
            first_sample=-1,
            last_sample=-1,
            first_stream_id=0,
            last_stream_id=0,
            first_tune_id=0,
            last_tune_id=0,
            sample_rate=int(sample_rate),
            encoding=encoding,
        )


class IQSource(ABC):
    """Common interface: background thread fills an :class:`IQRing` with IQ."""

    kind = "unknown"

    #: True when :meth:`tune` actually changes what the source is receiving, so
    #: a band search can sweep. A file replay cannot -- it is one capture at one
    #: frequency -- so callers must ask instead of assuming.
    retunable = True

    #: Whether the RF amplifier can be switched from the UI. True generally;
    #: the file and simulator sources override it, and a HackRF decides it by
    #: asking its transfer helper which flags it understands.
    amp_supported = True

    #: Byte encoding of the IQ this source produces. Subclasses must set this,
    #: because guessing is how the waveform got inverted in the first place.
    encoding: IQEncoding = IQEncoding.SIGNED_INT8

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = int(sample_rate)
        self.ring = IQRing()
        self.stats = SourceStats()
        self.frequency_hz = 0
        #: The frequency the hardware is known to be receiving. Tracked
        #: separately from ``frequency_hz``, which is only a *request*, so that a
        #: failed retune does not leave the source claiming a frequency it is not
        #: on and a retry does not short-circuit on a change that never happened.
        self._applied_hz = 0
        self.lna_gain_db = 0
        self.vga_gain_db = 0
        #: RF amplifier on/off. This is a real switch, not a proxy: the
        #: Mayhem ``hackrf_transfer`` takes ``-a 1`` / ``-a 0``, so the
        #: amplifier is genuinely bypassed rather than turned down. It is
        #: separate from the LNA gain -- asking for 0 dB of LNA gain is a
        #: different thing entirely, and the first version of this control got
        #: that wrong.
        self.amp_enabled = True
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tune_lock = threading.Lock()
        self._consumed = 0
        #: The trailing byte of a producer's last read, and which producer it
        #: belongs to. Zero or one byte, and every producer that fills the ring
        #: with bytes must go through :meth:`_take_carry`/:meth:`_keep_pairs` so
        #: the ring never holds half a sample -- see :func:`_whole_pairs` for
        #: what a dangling byte does to the pairing if it is dropped instead.
        self._carry_byte: bytes = b""
        self._carry_owner: object = None
        #: Serialises the drain cursor. The cursor is a single value read by
        #: :meth:`drain_iq` and moved by :meth:`reset_drain`, and both run on
        #: different threads (decoder, and a retune racing a scan). Without this,
        #: a reset landing mid-drain moves the cursor underneath the copy in
        #: flight and the block being returned is pre-reset data the caller
        #: believes was post-reset.
        self._drain_lock = threading.Lock()
        #: Bumped on every tuning the hardware actually accepted. Stamped into
        #: the ring with each write, so a consumer can be told that a block it
        #: was handed straddles a retune. The discriminator carries state from
        #: sample to sample and cannot demodulate such a block as one stream.
        self._tune_id = 0
        #: Stream identity: bumped whenever the *producer* is replaced rather
        #: than re-tuned -- a transfer process spawned or restarted, a replayed
        #: file passing its end, a source started afresh. It is the other half
        #: of the seam test, and it is the half that a tuning id cannot cover: a
        #: retune restarts the HackRF process, so two runs at the same frequency
        #: with the same tuning id are still two runs, and the samples either
        #: side of the restart are not one stream.
        self._stream_seq = 0
        self._stream_lock = threading.Lock()
        self._stream_id = 0
        self._deliveries = 0
        #: Why the previous span was abandoned, as recorded by
        #: :meth:`reset_drain`. Carried on the *next* delivery so a consumer can
        #: tell "the stream broke" from "somebody asked for a clean start".
        self._drain_reset_reason = ""
        #: The most recent structured hand-over. Kept beside the drain for the
        #: same reason :attr:`last_drain_tune_ids` is: callers want the facts
        #: about the block they just asked for, and a second, independent query
        #: would be exactly the race this API exists to remove.
        self._last_delivery: IQBlock | None = None
        #: Tuning ids of the last hand-over, kept for the same reason the ids
        #: exist in the ring.
        self.last_drain_tune_ids: tuple[int, int] = (0, 0)
        self._timer = TimerResolution()
        # TEMP DIAG. ``_drain_last_end`` is the absolute write position the
        # previous hand-over reached, so the gap between one delivery and the
        # next is measurable rather than inferred; the two marks turn the ring's
        # cumulative overrun counters into per-hand-over deltas. ``process_epoch``
        # counts transfer-process generations (a HackRF retune restarts the
        # child), which is the identity a decode has to be attributed to when
        # several tunings share one tuning id.
        self._diag = diag.section("iq")
        self._drain_last_end = 0
        self._drain_drop_mark = 0
        self._drain_over_mark = 0
        self.process_epoch = 0
        self.reclaims = 0

    # -- whole-sample discipline --------------------------------------------
    #
    # Two methods, so that no producer can fill the ring with a half sample by
    # accident. ``_take_carry`` is called before a read and ``_keep_pairs``
    # after it; between them the buffer begins with whatever the previous read
    # left over, and the pair split happens once, in one place.

    def _take_carry(self, owner: object, into: bytearray | None = None) -> bytes:
        """The byte held back from ``owner``'s previous read, if it is ours.

        ``owner`` is whatever identifies the producer -- the transfer process
        for hardware, the stream id for a replay -- and is what scopes the byte:
        a byte from a producer that has been replaced is dropped instead, because
        the replacement's stream starts on a sample boundary of its own.

        ``into``, when given, receives the byte (a producer reading into a fixed
        buffer puts it in the buffer's first byte and reads after it, so no copy
        of the payload is made); the byte is also returned for the producers that
        concatenate.
        """
        byte = b""
        if self._carry_owner is owner and self._carry_byte:
            byte = self._carry_byte
            if into is not None:
                into[0] = byte[0]
        self._carry_byte = b""
        self._carry_owner = None
        return byte

    def _keep_pairs(self, data, owner: object) -> memoryview:
        """Split off a dangling byte, keep it for ``owner``, return the pairs.

        ``data`` is what was received *including* any byte :meth:`_take_carry`
        returned. The returned prefix has an even length and is what may be
        written to the ring; anything left over is remembered against ``owner``.
        """
        pairs, self._carry_byte = _whole_pairs(data)
        self._carry_owner = owner if self._carry_byte else None
        return pairs

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.stats = SourceStats()
        self.process_epoch += 1          # TEMP DIAG: a new reader generation
        # A fresh reader is a fresh stream, whatever it turns out to read.
        # Nothing written before this can be a continuation of what comes after
        # it, and the ids are how a consumer learns that.
        self._begin_stream()
        self._timer.acquire()
        self._thread = threading.Thread(target=self._run, name=f"iq-{self.kind}", daemon=True)
        self._thread.start()

    # -- stream identity ---------------------------------------------------

    @property
    def stream_id(self) -> int:
        """Identity of the stream currently being produced."""
        return self._stream_id

    def _begin_stream(self) -> int:
        """Start a new stream identity and return it.

        Every call is a place where "the samples after this are a continuation
        of the samples before it" stops being true, whatever produced them: a
        transfer process spawned, a replayed file passing its end, a restarted
        reader. The id goes into the ring with each write, stamped by the
        writer, so it describes the bytes rather than the moment they were read.
        """
        with self._stream_lock:
            self._stream_seq += 1
            self._stream_id = self._stream_seq
            return self._stream_id

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.stats.running = False
        self._teardown()
        self._timer.release()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _teardown(self) -> None:
        pass

    #: True when another process can be given sole use of the radio while this
    #: source stands aside. Only a real radio can be handed over; a file or a
    #: simulator has no device to give away.
    exclusive_use = False

    def release_device(self) -> bool:
        """Give up whatever hardware this source holds, keeping the thread up.

        A no-op returning True by default, because most sources hold nothing
        exclusive and there is nothing to fail at.
        """
        return True

    def reclaim_device(self) -> bool:
        """Take hardware back after :meth:`release_device`. Always True here."""
        self.reclaims += 1               # TEMP DIAG: reclaim without an epoch bump
        return True

    @abstractmethod
    def _run(self) -> None:
        ...

    # -- control -----------------------------------------------------------

    def tune(self, frequency_hz: int, *, reconfigure: bool = False) -> bool:
        """Retune. Returns True if the change is in effect on the hardware.

        Public entry point: it takes ``_tune_lock`` and delegates to
        :meth:`_tune_locked`. The lock covers the whole operation because the
        apply talks to a child process, and two overlapping tunes would race for
        it -- the loser's frequency would be reported while the winner's
        configuration was on the air.

        ``reconfigure=True`` forces a re-apply even at an unchanged frequency.
        Gains and the amplifier switch are command-line arguments to the transfer
        helper, so they only reach the hardware when the process is replaced.
        Without this, moving a gain slider at the frequency already tuned changed
        a Python attribute and nothing else, and the control silently did nothing
        until some unrelated later action happened to restart reception.
        """
        with self._tune_lock:
            return self._tune_locked(int(frequency_hz), reconfigure=reconfigure)

    def _tune_locked(self, frequency_hz: int, *, reconfigure: bool = False) -> bool:
        """Body of :meth:`tune`, for callers that already hold ``_tune_lock``.

        Nothing here may take ``_tune_lock``: it is a plain ``Lock``, so
        re-acquiring it on the same thread blocks forever. Subclasses that want
        to refuse a tune override *this* rather than :meth:`tune`, for the same
        reason :meth:`retune_settled` is written in terms of it -- overriding
        ``tune`` alone would leave that path bypassing the refusal.
        """
        want = int(frequency_hz)
        if (want == self.frequency_hz and not reconfigure
                and self._applied_hz == want):
            return True
        previous = self.frequency_hz
        self.frequency_hz = want
        if not self._apply_tune():
            # Roll the request back to what is really on the air, so the two
            # cannot end up out of step in the direction that cannot recover.
            # Committing it and returning False meant the *next* identical
            # request matched ``frequency_hz``, took the early return above and
            # reported success without ever touching the radio: one refusal left
            # the receiver permanently claiming a frequency it was not on.
            # Rolling back keeps ``want != frequency_hz``, so a retry really is
            # a retry. ``_applied_hz`` is 0 only on a source that has never
            # applied a tune at all, and there the pre-request value is the more
            # truthful of the two.
            self.frequency_hz = self._applied_hz or previous
            self._diag.bump("tune_refused")       # TEMP DIAG
            return False
        # Recorded here rather than in each subclass, because several of them
        # legitimately short-circuit -- before the reader thread exists, or
        # while the device is lent to the sweep tool -- and a subclass that
        # forgets to set this would make tune() report a failure that never
        # happened. While released this is a promise about the pending
        # configuration, not about the air: reclaim_device() opens at
        # frequency_hz, which is what this records.
        self._applied_hz = want
        # Only a tuning the hardware accepted changes what is being received, so
        # only that is a seam. A refused tune leaves the air untouched, and
        # labelling it as a new tuning would make the decoder drop a perfectly
        # continuous stream for no reason.
        self._tune_id += 1
        # TEMP DIAG: seams are counted whether or not anything reads them, and
        # a refused tune is counted separately -- a sweep that keeps being
        # refused looks identical to one that keeps retuning until the decoder
        # drops a third of its samples.
        self._diag.bump("tune_applied")
        self._diag.gauge("tune_id", self._tune_id)
        self._diag.gauge("applied_hz", self._applied_hz)
        return True

    def _apply_tune(self) -> bool:
        return True

    def set_gains(self, lna_db: int, vga_db: int) -> bool:
        """Change gain and push it to the hardware. Returns True if applied.

        Returns a bool because the caller has to be able to tell the operator
        the truth: these are command-line flags on a child process, and a value
        that never reached that process is not a gain setting.
        """
        self.lna_gain_db = int(lna_db)
        self.vga_gain_db = int(vga_db)
        return self.tune(self.frequency_hz, reconfigure=True)

    @property
    def applied_frequency_hz(self) -> int:
        """The frequency the hardware is actually receiving, not merely asked for."""
        return self._applied_hz

    def describe(self) -> str:
        return f"{self.kind} @ {self.frequency_hz/1e6:.1f} MHz, {self.sample_rate/1e6:g} MS/s"

    def take_iq(self, n_samples: int) -> np.ndarray:
        """Newest ``n_samples`` complex samples as complex64.

        A snapshot read, not a consumption: it deliberately ignores the drain
        cursor, because callers that want "whatever is in the ring right now"
        (the settled-capture path, tests) need that and must not be surprised by
        a cursor they did not move. Streaming consumers use :meth:`drain_iq`.
        """
        if diag.dump_enabled():
            # TEMP DIAG: the settled-capture path never calls drain_iq, so
            # without this the dump would sit untriggered for every caller that
            # reads this way. Same non-consuming snapshot either path takes.
            diag.dump_tick(self._dump_snapshot)
        raw = self.ring.read_newest(int(n_samples) * 2)
        if len(raw) < 16:
            return np.zeros(0, dtype=np.complex64)
        return _u8_to_iq(raw, self.encoding)

    def _dump_snapshot(self, max_samples: int) -> tuple[bytes, dict]:
        """TEMP DIAG: the newest ``max_samples`` in the ring, and what they are.

        Returns the bytes and a metadata dict. The bytes and the absolute range
        they occupy come from one acquisition of the ring's lock
        (:meth:`IQRing.snapshot_newest`), so the numbers beside them describe
        exactly those bytes -- and nothing here moves the loss accounting, so
        reading the dump does not make the decoder's own accounting look better
        than it was.

        The metadata names the tuning and the producing process at each end of
        the range rather than one of each for the whole file. A snapshot that
        straddles a retune holds two frequencies, which no offline reader can
        see in the samples, so :attr:`RingSnapshot.spans_tune` is passed on and
        the segment is labelled rather than written as if it were continuous.
        A snapshot that straddles a process restart has the same problem for the
        same reason, which is :attr:`RingSnapshot.spans_stream`.
        """
        snap = self.ring.snapshot_newest(int(max_samples) * 2)
        return snap.data, {
            "sample_rate": self.sample_rate,
            "encoding": self.encoding.value,
            "kind": self.kind,
            "frequency_hz": self._applied_hz,
            "tune_id": snap.last_tune_id,
            "first_tune_id": snap.first_tune_id,
            "last_tune_id": snap.last_tune_id,
            "spans_tune": snap.spans_tune,
            "stream_id": snap.last_stream_id,
            "first_stream_id": snap.first_stream_id,
            "last_stream_id": snap.last_stream_id,
            "spans_stream": snap.spans_stream,
            "epoch": self.process_epoch,
            "first_byte": snap.first_byte,
            "last_byte": snap.last_byte,
        }

    def drain_iq(self, max_samples: int = 300_000) -> np.ndarray:
        """Newest unconsumed samples, each handed over exactly once.

        The samples-only view of :meth:`drain_block`, for callers that do not
        need to know where the samples came from: a settled-capture reader, a
        power measurement, an offline tool. A streaming *decoder* wants the other
        one, because only that can say whether this block continues the previous
        one or starts a new stream.

        TEMP DIAG: the bounded dump is driven from :meth:`drain_block`, before
        anything is consumed, so it sees the whole ring rather than the leftover
        after this drain.
        """
        return self.drain_block(max_samples).iq

    def drain_block(self, max_samples: int = 300_000) -> IQBlock:
        """Newest unconsumed samples *and* what they are.

        This is the delivery API a streaming consumer uses, and every field of
        the returned record comes out of one acquisition of the ring's lock:

        * the samples, and the absolute half-open range
          ``[first_sample, last_sample)`` they occupy on the ring's monotonic
          write counter -- so two hand-overs can be *tested* for adjacency
          instead of assumed to be;
        * the stream and tuning identity at each end, so a block that straddles
          a retune or a transfer-process restart is recognised as a mixture and
          refused rather than demodulated as one stream;
        * the sample rate and byte encoding, both properties of the producer
          rather than of the array;
        * how many samples the cursor moved past without delivering.

        A streaming decoder must consume each sample exactly once: re-reading a
        snapshot would re-demodulate the same data every frame, and any
        demodulator state (the FM discriminator's previous sample) would be
        applied to a discontinuity. This hands over only the new tail, so no
        work is repeated and no seam appears.

        The previous version set the cursor and then read the newest
        ``min(available, want)`` bytes regardless of where the cursor had been.
        The cursor was therefore write-only, and the method's own docstring was
        false: with a stalled, looping or disconnected source, every call
        returned the same samples again. That fed one capture into the decoder
        over and over, producing frames stamped with fresh times, so the frame
        rate and the freshness indicator both read as healthy while nothing new
        had arrived. Tracking the cursor properly is what makes those indicators
        mean anything.

        ``max_samples`` decides what happens to a backlog. Everything older than
        the newest ``max_samples`` is *consumed and discarded*, because the
        alternative compounds: a decoder that falls behind asks for a bigger
        chunk next time, takes longer to process it, and falls further behind.
        Measured, that turned a 23 ms/field decoder into a 3-second lag while the
        live view showed older and older video. A live viewer should show the
        newest picture it can decode and drop what it cannot keep up with, so the
        default is a little over one field's worth. The size of that drop is
        reported as :attr:`IQBlock.skipped_samples`, and it is a *policy* number:
        it is kept apart from a ring overrun, which is loss.

        The cursor tracks the ring's absolute write counter, which is monotonic
        and never wraps, so there is no wraparound case to reason about. A
        backlog larger than the ring is already gone: the read cannot reach it,
        and the cursor is advanced to the present so the lost span is not
        re-requested forever.

        Reading the counter and the bytes in two calls cannot be made correct,
        only less likely: a write landing in between was counted as consumed by
        the first call and never handed over by the second -- samples lost,
        silently, on a live stream. One lock acquisition is the whole reason the
        range, the identities and the samples are in a single record.

        Nothing about the returned block depends on the TEMP DIAG dump being on.
        """
        if diag.dump_enabled():
            diag.dump_tick(self._dump_snapshot)
        want = int(max_samples) * 2
        with self._drain_lock:
            cursor = int(self._consumed)
            read = self.ring.read_since(cursor, want)
            self._consumed = read.cursor
            self.last_drain_tune_ids = (read.first_tune_id, read.last_tune_id)
            iq = (
                _u8_to_iq(read.data, self.encoding)
                if len(read.data) >= 16
                else np.zeros(0, dtype=np.complex64)
            )
            self._deliveries += 1
            block = IQBlock(
                iq=iq,
                first_sample=read.first_byte // 2,
                last_sample=read.last_byte // 2,
                first_stream_id=read.first_stream_id,
                last_stream_id=read.last_stream_id,
                first_tune_id=read.first_tune_id,
                last_tune_id=read.last_tune_id,
                sample_rate=self.sample_rate,
                encoding=self.encoding,
                skipped_samples=read.skipped_samples,
                reset_reason=self._drain_reset_reason,
                delivery=self._deliveries,
            )
            # Cleared inside the lock, so a reason belongs to exactly one
            # delivery: two consumers draining concurrently would otherwise see
            # the same one twice, and neither could say which hand-over it
            # described.
            self._drain_reset_reason = ""
            self._last_delivery = block
            # TEMP DIAG: absolute ranges in, counts out. Read inside the same
            # lock as the data, so the numbers cannot disagree with the block.
            self._diag_drain(cursor, read, block)
        return block

    def last_delivery(self) -> IQBlock | None:
        """The most recent :meth:`drain_block`, or None before the first.

        For a consumer that already holds the samples and wants their metadata.
        It carries a delivery number, so a caller can tell whether it is looking
        at the hand-over it asked for or at a later one -- which is the
        difference between "these bytes have no position" and "about to describe
        somebody else's bytes".
        """
        with self._drain_lock:
            return self._last_delivery

    def _diag_drain(self, cursor: int, read: RingRead, block: IQBlock) -> None:
        """TEMP DIAG: account for one hand-over. Counts only.

        ``read_since`` takes the newest ``min(pending, max_bytes)`` and then
        moves the cursor to the present, so a backlog longer than the cap is
        consumed and dropped without being named anywhere. That drop is
        invisible from outside -- the block looks like any other -- and it is
        exactly what would silently eat the head of a field. So it is counted
        here, per hand-over, together with the absolute byte range delivered and
        the gap between consecutive hand-overs, which together say whether the
        decoder saw a continuous stream.
        """
        d = self._diag
        ring = self.ring
        samples = block.samples
        # In samples, like everything else counted here. The byte positions were
        # kept in a counter named ``..._samples``, next to ``delivered_samples``
        # and ``skipped_samples`` which are halved, so a drain of a megabyte
        # reported twice the samples it delivered.
        pending = max(0, int(read.cursor) - int(cursor)) // 2
        skipped = int(read.skipped_samples)
        d.bump("drains")
        d.bump("pending_samples", pending)
        d.bump("delivered_samples", samples)
        d.gauge("read_samples_last", samples)
        d.hist("read_kib", samples >> 10)
        if skipped:
            d.bump("skip_events")
            d.bump("skipped_samples", skipped)
            d.hist("skip_kib", skipped >> 11)
        if block.reset_reason:
            d.bump("reset_reason_drains")
            d.gauge("reset_reason", block.reset_reason)
        if self._drain_last_end and read.first_byte > self._drain_last_end:
            gap = read.first_byte - self._drain_last_end
            # Delivery did not resume where it stopped. Normally this is
            # exactly the capped skip just counted -- the two are the same
            # quantity when the cursor is where the last hand-over left it,
            # so their equality is the invariant worth reading. They part
            # company when the cursor moved without a delivery (a
            # reset_drain), which is precisely when they differ.
            d.bump("range_gap_events")
            d.bump("range_gap_samples", gap // 2)
            d.hist("gap_kib", gap >> 11)
        self._drain_last_end = read.cursor
        dropped = ring.dropped_bytes - self._drain_drop_mark
        self._drain_drop_mark = ring.dropped_bytes
        if dropped > 0:
            d.bump("overrun_events")
            d.bump("dropped_samples", dropped // 2)
        over = ring.overwritten_bytes - self._drain_over_mark
        self._drain_over_mark = ring.overwritten_bytes
        if over > 0:
            d.bump("overwritten_samples", over // 2)
        d.gauge("written_bytes", read.cursor)
        d.gauge("tune_id_first", read.first_tune_id)
        d.gauge("tune_id_last", read.last_tune_id)
        d.gauge("stream_id_first", read.first_stream_id)
        d.gauge("stream_id_last", read.last_stream_id)
        d.gauge("epoch", self.process_epoch)
        d.gauge("reclaims", self.reclaims)
        d.gauge("ring_seams", len(ring.diag_seams()))

    def reset_drain(self, reason: str = "") -> None:
        """Discard everything buffered so far without handing it to a decoder.

        This is a real read boundary now, not a counter nobody reads: the next
        :meth:`drain_block` returns nothing until genuinely new samples arrive.
        Used on retune, so a decoder cannot splice the tail of one frequency
        onto the head of the next.

        ``reason`` is a short label carried on the *next* delivery, so a consumer
        can tell "the stream broke on its own" from "somebody asked for a clean
        start here". It is diagnostics, never control: nothing is refused because
        of it.

        Shares :attr:`_drain_lock` with :meth:`drain_block`, and reads the counter
        through the ring rather than touching ``total_written`` directly. A reset
        that landed between a drain's read of the counter and its read of the
        data moved the cursor underneath the copy already in flight, and the
        caller received pre-reset samples believing they were post-reset -- the
        exact splice this method exists to prevent, arriving through the seam.

        A reset that discards a span leaves a real gap, and the next delivery
        reports it as one: the cursor moved, so ``first_sample`` is past the last
        hand-over's end by exactly the discarded size. That is the honest
        accounting -- the samples exist nowhere else -- and it is why the size is
        counted here as well as inferred from the range.
        """
        with self._drain_lock:
            # TEMP DIAG: one read of the counter, as before; the discarded
            # span is derived from it rather than from a second read that
            # could land on a different value.
            written = self.ring.write_position()
            discarded = max(0, written - self._consumed)
            self._consumed = written
            if reason:
                self._drain_reset_reason = str(reason)
        self._diag.bump("drain_resets")
        if discarded:
            self._diag.bump("reset_discarded_samples", discarded // 2)

    def wait_for_bytes(self, n_bytes: int, timeout: float = 2.0) -> bool:
        """Block until the ring's write counter has advanced by ``n_bytes``."""
        target = self.ring.total_written + int(n_bytes)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ring.total_written >= target:
                return True
            self._stop.wait(0.005)
        return self.ring.total_written >= target

    def retune_settled(
        self, frequency_hz: int, n_samples: int, timeout: float = 2.0
    ) -> np.ndarray:
        """Retune, then return ``n_samples`` of IQ captured *after* the retune.

        A HackRF retune restarts the transfer process, so for a moment the ring
        still holds the previous frequency's samples. Taking the newest block
        immediately would blend two frequencies and report a signal that is not
        there -- or, worse, report the old signal at the new frequency. Waiting
        for the byte counter to advance by the amount wanted is what makes the
        returned block provably post-change, since bytes are appended in order
        and the newest ``n`` are the last ``n`` written.

        Returns an empty array when the wait times out, and that is not a
        subtlety. The previous version discarded ``wait_for_bytes``'s result and
        returned the newest samples either way, so a receiver that had stalled
        or gone away handed back the previous frequency's capture and the caller
        scored it as a fresh measurement at the new one. With a radio lent to
        another process this was not an edge case: the hop-walk fallback ran
        entirely in that state and reported a full set of hop results built from
        one pre-scan capture. An empty result and a genuinely quiet channel are
        different answers, and the scanner must be able to tell them apart --
        ``assess_frequency`` already returns None below 4096 samples, so an empty
        capture becomes "could not measure" rather than "nothing there".

        The tune, the wait and the capture are one transaction, holding
        ``_tune_lock`` throughout. They used to be three steps with the lock
        released between them, which let a second tuner -- the sweep tool
        restoring what it found, a UI change, another hop -- move the radio
        before the wait, so the samples returned were the ones *that* caller
        asked for and were scored as a measurement at this frequency. The
        result is the same shape as before; what changed is that nothing can
        happen in the middle of it.
        """
        with self._tune_lock:
            # _tune_locked, not tune(): the lock is already held here and it is
            # a plain Lock, so re-taking it would deadlock. Routing through
            # _tune_locked rather than reaching past it into _apply_tune also
            # means a subclass that refuses a tune at this level -- FileSource --
            # still refuses, instead of being bypassed by this path.
            if not self._tune_locked(int(frequency_hz)):
                return np.zeros(0, dtype=np.complex64)
            if not self.wait_for_bytes(n_samples * 2, timeout=timeout):
                self.stats.read_errors += 1
                self.stats.last_error = (
                    f"no fresh samples at {int(frequency_hz)/1e6:.1f} MHz "
                    f"within {timeout:.1f}s -- measurement discarded, not reported"
                )
                return np.zeros(0, dtype=np.complex64)
            return self.take_iq(n_samples)


class HackrfSource(IQSource):
    """Live IQ from a HackRF via the ``hackrf_transfer`` receive-to-stdout path."""

    kind = "hackrf"

    #: hackrf_transfer writes two's-complement 8-bit I/Q, both to stdout and to
    #: its ``-r`` file. Declared rather than inferred: the format is a property
    #: of the tool, so there is nothing to detect, and a wrong answer here
    #: inverts the waveform while leaving power measurements plausible.
    encoding = IQEncoding.SIGNED_INT8

    def __init__(
        self,
        frequency_hz: int,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        lna_db: int = 32,
        vga_db: int = 16,
        executable: str | None = None,
    ) -> None:
        super().__init__(sample_rate)
        self.frequency_hz = int(frequency_hz)
        # The constructor's frequency is the one this source will be started at,
        # so it counts as the applied one. Without this, the first
        # tune(frequency_hz) would look like a change to be made and reconfigure
        # the receiver, which is wrong: there is nothing to change, and the
        # early return is exactly the behaviour that makes a repeated request
        # cheap.
        self._applied_hz = int(frequency_hz)
        self.lna_gain_db = int(lna_db)
        self.vga_gain_db = int(vga_db)
        self.executable = executable or find_hackrf_transfer()
        self._proc: subprocess.Popen | None = None
        self._err_thread: threading.Thread | None = None
        #: True while the radio has been handed to another process. See
        #: :meth:`release_device`.
        self._released = False
        self._block = bytearray(1 << 22)   # 4 MiB; readinto needs a fixed buffer
        self._view = memoryview(self._block)
        self._stderr_tail: list[str] = []

    def available(self) -> bool:
        return bool(self.executable)

    def describe(self) -> str:
        return f"HackRF @ {self.frequency_hz/1e6:.4f} MHz, {self.sample_rate/1e6:g} MS/s, LNA {self.lna_gain_db}, VGA {self.vga_gain_db}"

    def _cmdline(self) -> list[str]:
        assert self.executable
        # -r must come first: a bare '-' ends option parsing.
        cmd = [
            self.executable,
            "-r", "-",
            "-f", str(int(self.frequency_hz)),
            "-s", str(int(self.sample_rate)),
            "-l", str(int(self.lna_gain_db)),
            "-g", str(int(self.vga_gain_db)),
        ]
        # -a is a Mayhem extension. The stock Great Scott build has no such flag
        # and rejects the whole command line when it sees one, so passing it
        # unconditionally would turn a working install into a dead radio. Probe
        # once per executable instead.
        if self.amp_supported:
            cmd += ["-a", "1" if self.amp_enabled else "0"]
        return cmd

    @property
    def amp_supported(self) -> bool:
        return bool(self.executable) and supported_flags(self.executable) >= {"a"}

    def _spawn(self) -> None:
        assert self.executable
        cmd = self._cmdline()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # bufsize=0 is deliberate and load-bearing. A buffered reader is free
            # to pull more off the OS pipe than the caller asked for, and those
            # extra bytes sit in a buffer that a subsequent readinto() cannot
            # see, so the ring gets a hole. Unbuffered readinto() is the only
            # spelling where "n came back" means "n bytes were consumed".
            bufsize=0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        # The stream identity belongs to *this process*, not to whatever is
        # current at the moment its samples are read. A retune kills the old
        # process and installs a new one while the sampling thread is still
        # blocked in readinto() on the old one's pipe, so the thread wakes up
        # afterwards with bytes from the old transfer. Stamping them with
        # ``self.stream_id`` -- the new process's -- would record the restart in
        # the wrong place, which is to say nowhere: the seam would fall in the
        # gap between the two processes' own writes and the block holding it
        # would look continuous. So the id travels with the handle it describes.
        setattr(self._proc, "_fpv_stream_id", self._begin_stream())
        # stderr gets its own thread, and it has to. `bufsize=0` means the
        # stderr handle is a raw FileIO with no read1(), so a drain on the
        # sampling thread has to use read(n) -- and read() on a pipe blocks
        # until n bytes are available. hackrf_transfer writes its progress
        # about once a second, so read(4096) would stall the IQ reader for a
        # second at a time, and read(1) would stall it forever on a build that
        # writes nothing to stderr at all. Either way the *only* path to the
        # samples stops flowing, with no error and a process still running --
        # which is exactly what it looked like from the outside.
        self._err_thread = threading.Thread(
            target=self._drain_stderr, args=(self._proc,),
            name="hackrf-stderr", daemon=True,
        )
        self._err_thread.start()
        self.stats.running = True
        # TEMP DIAG: every spawn is a new process identity, and a retune is a
        # spawn. The tuning id says the air changed; this says the process did.
        self.process_epoch += 1
        self._diag.bump("spawns")
        self._diag.gauge("epoch", self.process_epoch)

    def _spawn_settled(self) -> bool:
        """Start a transfer and do not return until it is genuinely receiving.

        Must be called with ``_tune_lock`` held.

        Spawning is not enough. ``hackrf_open()`` fails outright with
        ``HackRF not found (-5)`` while the *previous* transfer process is still
        holding the device, and terminating a process does not wait for its USB
        handles to close. So immediately after a retune -- which is the only
        situation where a transfer process is started -- the first spawn is
        expected to fail. Left unretried it produced two distinct wrong answers:
        the hop reported an empty band because no samples arrived inside the
        dwell, and the reader's own backoff respawned at the old timing, so the
        sweep silently lagged the frequency it claimed to be measuring.

        Retrying only on an *open* failure is deliberate. Any other error
        (no device at all, bad arguments, unsupported rate) will not fix itself,
        and a retry loop over those just delays telling the operator what is
        actually wrong.

        The backoff starts at ``OPEN_RETRY_MIN`` and grows, rather than sitting
        at a flat 350 ms, because the common case is that the device is free
        within a few milliseconds and a long fixed wait is charged to every hop
        of a sweep. The ceiling and the attempt count are unchanged, so a device
        that really is still held gets the same total patience as before.
        """
        backoff = OPEN_RETRY_MIN
        for _ in range(OPEN_RETRIES):
            if self._stop.is_set():
                return False
            try:
                self._spawn()
            except Exception as exc:
                self.stats.last_error = f"spawn failed: {exc}"
                self._stop.wait(backoff)
                backoff = min(OPEN_RETRY_MAX, backoff * OPEN_RETRY_GROWTH)
                continue
            base = self.ring.total_written
            deadline = time.monotonic() + OPEN_ARRIVE_WAIT
            while time.monotonic() < deadline:
                p = self._proc
                if p is not None and p.poll() is not None:
                    break                       # died; judge it below
                if self.ring.total_written > base:
                    return True                # samples are arriving
                self._stop.wait(0.005)
            p = self._proc
            if p is not None and p.poll() is None:
                return True                    # alive, just not talking yet
            tail = self._stderr_tail[-1] if self._stderr_tail else ""
            self._kill(self._proc)
            low = tail.lower()
            if "hackrf_open" not in low and "not found" not in low:
                # A real fault, not a device-still-busy. Say so and stop.
                self.stats.last_error = (tail or "receiver exited")[-300:]
                return False
            self.stats.last_error = tail[-300:]
            self.stats.open_retries += 1
            self._stop.wait(backoff)
            backoff = min(OPEN_RETRY_MAX, backoff * OPEN_RETRY_GROWTH)
        return False

    def _kill(self, p: "subprocess.Popen | None" = None) -> None:
        """Stop one transfer process, without touching the reader's I/O.

        Two things this deliberately does not do.

        It does not close ``p.stdout``/``p.stderr``. The sampling thread is
        almost always blocked in ``readinto()`` on exactly that handle, and
        closing a Windows pipe handle while a blocking read is in flight does
        not return: the reader waits for bytes that can no longer arrive, the
        closer waits for the read to return, and the retune that asked for this
        never finishes. That is not a theoretical race -- it is what made the
        *second* retune hang every time, on hardware only, while the first one
        worked. Terminating the process instead gives the reader EOF, and the
        reader closes its own handle once ``readinto`` has come back.

        It does not kill "whatever is current". The reader discovers EOF for the
        process it was reading some time after a retune has already installed
        its replacement, and an unparameterised ``_kill()`` there tore down the
        new process and restarted the whole dance one hop late. The caller names
        the process it means.
        """
        p = self._proc if p is None else p
        if p is None:
            return
        if self._proc is p:
            self._proc = None
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.wait(timeout=1.5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
            # Wait for the kill too. Skipping this and returning straight away
            # leaves a process that is still shutting down -- holding its USB
            # handles, and about to be told by the caller that the radio is free.
            try:
                p.wait(timeout=1.5)
            except Exception:
                pass

    @staticmethod
    def _close_quietly(stream) -> None:
        try:
            if stream is not None and not stream.closed:
                stream.close()
        except Exception:
            pass

    def _teardown(self) -> None:
        self._kill()

    #: A real radio is the one thing here another process can be handed.
    exclusive_use = True

    def release_device(self) -> bool:
        """Free the radio so another process can open it.

        The sampling thread is left running on purpose. It is what makes this
        cheap -- it is the thread that owns the pipe handles, and it is the
        thread that has to be there to read the EOF that killing the transfer
        process produces. What it must not do is reopen the device, or it would
        take the radio straight back out of the other process's hands.

        So the thread is told to stand down, not shut up. :attr:`_released` is
        checked in the respawn branch of :meth:`_run`; without it that branch
        reopens the device about 20 ms after this returns, which is precisely
        long enough to lose the race every time.

        Note what a successful return does and does not promise. It means *this*
        process has let go, which is the part the caller can act on. It does not
        promise the next process will get in: stopping a ``hackrf_transfer``
        that is streaming with ``-r -`` leaves the device refusing new opens for
        an unpredictable period, and nothing done here can shorten that. The
        sweep path deals with it by falling back rather than by waiting -- see
        ``fpv_rf/fastscan.py``, and ``tools/probe_handover.py`` for the
        measurement behind that.

        Returns whether this process has let the radio go. False means it
        is still holding it, and the caller must not start another process
        on the radio as though it were not.
        """
        with self._tune_lock:
            self._released = True
            old, self._proc = self._proc, None
            self._kill(old)
        if old is None:
            return True
        # Wait for *that* process to actually be gone, and for the device to
        # settle. The wait has to name the process this released: ``self._proc``
        # was cleared above on purpose, so waiting on it would return instantly
        # and say the radio was free while the transfer process was still
        # shutting down and still holding its USB handles. That is not a
        # hypothetical -- it is what left ``hackrf_sweep`` blocked forever
        # waiting for a device the previous process had not given up yet, and
        # the scan it was part of reported a 90-second stall instead of a fast
        # answer.
        deadline = time.monotonic() + 5.0
        while old.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if old.poll() is not None:
            # A terminated process still has its USB handles closing, and the
            # next process to ask for the radio inside that window is refused.
            time.sleep(0.15)
            return True
        # Still alive after being terminated *and* killed. Report the truth: the
        # radio is not free, and starting the other tool now would be a race
        # that a caller cannot win.
        return False

    def reclaim_device(self) -> bool:
        """Reopen the radio at whatever frequency is current, and resume.

        Returns whether the device came back. A false here means the radio is
        gone, which the caller should surface rather than paper over: the
        operator is now watching a still picture and would have no idea why.
        """
        with self._tune_lock:
            self._released = False
            self.reclaims += 1           # TEMP DIAG: see IQSource.reclaim_device
            self._diag.bump("reclaims")
            if self._proc is not None:
                return True
            if not self._thread:
                return True
            ok = self._spawn_settled()
            if not ok:
                self._proc = None
            return ok

    def _apply_tune(self) -> bool:
        """Swap in a transfer process at the new frequency.

        Called by ``tune()`` with ``_tune_lock`` already held -- see its
        docstring for why this must not take it again.
        """
        if not self._thread:
            return True
        if self._released:
            # The device belongs to another process. The new frequency is
            # already recorded -- tune() sets it before calling here -- so
            # reclaim_device() will open at the right place. Spawning now would
            # take the radio out from under whoever was lent it.
            return True
        # Detach the *current* process by name. The reader may be blocked in
        # readinto() on it and will clean up after itself; all this side owes the
        # device is that the old process is gone before the new one asks for it,
        # because two processes opening the same HackRF is a race the firmware
        # resolves badly.
        old, self._proc = self._proc, None
        t0 = time.monotonic()
        self._kill(old)
        self.stats.restarts += 1
        ok = self._spawn_settled()
        if not ok:
            self._proc = None
        dt = time.monotonic() - t0
        self.stats.tune_total_s += dt
        self.stats.tune_count += 1
        self.stats.tune_last_ms = dt * 1000.0
        return ok

    def _drain_stderr(self, p: "subprocess.Popen") -> None:
        """Read the child's diagnostics until it exits or the pipe closes.

        Runs on its own thread; see :meth:`_spawn` for why it cannot share the
        sampling thread. Only lines that look like errors are kept as
        ``last_error`` -- ``hackrf_transfer`` reports its sample rate and
        frequency on stderr too, and treating that as a failure would leave the
        UI permanently reporting a problem that is just a startup banner.

        This thread owns its handle and closes it on the way out, so nothing
        else has to reach in and close a handle a read is sitting on.
        """
        err = p.stderr
        if err is None:
            return
        try:
            while True:
                try:
                    data = err.read(4096)
                except Exception:
                    return
                if not data:
                    return
                for line in data.decode("utf-8", "replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    self._stderr_tail.append(line)
                    del self._stderr_tail[:-8]
                    low = line.lower()
                    if any(
                        w in low
                        for w in ("error", "fail", "unable", "could not",
                                  "not found", "busy", "no such", "invalid",
                                  "abort")
                    ):
                        self.stats.last_error = line[-300:]
        finally:
            self._close_quietly(err)

    def _run(self) -> None:
        if not self.executable:
            self.stats.last_error = "hackrf_transfer not found"
            return
        while not self._stop.is_set():
            p = self._proc
            if p is not None and p.poll() is not None:
                # This process has exited. If a retune has already installed a
                # replacement, this is the expected end of the old one and there
                # is nothing to report and nothing to restart -- the reader must
                # not race the retune for the device, and must not treat a normal
                # hop boundary as a fault the operator needs to see.
                if self._proc is p:
                    self.stats.last_error = (
                        self._stderr_tail[-1][-300:]
                        if self._stderr_tail
                        else "receiver exited"
                    )
                    self._kill(p)          # clears self._proc
                else:
                    self._close_quietly(p.stdout)
                    self._kill(p)
                p = None
            if p is None and self._proc is None:
                if self._stop.is_set():
                    break
                if self._released:
                    # The radio is in another process's hands on purpose. Idle
                    # rather than reopen it -- see release_device(). The thread
                    # stays up so it is there to read the EOF and the samples
                    # when reclaim_device() puts the device back.
                    self._stop.wait(0.05)
                    continue
                # Nothing is running and nobody else is starting anything: the
                # device was unplugged, or the process died on its own. Attempt
                # an opportunistic nonblocking acquisition of _tune_lock so we
                # do not block a caller that currently holds it (e.g. retune_settled).
                acquired = self._tune_lock.acquire(blocking=False)
                if not acquired:
                    # Another thread holds the lock (likely a retune in progress).
                    # Do a short stop-aware wait and restart the outer loop,
                    # rereading self._proc -- once a replacement has appeared,
                    # read from it without taking the lock.
                    self._stop.wait(0.01)
                    continue
                try:
                    # _released is re-read *inside* the lock, not just by the
                    # branch above. release_device() takes this same lock, so the
                    # check and the spawn are now one atomic step: before, the
                    # flag could be set after the test and before the spawn, and
                    # this thread would take the radio straight back out of the
                    # hands of the process it had just handed it to. That window
                    # was the entire reason release_device() exists.
                    if self._proc is None and not self._stop.is_set() and not self._released:
                        if not self._spawn_settled():
                            self._stop.wait(0.25)
                finally:
                    self._tune_lock.release()
                p = self._proc
            if p is None or p.stdout is None:
                self._stop.wait(0.02)
                continue
            # Captured *before* the read, from the process being read, and used
            # whatever happens to `self._proc` while this call is in flight. See
            # _spawn: a replacement installed mid-read must not stamp these
            # bytes.
            stream_id = int(getattr(p, "_fpv_stream_id", 0))
            tune_id = self._tune_id
            # A read can end mid-sample; the byte it leaves behind goes back into
            # the front of the buffer so the next read completes that pair. It
            # is taken only if it belongs to *this* process -- a replacement
            # producer's stream starts on a sample boundary of its own.
            held = self._take_carry(p, into=self._block)
            try:
                n = p.stdout.readinto(self._view[1 if held else 0:])
            except Exception as exc:
                self.stats.read_errors += 1
                if self._proc is p:
                    self.stats.last_error = f"read: {exc}"
                self._close_quietly(p.stdout)
                self._kill(p)
                continue
            if not n:
                # EOF: this process has finished. Close its stdout here, on the
                # thread that was the one reading it. Any half-sample it left
                # behind has no partner coming and went with the pipe.
                self._close_quietly(p.stdout)
                self._kill(p)
                continue
            # Whole pairs only: the ring de-interleaves from the front, so a
            # dangling byte here would mis-pair every sample after it. The held
            # byte sits at index 0 and the read landed immediately after it, so
            # the bytes to consider are ``_view[:len(held) + n]`` -- counted from
            # the front, because the pair split has to see the held byte too.
            pairs = self._keep_pairs(self._view[: len(held) + n], p)
            self.ring.write(pairs, tune_id, stream_id)
            self.stats.bytes_total += n
        self.stats.running = False


class FileSource(IQSource):
    """Replay a recorded HackRF ``.u8`` capture at real-time speed.

    This is how the app is exercised with no VTX on the air: the same decode
    and scan paths run, only the samples come from disk.
    """

    kind = "file"

    #: A recorded capture is one frequency. Sweeping it would report the same
    #: signal at every hop, so the scanner must be told to assess it in place.
    retunable = False

    #: Bytes handed to the ring per replay iteration, and therefore the finest
    #: granularity of the pacing below.
    #:
    #: The old code read a whole 1 MiB block -- 524288 complex samples, 52 ms of
    #: a 10 MS/s stream -- pushed all of it into the ring at once, and *then*
    #: waited out its 52 ms. The wait is what made the average rate right, and
    #: everything else about it was wrong: the ring received 52 ms of samples
    #: instantaneously and then nothing for 52 ms, so a consumer that takes
    #: 175000 samples per field was always draining against a buffer that had
    #: just been slammed full. It fell behind, and because the ring counts an
    #: overrun as loss, every replay reported megabytes of dropped samples that
    #: the decoder had in fact consumed -- more visibly wrong than the timing,
    #: and it is what made this look like a fault in the ring rather than a
    #: fault in the producer.
    #:
    #: 64 KiB is 32768 samples, 3.3 ms at 10 MS/s: a small fraction of a field,
    #: fine enough that the ring is never more than a few milliseconds ahead of
    #: the consumer, coarse enough that a syscall every 3 ms costs nothing next
    #: to the decode it is feeding.
    CHUNK_BYTES = 64 << 10

    def __init__(
        self,
        path: str | Path,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        loop: bool = True,
        realtime: bool = True,
        frequency_hz: int = 0,
        encoding: IQEncoding | None = None,
    ) -> None:
        super().__init__(sample_rate)
        # A recording has no amplifier to switch, so the control would be a lie.
        self.amp_supported = False
        self.path = Path(path)
        self.loop = loop
        self.realtime = realtime
        #: The frequency this recording was made at, if known. A file cannot be
        #: retuned, so this is metadata, and the only honest thing to report.
        self.frequency_hz = int(frequency_hz)
        self._applied_hz = int(frequency_hz)
        #: ``None`` means "judge it from the file, and say so". Recordings exist
        #: in both encodings -- hackrf_transfer writes signed and :class:`SimSource`
        #: writes offset-binary -- so unlike hardware there is genuinely nothing to
        #: declare. But the data does not always settle it either (a loud recording
        #: looks the same either way), so the verdict is kept and reported rather
        #: than applied silently.
        self._encoding_choice = encoding
        self.encoding = encoding or IQEncoding.SIGNED_INT8
        self._encoding_reason = "declared" if encoding else "not yet examined"
        #: Set the first time :meth:`_detect_encoding` reaches a verdict, so the
        #: file is decoded one way from its first sample to its last. Re-deciding
        #: per block is how a recording that opens quiet and gets loud decoded
        #: as two different waveforms, with a discontinuity at the point the
        #: verdict flipped.
        self._encoding_settled = encoding is not None

    def _tune_locked(self, frequency_hz: int, *, reconfigure: bool = False) -> bool:
        """Refuse. A recording is one frequency and cannot become another.

        The base class happily stored the new value, which meant a replayed
        capture could be assessed, charted and alerted as a channel it was never
        recorded on -- the same samples reported under a different identity, and
        band and channel names attached to a frequency that had nothing to do
        with them. Declaring ``retunable = False`` was not enough on its own,
        because nothing enforced it: the scan button checked the flag and the
        manual frequency and channel controls did not.

        So this returns False and changes nothing. Correcting the *metadata* of
        a capture whose recorded frequency was mis-entered is a real need, but it
        is a different operation from tuning, and it is the constructor's
        ``frequency_hz`` argument. Conflating them is what made the same file
        describe itself as two different channels.

        The refusal sits at this level rather than in a ``tune`` override so that
        every route into a tune is covered. :meth:`IQSource.retune_settled` is
        the scan's measurement path and goes through ``_tune_locked``, so a
        ``tune``-level override was invisible to it: sweeping a recording would
        have gone straight to ``_apply_tune`` and reported hop results built from
        one fixed capture.
        """
        if int(frequency_hz) != self.frequency_hz:
            return False
        return True

    def set_gains(self, lna_db: int, vga_db: int) -> bool:
        # Recorded IQ is fixed. Storing the numbers would make the UI show a
        # gain the samples were never captured with.
        return True

    def describe(self) -> str:
        where = f" @ {self.frequency_hz/1e6:.4f} MHz" if self.frequency_hz else ""
        return (f"file {self.path.name}{where}, "
                f"{self.sample_rate/1e6:g} MS/s, "
                f"{self.encoding.value} ({self._encoding_reason})")

    def _detect_encoding(self, raw: bytes) -> None:
        """Settle the recording's byte format once, on the first real block.

        Runs before any of the data reaches the ring, so the file is decoded one
        way from its first sample rather than switching convention partway
        through a replay. Detection is confined to files, where the format is
        genuinely unknown; hardware declares its own.

        The "once" is enforced, not just described. This was called on every
        block, and a block is 64 KiB of whatever the recording happens to hold --
        so a file that opens with a quiet passage and gets loud later would be
        declared offset-binary at the start and signed a second in, with the
        decoder handed a discontinuity in the middle of one continuous
        recording. Freezing the first verdict makes the file one waveform, which
        is what a recording is.

        When the data is undecidable -- a loud recording, where the two encodings
        are not distinguishable -- it falls back to signed, the format the radio
        produces and the only one this app's own recordings come from, and says
        so in :meth:`describe`. A file that turns out to be offset-binary can be
        loaded with an explicit ``encoding=``, which wins over anything detected;
        what is not acceptable is pretending the data settled it when it did not.
        """
        if self._encoding_settled:
            return
        verdict = detect_iq_encoding(raw)
        # Settled only now: a raise above left the flag unset, so the next chunk
        # retries detection instead of freezing the default as "not yet examined".
        self._encoding_settled = True
        if verdict.encoding is None:
            self.encoding = IQEncoding.SIGNED_INT8
            self._encoding_reason = f"defaulted, {verdict.reason}"
        else:
            self.encoding = verdict.encoding
            self._encoding_reason = "from the data"

    def _run(self) -> None:
        if not self.path.exists():
            self.stats.last_error = f"capture not found: {self.path}"
            return
        self.stats.running = True
        bytes_per_s = self.sample_rate * 2
        chunk = max(2, self.CHUNK_BYTES)
        period = chunk / bytes_per_s
        # Paced on an absolute schedule rather than "sleep a chunk period after
        # reading", for the reason :meth:`SimSource._run` gives: a wait that
        # overshoots is paid again on every chunk and never washes out, so the
        # long-run rate is quietly below real time. Advancing a deadline and
        # simply not sleeping when it has passed makes the occasional late chunk
        # a catch-up chunk instead of a permanently late stream.
        deadline = time.perf_counter()
        while not self._stop.is_set():
            # Each pass over the file is a new stream, and it has to say so. A
            # looping replay hands over the end of one recording and the start
            # of another with no gap in the ring and no retune: the positions
            # are perfectly adjacent and the samples are not one recording, and
            # nothing but the file's own EOF says so. (An un-looped replay ends
            # the stream instead, which is the same boundary seen from the other
            # side.)
            stream_id = self._begin_stream()
            with open(self.path, "rb", buffering=0) as fh:
                while not self._stop.is_set():
                    t0 = time.monotonic()
                    data = fh.read(chunk)
                    if not data:
                        break
                    # Decide the format before the first byte reaches the ring,
                    # so the file is decoded consistently from end to end rather
                    # than switching convention partway through a replay.
                    self._detect_encoding(data)
                    # Whole samples only: a recording whose length is odd ends
                    # a read mid-pair, and the carried byte belongs to *this*
                    # pass -- the next pass is a new stream, and its first byte
                    # is an I of its own.
                    held = self._take_carry(stream_id)
                    pairs = self._keep_pairs(held + data if held else data, stream_id)
                    self.ring.write(pairs, self._tune_id, stream_id)
                    self.stats.bytes_total += len(data)
                    if self.realtime:
                        deadline += period
                        now = time.monotonic()
                        if deadline < now - period:
                            deadline = now        # fell a whole period behind
                        slack = deadline - now
                        if slack > 0.0005:
                            self._stop.wait(slack)
                    else:
                        # --fast-file: no pacing, but re-reading the file in a
                        # tight loop would spin a core with nothing throttled,
                        # starving the decoder thread of the GIL. wait(0) just
                        # yields the timeslice and keeps full throughput.
                        self._stop.wait(0)
            if not self.loop:
                break
        self.stats.running = False


class SimSource(IQSource):
    """Synthetic IQ, for exercising the app with no hardware and no signal.

    ``video=True`` synthesises a real FM composite signal -- sync pulses, blanking
    and a moving test pattern -- so the whole chain (demod, sync lock, raster,
    scan, alert) runs end to end. ``video=False`` produces band-limited noise,
    which is what a quiet band actually looks like, so the "no signal" path can
    be tested too.

    The signal follows the tuner: the generated content does not depend on
    ``frequency_hz``, so a band search that tunes to a frequency and then looks
    for energy finds energy there. That is what makes the scan logic testable
    without hardware, and it is also why the sim is not a spectrum reference --
    it will not reproduce real capture spectra.

    Set ``signal_hz`` to put the transmitter at one frequency and leave the rest
    of the band empty, which is what a band search actually has to cope with: a
    sweep that cannot tell a populated channel from a quiet one is not a sweep.
    """

    kind = "sim"

    #: The simulator *writes* offset-binary bytes (see ``_run``), so it reads them
    #: back the same way. Declared, because inheriting the hardware default here
    #: would mean the simulator validated itself against the wrong convention --
    #: which is exactly how the hardware bug stayed invisible for so long.
    encoding = IQEncoding.UNSIGNED_OFFSET

    def __init__(
        self,
        frequency_hz: int = 5_802_000_000,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        video: bool = True,
        ntsc: bool = True,
        amplitude: float = 0.20,
        noise: float = 0.012,
        seed: int = 1234,
        signal_hz: int | None = None,
        signal_span_hz: int = 5_000_000,
    ) -> None:
        super().__init__(sample_rate)
        self.frequency_hz = int(frequency_hz)
        # As for HackrfSource: the constructor frequency is the one the source
        # starts at, so it is the applied one and a repeated tune is free.
        self._applied_hz = int(frequency_hz)
        self.video = video
        self.ntsc = ntsc
        self.amplitude = amplitude
        self.noise = noise
        #: When set, only a tuner within ``signal_span_hz`` sees the transmitter.
        self.signal_hz = None if signal_hz is None else int(signal_hz)
        self.signal_span_hz = int(signal_span_hz)
        self._rng = np.random.default_rng(seed)
        self._phase = 0.0
        self._line0 = 0                    # absolute line index of the next block
        self._build_geometry()

    @property
    def signal_present(self) -> bool:
        """Whether the transmitter is inside the span the tuner is looking at.

        ``signal_span_hz`` defaults to the half-span of the sampled bandwidth, so
        "tuned to the transmitter" is the only condition: a hop 12 MHz away
        genuinely cannot see it, and the coarse sweep has to get that right.
        """
        if not self.video:
            return False
        if self.signal_hz is None:
            return True
        return abs(self.frequency_hz - self.signal_hz) <= self.signal_span_hz

    def describe(self) -> str:
        if self.signal_hz is None:
            mode = "synthetic video" if self.video else "synthetic noise"
        else:
            mode = (
                f"synthetic {'video' if self.video else 'noise'} at "
                f"{self.signal_hz/1e6:.4f} MHz"
            )
        return f"sim {mode}, tuned {self.frequency_hz/1e6:.4f} MHz"

    # -- geometry ----------------------------------------------------------
    #
    # The composite is built per *line*, not per sample, and the block size is a
    # whole number of lines. That is what makes the generator fast enough to be
    # worth having: the previous version evaluated the pattern independently for
    # all 131072 samples, so every block paid for a sin(), a power() and five
    # boolean-mask gathers over the whole buffer. The pattern is separable --
    #
    #     luma = A(line) + B(line) * checker_within_line + 0.35 * ramp_within_line
    #
    # because the checkerboard is ``(gx + gy) % 2`` and ``gx`` depends only on
    # position within the line while ``gy`` depends only on the line. So the
    # within-line part is two fixed vectors and each line costs two scalar
    # multiplies, broadcast over a row.

    def _build_geometry(self) -> None:
        fs = float(self.sample_rate)
        self._line_rate = 15_734.0 if self.ntsc else 15_625.0
        self._lpf = 240 if self.ntsc else 288
        self._blank_lines = 22
        self._vis_lines = self._lpf - self._blank_lines
        sync_frac = (4.7 if self.ntsc else 5.0) * 1e-6 * self._line_rate
        act_start = 0.145 if self.ntsc else 0.164
        act_span = 0.828 if self.ntsc else 0.813
        self._line_n = int(round(fs / self._line_rate))       # samples per line
        L = self._line_n
        self._sync_n = max(1, int(round(sync_frac * L)))
        a0 = int(round(act_start * L))
        a1 = int(round((act_start + act_span) * L))
        a1 = min(a1, L)
        self._active = np.zeros(L, dtype=bool)
        self._active[a0:a1] = True

        within = np.arange(L, dtype=np.float32) / np.float32(L)
        u = np.clip((within - np.float32(act_start)) / np.float32(act_span), 0.0, 1.0)
        gx = np.clip(u * np.float32(8.0), 0, 7).astype(np.int32)
        cp = (gx % 2).astype(np.float32)
        BLACK, WHITE, BLANK, SYNC = 0.00, 0.88, 0.12, 1.00
        gamma = 2.2
        # The pattern is fed through gamma before being scaled to WHITE, so the
        # luma that lands on the blanking level is BLANK/WHITE un-gamma'd. Folding
        # it in here means the per-sample branch for "not picture" disappears:
        # a blanked sample is just another luma value, and the whole line becomes
        # one gamma gather with no boolean `where` over the sample axis.
        blank_luma = np.float32((BLANK / WHITE) ** (1.0 / gamma))
        self._blank_luma = blank_luma
        bias = np.where(self._active, np.float32(0.0), blank_luma)
        # the two within-line terms, pre-summed with that bias
        self._basis_cp = np.where(self._active, np.float32(0.12) * cp, np.float32(0.0))
        self._basis_u = (np.where(self._active, np.float32(0.18) * u, np.float32(0.0)) + bias)
        self._level_white = np.float32(WHITE)
        self._level_sync = np.float32(SYNC)

        # gamma 2.2 on a lookup table: the input is a 0..1 luma, so 2048 entries
        # is finer than the 8-bit output can show, and it turns the most
        # expensive transcendental in the generator into a gather.
        q = 2048
        self._gamma_lut = np.power(
            np.arange(q, dtype=np.float32) / np.float32(q - 1), np.float32(gamma)
        )
        self._lut_q = q

        # ~10 ms of samples per block, rounded to whole lines
        self._lines_per_block = max(1, int(round(fs * 0.010) / L))
        self._block_n = self._lines_per_block * L
        self._block = bytearray(self._block_n * 2)
        # interleaved I/Q scratch, float32 so the cos/sin and the quantise both
        # run at half the bandwidth of a complex128 temporary
        self._scratch_iq = np.empty((self._block_n, 2), dtype=np.float32)
        self._phase_buf = np.empty(self._block_n, dtype=np.float32)
        self._dev_buf = np.empty(self._block_n, dtype=np.float32)

    def _deviation(self, nlines: int) -> np.ndarray:
        """Per-sample FM deviation (radians) for ``nlines`` whole video lines.

        Timings follow the standard (NTSC shown): sync tip 0.000-0.074 of a line,
        back porch to 0.145, active picture 0.145-0.973, with 22 blanked lines
        per field for vertical blanking.

        Levels are calibrated against the known-good capture rather than the
        textbook, because two details decide whether sync is findable at all:

        * **Sync sits at the top of the distribution and is flat-topped.**
          Measured on the real capture, the sync level is 3.1243 with a standard
          deviation of 0.0099 -- a hard, flat plateau, while the picture median
          sits at 4% of the picture's range. The detector's percentile threshold
          therefore lands cleanly on sync.
        * **The picture is mostly dark.** Its 99.5th percentile only just
          reaches the sync level, so almost no picture sample competes with the
          sync plateau for the top of the distribution. A test pattern that
          filled the line with mid-to-bright levels instead would put thousands
          of samples at the threshold, and the mask would break into fragments
          at half-line spacing -- which is exactly the failure it produces here.

        So the pattern is gamma-shaped to keep highlights sparse.

        The deviation is centred (``amplitude * (level - 0.5)``) so the carrier
        sits mid-band, and the step stays well inside +-pi so the discriminator
        never wraps. Note this makes a *narrowband* FM signal: at 0.20 rad per
        sample the occupied bandwidth is a few hundred kHz, not the ~15 MHz a
        real VTX uses. That is a deliberate simplification -- it decodes
        identically through this chain, and the alternative pushes the
        per-sample step close enough to pi that wrap-around recovery starts to
        matter. It does mean the sim will not reproduce real capture spectra.
        """
        L = self._line_n
        lines = self._line0 + np.arange(nlines, dtype=np.int64)
        self._line0 += nlines
        t_abs = self._line0 * L / float(self.sample_rate)

        fld = np.mod(lines, self._lpf)
        vis = fld < self._vis_lines
        gy = np.clip((fld / self._vis_lines) * 8.0, 0, 7).astype(np.int64)
        gp = (gy & 1).astype(np.float32)
        sweep = (0.5 + 0.5 * np.sin(2.0 * np.pi * (fld / self._lpf + t_abs * 0.8))).astype(
            np.float32
        )
        # luma = A + B*checker_within_line + ramp_within_line
        #
        # The coefficients are much smaller than a broadcast test pattern would
        # use, and that is deliberate rather than timid. A discriminator step is
        # proportional to the *video* step, so a pattern with eight hard
        # black-to-white checker transitions per line puts eight steps of 0.35
        # luma into the demodulator -- each bigger than the sync pulse itself.
        # Those crossings then land in the top few percent of the distribution
        # alongside sync, the threshold stops isolating the sync plateau, and the
        # detector locks onto picture edges instead. Worse, the edges are
        # line-periodic, so the result still *looks* perfect: quality 0.99 and a
        # clean 636-sample spacing, which is exactly how this went unnoticed.
        #
        # Real FPV video does not do this. On the known-good capture the picture
        # median sits near black and its 99.5th percentile is 3.1367 against a
        # sync level of 3.1243 -- only 0.5% of picture samples reach sync. The
        # pattern below keeps the same property: highlights stay sparse and well
        # below the sync tip, so a detected pulse really is a pulse.
        a = (0.15 + 0.12 * gp + 0.15 * sweep).astype(np.float32)
        b = (0.12 * (1.0 - 2.0 * gp)).astype(np.float32)
        # Vertical blanking folds into those per-line scalars, so it costs
        # nothing per sample: a blanked line is a line with no checker whose
        # luma lands on the blanking level, and the basis carries the bias that
        # does it. Both are scalars-per-line, so the whole field is still one
        # vectorised pass with no boolean branch over the sample axis.
        blank = self._blank_luma
        a = np.where(vis, a, blank).astype(np.float32)[:, None]
        b = np.where(vis, b, np.float32(0.0)).astype(np.float32)[:, None]

        lin = a + b * self._basis_cp
        lin += self._basis_u                               # carries the blank bias
        q = self._lut_q
        level = self._level_white * self._gamma_lut[
            np.clip((lin * q).astype(np.int32), 0, q - 1)
        ]
        level[:, : self._sync_n] = self._level_sync        # sync wins over all
        # straight to deviation, so no separate centre-and-scale pass
        level -= np.float32(0.5)
        level *= self.amplitude
        return level.reshape(-1)

    def _run(self) -> None:
        self.stats.running = True
        two_pi = np.float64(2.0 * np.pi)
        n = self._block_n
        iq = self._scratch_iq
        ph = self._phase_buf
        period = n / float(self.sample_rate)
        acc = np.empty(n, dtype=np.float64)
        # Pacing on an absolute schedule rather than "sleep a block period after
        # generating": a wait that overshoots by a few ms would otherwise be paid
        # again on every block, and the error never washes out. Advancing a
        # fixed deadline and simply not sleeping when it has already passed makes
        # the long-run average the right one -- the occasional late block is
        # followed by a catch-up block instead of a permanently late stream.
        deadline = time.perf_counter()
        while not self._stop.is_set():
            t_start = time.perf_counter()
            freq_at_start = self.frequency_hz
            if self.signal_present:
                dev = self._deviation(self._lines_per_block)
            else:
                # Quiet band. The trick is that the demodulator sees a phase, and
                # the spectrum of the *signal* is whatever the phase does. A
                # random walk in the deviation is a Lorentzian -- a huge hump at
                # DC falling as 1/f^2 -- which is nothing like thermal noise and
                # gives a band search an easy false positive to trip over near
                # the centre of the span.
                #
                # So the deviation is a first *difference* of white noise, of
                # length n+1. Its cumsum is then 3.0*white (plus a constant), so
                # the phase is uniformly distributed sample to sample and the
                # spectrum is flat across the whole span. 3.0 rad of phase spread
                # is ~4.8 MHz RMS deviation at 10 MS/s, which fills the span
                # without aliasing against it.
                self._line0 += self._lines_per_block
                w = self._rng.standard_normal(n + 1, dtype=np.float32)
                dev = w[1:] - w[:-1]
                dev *= np.float32(3.0)
            # Four things here are load-bearing.
            #
            # 1. cumsum in float64. A float32 cumsum over a 100k-sample block
            #    reaches ~3000 rad, where float32 spacing is 2.4e-4 rad, and the
            #    accumulated rounding random-walks far past the 0.03 rad signal
            #    step. Nothing downstream can recover a signal destroyed there.
            # 2. Carry the phase across blocks as the *last scalar sample*, not
            #    as the previous block's array. Adding a whole array of phase
            #    values to the new cumsum makes the offset vary sample to
            #    sample, which scrambles the ramp from the second block onward.
            # 3. Modulo 2*pi *before* adding the phase noise, so the noise is
            #    never itself wrapped and folded back into the signal.
            # 4. Only then narrow to float32, for the cos/sin and the quantise.
            #    2*pi in float32 is still good to 4e-7 rad, which is 1e-5 of the
            #    signal step, and it halves the bandwidth of the two hottest
            #    stages.
            np.cumsum(dev, dtype=np.float64, out=acc)
            acc += self._phase
            np.mod(acc, two_pi, out=acc)
            self._phase = float(acc[-1])
            np.copyto(ph, acc, casting="unsafe")
            ph += self._rng.standard_normal(n, dtype=np.float32) * np.float32(self.noise)
            # straight into the interleaved I/Q scratch: no complex temporary,
            # and cos/sin each land straight in their own column
            np.cos(ph, out=iq[:, 0])
            np.sin(ph, out=iq[:, 1])
            np.multiply(iq, np.float32(127.0), out=iq)
            np.rint(iq, out=iq)
            np.add(iq, np.float32(128.0), out=iq)          # -> 1..255, no clip
            u8 = iq.astype(np.uint8)
            # A retune part-way through a block would hand a band search a block
            # full of the *previous* hop's frequency, and it would then report
            # the signal one hop to the side of where it really is. 8.2 ms of
            # wasted generation to avoid that is the right trade during a scan;
            # the cost is paid once per retune, not once per block.
            if self.frequency_hz != freq_at_start:
                continue
            self.ring.write(u8.tobytes(), self._tune_id, self.stream_id)
            self.stats.bytes_total += u8.size
            # Pace to real time, allowing for what generation actually cost.
            # Waiting a flat n/fs *after* generating would hold the source below
            # real time and starve the decoder.
            deadline += period
            now = time.perf_counter()
            if deadline < now - period:
                deadline = now            # fell a whole period behind: resync
            slack = deadline - now
            if slack > 0.0005:
                self._stop.wait(slack)
        self.stats.running = False


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def make_source(
    mode: str,
    frequency_hz: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    lna_db: int = 32,
    vga_db: int = 16,
    file_path: str | None = None,
    sim_video: bool = True,
    executable: str | None = None,
) -> IQSource:
    """Build a source by name: ``hackrf`` | ``file`` | ``sim``."""
    m = mode.lower()
    if m == "file":
        if not file_path:
            raise ValueError("file source needs a capture path")
        return FileSource(file_path, sample_rate)
    if m == "sim":
        return SimSource(frequency_hz, sample_rate, video=sim_video)
    if m == "hackrf":
        return HackrfSource(frequency_hz, sample_rate, lna_db, vga_db, executable)
    raise ValueError(f"unknown source mode {mode!r}")


if __name__ == "__main__":  # tiny self-check
    src = SimSource()
    src.start()
    time.sleep(0.4)
    iq = src.take_iq(200_000)
    src.stop()
    print(f"sim produced {iq.size} samples, mean|IQ|={np.abs(iq).mean():.3f}")
    sys.exit(0 if iq.size > 1000 else 1)
