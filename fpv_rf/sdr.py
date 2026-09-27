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
from functools import lru_cache
from pathlib import Path

import numpy as np
import re

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

#: Where ``hackrf_transfer`` is looked for last, after ``$HACKRF_TRANSFER`` and
#: the copy beside this program. These are globs rather than fixed paths because
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
    """
    try:
        r = subprocess.run(
            [executable, "-h"], capture_output=True, text=True, timeout=10
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
    """Locate ``hackrf_transfer``: env override, beside the app, PATH, then known installs.

    A copy sitting next to the executable is preferred over ``PATH`` on purpose.
    A self-contained build that silently picked up a *different* version of the
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


class IQRing:
    """Byte ring buffer handing contiguous IQ blocks to the decoder.

    The reader thread appends; the decoder takes the newest ``n`` bytes as one
    contiguous copy. On overflow the *oldest* data is discarded, because for a
    live view stale video is worse than a dropped frame.
    """

    def __init__(self, size: int = RING_BYTES) -> None:
        self._buf = bytearray(size)
        self._size = size
        self._wpos = 0
        self._count = 0
        self._lock = threading.Lock()
        self.dropped_bytes = 0
        self.total_written = 0

    def write(self, data: bytes | memoryview) -> int:
        n = len(data)
        with self._lock:
            if n > self._size:
                # Far more than we can hold; keep only the tail.
                self.dropped_bytes += n - (self._size // 2)
                data = memoryview(data)[n - (self._size // 2) :]
                n = len(data)
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
                self.dropped_bytes += over
            self._count = min(self._count + n, self._size)
            self.total_written += n
        return n

    def available(self) -> int:
        with self._lock:
            return self._count

    def read_newest(self, n: int) -> bytearray:
        """Newest ``n`` bytes, oldest-first. Returns fewer if that is all there is."""
        with self._lock:
            n = min(n, self._count)
            if n <= 0:
                return bytearray()
            end = self._wpos
            start = (end - n) % self._size
            if start + n <= self._size:
                return bytearray(self._buf[start : start + n])
            first = self._size - start
            return bytearray(self._buf[start:] + self._buf[: n - first])


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


def _u8_to_iq(raw: bytes | bytearray | memoryview) -> np.ndarray:
    """Interleaved uint8 IQ (as hackrf writes it) to normalised complex64.

    This sits directly in the streaming path -- it runs on every field, so its
    cost is a fixed slice of the 16.7 ms budget. The obvious spelling allocates
    five full-size temporaries:

        frombuffer -> astype(float32) -> subtract -> divide -> reshape
                  -> two strided reads -> complex128 temp -> astype(complex64)

    which measured ~7 ms for a 250k-sample chunk. The version below does two
    conversions into the real and imaginary views and nothing else, so there is
    no complex128 intermediate and no separate scale pass; the de-interleave is
    the single strided copy.
    """
    arr = np.frombuffer(raw, dtype=np.uint8)
    n = arr.size // 2
    if n == 0:
        return np.zeros(0, dtype=np.complex64)
    iq = np.empty(n, dtype=np.complex64)
    re, im = iq.real, iq.imag
    k = np.float32(1.0 / 127.5)
    np.multiply(arr[0 : 2 * n : 2], k, out=re)
    np.multiply(arr[1 : 2 * n : 2], k, out=im)
    re -= np.float32(1.0)
    im -= np.float32(1.0)
    return iq


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

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = int(sample_rate)
        self.ring = IQRing()
        self.stats = SourceStats()
        self.frequency_hz = 0
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
        self._timer = TimerResolution()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.stats = SourceStats()
        self._timer.acquire()
        self._thread = threading.Thread(target=self._run, name=f"iq-{self.kind}", daemon=True)
        self._thread.start()

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

    @abstractmethod
    def _run(self) -> None:
        ...

    # -- control -----------------------------------------------------------

    def tune(self, frequency_hz: int) -> bool:
        """Retune. Returns True if the change took effect.

        This is the *only* entry point, and it owns ``_tune_lock`` for the whole
        operation. Subclasses implement ``_apply_tune`` and must not take the
        lock again: it is a plain ``Lock``, so re-acquiring it on the same
        thread blocks forever. That mistake is invisible until the first retune
        to a genuinely different frequency, because a retune to the frequency
        already tuned returns early and never reaches the second acquisition.
        """
        with self._tune_lock:
            if int(frequency_hz) == self.frequency_hz:
                return True
            self.frequency_hz = int(frequency_hz)
            return self._apply_tune()

    def _apply_tune(self) -> bool:
        return True

    def set_gains(self, lna_db: int, vga_db: int) -> None:
        self.lna_gain_db = int(lna_db)
        self.vga_gain_db = int(vga_db)

    def describe(self) -> str:
        return f"{self.kind} @ {self.frequency_hz/1e6:.1f} MHz, {self.sample_rate/1e6:g} MS/s"

    def take_iq(self, n_samples: int) -> np.ndarray:
        """Newest ``n_samples`` complex samples as complex64."""
        raw = self.ring.read_newest(int(n_samples) * 2)
        if len(raw) < 16:
            return np.zeros(0, dtype=np.complex64)
        return _u8_to_iq(raw)

    def drain_iq(self, max_samples: int = 300_000) -> np.ndarray:
        """Return the newest unconsumed samples, each handed over exactly once.

        A streaming decoder must consume each sample exactly once: re-reading a
        snapshot would re-demodulate the same data every frame, and any
        demodulator state (the FM discriminator's previous sample) would be
        applied to a discontinuity. This hands over only the new tail, so no
        work is repeated and no seam appears.

        ``max_samples`` also decides what happens to a backlog. Everything older
        than the newest ``max_samples`` is *consumed and discarded*, because the
        alternative compounds: a decoder that falls behind asks for a bigger
        chunk next time, takes longer to process it, and falls further behind.
        Measured, that turned a 23 ms/field decoder into a 3-second lag while the
        live view showed older and older video. A live viewer should show the
        newest picture it can decode and drop what it cannot keep up with, so the
        default is a little over one field's worth.
        """
        want = int(max_samples) * 2
        self._consumed = self.ring.total_written
        raw = self.ring.read_newest(min(self.ring.available(), want))
        if len(raw) < 16:
            return np.zeros(0, dtype=np.complex64)
        return _u8_to_iq(raw)

    def reset_drain(self) -> None:
        self._consumed = self.ring.total_written

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
        """
        if not self.tune(int(frequency_hz)):
            return np.zeros(0, dtype=np.complex64)
        self.wait_for_bytes(n_samples * 2, timeout=timeout)
        return self.take_iq(n_samples)


class HackrfSource(IQSource):
    """Live IQ from a HackRF via the ``hackrf_transfer`` receive-to-stdout path."""

    kind = "hackrf"

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
        self.lna_gain_db = int(lna_db)
        self.vga_gain_db = int(vga_db)
        self.executable = executable or find_hackrf_transfer()
        self._proc: subprocess.Popen | None = None
        self._err_thread: threading.Thread | None = None
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

    @staticmethod
    def _close_quietly(stream) -> None:
        try:
            if stream is not None and not stream.closed:
                stream.close()
        except Exception:
            pass

    def _teardown(self) -> None:
        self._kill()

    def _apply_tune(self) -> bool:
        """Swap in a transfer process at the new frequency.

        Called by ``tune()`` with ``_tune_lock`` already held -- see its
        docstring for why this must not take it again.
        """
        if not self._thread:
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
                # Nothing is running and nobody else is starting anything: the
                # device was unplugged, or the process died on its own. Take the
                # lock so this cannot overlap a retune's spawn.
                with self._tune_lock:
                    if self._proc is None and not self._stop.is_set():
                        if not self._spawn_settled():
                            self._stop.wait(0.25)
                p = self._proc
            if p is None or p.stdout is None:
                self._stop.wait(0.02)
                continue
            try:
                n = p.stdout.readinto(self._view)
            except Exception as exc:
                self.stats.read_errors += 1
                if self._proc is p:
                    self.stats.last_error = f"read: {exc}"
                self._close_quietly(p.stdout)
                self._kill(p)
                continue
            if not n:
                # EOF: this process has finished. Close its stdout here, on the
                # thread that was the one reading it.
                self._close_quietly(p.stdout)
                self._kill(p)
                continue
            self.ring.write(self._view[:n])
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

    def __init__(
        self,
        path: str | Path,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        loop: bool = True,
        realtime: bool = True,
        frequency_hz: int = 0,
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

    def describe(self) -> str:
        where = f" @ {self.frequency_hz/1e6:.4f} MHz" if self.frequency_hz else ""
        return f"file {self.path.name}{where}, {self.sample_rate/1e6:g} MS/s"

    def _run(self) -> None:
        if not self.path.exists():
            self.stats.last_error = f"capture not found: {self.path}"
            return
        self.stats.running = True
        bytes_per_s = self.sample_rate * 2
        while not self._stop.is_set():
            with open(self.path, "rb", buffering=0) as fh:
                while not self._stop.is_set():
                    want = min(len(self._block_view()), RING_BYTES)
                    t0 = time.monotonic()
                    data = fh.read(want)
                    if not data:
                        break
                    self.ring.write(data)
                    self.stats.bytes_total += len(data)
                    if self.realtime:
                        target = len(data) / bytes_per_s
                        delay = target - (time.monotonic() - t0)
                        if delay > 0:
                            self._stop.wait(delay)
            if not self.loop:
                break
        self.stats.running = False

    def _block_view(self) -> bytearray:
        if not hasattr(self, "_blk"):
            self._blk = bytearray(1 << 20)
        return self._blk


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

    def describe(self) -> str:
        mode = "synthetic video" if self.video else "synthetic noise"
        return f"sim {mode} @ {self.frequency_hz/1e6:.4f} MHz"

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
            self.ring.write(u8.tobytes())
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
