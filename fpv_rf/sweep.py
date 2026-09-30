"""Fast band search by sweeping the radio open, instead of reopening it per hop.

The original scanner walks the band one hop at a time, and on hardware each hop
costs about 360 ms -- almost all of it waiting for the USB device to re-open.
Measured on a HackRF One, 5650-5950 MHz took 12.5 s across 34 hops, and the
breakdown was:

    _kill (terminate + wait)      1.4 ms
    _spawn (Popen)                5.6 ms
    waiting for device open       352.6 ms   <- 98% of the hop
    analysis                      8.6 ms

Process creation is not the problem, so nothing in the transfer path can be
optimised away: the cost is the USB re-open, and it is paid per hop no matter
how little else happens. Reaching a one-to-two second sweep needs the radio to
stay open across the whole band.

The Mayhem firmware ships a tool that does exactly that. ``hackrf_sweep``
retunes inside one open device and reports an FFT magnitude per frequency bin,
which makes a sweep a single process launch:

    5650-5950 MHz, 1 MHz bins      0.12 s
    same, sustained                38 ms per full-band sweep

That is about a hundred times faster, and it is fast enough to be worth
watching rather than waiting for.

Three things about the tool shape the code below.

*It needs a DLL the bundle does not ship.* ``hackrf_sweep.exe`` imports
``libfftw3f-3.dll`` and the Mayhem release omits it, so the tool exits
``0xC0000135`` (STATUS_DLL_NOT_FOUND) and prints nothing at all. Windows finds
a DLL by file name but resolves symbols by export name, so a copy of any
single-precision FFTW under the expected name satisfies the loader -- provided
it really exports the single-precision API. :func:`fftw_exports_fft_symbols`
checks that by reading the PE export table rather than trusting the file name,
because a double-precision build has the same shape and none of the symbols.

*It will not honour the span you ask for.* The 20 MS/s sample rate forces the
sweep onto a 20 MHz grid, so ``-f 5800:5810`` really covers 5800-5820. Coverage
is therefore measured from the output and reported, never assumed.

*Its bin width is derived, not requested.* Asking for 500 kHz bins yields
0.379 MHz bins, so every width used here is read back out of ``hz_low`` and
``hz_high`` instead of from the argument.

Nothing in the app depends on this module being present. If the sweep tool is
missing or will not start, :func:`available` returns False and the caller keeps
using :class:`fpv_rf.scan.BandScanner`, which is slower but needs nothing but
``hackrf_transfer``.
"""

from __future__ import annotations

import os
import math
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import bands, dsp, scan

#: Sample rate the tool forces, and with it the granularity of the sweep grid.
#: Requested spans are rounded out to a multiple of this, so a request for
#: 5800-5810 MHz is honoured as 5800-5820 MHz. Measured, not documented.
TOOL_GRID_HZ = 20_000_000

#: Narrowest FFT bin the tool accepts, in Hz.
MIN_BIN_HZ = 2_445

#: Bin width asked for. 1 MHz is a reasonable compromise: an analogue FPV
#: transmission is ~25 MHz wide so it fills ~25 bins, and FPV channels sit tens
#: of MHz apart, so there is no confusion between them, while 300 bins across
#: the US band is still cheap to hold in memory.
BIN_HZ = 1_000_000

#: The single-precision FFTW entry points ``hackrf_sweep.exe`` was linked
#: against. A DLL exporting all of these can stand in for
#: ``libfftw3f-3.dll``; one that does not, cannot.
REQUIRED_FFTW_SYMBOLS = (
    "fftwf_malloc",
    "fftwf_free",
    "fftwf_plan_dft_r2c_1d",
    "fftwf_plan_dft_c2r_1d",
    "fftwf_execute_dft",
    "fftwf_destroy_plan",
)

#: Where a staged copy of the tool is kept. The user's Mayhem folder is never
#: written to -- it may be somewhere read-only, and putting a downloaded DLL
#: beside someone's firmware is not a thing to do unasked.
def _cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    p = Path(base) / "FPV-RF" / "sweep-tool"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Finding the tool, and making it runnable
# ---------------------------------------------------------------------------


def find_sweep_tool() -> Path | None:
    """Locate ``hackrf_sweep.exe``.

    Looked for beside ``hackrf_transfer.exe``, because that is the only
    directory known to hold a working Mayhem utils folder, and beside the
    running interpreter for the packaged build where the folder was bundled.
    """
    from .sdr import find_hackrf_transfer

    candidates: list[Path] = []
    transfer = find_hackrf_transfer()
    if transfer:
        candidates.append(Path(transfer).with_name("hackrf_sweep.exe"))
    here = Path(getattr(sys, "_MEIPASS", "") or Path(__file__).resolve().parent)
    candidates.append(here / "hackrf_sweep.exe")
    candidates.append(here / "utils" / "hackrf_sweep.exe")
    env = os.environ.get("HACKRF_SWEEP")
    if env:
        candidates.insert(0, Path(env))

    for c in candidates:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None


def pe_exports(path: Path) -> set[str]:
    """Export names from a PE image, or an empty set if it has no exports.

    Windows resolves an import by the DLL's *file name* and then binds symbols
    by *export name*, which is the whole reason a renamed copy of another
    FFTW build can satisfy ``libfftw3f-3.dll``. Reading the table is how that
    claim gets checked instead of assumed.
    """
    try:
        b = path.read_bytes()
        if b[:2] != b"MZ":
            return set()
        pe = struct.unpack_from("<I", b, 0x3C)[0]
        if b[pe:pe + 4] != b"PE\0\0":
            return set()
        nsec = struct.unpack_from("<H", b, pe + 6)[0]
        optsz = struct.unpack_from("<H", b, pe + 20)[0]
        opt = pe + 24
        magic = struct.unpack_from("<H", b, opt)[0]
        # Data directory 0 is the export table; its offset differs for PE32
        # and PE32+.
        dd = opt + (112 if magic == 0x20B else 96)
        exp_rva, _sz = struct.unpack_from("<II", b, dd)
        if not exp_rva:
            return set()

        secs: list[tuple[int, int, int]] = []
        so = opt + optsz
        for i in range(nsec):
            o = so + i * 40
            vsz, va, rsz, ra = struct.unpack_from("<IIII", b, o + 8)
            secs.append((va, max(vsz, rsz), ra))

        def r2o(rva: int) -> int:
            for va, span, ra in secs:
                if va <= rva < va + span:
                    return ra + (rva - va)
            raise ValueError("rva outside every section")

        nnames = struct.unpack_from("<I", b, r2o(exp_rva) + 24)[0]
        names_off = r2o(struct.unpack_from("<I", b, r2o(exp_rva) + 32)[0])
        out: set[str] = set()
        for i in range(nnames):
            p = r2o(struct.unpack_from("<I", b, names_off + i * 4)[0])
            out.add(b[p:b.index(b"\0", p)].decode("latin1"))
        return out
    except Exception:
        return set()


def fftw_exports_fft_symbols(path: Path) -> bool:
    """True when this DLL can stand in for ``libfftw3f-3.dll``.

    A double-precision FFTW has a comparable export table, so the name is no
    guide at all -- hence the symbol list.
    """
    ex = pe_exports(path)
    return all(s in ex for s in REQUIRED_FFTW_SYMBOLS)


@lru_cache(maxsize=1)
def find_fftw() -> Path | None:
    """Find a single-precision FFTW DLL, verified by its exports.

    A DLL already sitting beside the tool under the right name wins outright.
    Otherwise a short list of likely homes is tried, and the first one that
    actually exports the single-precision API is taken. This machine turned out
    to have one via radioconda, named ``fftw3f.dll``.
    """
    tool = find_sweep_tool()
    if tool is not None:
        beside = tool.with_name("libfftw3f-3.dll")
        if beside.is_file():
            if fftw_exports_fft_symbols(beside):
                return beside
    roots = []
    for env in ("CONDA_PREFIX",):
        if os.environ.get(env):
            roots.append(Path(os.environ[env]))
    roots += [
        Path(sys.prefix),
        Path(r"C:\ProgramData\radioconda"),
        Path(r"C:\ProgramData\miniconda3"),
        Path(r"C:\ProgramData\Anaconda3"),
        Path(r"C:\Program Files\GNU Radio"),
    ]
    names = ("libfftw3f-3.dll", "fftw3f.dll", "libfftw3f.dll")
    for root in roots:
        for sub in ("", "Library/bin", "bin", "Scripts/../Library/bin"):
            d = (root / sub) if sub else root
            for nm in names:
                p = d / nm
                try:
                    if p.is_file() and fftw_exports_fft_symbols(p):
                        return p
                except OSError:
                    continue
    return None


@lru_cache(maxsize=1)
def runnable_tool() -> Path | None:
    """A copy of ``hackrf_sweep.exe`` that will actually start, or None.

    Returns the tool's own path when it already has its DLL beside it.
    Otherwise stages the tool plus a renamed, export-verified FFTW into the
    app's own cache directory. The Mayhem folder is only ever read.
    """
    tool = find_sweep_tool()
    if tool is None:
        return None
    dll = tool.with_name("libfftw3f-3.dll")
    if dll.is_file():
        return tool
    src = find_fftw()
    if src is None:
        return None
    cache = _cache_dir()
    try:
        staged_exe = cache / "hackrf_sweep.exe"
        staged_dll = cache / "libfftw3f-3.dll"
        if not staged_dll.is_file() or not fftw_exports_fft_symbols(staged_dll):
            shutil.copy2(src, staged_dll)
        if not staged_exe.is_file():
            shutil.copy2(tool, staged_exe)
    except OSError:
        return None
    return staged_exe


def _probe(tool: Path) -> bool:
    """Will this tool start at all?

    Asked with ``-h`` rather than with a real one-megahertz sweep, and that is
    the load-bearing part. The question this function answers is "can the tool
    run", and the thing that stops it running is a DLL it cannot load -- the
    Mayhem bundle omits ``libfftw3f-3.dll``, so the image fails to start with
    ``0xC0000135`` before it executes a single instruction. ``-h`` settles that
    in about 20 ms and never touches the radio.

    An earlier version of this asked for an actual sweep instead, which was
    wrong in a way that only showed up in the app. While the app is streaming,
    ``hackrf_transfer`` holds the radio, so a probe that needs the device sees a
    busy radio and concludes the tool is broken -- and the UI then announced the
    slow hop-by-hop engine forever, on a machine where the fast one worked.
    Whether a radio is *free* is a different question, asked at a different
    moment, and belongs to the scan rather than to capability detection.

    The exit code is the discriminator, not the output: this tool prints its
    usage banner on *any* failure, including a radio it cannot open, and the two
    streams interleave in an order that varies between runs.
    """
    try:
        r = subprocess.run(
            [str(tool), "-h"],
            capture_output=True, text=True, timeout=20,
            cwd=str(tool.parent),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if r.returncode != 0:
        return False
    # A distinctive phrase from this tool's own help, so that some other binary
    # answering with a zero exit is not mistaken for a working sweep tool.
    text = (r.stdout or "") + (r.stderr or "")
    return "bin_width" in text


@lru_cache(maxsize=1)
def available() -> bool:
    """True when the fast sweep path can be used on this machine."""
    tool = runnable_tool()
    return bool(tool) and _probe(tool)  # type: ignore[arg-type]


def unavailable_reason() -> str:
    """Why not, in words a person can act on. Empty when there is nothing wrong.

    The empty-when-fine case is not decoration. The first version had no such
    branch and fell through to "did not start on this machine", so asking the
    question when the answer was yes produced a confident falsehood -- which is
    worse than no answer, because a caller cannot tell it from a real fault.
    Callers should test the string, not just print it.
    """
    if available():
        return ""
    if find_sweep_tool() is None:
        return (
            "hackrf_sweep.exe not found. It ships in the Mayhem firmware's "
            "utils folder, next to hackrf_transfer.exe."
        )
    if find_fftw() is None:
        return (
            "hackrf_sweep.exe needs libfftw3f-3.dll, which the Mayhem bundle "
            "omits. Drop FFTW's single-precision DLL (fftw3f.dll) beside "
            "hackrf_sweep.exe renamed to libfftw3f-3.dll. The app is using the "
            "slower hop-by-hop scan meanwhile."
        )
    return "hackrf_sweep.exe did not start on this machine."


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass
class SweepBin:
    """One FFT magnitude, at a frequency, in dB."""

    freq_hz: int
    db: float


@dataclass
class SweepTrace:
    """A parsed sweep: magnitudes in frequency order, plus what was asked for."""

    bins: list[SweepBin] = field(default_factory=list)
    #: Every reading kept per bin, before combining. Kept because the choice of
    #: statistic is a real fork in the road and its consequences should be
    #: inspectable rather than quietly baked in.
    raw: dict[int, list[float]] = field(default_factory=dict)
    #: "max" or "mean". See :func:`_combine`.
    stat: str = "max"
    requested_lo_hz: int = 0
    requested_hi_hz: int = 0
    lo_hz: int = 0
    hi_hz: int = 0
    #: Outer edges of what was actually measured. Distinct from ``lo_hz`` and
    #: ``hi_hz``, which are the *centres* of the first and last bins and so sit
    #: half a bin inside the real coverage. Anything deciding what the trace
    #: covers wants these: a span reported 0.5 MHz short at each end is a span
    #: with 1 MHz of the band unmeasured, and a band edge is exactly where a
    #: channel at the end of a band plan lives.
    coverage_lo_hz: int = 0
    coverage_hi_hz: int = 0
    elapsed_s: float = 0.0
    sweeps: int = 1
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.bins) and not self.error

    @property
    def actual_lo_hz(self) -> int:
        return self.bins[0].freq_hz if self.bins else 0

    @property
    def actual_hi_hz(self) -> int:
        return self.bins[-1].freq_hz if self.bins else 0

    @property
    def bin_width_hz(self) -> int:
        """Measured, not requested. The tool derives the width it delivers."""
        if len(self.bins) < 2:
            return BIN_HZ
        span = self.bins[-1].freq_hz - self.bins[0].freq_hz
        return max(1, int(round(span / (len(self.bins) - 1))))

    def floor_db(self) -> float:
        """Median magnitude across the span: the noise floor for this pass."""
        if not self.bins:
            return -120.0
        vals = sorted(b.db for b in self.bins)
        n = len(vals)
        return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])

    def excess_db(self, floor: float | None = None) -> list[SweepBin]:
        """Every bin re-referenced to the floor, loudest first."""
        f = self.floor_db() if floor is None else floor
        out = [SweepBin(b.freq_hz, b.db - f) for b in self.bins]
        out.sort(key=lambda b: -b.db)
        return out

    def describe(self) -> str:
        span = (f"{bands.format_mhz(self.coverage_lo_hz)}.."
                f"{bands.format_mhz(self.coverage_hi_hz)}")
        if (self.coverage_lo_hz, self.coverage_hi_hz) != (
                self.requested_lo_hz, self.requested_hi_hz):
            span += (f" (asked {bands.format_mhz(self.requested_lo_hz)}.."
                     f"{bands.format_mhz(self.requested_hi_hz)})")
        n = len(self.bins)
        return (f"swept {span} in {self.elapsed_s:.2f} s, {n} bins of "
                f"{self.bin_width_hz/1e6:.2f} MHz, floor {self.floor_db():.1f} dB")


def _combine(values: list[float], stat: str) -> float:
    """Reduce repeated readings of one frequency to a single number.

    The mean is the wrong statistic here, and measurably so. Sweeps are short
    enough that a periodic transmitter is only sometimes present in one, and
    averaging a burst that appeared in a tenth of the passes with nine quiet
    passes divides its amplitude by ten. Measured here: three consecutive
    single-pass scans of 2.4 GHz gave spreads of 13, 42 and 15 dB, one of them
    catching a real WiFi signal at +42 dB over floor and the other two missing
    it entirely. A mean over those would have reported a fifth of what is
    actually there.

    The question being asked is "is something transmitting here", not "how much
    on average". The loudest pass answers that, and it answers it the same way
    whether the transmitter runs continuously, as FPV video does, or beacons,
    as WiFi does.

    The cost is a known bias: the maximum of N noise samples sits above their
    mean even with no signal present, by about 4.34*log10(N) dB for power
    noise. :func:`max_bias_db` prices that in so the gate can be set above it
    rather than by guesswork.
    """
    if not values:
        return -120.0
    if stat == "mean":
        return sum(values) / len(values)
    return max(values)


def max_bias_db(n: int) -> float:
    """How far the maximum of N noise samples sits above their mean.

    FFT bin power is roughly exponentially distributed, and for that the
    expected maximum of N samples is about ln(N) in natural log above the mean,
    which is 4.34*log10(N) in dB. Ten passes bias the reading by about 4.3 dB,
    a hundred by about 8.7. The gate has to clear this or repeated sweeping
    would manufacture findings out of nothing.
    """
    if n <= 1:
        return 0.0
    return 4.34 * math.log10(n)


def _parse(path: Path, lo: int, hi: int, sweeps: int, elapsed: float,
           stat: str = "max") -> SweepTrace:
    """Turn the tool's CSV into a frequency-sorted trace.

    The rows arrive in transfer order, not frequency order -- the first two rows
    of a real run were 5650 and 5660 MHz -- so sorting is load-bearing rather
    than tidy. Widths come from ``hz_low``/``hz_high`` divided by the number of
    magnitudes on the row, because the requested bin width is not the width
    delivered: asking for 500 kHz bins produced 0.379 MHz bins.
    """
    acc: dict[int, list[float]] = {}
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        return SweepTrace(requested_lo_hz=lo, requested_hi_hz=hi,
                          elapsed_s=elapsed, sweeps=sweeps, error=str(exc))

    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 7:
            continue
        try:
            r_lo = float(parts[2])
            r_hi = float(parts[3])
            vals = [float(v) for v in parts[6:] if v.strip()]
        except ValueError:
            continue          # the header, or a truncated final line
        if not vals:
            continue
        step = (r_hi - r_lo) / len(vals)
        for i, v in enumerate(vals):
            f = int(round(r_lo + step * (i + 0.5)))
            acc.setdefault(f, []).append(v)

    bins = [SweepBin(f, _combine(v, stat)) for f, v in sorted(acc.items())]
    # Coverage edges, from the outermost bins' own half-width. Deriving them
    # from the first and last *centres* would report a span one bin short at
    # each end, which is where a channel at the end of a band plan sits.
    if len(bins) >= 2:
        w = (bins[-1].freq_hz - bins[0].freq_hz) / (len(bins) - 1)
    else:
        w = float(BIN_HZ)
    cov_lo = int(bins[0].freq_hz - w / 2) if bins else 0
    cov_hi = int(bins[-1].freq_hz + w / 2) if bins else 0
    return SweepTrace(
        bins=bins,
        raw=dict(acc),
        stat=stat,
        requested_lo_hz=lo,
        requested_hi_hz=hi,
        lo_hz=bins[0].freq_hz if bins else 0,
        hi_hz=bins[-1].freq_hz if bins else 0,
        coverage_lo_hz=cov_lo,
        coverage_hi_hz=cov_hi,
        elapsed_s=elapsed,
        sweeps=sweeps,
    )


# ---------------------------------------------------------------------------
# Running a sweep
# ---------------------------------------------------------------------------


def round_to_grid(lo_hz: int, hi_hz: int) -> tuple[int, int]:
    """Widen a span to the tool's 20 MHz grid, so coverage is predictable.

    Asking for 5800-5810 gets 5800-5820 whatever we would prefer, so the
    request is made explicit and the real coverage is reported separately.
    """
    lo = (int(lo_hz) // TOOL_GRID_HZ) * TOOL_GRID_HZ
    hi = -(-int(hi_hz) // TOOL_GRID_HZ) * TOOL_GRID_HZ     # ceiling
    if hi <= lo:
        hi = lo + TOOL_GRID_HZ
    return lo, hi


def run_sweep(
    lo_hz: int,
    hi_hz: int,
    sweeps: int = 1,
    lna_db: int = 32,
    vga_db: int = 16,
    amp: bool = True,
    bin_hz: int = BIN_HZ,
    stat: str = "max",
    cancel: threading.Event | None = None,
    timeout: float = 6.0,
) -> SweepTrace:
    """One sweep of the span, combining ``sweeps`` passes.

    ``sweeps`` is 1 by default and that is a deliberate choice rather than the
    lazy one. FPV video transmits continuously, so one pass sees it as well as
    ten would, and the single pass costs 0.13 s against 0.53 s for ten. More
    passes only help against a transmitter that is *intermittent* -- a beaconing
    WiFi access point, say -- and :func:`_combine` explains why that argues for
    the loudest pass rather than the average.
    """
    tool = runnable_tool()
    if tool is None:
        return SweepTrace(requested_lo_hz=lo_hz, requested_hi_hz=hi_hz,
                          error=unavailable_reason())
    lo, hi = round_to_grid(lo_hz, hi_hz)
    with tempfile.TemporaryDirectory(prefix="sweep_") as td:
        out = Path(td) / "trace.csv"
        cmd = [
            str(tool),
            "-f", f"{lo // 1_000_000}:{hi // 1_000_000}",
            "-w", str(max(MIN_BIN_HZ, int(bin_hz))),
            "-N", str(max(1, int(sweeps))),
            "-l", str(int(lna_db)),
            "-g", str(int(vga_db)),
            "-a", "1" if amp else "0",
            "-r", str(out),
        ]
        if sweeps <= 1:
            cmd.append("-1")
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                cwd=str(tool.parent),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            return SweepTrace(requested_lo_hz=lo_hz, requested_hi_hz=hi_hz,
                              error=f"could not start the sweep tool: {exc}")
        while proc.poll() is None:
            if cancel is not None and cancel.is_set():
                proc.kill()
                proc.wait(timeout=10)
                return SweepTrace(requested_lo_hz=lo_hz, requested_hi_hz=hi_hz,
                                  elapsed_s=time.monotonic() - t0,
                                  sweeps=sweeps, error="cancelled")
            # A sweep is 0.15 s. Anything still running at the timeout is not
            # slow, it is stuck -- the observed cause is another process still
            # holding the radio open. Left without this, a scan that would
            # otherwise be instant sat there for minutes, which in a GUI is a
            # window that looks frozen with no way out but killing the app.
            if time.monotonic() - t0 > timeout:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                return SweepTrace(
                    requested_lo_hz=lo_hz, requested_hi_hz=hi_hz,
                    elapsed_s=time.monotonic() - t0, sweeps=sweeps,
                    error=f"the sweep tool made no progress in {timeout:.0f}s "
                          f"(something else is holding the radio)")
            time.sleep(0.005)
        elapsed = time.monotonic() - t0
        if proc.returncode != 0:
            return SweepTrace(requested_lo_hz=lo_hz, requested_hi_hz=hi_hz,
                              elapsed_s=elapsed, sweeps=sweeps,
                              error=f"the sweep tool exited {proc.returncode}")
        # Parsed in here, not after the block: leaving the `with` deletes the
        # temporary directory, and the file this needs to read has gone with it.
        trace = _parse(out, lo, hi, sweeps, elapsed, stat)
        if trace.ok or trace.error:
            return trace
        # Exit code 0 with nothing to show for it. This is the failure mode that
        # must never pass silently: an empty trace reports as "nothing above the
        # noise floor", which reads as *the band is quiet* when the truth is
        # that the measurement never happened. A scan's whole value is that its
        # silence means something, so silence bought this way is a lie. Observed
        # once already, 38 s long, on a radio that had just been handed back.
        trace.error = (
            f"the sweep tool finished but returned no data for "
            f"{lo // 1_000_000}-{hi // 1_000_000} MHz after {elapsed:.1f}s"
        )
        return trace


# ---------------------------------------------------------------------------
# Turning a trace into the app's normal scan result
# ---------------------------------------------------------------------------

#: Minimum excess over the floor before a wideband region is called a
#: transmitter, in dB. A floor on the adaptive gate, not the whole rule.
#:
#: Set from the two numbers that bound it, not tuned until a test passed. The
#: widest excursion a genuinely empty band has produced on this machine is
#: +8.8 dB, measured on a busy USB bus where even the hop walk slowed from 48 ms
#: to 2000 ms per hop; on a healthy bus it is +2 to +4 dB. A real transmitter
#: measures +29 dB in the same 15 MHz window, and +40 dB when a burst lands in
#: the pass being read. So the gate has to clear 8.8 with room to spare and stay
#: well under 29, and 14 sits in the middle of that gap with roughly equal
#: margin either side. Anything below about 11 reports phantom transmitters on a
#: degraded bus -- which is worse than reporting nothing, because a pilot cannot
#: tell a phantom from a real one.
GATE_DB = 14.0

#: Sigmas above the measured scatter that also count as a finding. Three is
#: chosen against the numbers this machine produced rather than picked: a
#: single-pass 1 MHz bin scatters about 4.5 dB, the 15 MHz wideband window
#: brings that down to roughly 1.5-2 dB, and three of those sits near 6 dB --
#: comfortably under the +39 to +42 dB a real WiFi signal showed, and above the
#: +13 dB a single noise bin reached before the window suppressed it.
GATE_SIGMA = 3.0

#: Width of the window used to look for a transmitter, in Hz.
#:
#: A single 1 MHz bin is the wrong unit of measurement here. Adjacent bins come
#: from the same FFT and are strongly correlated, so a lone bin that happens to
#: land high looks as significant as its distance above the floor suggests,
#: and thresholding raw bins duly reports noise excursions as transmitters. On a
#: quiet 5.8 GHz band, single-bin hits at +9 to +11 dB came up while the only
#: thing on air was nothing at all.
#:
#: What is actually being looked for is an analogue FPV transmission, which is
#: around 25 MHz wide. Measuring power in a window at least that wide rejects
#: one-bin spikes on physical grounds rather than on a tuned threshold, and it
#: responds properly to a real signal whose skirts are weaker than its centre.
#: It matches :data:`fpv_rf.scan.VTX_WIDTH_HZ` deliberately.
WIDE_HZ = 15_000_000

#: Below this width a finding is reported as a narrowband carrier rather than a
#: video channel, because that is what it is. Nothing in 5.8 GHz that a pilot
#: wants to look at is 3 MHz wide.
NARROW_HZ = 6_000_000


def robust_sigma(db: list[float]) -> float:
    """Noise scatter, from the median absolute deviation.

    A plain standard deviation is the wrong tool on a spectrum: the few bins
    that hold a signal dominate the sum of squares and inflate the very number
    meant to describe the noise. The median absolute deviation does not, and the
    1.4826 factor makes it agree with a standard deviation for clean Gaussian
    noise.
    """
    if len(db) < 3:
        return 1.0
    med = statistics.median(db)
    mad = statistics.median([abs(v - med) for v in db])
    return max(0.25, 1.4826 * mad)


def robust_gate(sigma: float, n_sweeps: int = 1) -> float:
    """The level a bin must clear, in dB *above the floor*.

    Three terms, each earning its place:

    * ``GATE_DB`` as a floor, so a pathologically still band cannot produce a
      gate small enough to trigger on rounding.
    * ``GATE_SIGMA`` times the measured scatter, so the gate adapts to the band
      instead of assuming every band is as noisy as the last.
    * :func:`max_bias_db`, because the loudest-of-N statistic runs high even on
      pure noise. Without this term, sweeping ten times would raise every bin
      about 4.3 dB and manufacture findings out of an empty band -- which is
      precisely how a scan that is supposed to prove a band is quiet would end
      up reporting transmitters in it.

    Deliberately an *excess*, not an absolute level. Taking the floor as an
    argument made it look like one, and the first version subtracted the floor
    again when reporting the result -- printing a gate of 77 dB on a band whose
    floor sits near -66 dB and whose real gate is 12.
    """
    return max(GATE_DB, GATE_SIGMA * sigma + max_bias_db(n_sweeps))


def wideband_db(bins: list[SweepBin], floor: float) -> list[SweepBin]:
    """Power in a :data:`WIDE_HZ` window centred on each bin, over the floor.

    Returned in frequency order, with the same frequencies as the input. A
    window at the edge of the span is truncated rather than wrapped, so the
    first and last few bins are measured over less width than the rest and will
    read low; that is the safe direction, since it under-reports rather than
    invents.
    """
    if not bins:
        return []
    n = len(bins)
    width = (bins[-1].freq_hz - bins[0].freq_hz) / max(1, n - 1) or BIN_HZ
    half = max(1, int(WIDE_HZ / 2 / width))
    out: list[SweepBin] = []
    for i, b in enumerate(bins):
        lo_i, hi_i = max(0, i - half), min(n, i + half + 1)
        window = bins[lo_i:hi_i]
        mean = sum(w.db for w in window) / len(window)
        out.append(SweepBin(b.freq_hz, mean - floor))
    return out


def _reading(excess_db: float, wide: bool) -> dsp.ChannelReading:
    """A reading for something the sweep found, without claiming what it is.

    The sweep measures magnitude, not video, so this is honestly
    ``SignalKind.CARRIER`` in both cases: energy is present and its strength is
    known, but whether it is analogue video is not, and saying so would be
    inventing a measurement. The width is what separates the two cases, and it
    is carried in ``peak_to_median_db``'s place only as far as the app already
    uses that field -- for ranking. The decoder settles what it is, by
    re-tuning and reporting real sync.
    """
    return dsp.ChannelReading(
        kind=dsp.SignalKind.CARRIER,
        snr_db=excess_db,
        peak_to_median_db=excess_db,
        noise_db=0.0,
        sync_pulses=0,
        sync_density=0.0,
        sync_quality=0.0,
        line_rate_hz=0.0,
        spec=None,
        peak_offset_hz=0.0,
    )


def result_from_trace(trace: SweepTrace) -> scan.ScanResult:
    """Group the loudest wideband regions into candidates.

    Three steps, in this order, and the order matters:

    1. Find the noise floor as the median, and its scatter as a MAD.
    2. Measure power in a transmitter-width window, so the unit of measurement
       matches the thing being looked for.
    3. Group the bins that clear the gate into contiguous runs, and take each
       run as one transmitter rather than as a dozen separate findings.
    """
    res = scan.ScanResult(
        lo_hz=trace.coverage_lo_hz or trace.requested_lo_hz,
        hi_hz=trace.coverage_hi_hz or trace.requested_hi_hz,
        elapsed_s=trace.elapsed_s,
        hops=len(trace.bins),
        # Every bin of a successful trace was actually read, so the coverage a
        # result reports has to agree with the bins it was built from -- a result
        # claiming zero measured units would be read as no search having run.
        measured=len(trace.bins) if trace.ok else 0,
        noise_floor_db=trace.floor_db(),
        error=trace.error,
    )
    if not trace.ok:
        return res

    floor = trace.floor_db()
    # The scatter that matters is the wideband series', not the raw bins': the
    # gate is applied after the window, so sizing it from the narrower
    # measurement would make it far too strict.
    wide = wideband_db(trace.bins, floor)
    sigma = robust_sigma([b.db for b in wide])
    gate = robust_gate(sigma, trace.sweeps)
    width = ((trace.bins[-1].freq_hz - trace.bins[0].freq_hz)
             / max(1, len(trace.bins) - 1)) or BIN_HZ

    over = [b for b in wide if b.db >= gate]
    if not over:
        res.gate_peak_db = gate
        return res

    max_gap = max(2 * int(width), WIDE_HZ)

    runs: list[list[SweepBin]] = []
    for b in over:
        if runs and b.freq_hz - runs[-1][-1].freq_hz <= max_gap:
            runs[-1].append(b)
        else:
            runs.append([b])

    for run in runs:
        peak = max(run, key=lambda b: b.db)
        centre = (run[0].freq_hz + run[-1].freq_hz) // 2
        # The emitted width is the run plus one window's worth of skirt, since
        # a signal narrower than the window measures as wider than it is. The
        # uncertainty is therefore a lower bound, which is the honest direction
        # for a label that claims to be within a tolerance.
        span_hz = run[-1].freq_hz - run[0].freq_hz + int(width)
        raw = [b for b in trace.bins
               if run[0].freq_hz - width <= b.freq_hz <= run[-1].freq_hz + width]
        excess = max((b.db for b in raw), default=peak.db) - floor
        hit = scan.CoarseHit(
            hop_hz=int(width),
            bins_above=len(run),
            peak_db=excess,
            peak_offset_hz=peak.freq_hz - centre,
            floor_db=floor,
        )
        # The same rule the hop scan uses, and for the same reason: a run of
        # bins measures a transmitter to within its own width, so a chart
        # channel further away than half that width is not one this measurement
        # found. Before, off_chart was ``match is None``, which the nearest-match
        # call never returns for an in-band frequency -- so a fast sweep labelled
        # everything on-chart while the same finding from a hop scan was
        # reported off-chart, and the two labels were both on screen at once.
        match, off_chart = scan.match_within(centre, int(span_hz))
        cand = scan.Candidate(
            frequency_hz=centre,
            reading=_reading(excess, span_hz >= NARROW_HZ),
            match=match,
            uncertainty_hz=int(span_hz),
            coarse=hit,
            off_chart=off_chart,
            span_hz=span_hz,
            merged=len(run),
        )
        res.candidates.append(cand)
        res.hits.append(hit)
    res.candidates.sort(key=lambda c: -c.strength)
    res.gate_peak_db = gate
    return res
