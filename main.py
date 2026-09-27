"""FPV analogue video: band search, channel select, and RF alerting.

Run ``python main.py --help`` for the options. The defaults start on the sim
source so the app comes up and does something useful on a machine with no
HackRF and nothing transmitting, which is the normal state of a bench.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpv_rf import bands, sdr  # noqa: E402

LOG_NAME = "fpv-rf.log"


class _Tee(io.TextIOBase):
    """Write to a stream *and* to a log file, tolerating either being absent.

    A windowed build has no console at all: ``sys.stdout`` is ``None`` and every
    ``print`` in the program goes nowhere. That is exactly when diagnostics
    matter most, because a frozen build that misbehaves gives you nothing to
    look at. So stdout and stderr are replaced with these, and everything the
    program says lands in a log file beside the executable whether or not anyone
    is watching a terminal.
    """

    def __init__(self, original, path: Path, mirror_to_console: bool) -> None:
        self._original = original
        self._path = path
        self._mirror = mirror_to_console and original is not None
        try:
            # Held open rather than reopened per write: the program prints on
            # transitions, not in a loop, but a tee that opens a file for every
            # fragment of a traceback is a bad habit to leave in a program whose
            # failure path is the thing you will be reading.
            self._fh = path.open("a", encoding="utf-8", errors="replace")
        except Exception:
            self._fh = None

    def write(self, s: str) -> int:
        if self._mirror and self._original is not None:
            try:
                self._original.write(s)
            except Exception:
                pass
        if self._fh is not None:
            try:
                self._fh.write(s)
                self._fh.flush()
            except Exception:
                # A log that cannot be written must never be the reason the app
                # fails to carry on.
                self._fh = None
        return len(s)

    def flush(self) -> None:
        if self._mirror and self._original is not None:
            try:
                self._original.flush()
            except Exception:
                pass
        if self._fh is not None:
            try:
                self._fh.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return bool(self._mirror and getattr(self._original, "isatty", False)())


def install_logging() -> Path:
    """Send stdout and stderr to a log file. Returns the path written to."""
    from fpv_rf.sdr import app_dir

    path = app_dir() / LOG_NAME
    try:
        # A new session should not be buried under the last one's output.
        if path.exists():
            path.unlink()
    except Exception:
        pass
    frozen = bool(getattr(sys, "frozen", False))
    sys.stdout = _Tee(sys.stdout, path, not frozen)
    sys.stderr = _Tee(sys.stderr, path, not frozen)
    return path


def _alert_box(title: str, body: str) -> None:
    """Show a message box when there is no console to print to.

    ``--check`` on a windowed build would otherwise report its result into the
    void and exit, which is indistinguishable from crashing.

    ``FPV_RF_NO_DIALOG=1`` suppresses this. A modal box blocks until someone
    clicks it, which is right for a person and fatal for the build script that
    needs to verify the packaged app unattended.
    """
    if os.environ.get("FPV_RF_NO_DIALOG"):
        return
    if sys.stdout is not None and getattr(sys.stdout, "isatty", lambda: False)():
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, body, title, 0x40)
    except Exception:
        pass


def build_parser(default_mode: str = "sim") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fpv",
        description=(
            "Decode analogue FPV video from HackRF IQ, search the 5.8 GHz band "
            "for energy above the noise floor, and alert with band + channel + "
            "frequency."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("source")
    g.add_argument(
        "--mode", choices=("hackrf", "file", "sim"), default=default_mode,
        help="hackrf needs a radio; file replays a capture; sim is synthetic",
    )
    g.add_argument(
        "--file", default=None, metavar="PATH",
        help="capture to replay, for --mode file (raw interleaved uint8 IQ)",
    )
    g.add_argument(
        "--capture-frequency", type=float, default=0.0, metavar="MHZ",
        help=(
            "the frequency a --mode file capture was recorded at, in MHz. "
            "Required for file mode to name a band: a recording cannot be swept, "
            "so without this the app has the samples but not the channel."
        ),
    )
    g.add_argument("--fast-file", action="store_true",
                   help="replay a file as fast as it decodes, not in real time")

    g = p.add_argument_group("tuning")
    g.add_argument("--frequency", type=float, default=5802.0, metavar="MHZ",
                   help="initial tune, MHz")
    g.add_argument("--lna", type=int, default=32, help="LNA gain, dB (0-62)")
    g.add_argument("--vga", type=int, default=16, help="VGA gain, dB (0-62)")
    g.add_argument("--hackrf-transfer", default=None, metavar="EXE",
                   help="path to hackrf_transfer.exe; found automatically if omitted")
    g.add_argument(
        "--region", choices=[r.value for r in bands.REGIONS], default="us",
        help="region, for the band chart and the default search span",
    )

    g = p.add_argument_group("picture")
    g.add_argument("--width", type=int, default=160,
                   help="raster width in samples per line; halve it to decode "
                        "faster at the cost of horizontal detail")

    g = p.add_argument_group("search and alerts")
    g.add_argument("--no-autoscan", action="store_true",
                   help="do not start a band search on launch")
    g.add_argument("--no-beep", action="store_true", help="no audible alert")
    g.add_argument("--alert-cooldown", type=float, default=15.0, metavar="S",
                   help="minimum gap between repeat alerts for one finding")
    g.add_argument("--alert-forget", type=float, default=30.0, metavar="S",
                   help="how long a finding may be missing before it is lost")

    g = p.add_argument_group("diagnostics")
    g.add_argument("--check", action="store_true",
                   help="verify the source and exit, without starting the GUI")
    g.add_argument(
        "--selftest", action="store_true",
        help="build the window offscreen, check it end to end, and exit. This "
             "is how you tell a broken packaged build apart from a fault in "
             "the app itself")
    return p


def main(argv: list[str] | None = None) -> int:
    log_path = install_logging()
    # A frozen build is the one people double-click, and what they double-click
    # it for is the radio plugged into it. Defaulting that to the simulator --
    # which is the right default for the source tree, where the app has to do
    # something on a machine with no hardware -- would open a convincing fake
    # picture every time. --mode sim still selects it in the packaged app.
    default_mode = "hackrf" if getattr(sys, "frozen", False) else "sim"
    args = build_parser(default_mode=default_mode).parse_args(argv)
    args.region = bands.Region(args.region)
    args.frequency_hz = int(round(args.frequency * 1e6))

    if args.mode == "file" and not args.file:
        print("--mode file needs --file", file=sys.stderr)
        _alert_box("FPV RF", "--mode file needs --file")
        return 2
    if args.mode == "file" and not args.capture_frequency:
        print(
            "--mode file needs --capture-frequency, so the app can name the band "
            "the recording was made on",
            file=sys.stderr,
        )
        _alert_box(
            "FPV RF",
            "--mode file needs --capture-frequency, so the app can name the "
            "band the recording was made on.",
        )
        return 2

    source = sdr.make_source(
        args.mode,
        args.frequency_hz,
        lna_db=args.lna,
        vga_db=args.vga,
        file_path=args.file,
        executable=args.hackrf_transfer,
    )
    if args.mode == "file":
        source.frequency_hz = int(round(args.capture_frequency * 1e6))
        source.realtime = not args.fast_file

    lo, hi = bands.default_scan_span(args.region)
    print(f"log: {log_path}")
    print(f"source: {source.describe()}")
    print(f"retunable: {source.retunable}   "
          f"search span: {bands.format_mhz(lo)} - {bands.format_mhz(hi)}")
    if args.mode == "hackrf" and not source.available():
        print("hackrf_transfer.exe was not found", file=sys.stderr)
        _alert_box(
            "FPV RF - radio helper missing",
            "hackrf_transfer.exe was not found.\n\n"
            "Put it in the same folder as this program, or set the "
            "HACKRF_TRANSFER environment variable to its full path.\n\n"
            f"Details are in {log_path}",
        )
        return 3

    if args.check:
        return _check(source)

    if args.selftest:
        return _selftest()

    from fpv_rf import ui

    print("starting the window")
    try:
        return ui.run(source, args)
    except Exception:
        import traceback

        traceback.print_exc()
        _alert_box(
            "FPV RF - failed to start",
            f"The window could not be opened.\n\nSee {log_path} for the "
            "full error.",
        )
        raise


def _check(source: sdr.IQSource) -> int:
    """Start the source, prove it produces usable IQ, and report."""
    import io
    import time

    from fpv_rf import dsp

    # Collect the report as text as well as printing it, so a windowed build can
    # show it in a message box instead of a console nobody is looking at.
    report = io.StringIO()

    def say(*parts: str) -> None:
        line = " ".join(parts)
        print(line)
        report.write(line + "\n")

    source.start()
    time.sleep(0.5)
    try:
        st = source.stats
        iq = source.take_iq(1_000_000)
        if iq.size < 100_000:
            say(f"FAIL: source produced only {iq.size} samples")
            say(f"last error: {st.last_error or '(none)'}")
            _alert_box("FPV RF - check failed", report.getvalue().strip())
            return 1
        rate = st.bytes_total / 2 / max(1e-6, time.monotonic() - st.started_at)
        say(f"ok: {iq.size} samples, {rate/1e6:.2f} MS/s, "
            f"mean |IQ| {abs(iq).mean():.3f}")
        dropped = source.ring.dropped_bytes // 2
        if dropped:
            # Expected in this mode and not a warning: the check pulls a whole
            # 1 Msample snapshot at once, and the ring is 2 Msamples and still
            # filling, so the writer is guaranteed to overrun. The streaming
            # decoder drains continuously and does not do this.
            say(f"({dropped/1e6:.2f} Msamples overwritten while this check "
                f"read its snapshot -- the streaming decoder does not)")
        reading = dsp.assess_channel(iq, source.sample_rate)
        verdict = reading.describe()
        say(f"at {bands.format_mhz(source.frequency_hz)}: {verdict}")
        if st.last_error:
            say(f"radio helper said: {st.last_error}")
        _alert_box(f"FPV RF - check: {bands.format_mhz(source.frequency_hz)}",
                   report.getvalue().strip())
        return 0
    finally:
        source.stop()


def _selftest() -> int:
    """Build the real window offscreen and prove it works, then exit.

    A frozen build can fail in ways running from source cannot -- a missing Qt
    plugin, a data file left behind by the packager -- and the symptom is a
    window that never appears. This is how you tell that apart from a fault in
    the app itself without attaching a debugger to an .exe.
    """
    import os
    import runpy
    import tempfile
    from pathlib import Path

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    tools = Path(__file__).resolve().parent / "tools" / "check_ui.py"
    if not tools.exists():
        print(f"selftest unavailable: {tools} is not present")
        return 1
    print(f"running the offscreen window check from {tools}")
    # The check writes its screenshots relative to the working directory, so run
    # it somewhere disposable. Otherwise a self test on a packaged app litters
    # the application folder with a _validate_out directory every time it runs.
    prev = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="fpv-selftest-") as tmp:
        try:
            os.chdir(tmp)
            runpy.run_path(str(tools), run_name="__main__")
        except SystemExit as exc:
            code = int(exc.code or 0)
        except Exception:
            import traceback

            traceback.print_exc()
            code = 1
        finally:
            os.chdir(prev)
    print(f"selftest exit={code}")
    _alert_box("FPV RF - self test",
               "PASS - the window builds, decodes, scans and alerts."
               if code == 0 else f"FAIL (exit {code}). See the log.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
