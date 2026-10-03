"""Build a double-clickable Windows app.

    python build_exe.py

Produces ``dist/FPV-RF/FPV-RF.exe`` -- a folder, not a single file. That is a
deliberate choice and worth explaining, because "why isn't it one .exe?" is the
obvious question.

A one-file build unpacks its whole payload to a temp directory on *every* launch
and deletes it afterwards. With Qt and SciPy that is several hundred megabytes
copied to disk each time, which turns a two-second start into a thirty-second
one. For a program whose job is decoding live video, paying that before the
window appears is the wrong trade. The folder form starts in about a second and
is a normal, copyable, deletable application directory.

Run the resulting app with:

    FPV-RF.cmd                                   the launcher in this folder
    "dist\\FPV-RF\\FPV-RF.exe"                    the radio, tuned to 5802 MHz
    "dist\\FPV-RF\\FPV-RF.exe" --mode sim         a synthetic drone, no hardware
    "dist\\FPV-RF\\FPV-RF.exe" --check            verify the radio and exit
    "dist\\FPV-RF\\FPV-RF.exe" --selftest         check the packaged build

``FPV-RF.cmd`` answers the same "i dont see the exe in the project folder"
without moving anything: the exe cannot leave its folder -- see above for why
it needs _internal beside it -- so a launcher in the root starts it there and
forwards any arguments. This script writes that file, and rewrites it on every
build so it cannot drift out of step with the paths below.

The verification step matters more here than for a source-tree run: a packaged
build can fail for reasons that have nothing to do with the program (a Qt
platform plugin the packager left out, a data file that did not get copied), and
the symptom is a window that never appears. ``--selftest`` builds the real
window offscreen inside the frozen build, so "the exe is broken" and "the app is
broken" become distinguishable without a debugger.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#: No spaces, and the name says what it is. The first build went to
#: ``dist/FPV RF/FPV RF.exe`` and the response was "i dont see the exe in the
#: project folder" -- which is fair. A space in the folder name means the path
#: has to be quoted everywhere, Explorer hides it behind one more level of
#: nesting, and it is awkward to type into a Run box or a shell. The build
#: output should be the one thing in this project that is impossible to miss.
APP_NAME = "FPV-RF"

DIST = ROOT / "dist"
BUILD = ROOT / "build"

#: The one thing in the project root a person double-clicks. The exe itself
#: cannot move here: it is an onedir build, so without _internal and the
#: hackrf helpers beside it it does not start, and the one-file alternative is
#: rejected above for the thirty seconds it costs to launch. A two-line launcher
#: in the root gets the app to the same click, without copying 212 MB.
ROOT_LAUNCHER = ROOT / f"{APP_NAME}.cmd"

#: The radio helper has to travel with the app. It is found relative to the exe
#: at runtime, ahead of PATH, so the copy shipped here is the one that runs.
#:
#: Discovered through sdr rather than a path written down here, so there is only
#: one place in the project that knows where hackrf_transfer lives. Building
#: from a checkout with the helper already beside ``main.py`` wins, which is the
#: usual case once someone has put it there deliberately.
sys.path.insert(0, str(ROOT))
from fpv_rf.sdr import find_hackrf_transfer  # noqa: E402
from fpv_rf.sweep import find_sweep_tool  # noqa: E402

_found_transfer = find_hackrf_transfer()
HACKRF_TRANSFER = Path(_found_transfer) if _found_transfer else None

#: The fast search engine's helper, for the same reason and found the same way.
_found_sweep = find_sweep_tool()
HACKRF_SWEEP = Path(_found_sweep) if _found_sweep else None

#: Qt ships far more than this program uses. Only Core, Gui and Widgets are
#: imported, so everything else is dead weight -- and the big offenders (QtWebEngine
#: alone is several hundred megabytes) dominate the folder size.
#:
#: Excluding aggressively is only safe if the result is verified, which is what
#: the --selftest run at the end of this script is for.
EXCLUDES = [
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DRender",
    "PySide6.QtBluetooth",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNfc",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtPositioning",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSerialBus",
    "PySide6.QtSerialPort",
    "PySide6.QtSpatialAudio",
    "PySide6.QtSql",
    "PySide6.QtStateMachine",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtTest",
    "PySide6.QtTextToSpeech",
    "PySide6.QtUiTools",
    "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebSockets",
    "PySide6.QtXml",
    "PIL",
    "matplotlib",
    "tkinter",
    "pytest",
    "IPython",
    "notebook",
]


def run(cmd: list[str], **kw) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def launcher_text() -> str:
    """The text of the root launcher.

    Derived from the constants above rather than written down here, so a
    renamed app or a moved dist directory cannot leave a launcher pointing at a
    path that no longer exists. CRLF, because cmd.exe is the thing reading it.
    """
    rel = f"{DIST.name}\\{APP_NAME}\\{APP_NAME}.exe"
    return f"""@echo off
REM FPV-RF launcher. Generated by "python build_exe.py --root-launcher" -- edit
REM that, not this file, or the next build overwrites it.
REM
REM The packaged app is a folder, not a single exe: FPV-RF.exe needs _internal
REM and the hackrf helpers beside it, so it cannot be moved up here on its own.
REM This starts it where it is, from whatever directory you run it from, and
REM passes on the arguments you typed.

setlocal
set "FPV_RF_EXE=%~dp0{rel}"

if not exist "%FPV_RF_EXE%" (
    echo FPV-RF has not been built yet: "%FPV_RF_EXE%" is not there.
    echo Build it first:  python build_exe.py
    exit /b 1
)

"%FPV_RF_EXE%" %*
exit /b %ERRORLEVEL%
"""


def write_root_launcher() -> Path:
    """Write (or refresh) ``FPV-RF.cmd``. Returns the path written."""
    ROOT_LAUNCHER.write_text(launcher_text(), encoding="utf-8", newline="\r\n")
    return ROOT_LAUNCHER


def main() -> int:
    # HACKRF_TRANSFER is None when the helper was not found anywhere, which is
    # a perfectly ordinary machine -- a fresh checkout, or a build box with no
    # radio firmware on it. The message below exists precisely to tell that
    # operator that the app will still build, so calling .exists() on it first
    # turned the promised warning into an AttributeError: the one case the
    # script was written to handle was the one it could not survive.
    if HACKRF_TRANSFER is None or not HACKRF_TRANSFER.exists():
        where = HACKRF_TRANSFER if HACKRF_TRANSFER is not None else "not found on this machine"
        print(f"warning: {where}\n"
              f"         The app will still build, but it will not find a radio "
              f"unless hackrf_transfer.exe\n"
              f"         is copied next to the exe afterwards, or "
              f"HACKRF_TRANSFER is set.",
              file=sys.stderr)

    for d in (DIST, BUILD):
        if d.exists():
            print(f"removing {d}")
            shutil.rmtree(d, ignore_errors=True)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--windowed",                     # no console window on double-click
        "--name", APP_NAME,
        "--onedir",
        "--distpath", str(DIST),
        "--workpath", str(BUILD),
        "--specpath", str(ROOT),
        # A visible icon would be nicer; there isn't one to ship, and shipping a
        # default PyInstaller icon is worse than shipping none.
        "--hidden-import", "winsound",
        "--collect-submodules", "fpv_rf",
        # check_ui.py runs the real window offscreen; --selftest needs it inside
        # the frozen build to tell a broken package from a broken app.
        "--add-data", f"{ROOT / 'tools' / 'check_ui.py'}{os.pathsep}tools",
    ]
    for mod in EXCLUDES:
        cmd += ["--exclude-module", mod]
    cmd.append(str(ROOT / "main.py"))

    t0 = time.monotonic()
    run(cmd, cwd=ROOT)
    print(f"\npackaged in {time.monotonic() - t0:.0f} s")

    out_dir = DIST / APP_NAME
    exe = out_dir / f"{APP_NAME}.exe"
    if not exe.exists():
        print(f"expected {exe} and it is not there", file=sys.stderr)
        return 1

    if HACKRF_TRANSFER is not None and HACKRF_TRANSFER.exists():
        dest = out_dir / HACKRF_TRANSFER.name
        shutil.copy2(HACKRF_TRANSFER, dest)
        print(f"copied the radio helper to {dest.name}")
    else:
        # Already warned about at the top. Repeating it here would be noise; the
        # point of the copy block is that a build without the helper is a build
        # that ships, not one that stops.
        print("no radio helper to copy -- the packaged app will need one "
              "beside the exe to talk to a HackRF")

    # The sweep tool travels too, or the packaged app is stuck on the hop walk
    # for everyone who has not already got a Mayhem utils folder on the desktop.
    # It is the whole 0.14 s versus 12.5 s difference, so shipping the app
    # without it would quietly ship the slow version.
    #
    # FFTW is deliberately not shipped: libfftw3f-3.dll is GPL, and this project
    # is MIT. sweep.find_fftw() looks for a suitable build on the user's machine
    # and stages a copy at runtime; when it finds none, the app says the fast
    # path is unavailable and uses the hop walk, which is a working app rather
    # than a broken one.
    if HACKRF_SWEEP is not None and HACKRF_SWEEP.exists():
        dest = out_dir / HACKRF_SWEEP.name
        shutil.copy2(HACKRF_SWEEP, dest)
        print(f"copied the sweep helper to {dest.name}")
    else:
        print("warning: hackrf_sweep.exe was not found, so the packaged app\n"
              "         will use the hop walk (12.5 s) instead of the sweep\n"
              "         (0.14 s). Copy it next to the exe to fix that.",
              file=sys.stderr)

    size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"\n{exe}")
    print(f"{out_dir}  ({size / 1e6:.0f} MB, {len(list(out_dir.rglob('*')))} files)")

    # The app stays where it is; this is what makes it findable from the root.
    print(f"wrote {write_root_launcher().name}")

    # The build is not finished until the thing it produced has been run.
    # FPV_RF_NO_DIALOG matters here: these are GUI-subsystem exes, so a result
    # report would come back as a modal message box and block until clicked.
    env = dict(os.environ, FPV_RF_NO_DIALOG="1", QT_QPA_PLATFORM="offscreen")

    # Order matters, and putting it back is expensive. The self test measures
    # sustained publish throughput, so it fails if something else is competing for
    # the CPU -- and opening a HackRF immediately before it is exactly that. An
    # earlier build ran the radio check first and the self test failed on a binary
    # that was otherwise fine. So the throughput measurement goes first, on a
    # quiet machine, and anything that touches the radio comes after it. The gate
    # itself is check_ui.MIN_FRAME_RATE (0.7 * PREVIEW_HZ, 21 fps against the 30 Hz
    # cadence), quoted from the cadence so the two cannot drift apart.

    log = out_dir / "fpv-rf.log"

    def read_log() -> str:
        return (log.read_text(encoding="utf-8", errors="replace")
                if log.exists() else "")

    print("\nverifying the packaged build: --selftest (throughput, run first)",
          flush=True)
    rc2 = subprocess.run([str(exe), "--selftest"], cwd=out_dir, env=env).returncode
    print(f"--selftest exit={rc2}")
    # Read it now. Every run of the app clears the log at startup, so the self
    # test's verdict is destroyed by the checks that follow -- and the self test
    # is the one whose output is worth reading.
    selftest_log = read_log()

    print("\nverifying the packaged build: --check (sim)", flush=True)
    rc = subprocess.run(
        [str(exe), "--check", "--mode", "sim"], cwd=out_dir, env=env
    ).returncode
    print(f"--check exit={rc}")

    print("\nverifying the packaged build: --check (hackrf)", flush=True)
    rc_radio = subprocess.run([str(exe), "--check"], cwd=out_dir, env=env).returncode
    print(f"--check (hackrf) exit={rc_radio}")

    radio_log = read_log()
    print("\n--- self test (the real window, offscreen, inside the .exe) ---")
    for ln in selftest_log.splitlines():
        if ln.startswith("RESULT") or "frames at" in ln or ln.strip().startswith("FAIL"):
            print(f"  {ln.strip()}")
    print("\n--- live radio check ---")
    for ln in radio_log.splitlines():
        if ln.startswith("ok:") or ln.startswith("FAIL") or ln.startswith("at "):
            print(f"  {ln.strip()}")

    if rc2:
        print(
            "\nThe self test failed. It runs the real window offscreen and checks\n"
            "decode throughput among other things, so a failure here is the app,\n"
            "not the packaging. The full report is in the log above.",
            file=sys.stderr,
        )
        return 1
    if rc:
        print("\nBUILD OK BUT VERIFICATION FAILED", file=sys.stderr)
        return 1
    if rc_radio:
        print(
            "\nWARNING: the packaged build is sound but the live radio check "
            "failed. Check the hardware, and see the log.",
            file=sys.stderr,
        )

    # The exe was built, verified, and launched twice at this point, so it is
    # certainly not the absence of a file that would be reported here. Still
    # check: antivirus software and disk-cleanup tools are fond of removing a
    # freshly written unsigned executable, and "the build succeeded but there is
    # nothing in dist" is a much more confusing report than an explicit one.
    if not exe.exists():
        print(f"\nTHE BUILD RAN BUT {exe} IS NOT ON DISK", file=sys.stderr)
        print("Something removed it -- most often antivirus. See the path above.",
              file=sys.stderr)
        return 1

    print()
    print("=" * 68)
    print(f"  THE APP IS HERE -- DOUBLE-CLICK THIS FILE")
    print()
    print(f"    {exe}")
    print(f"    or {ROOT_LAUNCHER.name}, in the project root")
    print()
    print(f"  {size / 1e6:.0f} MB in {len(list(out_dir.rglob('*')))} files.")
    print("  Copy the whole folder to put the app somewhere else.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    if "--root-launcher" in sys.argv[1:]:
        # Refreshed without rebuilding, so moving or renaming the app folder
        # costs a second rather than another PyInstaller run.
        print(write_root_launcher())
        raise SystemExit(0)
    raise SystemExit(main())
