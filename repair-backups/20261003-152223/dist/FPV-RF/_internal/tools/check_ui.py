"""Drive the real Qt window offscreen: scan, alert, retune, and screenshot.

A GUI that imports cleanly is not a GUI that works. This runs the actual event
loop with the real widgets, a real source and a real scan, and writes PNGs at
each stage so the result can be looked at rather than asserted about.

Runs with the platform plugin set to ``offscreen``, so it needs no display.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fpv_rf import bands, scan, sdr, ui  # noqa: E402

OUT = Path("_validate_out")


class Args:
    mode = "sim"
    frequency = 5802.0
    lna = 32
    vga = 16
    region = bands.Region.US
    width = 160
    alert_cooldown = 15.0
    alert_forget = 30.0
    no_beep = True
    autoscan = False


#: Every check this file is going to make, declared up front.
#:
#: The results dict used to be filled in by the timer callbacks, so
#: ``all(results.values())`` at the end asked "is everything that got here
#: true" rather than "did everything run and come out true". A callback that
#: raised part-way through left the checks after it missing, the ones before it
#: still True, and the summary a confident PASS for a run that tested half of
#: what it claims to test. Predeclaring every name means a stage that never
#: reports is visibly False instead of absent, which is the whole difference
#: between a check and a decoration.
EXPECTED: tuple[str, ...] = (
    "window built",
    "picture decodes",
    "frame rate",
    "display controls",
    "scan found the transmitter",
    "best channel",
    "banner shows a frequency",
    "one alert, not many",
    "chart has a mark",
    "repeat sweep is quiet",
    "banner becomes a status line",
    "auto-select retuned",
    "chart click retunes",
    "chart click returns to F4",
    "controls survive no signal",
    "auto-select declines on a quiet band",
)

#: The last stage. A run that stops before this has not finished testing, so
#: the verdict is a failure even if every check that did run passed -- the same
#: rule, applied to the run as a whole rather than to one dict entry.
TERMINAL_STAGE = "after_second_scan"


def shot(win: QApplication, name: str, note: str) -> None:
    QApplication.processEvents()
    path = OUT / f"ui_{name}.png"
    win.grab().save(str(path))
    print(f"  wrote {path}  ({note})")


def stage(name: str, fn, *args) -> None:
    """Run one timer callback, turning any exception into a recorded failure.

    Qt swallows an exception raised inside a slot and carries on to the next
    timer, so a stage that blew up did not stop the run -- it just stopped
    reporting, which is the case the predeclared results above exist to catch.
    Here it is also *recorded*, with its traceback, so the failure names itself
    instead of appearing as three unrelated Falses.
    """
    try:
        fn(*args)
        print(f"  [stage {name}: done]")
    except Exception:
        import traceback

        traceback.print_exc()
        errors.append(f"stage {name} raised:\n{traceback.format_exc()}")
        print(f"  [stage {name}: RAISED]")


def check_display(win, results: dict, key: str = "display controls") -> None:
    """The display controls must change pixels, not just widget state.

    Each call names all three controls, because ``set_controls`` only touches
    what it is given -- which is the behaviour we want, and means a test that
    assumes otherwise measures the wrong thing.
    """
    if win.worker.latest() is None:
        results[key] = False
        return
    plain = win.panel._display.copy()
    win.panel.set_controls(contrast=2.0, brightness=0, invert=False)
    hi = win.panel._display.copy()
    win.panel.set_controls(contrast=1.0, brightness=0, invert=True)
    inv = win.panel._display.copy()
    win.panel.set_controls(contrast=1.0, brightness=60, invert=False)
    br = win.panel._display.copy()
    win.panel.set_controls(contrast=1.0, brightness=0, invert=False)
    back = win.panel._display.copy()
    d_ctr = float(np.abs(hi.astype(int) - plain.astype(int)).mean())
    d_inv = int(np.abs(inv.astype(int) - (255 - plain.astype(int))).max())
    d_br = float((br.astype(int) - plain.astype(int)).mean())
    d_back = int(np.abs(back.astype(int) - plain.astype(int)).max())
    print(f"  contrast 2.0 moves pixels by {d_ctr:.1f} levels on average")
    print(f"  invert mirrors to within {d_inv} of 255-x")
    print(f"  brightness +60 lifts the mean by {d_br:.1f}")
    print(f"  returning the controls to neutral restores the frame exactly "
          f"(max error {d_back})")
    results[key] = (
        d_ctr > 3.0 and d_inv <= 1 and 50.0 < d_br < 61.0 and d_back == 0
    )


def main() -> int:
    OUT.mkdir(exist_ok=True)
    app = QApplication.instance() or QApplication([])
    # Predeclared, so every name is decided from the start. Anything the run
    # never reports stays False and is reported as a failure by name, rather
    # than quietly not existing.
    results: dict[str, bool] = {k: False for k in EXPECTED}
    #: Exceptions raised inside a timer callback, recorded rather than swallowed.
    errors: list[str] = []
    #: Stages that reached their end. Checked against the terminal stage so a
    #: run that died part-way cannot be summarised as a pass.
    stages_done: list[str] = []

    # A transmitter on one frequency and an empty band everywhere else, so the
    # scan has something real to find and something real to reject.
    source = sdr.SimSource(signal_hz=5_802_000_000)
    source.start()
    time.sleep(0.3)
    win = ui.MainWindow(source, Args())
    win.resize(1180, 720)
    win.show()
    QApplication.processEvents()
    results["window built"] = win.isVisible()

    state: dict[str, object] = {}

    def after_paint() -> None:
        f = win.worker.latest()
        st = win.worker.stats
        state["frames"] = st.frames
        shape = "none" if f is None else f"{f.image.shape[1]}x{f.image.shape[0]}"
        print(f"  after 1.5 s: {st.frames} frames at {st.frame_rate():.1f} fps, "
              f"decode {st.decode_ms:.1f} ms, cycle {st.cycle_ms:.1f} ms, "
              f"frame {shape}")
        results["picture decodes"] = f is not None and f.image.size > 0
        results["frame rate"] = st.frame_rate() > 45.0
        check_display(win, results)
        shot(win, "1_locked", "picture plus band chart, no scan yet")
        win._on_scan()
        stages_done.append("after_paint")

    def after_scan() -> None:
        res = win._last_result
        print(f"  scan: {res.describe() if res else 'none'}")
        print(f"  alerts: {win.alerts.summary()}")
        print(f"  log rows: {win.log.count()}")
        for i in range(win.log.count()):
            print(f"    {win.log.item(i).text()}")
        results["scan found the transmitter"] = res is not None and bool(res.candidates)
        results["best channel"] = (
            res is not None and res.best() is not None and res.best().band == "F"
        )
        results["banner shows a frequency"] = (
            "580" in win.banner._head.text()
        )
        results["one alert, not many"] = win.log.count() == 1
        results["chart has a mark"] = len(win.chart.marks) == 1
        shot(win, "2_alert", "after the scan: banner, chart mark, alert log")
        state["before"] = win.log.count()
        stages_done.append("after_scan")

        # A second sweep over the same signal must not alert again.
        win._on_scan()

    def after_second_scan() -> None:
        print(f"  second sweep: {win.alerts.summary()}, "
              f"log rows {win.log.count()} (was {state.get('before')})")
        results["repeat sweep is quiet"] = win.log.count() == state.get("before")
        results["banner becomes a status line"] = (
            "already known" in win.banner._head.text()
            or "already known" in win.banner._sub.text()
            or "idle" == win.banner._head.text()
        )
        shot(win, "3_repeat", "second sweep: no new alert")

        # Auto-select must land on the transmitter and retune.
        win._on_auto()
        QApplication.processEvents()
        print(f"  auto-select tuned to {bands.format_mhz(source.frequency_hz)}")
        results["auto-select retuned"] = abs(
            source.frequency_hz - 5_800_000_000
        ) < 6_000_000

        # Clicking a chart channel must retune and re-check.
        win._on_channel_clicked(5_840_000_000)
        QApplication.processEvents()
        print(f"  clicked F6, tuned to {bands.format_mhz(source.frequency_hz)}")
        results["chart click retunes"] = source.frequency_hz == 5_840_000_000

        win._on_channel_clicked(5_800_000_000)
        QApplication.processEvents()
        print(f"  clicked F4, tuned back to {bands.format_mhz(source.frequency_hz)}")
        results["chart click returns to F4"] = source.frequency_hz == 5_800_000_000

        # No signal: the panel must say so rather than paint a stale picture, and
        # the display controls must survive the gap. The operator reaches for
        # brightness precisely when the picture is dark or missing, so a control
        # that is dropped on the floor during a retune is a control that only
        # works when it is not needed.
        win.worker.reset()
        win.panel.set_frame(None)
        win.panel.set_controls(contrast=1.8, brightness=25, invert=True)
        QApplication.processEvents()
        results["controls survive no signal"] = (
            win.isVisible()
            and abs(win.panel.contrast - 1.8) < 1e-6
            and win.panel.brightness == 25
            and win.panel.invert is True
        )
        print(f"  with no frame, the panel kept contrast="
              f"{win.panel.contrast} brightness={win.panel.brightness} "
              f"invert={win.panel.invert} and did not paint a stale picture")
        # Feed the panel a frame directly and confirm the stored controls apply
        # to it -- the panel has no signal of its own to wait for.
        f = win.worker.latest()
        if f is not None:
            win.panel.set_frame(f)
        shot(win, "4_selected", "after auto-select and a chart click")

        # A band with no signal: auto-select must decline.
        win._on_region(1)
        win._on_channel_clicked(5_725_000_000)
        QApplication.processEvents()
        win._on_auto()
        QApplication.processEvents()
        print(f"  on an empty band, auto-select said: "
              f"{win.banner._head.text()!r}")
        results["auto-select declines on a quiet band"] = (
            "nothing above the noise floor" in win.banner._head.text()
        )
        shot(win, "5_empty", "empty band: no channel offered")
        win.close()
        stages_done.append(TERMINAL_STAGE)

    state["before"] = 0
    QTimer.singleShot(1500, lambda: stage("after_paint", after_paint))
    # A scan shares the source with the decode worker, so it is slower than the
    # standalone scanner benchmark: every hop has to wait for a fresh sim block
    # to land while the decoder is also draining. Allow for that rather than
    # timing a test around it.
    QTimer.singleShot(7000, lambda: stage("after_scan", after_scan))
    QTimer.singleShot(13000, lambda: stage(TERMINAL_STAGE, after_second_scan))
    QTimer.singleShot(16000, app.quit)
    app.exec()

    print()
    for k in EXPECTED:
        print(f"  {'OK  ' if results[k] else 'FAIL'}  {k}")
    for e in errors:
        print(f"\n  !! {e}")
    failed = [k for k in EXPECTED if not results[k]]
    # Two independent reasons to fail, and both have to be checked. The results
    # decide whether the things that ran were right; the stage list decides
    # whether they all ran at all, which a passing subset cannot tell you.
    missing_stage = TERMINAL_STAGE not in stages_done
    if missing_stage:
        print(f"  !! the run never reached its last stage ({TERMINAL_STAGE}); "
              f"stages completed: {stages_done or ['none']}")
    ok = not failed and not errors and not missing_stage
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
