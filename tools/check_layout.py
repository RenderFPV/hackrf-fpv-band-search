"""Layout sanity: nothing clipped, nothing squeezed to nothing.

The GUI works, but "works" and "can be read" are different claims. This checks
the numbers: every chart row inside the widget, every control big enough to hit
with a mouse, and the picture panel actually larger than the chart.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from fpv_rf import bands, sdr, ui  # noqa: E402


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


def main() -> int:
    app = QApplication.instance() or QApplication([])
    src = sdr.SimSource(signal_hz=5_802_000_000)
    src.start()
    time.sleep(0.2)
    win = ui.MainWindow(src, Args())
    win.resize(1180, 720)
    win.show()
    for _ in range(8):
        app.processEvents()
        time.sleep(0.05)

    bad: list[str] = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        print(f"  {'OK  ' if ok else 'FAIL'}  {what}{('  ' + detail) if detail else ''}")
        if not ok:
            bad.append(what)

    print(f"window {win.width()}x{win.height()}")
    print("\nleft column (picture + chart)")
    c = win.chart
    plot = c._plot()
    need = 20.0 + len(c.band_list) * (c.ROW_H + c.ROW_GAP)
    check(
        c.height() >= need,
        f"chart fits all {len(c.band_list)} band rows",
        f"needs {need:.0f}px, has {c.height()}px",
    )
    check(
        c.width() >= c.MARGIN_L + c.MARGIN_R + 200,
        "chart axis has room to draw in",
        f"{c.width()}px wide, {c.MARGIN_L}px of it is the label column",
    )
    check(
        plot.right() <= c.width() and plot.bottom() <= c.height(),
        "plot area is inside the widget",
        f"plot right {plot.right():.0f} of {c.width()}, "
        f"bottom {plot.bottom():.0f} of {c.height()}",
    )
    # The rightmost chart channel must not fall off the right-hand edge.
    rightmost = max(max(b.channels) for b in c.band_list)
    leftmost = min(min(b.channels) for b in c.band_list)
    x_r, x_l = c._x(rightmost), c._x(leftmost)
    check(
        c.lo_hz <= leftmost and rightmost <= c.hi_hz,
        "every chart channel is inside the drawn span",
        f"{bands.format_mhz(leftmost)} to {bands.format_mhz(rightmost)} "
        f"in a span drawn {bands.format_mhz(c.lo_hz)} to {bands.format_mhz(c.hi_hz)}",
    )
    # One pixel per MHz, and channel ticks 19-37 MHz apart, so this is also a
    # check that two adjacent channels do not land on the same pixel.
    px_per_mhz = plot.width() / (c.hi_hz - c.lo_hz) * 1e6
    gaps = []
    for b in c.band_list:
        ch = sorted(b.channels)
        gaps += [(ch[i + 1] - ch[i]) / 1e6 for i in range(len(ch) - 1)]
    check(
        px_per_mhz * min(gaps) > 3.0,
        "adjacent channels on a band are far enough apart to be told apart",
        f"{px_per_mhz:.1f} px/MHz, closest pair on a plan is "
        f"{min(gaps):.0f} MHz = {px_per_mhz*min(gaps):.0f}px",
    )
    check(
        c._x(c.hi_hz) - c._x(c.lo_hz) > 0.99 * plot.width() * 0 + 1,
        "frequency maps across the full axis",
        f"{bands.format_mhz(c.lo_hz)} at x={c._x(c.lo_hz):.0f}, "
        f"{bands.format_mhz(c.hi_hz)} at x={c._x(c.hi_hz):.0f}",
    )
    p = win.panel
    check(
        p.width() >= 320 and p.height() >= 240,
        "picture panel is at least the 4:3 minimum",
        f"{p.width()}x{p.height()}",
    )
    # The panel and the chart are stacked in one column, so they share a width by
    # construction; height is the axis the operator competes for.
    check(
        p.height() > c.height(),
        "the picture gets more height than the chart",
        f"panel {p.height()}px tall vs chart {c.height()}px",
    )
    check(
        c.height() < p.height() + 200,
        "the chart does not take over the window",
        f"chart is {c.height()/(c.height()+p.height())*100:.0f}% of the column",
    )

    print("\nright column (controls)")
    for name in (
        "banner", "mode", "freq", "lna", "vga", "region", "hop",
        "scan_btn", "auto_btn", "cancel_btn", "progress", "bright",
        "contrast", "invert", "smooth", "log",
    ):
        w = getattr(win, name)
        h = w.height() if hasattr(w, "height") else 0
        wdt = w.width() if hasattr(w, "width") else 0
        check(
            wdt > 0 and h >= 16,
            f"{name} is big enough to use",
            f"{wdt}x{h}",
        )
    check(
        win.banner.height() >= 64,
        "the alert banner is tall enough to read across a room",
        f"{win.banner.height()}px",
    )
    check(
        win.log.height() >= 100,
        "the alert log shows several lines at once",
        f"{win.log.height()}px, {win.log.count()} row(s)",
    )
    check(
        win.freq.minimum() * 1_000_000 <= 5_650_000_000
        and win.freq.maximum() * 1_000_000 >= 5_950_000_000,
        "the tune spinner covers the whole search span",
        f"{win.freq.minimum()}-{win.freq.maximum()} MHz",
    )
    # Whatever is forcing the column wide, name it -- a 288px sideways scroll
    # means something in there has a minimum size nobody asked for.
    print("\nwhat is setting the control column's minimum width")
    outer = win.scroll.widget()
    print(f"  column total minimumSizeHint {outer.minimumSizeHint().width()}px")
    lay = outer.layout()
    for k in range(lay.count()):
        wid = lay.itemAt(k).widget()
        if wid is None:
            continue
        title = wid.title() if hasattr(wid, "title") else ""
        m = wid.minimumSizeHint()
        print(f"    {m.width():5d}px  {type(wid).__name__:16s} {title!r}")
        sub = wid.layout()
        if sub is None:
            continue
        for j in range(sub.count()):
            it = sub.itemAt(j)
            if it is None:
                continue
            w2 = it.widget()
            if w2 is None:
                continue
            m2 = w2.minimumSizeHint()
            if m2.width() > 90:
                t2 = w2.text()[:34] if hasattr(w2, "text") else ""
                print(f"           {m2.width():5d}px  {type(w2).__name__:14s} {t2!r}")
    check(
        outer.minimumSizeHint().width() <= win.scroll.width() + 8,
        "no control column is wider than the space it is given",
        f"wants {outer.minimumSizeHint().width()}px, has {win.scroll.width()}px",
    )
    check(
        win.scroll.horizontalScrollBar().maximum() == 0,
        "the control column does not need scrolling sideways",
        f"h-scroll max {win.scroll.horizontalScrollBar().maximum()}, "
        f"column wants {win.scroll.widget().minimumSizeHint().width()}px in "
        f"{win.scroll.width()}px",
    )

    print("\nalert text, measured against its box")
    win.banner.set_event(
        __import__("fpv_rf.alerts", fromlist=["x"]).AlertEvent(
            kind=__import__("fpv_rf.alerts", fromlist=["x"]).AlertKind.FOUND,
            at=time.time(), key="k", band="F", channel=4,
            frequency_hz=5_800_000_000,
            signal=__import__("fpv_rf.dsp", fromlist=["x"]).SignalKind.VIDEO,
            detail="decodable NTSC video, sync 100%, SNR 9.7 dB",
            label="F4",
        )
    )
    app.processEvents()
    # With word wrap on, a long line is a taller box, not a clipped one -- so
    # what matters is that the wrapped text has vertical room.
    fm = win.banner._head.fontMetrics()
    have_w = win.banner._head.width()
    lines = max(1, -(-fm.horizontalAdvance(win.banner._head.text()) // have_w))
    need_h = lines * fm.height()
    check(
        need_h <= win.banner._head.height(),
        "the alert headline has room for the lines it wraps to",
        f"{lines} line(s) needs {need_h}px, label has {win.banner._head.height()}px",
    )
    fsub = win.banner._sub.fontMetrics()
    sub_lines = max(
        1, -(-fsub.horizontalAdvance(win.banner._sub.text()) // have_w)
    )
    check(
        win.banner.height() >= win.banner._head.height() + sub_lines * fsub.height(),
        "the banner grows to hold both lines",
        f"banner {win.banner.height()}px, headline {win.banner._head.height()}px "
        f"+ {sub_lines} detail line(s) of {fsub.height()}px",
    )

    win.close()
    src.stop()
    print("\nRESULT:", "PASS" if not bad else f"FAIL ({len(bad)})")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
