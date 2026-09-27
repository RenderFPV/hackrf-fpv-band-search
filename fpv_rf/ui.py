"""Qt front end: live picture, VTX band chart, and the alert panel.

Three things here are worth knowing before reading the code.

*The panel repaints at 60 Hz from whatever frame is newest, and the conversion
to a QImage is cached.* The decoder already runs at the field rate, so there is
nothing to gain from painting faster than the picture changes and a lot to lose
in per-paint allocation. The numpy array is wrapped rather than copied --
``QImage`` over ``Format_Grayscale8`` holds a pointer to the buffer -- so the
numpy array is kept alive on this object for exactly as long as the QImage that
points at it, and brightness/contrast/invert are recomputed only when one of
those changes or a new frame arrives.

*Scanning runs off the GUI thread.* A full sweep is seconds of tuner settling
time, and doing it here would freeze the window and the repaints with it.

*The scan thread and the decode thread share one source*, so changing channel is
a retune plus a buffer reset, not a restart. That is deliberate: a HackRF retune
is a process restart costing a few hundred milliseconds, and paying that
whenever the operator clicks a channel would make the chart feel broken.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from . import alerts, bands, dsp, scan
from .sdr import DEFAULT_SAMPLE_RATE, IQSource, make_source
from .video import DecodeWorker, VideoFrame

#: Repaint period. 60 Hz because that is the field rate; the decoder will not
#: beat it and painting slower than the picture changes just looks like judder.
PAINT_MS = 16

#: How often the numeric readouts are refreshed. 4 Hz is fast enough to feel
#: live and slow enough that the text does not shimmer.
STATS_MS = 250

KIND_COLOURS = {
    dsp.SignalKind.VIDEO: QColor(70, 220, 120),
    dsp.SignalKind.CARRIER: QColor(240, 190, 60),
    dsp.SignalKind.NOISE: QColor(150, 150, 160),
}


# --------------------------------------------------------------------------
# Picture
# --------------------------------------------------------------------------


class VideoPanel(QWidget):
    """The picture, plus the display controls that act on it.

    The image is *not* stretched to fill the widget. FPV video is 4:3 and a
    window that is not, and stretching to fit distorts the picture enough to make
    a wide-band test pattern look like a sync fault. Letterboxing also keeps the
    frame rate at 60 Hz: a widget that changes size every repaint cannot be
    blitted cheaply.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(16, 16, 18))
        self.setPalette(pal)

        self._frame: VideoFrame | None = None
        self._image: QImage | None = None
        # The QImage points into these, so they must outlive it. Not a
        # subtlety: without the reference the buffer is freed and the QImage
        # paints freed memory, which looks like a beautiful, plausible picture.
        self._buf = np.zeros((240, 160), dtype=np.uint8)
        self._display = np.zeros((240, 160), dtype=np.uint8)

        self.brightness = 0
        self.contrast = 1.0
        self.invert = False
        self.smooth = True
        self._key: tuple[int, int, float, bool, bool] | None = None

    # -- data in -----------------------------------------------------------

    def set_frame(self, frame: VideoFrame | None) -> None:
        """Take the newest decoded frame, if it is not the one already shown."""
        if frame is None:
            self._frame = None
            self._image = None
            self.update()
            return
        if self._frame is not None and frame.timestamp == self._frame.timestamp:
            return
        self._frame = frame
        self._rebuild()

    def _rebuild(self) -> None:
        """Apply the display controls and wrap the result for Qt.

        Does nothing when there is no frame. That is not just for tidiness: the
        operator turns brightness up *because* the picture is too dark, and the
        picture being too dark is a state the decoder passes through every time
        it retunes. Asserting a frame here would crash the app at exactly the
        moment the user reaches for the controls.
        """
        f = self._frame
        if f is None:
            self._key = None
            return
        src = f.image
        key = (src.ctypes.data, src.size, self.contrast, bool(self.invert), self.brightness)
        if key == self._key and self._image is not None:
            return
        self._key = key
        if src.shape != self._display.shape:
            self._display = np.empty(src.shape, dtype=np.uint8)
        out = self._display
        if self.contrast == 1.0 and self.brightness == 0 and not self.invert:
            np.copyto(out, src)
        else:
            # float32 rather than int16: 38400 pixels at 60 Hz is nothing, but a
            # float16 intermediate would clip on the multiply and the picture
            # would go flat in the highlights exactly when contrast is turned up.
            v = src.astype(np.float32)
            if self.contrast != 1.0:
                v -= np.float32(128.0)
                v *= np.float32(self.contrast)
                v += np.float32(128.0)
            if self.brightness:
                v += np.float32(self.brightness)
            if self.invert:
                np.subtract(np.float32(255.0), v, out=v)
            # Clip in float, then narrow once. Clipping straight into the uint8
            # output is a casting error, and clipping after narrowing would
            # already have wrapped -- 250 + 20 would be 14, not 255.
            np.clip(v, 0.0, 255.0, out=v)
            np.copyto(out, v, casting="unsafe")
        self._buf = out
        h, w = out.shape
        self._image = QImage(
            out.data, w, h, w, QImage.Format.Format_Grayscale8
        ).copy()
        self.update()

    def set_controls(
        self, brightness: int | None = None, contrast: float | None = None,
        invert: bool | None = None,
    ) -> None:
        changed = False
        if brightness is not None and brightness != self.brightness:
            self.brightness = brightness
            changed = True
        if contrast is not None and abs(contrast - self.contrast) > 1e-6:
            self.contrast = contrast
            changed = True
        if invert is not None and invert != self.invert:
            self.invert = invert
            changed = True
        if changed:
            self._key = None
            self._rebuild()

    # -- paint -------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(16, 16, 18))
        img = self._image
        f = self._frame
        if img is None or f is None:
            p.setPen(QColor(120, 120, 130))
            p.setFont(QFont("", 12))
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "no picture\n\nthe decoder has not locked a sync train\n"
                "run a band search to find a channel",
            )
            return

        # Letterbox, preserving 4:3.
        avail = self.rect()
        scale = min(avail.width() / img.width(), avail.height() / img.height())
        w = int(img.width() * scale)
        h = int(img.height() * scale)
        target = QRectF(
            avail.x() + (avail.width() - w) / 2.0,
            avail.y() + (avail.height() - h) / 2.0,
            w,
            h,
        )
        p.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform, self.smooth
        )
        p.drawImage(target, img)

        # Overlays, drawn at widget scale so they stay readable.
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont("", max(9, int(11 * min(1.0, scale / 3.0))))
        p.setFont(font)
        p.setPen(QColor(0, 0, 0, 160))
        bar = target.adjusted(0, 0, 0, -font.pointSize() * 1.8)
        p.fillRect(QRectF(bar.left(), bar.bottom() - 1, bar.width(), 1), QColor(0, 0, 0, 120))
        p.setPen(QColor(210, 210, 220))
        p.drawText(
            QRectF(bar.left() + 6, bar.bottom() - 2, bar.width() - 12, font.pointSize() * 1.6),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"{f.standard}  {f.line_rate_hz/1000:.2f} kHz lines  "
            f"sync {f.sync_quality*100:.0f}%",
        )
        if not f.locked:
            p.setPen(QColor(240, 120, 90))
            p.drawText(
                QRectF(bar.left() + 6, bar.top() + font.pointSize() * 1.4,
                       bar.width() - 12, font.pointSize() * 1.4),
                Qt.AlignmentFlag.AlignLeft,
                "no lock",
            )


# --------------------------------------------------------------------------
# Band chart
# --------------------------------------------------------------------------


@dataclass
class _Mark:
    """Something the chart wants to show about a frequency."""

    lo_hz: int
    hi_hz: int
    text: str
    colour: QColor
    strong: bool = False


class BandChart(QWidget):
    """Every channel of every band plan, over the region's frequency span.

    One row per band plan, which is how a race-timer screen shows it and for a
    good reason: on a single 300 MHz axis, eleven plans' worth of channels
    overlap into an unreadable thicket, and the whole point of the chart is to
    let the operator see that the thing they want is 3 MHz from where the scan
    says it is.

    Click a channel to tune it. Click anywhere else in a row to make that band
    plan the active one, which scopes both the search and auto-select to it.
    Shift-click a row to sweep that band plan and nothing else.
    """

    # object, not int, and this is not a style preference. Qt's `int` is a 32-bit
    # C int, and a frequency in Hz is around 5.8e9. Declaring this `Signal(int)`
    # truncated every click to a 32-bit value (5865000000 -> 1570032704) and the
    # slot then failed to resolve at all, so clicking a channel on the chart did
    # nothing and Qt logged an AttributeError nobody was watching for. The
    # integer arrives intact.
    channelClicked = Signal(object)       # frequency in Hz, a Python int
    bandActivated = Signal(int)           # band index, now the active plan
    bandScanRequested = Signal(int)       # band index, sweep this plan only

    MARGIN_L = 58
    MARGIN_R = 12
    ROW_H = 22
    ROW_GAP = 3

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.lo_hz, self.hi_hz = bands.default_scan_span(bands.Region.US)
        self.band_list: list[bands.Band] = list(bands.BANDS_58)
        self.marks: list[_Mark] = []
        self.tune_hz = 0
        self.selected_hz = 0
        self.active_row: int | None = None
        self._hover: tuple[int, int] | None = None     # (row, channel index)

    @property
    def active_band(self) -> bands.Band | None:
        """The band plan the operator has chosen, if any."""
        if self.active_row is None:
            return None
        if 0 <= self.active_row < len(self.band_list):
            return self.band_list[self.active_row]
        return None

    def set_active_row(self, row: int | None) -> None:
        self.active_row = row
        self.update()

    def set_span(self, lo_hz: int, hi_hz: int) -> None:
        """Set the region span, widened to cover the band plans being drawn.

        The chart shows 10 band plans, and the B plan runs 5645-5945 MHz while
        the US region span is 5650-5950. Drawing the chart to the region's span
        would clip B1 and B8 off the ends of the axis, which is a bad way to
        discover that the channel you want is on the far left. So the axis covers
        the chart, and the region only decides what is *searched* and which
        channels are legal -- a distinction the tooltip makes explicit.
        """
        lo, hi = int(lo_hz), int(hi_hz)
        if self.band_list:
            lo = min(lo, min(min(b.channels) for b in self.band_list))
            hi = max(hi, max(max(b.channels) for b in self.band_list))
        self.lo_hz, self.hi_hz = lo, hi
        self.update()

    def set_bands(self, band_list: list[bands.Band]) -> None:
        self.band_list = list(band_list)
        self.set_span(self.lo_hz, self.hi_hz)

    def set_marks(self, marks: list[_Mark]) -> None:
        self.marks = marks
        self.update()

    def set_tune(self, freq_hz: int, selected_hz: int = 0) -> None:
        self.tune_hz = int(freq_hz)
        self.selected_hz = int(selected_hz or freq_hz)
        self.update()

    # -- geometry ----------------------------------------------------------

    def _plot(self) -> QRectF:
        top = 20.0
        h = len(self.band_list) * (self.ROW_H + self.ROW_GAP)
        return QRectF(
            self.MARGIN_L,
            top,
            max(1.0, self.width() - self.MARGIN_L - self.MARGIN_R),
            max(1.0, h),
        )

    def _x(self, freq_hz: float) -> float:
        r = self._plot()
        span = max(1, self.hi_hz - self.lo_hz)
        return r.left() + r.width() * (freq_hz - self.lo_hz) / span

    def _f(self, x: float) -> int:
        r = self._plot()
        span = max(1, self.hi_hz - self.lo_hz)
        return int(self.lo_hz + (x - r.left()) / r.width() * span)

    def _row_at(self, y: float) -> int:
        return int((y - self._plot().top()) // (self.ROW_H + self.ROW_GAP))

    def _channel_at(self, pos: QPointF) -> tuple[int, int] | None:
        row = self._row_at(pos.y())
        if not (0 <= row < len(self.band_list)):
            return None
        f = self._f(pos.x())
        chans = self.band_list[row].channels
        for i, c in enumerate(chans):
            if abs(f - c) <= 9_000_000:      # 18 MHz hit box, wider than the tick
                return row, i
        return None

    # -- paint -------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(22, 22, 26))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = self._plot()
        font = QFont("", 8)
        p.setFont(font)

        # MHz grid.
        p.setPen(QPen(QColor(48, 48, 56), 1))
        step = 50_000_000
        f = (self.lo_hz // step) * step
        while f <= self.hi_hz:
            x = self._x(f)
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(QColor(120, 120, 132))
            p.drawText(
                QRectF(x - 22, 2, 44, 14),
                Qt.AlignmentFlag.AlignCenter,
                f"{f/1e6:.0f}",
            )
            p.setPen(QPen(QColor(48, 48, 56), 1))
            f += step
        p.setPen(QColor(120, 120, 132))
        p.drawText(
            QRectF(2, 2, self.MARGIN_L - 4, 14),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            "MHz",
        )

        occupied = [(m.lo_hz, m.hi_hz, m.colour) for m in self.marks]

        for row, band in enumerate(self.band_list):
            y = r.top() + row * (self.ROW_H + self.ROW_GAP)
            rect = QRectF(r.left(), y, r.width(), self.ROW_H)
            active = row == self.active_row
            p.fillRect(rect, QColor(38, 42, 56) if active else QColor(30, 30, 36))
            p.setPen(QColor(120, 170, 255) if active else QColor(150, 150, 160))
            p.drawText(
                QRectF(2, y, self.MARGIN_L - 6, self.ROW_H),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                band.key,
            )
            if active:
                # An outline, not just a fill: the active row has to be
                # identifiable at a glance while the operator is watching the
                # picture rather than the chart.
                p.setPen(QPen(QColor(90, 150, 255), 1))
                p.drawRect(rect.adjusted(0, 0, -1, -1))
            for m in self.marks:
                if band.key not in m.text:
                    continue
                x0, x1 = self._x(m.lo_hz), self._x(m.hi_hz)
                p.fillRect(
                    QRectF(x0, y + 2, max(3.0, x1 - x0), self.ROW_H - 4), m.colour
                )
            for i, c in enumerate(band.channels):
                x = self._x(c)
                hot = any(lo - 9e6 <= c <= hi + 9e6 for lo, hi, _ in occupied)
                p.setPen(
                    QPen(QColor(90, 220, 130) if hot else QColor(96, 96, 110), 2)
                )
                p.drawLine(
                    QPointF(x, y + 4), QPointF(x, y + self.ROW_H - 4)
                )
                if self._hover == (row, i):
                    p.setPen(QColor(255, 255, 255))
                    p.drawText(
                        QRectF(x - 30, y - 2, 60, 12),
                        Qt.AlignmentFlag.AlignCenter,
                        f"{c/1e6:.0f}",
                    )

        # What the tuner is on, and what is selected.
        if self.tune_hz:
            x = self._x(self.tune_hz)
            p.setPen(QPen(QColor(90, 170, 255), 2, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(x, r.top() - 4), QPointF(x, r.bottom() + 4))
        if self.selected_hz:
            x = self._x(self.selected_hz)
            p.setPen(QPen(QColor(255, 120, 120), 1))
            p.drawLine(QPointF(x, r.top() - 4), QPointF(x, r.bottom() + 4))

        p.setPen(QColor(100, 100, 112))
        active = self.active_band
        scope = f"scoping to {active.key}" if active else "all bands"
        p.drawText(
            QRectF(r.left(), r.bottom() + 4, r.width(), 14),
            Qt.AlignmentFlag.AlignRight,
            f"{len(self.marks)} finding(s)  |  {scope}  |  "
            f"click a tick to tune, a row to scope, shift-click to sweep",
        )

    # -- input -------------------------------------------------------------

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        hit = self._channel_at(event.position())
        if hit != self._hover:
            self._hover = hit
            self.update()
        if hit:
            band = self.band_list[hit[0]]
            f = band.channels[hit[1]]
            self.setToolTip(
                f"{band.label} channel {hit[1]+1}\n"
                f"{bands.format_mhz(f)}  ({f/1e6:.0f} MHz)\n"
                f"{'legal here' if bands.is_legal(f) else 'outside this region'}"
            )
        else:
            self.setToolTip("")

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = None
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Tune a tick, pick a band, or sweep a band.

        Three gestures on one widget, so which one a click means has to be
        decided by what is under the pointer and never by what feels natural to
        guess at. A tick is a frequency and tunes it. The rest of the row is
        band *identity*, and selecting it is what scopes the search -- which is
        the thing an operator racing on one plan actually wants, and which
        previously had no gesture at all outside a 9 MHz box around each tick.
        Shift is the explicit "sweep this plan" and is deliberately not the
        default, because it starts a sweep the operator then has to wait for.
        """
        pos = event.position()
        hit = self._channel_at(pos)
        row = self._row_at(pos.y())
        in_row = 0 <= row < len(self.band_list)
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        if in_row and shift:
            self.bandScanRequested.emit(row)
            return
        if hit:
            self.channelClicked.emit(self.band_list[hit[0]].channels[hit[1]])
            return
        if in_row:
            self.bandActivated.emit(row)
            return
        # A click on empty space below the last row clears the scoping, so
        # there is a way back to "search everything" that is not a restart.
        if event.button() == Qt.MouseButton.LeftButton:
            self.bandActivated.emit(-1)


# --------------------------------------------------------------------------
# Alert panel
# --------------------------------------------------------------------------


class AlertBanner(QFrame):
    """The one line that has to be readable across the room."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(64)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 6, 10, 6)
        self._head = QLabel("idle")
        self._head.setWordWrap(True)
        f = QFont("", 15)
        f.setBold(True)
        self._head.setFont(f)
        self._sub = QLabel("no band search has run yet")
        # Word wrap on *both* labels, and this is load-bearing rather than
        # cosmetic: a QLabel without it reports the full unwrapped text width as
        # its minimum, so one long detail line silently sets the minimum width
        # of the entire control column and the scroll area answers with a
        # horizontal scrollbar.
        self._sub.setWordWrap(True)
        self._sub.setStyleSheet("color: #a0a0ad;")
        self._layout.addWidget(self._head)
        self._layout.addWidget(self._sub)

    def set_idle(self, message: str = "idle") -> None:
        self._head.setText(message)
        self._head.setStyleSheet("color: #d0d0da;")
        self._sub.setText("run a band search to look for energy above the noise floor")
        self.setStyleSheet("background: #26262c;")

    def set_event(self, ev: alerts.AlertEvent, repeats: int = 1) -> None:
        if ev.kind is alerts.AlertKind.LOST:
            self._head.setText(f"LOST  {ev.label}")
            self._head.setStyleSheet("color: #f0b060;")
            self.setStyleSheet("background: #3a2c1c;")
        else:
            colour = KIND_COLOURS.get(ev.signal, QColor(220, 220, 220))
            self._head.setText(
                f"RF ABOVE NOISE FLOOR   {ev.label} @ "
                f"{bands.format_mhz(ev.frequency_hz)}"
            )
            self._head.setStyleSheet(f"color: {colour.name()};")
            self.setStyleSheet("background: #1d2a22;")
        extra = f"  (seen {repeats}x)" if repeats > 1 else ""
        self._sub.setText(ev.detail + extra)


# --------------------------------------------------------------------------
# Scan thread
# --------------------------------------------------------------------------


class ScanThread(QThread):
    """Runs one band search off the GUI thread.

    The tuner settles for hundreds of milliseconds per hop and a real sweep is
    tens of them, so this is seconds of work. Doing it on the GUI thread would
    stop the repaints, and with them the picture -- which is the thing the
    operator is looking at while they wait.
    """

    progress = Signal(str, float)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, source: IQSource, lo_hz: int, hi_hz: int,
                 hop_frac: float, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.source = source
        self.lo_hz, self.hi_hz = int(lo_hz), int(hi_hz)
        self.hop_frac = float(hop_frac)
        self._scanner = scan.BandScanner(source, hop_frac=self.hop_frac)

    def cancel(self) -> None:
        self._scanner.cancel()

    def run(self) -> None:  # noqa: D102
        try:
            res = self._scanner.scan(
                lo_hz=self.lo_hz, hi_hz=self.hi_hz,
                progress=lambda m, f: self.progress.emit(m, f),
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished_ok.emit(res)


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self, source: IQSource, args) -> None:
        super().__init__()
        self.args = args
        self.source = source
        self.scanner = scan.BandScanner(source)
        self.worker = DecodeWorker(source, width=args.width)
        self.alerts = alerts.AlertEngine(
            cooldown_s=args.alert_cooldown, forget_s=args.alert_forget
        )
        self.beeper = alerts.Beeper(enabled=not args.no_beep)
        self.alerts.add_listener(self._on_alert)     # from the scan thread
        self._scan_thread: ScanThread | None = None
        self._last_result: scan.ScanResult | None = None
        self._last_event: alerts.AlertEvent | None = None
        #: Band plan the operator picked on the chart, scoping search and
        #: auto-select. None means "every plan", which is the startup state
        #: because a default would silently hide channels on the other plans.
        self._active_band: bands.Band | None = None
        #: LNA setting to restore when the amp is switched back on. Remembered
        #: rather than hard-coded so switching the amp off and on again is not
        #: a way to lose your gain setup.
        self._lna_saved_db = int(source.lna_gain_db)

        self.setWindowTitle("FPV band search and alert")
        self._build()

        self.worker.start()
        self._paint = QTimer(self)
        self._paint.timeout.connect(self._on_paint)
        self._paint.start(PAINT_MS)
        self._stats = QTimer(self)
        self._stats.timeout.connect(self._on_stats)
        self._stats.start(STATS_MS)

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_video())
        split.addWidget(self._build_side())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([720, 460])
        self.setCentralWidget(split)

        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status = sb

        self.banner.set_idle()
        # The status bar and the scan button both read the active band, so this
        # has to come after both exist.
        self._update_scan_label()

    def _build_video(self) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(6, 6, 6, 6)
        # 3:2, not 1:2. The picture is what the operator is here for, and the
        # chart needs 20 + rows*25 px to be readable -- 270 for ten plans. That
        # ratio puts the panel at ~420px and the chart at exactly what it needs
        # in a 720px window, with the panel taking the slack if there is any.
        self.panel = VideoPanel()
        v.addWidget(self.panel, 3)
        self.chart = BandChart()
        self.chart.set_span(*bands.default_scan_span(self.args.region))
        self.chart.channelClicked.connect(self._on_channel_clicked)
        self.chart.bandActivated.connect(self._on_band_activated)
        self.chart.bandScanRequested.connect(self._on_band_scan)
        v.addWidget(self.chart, 2)
        return box

    def _build_side(self) -> QWidget:
        outer = QWidget()
        v = QVBoxLayout(outer)
        v.setContentsMargins(6, 6, 6, 6)

        self.banner = AlertBanner()
        v.addWidget(self.banner)

        # -- source ------------------------------------------------------
        g = QGroupBox("source")
        gv = QGridLayout(g)
        self.mode = QComboBox()
        self.mode.addItems(["hackrf", "file", "sim"])
        self.mode.setCurrentText(self.args.mode)
        self.mode.setEnabled(False)
        self.mode.setToolTip("fixed at startup; restart to change source")
        gv.addWidget(QLabel("mode"), 0, 0)
        gv.addWidget(self.mode, 0, 1)

        self.freq = QSpinBox()
        self.freq.setRange(5000, 7100)
        self.freq.setSuffix(" MHz")
        self.freq.setSingleStep(1)
        self.freq.setValue(int(round(self.source.frequency_hz / 1e6)))
        self.freq.valueChanged.connect(self._on_freq_spin)
        gv.addWidget(QLabel("tune"), 1, 0)
        gv.addWidget(self.freq, 1, 1)

        self.lna = QSpinBox()
        self.lna.setRange(0, 62)
        self.lna.setValue(self.source.lna_gain_db)
        self.lna.valueChanged.connect(self._on_gain)
        gv.addWidget(QLabel("LNA dB"), 2, 0)
        gv.addWidget(self.lna, 2, 1)
        self.vga = QSpinBox()
        self.vga.setRange(0, 62)
        self.vga.setValue(self.source.vga_gain_db)
        self.vga.valueChanged.connect(self._on_gain)
        gv.addWidget(QLabel("VGA dB"), 3, 0)
        gv.addWidget(self.vga, 3, 1)

        self.amp = QCheckBox("RF amp")
        self.amp.setChecked(True)
        self.amp.setToolTip(
            "Switch the RF amplifier off, or back on.\n\n"
            "A real bypass (-a 0), not a gain reduction: the amplifier is off, "
            "not turned down. Requires a transfer helper that supports the "
            "flag; the stock Great Scott build does not, and then this control "
            "is inert.\n\n"
            "The point of it is diagnosis. Amplified noise floor looks exactly "
            "like a weak transmitter, so switch the amplifier off: if the "
            "reading survives, the energy is really out there.\n\n"
            "It costs a retune, like any change to the gains."
        )
        self.amp.toggled.connect(self._on_amp)
        if not getattr(self.source, "amp_supported", True):
            self.amp.setChecked(False)
            self.amp.setEnabled(False)
            self.amp.setToolTip(
                "This build of hackrf_transfer has no -a flag, so the "
                "amplifier cannot be switched from here.\n\n"
                "The Mayhem build has it. Download that firmware's utils "
                "folder and point HACKRF_TRANSFER at its hackrf_transfer.exe."
            )
        gv.addWidget(self.amp, 2, 2)
        v.addWidget(g)

        # -- search ------------------------------------------------------
        g = QGroupBox("band search")
        gv = QGridLayout(g)
        self.region = QComboBox()
        for r in bands.REGIONS:
            self.region.addItem(bands.REGIONS[r].label, r)
        self.region.setCurrentIndex(
            max(0, self.region.findData(self.args.region))
        )
        # Let the combo shrink. Its natural width is the widest item
        # ("United States (5.65-5.95 GHz)"), which sets a floor on the whole
        # column for a widget whose job is to be scanned, not read in full.
        self.region.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.region.setMinimumContentsLength(14)
        self.region.setToolTip(
            "Sets the default search span and which channels count as legal.\n"
            + "\n".join(
                f"{bands.REGIONS[r].label}: {bands.REGIONS[r].note}"
                for r in bands.REGIONS
            )
        )
        self.region.currentIndexChanged.connect(self._on_region)
        gv.addWidget(QLabel("region"), 0, 0, 1, 2)
        gv.addWidget(self.region, 0, 2)

        self.hop = QComboBox()
        self.hop.addItem("fast  9 MHz grid", 0.9)
        self.hop.addItem("medium  5 MHz grid", 0.5)
        self.hop.addItem("fine  2 MHz grid", 0.2)
        self.hop.setCurrentIndex(0)
        self.hop.setToolTip(
            "The grid sets how precisely a transmitter's frequency is known:\n"
            "a 9 MHz grid pins it to +-4.5 MHz, a 2 MHz grid to +-1 MHz.\n"
            "Finer costs proportionally more sweeps -- and on real hardware,\n"
            "one tuner settling period per hop."
        )
        gv.addWidget(QLabel("grid"), 1, 0, 1, 2)
        gv.addWidget(self.hop, 1, 2)

        self.scan_btn = QPushButton("Scan band")
        self.scan_btn.setDefault(True)
        self.scan_btn.setToolTip(
            "Sweep the whole search span at a coarse grid and report every "
            "frequency with energy above the noise floor."
        )
        self.scan_btn.clicked.connect(self._on_scan)
        gv.addWidget(self.scan_btn, 2, 0, 1, 3)
        self.auto_btn = QPushButton("Auto-select")
        self.auto_btn.setToolTip(
            "Tune the strongest decodable channel from the last search and "
            "watch it.\n\nOn a band with nothing on it this deliberately does "
            "nothing: a search that always names a channel teaches you to "
            "ignore it."
        )
        self.auto_btn.clicked.connect(self._on_auto)
        gv.addWidget(self.auto_btn, 3, 0, 1, 2)
        self.cancel_btn = QPushButton("Stop")
        self.cancel_btn.setToolTip("Stop the running search. A stopped search "
                                   "reports nothing, rather than reporting that "
                                   "everything it did not reach has gone.")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        gv.addWidget(self.cancel_btn, 3, 2)

        self.autotune = QCheckBox("tune it automatically")
        self.autotune.setChecked(True)
        self.autotune.setToolTip(
            "When a search finds something, tune it and show it, instead of\n"
            "making you press Auto-select afterwards.\n\n"
            "Still declines when there is nothing above the noise floor, and\n"
            "when the active band has nothing on it. Finding a signal and\n"
            "watching it are one action, not two."
        )
        gv.addWidget(self.autotune, 4, 0, 1, 3)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        gv.addWidget(self.progress, 5, 0, 1, 3)
        v.addWidget(g)

        # -- display -----------------------------------------------------
        g = QGroupBox("picture")
        gv = QGridLayout(g)
        self.bright = QSlider(Qt.Orientation.Horizontal)
        self.bright.setRange(-100, 100)
        # Qt's default slider is 15px tall, which is a 15px target for a
        # control the operator is meant to be able to grab quickly. These are
        # the controls people reach for mid-flight.
        self.bright.setMinimumHeight(22)
        self.bright.setStyleSheet("QSlider::handle:horizontal {width: 14px;}")
        self.bright.valueChanged.connect(
            lambda v_: self.panel.set_controls(brightness=v_)
        )
        gv.addWidget(QLabel("brightness"), 0, 0)
        gv.addWidget(self.bright, 0, 1)
        self.contrast = QSlider(Qt.Orientation.Horizontal)
        self.contrast.setRange(50, 250)
        self.contrast.setValue(100)
        self.contrast.setMinimumHeight(22)
        self.contrast.setStyleSheet("QSlider::handle:horizontal {width: 14px;}")
        self.contrast.valueChanged.connect(
            lambda v_: self.panel.set_controls(contrast=v_ / 100.0)
        )
        gv.addWidget(QLabel("contrast"), 1, 0)
        gv.addWidget(self.contrast, 1, 1)
        self.invert = QCheckBox("invert")
        self.invert.toggled.connect(self.panel.set_controls)
        self.smooth = QCheckBox("smooth scaling")
        self.smooth.setChecked(True)
        self.smooth.toggled.connect(self._on_smooth)
        row = QHBoxLayout()
        row.addWidget(self.invert)
        row.addWidget(self.smooth)
        gv.addLayout(row, 2, 0, 1, 2)
        v.addWidget(g)

        # -- log ---------------------------------------------------------
        # The group title is just "alerts". Putting the running summary in the
        # title looks tidy and is quietly disastrous: a QGroupBox's minimum
        # width includes its title, so a 47-character title sets a floor on the
        # entire control column and the scroll area answers with a horizontal
        # scrollbar. The summary is a wrapping label inside instead.
        g = QGroupBox("alerts")
        gv = QVBoxLayout(g)
        self.log = QListWidget()
        self.log.setMinimumHeight(120)
        gv.addWidget(self.log, 1)
        row = QHBoxLayout()
        self.alert_count = QLabel(self.alerts.summary())
        self.alert_count.setWordWrap(True)
        self.alert_count.setStyleSheet("color: #a0a0ad;")
        row.addWidget(self.alert_count, 1)
        clear = QPushButton("clear")
        clear.setToolTip("Empty the alert log. Findings already alerted on stay "
                         "known, so clearing the list does not re-alert them.")
        clear.clicked.connect(self._clear_log)
        row.addWidget(clear, 0)
        gv.addLayout(row)
        v.addWidget(g, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(outer)
        scroll.setMinimumWidth(400)
        self.scroll = scroll
        return scroll

    # -- callbacks ---------------------------------------------------------

    def _on_smooth(self, on: bool) -> None:
        self.panel.smooth = bool(on)
        self.panel.update()

    def _clear_log(self) -> None:
        """Empty the visible list, keeping the engine's memory of what it has said.

        These are different things on purpose. The list is a reading of past
        events; the engine's `_seen` map is what stops the same drone alerting
        every sweep. Clearing the list because you tidied the screen should not
        turn into eight identical alerts on the next sweep.
        """
        self.log.clear()

    def _on_region(self, _idx: int) -> None:
        region = self.region.currentData()
        lo, hi = bands.default_scan_span(region)
        self.chart.set_span(lo, hi)
        # A new region is a new set of legal channels and a new default sweep,
        # so a band chosen under the old one would silently keep scoping the new
        # search to a plan the operator never picked here.
        self._active_band = None
        self.chart.set_active_row(None)

    def _update_scan_label(self) -> None:
        """Put the actual sweep width on the button.

        On hardware the sweep is dominated by one tuner settling period per
        hop, so the width of the span *is* the wait. An operator who cannot see
        it has no way to connect "this takes ages" to the control that fixes it.
        """
        lo, hi = self._search_span()
        width_mhz = (hi - lo) / 1e6
        self.scan_btn.setText(f"Scan band  ({width_mhz:.0f} MHz)")
        self.scan_btn.setToolTip(
            f"Sweep {bands.format_mhz(lo)}-{bands.format_mhz(hi)} "
            f"({width_mhz:.0f} MHz) at a coarse grid and report every frequency "
            f"with energy above the noise floor.\n\n"
            f"On real hardware each hop costs a tuner settling period, so the "
            f"time is roughly proportional to this width. Click a band row on "
            f"the chart to scope the sweep to one plan."
        )

    def _on_freq_spin(self, mhz: int) -> None:
        self._tune(int(mhz) * 1_000_000, ask_alert=True)

    def _on_channel_clicked(self, freq_hz: int) -> None:
        m = bands.match_frequency(freq_hz)
        if m is not None:
            self.freq.blockSignals(True)
            self.freq.setValue(int(round(m.frequency_hz / 1e6)))
            self.freq.blockSignals(False)
        self._tune(freq_hz, ask_alert=True)

    def _on_band_activated(self, row: int) -> None:
        """Click a band row: make that plan the one search and auto-select use.

        ``row < 0`` is a click on empty space, which clears the scoping. Without
        it, picking a band would be a one-way door for the rest of the session.
        """
        if row < 0:
            self._active_band = None
            self.chart.set_active_row(None)
            self._update_scan_label()
            self._status.showMessage("searching every band plan")
            return
        band = self.chart.band_list[row]
        self._active_band = band
        self.chart.set_active_row(row)
        self._update_scan_label()
        lo, hi = band.frequency_span()
        self._status.showMessage(
            f"{band.label} is the active band -- search and auto-select now "
            f"look only at {bands.format_mhz(lo)}-{bands.format_mhz(hi)}"
        )

    def _on_band_scan(self, row: int) -> None:
        """Shift-click a band row: sweep just that band plan."""
        band = self.chart.band_list[row]
        self._status.showMessage(f"scanning the {band.label} band plan only")
        self._start_scan(band.frequency_span()[0], band.frequency_span()[1])

    def _on_amp(self, on: bool) -> None:
        """Switch the RF amplifier off, or back on.

        This is a real switch, not a gain: the transfer helper takes ``-a 0`` to
        bypass the amplifier entirely. That is a different thing from asking for
        0 dB of LNA gain, which leaves the LNA powered and only stops it
        amplifying -- the first version of this control did that, and it was
        wrong.

        Worth having as its own control for what it diagnoses: with the
        amplifier off, a signal that was really just your own noise floor
        amplified stops looking like a transmitter. If a strong reading survives
        switching it off, the energy is out there.

        Costs a retune, like any LNA or gain change: the flags are command-line
        arguments, so they can only change when the process is replaced. Not a
        control to flick mid-sweep.
        """
        if not on:
            self._lna_saved_db = self.lna.value()
        self.source.amp_enabled = bool(on)
        self._tune(self.source.frequency_hz, ask_alert=False)

    def _on_gain(self, _v: int) -> None:
        self.source.lna_gain_db = self.lna.value()
        self.source.vga_gain_db = self.vga.value()
        self._tune(self.source.frequency_hz, ask_alert=False)

    def _tune(self, freq_hz: int, ask_alert: bool = False,
              clear_alerts: bool = True) -> None:
        if not self.source.tune(freq_hz):
            QMessageBox.warning(
                self, "Tune failed",
                f"Could not tune to {bands.format_mhz(freq_hz)}.\n"
                f"{self.source.stats.last_error or 'the source reported no reason.'}",
            )
            return
        # The ring still holds the previous frequency. Dropping it means the
        # decoder does not spend a field trying to lock a sync train in samples
        # from a different channel.
        self.source.reset_drain()
        self.worker.reset()
        # Clearing is right when the operator moved the tuner themselves: what
        # was found on the old channel is no longer what they are looking at.
        # It is wrong when *we* moved the tuner onto a channel the search just
        # found, because the finding is still true and still current -- clearing
        # there would drop it from the seen-map and make the next sweep raise an
        # identical alert for the same drone.
        if clear_alerts:
            self.alerts.clear()
        self.chart.set_tune(self.source.frequency_hz)
        if ask_alert:
            self._check_current()

    def _search_span(self) -> tuple[int, int]:
        """What 'Scan band' sweeps: the active band plan if one is chosen.

        Scoping the *sweep* to the active band is the point of choosing one. It
        is also the single biggest lever on sweep time, since on hardware every
        MHz of span is retune time -- Raceband's 8 channels are a fifth of the
        US region span, so scoping is the difference between a sweep you wait
        for and one you do not.
        """
        if self._active_band is not None:
            return self._active_band.frequency_span()
        return bands.default_scan_span(self.region.currentData())

    def _on_scan(self) -> None:
        lo, hi = self._search_span()
        self._start_scan(lo, hi)

    def _on_auto(self) -> None:
        self._select_best(announce=True)

    def _select_best(self, announce: bool) -> bool:
        """Tune the best decodable candidate from the last search, if any.

        Returns whether it tuned something. Declines on an empty band, and
        declines when the active band plan has nothing on it even if another
        plan does. Both refusals are the feature: a selector that always
        produces an answer is worse than no selector, because the operator
        cannot tell the difference between "found your quad" and "found the
        noise floor".
        """
        if self._last_result is None:
            if announce:
                self._status.showMessage("run a band search first")
            return False
        best = self.scanner.suggest(self._last_result, self._active_band)
        if best is None:
            if self._active_band is not None and self._last_result.decodable:
                self._status.showMessage(
                    f"nothing decodable on {self._active_band.label} -- "
                    f"cleared to search every band"
                )
            else:
                self.banner.set_idle("nothing above the noise floor")
                if announce:
                    self._status.showMessage(
                        "no channel found -- auto-select deliberately declines "
                        "to guess, so an empty band stays empty"
                    )
            return False
        self.freq.blockSignals(True)
        self.freq.setValue(int(round(best.frequency_hz / 1e6)))
        self.freq.blockSignals(False)
        self._tune(best.frequency_hz, ask_alert=False, clear_alerts=False)
        self._status.showMessage(f"watching {best.label()}")
        return True

    def _on_cancel(self) -> None:
        if self._scan_thread:
            self._scan_thread.cancel()
            self._status.showMessage("stopping the search...")

    def _start_scan(self, lo_hz: int, hi_hz: int) -> None:
        if self._scan_thread and self._scan_thread.isRunning():
            self._status.showMessage("a search is already running")
            return
        if not self.source.retunable:
            QMessageBox.information(
                self, "Cannot sweep a recording",
                f"This source is a recorded capture, taken at "
                f"{bands.format_mhz(self.source.frequency_hz)}.\n\n"
                "It holds one frequency, so sweeping it would report the same "
                "signal at every hop. It is assessed in place instead.",
            )
            return
        self.progress.setValue(0)
        self.scan_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        # Deliberately *not* alerts.clear() here. The whole point of the
        # cooldown and the seen-count is that the second sweep over the same
        # drone says "already known" instead of raising an identical alert; the
        # history has to survive the sweep that follows the one that found it.
        th = ScanThread(self.source, lo_hz, hi_hz, self.hop.currentData(), self)
        th.progress.connect(self._on_progress)
        th.finished_ok.connect(self._on_scan_done)
        th.failed.connect(self._on_scan_failed)
        self._scan_thread = th
        th.start()
        self._status.showMessage(
            f"sweeping {bands.format_mhz(lo_hz)}-{bands.format_mhz(hi_hz)}..."
        )

    def _on_progress(self, message: str, frac: float) -> None:
        self.progress.setValue(int(frac * 1000))

    def _on_scan_done(self, res: scan.ScanResult) -> None:
        self.progress.setValue(1000)
        self.scan_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._last_result = res
        events = self.alerts.update(res)
        self._refresh_marks()
        self.alert_count.setText(self.alerts.summary())
        self._status.showMessage(res.describe())
        if not events:
            self.banner.set_idle(
                "nothing above the noise floor"
                if not res.candidates
                else f"already known: {self.alerts.active[0].label()}"
                if self.alerts.active
                else "idle"
            )
        if self.autotune.isChecked():
            # After the alert bookkeeping, not before: _tune() clears the
            # alert engine, and doing that first would make a brand new finding
            # look like it had never been raised.
            self._select_best(announce=False)

    def _on_scan_failed(self, message: str) -> None:
        self.scan_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        QMessageBox.warning(self, "Band search failed", message)
        self._status.showMessage(f"band search failed: {message}")

    def _on_alert(self, ev: alerts.AlertEvent) -> None:
        """Called on the scan thread. Must hand over to the GUI thread.

        The one thing a Qt front end must not do is touch widgets from another
        thread. A queued connection is the whole of the fix, and getting it
        wrong does not usually crash -- it corrupts the banner at random
        intervals, which is worse.
        """
        QTimer.singleShot(0, lambda: self._apply_alert(ev))

    def _apply_alert(self, ev: alerts.AlertEvent) -> None:
        self._last_event = ev
        repeats = self._repeats_for(ev.frequency_hz)
        self.banner.set_event(ev, repeats)
        self.log.addItem(f"{time.strftime('%H:%M:%S')}  {ev.headline()}")
        self.log.scrollToBottom()
        # Cap the visible list; an unbounded QListWidget in a long session is a
        # slow leak dressed up as a feature.
        while self.log.count() > 200:
            self.log.takeItem(0)
        self.alert_count.setText(self.alerts.summary())
        self.beeper.beep(ev.kind)

    def _repeats_for(self, freq_hz: int) -> int:
        for c in self.alerts.active:
            if abs(c.frequency_hz - freq_hz) <= 1_000_000:
                return self.alerts.repeat_count(c)
        return 1

    def _check_current(self) -> None:
        """Assess the channel the operator just chose and alert if it is live."""
        cand = self.scanner.assess_frequency(self.source.frequency_hz)
        if cand is None:
            return
        res = scan.ScanResult(
            lo_hz=self.source.frequency_hz, hi_hz=self.source.frequency_hz
        )
        res.candidates = [cand]
        events = self.alerts.update(res)
        self._last_result = res
        self._refresh_marks()
        if not events:
            self._apply_alert_silent(cand)

    def _apply_alert_silent(self, cand: scan.Candidate) -> None:
        self.banner.set_event(
            alerts.AlertEvent(
                kind=alerts.AlertKind.FOUND,
                at=time.time(),
                key=str(cand.frequency_hz),
                band=cand.band,
                channel=cand.channel,
                frequency_hz=cand.frequency_hz,
                signal=cand.reading.kind,
                detail=alerts._describe(cand),
                label=cand.match.label if cand.match else "off-chart",
            ),
            self.alerts.repeat_count(cand),
        )

    def _refresh_marks(self) -> None:
        marks = []
        for c in self.alerts.active:
            lo = c.frequency_hz - max(c.half_width_hz, 1_000_000)
            hi = c.frequency_hz + max(c.half_width_hz, 1_000_000)
            marks.append(
                _Mark(
                    lo, hi, f"{c.band} {c.frequency_hz}",
                    KIND_COLOURS.get(c.reading.kind, QColor(200, 200, 200)),
                    strong=c.decodable,
                )
            )
        self.chart.set_marks(marks)

    # -- periodic ----------------------------------------------------------

    def _on_paint(self) -> None:
        self.panel.set_frame(self.worker.latest())

    def _on_stats(self) -> None:
        st = self.worker.stats
        src = self.source.stats
        dropped = self.source.ring.dropped_bytes // 2
        elapsed = max(1e-6, time.monotonic() - st.started_at)
        rate = src.bytes_total / 2 / elapsed / 1e6
        lock = "locked" if st.locked_frames else "no lock"
        self._status.showMessage(
            f"{st.frames} frames  {st.frame_rate():4.1f} fps  "
            f"decode {st.decode_ms:4.1f} ms  cycle {st.cycle_ms:4.1f} ms  "
            f"{lock}  sync {st.sync_quality*100:3.0f}%  "
            f"source {rate:4.2f} MS/s"
            + (f"  dropped {dropped/1e6:.1f} Ms" if dropped else "")
        )

    # -- shutdown ----------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._scan_thread and self._scan_thread.isRunning():
            self._scan_thread.cancel()
            self._scan_thread.wait(3000)
        self._paint.stop()
        self._stats.stop()
        self.worker.stop()
        self.source.stop()
        super().closeEvent(event)


def run(source: IQSource, args) -> int:
    """Start the source and run the Qt event loop. Returns the exit code."""
    app = QApplication.instance() or QApplication([])
    source.start()
    win = MainWindow(source, args)
    win.show()
    if getattr(args, "autoscan", True):
        QTimer.singleShot(300, win._on_scan)
    return app.exec()
