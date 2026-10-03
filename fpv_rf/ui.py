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

from . import acquisition, alerts, bands, dsp, fastscan, scan
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

    #: Width / height of the picture as it is *displayed*.
    #:
    #: This is the shape of analogue video, and it is deliberately not derived
    #: from the frame. A decoded frame is 240 lines by ``--width`` samples, so at
    #: the default 160 that is 2:3 and at 320 it is 3:4 -- neither of which is
    #: 4:3. Scaling the image by its own aspect therefore painted a 2:3 picture,
    #: made ``--width`` a layout control, and put the geometry of the display
    #: under the control of a decode-resolution flag. The aspect belongs to the
    #: signal; the sample count is an implementation detail of the rasteriser.
    DISPLAY_ASPECT = 4.0 / 3.0

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

        # Letterbox to 4:3, which is the picture's shape and not the shape of
        # the samples it was decoded into.
        #
        # The image is one raster of 240 lines by whatever ``--width`` says, so
        # it is 2:3 at the default 160 and 3:4 at 320. Fitting *that* aspect
        # into the widget made the picture a different shape for every width,
        # and none of them were 4:3 -- so the app was simultaneously distorting
        # the picture and letting a decode-resolution flag change the layout.
        # The sample aspect is a sampling artifact; the display aspect is a
        # property of analogue video, and the widget is fitted to that instead.
        avail = self.rect()
        aspect = self.DISPLAY_ASPECT
        if avail.width() / max(1.0, avail.height()) > aspect:
            h = float(avail.height())
            w = h * aspect
        else:
            w = float(avail.width())
            h = w / aspect
        scale = min(w / img.width(), h / img.height())
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
        #: The region whose legality rules apply to this chart. Kept alongside
        #: the span because the span is *widened* to cover the band plans, so it
        #: cannot be used to recover which region was chosen -- and the tooltip
        #: has to answer with the operator's rules, not the US ones.
        self.region: bands.Region = bands.Region.US
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

    def set_region(self, region: bands.Region) -> None:
        """Record the region in force, for the tooltip's legality answer.

        Separate from :meth:`set_span` because the two answer different
        questions. The span is where the axis is drawn, and it is deliberately
        wider than the region. The region is whose channel list is law, and the
        tooltip is where that is applied.
        """
        self.region = bands.Region(region)
        self.update()

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
            # The region the operator actually selected. ``is_legal`` defaults
            # to US, and that default is wrong for every operator outside it: a
            # CE, AU or JP user hovering a channel was told "legal here" or
            # "outside this region" about the US band list while the combo box
            # above said a different country.
            legal = bands.is_legal(f, self.region)
            self.setToolTip(
                f"{band.label} channel {hit[1]+1}\n"
                f"{bands.format_mhz(f)}  ({f/1e6:.0f} MHz)\n"
                f"{'legal here' if legal else 'outside this region'}"
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
                f"ANALOGUE VIDEO DETECTED   {ev.label} @ "
                f"{bands.format_mhz(ev.frequency_hz)}"
            )
            self._head.setStyleSheet(f"color: {colour.name()};")
            self.setStyleSheet("background: #1d2a22;")
        extra = f"  (seen {repeats}x)" if repeats > 1 else ""
        self._sub.setText(ev.detail + extra)


# --------------------------------------------------------------------------
# Scan thread
# --------------------------------------------------------------------------


#: Search threads that were still running when the window gave up waiting for
#: one. Held at module scope so the wrapper object outlives the run: a QThread
#: deleted while it is still executing is a crash, and the operator closing the
#: app is not a reason to hand them one.
_ORPHANED_SCAN_THREADS: list["ScanThread"] = []


class ScanThread(QThread):
    """Runs one band search off the GUI thread.

    The tuner settles for hundreds of milliseconds per hop and a real sweep is
    tens of them, so this is seconds of work. Doing it on the GUI thread would
    stop the repaints, and with them the picture -- which is the thing the
    operator is looking at while they wait.

    Which engine does the work is :mod:`fpv_rf.fastscan`'s decision, not this
    class's: the sweep tool is about a hundred times faster where it exists and
    is the only way to get a whole-band search into a fraction of a second, and
    the hop walk is the fallback. Both return the same type, so everything
    downstream of here is identical either way.
    """

    progress = Signal(str, float)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, source: IQSource, lo_hz: int, hi_hz: int,
                 hop_frac: float, sweeps: int = 1,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.source = source
        self.lo_hz, self.hi_hz = int(lo_hz), int(hi_hz)
        self.hop_frac = float(hop_frac)
        self.sweeps = max(1, int(sweeps))
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:  # noqa: D102
        try:
            res = acquisition.acquire(
                self.source, self.lo_hz, self.hi_hz,
                cancel=self._cancel,
                frequencies=getattr(self, "frequencies", None),
                auto_gain=getattr(self, "auto_gain", True),
                progress=lambda m, f: self.progress.emit(m, f),
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        if (res.error and not res.candidates
                and (res.engine in ("sweep", "none") or res.incomplete)):
            # A sweep that could not run is a failure, not an empty band. Saying
            # "no confirmed analogue video" about a scan that never happened
            # is the one answer that must never come out of this.
            #
            # "none" is the engine fastscan._acquisition_failure() stamps on a
            # result meaning the band was never measured -- the radio did not
            # come back after the sweep, or it reopened and delivered nothing.
            # It was missing from this test, so that result fell through to
            # finished_ok and the UI reported a dead radio as a scan that
            # completed and found nothing.
            #
            # ``incomplete`` covers the hop walk, which this condition used to
            # miss entirely. A hop whose retune was refused or timed out returns
            # no samples, and the scanner used to record that as a hop that saw
            # nothing -- so a radio that gave up on the tenth hop reported a
            # confident scan of a band it had measured a tenth of, and "nothing
            # above the noise floor" across the other thirty hops meant nothing
            # at all. A result that could not measure part of the span is a
            # failed search, and this is where it becomes one.
            self.failed.emit(res.error)
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

        #: The channel the operator was on when the last search started, so it
        #: can be restored if the search does not end up moving the receiver
        #: somewhere new. Zero means "nothing to restore".
        self._pre_scan_hz = int(source.frequency_hz)

        #: Set once the window starts closing. Suppresses the search-failure
        #: dialog on the way out, where a modal box blocks the shutdown that
        #: would have reported it and there is no operator left to read it.
        self._closing = False
        self.monitor = acquisition.Monitor(enabled=not getattr(
            args, "no_autoscan", not getattr(args, "autoscan", True)))
        #: Prior enabled states of the tuning controls while a search runs.
        self._tuning_was: dict = {}
        #: When the close must stop waiting, as a monotonic deadline.
        self._close_deadline = 0.0

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
        self.chart.set_region(self.args.region)
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
        if not self.source.retunable:
            # A recording is one frequency. Offering a tuner that cannot tune is
            # worse than offering none: dragging it used to relabel the capture
            # with a channel it was never recorded on, and the source now refuses
            # that, so the control would do nothing but raise an error. Disabled,
            # with the reason on the tooltip.
            self.freq.setEnabled(False)
            self.freq.setToolTip(
                f"This source is a recorded capture, fixed at "
                f"{bands.format_mhz(self.source.frequency_hz)}. "
                f"Its samples cannot be retuned."
            )
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
            "one tuner settling period per hop.\n\n"
            "This only applies to the hop-by-hop engine. The sweep engine\n"
            "always resolves to 1 MHz bins and ignores this."
        )
        self.hop.hide()  # retained for diagnostic API compatibility

        self.sweeps = QComboBox()
        self.sweeps.addItem("1 pass", 1)
        self.sweeps.addItem("4 passes", 4)
        self.sweeps.addItem("10 passes", 10)
        self.sweeps.setCurrentIndex(0)
        self.sweeps.setToolTip(
            "How many times to sweep the band and keep the loudest reading.\n\n"
            "One pass is right for FPV video, which transmits continuously and\n"
            "is therefore caught every time -- a whole-band pass takes about a\n"
            "fifth of a second, so a second pass is only worth having for a\n"
            "transmitter that comes and goes, such as a beaconing access point.\n\n"
            "More passes never find a weaker signal, only one that was not\n"
            "transmitting at the moment. The strongest reading is kept, not the\n"
            "average, so a pass that caught a burst is not diluted by the quiet\n"
            "ones around it."
        )
        self.sweeps.hide()  # spectrum passes do not apply to video acquisition

        self.scan_btn = QPushButton("Scan band")
        self.scan_btn.setDefault(True)
        self.scan_btn.setToolTip(
            "Check channel presets and stop on independently verified analogue video."
        )
        self.scan_btn.clicked.connect(self._on_scan)
        gv.addWidget(self.scan_btn, 3, 0, 1, 3)
        self.auto_btn = QPushButton("Auto-select")
        self.auto_btn.setToolTip(
            "Tune the strongest decodable channel from the last search and "
            "watch it.\n\nOn a band with nothing on it this deliberately does "
            "nothing: a search that always names a channel teaches you to "
            "ignore it."
        )
        self.auto_btn.clicked.connect(self._on_auto)
        gv.addWidget(self.auto_btn, 4, 0, 1, 2)
        self.cancel_btn = QPushButton("Stop")
        self.cancel_btn.setToolTip("Stop the running search. A stopped search "
                                   "reports nothing, rather than reporting that "
                                   "everything it did not reach has gone.")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        gv.addWidget(self.cancel_btn, 4, 2)

        self.autotune = QCheckBox("tune verified video automatically")
        self.autotune.setChecked(True)
        self.autotune.setToolTip(
            "When a search finds something, tune it and show it, instead of\n"
            "making you press Auto-select afterwards.\n\n"
            "Still declines when there is nothing above the noise floor, and\n"
            "when the active band has nothing on it. Finding a signal and\n"
            "watching it are one action, not two."
        )
        gv.addWidget(self.autotune, 5, 0, 1, 3)
        self.continuous = QCheckBox("keep searching / reacquire lost video")
        self.continuous.setChecked(self.monitor.enabled)
        self.continuous.toggled.connect(self._monitor_changed)
        gv.addWidget(self.continuous, 6, 0, 1, 3)
        self.auto_gain = QCheckBox("automatic receiver gain")
        self.auto_gain.setChecked(True)
        gv.addWidget(self.auto_gain, 7, 0, 1, 3)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        gv.addWidget(self.progress, 8, 0, 1, 3)
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
        # Connected through an explicit keyword, not straight to set_controls.
        # ``toggled`` emits a bool, and set_controls' first positional parameter
        # is *brightness*, so connecting it directly made ticking this box set
        # brightness to 1 and leave invert False. The two sliders above are
        # already written the safe way; this one was not, and it is invisible
        # unless you actually click the checkbox rather than call the setter.
        self.invert.toggled.connect(
            lambda v_: self.panel.set_controls(invert=bool(v_))
        )
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
        self.chart.set_region(region)
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

        ``reconfigure=True`` is what makes that cost actually happen. Toggling
        the switch while already tuned to this frequency used to call
        ``_tune(frequency_hz)``, which short-circuits on an unchanged frequency
        and never restarts the process -- so the checkbox moved and the
        amplifier carried on amplifying, with no indication that anything had
        failed to happen.
        """
        if not on:
            self._lna_saved_db = self.lna.value()
        self.source.amp_enabled = bool(on)
        if not self._reconfigure():
            self.amp.blockSignals(True)
            self.amp.setChecked(self.source.amp_enabled)
            self.amp.blockSignals(False)
            self._status.showMessage("the amplifier switch did not reach the radio")

    def _on_gain(self, _v: int) -> None:
        if not self.source.set_gains(self.lna.value(), self.vga.value()):
            self._status.showMessage("the gain change did not reach the radio")

    def _reconfigure(self) -> bool:
        """Push the current settings to the hardware without moving frequency.

        The single place that answers "how do changed settings reach the radio",
        so the gain sliders and the amplifier switch cannot drift apart again.
        """
        return self.source.tune(self.source.frequency_hz, reconfigure=True)

    def _tune(self, freq_hz: int, ask_alert: bool = False,
              clear_alerts: bool = True) -> bool:
        """Move the receiver. Returns True only if the hardware actually moved.

        The return value is load-bearing. Callers used to ignore it, so
        auto-select could report "watching F4 @ 5800 MHz" after a tune that had
        failed and left the radio exactly where it was -- a success message
        about a channel the app was not listening to.
        """
        if not self.source.tune(freq_hz):
            # Say where the receiver actually is, not just that the move failed.
            # The two frequencies being different is the whole diagnosis, and
            # leaving it to the operator to work out from the fact that the
            # channel they asked for is not the one the box is still showing is
            # a step too many.
            have = self.source.applied_frequency_hz
            where = (f"\n\nThe receiver is still on {bands.format_mhz(have)}."
                     if have else "")
            QMessageBox.warning(
                self, "Tune failed",
                f"Could not tune to {bands.format_mhz(freq_hz)}.{where}\n"
                f"{self.source.stats.last_error or 'the source reported no reason.'}",
            )
            return False
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
            self.monitor.completed(time.monotonic())
            self.alerts.clear()
        # The applied frequency, not the requested one. They are normally the
        # same, and when they are not the marker is claiming the panel is
        # watching a frequency the radio is not on -- the same claim the tune
        # failure above refuses to make.
        self.chart.set_tune(self.source.applied_frequency_hz)
        if ask_alert:
            self._check_current()
        return True

    #: Every control that changes what the radio is receiving, whether by
    #: moving it, by reconfiguring it, or by asking for the last result to be
    #: applied. Disabled for the duration of a search.
    #:
    #: A search drives the *same* source, hop by hop. A frequency change, a gain
    #: change or an auto-select arriving in the middle of one is not a conflict
    #: the app reports: the tune takes the tuning lock, the search's next hop
    #: takes it back, and both proceed. The operator gets a scan of a band that
    #: includes a frequency they chose mid-sweep, with a picture on screen from
    #: a hop inside the sweep -- and no indication that anything overlapped.
    #: Reconfiguring the receiver is worse, because a spawn during a sweep is
    #: the handover race the sweep path goes to such lengths to avoid.
    TUNING_CONTROLS = ("freq", "lna", "vga", "amp", "auto_btn", "chart", "region", "auto_gain")

    def _set_scan_controls(self, running: bool) -> None:
        """Enable or disable everything that moves the receiver.

        The prior states are remembered, not assumed, because several of these
        are already disabled for reasons of their own -- a recording cannot be
        retuned, this build of the transfer helper has no ``-a`` flag -- and
        blindly re-enabling them when the search ends would hand the operator a
        control that does nothing, which is a smaller lie than a greyed one but
        still a lie.
        """
        controls = [getattr(self, name) for name in self.TUNING_CONTROLS]
        if running:
            self._tuning_was = {w: w.isEnabled() for w in controls}
            for w in controls:
                w.setEnabled(False)
            return
        for w, was in getattr(self, "_tuning_was", {}).items():
            w.setEnabled(was)
        self._tuning_was = {}

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
        verified = scan.ScanResult(self._last_result.lo_hz, self._last_result.hi_hz,
            candidates=[c for c in self._last_result.candidates
                        if acquisition.video_present(c.reading)])
        best = self.scanner.suggest(verified, self._active_band)
        if best is None:
            if self._active_band is not None and self._last_result.decodable:
                self._status.showMessage(
                    f"nothing decodable on {self._active_band.label} -- "
                    f"cleared to search every band"
                )
            else:
                self.banner.set_idle("no confirmed analogue video")
                if announce:
                    self._status.showMessage(
                        "no channel found -- auto-select deliberately declines "
                        "to guess, so an empty band stays empty"
                    )
            return False
        self.freq.blockSignals(True)
        self.freq.setValue(int(round(best.frequency_hz / 1e6)))
        self.freq.blockSignals(False)
        if not self._tune(best.frequency_hz, ask_alert=False, clear_alerts=False):
            # _tune has already told the operator why. Returning True here would
            # overwrite that with a confident "watching <channel>" for a channel
            # the receiver never moved to, and the caller would carry on as
            # though it had.
            self._status.showMessage("auto-select failed: the radio did not move")
            return False
        self._status.showMessage(f"watching {best.label()}")
        return True

    def _monitor_changed(self, enabled: bool) -> None:
        self.monitor.enabled = enabled
        self.monitor.next_scan = time.monotonic() + self.monitor.retry_s

    def _on_cancel(self) -> None:
        self.continuous.setChecked(False)
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
        # Remember where the operator was, so a search that does not move the
        # receiver anywhere can put it back. The hop walk drives the shared
        # source and leaves it on the last hop it visited.
        self._pre_scan_hz = self.source.frequency_hz
        self.progress.setValue(0)
        self.scan_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        # Everything that moves the receiver goes dead for the duration. The
        # search and the operator were both driving one source, and neither side
        # knew about the other.
        self._set_scan_controls(True)
        # Deliberately *not* alerts.clear() here. The whole point of the
        # cooldown and the seen-count is that the second sweep over the same
        # drone says "already known" instead of raising an identical alert; the
        # history has to survive the sweep that follows the one that found it.
        th = ScanThread(self.source, lo_hz, hi_hz, self.hop.currentData(),
                        sweeps=self.sweeps.currentData(), parent=self)
        th.frequencies = (list(self._active_band.channels)
                          if self._active_band is not None else None)
        th.auto_gain = self.auto_gain.isChecked()
        th.progress.connect(self._on_progress)
        th.finished_ok.connect(self._on_scan_done)
        th.failed.connect(self._on_scan_failed)
        self._scan_thread = th
        th.finished.connect(lambda: self._retire_scan(th))
        th.start()
        self._status.showMessage("Searching for analogue video")

    def _retire_scan(self, thread) -> None:
        if self._scan_thread is thread:
            self._scan_thread = None
        thread.deleteLater()

    def _on_progress(self, message: str, frac: float) -> None:
        self.progress.setValue(int(frac * 1000))

    def _on_scan_done(self, res: scan.ScanResult) -> None:
        self._finalise_scan(res)

    def _on_scan_failed(self, message: str) -> None:
        self._finalise_scan(None, failure=message)

    def _finalise_scan(self, res: scan.ScanResult | None,
                       failure: str = "") -> None:
        """Everything that happens when a search stops, however it stopped.

        One function because the three ways a search ends -- found something,
        cancelled, or failed outright -- have to leave the app in the same state,
        and they used to be two functions that did not. ``_on_scan_failed``
        re-enabled the buttons, showed a dialog and stopped: it never restored
        the tuning, so a search that failed halfway through the hop walk left
        the radio on whichever hop it died on, with the frequency box still
        reading the operator's channel and the picture panel showing a frequency
        nobody chose. That is the one case where the operator most needs their
        channel back, and it was the one that lost it. The docstring on
        :meth:`_restore_pre_scan_tuning` claimed it covered "a search that could
        not run at all"; nothing called it from there.

        ``res`` is None for a search that produced no result at all. A failure
        still has to be announced -- silently returning to the previous channel
        would leave the operator thinking the search simply found nothing.
        """
        self.monitor.completed(time.monotonic())
        for control, value in ((self.lna, self.source.lna_gain_db),
                               (self.vga, self.source.vga_gain_db)):
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        self.scan_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._set_scan_controls(False)
        if res is not None:
            self.progress.setValue(1000)
            # Only verified video is an acquisition. Spectrum-only findings
            # remain available in the diagnostic scanner, never as FPV beeps.
            res.candidates = [c for c in res.candidates if acquisition.video_present(c.reading)]
            self._last_result = res
            events = self.alerts.update(res)
            self._refresh_marks()
            self.alert_count.setText(self.alerts.summary())
            self._status.showMessage(res.describe())
            if not events:
                self.banner.set_idle(
                    "no confirmed analogue video"
                    if not res.candidates
                    else f"already known: {self.alerts.active[0].label()}"
                    if self.alerts.active
                    else "idle"
                )
        elif failure:
            if not self._closing and not self.monitor.enabled:
                # Suppressed while closing: a modal dialog on a window that is
                # on its way out blocks the shutdown it was supposed to report
                # on, and there is no operator left to read it.
                QMessageBox.warning(self, "Band search failed", failure)
            self._status.showMessage(f"band search failed: {failure}")
        # Put the receiver back where the operator left it unless the search
        # moved it to something new.
        #
        # A hop walk retunes the shared source for every hop, so when the search
        # finishes the radio is sitting on whichever hop came last -- typically the
        # top of the band. Previously the only thing that could move it was
        # auto-select, and only when it succeeded. So the two obvious cases both
        # left the receiver somewhere the operator did not choose: auto-select
        # unticked, or a search that found nothing. The frequency box still read
        # the original channel, so the app claimed to be watching a channel it was
        # not receiving, and said "no confirmed analogue video" while frozen on
        # an arbitrary frequency.
        if res is not None and self.autotune.isChecked():
            # After the alert bookkeeping, not before: _tune() clears the
            # alert engine, and doing that first would make a brand new finding
            # look like it had never been raised.
            if self._select_best(announce=False):
                self._pre_scan_hz = self.source.frequency_hz
                self.monitor.acquired(time.monotonic())
                return
        self._restore_pre_scan_tuning(res)

    def _on_alert(self, ev: alerts.AlertEvent) -> None:
        """Called on the scan thread. Must hand over to the GUI thread.

        The one thing a Qt front end must not do is touch widgets from another
        thread. A queued connection is the whole of the fix, and getting it
        wrong does not usually crash -- it corrupts the banner at random
        intervals, which is worse.
        """
        QTimer.singleShot(0, lambda: self._apply_alert(ev))

    def _apply_alert(self, ev: alerts.AlertEvent) -> None:
        if ev.kind is alerts.AlertKind.FOUND and ev.signal is not dsp.SignalKind.VIDEO:
            return
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

    def _restore_pre_scan_tuning(self, res: scan.ScanResult | None) -> None:
        """Return the receiver to the channel the operator was on.

        Covers every path where a search finished without choosing a channel:
        auto-select unticked, nothing found, a cancelled search, and a search
        that could not run at all. All four used to leave the radio on the last
        hop, and only the first two were even reachable before -- this is called
        from the failure path too, which is the case that most needs it.

        ``res`` is None when the search failed before producing a result, so the
        "why" it reports is a failure rather than a cancellation.

        The restore does *not* clear the alert engine, and that is deliberate.
        This move is not the operator choosing a channel -- it is this window
        undoing a side effect the search caused, returning the receiver to where
        it already was. Clearing here threw away what the search had just
        learned: with auto-select off, the findings were recorded, the engine
        was emptied on the way back to the original channel, and the next
        identical sweep raised the same alert again -- inside the cooldown the
        cooldown was supposed to enforce, because the memory of it had been
        destroyed a moment earlier. The same reasoning the scan path uses at
        :meth:`_tune`'s ``clear_alerts`` applies here, in the same direction.

        Nothing is done when the receiver is already where it should be. That is
        not just an optimisation: the fast sweep engine never moves the radio at
        all, so this is the path every sweep takes, and clearing the decoder
        there would blank a perfectly good picture after every search.
        """
        want = self._pre_scan_hz
        if not want or want == self.source.frequency_hz:
            return
        if not self.source.retunable:
            # A file or simulator cannot move, and pretending otherwise would
            # fail. Nothing to restore: it never left.
            return
        if self._tune(want, ask_alert=False, clear_alerts=False):
            if res is None:
                why = "search could not run"
            elif res.stopped_early:
                why = "search cancelled"
            elif res.error:
                why = "search could not run"
            else:
                why = "nothing found"
            self._status.showMessage(
                f"{why} -- back on {bands.format_mhz(want)}"
            )
        else:
            # The restore itself failed, so the radio is not on the operator's
            # channel and the panel is showing whatever the search left it on.
            # Drop the picture rather than leave a frame from a frequency nobody
            # is listening to under a status line that says otherwise. _tune has
            # already told the operator why the move failed.
            self.source.reset_drain()
            self.worker.reset()

    def _check_current(self) -> None:
        """Assess the channel the operator just chose and alert if it is live.

        A quiet channel gets a report, not an alert. This used to hand every
        assessment to the alert engine, and since an assessment always yields a
        candidate -- including one classified NOISE -- asking about an empty
        channel produced a FOUND event, a banner and an audible beep for a
        channel with nothing on it. An alert that fires on silence trains the
        operator to ignore it, which costs the one property an alert exists to
        have.

        ``snap=False`` because this is an *explicit* choice by the operator. The
        default snaps a frequency onto the nearest band-plan channel, which is
        right for a search reporting a peak it located itself and wrong here:
        type 5802, get measured at 5800 and told so. That is a measurement of a
        different frequency than the one on the display and in the status line,
        and a 2 MHz error is enough to be confident and wrong about a channel
        nobody is on. Verbatim is also what :meth:`BandScanner.assess_frequency`
        documents for a hand-picked frequency.
        """
        cand = acquisition.confirm(self.source, self.source.frequency_hz)
        if cand is None:
            self._last_result = scan.ScanResult(self.source.frequency_hz, self.source.frequency_hz)
            self._status.showMessage("No confirmed analogue video at this frequency")
            self.banner.set_idle("no confirmed analogue video")
            return
        res = scan.ScanResult(
            lo_hz=self.source.frequency_hz, hi_hz=self.source.frequency_hz
        )
        if acquisition.video_present(cand.reading):
            res.candidates = [cand]
            events = self.alerts.update(res)
        else:
            # Nothing there. Record the result so the chart and the band display
            # stay consistent with what was just measured, but keep it out of the
            # alert engine entirely.
            events = []
            self._status.showMessage(
                f"{bands.format_mhz(cand.frequency_hz)}: nothing above the "
                f"noise floor ({cand.reading.peak_to_median_db:+.1f} dB)"
            )
        self._last_result = res
        self._refresh_marks()
        if not events and cand.reading.occupied:
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
        now = time.monotonic()
        busy = bool(self._scan_thread and self._scan_thread.isRunning())
        locked = self.worker.locked and self.worker.stats.sync_quality >= .60
        if not self._closing and self.source.retunable and self.monitor.due(now, locked, busy):
            if self.monitor.tracking:
                self.alerts.lost(self.source.frequency_hz)
            self.monitor.completed(now)
            self._on_scan()
            return
        st = self.worker.stats
        src = self.source.stats
        dropped = self.source.ring.dropped_bytes // 2
        elapsed = max(1e-6, time.monotonic() - st.started_at)
        rate = src.bytes_total / 2 / elapsed / 1e6
        # Judged on how recently a frame was published, not on whether one has
        # ever been published. ``locked_frames`` is cumulative, so it could only
        # ever say "locked" afterwards: stop the receiver, lose the transmitter,
        # and the status line kept claiming a lock it no longer had.
        lock = "locked" if self.worker.locked else "no lock"
        if self.worker.stats.decode_errors:
            lock = f"decode errors x{self.worker.stats.decode_errors}"
        # The sync figure is only meaningful while locked, and printing it
        # unconditionally is what concealed a real decoder fault. On a genuine
        # 5802 MHz capture the sync detector reported 90% for as long as it was
        # fed noise or a signal whose lines it could not assemble, while not one
        # frame was ever produced -- so the status line read "no lock  sync 90%"
        # and a user had no way to tell that from a healthy-but-noisy lock. The
        # number now appears only when there is a lock to justify it.
        sync = f"  sync {st.sync_quality*100:3.0f}%" if self.worker.locked else ""
        # Likewise, a field that needed most of its line positions reconstructed
        # is a weaker result than a fully detected one, and the difference is
        # visible rather than absorbed.
        grid = (f"  {st.lines_from_grid}/240 lines from grid"
                if st.lines_from_grid else "")
        # Asked for, and actually receiving. These are the same number almost
        # always, and the exception is the whole reason to print both: a source
        # that has lent its radio to another process, or whose last retune was
        # refused, leaves the requested frequency and the applied one apart, and
        # every other figure on this line -- frame rate, sync, MS/s -- describes
        # what is arriving, not what was asked for. Without this the status bar
        # would describe a healthy picture of a channel the radio is not on.
        want = self.source.frequency_hz
        have = self.source.applied_frequency_hz
        asked = (f"  asked {bands.format_mhz(want)}, receiving "
                 f"{bands.format_mhz(have)}" if want and have and want != have
                 else "")
        self._status.showMessage(
            f"{st.frames} frames  {st.frame_rate():4.1f} fps  "
            f"decode {st.decode_ms:4.1f} ms  cycle {st.cycle_ms:4.1f} ms  "
            f"{lock}{sync}{grid}  "
            f"source {rate:4.2f} MS/s"
            + (f"  dropped {dropped/1e6:.1f} Ms" if dropped else "")
            + asked
        )

    # -- shutdown ----------------------------------------------------------

    #: How long a closing window waits for a running search to stop, and how
    #: often it re-checks while waiting.
    #:
    #: 10 s because reclaiming the radio after a sweep is a real USB reopen and
    #: the handover wedge can take a while. A search that has not finished by
    #: then is not going to, and the window must still close: an app that cannot
    #: be closed is worse than a search that was abandoned.
    CLOSE_WAIT_S = 10.0
    CLOSE_POLL_MS = 200

    def closeEvent(self, event) -> None:  # noqa: N802
        """Cancel the search, wait for it from the event loop, then tear down.

        The wait is a poll, not ``QThread.wait()``. A blocking wait on the GUI
        thread spins a modal event loop to stay responsive, which means the scan
        thread's ``finished_ok`` signal is delivered *during* the close -- and
        the result handler touches the banner, the chart and the status line of
        a window that is halfway through being dismantled. That was a 3 s wait
        with the result discarded, so any search still running when the operator
        closed the window (typically stuck reclaiming a slow radio) had its
        completion delivered into a dying window.

        With a poll the wait happens in the event loop without re-entering it, so
        nothing is delivered until the thread has finished, and the teardown
        runs exactly once, from the GUI thread, at a point where the app is
        still whole. If the search overruns its budget the thread is detached
        rather than destroyed: deleting a running QThread underneath itself is a
        crash, and refusing to close is worse.
        """
        self.monitor.enabled = False
        if self._scan_thread and self._scan_thread.isRunning():
            self._closing = True
            self._scan_thread.cancel()
            self._close_deadline = time.monotonic() + self.CLOSE_WAIT_S
            self._status.showMessage("closing: waiting for the search to stop")
            event.ignore()
            QTimer.singleShot(self.CLOSE_POLL_MS, self._poll_for_close)
            return
        self._teardown()
        super().closeEvent(event)

    def _poll_for_close(self) -> None:
        """One step of the close: has the search stopped, or is it out of time?"""
        th = self._scan_thread
        if th is not None and th.isRunning():
            left = self._close_deadline - time.monotonic()
            if left > 0.0:
                self._status.showMessage(
                    f"closing: waiting for the search to stop ({left:.0f} s left)"
                )
                QTimer.singleShot(self.CLOSE_POLL_MS, self._poll_for_close)
                return
            # Over budget. Detach: the QThread object must outlive the run, and
            # it must not deliver anything to a window that has gone. The
            # signals are disconnected for the same reason, and the module-level
            # list keeps a reference so the wrapper is not collected while the
            # thread is still inside it.
            self._status.showMessage("closing: the search did not stop in time")
            for sig in (th.progress, th.failed, th.finished_ok):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            th.setParent(None)
            _ORPHANED_SCAN_THREADS.append(th)
        self._teardown()
        self.close()

    def _teardown(self) -> None:
        """Stop the periodic work, then the decoder, then the radio.

        In that order, and once. Called from :meth:`closeEvent` on the way to a
        clean close and from :meth:`_poll_for_close` when the window is closing
        while a search runs; every step is idempotent, so being called twice
        costs nothing.
        """
        self._paint.stop()
        self._stats.stop()
        self.worker.stop()
        self.source.stop()


def run(source: IQSource, args) -> int:
    """Start the source and run the Qt event loop. Returns the exit code."""
    app = QApplication.instance() or QApplication([])
    source.start()
    win = MainWindow(source, args)
    win.show()
    # ``--no-autoscan`` is a store_true flag, so argparse spells it
    # ``no_autoscan`` and there is no ``autoscan`` attribute to read. Reading
    # one for the other is how the flag became dead: the default of ``True``
    # always won, so passing it did nothing and the app searched the band on
    # launch anyway. The nested getattr keeps the test ``Args`` class working --
    # it declares the positive form, which is what a harness wants to say
    # "do not scan" in one attribute.
    if not getattr(args, "no_autoscan", not getattr(args, "autoscan", True)):
        QTimer.singleShot(300, win._on_scan)
    return app.exec()
