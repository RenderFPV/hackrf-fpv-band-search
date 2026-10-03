"""Regression for the Rasteriser's usable-line bound (dsp.Rasteriser.push).

The bug: ``push`` decided which sync detections were renderable by comparing
``start + (offset + span)`` against ``size - (offset + span)`` -- two active
windows of padding where one is needed. A field sized so its last row's window
ends *exactly* on the last sample therefore lost that row, and with 239 of
NTSC's 240 lines left there is no 240-line run for ``select_run`` to find, so
every perfectly-formed field was refused with ``run_span``. Same for PAL's 288.

The cases below are built from the render's own addressing -- ``int64`` start
plus ``int(round(offset))``, ``int(round(span))`` samples wide -- because the
other half of the defect is that the bound used *floats*. A float end of a row
landing a fraction past an integer sample count throws the row away just as
surely as the doubled window did, so a bound that is correct only for a whole
number of samples per line still fails on a measured 640.37.

What is pinned:

  1. An exactly-fitting NTSC 240-row and PAL 288-row field both decode.
  2. One sample short of that, the *final* row and only the final row is lost,
     and the field is refused -- an incomplete row must never be rendered.
  3. A fractional line length agrees with ``_render``'s rounding.
  4. No row is gathered from a negative or out-of-range index, which is what a
     negative ``crop_start`` would otherwise cause: a negative NumPy index does
     not raise, it wraps round to the far end of the buffer.
  5. ``first_sample``/``last_sample`` name the samples the picture was *gathered
     from* -- detections plus the crop offset -- rather than the sync detections
     themselves, and they are read off the block the renderer actually built
     rather than off the same arithmetic twice.

Run:  python tools/check_raster_window.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv_rf import diag, dsp  # noqa: E402

FAILURES: list[str] = []

#: Sample rate the synthetic line lengths are quoted against. Only the ratio
#: matters (samples per line is what the rasteriser sees), but a plausible one
#: keeps ``line_rate_hz`` sane for anyone reading the output.
FS = 10_000_000.0


def check(name: str, good: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if good else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not good:
        FAILURES.append(name)
    return bool(good)


def field(spec: dsp.VideoSpec, line: float, rows: int | None = None):
    """A ramp ``demod`` buffer and a :class:`SyncResult` whose rows fit exactly.

    The buffer is a ramp rather than noise so that which samples a row was
    gathered from is readable back out of the rendered block: a wrapped
    negative index shows up as a value from the wrong end of the buffer.

    The size is ``_render``'s own requirement for the last row and nothing
    more, derived exactly as ``_render`` derives it. ``rows`` overrides the
    number of detections, so a case can hand the rasteriser one more line than
    the field needs.
    """
    rows = int(spec.visible_lines if rows is None else rows)
    starts = np.arange(rows, dtype=np.float64) * float(line)
    off_i = int(round(line * spec.crop_start))
    n_src = int(round(line * spec.crop_width))
    exact = int(starts[-1]) + off_i + n_src
    demod = np.arange(exact, dtype=np.float32)
    sync = dsp.SyncResult(
        pulse_starts=starts,
        pulse_centres=starts + 4.0,
        line_samples=float(line),
        line_rate_hz=FS / float(line),
        polarity=1,
        threshold=0.0,
        quality=1.0,
        spec=spec,
    )
    return demod, sync, rows, off_i, n_src


def push(demod: np.ndarray, sync: dsp.SyncResult, origin: int = -1):
    """Push one buffer and report the outcome, its reject reason and usable count.

    The gauges are reset first because they belong to the ``raster`` section
    rather than to a Rasteriser, so without this a case that rejects before
    stamping them reports whatever the last one left behind.

    ``origin`` is the absolute source index of ``demod[0]``; -1 is "the caller
    does not know" and is the default so the earlier cases are unchanged.
    """
    diag.reset()
    r = dsp.Rasteriser(width=160)
    fr = r.push(demod, sync, origin=origin)
    return fr, r._diag.gauge_value("reject"), int(r._diag.gauge_value("usable") or 0)


def case_exact_fit() -> None:
    print("\n1. a field whose last row ends on the last sample decodes")
    for tag, spec, line in (("NTSC", dsp.NTSC, 640.0), ("PAL", dsp.PAL, 640.0)):
        demod, sync, rows, off_i, n_src = field(spec, line)
        fr, reject, usable = push(demod, sync)
        check(
            f"{tag}: {rows} rows fit in {demod.size} samples and all are rendered",
            fr is not None and fr.height == rows and fr.lines_used == rows
            and usable == rows and reject == "none",
            f"usable={usable}/{rows} shape={None if fr is None else fr.image.shape} "
            f"reject={reject}",
        )
        check(
            f"{tag}: nothing beyond the buffer was read",
            fr is not None and usable == rows,
            f"usable={usable} window=[{off_i}, {off_i + n_src})",
        )


def case_one_short() -> None:
    print("\n2. one sample short, the final row is lost and the field refused")
    for tag, spec, line in (("NTSC", dsp.NTSC, 640.0), ("PAL", dsp.PAL, 640.0)):
        demod, sync, rows, _, _ = field(spec, line)
        fr, reject, usable = push(demod[:-1], sync)
        check(
            f"{tag}: exactly one row fewer is not renderable",
            fr is None and usable == rows - 1,
            f"usable={usable}/{rows} reject={reject}",
        )
        check(
            f"{tag}: refused for want of a run, not for want of samples",
            reject == "run_span",
            f"reject={reject}",
        )
        # And the row that was dropped is the last one, not an arbitrary one:
        # the run spans all that is left, so the loss is at the fresh end.
        d: dict = {}
        starts = sync.pulse_starts
        dsp.Rasteriser.select_run(starts[:-1], line, rows, diag_out=d)
        check(
            f"{tag}: the loss is the freshest row",
            d.get("reason") == "span" and int(d.get("max_span", 0)) == rows - 1,
            f"reason={d.get('reason')} max_span={d.get('max_span')}",
        )


def case_fractional_line() -> None:
    print("\n3. a fractional line length agrees with _render's rounding")
    # 640.37 samples/line: the float end of the last row is 153650.378 while the
    # render's own integer indexing puts it at 153650, so a float bound loses the
    # row here and the integer one keeps it.
    line = 640.37
    demod, sync, rows, off_i, n_src = field(dsp.NTSC, line)
    float_end = float(sync.pulse_starts[-1]) + line * dsp.NTSC.crop_start + line * dsp.NTSC.crop_width
    fr, reject, usable = push(demod, sync)
    check(
        "NTSC at 640.37 samples/line: all rows render at the exact fit",
        fr is not None and usable == rows and reject == "none",
        f"usable={usable}/{rows} reject={reject} "
        f"float_end={float_end:.3f} render_end={demod.size}",
    )
    check(
        "the float end of the last row is past the exact fit, so a float bound"
        " would have dropped it",
        float_end > demod.size,
        f"float_end={float_end:.3f} > {demod.size}",
    )
    fr, reject, usable = push(demod[:-1], sync)
    check(
        "NTSC at 640.37 samples/line: one sample short still refuses",
        fr is None and usable == rows - 1 and reject == "run_span",
        f"usable={usable}/{rows} reject={reject}",
    )
    # A period whose crop offsets land on a different rounding, to catch a bound
    # hard-coded to one field's geometry rather than derived per call.
    line2 = 639.62
    demod2, sync2, rows2, _, _ = field(dsp.NTSC, line2)
    fr2, reject2, usable2 = push(demod2, sync2)
    check(
        "NTSC at 639.62 samples/line: all rows render at the exact fit",
        fr2 is not None and usable2 == rows2 and reject2 == "none",
        f"usable={usable2}/{rows2} reject={reject2}",
    )


def case_indexing() -> None:
    print("\n4. no row is gathered from a negative or out-of-range index")
    real = dsp._resample_2d
    seen: list[np.ndarray] = []

    def spy(block, out_h, width):
        seen.append(block.copy())
        return real(block, out_h, width)

    dsp._resample_2d = spy
    try:
        # A negative crop_start puts the window's first samples before the
        # buffer. The earliest row must be dropped, not wrapped round the end.
        spec = replace(dsp.NTSC, visible_lines=9, crop_start=-0.20, crop_width=0.50)
        line = 640.0
        rows = int(spec.visible_lines)
        # One row more than the field needs, so the freshest-run choice really
        # has to choose: the row at sample 0 is the incomplete one.
        demod, sync, detections, off_i, n_src = field(spec, line, rows=rows + 1)
        starts = sync.pulse_starts
        fr, reject, usable = push(demod, sync)
        check(
            "a row whose window starts before the buffer is not usable",
            fr is not None and usable == rows and usable == detections - 1,
            f"usable={usable} of {detections} detections, off_i={off_i}, "
            f"reject={reject}",
        )
        if seen:
            block = seen[-1]
            expected = int(starts[1]) + off_i
            check(
                "the rendered block came from non-negative, in-range indices",
                expected >= 0 and block.shape == (rows, n_src)
                and int(block[0, 0]) == int(demod[expected]),
                f"first gathered sample {int(block[0, 0])} == demod[{expected}]; "
                f"a negative index would have read demod[{demod.size + expected}]",
            )
        else:
            check("the rendered block was captured", False, "nothing was rendered")
    finally:
        dsp._resample_2d = real

    # Out of range at the top end: the exact fit reads the last sample and not
    # one past it, and a buffer one sample shorter renders nothing at all.
    demod, sync, rows, off_i, n_src = field(dsp.NTSC, 640.0)
    seen.clear()
    dsp._resample_2d = spy
    try:
        fr, reject, usable = push(demod, sync)
    finally:
        dsp._resample_2d = real
    if fr is not None and seen:
        first_row = int(sync.pulse_starts[-1]) + off_i
        block = seen[-1]
        check(
            "the last row's window ends on the last sample",
            block.shape == (rows, n_src) and first_row + n_src == demod.size
            and int(block[-1, -1]) == int(demod[-1]),
            f"window=[{first_row}, {first_row + n_src}) of {demod.size}",
        )
    else:
        check("the last row's window was rendered", False, "no block captured")

    # A buffer too short for even one active window is named, not rendered.
    fr, reject, usable = push(demod[: off_i + n_src - 1], sync)
    check(
        "a buffer short of one active window is refused as such",
        fr is None and reject == "buffer_short",
        f"reject={reject} usable={usable}",
    )


def case_provenance() -> None:
    print("\n5. frame provenance names the gathered samples, crop offset included")
    # The spy sees the block *before* resampling, so its values are the indices
    # that were read out of the ramp ``demod`` -- a direct measurement of what
    # the picture was made of, independent of the arithmetic push() uses to
    # describe it. If the two ever stop agreeing, one of them is lying.
    real = dsp._resample_2d
    seen: list[np.ndarray] = []

    def spy(block, out_h, width):
        seen.append(np.asarray(block, dtype=np.float64).copy())
        return real(block, out_h, width)

    origin = 1_234_567
    demod, sync, rows, off_i, n_src = field(dsp.NTSC, 640.0)
    check(
        "NTSC's crop offset is non-zero, so this case can see it",
        off_i > 0,
        f"crop_start={dsp.NTSC.crop_start} -> off_i={off_i} n_src={n_src}",
    )
    dsp._resample_2d = spy
    try:
        fr, reject, usable = push(demod, sync, origin=origin)
    finally:
        dsp._resample_2d = real
    check(
        "the field decodes so there is provenance to check",
        fr is not None and usable == rows,
        f"usable={usable}/{rows} reject={reject}",
    )
    if fr is None or not seen:
        check("the rendered block was captured", False, "nothing was rendered")
        return
    block = seen[-1]
    lo_got, hi_got = int(block.min()), int(block.max()) + 1
    check(
        "first_sample/last_sample are the gathered interval",
        fr.first_sample == origin + lo_got and fr.last_sample == origin + hi_got,
        f"reported=[{fr.first_sample}, {fr.last_sample}) "
        f"gathered=[{origin + lo_got}, {origin + hi_got})",
    )
    # The bug this pins: the interval quoted from the *detections*, which sit
    # off_i samples before the picture at both ends. It was a plausible number
    # and it was wrong on every field, by a constant that a reader cannot see.
    naive_lo = origin + int(sync.pulse_starts[0])
    naive_hi = origin + int(sync.pulse_starts[-1]) + n_src
    check(
        "the detections-only interval is different, and was the wrong one",
        naive_lo != fr.first_sample and naive_hi != fr.last_sample,
        f"detections=[{naive_lo}, {naive_hi}) vs reported "
        f"[{fr.first_sample}, {fr.last_sample}), off_i={off_i}",
    )
    check(
        "last_sample is exclusive of the interval it names",
        fr.last_sample - fr.first_sample == int(sync.pulse_starts[-1])
        - int(sync.pulse_starts[0]) + n_src,
        f"span={fr.last_sample - fr.first_sample} "
        f"expected={int(sync.pulse_starts[-1]) - int(sync.pulse_starts[0]) + n_src}",
    )
    # And an unknown origin stays unknown rather than becoming a small number.
    fr2, _, _ = push(demod, sync)
    check(
        "an unknown origin reports -1 rather than a relative position",
        fr2 is not None and fr2.first_sample == -1 and fr2.last_sample == -1,
        f"first_sample={None if fr2 is None else fr2.first_sample} "
        f"last_sample={None if fr2 is None else fr2.last_sample}",
    )

    # The lower end is taken from the *run*, not from the detections array: here
    # the first detection is unusable, so the picture starts at the second one.
    spec = replace(dsp.NTSC, visible_lines=9, crop_start=-0.20, crop_width=0.50)
    demod2, sync2, _, off_i2, n_src2 = field(spec, 640.0, rows=int(spec.visible_lines) + 1)
    seen.clear()
    dsp._resample_2d = spy
    try:
        fr3, reject3, usable3 = push(demod2, sync2, origin=origin)
    finally:
        dsp._resample_2d = real
    starts2 = sync2.pulse_starts
    check(
        "a field whose first detection is dropped decodes from the run",
        fr3 is not None and usable3 == int(spec.visible_lines),
        f"usable={usable3}/{int(spec.visible_lines)} reject={reject3} "
        f"detections={starts2.size}",
    )
    if fr3 is not None and seen:
        block2 = seen[-1]
        lo2, hi2 = int(block2.min()), int(block2.max()) + 1
        check(
            "provenance follows the run's first row, not the dropped detection",
            fr3.first_sample == origin + lo2 and fr3.last_sample == origin + hi2,
            f"reported=[{fr3.first_sample}, {fr3.last_sample}) "
            f"gathered=[{origin + lo2}, {origin + hi2}) off_i={off_i2} "
            f"dropped detection at {int(starts2[0])}",
        )
        check(
            "the dropped detection would have been a different interval",
            origin + int(starts2[0]) + off_i2 != fr3.first_sample,
            f"if the detections had been used: {origin + int(starts2[0]) + off_i2}",
        )
        check(
            "every gathered sample was inside the buffer",
            lo2 >= 0 and hi2 <= demod2.size,
            f"gathered=[{lo2}, {hi2}) of {demod2.size} n_src={n_src2}",
        )
    else:
        check("the rendered block was captured", False, "nothing was rendered")


def main() -> int:
    print("Rasteriser usable-line bound check")
    case_exact_fit()
    case_one_short()
    case_fractional_line()
    case_indexing()
    case_provenance()
    print("\n" + ("RESULT: FAIL -- " + ", ".join(FAILURES) if FAILURES else "RESULT: PASS"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())