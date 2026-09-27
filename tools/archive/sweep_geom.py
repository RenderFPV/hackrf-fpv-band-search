"""Sweep rasteriser geometry to find what actually maximises picture coherence.

Answers three questions at once:
  * is the picture present but horizontally mis-cropped?   (sweep offset/width)
  * does the choice of 240-line run matter?               (sweep run index)
  * is per-line raw anchoring better than a global grid?  (sweep grid mode)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Two levels up is the repository root. The tools directory is needed as
# well: these scripts sit one level deeper now, and they import `_paths`
# and each other by bare name.
_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT), str(_ROOT / "tools")]
from fpv_rf import dsp  # noqa: E402
from tools.check_frame import col_correlation, row_correlation  # noqa: E402
from tools.validate_dsp import DEFAULT, load_u8  # noqa: E402

FS = 10_000_000.0


def runs_of(starts: np.ndarray, line: float, tol: float = 0.15) -> list[tuple[int, int]]:
    good = np.abs(np.diff(starts) - line) <= line * tol
    bounds = np.concatenate(([0], np.flatnonzero(~good) + 1, [starts.size]))
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(bounds.size - 1)]


def render(demod, run, line, off_frac, span_frac, width=160, grid=False, line_grid_period=None, line_grid_int=None):
    a0 = run.astype(np.int64) + int(round(line * off_frac))
    n_src = int(round(line * span_frac))
    if a0.max() + n_src > demod.size:
        return None
    idx = a0[:, None] + np.arange(n_src, dtype=np.int64)[None, :]
    block = demod[idx]
    gx = np.linspace(0.0, n_src - 1.0, width)
    sx = np.arange(n_src, dtype=np.float32)
    rows = np.empty((a0.size, width), dtype=np.float32)
    for i in range(a0.size):
        rows[i] = np.interp(gx, sx, block[i])
    lo, hi = np.percentile(rows, [2.0, 98.0])
    if hi <= lo:
        return None
    y = np.clip((rows - lo) / (hi - lo), 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def main() -> int:
    iq = load_u8(DEFAULT, 8_000_000)
    demod = dsp.fm_demodulate(iq)
    sync = dsp.detect_sync(demod, FS, polarity="auto")
    s, line = sync.pulse_starts, float(sync.line_samples)
    want = sync.spec.visible_lines
    print(f"pulses={s.size} line={line:.2f} want={want} spec={sync.spec.name}")
    rs = runs_of(s, line)
    longs = [(lo, hi) for lo, hi in rs if hi - lo >= want]
    print(f"runs={len(rs)} runs>={want}={len(longs)}")

    # --- 1. sweep geometry on the freshest qualifying run ---------------------
    best = None
    for k, (lo, hi) in enumerate(longs[-6:]):
        run = s[hi - want : hi]
        for off in np.arange(0.10, 0.281, 0.01):
            for span in np.arange(0.66, 0.90, 0.02):
                img = render(demod, run, line, off, span)
                if img is None:
                    continue
                sc = row_correlation(img)
                if best is None or sc > best[0]:
                    best = (sc, col_correlation(img), off, span, k)
    sc, cc, off, span, k = best
    print(f"\nBEST geometry: off={off:.3f} span={span:.2f} run#{k}  row={sc:+.4f} col={cc:+.4f}")
    print("  (defaults: off=0.145 span=0.828)")

    # --- 2. compare default vs best, on same run -----------------------------
    run = s[longs[-6:][k][1] - want : longs[-6:][k][1]]
    for tag, (o, sp) in (("default", (0.145, 0.828)), ("best", (off, span))):
        img = render(demod, run, line, o, sp)
        print(f"  {tag:8s} off={o:.3f} span={sp:.3f} row={row_correlation(img):+.4f} col={col_correlation(img):+.4f} mean={img.mean():.1f}")

    # --- 3. does the run choice matter? -------------------------------------
    print("\nper-run (best geometry):")
    for k, (lo, hi) in enumerate(longs[-8:]):
        img = render(demod, s[hi - want : hi], line, off, span)
        if img is not None:
            print(f"  run#{k} start={s[hi-want]:.0f} row={row_correlation(img):+.4f} col={col_correlation(img):+.4f}")

    # --- 4. reference geometry, for comparison ------------------------------
    print("\nreference geometry (off=0.18 span=0.76):")
    img = render(demod, run, line, 0.18, 0.76)
    print(f"  row={row_correlation(img):+.4f} col={col_correlation(img):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
