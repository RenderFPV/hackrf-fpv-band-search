"""Numeric sanity check on a decoded frame, since images cannot be eyeballed here.

Checks the properties a real test pattern must have and noise must not:
  * rows must be internally smooth (a lock failure makes every row differ)
  * there must be a left-to-right gradient (the horizontal ramp)
  * there must be a top-to-bottom band structure (the checkerboard)
  * the histogram must use the available range, not sit in a few levels
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def describe(name: str, img: np.ndarray) -> None:
    a = img.astype(np.float64)
    rows = a.mean(axis=1)
    cols = a.mean(axis=0)
    # structure: correlation between neighbouring rows and neighbouring columns
    row_corr = np.corrcoef(a[:-1].ravel(), a[1:].ravel())[0, 1]
    col_corr = np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1]
    rng = float(a.max() - a.min())
    # how much of the variance is explained by the row means alone
    band = float(np.var(rows) / max(1e-9, np.var(a)))
    print(f"{name:14s} {img.shape[1]}x{img.shape[0]}  "
          f"range={a.min():.0f}..{a.max():.0f} (span {rng:.0f})  "
          f"mean={a.mean():.1f}")
    print(f"{'':14s} row-pair corr={row_corr:+.3f}  col-pair corr={col_corr:+.3f}  "
          f"var explained by row means={band:.2f}")
    print(f"{'':14s} row means span {rows.max()-rows.min():.1f}, "
          f"col means span {cols.max()-cols.min():.1f}")
    # a noise-only image has row-pair corr near 0 and a small range
    ok = row_corr > 0.3 and col_corr > 0.3 and rng > 40
    print(f"{'':14s} -> {'STRUCTURED' if ok else 'NOISE-LIKE / FLAT'}")


def main() -> int:
    out = Path("_validate_out")
    seen = missing = 0
    for fn in ("worker_frame.png", "chk_sim_video.png", "chk_real_capture.png"):
        p = out / fn
        if p.exists():
            describe(fn, np.array(Image.open(p)))
            print()
            seen += 1
        else:
            missing += 1
    # This script reports on frames other checks have already written; it has
    # no assertion of its own and always exits 0. That is a deliberate choice,
    # but a silent one is a trap: finding no files printed nothing at all and
    # looked exactly like a pass. Say what was examined instead.
    if not seen:
        print(f"no frames to examine in {out}/ -- run check_worker.py or "
              f"check_sources.py first to produce some")
    elif missing:
        print(f"examined {seen} frame(s); {missing} not present (the producing "
              f"check was skipped, or has not been run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
