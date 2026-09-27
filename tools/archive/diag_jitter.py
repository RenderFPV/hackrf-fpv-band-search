"""Diagnose the dominant source of line-to-line decorrelation.

Hypothesis under test: per-line sync centroids jitter by a few samples, so each
scanline is resampled at a slightly different horizontal phase, which shears the
picture and destroys vertical correlation.

We measure the residual of each detected pulse against a perfectly periodic
grid, then compare three rasterisation strategies on the same data.
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
from _paths import CAPTURE  # noqa: E402

CAP = CAPTURE  # noqa: F821  (from _paths, below)
FS = 10_000_000.0


def load(path: Path, n: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8, count=n * 2)[: (n * 2) // 2 * 2]
    v = raw.astype(np.float32).reshape(-1, 2)
    v = (v - 127.5) / 127.5
    return (v[:, 0] + 1j * v[:, 1]).astype(np.complex64)


def rowcorr(img: np.ndarray) -> float:
    a = img[:-1].astype(np.float32).ravel()
    b = img[1:].astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d else 0.0


def raster(env, centres, line, spec, width=320, out_h=240, start_frac=0.18, width_frac=0.68):
    start = line * start_frac
    stop = start + line * width_frac
    rows = []
    for k in range(centres.size - 1):
        a, b = int(centres[k] + start), int(centres[k] + stop)
        if a < 0 or b >= env.size or b - a < 8:
            continue
        src = env[a:b]
        rows.append(np.interp(np.linspace(0, src.size - 1, width), np.arange(src.size), src))
    if len(rows) < out_h + 2:
        return None
    block = np.stack(rows[: len(rows) // out_h * out_h])
    p_w, p_b = np.percentile(env, 10), np.percentile(env, 90)
    blk = np.clip((p_b - block.reshape(out_h, -1, width).mean(axis=1)) * 255.0 / max(p_b - p_w, 1e-6), 0, 255)
    return blk.astype(np.uint8)


def main() -> int:
    iq = load(CAP, 8_000_000)
    demod = dsp.fm_demodulate(iq)
    env = np.abs(demod)
    sync = dsp.detect_sync(demod, FS, polarity="auto")
    c = sync.pulse_centres
    L = sync.line_samples
    print(f"pulses={c.size}  line={L:.3f}  spec={sync.spec.name}  quality={sync.quality:.3f}")

    # --- jitter measurement (with true line numbering) ----------------------
    d = np.diff(c)
    steps = np.maximum(1, np.rint(d / L).astype(np.int64))
    line_no = np.concatenate(([0], np.cumsum(steps)))
    grid = c[0] + line_no * L
    resid = c - grid
    contig = steps <= 3
    r = resid[:-1][contig]
    print(f"\nresidual vs line-numbered grid (contiguous runs, n={r.size}):")
    print(f"  std={r.std():.3f} samples   p95|r|={np.percentile(np.abs(r),95):.3f}   max={np.abs(r).max():.1f}")
    print(f"  1 sample = {L/FS*1e6:.2f} us = {1/320*100:.3f}% of image width")
    hist = dict(zip(*[x.tolist() for x in np.unique(steps, return_counts=True)]))
    print(f"  gap histogram (lines:count): {hist}")

    spec = sync.spec
    fitted, a, b = dsp.line_grid(c, L)
    print(f"  fitted grid: intercept={a:.1f} period={b:.4f} (nominal {L:.4f})")
    variants = {
        "A per-line centroid": c,
        "B perfect global grid": grid,
        "E line-numbered fit": fitted,
    }
    print()
    for name, centres in variants.items():
        img = raster(env, centres, L, spec)
        if img is None:
            print(f"{name:24s} -> insufficient lines")
            continue
        print(f"{name:24s} -> rowcorr={rowcorr(img):+.4f}  mean={img.mean():.1f} std={img.std():.1f}")

    # --- crop sweep ---------------------------------------------------------
    print("\ncrop sweep (line-numbered fit), start x width -> rowcorr")
    best = None
    for sf in (0.20, 0.22, 0.24, 0.26, 0.28, 0.30):
        cells = []
        for wf in (0.70, 0.74, 0.78, 0.82, 0.86, 0.90):
            img = raster(env, fitted, L, spec, start_frac=sf, width_frac=wf)
            if img is None:
                continue
            rc = rowcorr(img)
            cells.append(f"w{wf:.2f}:{rc:+.3f}")
            if best is None or rc > best[0]:
                best = (rc, sf, wf, img)
        print(f"  start={sf:.2f}  " + "  ".join(cells))
    if best:
        print(f"\nbest: start={best[1]:.2f} width={best[2]:.2f} rowcorr={best[0]:+.4f}")
        np.save(Path(__file__).parents[1] / "_validate_out" / "best_crop.npy", best[3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
