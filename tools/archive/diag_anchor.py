"""Decide the line anchor: sync leading edge vs threshold-crossing centroid.

Also switches to a banding-proof quality metric. Plain adjacent-row
correlation is confounded: a frame that is nothing but horizontal bands scores
~0.45 because every row looks like every other. Subtracting each row's own mean
removes any per-row DC (i.e. all banding) and leaves only real picture
structure, which is what we actually want to maximise.
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


def detrend(img: np.ndarray) -> np.ndarray:
    return img - img.mean(axis=1, keepdims=True)


def rowcorr(img: np.ndarray, detr: bool = True) -> float:
    x = detrend(img) if detr else img
    a = x[:-1].astype(np.float32).ravel()
    b = x[1:].astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d else 0.0


def raster(env, centres, line, spec, width=320, out_h=240, start_frac=None, width_frac=None):
    sf = spec.active_start_frac if start_frac is None else start_frac
    wf = spec.active_width_frac if width_frac is None else width_frac
    start = line * sf
    stop = start + line * wf
    rows = []
    for k in range(centres.size - 1):
        a, b = int(centres[k] + start), int(centres[k] + stop)
        if a < 0 or b >= env.size or b - a < 8:
            continue
        src = env[a:b]
        rows.append(np.interp(np.linspace(0, src.size - 1, width), np.arange(src.size), src))
    if len(rows) < out_h + 2:
        return None
    p_w, p_b = np.percentile(env, 10), np.percentile(env, 90)
    blk = np.stack(rows[: len(rows) // out_h * out_h]).reshape(out_h, -1, width).mean(axis=1)
    return np.clip((p_b - blk) * 255.0 / max(p_b - p_w, 1e-6), 0, 255).astype(np.uint8)


def main() -> int:
    iq = load(CAP, 6_000_000)
    demod = dsp.fm_demodulate(iq)
    env = np.abs(demod)
    sync = dsp.detect_sync(demod, FS, polarity="auto")
    L, spec = sync.line_samples, sync.spec
    print(f"line={L:.2f}  spec={spec.name}  pulses={sync.pulse_centres.size}")
    print(f"physical crop from sync leading edge: start={spec.active_start_frac:.3f} "
          f"width={spec.active_width_frac:.3f}")

    # metric sanity
    rng = np.random.default_rng(3)
    bands = np.tile(np.arange(240, dtype=np.float32)[:, None], (1, 320)).astype(np.uint8)
    print(f"\nmetric check: bands detrended={rowcorr(bands):+.4f}  noise={rowcorr(rng.integers(0,255,(240,320),dtype=np.uint8)):+.4f}")

    grid_c, _, _ = dsp.line_grid(sync.pulse_centres, L)
    # rising-edge anchor: re-derive starts the same way detect_sync does
    smooth_taps = max(1, int(round(FS * 1.0e-6)))
    env_s = dsp._smooth(env, smooth_taps)
    thr = float(np.percentile(env_s, 98.5))
    starts = dsp._rising_edges(env_s > thr)
    starts = dsp._cluster(starts, max(4, int(L * 0.5)))
    grid_s, _, _ = dsp.line_grid(starts.astype(np.float64), L)
    print(f"rising edges kept: {starts.size} (centroids {sync.pulse_centres.size})")

    print(f"\n{'variant':38s} {'rowcorr':>9s} {'plain':>9s} {'mean':>7s} {'std':>7s}")
    cands = {
        "centroid + grid, phys crop": (grid_c, None, None),
        "rising-edge + grid, phys crop": (grid_s, None, None),
    }
    for name, (c, sf, wf) in cands.items():
        img = raster(env, c, L, spec, start_frac=sf, width_frac=wf)
        if img is None:
            print(f"{name:38s} insufficient lines")
            continue
        print(f"{name:38s} {rowcorr(img):+9.4f} {rowcorr(img, False):+9.4f} "
              f"{img.mean():7.1f} {img.std():7.1f}")

    # fine sweep on the leading-edge anchor to confirm the physical crop
    print("\nleading-edge anchor sweep -> detrended rowcorr")
    best = None
    for sf in (0.10, 0.12, 0.145, 0.17, 0.20):
        cells = []
        for wf in (0.72, 0.78, 0.828, 0.88):
            img = raster(env, grid_s, L, spec, start_frac=sf, width_frac=wf)
            if img is None:
                continue
            rc = rowcorr(img)
            cells.append(f"w{wf:.3f}:{rc:+.3f}")
            if best is None or rc > best[0]:
                best = (rc, sf, wf, img)
        print(f"  start={sf:.3f}  " + "  ".join(cells))
    if best:
        print(f"\nbest: start={best[1]:.3f} width={best[2]:.3f} rowcorr={best[0]:+.4f}")
        out = Path(__file__).parents[1] / "_validate_out" / "best_crop.png"
        from PIL import Image

        Image.fromarray(best[3], "L").save(out)
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
