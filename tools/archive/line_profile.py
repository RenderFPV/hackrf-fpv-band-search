"""Inspect the average per-line envelope profile to locate sync/blanking.

For a correctly tuned analogue signal the mean envelope across many lines has
an unmistakable shape: a narrow sync tip, a low blanking shoulder, then the
picture plateau. Locating those features data-driven removes the magic
constants from the crop and lets the decoder self-calibrate for NTSC vs PAL and
for whatever overscan a given VTX uses.
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


def main() -> int:
    iq = load(CAP, 4_000_000)
    demod = dsp.fm_demodulate(iq)
    env = np.abs(demod)
    sync = dsp.detect_sync(demod, FS, polarity="auto")
    L = sync.line_samples
    fitted, a, b = dsp.line_grid(sync.pulse_centres, L)

    # average envelope as a function of position within a line
    nlines = int(min(fitted.size, 3000))
    pos = fitted[:nlines]
    offsets = np.arange(int(round(L)))
    idx = pos[:, None] + offsets[None, :]              # (nlines, L)
    ok = (idx >= 0) & (idx < env.size)
    good = ok.all(axis=1)
    idx = idx[good].astype(np.int64)
    if idx.shape[0] < 16:
        print("not enough in-range lines for a profile")
        return 1
    prof = env[idx].mean(axis=0)
    print(f"lines used={idx.shape[0]}  line={L:.2f} samples")
    print(f"profile min={prof.min():.4f} max={prof.max():.4f} mean={prof.mean():.4f}")

    # locate the sync tip = global max of the profile
    tip = int(np.argmax(prof))
    print(f"\nsync tip at offset {tip} ({tip/L*100:.1f}% of line)")
    hi = float(np.percentile(prof, 90))
    lo = float(np.percentile(prof, 10))
    thr = lo + 0.35 * (hi - lo)
    above = prof > thr
    start = None
    for i in range(tip + 1, prof.size):
        if above[i] and above[i : i + 20].all():
            start = i
            break
    end = None
    if start is not None:
        below = np.flatnonzero(~above[start:])
        if below.size:
            end = int(start + below[0])
    print(f"picture starts ~{start} ({None if start is None else round(start/L,4)} of line)")
    print(f"picture ends   ~{end} ({None if end is None else round(end/L,4)} of line)")
    if start is not None and end:
        print(f"=> hblank_start_frac ~ {start/L:.4f}   active_width_frac ~ {(end-start)/L:.4f}")

    p = (prof - prof.min()) / (prof.max() - prof.min() or 1.0)
    print("\naverage line profile, whole line in rows of ~100 samples")
    print("(0..1 normalised; ' '=low envelope=white, '@'=high envelope=black/sync)")
    for a0 in range(0, p.size, 100):
        seg = p[a0 : a0 + 100]
        print(f"  {a0:4d} |" + "".join(" .:-=+*#%@"[min(9, int(v * 10))] for v in seg) + "|")
    print("\nnumeric samples every 10:")
    print("  " + " ".join(f"{i}:{p[i]:.2f}" for i in range(0, p.size, 10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
