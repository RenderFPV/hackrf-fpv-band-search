"""Verify the vectorised _centroids against a slow reference, and time it."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Two levels up is the repository root. The tools directory is needed as
# well: these scripts sit one level deeper now, and they import `_paths`
# and each other by bare name.
_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT), str(_ROOT / "tools")]
from fpv_rf import dsp  # noqa: E402
from tools.validate_dsp import DEFAULT, load_u8  # noqa: E402


def ref_centroids(score, mask, starts):
    out = []
    n = score.size
    for s in starts:
        s = int(s)
        e = s
        while e < n and mask[e]:
            e += 1
        seg = score[s:e]
        if seg.size == 0:
            out.append(float(s))
            continue
        w = np.clip(seg.astype(np.float64) - float(seg.min()), 0, None)
        t = w.sum()
        out.append(s + (float((w * np.arange(seg.size)).sum() / t) if t > 0 else 0.0))
    return np.array(out)


def timeit(fn, n=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def main() -> int:
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(12):
        n = int(rng.integers(500, 4000))
        score = rng.normal(size=n).astype(np.float32)
        mask = score > np.percentile(score, 90)
        st = dsp._rising_edges(mask)
        a = dsp._centroids(score, mask, st)
        b = ref_centroids(score, mask, st)
        assert a.size == b.size, (a.size, b.size)
        if a.size:
            worst = max(worst, float(np.abs(a - b).max()))
    print(f"max |vectorised - reference| over 12 random cases: {worst:.3e}")

    iq = load_u8(DEFAULT, 600_000)
    d = dsp.fm_demodulate(iq)
    m = d > np.percentile(d, 97.5)
    st = dsp._rising_edges(m)
    print(f"centroids now   {timeit(lambda: dsp._centroids(d, m, st)):6.2f} ms")
    print(f"detect_sync now {timeit(lambda: dsp.detect_sync(d, 10e6, polarity='auto')):6.2f} ms")
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
