"""Randomised check: _cluster must match a plain greedy min-gap filter exactly."""

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


def greedy(idx: np.ndarray, min_gap: int) -> np.ndarray:
    out = [float(idx[0])]
    for v in idx[1:]:
        v = float(v)
        if v - out[-1] >= min_gap:
            out.append(v)
    return np.asarray(out, dtype=np.float64)


def invariant_ok(idx: np.ndarray, min_gap: int) -> bool:
    """Consecutive kept edges must be at least min_gap apart."""
    kept = dsp._cluster(idx, min_gap)
    if kept.size < 2:
        return True
    return bool(np.all(np.diff(kept) >= min_gap))


def main() -> int:
    rng = np.random.default_rng(7)
    # 1. the invariant, on dense adversarial input
    bad_inv = 0
    for _ in range(500):
        idx = np.sort(
            rng.choice(np.arange(0, 8000), size=int(rng.integers(2, 80)), replace=False)
        ).astype(float)
        if not invariant_ok(idx, int(rng.integers(2, 600))):
            bad_inv += 1
    print(f"min-gap invariant held   : {500 - bad_inv}/500")

    # 2. agreement with greedy on sync-like spacing: one line apart with jitter,
    #    extra crossings inside pulses, and vertical-blanking sized holes
    agree = 0
    trials = 300
    for _ in range(trials):
        n = int(rng.integers(20, 400))
        gaps = rng.normal(636, 12, size=n)
        # spurious extra crossings inside some pulses (a few samples apart)
        extra = rng.random(n) < 0.3
        gaps = gaps + extra * rng.uniform(2, 40, n)
        # vertical blanking / dropped blocks
        holes = rng.random(n) < 0.02
        gaps = gaps + holes * 636 * rng.integers(20, 90, n)
        idx = np.cumsum(np.concatenate(([0], gaps)))
        idx = np.maximum.accumulate(idx)
        if np.array_equal(dsp._cluster(idx, 318), greedy(idx, 318)):
            agree += 1
    print(f"matches greedy on sync   : {agree}/{trials}")

    # 3. the hand-computed cases cited in the docstring
    for idx, mg, want in (
        (np.array([0.0, 400, 500, 900]), 318, [0, 400, 900]),
        (np.array([0.0, 200, 400]), 318, [0]),
    ):
        got = dsp._cluster(idx, mg)
        ok = np.array_equal(got, np.asarray(want, dtype=float))
        print(f"case {idx.tolist()} -> {got.tolist()} {'OK' if ok else 'WRONG'}")

    iq = load_u8(DEFAULT, 600_000)
    d = dsp.fm_demodulate(iq)
    m = d > np.percentile(d, 97.5)
    st = dsp._rising_edges(m)
    t0 = time.perf_counter()
    for _ in range(50):
        dsp._cluster(st, 318)
    dt = (time.perf_counter() - t0) / 50 * 1000
    print(f"_cluster on {st.size} edges: {dt:.3f} ms")
    return 0 if bad_inv == 0 and agree == trials else 1


if __name__ == "__main__":
    raise SystemExit(main())
