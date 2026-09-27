"""Profile one block of SimSource generation."""

from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

# Two levels up is the repository root. The tools directory is needed as
# well: these scripts sit one level deeper now, and they import `_paths`
# and each other by bare name.
_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT), str(_ROOT / "tools")]
from fpv_rf import sdr  # noqa: E402


def gen(s: sdr.SimSource) -> None:
    two_pi = np.float64(2.0 * np.pi)
    n = s._block_n
    iq = s._scratch_iq
    ph = s._phase_buf
    acc = np.empty(n, dtype=np.float64)
    dev = s._deviation(s._lines_per_block)
    np.cumsum(dev, dtype=np.float64, out=acc)
    acc += s._phase
    np.mod(acc, two_pi, out=acc)
    s._phase = float(acc[-1])
    np.copyto(ph, acc, casting="unsafe")
    ph += s._rng.standard_normal(n, dtype=np.float32) * np.float32(s.noise)
    np.cos(ph, out=iq[:, 0])
    np.sin(ph, out=iq[:, 1])
    np.multiply(iq, np.float32(127.0), out=iq)
    np.rint(iq, out=iq)
    np.add(iq, np.float32(128.0), out=iq)
    u8 = iq.astype(np.uint8)
    s.ring.write(u8.tobytes())


def stage_times(s: sdr.SimSource, reps: int = 20) -> None:
    n = s._block_n
    iq = s._scratch_iq
    ph = s._phase_buf
    acc = np.empty(n, dtype=np.float64)
    two_pi = np.float64(2.0 * np.pi)

    def t(label, fn):
        fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        print(f"  {label:22s} {(time.perf_counter()-t0)/reps*1000:6.2f} ms")

    t("deviation", lambda: s._deviation(s._lines_per_block))
    dev = s._deviation(s._lines_per_block)
    t("cumsum f64", lambda: np.cumsum(dev, dtype=np.float64))
    t("normal f32", lambda: s._rng.standard_normal(n, dtype=np.float32))
    np.copyto(ph, acc, casting="unsafe")
    t("cos+sin -> iq", lambda: (np.cos(ph, out=iq[:, 0]), np.sin(ph, out=iq[:, 1])))
    t("quantise", lambda: (np.multiply(iq, np.float32(127.0), out=iq),
                           np.rint(iq, out=iq),
                           np.add(iq, np.float32(128.0), out=iq),
                           iq.astype(np.uint8)))
    u8 = iq.astype(np.uint8)
    t("ring write", lambda: s.ring.write(u8.tobytes()))


def main() -> int:
    s = sdr.SimSource(video=True)
    n = s._block_n
    budget = n / s.sample_rate * 1000
    print(f"block {n} samples, real-time budget {budget:.2f} ms")
    stage_times(s)
    gen(s)
    t0 = time.perf_counter()
    reps = 30
    for _ in range(reps):
        gen(s)
    dt = (time.perf_counter() - t0) / reps * 1000
    print(f"\ntotal generate {dt:.2f} ms -> {n/dt*1e-6:.1f} MS/s max"
          f" = {n/(dt/1000)/s.sample_rate:.2f}x realtime")
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(20):
        gen(s)
    pr.disable()
    pstats.Stats(pr).sort_stats("tottime").print_stats(12)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
