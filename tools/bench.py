"""Time each stage of the decode path, and check the fast demod is exact."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv_rf import dsp  # noqa: E402
from tools.validate_dsp import DEFAULT, load_u8  # noqa: E402


def timeit(fn, n=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def main() -> int:
    iq = load_u8(DEFAULT, 600_000)
    fast = dsp.fm_demodulate(iq)
    ref = np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float32)
    err = float(np.abs(fast - ref).max())
    print(f"fm_demodulate fast vs np.angle: max abs err = {err:.3e}")

    d = fast
    print(f"input {d.size} samples")
    print(f"fm_demodulate 600k   {timeit(lambda: dsp.fm_demodulate(iq)):6.2f} ms")
    print(f"percentile 97.5      {timeit(lambda: np.percentile(d, 97.5)):6.2f} ms")

    # the subsampled threshold must land inside the same plateau
    full = float(np.percentile(d, 97.5))
    sub = dsp._high_percentile(d, 97.5)
    plateau = (d > full).mean()
    print(
        f"threshold full={full:.5f} subsampled={sub:.5f} "
        f"delta={abs(sub-full):.5f} = {abs(sub-full)/d.std():.3%} of a std dev;"
        f" plateau is {plateau:.2%} of samples"
    )
    n_syn = (d > sub).sum()
    n_syn_full = (d > full).sum()
    print(f"  samples above: full {n_syn_full}, subsampled {n_syn}, "
          f"ratio {n_syn/max(1,n_syn_full):.4f}")
    print(f"_high_percentile     {timeit(lambda: dsp._high_percentile(d, 97.5)):6.2f} ms")
    thr = float(np.percentile(d, 97.5))
    m = d > thr
    print(f"mask + rising        {timeit(lambda: dsp._rising_edges(d > np.percentile(d, 97.5))):6.2f} ms")
    st = dsp._rising_edges(m)
    print(f"  edges {st.size}")
    print(f"cluster              {timeit(lambda: dsp._cluster(st, 318)):6.2f} ms")
    print(f"centroids            {timeit(lambda: dsp._centroids(d, m, st)):6.2f} ms")
    print(f"detect_sync 600k     {timeit(lambda: dsp.detect_sync(d, 10e6, polarity='auto')):6.2f} ms")

    for w in (200_000, 240_000, 320_000):
        seg = d[-w:]
        print(
            f"detect_sync {w//1000:>3}k     "
            f"{timeit(lambda seg=seg: dsp.detect_sync(seg, 10e6, polarity='auto')):6.2f} ms"
        )
    s = dsp.detect_sync(d, 10e6)
    print(f"  pulses {s.pulse_starts.size}")
    print(f"raster.push          {timeit(lambda: dsp.Rasteriser(width=160).push(d, s)):6.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
