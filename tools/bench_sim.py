"""What sample rate does each source actually sustain, end to end?"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv_rf import sdr  # noqa: E402


def measure(src: sdr.IQSource, seconds: float = 3.0) -> float:
    src.start()
    time.sleep(0.2)
    t0 = time.perf_counter()
    b0 = src.stats.bytes_total
    time.sleep(seconds)
    rate = (src.stats.bytes_total - b0) / (time.perf_counter() - t0) / 2.0
    src.stop()
    return rate


def main() -> int:
    fs = sdr.DEFAULT_SAMPLE_RATE
    print(f"target {fs/1e6:g} MS/s\n")
    cases = [
        ("sim video", lambda: sdr.SimSource(video=True)),
        ("sim noise", lambda: sdr.SimSource(video=False)),
    ]
    worst = 1.0
    for name, make in cases:
        src = make()
        rate = measure(src)
        worst = min(worst, rate / fs)
        print(
            f"{name:10s} {rate/1e6:6.2f} MS/s  = {rate/fs:4.2f}x realtime"
            f"  {'OK' if rate >= 0.98*fs else 'SLOW'}"
        )
    print(
        "\nverdict:", "OK" if worst >= 0.95 else "SLOW",
        f"(worst {worst:.2f}x; the sim must sustain real time or it starves the decoder)",
    )

    # is the pattern still sane after the rewrite?
    src = sdr.SimSource(video=True)
    src.start()
    time.sleep(0.5)
    iq = src.take_iq(400_000)
    src.stop()
    print(f"\nsim produced {iq.size} samples, mean|IQ|={np.abs(iq).mean():.4f}")
    return 0 if worst >= 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
