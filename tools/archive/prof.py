"""cProfile the decode path to find where the time actually goes."""

from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path

import numpy as np

# Two levels up is the repository root. The tools directory is needed as
# well: these scripts sit one level deeper now, and they import `_paths`
# and each other by bare name.
_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT), str(_ROOT / "tools")]
from fpv_rf import dsp  # noqa: E402
from tools.validate_dsp import DEFAULT, load_u8  # noqa: E402


def main() -> int:
    iq = load_u8(DEFAULT, 600_000)
    d = dsp.fm_demodulate(iq)
    s = dsp.detect_sync(d, 10e6)

    def work():
        for _ in range(10):
            dsp.detect_sync(d, 10e6, polarity="auto")
            dsp.Rasteriser(width=160).push(d, s)

    pr = cProfile.Profile()
    pr.enable()
    work()
    pr.disable()
    st = pstats.Stats(pr)
    st.sort_stats("tottime").print_stats(16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
