"""Where do the extra sync edges in the sim stream come from?"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv_rf import dsp, sdr  # noqa: E402


def main() -> int:
    src = sdr.SimSource(video=True)
    src.start()
    time.sleep(0.4)
    iq = src.take_iq(600_000)
    src.stop()

    d = dsp.fm_demodulate(iq)
    print(f"line_n = {src._line_n}  sync_n = {src._sync_n}  "
          f"lines/block = {src._lines_per_block}")

    # what does the detector see, before any clustering?
    thr = dsp._high_percentile(d, 97.5)
    print(f"threshold {thr:.4f}")
    lo, hi = float(np.percentile(d, 99.9)), float(d.max())
    print(f"sync plateau: max {hi:.4f}   p99.9 {lo:.4f}")
    # how much of the line is above threshold, and where within the line?
    above = d > thr
    print(f"fraction above threshold: {above.mean():.4%}  (sync duty should be "
          f"{src._sync_n/src._line_n:.4%})")
    L = src._line_n
    # position within the line of the samples above threshold, modulo the grid
    pos = (np.flatnonzero(above) % L)
    hist = np.bincount(pos, minlength=L)
    nz = np.flatnonzero(hist)
    print(f"samples-above positions mod {L}: {nz.min()}..{nz.max()}")
    print("  strongest positions:", [(int(p), int(hist[p])) for p in
                                     np.argsort(hist)[-6:][::-1]])

    edges = dsp._rising_edges(above)
    print(f"\nraw rising edges: {edges.size}")
    gaps = np.diff(edges)
    vals, cnt = np.unique(gaps.astype(int), return_counts=True)
    top = np.argsort(cnt)[-8:][::-1]
    print("  most common raw gaps:", [(int(vals[t]), int(cnt[t])) for t in top])

    cl = dsp._cluster(edges, int(0.5 * L))
    print(f"after _cluster: {cl.size}")
    g2 = np.diff(cl)
    vals2, cnt2 = np.unique(g2.astype(int), return_counts=True)
    top2 = np.argsort(cnt2)[-8:][::-1]
    print("  most common clustered gaps:",
          [(int(vals2[t]), int(cnt2[t])) for t in top2])
    odd = g2[(g2 < L * 0.85) | (g2 > L * 1.15)]
    print(f"  off-nominal gaps: {odd.size} of {g2.size}; first 12: {odd[:12].astype(int)}")

    # The check that matters: are the detected edges at the *sync pulse*, i.e.
    # within sync_n samples of the line start? A detector that has locked onto
    # periodic picture edges still reports perfect 636-sample spacing, so only
    # the position within the line reveals the difference.
    sync_pos = (cl % L).astype(np.int64)
    in_sync = (sync_pos <= src._sync_n).mean()
    # the grid origin is whatever the snapshot happened to start on, so compare
    # the spread of sync_pos rather than assuming it starts at 0
    origin = np.bincount(sync_pos, minlength=L).argmax()
    offset = (sync_pos - origin) % L
    within = int((offset <= 2 * src._sync_n).sum())
    print(f"\nsync position check: {within}/{cl.size} edges "
          f"({within/max(1,cl.size):.1%}) within 2*sync_n of the line start "
          f"(origin {origin}, sync_n {src._sync_n})")
    print(f"  offset histogram (top 6): "
          f"{[(int(o), int(c)) for o, c in zip(*np.unique(offset, return_counts=True))][:6]}")
    verdict = within == cl.size and odd.size == 0
    print(f"  -> detector is locking onto {'SYNC' if verdict else 'PICTURE EDGES'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
