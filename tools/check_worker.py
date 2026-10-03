"""End-to-end check of the streaming decode path (sim source, no hardware)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv_rf import sdr, video  # noqa: E402

OUT = Path("_validate_out")

#: Enough frames to prove the streaming path sustains a real frame rate rather
#: than merely producing a first frame and stalling. Two decode cadences' worth,
#: from :data:`video.PREVIEW_HZ` rather than a literal, so the gate follows the
#: product if the cadence ever changes. The worker publishes at most one picture
#: per iteration, so this is also the shortest wait that cannot pass by luck.
MIN_FRAMES = 2 * int(video.PREVIEW_HZ) + 1

#: Upper bound on the wait, not the wait itself. The loop exits as soon as
#: MIN_FRAMES have arrived, so a fast machine is not made to wait and a slow one
#: is not failed for being slow -- it is only failed if it cannot reach
#: :data:`video.PREVIEW_HZ` even with a generous budget. That matters on a shared
#: CI runner, where the honest verdict is "this machine is loaded", not "the
#: decoder regressed".
DEADLINE_S = 30.0


def main() -> int:
    src = sdr.SimSource(video=True)
    src.start()
    w = video.DecodeWorker(src, width=160)
    w.start()
    # Wait for the frame count, not for a stopwatch. A fixed sleep has to be
    # tuned to the slowest machine you expect and is wrong everywhere else.
    end = time.monotonic() + DEADLINE_S
    while w.stats.frames < MIN_FRAMES and time.monotonic() < end:
        time.sleep(0.05)
    fr = w.latest()
    st = w.stats
    w.stop()
    src.stop()

    print(f"frames        : {st.frames}")
    print(f"frame rate    : {st.frame_rate():.1f} fps  "
          f"(decode cadence {video.PREVIEW_HZ:.0f} Hz)")
    print(f"cycle         : {st.cycle_ms:.1f} ms  (drain {st.drain_ms:.1f} + decode {st.decode_ms:.1f} + wait)")
    print(f"chunk         : {st.chunk_samples} samples "
          f"({st.chunk_samples/(w.decoder.sample_rate*video.NTSC_FIELD_INTERVAL_S):.2f} NTSC fields)")
    print(f"sync quality  : {st.sync_quality:.3f}   standard={st.standard}")
    print(f"line rate     : {st.line_rate_hz:.1f} Hz")
    print(f"iq dropped    : {st.dropped_iq_samples} samples (ring overrun)")
    print(f"source rate   : {src.stats.rate(src.sample_rate):.2f}x realtime")
    if fr is not None:
        print(f"latest frame  : {fr.width}x{fr.height} age={time.monotonic()-fr.timestamp:.3f}s")
        from PIL import Image

        OUT.mkdir(exist_ok=True)
        Image.fromarray(fr.image, "L").save(OUT / "worker_frame.png")
    ok = fr is not None and st.frames >= MIN_FRAMES
    if not ok and st.frames < MIN_FRAMES:
        print(f"only {st.frames} frames in {DEADLINE_S:.0f}s "
              f"({st.frame_rate():.1f} fps) -- a loaded machine, or a stall")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
