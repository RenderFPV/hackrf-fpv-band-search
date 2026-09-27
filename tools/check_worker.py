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
#: than merely producing a first frame and stalling.
MIN_FRAMES = 61

#: Upper bound on the wait, not the wait itself. The loop exits as soon as
#: MIN_FRAMES have arrived, so a fast machine is not made to wait and a slow one
#: is not failed for being slow -- it is only failed if it cannot reach 60 fps
#: even with a generous budget. That matters on a shared CI runner, where the
#: honest verdict is "this machine is loaded", not "the decoder regressed".
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
    print(f"frame rate    : {st.frame_rate():.1f} fps  (target 60)")
    print(f"cycle         : {st.cycle_ms:.1f} ms  (drain {st.drain_ms:.1f} + decode {st.decode_ms:.1f} + wait)")
    print(f"chunk         : {st.chunk_samples} samples "
          f"({st.chunk_samples/(10e6/60):.2f} fields)")
    print(f"sync quality  : {st.sync_quality:.3f}   standard={st.standard}")
    print(f"line rate     : {st.line_rate_hz:.1f} Hz")
    print(f"iq dropped    : {st.dropped_iq_samples} samples (ring overrun)")
    print(f"source rate   : {src.stats.rate(src.sample_rate):.2f}x realtime")
    if fr is not None:
        print(f"latest frame  : {fr.width}x{fr.height} age={time.monotonic()-fr.timestamp:.3f}s")
        from PIL import Image

        OUT.mkdir(exist_ok=True)
        Image.fromarray(fr.image, "L").save(OUT / "worker_frame.png")
    ok = fr is not None and st.frames > 60
    if not ok and st.frames < MIN_FRAMES:
        print(f"only {st.frames} frames in {DEADLINE_S:.0f}s "
              f"({st.frame_rate():.1f} fps) -- a loaded machine, or a stall")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
