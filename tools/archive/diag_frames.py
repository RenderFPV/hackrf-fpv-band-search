"""Per-iteration diagnostics: why does the streaming decoder skip frames?"""

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
from fpv_rf import dsp, sdr, video  # noqa: E402


def main() -> int:
    src = sdr.SimSource(video=True)
    src.start()
    dec = video.VideoDecoder(src.sample_rate, width=160)
    field = int(src.sample_rate / 60)
    max_chunk = int(field * 1.05)
    print(f"field={field} max_chunk={max_chunk} window={dec.window}")
    print(f"{'iter':>5} {'chunk':>7} {'pulses':>7} {'qual':>5} {'line':>8} "
          f"{'usable':>7} {'run':>4} {'frame':>5}  {'t_ms':>6}")
    t_dec = 0.0
    emitted = 0
    n = 0
    t_end = time.monotonic() + 3.0
    while time.monotonic() < t_end:
        iq = src.drain_iq(max_chunk)
        n += 1
        if iq.size < 64:
            time.sleep(0.002)
            continue
        t0 = time.perf_counter()
        new = dsp.fm_demodulate(iq)
        dec._demod = np.concatenate((dec._demod, new))[-dec.window:]
        sig = dec._demod
        sync = dsp.detect_sync(sig, src.sample_rate, polarity="auto")
        fr = dec.raster.push(sig, sync)
        dt = (time.perf_counter() - t0) * 1000
        t_dec += dt

        spec = sync.spec
        want = spec.visible_lines
        line = sync.line_samples
        offset = line * spec.crop_start
        span = line * spec.crop_width
        limit = sig.size - (offset + span)
        usable = sync.pulse_starts[sync.pulse_starts + offset + span <= limit]
        run = dsp.Rasteriser.select_run(usable, line, want) if usable.size >= want else None
        got = fr is not None
        emitted += got
        gaps = np.diff(usable) if usable.size > 1 else np.zeros(0)
        bad = gaps[(gaps < line * 0.85) | (gaps > line * 1.15)]
        if n <= 12 or not got:
            print(f"{n:5d} {iq.size:7d} {sync.pulse_starts.size:7d} {sync.quality:5.2f} "
                  f"{line:8.2f} {usable.size:7d} {0 if run is None else run.size:4d} "
                  f"{'yes' if got else 'NO':>5}  {dt:6.1f}"
                  f"  badgaps={bad.size} {[int(g) for g in bad[:8]]}")
        if dt < 1 / 60:
            time.sleep(1 / 60 - dt)
    src.stop()
    print(f"\niterations {n}, frames {emitted} ({emitted/max(1,n):.0%}), "
          f"mean decode {t_dec/max(1,n):.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
