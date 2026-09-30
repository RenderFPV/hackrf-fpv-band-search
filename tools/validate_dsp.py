"""Offline validation of the DSP chain against the known-good R5 capture.

Run:  python tools/validate_dsp.py [capture.u8]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpv_rf import dsp, sdr  # noqa: E402
from _paths import CAPTURE  # noqa: E402

DEFAULT = CAPTURE  # noqa: F821  (from _paths, below)
OUT = Path(__file__).resolve().parents[1] / "_validate_out"


def load_u8(path: Path, max_samples: int) -> np.ndarray:
    """Interleaved uint8 capture file to complex64, the way the app reads one.

    This used to open-code the conversion: read the bytes, subtract 127.5 from
    every one of them, and call the result IQ. That is the *offset-binary*
    mapping, and it is not what a HackRF writes -- ``hackrf_transfer`` emits
    two's-complement samples, and :class:`fpv_rf.sdr.HackrfSource` says so and
    decodes them as such. So this loader disagreed with the program under test
    about the format of its own recordings, and the disagreement is invisible in
    any single output: reading signed data as offset-binary inverts the waveform
    while preserving its power, so a signal-strength figure stays plausible and
    only a picture comes out wrong.

    So it goes through the app's own conversion with the hardware format stated
    explicitly. One implementation of "how are these bytes turned into IQ",
    which is the only way the offline numbers and the running app can be
    compared at all.
    """
    raw = np.fromfile(path, dtype=np.uint8, count=max_samples * 2)
    return sdr.u8_to_iq(raw, sdr.IQEncoding.SIGNED_INT8)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"capture not found: {path}")
        return 2
    OUT.mkdir(exist_ok=True)
    fs = 10_000_000.0

    iq = load_u8(path, 8_000_000)
    print(f"capture   : {path.name}  ({iq.size} samples, {iq.size / fs:.3f} s @ {fs/1e6:g} MS/s)")
    print(f"mean |IQ| : {np.mean(np.abs(iq)):.4f}")

    t0 = time.perf_counter()
    occ = dsp.assess_channel(iq, fs, dsp.NTSC)
    t_occ = time.perf_counter() - t0
    print(
        f"\nchannel   : kind={occ.kind.value}  snr={occ.snr_db:6.2f} dB  "
        f"peak/med={occ.peak_to_median_db:5.1f} dB  occupied={occ.occupied}"
    )
    print(f"            sync pulses={occ.sync_pulses} density={occ.sync_density*100:.0f}% "
          f"quality={occ.sync_quality*100:.0f}%  line={occ.line_rate_hz:.1f} Hz")
    print(f"            -> {occ.describe()}   (strength {occ.strength:.0f}, [{t_occ*1000:.0f} ms])")

    t0 = time.perf_counter()
    demod = dsp.fm_demodulate(iq)
    t_demod = time.perf_counter() - t0
    print(f"fm demod  : {t_demod*1000:.0f} ms for {iq.size/1e6:.1f}M samples "
          f"({iq.size/t_demod/1e6:.1f} MS/s)")

    for cut in (None, 5_000_000.0):
        y = demod if cut is None else dsp.limit_video_bandwidth(demod, fs, cut)
        sync = dsp.detect_sync(y, fs, polarity="auto")
        tag = "none" if cut is None else f"{int(cut/1e6)}m"
        print(
            f"\n-- cutoff={tag} --"
            f"\n  pulses   : {sync.pulse_centres.size}"
            f"\n  line     : {sync.line_samples:.2f} samples  "
            f"({sync.line_rate_hz:.1f} Hz -> {sync.spec.name})"
            f"\n  quality  : {sync.quality*100:.1f}%   polarity={sync.polarity:+d}"
        )
        if sync.pulse_centres.size < 10:
            continue
        r = dsp.Rasteriser(width=320)
        fr = r.push(y, sync, brightness=1.0, contrast=1.0)
        if fr is None:
            print("  raster   : no complete field yet")
            continue
        p = OUT / f"frame_{tag}.png"
        try:
            from PIL import Image

            Image.fromarray(fr.image, "L").save(p)
        except Exception:
            np.save(str(p) + ".npy", fr.image)
        print(f"  raster   : {fr.height}x{fr.width} lines_used={fr.lines_used}")
        print(f"  saved    : {p}")
        print(f"  luma     : min={fr.image.min()} max={fr.image.max()} mean={fr.image.mean():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
