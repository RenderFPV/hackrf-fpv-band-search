"""End-to-end check: sim, file, and the real capture all through decode+scan."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv_rf import dsp, sdr  # noqa: E402
from tools.check_frame import row_correlation  # noqa: E402
from tools.validate_dsp import DEFAULT, load_u8  # noqa: E402

FS = 10_000_000
OUT = Path("_validate_out")


def report(tag, iq):
    print(f"--- {tag}: {iq.size} samples")
    occ = dsp.assess_channel(iq, FS, dsp.NTSC)
    exp = int(iq.size * dsp.NTSC.line_rate_hz / FS)
    print(f"    assess: kind={occ.kind.value} pulses={occ.sync_pulses} "
          f"(expect ~{exp}) qual={occ.sync_quality:.2f} line={occ.line_rate_hz:.1f}Hz "
          f"{occ.spec.name if occ.spec else '?'} density={occ.sync_density:.2f}")
    demod = dsp.fm_demodulate(iq)
    s = dsp.detect_sync(demod, FS, polarity="auto")
    print(f"    sync  : pol={s.polarity:+d} line={s.line_samples:.2f} qual={s.quality:.3f} "
          f"pulses={s.pulse_starts.size}")
    r = dsp.Rasteriser(width=160)
    fr = r.push(demod, s)
    if fr is None:
        print("    frame : NONE")
        return
    print(f"    frame : {fr.image.shape[1]}x{fr.image.shape[0]} "
          f"row={row_correlation(fr.image):+.3f} mean={fr.image.mean():.1f}")
    from PIL import Image

    Image.fromarray(fr.image, "L").save(OUT / f"chk_{tag}.png")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    src = sdr.SimSource(video=True)
    src.start()
    time.sleep(1.4)
    report("sim_video", src.take_iq(1_000_000))
    src.stop()

    src = sdr.SimSource(video=False)
    src.start()
    time.sleep(0.6)
    report("sim_noise", src.take_iq(400_000))
    src.stop()

    # No capture is the normal case for a fresh clone, and it is not a failure:
    # the sim and noise cases above are what CI can actually verify. Only the
    # extra realism of a real recording is unavailable, and saying so is more
    # useful than dying inside np.fromfile with a bare FileNotFoundError.
    if DEFAULT.exists():
        report("real_capture", load_u8(DEFAULT, 8_000_000))
    else:
        print("--- real_capture: SKIPPED (no capture found)")
        print(f"    looked for {DEFAULT}")
        print("    record one with:  python tools/record_sample.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
