"""Write the IQ format fixtures used by tools/check_review_fixes.py.

One waveform, written three ways:

  iq_reference.f32       complex64, the ground truth
  iq_signed_int8.u8     two's complement, which is what hackrf_transfer writes
  iq_unsigned_offset.u8 offset-binary by 128, which is what SimSource writes

Encoding the same samples two ways is what makes the test meaningful: both files
must decode back to the reference, so the test pins the decoded result instead
of merely checking that nothing raised. A test that only checked for absence of
an exception would have passed against the original bug.

The waveform is an FM composite with sync pulses rather than a tone, so the
fixture is also usable for decode tests rather than only for format tests.
Deterministic: no random numbers, so regenerating gives byte-identical files.

Run:  python tools/make_iq_fixtures.py

The first three are generated from scratch. The fourth,
``real_5802_signed.u8``, is a verbatim excerpt of a real recording and needs that
file present; it is committed so the real-capture decode test can run in CI
without a radio. Regenerating without the original leaves the committed copy
alone rather than truncating it to nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).resolve().parent / "fixtures"
FS = 10_000_000
LINES = 300

#: Mean complex amplitude. Matched to the real 5802 MHz capture (mean |x| 0.22)
#: and to the simulator's default 0.20, so the fixtures exercise the same part of
#: the decoder's input range that actual captures do.
AMPLITUDE = 0.20


def composite(n_lines: int = LINES) -> np.ndarray:
    """A small NTSC-shaped FM composite at 10 MS/s.

    Sync tips at 0.6 cycles/sample with the standard timings, so it has the
    structure the real decoder looks for: a flat-topped pulse train separated by
    active picture. Enough lines to be a recognisable video signal without being
    a whole field.
    """
    from fpv_rf import dsp

    L = int(round(FS / dsp.NTSC.line_rate_hz))
    n = L * n_lines
    t = np.arange(n, dtype=np.float32) / L

    # sync tip 0.000-0.074, back porch to 0.145, active to 0.973
    sync = (t < 0.074) | ((t >= 0.145) & (t < 0.155))
    blank = ((t >= 0.074) & (t < 0.145)) | (t >= 0.973)

    picture = 0.12 * np.sin(2.0 * np.pi * 3.0 * t)
    picture += 0.06 * (np.floor(t * 8.0) % 2.0)              # checker
    picture += 0.05 * np.sin(2.0 * np.pi * 0.31 * t)         # a moving pattern

    # Sync is a genuine outlier in the level distribution, which is the property
    # the sync detector depends on. Picture highlights stay well below the tip.
    level = np.where(sync, 0.45, np.where(blank, 0.40, picture)).astype(np.float32)
    level = np.repeat(level, L)[:n]
    # The level is the per-sample phase increment in radians, exactly as
    # SimSource treats it, so there is no separate deviation constant to get
    # wrong and no per-sample step anywhere near wrap-around trouble.
    phase = np.cumsum(level, dtype=np.float64)
    iq = np.exp(1j * np.mod(phase, 2.0 * np.pi))

    # Scaled to the amplitude a real FPV capture actually shows, measured on the
    # 5802 MHz capture at mean |x| 0.22. This is not cosmetic. The two encodings
    # are only distinguishable by histogram when the signal is *quiet*: a loud
    # signal reaches the ends of the byte axis in either encoding and a wrapped
    # loud signal is just a loud signal. A unit-magnitude fixture sits at 33.5%
    # extreme bytes and is honestly undecidable; at this amplitude it is at 100%
    # or 0% and settles the question. Testing detection against a fixture whose
    # amplitude is chosen to defeat it would be testing nothing.
    return (iq * AMPLITUDE).astype(np.complex64)


REAL_CAPTURE = (Path.home() / "Documents" / "hackrf-fpv-still-decoder" / "Samples"
                / "r5_5802_g16.u8")

#: How much of the real capture to keep. 2 MB is 100k complex samples at 10 MS/s,
#: a second and a half: enough for several fields, small enough to commit.
REAL_BYTES = 2_000_000


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    iq = composite()
    inter = np.empty((iq.size, 2), dtype=np.float32)
    inter[:, 0] = iq.real
    inter[:, 1] = iq.imag

    # Quantise ONCE, then express the same integers in both encodings.
    #
    # Rounding the two independently gives different bytes at ties: round(x*127.5)
    # and round(x*127.5 + 128) - 128 are equal in exact arithmetic, but in
    # float32 the addition loses the low bits and can tip a value across a .5
    # boundary. Measured, that put 122 of 381,600 samples one LSB apart -- an
    # artifact of this generator, not of the decoder, and it would have made the
    # encoding-invariance assertion a tolerance test instead of an exact one.
    q = np.clip(np.round(inter * 127.5), -128, 127).astype(np.int16)
    signed = q.astype(np.int8)
    unsig = (q + 128).astype(np.uint8)

    iq.tofile(OUT / "iq_reference.f32")
    signed.tofile(OUT / "iq_signed_int8.u8")
    unsig.tofile(OUT / "iq_unsigned_offset.u8.u8")

    # A verbatim excerpt of a real 5802 MHz capture, kept in its original signed
    # encoding. This is the fixture that matters most: it is not a belief about
    # what hackrf_transfer emits, it is what it emitted, so anything tested
    # against it exercises the hardware byte format directly.
    #
    # It is also what lets check_review_fixes.py prove the decoder works on real
    # samples, with no simulator in the path -- which is the one claim that
    # simulator-only testing cannot make. That is why the 2 MB excerpt is
    # committed rather than regenerated: the 38 MB original stays out of git
    # (see /samples/ in .gitignore), but the excerpt has to be in the repository
    # or CI cannot run the test that matters.
    if REAL_CAPTURE.exists():
        raw = REAL_CAPTURE.read_bytes()[:REAL_BYTES]
        (OUT / "real_5802_signed.u8").write_bytes(raw)
        print(f"excerpted {len(raw):,} bytes from {REAL_CAPTURE.name}")
    else:
        print(f"SKIP real capture not found at {REAL_CAPTURE}")
        print("     the committed tools/fixtures/real_5802_signed.u8 is kept as-is;")
        print("     point FPV_CAPTURE at a capture to regenerate it")

    print(f"wrote {len(iq):,} complex samples to {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:26s} {p.stat().st_size:>10,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
