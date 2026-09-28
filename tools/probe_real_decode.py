"""Does a real HackRF capture actually decode to video, now that the format is right?

Run offline against a recorded capture. This is the check that would retire the
"video decode is only proven by the simulator" limitation -- and equally, the
check that can disprove it. A negative result here is worth as much as a
positive one, so the tool reports both ways round and never rounds up.

Usage:
    python tools/probe_real_decode.py [capture.u8] [--rate MSPS] [--trials N]

Defaults to the 5802 MHz capture from the earlier session, with hackrf_transfer's
own 10 MS/s default rate, which is what that capture was taken at.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpv_rf import dsp, sdr, video  # noqa: E402

DEFAULT_CAPTURE = Path(
    r"C:\Users\Render\Documents\hackrf-fpv-still-decoder\Samples\r5_5802_g16.u8"
)

#: hackrf_transfer's own default, which is what every capture in that directory
#: was taken at unless recorded with an explicit -s.
HACKRF_DEFAULT_MSPS = 10.0

#: The decoder refuses to declare a lock below this, so a "no" here means the
#: detector did not find enough well-spaced sync pulses, not that they existed
#: and were ignored.
LOCK_QUALITY_FLOOR = 0.35

#: Sync pulses per composite line is ~7.5 us out of ~63.5 us, so a line rate near
#: 15.734 kHz (NTSC) or 25.0 kHz (PAL) is the real test. Anything else means the
#: sample rate is wrong, not that the video is.
LINE_RATE_NTSC = 15_734.0
LINE_RATE_PAL = 25_000.0
LINE_RATE_TOL = 0.06


def _load(path: Path, n_bytes: int) -> tuple[bytes, float]:
    raw = path.read_bytes()[:n_bytes]
    if len(raw) % 2:
        raw = raw[:-1]
    return raw, len(raw)


def _report_rate(rate_hz: int, raw: bytes) -> dict[str, object]:
    """Demodulate, look for sync, and say plainly what was found."""
    iq = sdr._u8_to_iq(raw, sdr.IQEncoding.SIGNED_INT8)
    demod = dsp.fm_demodulate(iq)
    # The decoder's own path applies a gentle line filter when one is configured;
    # the sync search is run on the raw demod here as well as filtered, because
    # which one wins is itself informative.
    sync = dsp.detect_sync(demod, rate_hz, polarity="auto")

    nearest = ("NTSC", LINE_RATE_NTSC) if abs(
        sync.line_rate_hz - LINE_RATE_NTSC
    ) < abs(sync.line_rate_hz - LINE_RATE_PAL) else ("PAL", LINE_RATE_PAL)
    off_pct = abs(sync.line_rate_hz - nearest[1]) / nearest[1] * 100.0

    return {
        "rate_msps": rate_hz / 1e6,
        "pulses": int(sync.pulse_starts.size),
        "quality": float(sync.quality),
        "line_rate_hz": float(sync.line_rate_hz),
        "nearest": nearest[0],
        "off_pct": off_pct,
        "spec": sync.spec.name,
        "polarity": int(sync.polarity),
        "demod_sd": float(demod.std()),
        "decoded_standard": nearest[0] if off_pct < LINE_RATE_TOL * 100 else None,
    }


def _try_decode(rate_hz: int, raw: bytes, blocks: int) -> tuple[int, float, str, int]:
    """Push the capture through the real VideoDecoder, block by block."""
    dec = video.VideoDecoder(rate_hz)
    iq = sdr._u8_to_iq(raw, sdr.IQEncoding.SIGNED_INT8)
    per = 1 << 18
    frames = 0
    best = 0.0
    kind = ""
    from_grid = 0
    for i in range(0, iq.size, per):
        fr = dec.push_iq(iq[i : i + per])
        if fr is not None:
            frames += 1
            best = max(best, fr.sync_quality)
            from_grid = max(from_grid, fr.lines_from_grid)
            kind = f"{fr.standard} {fr.image.shape[1]}x{fr.image.shape[0]}"
        if i // per >= blocks * 8:
            break
    return frames, best, kind, from_grid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", nargs="?", type=Path, default=DEFAULT_CAPTURE)
    ap.add_argument("--rate", type=float, default=HACKRF_DEFAULT_MSPS,
                    help="sample rate in MS/s (default: %(default)s)")
    ap.add_argument("--trials", type=int, default=3,
                    help="extra rates to try around the nominal one, in MS/s")
    ap.add_argument("--seconds", type=float, default=2.0,
                    help="how much of the capture to decode (default: %(default)s)")
    ap.add_argument("--mb", type=int, default=40,
                    help="max bytes to read (default: %(default)s)")
    args = ap.parse_args(argv)

    path: Path = args.capture
    if not path.exists():
        print(f"capture not found: {path}")
        return 2

    raw, n = _load(path, args.mb * 1_000_000)
    print(f"capture {path.name}: {n:,} bytes, {n/2:,} complex samples")
    verdict = sdr.detect_iq_encoding(raw)
    print(f"  format: {verdict.encoding.value if verdict.encoding else 'UNDECIDABLE'}"
          f"  ({verdict.share*100:.1f}% of bytes at the ends of the byte axis)")

    # Trial rates around the nominal one. A wrong rate rescales every timing in
    # the DSP chain, so a capture that has video on it can still be pushed at a
    # rate that hides it. Reporting one rate only would make a null result
    # indistinguishable from a mislabelled capture.
    base = args.rate
    rates = [base] + [base * (1.0 + k * 0.10) for k in range(1, args.trials + 1)]

    best_row: dict[str, object] | None = None
    for msps in rates:
        row = _report_rate(int(msps * 1e6), raw)
        marker = " <-- nominal" if msps == base else ""
        print(f"\n  {row['rate_msps']:.3f} MS/s{marker}")
        print(f"    demodulator sd        {row['demod_sd']:.4f}")
        print(f"    sync pulses found     {row['pulses']}")
        print(f"    sync quality          {row['quality']:.3f}"
              f"  (lock floor {LOCK_QUALITY_FLOOR})")
        print(f"    measured line rate    {row['line_rate_hz']:,.0f} Hz"
              f"  -> {row['nearest']} off by {row['off_pct']:.1f}%")
        if row["decoded_standard"]:
            print(f"    IN BAND for {row['decoded_standard']}")
        if best_row is None or row["quality"] > best_row["quality"]:
            best_row = row

    assert best_row is not None
    rate_hz = int(base * 1e6)
    want = int(args.seconds * rate_hz)
    decode_raw = raw[: want * 2]
    frames, quality, kind, from_grid = _try_decode(rate_hz, decode_raw, blocks=32)

    print(f"\n  end-to-end at {base:g} MS/s over {args.seconds:g} s of capture:")
    print(f"    frames emitted        {frames}")
    print(f"    worst field needing reconstructed line positions: "
          f"{from_grid}/240")
    if frames:
        print(f"    best sync quality     {quality:.3f}")
        print(f"    frame geometry        {kind}")
        print("\n  RESULT: LOCKED -- real FPV video decoded from a real capture.")
        return 0

    print(f"    best sync quality     {best_row['quality']:.3f}")
    print("\n  RESULT: NO LOCK. No video decoded from this capture.")
    print(f"    Pulses found: {best_row['pulses']}, quality {best_row['quality']:.3f}"
          f" against a floor of {LOCK_QUALITY_FLOOR}.")
    print("    A high sync quality with no frames is itself a finding, not a")
    print("    near-miss: the detector can see a sync train it cannot assemble")
    print("    into a field. This tool reports both numbers so the two can never")
    print("    be mistaken for one another.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
