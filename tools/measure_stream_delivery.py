"""Measure what the streaming decode path actually delivers, on real IQ.

The counters in :class:`fpv_rf.video.DecodeStats` say *how much* arrived; this
says how it arrived. It runs the same two objects the app runs -- an
:class:`fpv_rf.sdr.IQSource` filling a ring, and a
:class:`fpv_rf.video.DecodeWorker` draining it on its own cadence -- and reports
delivery, loss, gaps by cause, picture rate, per-phase timings and lag.

Headless by construction: it imports no GUI module, so it runs over SSH, in CI
and on a machine with no display. It touches nothing but the source and the
worker, and it never writes to the capture.

Two honest caveats, both stated in the output rather than hidden:

* **The decode cadence is not the field rate.** :data:`fpv_rf.video.PREVIEW_HZ`
  is 30 Hz, so at most 30 pictures a second can be published no matter how fast
  the transmitter sends. A published rate below 30/s is a decoder that is missing
  its own cadence; a published rate of exactly 30/s with a 59.94 field rate is a
  decoder keeping up and choosing to skip every second field, which is what
  :data:`PREVIEW_HZ` asks for. Both are reported, the field rate alongside.
* **Without ``--realtime`` the source is not a clock.** The file is then read as
  fast as the disk and the decoder allows, so the stream is always "behind" in
  wall-clock terms and the picture rate measures decode throughput rather than
  delivery. Use ``--realtime`` to measure the live path.

Run:

    python tools/measure_stream_delivery.py --realtime --seconds 2

With no ``--file`` the usual capture search runs (``$FPV_CAPTURE``, then
``samples/``, then a shallow walk of the home directory -- see ``tools/_paths.py``).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv_rf import dsp, sdr, video  # noqa: E402
from _paths import CAPTURE  # noqa: E402


# ---------------------------------------------------------------------------
# Timing the phases
# ---------------------------------------------------------------------------


class Timers:
    """Per-call wall time for one phase, in milliseconds.

    The phases are measured by wrapping the calls the decoder actually makes
    rather than by reading the exponential averages in
    :class:`~fpv_rf.video.DecodeStats`: an average hides exactly what a
    measurement is for, which is the slow iteration. Wrapping is exact for the
    calls it wraps and says nothing about anything else -- in particular the
    rolling window is built inside ``fm``, and ``sync`` and ``raster`` each see
    that whole window, so the three phases together exceed the length of the
    block that produced them. That is the real cost, and it is why the window is
    42 ms rather than one field.
    """

    def __init__(self) -> None:
        self.ms: dict[str, list[float]] = {}

    def wrap(self, owner, name: str, label: str):
        """Time ``owner.name`` into ``label``; return what to put back."""
        original = getattr(owner, name)
        sink = self.ms.setdefault(label, [])

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                sink.append((time.perf_counter() - t0) * 1000.0)

        setattr(owner, name, timed)
        return original

    def restore(self, owner, name: str, original) -> None:
        setattr(owner, name, original)

    def report(self, label: str, count: int | None = None) -> str:
        """``mean p50 p95 p99 max n`` for one phase, in ms."""
        values = self.ms.get(label, [])
        if not values:
            return f"no calls recorded{'' if count is None else f' of {count}'}"
        a = np.asarray(values, dtype=np.float64)
        p50, p95, p99 = np.percentile(a, [50.0, 95.0, 99.0])
        return (f"{a.mean():7.2f} {p50:7.2f} {p95:7.2f} {p99:7.2f} {a.max():7.2f} "
                f"{a.size:6d}")


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def share(part: float, whole: float) -> str:
    """``part/whole`` as a percentage, or ``n/a`` when the denominator is zero."""
    return f"{100.0 * part / whole:6.2f}%" if whole else "   n/a"


def ms(value: float) -> str:
    return f"{value:8.2f}"


def spread(values: list[float], scale: float = 1000.0, unit: str = "ms") -> str:
    """``mean p50 p95 max`` of a list, scaled out of seconds."""
    if not values:
        return "no samples"
    a = np.asarray(values, dtype=np.float64) * scale
    p50, p95 = np.percentile(a, [50.0, 95.0])
    return (f"mean {a.mean():.2f} {unit}, p50 {p50:.2f}, p95 {p95:.2f}, "
            f"max {a.max():.2f} (n={a.size})")


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure(args) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"no capture at {path}")
        print("Set $FPV_CAPTURE to a hackrf_transfer recording, or pass --file.")
        return 2

    encoding = None
    if args.encoding != "detect":
        try:
            encoding = sdr.IQEncoding(args.encoding)
        except ValueError:
            print(f"unknown encoding {args.encoding!r}; expected one of "
                  + ", ".join(e.value for e in sdr.IQEncoding))
            return 2

    size = path.stat().st_size
    in_file = size // 2
    rate = int(args.sample_rate)
    stream_s = in_file / rate

    src = sdr.FileSource(path, sample_rate=rate, loop=False,
                         realtime=bool(args.realtime), encoding=encoding)
    worker = video.DecodeWorker(src, width=int(args.width))

    timers = Timers()
    deliveries: list[dict] = []
    published: list[float] = []

    # -- wrap the five calls the numbers are made of ------------------------
    real_drain = src.drain_block
    conv = timers.ms.setdefault("conv", [])

    def timed_drain(max_samples=300_000):
        t0 = time.perf_counter()
        try:
            block = real_drain(max_samples)
        finally:
            conv.append((time.perf_counter() - t0) * 1000.0)
        if not block.empty:
            deliveries.append({
                "first": int(block.first_sample),
                "last": int(block.last_sample),
                "samples": int(block.samples),
                "skipped": int(block.skipped_samples),
                "seam": block.seam,
                "reset": block.reset_reason,
                "at": time.perf_counter(),
            })
        return block

    real_step = worker.step
    steps = timers.ms.setdefault("total", [])
    last_published_at = [0.0]

    def timed_step():
        t0 = time.perf_counter()
        try:
            return real_step()
        finally:
            steps.append((time.perf_counter() - t0) * 1000.0)
            # ``_published_at`` only moves when a picture reaches the panel, so
            # the publication times fall out of it rather than out of the
            # return value, which the worker thread keeps to itself.
            at = worker._published_at
            if at and at != last_published_at[0]:
                last_published_at[0] = at
                published.append(at)

    src.drain_block = timed_drain
    worker.step = timed_step
    restore_fm = timers.wrap(dsp, "fm_demodulate", "fm")
    restore_sync = timers.wrap(dsp, "detect_sync", "sync")
    # On the *class*, not on the instance: ``VideoDecoder.reset()`` builds a new
    # Rasteriser, so an instance attribute would be thrown away by the first
    # segment boundary and the phase would then read as never called.
    restore_raster = timers.wrap(dsp.Rasteriser, "push", "raster")

    t0 = time.perf_counter()
    src.start()
    worker.start()
    time.sleep(float(args.seconds))
    worker.stop()
    src.stop()
    elapsed = time.perf_counter() - t0

    timers.restore(dsp, "fm_demodulate", restore_fm)
    timers.restore(dsp, "detect_sync", restore_sync)
    timers.restore(dsp.Rasteriser, "push", restore_raster)
    src.drain_block = real_drain
    worker.step = real_step

    st = worker.stats
    overrun = int(src.ring.dropped_bytes) // 2

    # -- gaps, measured off the hand-over records ---------------------------
    gaps: list[tuple[int, str]] = []
    overlaps: list[tuple[int, str]] = []
    cursor = None
    for d in deliveries:
        if cursor is not None and d["first"] > cursor:
            gaps.append((d["first"] - cursor, d["seam"] or "gap"))
        elif cursor is not None and d["first"] < cursor:
            overlaps.append((cursor - d["first"], d["seam"] or "overlap"))
        cursor = max(cursor or 0, d["last"])
    skipped_drains = [(d["skipped"], d["seam"] or d["reset"] or "cap")
                      for d in deliveries if d["skipped"]]

    # -- lag: how far behind the stream the newest sample is ----------------
    lag: list[float] = []
    for d in deliveries:
        behind = (d["at"] - t0) - d["last"] / rate
        lag.append(behind)
    cadence = 1.0 / video.PREVIEW_HZ

    # -- report -------------------------------------------------------------
    print("Stream delivery measurement")
    print(f"  capture            {path.name} ({size:,} bytes, "
          f"{in_file:,} samples, {stream_s:.2f} s at {rate/1e6:g} MS/s)")
    print(f"  encoding           {src.encoding.value} ({src._encoding_reason})")
    print(f"  source             {src.describe()}, "
          f"{'realtime-paced' if args.realtime else 'read as fast as it goes'}"
          f", loop off")
    print(f"  decode             {args.width} px wide, "
          f"{video.WINDOW_S * 1000:.0f} ms window ({video.DEFAULT_WINDOW:,} samples), "
          f"{video.PREVIEW_HZ:.0f} Hz cadence, cap "
          f"{int(rate * video.WINDOW_S):,} samples")
    print(f"  window             measured {elapsed:.2f} s, "
          f"{len(deliveries):,} hand-over(s), {len(steps):,} decode iteration(s)")

    print("\n  Delivery")
    print(f"    written           {st.iq_written:>13,} samples  "
          f"{share(st.iq_written, in_file)} of the capture")
    print(f"    received          {st.iq_delivered:>13,} samples  "
          f"{share(st.iq_delivered, st.iq_written)} of written")
    print(f"    accepted          {st.iq_accepted:>13,} samples  "
          f"{share(st.iq_accepted, st.iq_delivered)} of received")
    print(f"    rejected          {st.iq_rejected:>13,} samples  "
          f"{share(st.iq_rejected, st.iq_delivered)} of received "
          f"(delivered, refused: a seam, a stale epoch, a decode that raised)")
    print(f"    skipped, capped   {st.capped_skip_samples:>13,} samples  "
          f"{share(st.capped_skip_samples, st.iq_written)} of written "
          f"({st.capped_skips} skip(s) -- policy: a backlog longer than the cap)")
    print(f"    skipped, overrun  {overrun:>13,} samples  "
          f"{share(overrun, st.iq_written)} of written "
          f"(loss: the ring destroyed it unread)")
    ok = st.iq_delivered == st.iq_accepted + st.iq_rejected
    print(f"    identity          delivered == accepted + rejected: "
          f"{'holds' if ok else 'BROKEN'} "
          f"({st.iq_delivered:,} == {st.iq_accepted:,} + {st.iq_rejected:,})")

    print("\n  Continuity")
    causes = {k: int(v) for k, v in sorted(st.gap_causes.items())}
    W = video.DecodeWorker                       # the eight fixed cause names
    for name in (W.GAP_FIRST, W.GAP_GAP, W.GAP_OVERLAP, W.GAP_STREAM, W.GAP_TUNE,
                 W.GAP_RANGE, W.GAP_UNPLACED, W.GAP_TINY):
        print(f"    {name:<14} {causes.get(name, 0):6d}")
    print(f"    {'discontinuities':<14} {st.discontinuities:6d} "
          f"(a 'first' is the start of a segment, not a break in one)")
    print(f"    {'segments':<14} {st.segments:6d} uninterrupted run(s)")
    print(f"    {'missing':<14} {st.missing_samples:6d} sample(s) that never arrived, "
          f"= {100.0 * st.missing_samples / max(1, st.iq_written):.3f}% of written")
    if gaps:
        top = sorted(gaps, reverse=True)[:3]
        print("    largest gaps      "
              + ", ".join(f"{n:,} ({1000.0 * n / rate:.1f} ms, {why})"
                          for n, why in top))
    else:
        print("    largest gaps      none: every hand-over was adjacent")
    if overlaps:
        top = sorted(overlaps, reverse=True)[:3]
        print("    overlaps          "
              + ", ".join(f"{n:,} ({why})" for n, why in top))
    if skipped_drains:
        top = sorted(skipped_drains, reverse=True)[:3]
        print("    capped skips      "
              + ", ".join(f"{n:,} ({why})" for n, why in top))

    print("\n  Resets and errors")
    print(f"    {'discontinuities':<18} {st.discontinuities:6d}")
    print(f"    {'decoder resets':<18} {st.fm_resets:6d} "
          f"(a segment's first block, and any decode that raised)")
    print(f"    {'boundary steps':<18} {st.boundary_steps:6d} "
          f"proven-adjacent joins")
    print(f"    {'refused blocks':<18} {st.seam_blocks:6d} "
          f"({st.tune_seams_internal} from a retune)")
    print(f"    {'ring overruns':<18} {overrun:6,d} sample(s) destroyed unread")
    print(f"    {'decode errors':<18} {st.decode_errors:6d}"
          + (f"  last: {st.last_decode_error}" if st.last_decode_error else ""))
    print(f"    {'read errors':<18} {src.stats.read_errors:6d} "
          f"({src.stats.restarts} source restart(s))")
    if src.stats.last_error:
        print(f"    {'source error':<18} {src.stats.last_error}")

    fields = (video.PAL_FIELD_INTERVAL_S
              if st.standard == "PAL" else video.NTSC_FIELD_INTERVAL_S)
    pub_rate = st.frames / elapsed
    uniq_rate = st.unique_frames / elapsed
    print("\n  Pictures")
    print(f"    published         {st.frames:6d} in {elapsed:.2f}s = "
          f"{pub_rate:5.1f}/s ({100.0 * pub_rate / video.PREVIEW_HZ:.0f}% of the "
          f"{video.PREVIEW_HZ:.0f} Hz cadence)")
    print(f"    unique            {st.unique_frames:6d} in {elapsed:.2f}s = "
          f"{uniq_rate:5.1f}/s ({100.0 * uniq_rate / video.PREVIEW_HZ:.0f}% of cadence)")
    print(f"    duplicates        {st.dup_frames:6d} re-renders of a field already "
          f"shown, counted and not published")
    print(f"    against the wire  {1.0 / fields:.2f} fields/s in {st.standard or 'NTSC?'}"
          f" -- the cadence is below that by design, so a decoder keeping up "
          f"skips every second field")
    print(f"    missed fields     {st.fields_missed:6d} "
          f"(miss rate {st.miss_rate():.2f})")
    print(f"    sync accepted     {st.sync_accepted:6d} of {st.sync_attempts:6d} "
          f"attempts ({st.sync_quality:.2f} quality, {st.line_rate_hz:.0f} lines/s)")
    print(f"    grid rows         {st.grid_rows:6d} line(s) placed by fitting "
          f"rather than by a detected sync pulse")
    print(f"    newest picture    [{st.frame_interval[0]:,}, "
          f"{st.frame_interval[1]:,}), age {worker.frame_age_s():.2f}s, "
          f"locked {worker.locked}")

    print("\n  Phase timings (ms)")
    print(f"    {'phase':<10} {'mean':>7} {'p50':>7} {'p95':>7} {'p99':>7} "
          f"{'max':>7} {'n':>6}")
    for label in ("conv", "fm", "sync", "raster", "total"):
        print(f"    {label:<10} {timers.report(label)}")
    print(f"    the decode cadence is {1000.0 * cadence:.1f} ms, so 'total' above "
          f"is what fits in it: "
          f"{'keeping up' if np.median(timers.ms.get('total', [0.0])) < 1000.0 * cadence else 'NOT keeping up'}")

    print("\n  Lag and intervals")
    print(f"    stream lag        {spread(lag)} -- how long after the moment it "
          f"arrived the newest sample was decoded")
    print(f"    picture age       {worker.frame_age_s():.2f}s at the end "
          f"(a live picture is under {video.STALE_FRAME_S:g}s)")
    if len(published) > 1:
        print(f"    publish interval  {spread(np.diff(published).tolist())}")
    if len(deliveries) > 1:
        print(f"    delivery interval {spread(np.diff([d['at'] for d in deliveries]).tolist())}")
        sizes = np.asarray([d["samples"] for d in deliveries], dtype=np.float64)
        print(f"    chunk size        mean {sizes.mean():,.0f} samples, "
              f"min {sizes.min():,.0f}, max {sizes.max():,.0f}, "
              f"empty hand-overs {len(steps) - len(deliveries):,}")
    print(f"    worker's EWMA     drain {st.drain_ms:.2f} ms, decode "
          f"{st.decode_ms:.2f} ms, cycle {st.cycle_ms:.2f} ms")

    print("\n  Note: published rates are bounded by the "
          f"{video.PREVIEW_HZ:.0f} Hz cadence, not by the "
          f"{1.0 / fields:.2f} fields/s on the wire; a rate at the cadence with "
          "a high unique rate is a decoder keeping up.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure IQ -> picture delivery on a real capture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--file", default=str(CAPTURE),
                    help="hackrf_transfer recording to replay (.u8, interleaved I/Q)")
    ap.add_argument("--sample-rate", type=int, default=sdr.DEFAULT_SAMPLE_RATE,
                    help="sample rate the capture was made at")
    ap.add_argument("--encoding", default="detect",
                    choices=["detect"] + [e.value for e in sdr.IQEncoding],
                    help="byte encoding of the capture; 'detect' judges it from the data")
    ap.add_argument("--realtime", action="store_true",
                    help="pace the replay to the sample rate, as the radio would; "
                         "without it the file is read as fast as it goes and the "
                         "numbers measure decode throughput instead")
    ap.add_argument("--seconds", type=float, default=2.0,
                    help="how long to run the measurement")
    ap.add_argument("--width", type=int, default=160,
                    help="decode width in pixels")
    return measure(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())