"""Continuity of the streamed IQ -> decode path: adjacency, identity, accounting.

A streaming decoder is only allowed to carry demodulator state across a block
boundary if it can *prove* the two blocks are adjacent. Everything here is about
that proof, and about what happens when it cannot be given. The cases are built
from the delivery primitive (``IQSource.drain_block``) rather than from the live
thread, because a dropped USB packet, a retune and a replayed file passing its
end are timing accidents that cannot be provoked on demand -- but a gap of
exactly one sample, an internal seam or a stale generation can be.

What is pinned:

  1. **Adjacent-chunk equivalence.** The same stream decoded in irregular chunks
     with every boundary proven adjacent is *bit-identical* to the one-shot
     decode: the same float32 demodulator output and the same pixels. An
     uninterrupted segment of N IQ samples yields exactly N-1 outputs, and the
     boundary steps are the N-1 - (blocks - 1) missing ones -- computed from the
     sample before the join, which is the one thing a per-block discriminator
     cannot see for itself.
  2. **Gap injection.** One missing sample and a larger one, each produced the
     only way a gap can really happen in this design (a drain reset that
     discards a known number of samples). No picture is published across the
     omission, and two short fragments on either side of it are not combined
     into a field.
  3. **Identity boundaries.** A retune, a block that straddles one internally, a
     transfer-process restart at an unchanged frequency, a replayed file passing
     its end, a drain reset, and a reset racing a decode in flight. Nothing is
     demodulated across any of them, each costs exactly one field, and the block
     after the break starts a fresh segment rather than another gap.
  4. **Window capacity.** 42 ms is 2.5 NTSC fields and 2.1 PAL fields, so both
     standards must decode out of a single 42 ms segment *in isolation* -- at
     several phases, because a window that only worked at phase zero would be a
     coincidence.
  5. **Burst adjacency.** A burst of back-to-back writes drained one chunk at a
     time is still one segment, with no gap invented between the chunks.
  6. **Freshness.** At end of file, on an empty drain and on a stalled source,
     the picture on screen does not gain a new timestamp: the rolling window
     still holds the field it last drew, and re-publishing it would make a dead
     transmitter look live.
  7. **Accounting identities.** ``delivered == accepted + rejected`` always, and
     every written sample lands in exactly one bucket. The buckets overlap in one
     place -- a drain skip shows up both as a capped skip and as the gap it
     leaves -- so the tool prints the decomposition and checks the disjoint form
     ``written == accepted + rejected + missing`` as well as the five-term form
     on the scenarios where the buckets cannot overlap.
  8. **Snapshot neutrality.** Taking ring snapshots for the diagnostic dump must
     not move the drain cursor, consume anything or improve the loss accounting;
     the delivery a worker sees is the same with and without.
  9. **Overlap.** A delivery that reaches back into samples the decoder already
     holds has its repeated head removed *before* the decoder sees it, and the
     demodulator output is bit-identical to a bench that was never rewound. A
     block wholly inside the repeat decodes nothing and costs nothing, and a
     remainder below the minimum is refused rather than demodulated as a
     window that is mostly boundary step. A repeat reaching into a span that was
     refused or poisoned rather than decoded is still trimmed -- the new samples
     are accepted, because refusing them would discard a field over a hole
     already counted -- but that run continues nothing, so it starts a fresh
     segment: no boundary step across the span, and the demodulator's origin is
     the new run's own first sample.
 10. **Whole-sample discipline.** A producer that hands over an odd number of
     bytes -- which a pipe read may do at any time -- does not leave the ring
     holding half a sample: the dangling byte is carried into the *next* read of
     the *same* producer, and dropped when the producer is replaced, because a
     replacement's stream starts on a sample boundary of its own.
 11. **An oversized write is not continuity.** A write too large for the ring
     invalidates what the ring already held, and the bytes it destroyed are
     counted as loss rather than left behind as "the oldest data still in the
     buffer", presented at positions contiguous with the new tail.
 12. **Seams survive a wrap.** The identity in force at the head of the ring is
     an *anchor* and is not pruned when it falls below the floor, so a seam
     recorded several ring-lengths earlier is still reported after the buffer
     has turned over several times.

Run:  python tools/check_stream_continuity.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv_rf import diag, dsp, sdr, video  # noqa: E402
from _paths import CAPTURE  # noqa: E402

OUT = Path("_validate_out")
FAILURES: list[str] = []
SKIPPED: list[str] = []

#: The reference rate. Everything below is quoted in samples and the durations
#: are derived from it, so the numbers mean the same thing at any rate.
FS = 10_000_000

#: 42 ms of stream: the delivery cap and the demodulator window alike.
CHUNK = int(round(FS * video.WINDOW_S))

#: Simulator samples each case family consumes from the front of one shared
#: array. The identity cases are the hungriest: they lock, then spend six 42 ms
#: chunks proving that a break costs one field and not one field per block.
NTSC_SAMPLES = 4_400_000
PAL_SAMPLES = 800_000


def check(name: str, good: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if good else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not good:
        FAILURES.append(name)
    return bool(good)


def skip(name: str, why: str) -> None:
    print(f"  skip  {name} -- {why}")
    SKIPPED.append(f"{name}: {why}")


# ---------------------------------------------------------------------------
# Signal: real composite video from the simulator, taken through the real drain
# ---------------------------------------------------------------------------


def composite_iq(n_samples: int, *, ntsc: bool = True, timeout: float = 90.0) -> np.ndarray:
    """Composite-video IQ from the simulator, captured once and reused.

    Taken with :meth:`IQSource.drain_block` rather than generated directly, so
    the samples used here are the ones the app's own delivery path produces --
    quantised to the simulator's byte encoding and converted back -- and not a
    private copy of the generator's arithmetic.
    """
    src = sdr.SimSource(video=True, ntsc=ntsc)
    src.start()
    parts: list[np.ndarray] = []
    got = 0
    end = time.monotonic() + timeout
    while got < n_samples and time.monotonic() < end:
        block = src.drain_block(1 << 19)
        if not block.empty:
            parts.append(block.iq)
            got += block.samples
    src.stop()
    if not parts:
        return np.zeros(0, dtype=np.complex64)
    return np.concatenate(parts)[:n_samples]


def iq_to_u8(iq: np.ndarray) -> bytes:
    """Interleaved bytes the simulator's own ring would have produced.

    The inverse of the read path for :data:`sdr.IQEncoding.UNSIGNED_OFFSET`, so
    what a case writes into a ring and what a case decodes are the same values
    the simulator thread would have delivered.
    """
    raw = np.empty((iq.size, 2), dtype=np.float32)
    raw[:, 0] = iq.real
    raw[:, 1] = iq.imag
    v = np.clip(np.rint(raw * 127.0) + 128.0, 1.0, 255.0).astype(np.uint8)
    out = np.empty(iq.size * 2, dtype=np.uint8)
    out[0::2] = v[:, 0]
    out[1::2] = v[:, 1]
    return out.tobytes()


class Bench:
    """A worker over a ring this script fills, so every break is chosen.

    The source is never started: nothing writes to the ring but the cases below,
    which is what makes a one-sample gap or a mid-block retune reproducible
    instead of a matter of luck.
    """

    def __init__(self, *, width: int = 160) -> None:
        self.src = sdr.SimSource(video=True)
        self.w = video.DecodeWorker(self.src, width=width)
        self.tune = 0
        self.stream = self.src._begin_stream()

    def feed(self, iq: np.ndarray) -> int:
        """Write one run of samples under the current identities."""
        return self.src.ring.write(iq_to_u8(iq), self.tune, self.stream)

    def step(self) -> bool:
        return self.w.step()

    @property
    def st(self):
        return self.w.stats


def run_to_lock(bench: Bench, iq: np.ndarray, blocks: int = 3) -> int:
    """Feed ``blocks`` 42 ms chunks and return how many pictures were published."""
    before = bench.st.frames
    pos = 0
    for _ in range(blocks):
        seg = iq[pos : pos + CHUNK]
        pos += seg.size
        if seg.size == 0:
            break
        bench.feed(seg)
        bench.step()
    return bench.st.frames - before


#: The counters the identity cases move, so a delta is one dict rather than a
#: tuple of nine ``before`` variables. Read together because the interesting
#: claim is always a *combination*: a refusal that costs a field but publishes
#: nothing and starts a new segment.
_COUNTERS = (
    "frames", "segments", "discontinuities", "seam_blocks", "tune_seams",
    "tune_seams_internal", "iq_delivered", "iq_accepted", "iq_rejected",
    "missing_samples", "capped_skip_samples", "boundary_steps",
)
_CAUSES = ("tune", "stream", "gap", "overlap", "first", "tiny", "unplaced", "range")


def mark(st) -> dict:
    """Every counter an identity assertion might want, in one snapshot."""
    out = {k: int(getattr(st, k)) for k in _COUNTERS}
    out.update({f"cause:{c}": int(st.gap_causes.get(c, 0)) for c in _CAUSES})
    return out


def delta(before: dict, st) -> dict:
    """How each of those counters moved since ``before``."""
    now = mark(st)
    return {k: now[k] - before[k] for k in now}


def describe(d: dict) -> str:
    return ", ".join(f"{k} {v:+d}" for k, v in d.items() if v)


# ---------------------------------------------------------------------------
# 1. adjacent-chunk equivalence
# ---------------------------------------------------------------------------


def case_equivalence(iq: np.ndarray, *, tag: str = "") -> None:
    if tag:
        print(f"\n1{tag}. adjacent chunks decode identically to the one-shot decode")
    else:
        print("\n1. adjacent chunks decode identically to the one-shot decode")
    n = int(iq.size)
    window = video.DEFAULT_WINDOW
    reference = dsp.fm_demodulate(iq)

    mono = video.VideoDecoder(FS, width=160, window=window)
    frame_mono = mono.push_iq(iq, continuous=False, first_sample=0)
    check(
        "the one-shot decode of the whole stream gives N-1 demodulator outputs",
        mono.fm_samples == n - 1 and mono.boundary_steps == 0,
        f"fm={mono.fm_samples} of {n} IQ, boundary steps={mono.boundary_steps}",
    )

    # Irregular chunk sizes, deliberately not a divisor of anything: no two of
    # them the same, so a decoder that assumed a fixed cadence would drift.
    rng = np.random.default_rng(11)
    sizes: list[int] = []
    left = n
    while left > 0:
        take = min(int(rng.integers(1_000, 250_000)), left)
        sizes.append(take)
        left -= take

    dec = video.VideoDecoder(FS, width=160, window=window)
    stitched: list[np.ndarray] = []
    filled = 0                      # next demodulator index not yet collected
    frames = []
    loose = []                      # chunks whose window lost output already taken
    pos = 0
    for take in sizes:
        seg = iq[pos : pos + take]
        start = pos
        pos += take
        frame = dec.push_iq(seg, continuous=len(stitched) > 0, first_sample=start)
        origin = dec.demod_origin
        buf = dec._demod
        if not origin <= filled <= origin + buf.size:
            loose.append((start, take, origin, filled, int(buf.size)))
        stitched.append(buf[filled - origin :].copy())
        filled = origin + buf.size
        if frame is not None:
            frames.append(frame)
    joined = np.concatenate(stitched)

    check(
        "every chunk's output is contiguous with the last one's collected",
        not loose,
        f"{len(sizes)} chunks, {len(loose)} loose ({loose[:2]})" if loose
        else f"{len(sizes)} chunks, all contiguous",
    )
    check(
        "the demodulator output is bit-identical to the one-shot decode",
        joined.size == n - 1 and np.array_equal(joined.view(np.uint32), reference.view(np.uint32)),
        f"{joined.size} of {n - 1} outputs, "
        f"{int(np.count_nonzero(joined != reference))} value(s) differ"
        + (f", max {np.abs(joined.astype(np.float64) - reference).max():.3e}"
           if joined.size == reference.size else ""),
    )
    check(
        "N-1 holds across the chunking, and every boundary step was computed",
        dec.fm_samples == n - 1 and dec.boundary_steps == len(sizes) - 1
        and dec.fm_resets == 0,
        f"fm={dec.fm_samples}, boundary steps={dec.boundary_steps} of "
        f"{len(sizes) - 1}, decoder resets={dec.fm_resets}",
    )

    # The picture checks need a field, which a capture of noise will not have.
    if not frames:
        skip(
            f"picture comparison{(' on ' + tag) if tag else ''}",
            "no field in this stream to compare (the signal did not lock)",
        )
        return
    check(
        f"the chunked decode published {len(frames)} picture(s)",
        len(frames) >= 2,
        f"{len(frames)} of {len(sizes)} pushes rendered",
    )
    if frame_mono is None:
        skip("the last picture against the one-shot picture", "the one-shot decode rendered nothing")
        return
    last = frames[-1]
    same_interval = (last.iq_first, last.iq_last) == (frame_mono.iq_first, frame_mono.iq_last)
    check(
        "the last picture is drawn from the same stream interval",
        same_interval,
        f"[{last.iq_first}, {last.iq_last}) vs "
        f"[{frame_mono.iq_first}, {frame_mono.iq_last})",
    )
    diff = np.abs(last.image.astype(np.int32) - frame_mono.image.astype(np.int32))
    check(
        "and it is pixel-identical to the one-shot picture",
        diff.max() == 0,
        f"{int(np.count_nonzero(diff))} of {diff.size} pixels differ, "
        f"worst {int(diff.max())} levels",
    )


def case_equivalence_real_capture() -> None:
    print("\n1b. the same equivalence on the real capture, when one is present")
    path = Path(CAPTURE)
    if not path.exists():
        skip("real-capture equivalence", f"no capture at {path}")
        return
    raw = np.fromfile(path, dtype=np.uint8, count=2 * 2_000_000)
    iq = sdr.u8_to_iq(raw, sdr.IQEncoding.SIGNED_INT8)
    case_equivalence(iq, tag="b")


# ---------------------------------------------------------------------------
# 2. gap injection
# ---------------------------------------------------------------------------


def case_gaps(iq: np.ndarray) -> None:
    print("\n2. a gap of one sample, and a gap of many, are both visible")
    bench = Bench()
    st = bench.st
    run_to_lock(bench, iq, blocks=3)
    check("the stream is locked before the gaps start", st.frames > 0,
          f"{st.frames} picture(s), sync {st.sync_quality:.2f} {st.standard}")

    # (a) Exactly one sample lost. The only way to express "a sample is missing"
    # in this design is for the cursor to move past it, so that is how it is
    # done: one sample is written, then discarded without being handed over.
    before = mark(st)
    end_before = bench.src.ring.write_position() // 2
    bench.feed(iq[end_before : end_before + 1])
    bench.src.reset_drain("test")
    check(
        "the single sample really was written and then discarded unread",
        bench.src.ring.write_position() // 2 - end_before == 1,
        f"1 sample written at {end_before} and dropped by the reset",
    )
    pos = end_before + 1
    bench.feed(iq[pos : pos + CHUNK])
    bench.step()
    d = delta(before, st)
    gap_at = end_before
    interval = st.frame_interval
    check(
        "a one-sample gap is counted as one sample, under its own cause",
        d["missing_samples"] == 1 and d["cause:gap"] == 1,
        describe(d),
    )
    check(
        "the decoder started a new segment rather than stepping across it",
        d["segments"] == 1 and d["boundary_steps"] == 0,
        describe(d),
    )
    check(
        "and the gap is a discontinuity, like every other break",
        d["discontinuities"] == 1,
        f"{d['discontinuities']} discontinuity, gap_causes={st.gap_causes}",
    )
    check(
        "no picture was published out of samples that span the omission",
        interval[0] > gap_at,
        f"newest picture [{interval[0]}, {interval[1]}), omission at {gap_at}",
    )
    check(
        "the picture did come back after it",
        st.frames > before["frames"],
        f"{before['frames']} -> {st.frames} picture(s)",
    )

    # (b) A larger gap, from a reset that discards a known number of samples.
    before = mark(st)
    end = bench.src.ring.write_position() // 2
    bench.feed(iq[end : end + 250_000])
    bench.src.reset_drain("test")
    end2 = bench.src.ring.write_position() // 2
    bench.feed(iq[end2 : end2 + CHUNK])
    bench.step()
    d = delta(before, st)
    check(
        "a 250,000-sample omission is counted as 250,000 samples",
        d["missing_samples"] == 250_000,
        describe(d),
    )

    # (c) Short fragments on either side of a boundary must not be combined into
    # a field: the rolling window must not still be holding the old stream.
    size_before = int(bench.w.decoder._demod.size)
    bench.w.reset()
    before = mark(st)
    seg_a = iq[end2 + CHUNK : end2 + CHUNK + 2_000]
    seg_b = iq[end2 + CHUNK + 2_000 : end2 + CHUNK + 4_000]
    bench.feed(seg_a)
    bench.step()
    bench.feed(seg_b)
    bench.step()
    window = int(bench.w.decoder._demod.size)
    d = delta(before, st)
    check(
        "two short fragments after a reset are not appended to the old window",
        window == 3_999 and size_before >= CHUNK - 1,
        f"window was {size_before} samples, now {window} after two 2,000-sample "
        f"fragments",
    )
    check(
        "and the two fragments adjacent to each other are joined, exactly once",
        d["boundary_steps"] == 1,
        describe(d),
    )
    check(
        "four thousand samples cannot make a field",
        d["frames"] == 0,
        f"{d['frames']} picture(s) from 3,999 samples",
    )


# ---------------------------------------------------------------------------
# 3. identity boundaries
# ---------------------------------------------------------------------------


def case_identities(iq: np.ndarray) -> None:
    print("\n3. every change of identity is visible and handled differently")
    bench = Bench()
    st = bench.st
    run_to_lock(bench, iq, blocks=2)
    check("locked to start with", st.frames > 0,
          f"{st.frames} picture(s), sync {st.sync_quality:.2f} {st.standard}")
    pos = CHUNK * 2

    def one_block() -> bool:
        """Write one 42 ms chunk under the current identities, then drain it.

        A drain hands over at most the cap, so a second chunk is what makes a
        *following* delivery visible: without one there is simply nothing to
        deliver and every counter would stay still.
        """
        nonlocal pos
        bench.feed(iq[pos : pos + CHUNK])
        pos += CHUNK
        return bench.step()

    # (a) A retune. Nothing after it may be demodulated as a continuation of
    # what came before -- different frequency, and the discriminator's
    # predecessor is a sample of the old one -- so the block is refused and the
    # field it held is lost. (Counted as ``tune_seams_internal``, which is
    # narrower than the event: ``_join`` treats "not the tuning I am holding" and
    # "a tuning change inside this block" as one kind of break.)
    bench.tune = 5
    before = mark(st)
    published = one_block()
    d = delta(before, st)
    check(
        "a retune is refused rather than demodulated across",
        not published and d["iq_rejected"] == CHUNK and d["iq_accepted"] == 0
        and d["cause:tune"] == 1 and d["discontinuities"] == 1,
        describe(d),
    )
    # ... and it costs exactly one field, because the block after it is accepted
    # as the start of a new segment rather than refused again.
    before = mark(st)
    published = one_block()
    d = delta(before, st)
    check(
        "the block after the retune starts a new segment instead of another gap",
        published and d["segments"] == 1 and d["iq_rejected"] == 0
        and d["iq_accepted"] == CHUNK and d["cause:tune"] == 0,
        describe(d),
    )

    # (b) A block that holds two tunings *internally*. Also refused: its halves
    # are each fine and together they are one picture out of two different
    # pictures.
    before = mark(st)
    bench.src.ring.write(iq_to_u8(iq[pos : pos + CHUNK // 2]), 5, bench.stream)
    bench.tune = 6
    bench.src.ring.write(iq_to_u8(iq[pos + CHUNK // 2 : pos + CHUNK]), 6, bench.stream)
    pos += CHUNK
    published = bench.step()
    d = delta(before, st)
    check(
        "a block straddling a retune internally is refused outright",
        not published and d["seam_blocks"] == 1 and d["tune_seams"] == 1
        and d["tune_seams_internal"] == 1,
        describe(d),
    )
    check(
        "and every sample of it is accounted as rejected",
        d["iq_rejected"] == CHUNK and d["iq_accepted"] == 0,
        f"{d['iq_rejected']} of {CHUNK} samples rejected",
    )
    before = mark(st)
    published = one_block()
    d = delta(before, st)
    check(
        "and the one after it is accepted as a new segment, not refused again",
        published and d["segments"] == 1 and d["iq_rejected"] == 0,
        describe(d),
    )

    # (c) A restart at the *same* frequency: the tuning id is unchanged, so only
    # the stream identity can tell it.
    before = mark(st)
    bench.stream = bench.src._begin_stream()
    published = one_block()
    d = delta(before, st)
    check(
        "a replaced producer at an unchanged frequency is recognised",
        not published and d["cause:stream"] == 1 and d["seam_blocks"] == 1,
        describe(d),
    )
    check(
        "and it is refused as the mixture it is, not counted as a retune",
        d["iq_rejected"] == CHUNK and d["tune_seams"] == 0,
        f"{d['iq_rejected']} sample(s) rejected, {d['tune_seams']} retune seam(s)",
    )
    before = mark(st)
    published = one_block()
    d = delta(before, st)
    check(
        "and the next block recovers from it as well",
        published and d["segments"] == 1 and d["iq_rejected"] == 0,
        describe(d),
    )

    # (d) A drain reset labels the next delivery, and labels exactly one.
    before = mark(st)
    bench.feed(iq[pos : pos + CHUNK])
    pos += CHUNK
    bench.src.reset_drain("retune")
    labelled = bench.src.drain_block(CHUNK)
    check(
        "a drain reset's reason travels on the next delivery and only that one",
        labelled.reset_reason == "retune" and labelled.empty
        and bench.src.drain_block(CHUNK).reset_reason == "",
        f"first={labelled.reset_reason!r} ({labelled.samples} samples), next="
        f"{bench.src.drain_block(CHUNK).reset_reason!r}",
    )
    # ... and the samples it discarded are a real gap, not a silence: they were
    # written, never delivered, and never accepted either.
    one_block()
    d = delta(before, st)
    check(
        "what the reset discarded is counted as missing, not as nothing",
        d["missing_samples"] == CHUNK and d["iq_accepted"] == CHUNK,
        describe(d),
    )
    check(
        "and counted as a gap rather than as a policy skip or a loss",
        d["cause:gap"] == 1 and d["capped_skip_samples"] == 0
        and d["iq_rejected"] == 0,
        f"capped {d['capped_skip_samples']} sample(s), rejected "
        f"{d['iq_rejected']}, overrun {bench.src.ring.dropped_bytes // 2}",
    )

    # (e) A reset racing a decode in flight. The generation is bumped before the
    # decoder is touched, so the decode that is already running throws its result
    # away rather than publishing a picture of the frequency just left.
    class Slow:
        def __init__(self, real):
            self._real = real

        def reset(self):
            self._real.reset()

        def push_iq(self, *a, **k):
            time.sleep(0.30)
            return self._real.push_iq(*a, **k)

        def __getattr__(self, name):
            return getattr(self._real, name)

    bench2 = Bench()
    bench2.w.decoder = Slow(bench2.w.decoder)
    run_to_lock(bench2, iq, blocks=2)
    frames_before = bench2.st.frames
    before2 = mark(bench2.st)
    bench2.feed(iq[0:CHUNK])
    outcome: dict = {}

    def decode():
        outcome["published"] = bench2.step()

    thread = threading.Thread(target=decode)
    thread.start()
    time.sleep(0.08)                 # inside the slow push, before it returns
    bench2.w.reset()
    thread.join(timeout=5.0)
    check(
        "a reset landing mid-decode publishes nothing from the old frequency",
        outcome.get("published") is False and bench2.st.frames == frames_before
        and bench2.w.latest() is None,
        f"published={outcome.get('published')} frames "
        f"{frames_before} -> {bench2.st.frames}",
    )
    check(
        "and the samples it decoded are accounted once, as accepted",
        delta(before2, bench2.st)["iq_accepted"] == CHUNK
        and delta(before2, bench2.st)["iq_rejected"] == 0
        and delta(before2, bench2.st)["missing_samples"] == 0,
        f"{CHUNK} sample(s) in the block: {describe(delta(before2, bench2.st))} "
        f"-- the picture was withheld, not the accounting lost",
    )


def case_file_loop(iq: np.ndarray) -> None:
    print("\n3b. a replayed file passing its end is a new stream, not a seam of bytes")
    OUT.mkdir(exist_ok=True)
    path = OUT / "continuity_loop.u8"
    # Signed int8, which is what hackrf_transfer writes, and stated explicitly so
    # the encoding detector cannot make the loop point ambiguous. Six caps long,
    # because a file shorter than two drains makes *every* hand-over straddle the
    # loop point -- the seam is then the normal case and proves nothing about
    # what a decoder does with a clean block.
    path.write_bytes(
        np.clip(np.rint(_split(iq[:2_500_000]) * 127.0), -127, 127).astype(np.uint8).tobytes()
    )

    # (a) The loop point itself, built the way a replay thread builds it: the
    # tail of one pass under its own stream id, then the head of the next under
    # a new one, with the positions either side of it perfectly adjacent. There
    # is no gap and no retune -- nothing but the identity says the samples are
    # two recordings -- which is why the seam has to be refused rather than
    # stepped across.
    bench = Bench()
    st = bench.st
    bench.feed(iq[:CHUNK])
    bench.step()
    first = bench.src._last_delivery
    check("the replay decodes before its end is reached", st.frames > 0,
          f"{st.frames} picture(s), interval {st.frame_interval}")
    half = CHUNK // 2
    before = mark(st)
    bench.feed(iq[CHUNK : CHUNK + half])          # tail of the pass, old stream
    bench.stream = bench.src._begin_stream()
    bench.src.ring.write(iq_to_u8(iq[CHUNK + half : 2 * CHUNK]), 0, bench.stream)
    published = bench.step()
    block = bench.src._last_delivery
    d = delta(before, st)
    check(
        "the loop point is one block holding two producers, adjacent in position",
        block.spans_stream and block.seam == "stream"
        and block.first_sample == first.last_sample,
        f"stream {block.first_stream_id} -> {block.last_stream_id} at "
        f"[{block.first_sample}, {block.last_sample}), seam={block.seam!r}, "
        f"the block before it ended at {first.last_sample}",
    )
    check(
        "and a worker refuses it: two recordings, one picture, not a field",
        not published and d["cause:stream"] == 1 and d["iq_rejected"] == CHUNK
        and d["iq_accepted"] == 0,
        describe(d),
    )
    before = mark(st)
    bench.feed(iq[2 * CHUNK : 3 * CHUNK])
    published = bench.step()
    d = delta(before, st)
    check(
        "and the next block after the loop decodes normally",
        published and d["segments"] == 1 and d["iq_rejected"] == 0,
        describe(d),
    )

    # (b) The real replay, to show that the identity in (a) is one the source
    # itself stamps rather than something this script invented.
    replay = sdr.FileSource(path, sample_rate=FS, loop=True, realtime=False,
                            encoding=sdr.IQEncoding.SIGNED_INT8)
    replay.start()
    worker = video.DecodeWorker(replay, width=160)
    end = time.monotonic() + 6.0
    while time.monotonic() < end and (worker.stats.frames < 4
                                       or replay.stream_id < 3):
        worker.step()
    streams = replay.stream_id
    worker.stop()
    replay.stop()
    st = worker.stats
    check(
        "every pass of a looping replay begins a new stream identity",
        streams >= 3,
        f"{streams} stream id(s) after {st.iq_delivered} samples delivered",
    )
    check(
        "and a worker fed that file refuses the loop points, and keeps decoding",
        st.frames > 0 and st.gap_causes.get("stream", 0) > 0,
        f"{st.frames} picture(s), gap_causes={st.gap_causes}, "
        f"seam_blocks={st.seam_blocks}",
    )


def _split(iq: np.ndarray) -> np.ndarray:
    """Interleaved real/imaginary view, for writing a capture file directly."""
    return np.stack((iq.real, iq.imag), axis=-1).reshape(-1)


# ---------------------------------------------------------------------------
# 4. window capacity: NTSC and PAL out of one isolated 42 ms segment
# ---------------------------------------------------------------------------


def case_capacity(ntsc_iq: np.ndarray, pal_iq: np.ndarray) -> None:
    print("\n4. one isolated 42 ms segment carries a whole field, both standards")
    for tag, iq, spec in (("NTSC", ntsc_iq, dsp.NTSC), ("PAL", pal_iq, dsp.PAL)):
        ok_phase = []
        for phase in (0, 131_071, 262_144):
            seg = iq[phase : phase + CHUNK]
            dec = video.VideoDecoder(FS, width=160)     # the default 42 ms window
            dec.reset()                                  # an isolated segment
            frame = dec.push_iq(seg, continuous=False, first_sample=phase)
            got = frame is not None and frame.height == spec.visible_lines
            ok_phase.append(got)
            check(
                f"{tag} at phase {phase}: one isolated 42 ms segment renders a "
                f"{spec.visible_lines}-line field",
                got,
                f"{'none' if frame is None else f'{frame.height}x{frame.width}'} "
                f"from {seg.size} samples, grid rows "
                f"{0 if frame is None else frame.lines_from_grid}",
            )
        field_ms = spec.visible_lines * 640.0 / FS * 1000.0
        check(
            f"{tag}: capacity is phase-independent",
            all(ok_phase),
            f"{field_ms:.1f} ms of field inside a 42 ms window "
            f"({spec.visible_lines} lines), phases 0/131071/262144",
        )
        dec = video.VideoDecoder(FS, width=160)
        seg = iq[:CHUNK]
        dec.reset()
        dec.push_iq(seg, continuous=False, first_sample=0)
        check(
            f"{tag}: an isolated segment contributes its own length minus one",
            dec.fm_samples == seg.size - 1 and dec.boundary_steps == 0,
            f"{dec.fm_samples} outputs from {seg.size} IQ samples",
        )

    # Multi-phase continuity: the same stream, chunked adjacent, must keep
    # N-1 across the whole run and must not reset the decoder anywhere.
    bounds = [0, 40_000, 101_000, 256_000, 346_000]
    for tag, iq in (("NTSC", ntsc_iq), ("PAL", pal_iq)):
        total = bounds[-1]
        dec = video.VideoDecoder(FS, width=160)
        for k in range(1, len(bounds)):
            start, end = bounds[k - 1], bounds[k]
            dec.push_iq(iq[start:end], continuous=True, first_sample=start)
        check(
            f"{tag}: {len(bounds) - 1} adjacent chunks of one stream, no reset",
            dec.fm_samples == total - 1 and dec.boundary_steps == len(bounds) - 2
            and dec.fm_resets == 0,
            f"fm={dec.fm_samples} of {total - 1}, boundary steps="
            f"{dec.boundary_steps} (one per join), resets={dec.fm_resets}",
        )


# ---------------------------------------------------------------------------
# 5. burst adjacency
# ---------------------------------------------------------------------------


def case_burst(iq: np.ndarray) -> None:
    print("\n5. a burst of small writes drains as one segment")
    bench = Bench()
    st = bench.st
    small = 30_000
    # The delivery cap is one 42 ms window, which is wider than the whole burst,
    # so without this the burst would arrive in a single hand-over and there
    # would be nothing to prove. Narrowing the cap is the delivery primitive's
    # own policy knob, not a decoder setting.
    bench.w._max_chunk = small
    # Written and drained back to back, with no pause: the writer never waits
    # for the reader, and each drain takes exactly what the last write left.
    # (Filling the ring with all twelve writes first would not test this -- the
    # cap would then discard the backlog on purpose, which is case 7's subject
    # and a loss this case must not confuse with adjacency.)
    for k in range(12):
        bench.feed(iq[k * small : (k + 1) * small])
        bench.step()
    total = 12 * small
    check(
        "the burst was delivered whole, one small chunk at a time",
        st.iq_delivered == total and st.chunk_samples == small,
        f"delivered={st.iq_delivered} of {total} in 12 hand-over(s), last chunk "
        f"{st.chunk_samples} sample(s)",
    )
    check(
        "with no backlog skipped to get there",
        st.capped_skip_samples == 0 and int(bench.src.ring.dropped_bytes) == 0,
        f"capped={st.capped_skip_samples} overrun={bench.src.ring.dropped_bytes // 2}",
    )
    check(
        "no gap was invented between the chunks of the burst",
        st.missing_samples == 0 and st.discontinuities == 0,
        f"missing={st.missing_samples} discontinuities={st.discontinuities} "
        f"gap_causes={st.gap_causes}",
    )
    check(
        "and the decoder treated it as one segment",
        st.segments == 1 and st.boundary_steps == 11 and st.fm_resets == 1,
        f"segments={st.segments} boundary steps={st.boundary_steps} "
        f"(decoder resets {st.fm_resets}, one for the first block of a segment)",
    )
    check(
        "the N-1 invariant survives the burst",
        st.fm_samples == total - 1,
        f"fm={st.fm_samples} of {total - 1}",
    )


# ---------------------------------------------------------------------------
# 6. freshness
# ---------------------------------------------------------------------------


def case_freshness(iq: np.ndarray) -> None:
    print("\n6. nothing new to show must not look like a new picture")
    bench = Bench()
    st = bench.st
    run_to_lock(bench, iq, blocks=4)
    locked = st.frames
    check("locked to start with", locked >= 3,
          f"{locked} picture(s), {st.unique_frames} unique, {st.dup_frames} duplicate")

    # (a) New samples that do not complete a new field: the rolling window still
    # holds the field it last drew, so the rasteriser can legitimately produce it
    # again -- and it must not be published as new. A fifth of a scanline is
    # deliberately little: enough for the rasteriser to run over the window, not
    # enough to finish another line, let alone another field.
    pos = CHUNK * 4
    before = mark(st)
    bench.feed(iq[pos : pos + 200])
    pos += 200
    at = st.last_frame_at
    published = bench.step()
    d = delta(before, st)
    check(
        "a re-render of the field already on screen is counted, not published",
        d["frames"] == 0 and not published and st.dup_frames > 0
        and st.last_frame_at == at,
        f"{st.raster_accepts} field(s) rasterised in total, dup_frames="
        f"{st.dup_frames}, publish time "
        f"{'moved' if st.last_frame_at != at else 'unchanged'}",
    )

    # (b) An empty drain, repeatedly: a stalled or disconnected source, polled at
    # the rate the app polls at. Starving the poll is the case worth testing --
    # one starved read must not cost a field.
    before = mark(st)
    for _ in range(40):
        bench.step()
        time.sleep(0.02)
    d = delta(before, st)
    check(
        "forty empty drains publish nothing and break nothing",
        d["frames"] == 0 and d["iq_delivered"] == 0 and d["discontinuities"] == 0
        and d["iq_accepted"] == 0 and d["segments"] == 0,
        describe(d),
    )
    check(
        "and the picture on screen is still the same picture",
        st.frames == locked and st.unique_frames == locked,
        f"frames={st.frames} unique={st.unique_frames} "
        f"(interval {st.frame_interval})",
    )
    # ... until it is old enough to stop being called live, at which point the
    # stall is what the counters must be reporting.
    time.sleep(video.STALE_FRAME_S + 0.2)
    bench.step()
    check(
        "the picture ages out of 'live' instead of standing there for ever",
        bench.w.frame_age_s() > video.STALE_FRAME_S and bench.w.latest() is None,
        f"age={bench.w.frame_age_s():.2f}s against a {video.STALE_FRAME_S}s "
        f"limit, locked={bench.w.locked}",
    )
    check(
        "and the stall is counted as missed fields rather than silence",
        st.fields_missed > 0,
        f"fields_missed={st.fields_missed}, miss rate {st.miss_rate():.2f}",
    )

    # (c) End of file: a replay that has nothing more to give.
    OUT.mkdir(exist_ok=True)
    path = OUT / "continuity_eof.u8"
    path.write_bytes(
        np.clip(np.rint(_split(iq[:500_000]) * 127.0), -127, 127).astype(np.uint8).tobytes()
    )
    src = sdr.FileSource(path, sample_rate=FS, loop=False, realtime=False,
                         encoding=sdr.IQEncoding.SIGNED_INT8)
    src.start()
    worker = video.DecodeWorker(src, width=160)
    end = time.monotonic() + 5.0
    while time.monotonic() < end and worker.stats.frames < 2:
        worker.step()
    at_eof = worker.stats.last_frame_at
    frames_at_eof = worker.stats.frames
    drained = 0
    end = time.monotonic() + 1.5
    while time.monotonic() < end:
        if worker.step():
            drained += 1
    worker.stop()
    src.stop()
    check(
        "at end of file the picture stops advancing and nothing duplicates",
        worker.stats.frames == frames_at_eof and drained == 0
        and worker.stats.last_frame_at == at_eof,
        f"{worker.stats.frames - frames_at_eof} new picture(s) in 1.5 s past EOF, "
        f"unique={worker.stats.unique_frames} dup={worker.stats.dup_frames}",
    )
    check(
        "the EOF is visible as a stall, not as a live channel",
        not worker.locked and worker.latest() is None,
        f"locked={worker.locked} age={worker.frame_age_s():.2f}s",
    )


# ---------------------------------------------------------------------------
# 7. accounting identities
# ---------------------------------------------------------------------------


def buckets(bench: Bench) -> dict:
    """The five buckets, each read from the counter that owns it."""
    st = bench.st
    written = int(st.iq_written)
    delivered = int(st.iq_delivered)
    accepted = int(st.iq_accepted)
    rejected = int(st.iq_rejected)
    capped = int(st.capped_skip_samples)
    missing = int(st.missing_samples)
    overrun = int(bench.src.ring.dropped_bytes) // 2
    return {
        "written": written,
        "delivered": delivered,
        "accepted": accepted,
        "rejected": rejected,
        "capped": capped,
        "missing": missing,
        "overrun": overrun,
        "deliberate": capped - overrun,
        "reset_discard": written - accepted - rejected - capped,
    }


def show(tag: str, b: dict) -> None:
    print(
        f"    {tag}: written={b['written']} delivered={b['delivered']} "
        f"accepted={b['accepted']} rejected={b['rejected']} "
        f"capped={b['capped']} missing={b['missing']} overrun={b['overrun']}"
    )


def case_accounting(iq: np.ndarray) -> None:
    print("\n7. every written sample lands in exactly one bucket")
    bench = Bench()
    st = bench.st

    # (a) Nothing unusual: the buckets are trivially disjoint.
    run_to_lock(bench, iq, blocks=3)
    b = buckets(bench)
    show("clean", b)
    check(
        "delivered == accepted + rejected",
        b["delivered"] == b["accepted"] + b["rejected"],
        f"{b['delivered']} = {b['accepted']} + {b['rejected']}",
    )
    check(
        "written == accepted + rejected (no cap, no loss)",
        b["written"] == b["accepted"] + b["rejected"] + b["missing"] + b["overrun"]
        and b["missing"] == 0 and b["overrun"] == 0,
        f"{b['written']} == {b['accepted']} + {b['rejected']} + "
        f"{b['missing']} + {b['overrun']}",
    )

    # (b) A backlog longer than the cap: dropped on purpose to stay current.
    pos = CHUNK * 3
    bench.feed(iq[pos : pos + 3 * CHUNK])
    pos += 3 * CHUNK
    bench.step()
    b = buckets(bench)
    show("capped skip", b)
    check(
        "a capped skip is policy, not loss, and is counted",
        st.capped_skips >= 1 and b["capped"] == 2 * CHUNK and b["overrun"] == 0,
        f"capped_skips={st.capped_skips} capped={b['capped']} overrun={b['overrun']}",
    )
    check(
        "delivered == accepted + rejected still holds",
        b["delivered"] == b["accepted"] + b["rejected"],
        f"{b['delivered']} = {b['accepted']} + {b['rejected']}",
    )
    check(
        "written == accepted + rejected + missing",
        b["written"] == b["accepted"] + b["rejected"] + b["missing"],
        f"{b['written']} == {b['accepted']} + {b['rejected']} + {b['missing']}"
        f"  (the {b['capped']} capped samples also show up as that gap, which is"
        f" the one place the buckets overlap)",
    )

    # (c) A ring overrun: bytes destroyed before anybody read them. Twice the
    # ring, so the head of the write certainly lands on data still unread.
    bench.src.ring.write(b"\x00" * (2 * sdr.RING_BYTES), bench.tune, bench.stream)
    bench.step()
    b = buckets(bench)
    show("overrun", b)
    check(
        "an overrun is loss, and it is counted against the decoder",
        b["overrun"] > 0 and b["deliberate"] >= 0,
        f"overrun={b['overrun']} deliberate={b['deliberate']} "
        f"(the skipped span is overrun + deliberate)",
    )
    check(
        "delivered == accepted + rejected still holds after loss",
        b["delivered"] == b["accepted"] + b["rejected"],
        f"{b['delivered']} = {b['accepted']} + {b['rejected']}",
    )
    check(
        "written == accepted + rejected + missing holds through a loss",
        b["written"] == b["accepted"] + b["rejected"] + b["missing"],
        f"{b['written']} == {b['accepted']} + {b['rejected']} + {b['missing']}",
    )

    # (d) A drain reset discarding a known span.
    bench.feed(iq[pos : pos + CHUNK])
    pos += CHUNK
    skips_before = st.capped_skips
    bench.src.reset_drain("retune")
    bench.step()
    bench.feed(iq[pos : pos + CHUNK])
    pos += CHUNK
    bench.step()
    b = buckets(bench)
    show("drain reset", b)
    check(
        "what a drain reset discarded is in the accounting, not in the air",
        b["reset_discard"] > 0
        and b["written"] == b["accepted"] + b["rejected"] + b["capped"] + b["reset_discard"],
        f"written={b['written']} = accepted {b['accepted']} + rejected "
        f"{b['rejected']} + capped {b['capped']} + reset {b['reset_discard']}",
    )
    check(
        "written == accepted + rejected + missing, once more",
        b["written"] == b["accepted"] + b["rejected"] + b["missing"],
        f"{b['written']} == {b['accepted']} + {b['rejected']} + {b['missing']}",
    )
    check(
        "and the reset discard is exactly what the gap counter saw",
        b["missing"] == b["capped"] + b["reset_discard"],
        f"missing={b['missing']} = capped {b['capped']} + reset {b['reset_discard']}",
    )
    check(
        "a reset is neither a cap nor a rejection",
        st.capped_skips == skips_before and b["rejected"] == 0,
        f"capped_skips {skips_before} -> {st.capped_skips}, rejected={b['rejected']}",
    )


# ---------------------------------------------------------------------------
# 8. snapshot neutrality
# ---------------------------------------------------------------------------


def feed_script(bench: Bench, iq: np.ndarray, snapshots: bool) -> None:
    """The same delivery every time, optionally reading the ring in between."""
    pos = 0
    for _ in range(6):
        seg = iq[pos : pos + CHUNK]
        pos += seg.size
        bench.feed(seg)
        if snapshots:
            diag._DUMP._next = 0.0
            diag.dump_tick(bench.src._dump_snapshot)
        bench.step()
    bench.src.ring.write(b"\x00" * (2 * sdr.RING_BYTES), bench.tune, bench.stream)   # an overrun
    if snapshots:
        diag._DUMP._next = 0.0
        diag.dump_tick(bench.src._dump_snapshot)
    bench.step()


def case_snapshot_neutrality(iq: np.ndarray) -> None:
    print("\n8. dumping snapshots must not disturb the stream")
    OUT.mkdir(exist_ok=True)
    dump_dir = OUT / "continuity_dump"
    for stale in dump_dir.glob("*"):
        stale.unlink()
    dump_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, snapshots in (("without snapshots", False), ("with snapshots", True)):
        bench = Bench()
        if snapshots:
            os.environ[diag.ENV_DUMP] = str(dump_dir / "seg")
            os.environ[diag.ENV_DUMP_SAMPLES] = "100000"
            os.environ[diag.ENV_DUMP_FILES] = "2"
            os.environ[diag.ENV_DUMP_FLUSH_S] = "1"
            diag._DUMP.reset()
        feed_script(bench, iq, snapshots)
        st = bench.st
        results[label] = (
            st.iq_written, st.iq_delivered, st.iq_accepted, st.iq_rejected,
            st.missing_samples, st.capped_skip_samples, int(bench.src.ring.dropped_bytes),
            st.frames, st.fm_samples, st.boundary_steps,
        )
    for name in (diag.ENV_DUMP, diag.ENV_DUMP_SAMPLES, diag.ENV_DUMP_FILES,
                 diag.ENV_DUMP_FLUSH_S):
        os.environ.pop(name, None)
    diag._DUMP.reset()

    plain = results["without snapshots"]
    dumped = results["with snapshots"]
    names = ("iq_written", "iq_delivered", "iq_accepted", "iq_rejected",
             "missing_samples", "capped_skip_samples", "ring dropped bytes",
             "frames", "fm_samples", "boundary_steps")
    check(
        "the delivery and the accounting are identical with and without snapshots",
        plain == dumped,
        ", ".join(f"{n}: {a} vs {b}" for n, a, b in zip(names, plain, dumped)
                  if a != b) or "every counter matches",
    )
    check(
        "snapshots were actually taken while it ran",
        bool(list(dump_dir.glob("*.json"))),
        f"{len(list(dump_dir.glob('*')))} file(s) in {dump_dir}",
    )

    # The cursor, specifically: a snapshot must not move it. Written in two
    # instalments because a drain takes the *newest* cap, so one big write
    # would leave nothing for a second hand-over to find.
    src = sdr.SimSource(video=True)
    src._begin_stream()
    src.ring.write(iq_to_u8(iq[:200_000]), 0, src.stream_id)
    first = src.drain_block(200_000)
    for _ in range(5):
        snap = src.ring.snapshot_newest(200_000)
    src.ring.write(iq_to_u8(iq[200_000:400_000]), 0, src.stream_id)
    second = src.drain_block(200_000)
    check(
        "five snapshots in a row did not move the drain cursor",
        second.first_sample == first.last_sample and second.samples == 200_000
        and snap.last_byte - snap.first_byte == len(snap.data),
        f"first ended at {first.last_sample}, second starts at "
        f"{second.first_sample} ({second.samples} samples)",
    )
    check(
        "and they did not count as consumption: nothing was lost either",
        src.ring.dropped_bytes == 0 and src.ring.total_written // 2 == 400_000,
        f"written={src.ring.total_written // 2} samples, dropped="
        f"{src.ring.dropped_bytes // 2}",
    )


# ---------------------------------------------------------------------------
# 9. overlap: a delivery that reaches back into samples already decoded
# ---------------------------------------------------------------------------


def rewind(bench: Bench, samples: int) -> None:
    """Make the next drain hand over ``samples`` samples it has already given.

    Done on the cursor under its own lock, which is the whole state of the
    delivery: the ring still holds those bytes, they are still stamped with the
    right identities, and nothing about them changed. Only the consumer's
    position moved -- which is the situation a rewind of the absolute stream
    counter produces, and the only way to see what the join does with a block
    that starts *before* the end of what it already accepted.
    """
    with bench.src._drain_lock:
        bench.src._consumed -= 2 * int(samples)


def demod_state(bench: Bench) -> tuple[int, np.ndarray]:
    """A bench's demodulator output and where it starts, for exact comparison."""
    dec = bench.w.decoder
    return int(dec.demod_origin), dec._demod.copy()


def case_overlap(iq: np.ndarray) -> None:
    print("\n9. a repeated head is removed before the decoder sees it")
    repeat = 100_000
    plain = Bench()
    again = Bench()
    pos = 0

    def both(chunk: int) -> None:
        """One matched feed-and-step on both benches, from one shared segment."""
        nonlocal pos
        seg = iq[pos : pos + chunk]
        pos += seg.size
        for bench in (plain, again):
            bench.feed(seg)
            bench.step()

    both(CHUNK)
    both(CHUNK)
    check("both benches are locked before the rewind", plain.st.frames > 0,
          f"{plain.st.frames} and {again.st.frames} picture(s), "
          f"sync {plain.st.sync_quality:.2f} {plain.st.standard}")

    # (a) A partial repeat: the block starts 100,000 samples before the end of
    # the accepted run, so it is a genuine overlap rather than a seam or a hole.
    # The delivery is still exactly one cap wide -- the *new* samples were made
    # shorter to pay for the repeat -- which is what a backlog of that size looks
    # like to a consumer that is a little behind.
    short = CHUNK - repeat
    rewind(again, repeat)
    before = mark(again.st)
    both(short)
    d = delta(before, again.st)
    check(
        "the repeated head is counted as refused, under its own cause",
        d["cause:overlap"] == 1 and d["iq_rejected"] == repeat,
        describe(d),
    )
    check(
        "and only what is new is accepted -- the same as an unrewound bench",
        d["iq_accepted"] == short and d["iq_accepted"] == short
        and plain.st.iq_accepted == again.st.iq_accepted,
        f"accepted {d['iq_accepted']} vs {short} expected, "
        f"benches {plain.st.iq_accepted}/{again.st.iq_accepted}",
    )
    check(
        "delivered == accepted + rejected still holds across the trim",
        again.st.iq_delivered == again.st.iq_accepted + again.st.iq_rejected,
        f"{again.st.iq_delivered} = {again.st.iq_accepted} + {again.st.iq_rejected}",
    )
    check(
        "a trim is not a break: the demodulator predecessor is carried across",
        d["segments"] == 0 and d["boundary_steps"] == 1,
        f"{describe(d)} -- the join was stepped, not restarted",
    )
    check(
        "the overlap is measured too, in samples",
        again.st.overlap_samples >= repeat,
        f"overlap_samples={again.st.overlap_samples} of {repeat} repeated",
    )
    origin_a, demod_a = demod_state(again)
    origin_b, demod_b = demod_state(plain)
    same = (
        origin_a == origin_b
        and demod_a.size == demod_b.size
        and np.array_equal(demod_a.view(np.uint32), demod_b.view(np.uint32))
    )
    worst = (
        float(np.abs(demod_a.astype(np.float64) - demod_b).max())
        if demod_a.size == demod_b.size and demod_a.size
        else float("inf")
    )
    check(
        "and the demodulator output is bit-identical to an unrewound bench",
        same,
        f"origin {origin_a} vs {origin_b}, {demod_a.size} outputs"
        + ("" if same else f", worst difference {worst:.3e}"),
    )

    # (b) A block *wholly* inside what has already been decoded. Nothing is left
    # to decode, and -- the part that matters -- the continuity record is left
    # alone, so the next block is still adjacent to the accepted run instead of
    # starting a segment over a hole that was never there.
    inside = 200_000
    before = mark(again.st)
    steps_before = again.w.decoder.boundary_steps
    rewind(again, inside)
    published = again.step()
    d = delta(before, again.st)
    check(
        "a block wholly inside the repeat decodes nothing and is not a break",
        not published and d["cause:overlap"] == 1 and d["iq_rejected"] == inside
        and d["iq_accepted"] == 0 and d["frames"] == 0
        and again.w.decoder.boundary_steps == steps_before,
        describe(d),
    )
    both(CHUNK)
    d2 = delta(before, again.st)
    check(
        "the block after it is still adjacent, so it is a segment and not a gap",
        d2["segments"] == 0 and d2["boundary_steps"] == 1 and d2["iq_accepted"] == CHUNK,
        describe(d2),
    )
    origin_a, demod_a = demod_state(again)
    origin_b, demod_b = demod_state(plain)
    check(
        "an emptied overlap does not perturb the demodulator either",
        origin_a == origin_b and demod_a.size == demod_b.size
        and np.array_equal(demod_a.view(np.uint32), demod_b.view(np.uint32)),
        f"origin {origin_a} vs {origin_b}, {demod_a.size} outputs",
    )

    # (c) What is left after the repeat is *itself* too small to demodulate. The
    # size that decides is the size of what is new, not of what arrived, so this
    # is the same refusal a tiny block gets anywhere else -- and the span is left
    # open, so the next block starts a segment rather than stepping across
    # samples that were never decoded.
    remnant = 10
    trim = 150_000
    rewind(again, trim)
    seg = iq[pos : pos + remnant]
    pos += remnant
    again.feed(seg)
    before = mark(again.st)
    published = again.step()
    d = delta(before, again.st)
    check(
        "a remainder below the minimum is refused, not demodulated",
        not published and d["cause:overlap"] == 1 and d["cause:tiny"] == 1
        and d["iq_rejected"] == trim + remnant and d["iq_accepted"] == 0,
        describe(d),
    )
    check(
        "and both the repeat and the refusal are counted as discontinuities",
        d["discontinuities"] == 2,
        f"{d['discontinuities']} discontinuity(ies), gap_causes={again.st.gap_causes}",
    )
    check(
        "delivered == accepted + rejected holds across the trim-then-refuse",
        again.st.iq_delivered == again.st.iq_accepted + again.st.iq_rejected,
        f"{again.st.iq_delivered} = {again.st.iq_accepted} + {again.st.iq_rejected}",
    )
    before = mark(again.st)
    both(CHUNK)
    d = delta(before, again.st)
    check(
        "the span is left open, so the next block starts a fresh segment",
        d["segments"] == 1 and d["boundary_steps"] == 0 and d["iq_rejected"] == 0
        and d["iq_accepted"] == CHUNK,
        describe(d),
    )

    # (d) The same refusal, but the delivery that follows it reaches *back* over
    # the refused span -- and past it, into samples the decoder does still hold,
    # so the trim is as exact as ever. What is left is accepted: refusing it
    # would throw away 22 ms of good video over a hole that is already counted.
    # But it begins after samples nobody decoded, so it continues nothing, and is
    # a fresh segment with no boundary step across the refusal.
    rewind(again, trim)
    seg = iq[pos : pos + remnant]
    pos += remnant
    again.feed(seg)
    before = mark(again.st)
    again.step()
    d = delta(before, again.st)
    check(
        "the trim-then-tiny refusal is in place, and the span is open",
        d["cause:overlap"] == 1 and d["cause:tiny"] == 1
        and d["iq_accepted"] == 0 and again.w._cont.open,
        f"{describe(d)}, open={again.w._cont.open}, end={again.w._cont.end}",
    )
    reach = 200_000
    rewind(again, reach)
    seg = iq[pos : pos + CHUNK - reach]
    pos += seg.size
    again.feed(seg)
    before = mark(again.st)
    again.step()
    d = delta(before, again.st)
    fresh = CHUNK - reach
    check(
        "an overlap reaching into a refused span is trimmed and accepted",
        d["cause:overlap"] == 1 and d["iq_rejected"] == reach
        and d["iq_accepted"] == fresh and d["discontinuities"] == 1,
        f"{describe(d)} -- {fresh} of {CHUNK} is new",
    )
    check(
        "but it continues nothing: a fresh segment and no step across the refusal",
        d["segments"] == 1 and d["boundary_steps"] == 0,
        f"{describe(d)} -- the decoder holds no predecessor to step from",
    )
    check(
        "delivered == accepted + rejected holds across a refused-span overlap",
        again.st.iq_delivered == again.st.iq_accepted + again.st.iq_rejected,
        f"{again.st.iq_delivered} = {again.st.iq_accepted} + {again.st.iq_rejected}",
    )
    started = pos - seg.size
    origin = int(again.w.decoder.demod_origin)
    check(
        "and the demodulator output starts where the new run does, not one early",
        origin == started,
        f"demod origin {origin}, the run starts at {started} -- a sample early"
        f" would be the offset a taken boundary step had bought",
    )
    seg2 = iq[pos : pos + CHUNK]
    pos += seg2.size
    again.feed(seg2)
    before = mark(again.st)
    again.step()
    d = delta(before, again.st)
    check(
        "the block after that segment is adjacent to it, so the step resumes",
        d["segments"] == 0 and d["boundary_steps"] == 1 and d["iq_accepted"] == CHUNK,
        f"{describe(d)} -- {fresh + CHUNK} samples in one segment",
    )

    # (e) The other way the span is left open: a decode that raised. The samples
    # were taken into the window and then the decode failed, so the window holds
    # samples counted as refused and its predecessor is the last sample of a run
    # nobody demodulated. An overlap afterwards must therefore restart rather
    # than carry, or the state the exception poisoned is kept and stepped across.
    real = again.w.decoder

    class Boom:
        """The real decoder, except that one push takes the samples in and fails."""

        def __init__(self, inner) -> None:
            self.inner = inner
            self.armed = True

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def reset(self) -> None:
            self.inner.reset()

        def push_iq(self, iq, *, continuous=False, first_sample=None,
                    provenance=None):
            # Mirrors the real signature, including the TEMP DIAG provenance
            # keyword: the worker names it, so a stand-in that omitted it would
            # fail on its *signature* -- a TypeError counted as a decode error,
            # which is a different failure from the one this case is about.
            if not self.armed:
                return self.inner.push_iq(
                    iq, continuous=continuous, first_sample=first_sample,
                    provenance=provenance,
                )
            self.armed = False
            try:
                self.inner.push_iq(
                    iq, continuous=continuous, first_sample=first_sample,
                    provenance=provenance,
                )
            except Exception:
                pass
            raise ValueError("deliberate")

    boom = Boom(real)
    again.w.decoder = boom
    seg3 = iq[pos : pos + CHUNK]
    pos += seg3.size
    again.feed(seg3)
    before = mark(again.st)
    errors = again.st.decode_errors
    again.step()
    d = delta(before, again.st)
    left = int(real._demod.size)
    check(
        "a decode that raised is counted, and its samples leave the span open",
        again.st.decode_errors == errors + 1 and again.w._cont.open
        and d["iq_accepted"] == 0 and d["iq_rejected"] == CHUNK
        and again.st.iq_delivered == again.st.iq_accepted + again.st.iq_rejected,
        f"{describe(d)}, error {again.st.last_decode_error!r}, "
        f"{left} refused samples left in the window",
    )
    reach = 150_000
    rewind(again, reach)
    seg4 = iq[pos : pos + CHUNK - reach]
    pos += seg4.size
    again.feed(seg4)
    before = mark(again.st)
    again.step()
    d = delta(before, again.st)
    again.w.decoder = real
    fresh = CHUNK - reach
    check(
        "an overlap after a poisoned span is trimmed and accepted",
        d["cause:overlap"] == 1 and d["iq_rejected"] == reach
        and d["iq_accepted"] == fresh,
        f"{describe(d)} -- {fresh} of {CHUNK} is new",
    )
    check(
        "and it drops the window the exception left behind",
        d["segments"] == 1 and d["boundary_steps"] == 0,
        f"{describe(d)} -- none of the {left} refused samples are carried",
    )
    started = pos - seg4.size
    origin, demod = demod_state(again)
    check(
        "so the demodulator starts where the new run does, not one early",
        origin == started and demod.size == fresh - 1,
        f"demod origin {origin} (the run starts at {started}), "
        f"{demod.size} outputs of the {fresh - 1} this segment can have",
    )


# ---------------------------------------------------------------------------
# 10. whole-sample discipline at the producer
# ---------------------------------------------------------------------------


def iq_payload(n_bytes: int) -> bytes:
    """Interleaved bytes whose samples carry a checkable identity.

    Sample *s* is ``I = s%64 + 20``, ``Q = s%64 + 40``, so ``Q - I`` is 20 for
    every sample and for no other pairing. A ring that mis-paired -- a Q where an
    I belonged, from a dropped or reordered half-sample -- yields -19 or -83 at
    the join, so this detects a mis-pair rather than merely a wrong count: both
    counts would still add up, which is exactly why the defect survived a while.

    Both components stay inside the signed byte range on purpose. The live path
    decodes as signed int8, so a "Q" above 127 would read back as negative and
    the invariant would be checking a fixture rather than the pairing.
    """
    n = int(n_bytes) // 2
    s = np.arange(n, dtype=np.int16)
    out = np.empty(n * 2, dtype=np.uint8)
    out[0::2] = (s % 64 + 20).astype(np.uint8)
    out[1::2] = (s % 64 + 40).astype(np.uint8)
    return out.tobytes()


def pairs_intact(raw: bytes, label: str = "") -> tuple[bool, str]:
    """Are all the samples in ``raw`` correctly paired? Plus what to print."""
    if len(raw) % 2:
        return False, f"{len(raw)} bytes: an odd count, so half a sample"
    if not raw:
        return True, "empty"
    arr = np.frombuffer(raw, dtype=np.int8)
    gap = arr[1::2].astype(np.int16) - arr[0::2].astype(np.int16)
    bad = int(np.count_nonzero(gap != 20))
    return (
        bad == 0,
        f"{arr.size // 2} sample(s){(' ' + label) if label else ''}, "
        f"{bad} mis-paired",
    )


class _StepPipe:
    """A stdout stand-in that hands over prescribed sizes, then EOF.

    The sizes come from ``plan`` cyclically, so the parity of every hand-over is
    chosen rather than hoped for: a pipe read may end anywhere, and an odd count
    is an ordinary answer from ``readinto`` rather than a fault to be provoked by
    luck.

    Once the plan is spent the pipe parks on ``park`` when it was given one, the
    way a live process with nothing more to say blocks in ``readinto``; setting
    the event is what lets that read come back with EOF. It has to park rather
    than answer zero at once, because ``HackrfSource._run`` reads *any*
    zero-length read as EOF and tears the producer down -- so a test wanting a
    replacement to be installed while the old pipe is still parked would be
    measuring the EOF path instead of the replacement.
    """

    def __init__(self, payload: bytes = b"", plan: tuple[int, ...] = (8,),
                 park: threading.Event | None = None) -> None:
        self._payload = payload
        self._plan = plan
        self._at = 0
        self._park = park
        self.closed = False
        self.reads = 0
        self.odd_reads = 0

    def readinto(self, view) -> int:
        self.reads += 1
        if self.closed:
            return 0
        if self._at >= len(self._payload):
            if self._park is not None:
                self._park.wait(3.0)
            return 0
        want = self._plan[(self.reads - 1) % len(self._plan)]
        n = min(want, len(view), len(self._payload) - self._at)
        if n & 1:
            self.odd_reads += 1
        view[:n] = self._payload[self._at : self._at + n]
        self._at += n
        return n

    def close(self) -> None:
        self.closed = True


class _Proc:
    """Just enough ``subprocess.Popen`` for :meth:`HackrfSource._run`."""

    def __init__(self, stdout, stream_id: int = 1) -> None:
        self.stdout = stdout
        self.stderr = None
        self.returncode: int | None = None
        self.terminated = False
        self._fpv_stream_id = stream_id

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


def reader_source() -> "sdr.HackrfSource":
    """A HackrfSource wired to run the real ``_run``, minus the radio."""
    src = sdr.HackrfSource(frequency_hz=5_802_000_000, executable="hackrf_transfer")
    src._stop.clear()

    def no_spawn() -> bool:
        # The pipes below end in EOF, and EOF must look like the end of this
        # test rather than like an unplugged radio the reader retries for ever.
        src._stop.set()
        return False

    src._spawn_settled = no_spawn
    return src


def ring_bytes(src: "sdr.HackrfSource", n_bytes: int) -> bytes:
    """The whole ring, oldest-first. Only for a source nothing else is reading."""
    return bytes(src.ring.read_newest(n_bytes))


def case_whole_samples() -> None:
    print("\n10. the ring never holds half a sample, whoever the producer is")
    # The helper on its own, so the rule is pinned where it is written.
    even, odd = sdr._whole_pairs(b"\x01\x02\x03\x04")
    check(
        "_whole_pairs keeps a whole buffer whole and invents no phantom byte",
        bytes(even) == b"\x01\x02\x03\x04" and odd == b"",
        f"even={len(even)} bytes, remainder={len(odd)}",
    )
    even, odd = sdr._whole_pairs(b"\x01\x02\x03")
    check(
        "_whole_pairs holds back exactly one byte from an odd read",
        bytes(even) == b"\x01\x02" and odd == b"\x03",
        f"even={len(even)} bytes, remainder={len(odd)}",
    )
    even, odd = sdr._whole_pairs(b"\x07")
    check(
        "and a buffer of one byte is all remainder, with nothing even to keep",
        len(even) == 0 and odd == b"\x07",
        f"even={len(even)} bytes, remainder={len(odd)}",
    )

    # (a) The live hardware path, driven through the real sampling thread against
    # a pipe that ends every read between the halves of a sample.
    payload = iq_payload(8194)
    pipe = _StepPipe(payload, plan=(7, 5, 3, 9, 11, 1, 13))
    src = reader_source()
    src._proc = _Proc(pipe, stream_id=3)
    th = threading.Thread(target=src._run, name="iq-hackrf-pairs", daemon=True)
    th.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and src.ring.total_written < len(payload):
        time.sleep(0.002)
    src._stop.set()
    th.join(timeout=2.0)
    raw = ring_bytes(src, len(payload))
    ok_pairs, detail = pairs_intact(raw)
    check(
        "every sample the pipe delivered reaches the ring correctly paired",
        ok_pairs and raw == payload,
        detail + ("" if raw == payload else f", {len(raw)} of {len(payload)} bytes"),
    )
    check(
        "the reads really were odd, most of them",
        pipe.odd_reads >= 5 and pipe.reads > pipe.odd_reads // 2,
        f"{pipe.odd_reads} odd read(s) of {pipe.reads}",
    )
    check(
        "nothing was left behind, and no half-sample is owed to anybody",
        src.ring.total_written == len(payload) and src._carry_byte == b""
        and src._carry_owner is None,
        f"wrote {src.ring.total_written} of {len(payload)} bytes, "
        f"carry={src._carry_byte!r} owner={src._carry_owner!r}",
    )
    check(
        "and the reader stopped when told, rather than spinning on EOF",
        not th.is_alive(),
        "thread joined inside 2 s",
    )

    # (b) An odd *last* read: the half-sample left behind goes with the pipe, and
    # the ring holds the whole samples only.
    payload = iq_payload(8192) + bytes([0x2A])       # 4096 samples plus half of one
    src = reader_source()
    src._proc = _Proc(_StepPipe(payload, plan=(1024, 1024, 1024, 1024, 1024)))
    th = threading.Thread(target=src._run, name="iq-hackrf-pairs", daemon=True)
    th.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and src.ring.total_written < len(payload) - 1:
        time.sleep(0.002)
    src._stop.set()
    th.join(timeout=2.0)
    raw = ring_bytes(src, len(payload) - 1)
    ok_pairs, detail = pairs_intact(raw)
    check(
        "a recording ending mid-sample leaves the whole samples and no half one",
        ok_pairs and src.ring.total_written == len(payload) - 1
        and raw == payload[:-1],
        f"{detail}, {src.ring.total_written} of {len(payload)} bytes",
    )

    # (c) The replacement race, which is the reason the carry is scoped to a
    # producer. The first process ends its last read between two halves; the
    # reader is blocked in ``readinto`` on it; a second process is installed in
    # exactly that window and then the first one is allowed to finish. Both
    # pipes park rather than reach EOF, so "which process is the reader on" is
    # a fact about a blocked thread instead of a race against the clock.
    park = threading.Event()
    park2 = threading.Event()
    head_payload = iq_payload(1024) + bytes([0x2A])    # 512 samples plus half of one
    first = _StepPipe(head_payload, plan=(1024, 1), park=park)
    second = _StepPipe(iq_payload(2048), plan=(7, 5, 3), park=park2)
    src = reader_source()
    p1 = _Proc(first, stream_id=11)
    p2 = _Proc(second, stream_id=12)
    src._proc = p1
    th = threading.Thread(target=src._run, name="iq-hackrf-pairs", daemon=True)
    th.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and first.reads < 3:
        time.sleep(0.002)
    # The reader has taken process 11's dangling byte and is now blocked in its
    # next readinto -- the exact state a retune interrupts. The byte is in the
    # buffer's first slot, which is where the carry was put.
    carried = bytes(src._block[:1])
    src._proc = p2                              # installed mid-read, as a retune does
    park.set()                                  # ... and the old one lets go
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and src.ring.total_written < 1024 + 2048:
        time.sleep(0.002)
    # The reader is now parked inside the *replacement's* read, so nothing below
    # can move until it is released.
    current = src._proc
    raw = ring_bytes(src, 1024 + 2048)
    want_head, want_tail = iq_payload(1024), iq_payload(2048)
    ok_pairs, detail = pairs_intact(raw, "across the replacement")
    check(
        "the outgoing process really did leave a half-sample held for it",
        carried == head_payload[-1:] and first.reads >= 3,
        f"held {carried!r} for process 11 after {first.reads} read(s), and the "
        f"producer's last byte was {head_payload[-1:]!r}",
    )
    check(
        "and it is not carried across to its successor",
        raw == want_head + want_tail,
        f"ring={len(raw)} bytes, "
        + ("both producers' bytes in order" if raw == want_head + want_tail
           else "not both producers' bytes in order"),
    )
    check(
        "so every sample on both sides of the replacement is correctly paired",
        ok_pairs,
        detail,
    )
    snap = src.ring.snapshot_newest(1024 + 2048)
    check(
        "and the two producers are still told apart across the boundary",
        snap.first_stream_id == 11 and snap.last_stream_id == 12
        and snap.first_byte == 0 and snap.last_byte == 3072
        and snap.last_byte - snap.first_byte == len(snap.data)
        and src.ring.total_written == 3072,
        f"the whole ring [{snap.first_byte}, {snap.last_byte}) carries stream "
        f"{snap.first_stream_id} -> {snap.last_stream_id}, and the reader wrote "
        f"{src.ring.total_written} bytes",
    )
    check(
        "the outgoing process was torn down without touching its replacement",
        p1.terminated and not p2.terminated and current is p2,
        f"old terminated={p1.terminated}, new untouched={not p2.terminated}, "
        f"reader moved on to the new one={current is p2}",
    )
    park2.set()                                 # let the parked read come back
    src._stop.set()
    th.join(timeout=2.0)

    # (d) The replay path, which pairs at the same place from a file read. The file
    # is one whole sample plus half of another, and each pass reads it in one
    # go, so every pass leaves a dangling byte that must go with the pass rather
    # than into the next one.
    OUT.mkdir(exist_ok=True)
    path = OUT / "continuity_odd.u8"
    whole = iq_payload(1024)                   # 512 whole samples
    body = whole + bytes([0x63])               # ... and half of a 513th
    path.write_bytes(body)
    span = len(whole)

    def start_replay(loop: bool) -> "sdr.FileSource":
        src = sdr.FileSource(path, sample_rate=FS, loop=loop, realtime=False,
                             encoding=sdr.IQEncoding.SIGNED_INT8)
        src.CHUNK_BYTES = len(body)            # one read per pass, ending odd
        src.start()
        return src

    replay = start_replay(False)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and replay.ring.total_written < span:
        time.sleep(0.002)
    replay.stop()
    raw = replay.ring.snapshot_newest(span).data
    ok_pairs, detail = pairs_intact(raw, "from the file")
    check(
        "a replay whose read ends mid-sample fills the ring with whole samples",
        ok_pairs and raw == whole and replay.ring.total_written == span,
        f"{detail}, {replay.ring.total_written} of {len(body)} bytes, "
        f"the half-sample left at the end of the recording",
    )

    # ... and at the loop point that dangling byte is *not* the next pass's first
    # half of a sample. Read the whole ring rather than the newest slice, so how
    # far the later passes have got does not matter: the loop point is at a
    # known absolute position and everything before it is already written.
    replay = start_replay(True)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and replay.ring.total_written < 2 * span:
        time.sleep(0.002)
    streams = replay.stream_id
    snap = replay.ring.snapshot_newest(1 << 20)
    replay.stop()
    head = snap.data[: 2 * span]
    ok_pairs, detail = pairs_intact(head, "across the loop point")
    seams = [p for p, _t, _s in replay.ring.diag_seams() if p == span]
    check(
        "the loop point is a seam, and its dangling byte is not glued to the "
        "next pass",
        ok_pairs and head == whole * 2 and bool(seams),
        f"{detail}, first {len(head)} bytes "
        f"{'match' if head == whole * 2 else 'differ from'} two whole passes, "
        f"seam at {span}: {'found' if seams else 'missing'}",
    )
    check(
        "and each pass of the loop starts its own stream, as it must",
        streams >= 3 and snap.first_stream_id < snap.last_stream_id,
        f"stream id {streams}, ring spans stream {snap.first_stream_id} -> "
        f"{snap.last_stream_id} over [{snap.first_byte}, {snap.last_byte})",
    )


# ---------------------------------------------------------------------------
# 11. an oversized write is not continuity
# ---------------------------------------------------------------------------


def case_oversized_write() -> None:
    print("\n11. a write too large for the ring destroys the history it claimed")
    size = 4096
    ring = sdr.IQRing(size)
    old = bytes([0xA5]) * 2000
    ring.write(old, tune_id=1, stream_id=1)
    check(
        "the ring starts holding only what was written into it",
        ring.available() == len(old) and ring.total_written == len(old),
        f"available={ring.available()} total_written={ring.total_written}",
    )

    fresh = bytes([0x5A]) * 9000                  # more than twice the ring
    kept = ring.write(fresh, tune_id=1, stream_id=1)
    head = len(fresh) - (size // 2)
    raw = ring.read_newest(size)
    stale = int(np.count_nonzero(np.frombuffer(raw, dtype=np.uint8) == 0xA5))
    check(
        "only the tail of the oversized write is stored",
        kept == size // 2 and raw == fresh[-size // 2 :]
        and ring.available() == size // 2,
        f"kept {kept} of {len(fresh)} bytes, available={ring.available()}",
    )
    check(
        "and none of the history it replaced is still readable",
        stale == 0,
        f"{stale} of the previous write's {len(old)} bytes survive in the "
        f"{len(raw)} bytes handed out",
    )
    check(
        "the bytes it destroyed are counted as loss, not quietly overwritten",
        ring.dropped_bytes == len(old) + head and ring.total_written == 11000,
        f"dropped={ring.dropped_bytes} = {len(old)} destroyed + {head} of the "
        f"write itself, total_written={ring.total_written}",
    )
    snap = ring.snapshot_newest(size)
    check(
        "the retained tail is one contiguous run at its own absolute range",
        snap.data == fresh[-size // 2 :] and snap.last_byte == 11000
        and snap.last_byte - snap.first_byte == size // 2,
        f"[{snap.first_byte}, {snap.last_byte}) is "
        f"{snap.last_byte - snap.first_byte} bytes of the 4096 requested",
    )
    check(
        "and a consumer draining it gets those samples and nothing older",
        bytes(ring.read_newest(size)) == fresh[-size // 2 :],
        "the newest bytes are exactly the tail of the oversized write",
    )


# ---------------------------------------------------------------------------
# 12. seams survive the buffer turning over
# ---------------------------------------------------------------------------


def case_seams_survive_wrap() -> None:
    print("\n12. a seam is still reported after the ring has wrapped")
    size = 4096
    ring = sdr.IQRing(size)
    # One producer change, then enough of the second producer's stream to turn
    # the buffer over several times -- which is what pushes the seam below the
    # floor the tag list is pruned against.
    ring.write(bytes([1]) * 3000, tune_id=1, stream_id=1)     # tags: [(0, t1/s1)]
    for _ in range(3):
        ring.write(bytes([2]) * 3000, tune_id=2, stream_id=2)
    ring.write(bytes([2]) * 1000, tune_id=2, stream_id=2)
    tags = ring.diag_seams()
    check(
        "the seam is still recorded with three ring-lengths written after it",
        any(p == 3000 for p, _t, _s in tags) and ring.total_written == 13000,
        f"{ring.total_written / size:.1f} ring-lengths written, tags="
        + ", ".join(f"@{p}:t{t}/s{s}" for p, t, s in tags),
    )
    snap = ring.snapshot_newest(size)
    check(
        "a read of the oldest retained bytes is labelled with the stream that "
        "wrote them",
        snap.first_tune_id == 2 and snap.first_stream_id == 2
        and snap.last_tune_id == 2 and snap.last_stream_id == 2,
        f"oldest byte {snap.first_byte} answered as tune "
        f"{snap.first_tune_id}/stream {snap.first_stream_id}, newest as "
        f"{snap.last_tune_id}/{snap.last_stream_id}",
    )

    # Now the producer is replaced. Everything the ring still holds was written
    # by the *old* process, and the seam it is now straddling was recorded four
    # ring-lengths ago -- below the floor, so this is where a prune that keeps
    # only the recent entries throws away the one record that can answer for the
    # retained bytes and answers them with the replacement's identity instead.
    ring.write(bytes([3]) * 2000, tune_id=3, stream_id=3)
    snap = ring.snapshot_newest(size)
    floor = ring.total_written - size
    check(
        "a seam recorded below the floor is still reported to a straddling read",
        snap.first_tune_id == 2 and snap.last_tune_id == 3
        and snap.first_stream_id == 2 and snap.last_stream_id == 3,
        f"read [{snap.first_byte}, {snap.last_byte}) straddles the seam at "
        f"3000: tune {snap.first_tune_id}->{snap.last_tune_id}, stream "
        f"{snap.first_stream_id}->{snap.last_stream_id} (floor {floor})",
    )
    check(
        "and the identity in force at the head of the ring is the anchor, not "
        "the replacement's",
        any(p < floor and t == 2 and s == 2 for p, t, s in ring.diag_seams()),
        "tags below the floor: "
        + ", ".join(f"@{p}:t{t}/s{s}" for p, t, s in ring.diag_seams() if p < floor),
    )
    check(
        "the buffer really did turn over more than once on the way there",
        ring.total_written > 3 * size,
        f"{ring.total_written} bytes written into a {size}-byte ring "
        f"({ring.total_written / size:.1f} wraps)",
    )
    raw = ring.read_newest(size)
    check(
        "and the retained bytes are one contiguous run of the two producers",
        raw == bytes([2]) * 2096 + bytes([3]) * 2000,
        f"{len(raw)} bytes handed out, patterns {sorted(set(raw))} "
        f"(stream 2's tail then the replacement's whole write)",
    )


# ---------------------------------------------------------------------------


def main() -> int:
    print("Stream continuity check")
    t0 = time.perf_counter()
    ntsc = composite_iq(NTSC_SAMPLES)
    pal = composite_iq(PAL_SAMPLES, ntsc=False)
    print(f"  captured {ntsc.size} NTSC and {pal.size} PAL simulator samples "
          f"in {time.perf_counter() - t0:.1f}s")
    if ntsc.size < 4 * CHUNK + 2_000_000 or pal.size < 262_144 + CHUNK:
        print(f"  the simulator produced too little to test with (wanted "
              f"{4 * CHUNK + 2_000_000} NTSC and {262_144 + CHUNK} PAL)")
        return 2
    case_equivalence(ntsc)
    case_gaps(ntsc)
    case_identities(ntsc)
    case_file_loop(ntsc)
    case_capacity(ntsc, pal)
    case_burst(ntsc)
    case_freshness(ntsc)
    case_accounting(ntsc)
    case_snapshot_neutrality(ntsc)
    case_overlap(ntsc)
    case_whole_samples()
    case_oversized_write()
    case_seams_survive_wrap()
    case_equivalence_real_capture()
    print("\n" + ("RESULT: FAIL -- " + ", ".join(FAILURES) if FAILURES else "RESULT: PASS"))
    for line in SKIPPED:
        print(f"  skipped: {line}")
    print(f"  {time.perf_counter() - t0:.1f}s total")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())