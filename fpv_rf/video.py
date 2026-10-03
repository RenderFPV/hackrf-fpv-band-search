"""Live decode pipeline: IQ in, 60 fps luminance frames out.

Threading model
---------------
One background thread (:class:`DecodeWorker`) owns all decoding. The GUI never
touches the sample path -- it only ever reads the most recently published frame.
That keeps the numpy work off Qt's thread (which would stall painting) and means
a slow field simply means the *previous* frame stays on screen, which is the
right behaviour for a live view.

Why the decoder consumes a stream rather than a snapshot
--------------------------------------------------------
The FM discriminator is a phase difference, so it carries state from one sample
to the next. Feeding it a fresh overlapping snapshot each time re-does the same
arithmetic and puts a false edge at the seam. :meth:`IQSource.drain_block` hands
over only newly produced samples, so every sample is demodulated exactly once
and a rolling demodulator buffer is continuous.

Continuity has to be *proved*, not assumed
------------------------------------------
Exactly once is not the same as in order. Handing over only new samples means the
seams land wherever the delivery boundaries happen to fall, and two deliveries
are adjacent only if the second starts where the first ended. Nothing in the
samples can say whether it does: a dropped second of video and a continuous
stream look alike to a discriminator, and a stream that restarted at the same
frequency looks like one that never stopped. So the drain reports an absolute
range and the identity of the producer at each end, and the worker joins two
blocks only when all of that lines up:

* ``block.first_sample == previous.last_sample`` -- no gap and no overlap;
* the stream identity is unchanged -- no transfer-process restart, no replayed
  file passing its end;
* the tuning identity is unchanged -- no retune inside the window;
* and the block itself holds one identity at both ends, so it is not a mixture
  of two producers' samples.

When any of that fails, the decoder state is reset and a new segment begins,
which costs one field of picture and keeps everything after it honest. The
missing range is counted rather than absorbed. Blocks that straddle a transition
internally are refused outright: the samples on either side of it are each fine,
and together they are a picture assembled out of two different pictures.

Cost, honestly
--------------
At 10 MS/s a field is ~166k samples. Demodulating only the new samples makes a
field cost roughly 4 ms of numpy, plus ~3 ms to detect sync over the rolling
buffer and ~2 ms to rasterise -- comfortably inside the 16.7 ms field budget. The
decode side can therefore decode a field in well under its own duration, and the
panel repaints at up to 60 fps from whatever the newest published frame is. The
*decode* cadence is set separately, at :data:`PREVIEW_HZ`: running it at the
field rate decodes more pictures than any display can show, and every one of
them is a second pass over a rolling window that the previous pass already
read. Measured rates are exposed through :class:`DecodeStats` rather than
assumed, so the UI can show what is actually happening.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import diag, dsp
from .sdr import IQBlock, IQSource

#: How long a delivered block of IQ is, and how long the rolling demodulator
#: window is. One duration for both, because they answer the same question: how
#: much continuous stream is enough to assemble a field and not so much that a
#: decoder which has fallen behind is handed an unbounded backlog.
#:
#: 42 ms is 2.5 NTSC fields (16.7 ms) and 2.1 PAL fields (20 ms). Both matter:
#: PAL's 312-line field is ~200k samples at 10 MS/s and the rasteriser needs a
#: complete contiguous run of 288 lines out of the window, so a window that only
#: covered an NTSC field's worth could never assemble a PAL frame at all. The
#: earlier 320k window had 2.0 NTSC fields and 1.6 PAL, and the earlier
#: 16.7 ms *cap* took a PAL stream's whole active picture and threw the rest
#: away, so a locked PAL picture could never survive one iteration.
#:
#: 320k samples is the reference value at 10 MS/s; a source at another rate gets
#: the same duration scaled to it, because it is a time and not a sample count.
WINDOW_S = 0.042
DEFAULT_WINDOW = 420_000


def window_samples(sample_rate: int) -> int:
    """Samples in one decoder window at ``sample_rate``.

    Named rather than inlined because two callers need the same number -- the
    decoder's default window, and the publication capture's ledger, which has to
    be sized against the window it exists to describe. Spelled twice it would
    eventually be spelled differently, and the ledger would then quietly stop
    covering the oldest lines of the picture.
    """
    return int(round(WINDOW_S * int(sample_rate)))

#: Smallest block worth demodulating. Below this the discriminator's output is
#: mostly the boundary step, and the rolling window gains nothing but a seam.
MIN_BLOCK_SAMPLES = 64

#: Decode cadence. Deliberately half the field rate and independent of it: the
#: panel repaints at 60 fps from the newest published frame, so decoding faster
#: than this produces pictures no display can show, while decoding slower throws
#: away fields the window could have assembled. 30 Hz costs ~8 ms of CPU per
#: second less than 60 and leaves the newest picture ~16 ms old.
PREVIEW_HZ = 30.0

#: Field intervals, used for the missed-field accounting. **Separate from
#: :data:`PREVIEW_HZ` on purpose.** Counting missed fields against the decode
#: cadence rather than the standard makes the number a property of this
#: program's pacing instead of the transmitter's: at 30 Hz every second field is
#: "missed" by construction, which is exactly the sort of counter that looks
#: healthy because it is always small and means nothing.
NTSC_FIELD_INTERVAL_S = 1.0 / 59.94
PAL_FIELD_INTERVAL_S = 1.0 / 50.0

#: How old a published frame may be before it stops counting as live video.
#:
#: NTSC analogue video is 59.94 fields per second, so a field is 16.7 ms and a
#: whole frame is 33.3 ms. Half a second is therefore about fifteen frames of
#: slack: long enough that an ordinary hiccup -- one lost field, a retune, the
#: decoder taking a moment to re-lock -- does not flicker the panel or flap the
#: status line, and short enough that a transmitter which has gone away, or a
#: radio which has been unplugged, stops being described as locked well before
#: anyone could mistake the frozen picture for a live one.
STALE_FRAME_S = 0.5


@dataclass
class DecodeStats:
    frames: int = 0
    fields_missed: int = 0
    locked_frames: int = 0
    dropped_iq_samples: int = 0
    drain_ms: float = 0.0
    decode_ms: float = 0.0
    cycle_ms: float = 0.0
    last_frame_at: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    line_rate_hz: float = 0.0
    sync_quality: float = 0.0
    standard: str = ""
    chunk_samples: int = 0
    #: Decode exceptions that were caught and survived. Non-zero means the
    #: decoder itself is misbehaving, which is a different fault from "no video
    #: is arriving" and was previously invisible.
    decode_errors: int = 0
    last_decode_error: str = ""
    #: Scanlines of the most recent field that were placed on the fitted line grid
    #: because no sync pulse was detected on them. The samples there are real
    #: video; only the lines' positions were reconstructed. Reported because a
    #: field that needed 40 of 240 positions reconstructed is a weaker claim than
    #: one that needed none, and a viewer is entitled to know which they have.
    lines_from_grid: int = 0
    #: Times a block of IQ was discarded because it straddled a retune. Each one
    #: is a field of real video thrown away on purpose -- the samples either side
    #: of a retune are from two different transmitters and the discriminator
    #: cannot treat them as one stream -- so the cost of staying correct is worth
    #: showing rather than hiding.
    tune_seams: int = 0

    # -- delivery accounting -------------------------------------------------
    #
    # Four totals, and they are four because they answer four different
    # questions. Collapsing them is how a decode that is quietly being starved
    # can report a healthy frame rate: frames are counted from *published*
    # frames, and a publisher that is fed nothing looks identical to one that is
    # fed everything and chose to draw nothing.
    #
    #   written    the source produced this many samples
    #   delivered  the drain handed this many to the worker
    #   accepted   the decoder took this many into its demodulator
    #   rejected   delivered but refused: an internal seam, a stale generation,
    #              an out-of-range block, or a decode that raised
    #
    # delivered == accepted + rejected, always. written - delivered is what never
    # arrived, and it splits into two different things which are counted
    # separately because they mean opposite things: a *capped skip* is the drain
    # discarding a backlog on purpose to stay current, and a *ring overrun* is
    # the ring destroying data no consumer had read.

    #: Samples the source produced, from the ring's absolute write counter.
    iq_written: int = 0
    #: Samples the drain handed over.
    iq_delivered: int = 0
    #: Samples the decoder demodulated.
    iq_accepted: int = 0
    #: Samples delivered and then refused, with the reasons below.
    iq_rejected: int = 0
    #: Blocks refused because they hold more than one identity internally.
    seam_blocks: int = 0
    #: ... of those, the cause was a retune rather than a process restart.
    tune_seams_internal: int = 0
    #: Blocks refused because their range could not be true.
    oob_samples: int = 0
    #: Samples of *overlap* between consecutive blocks. Counted apart from a gap
    #: and never concatenated: a block that reaches back into what has already
    #: been demodulated is a repeat, and the only honest response is to refuse
    #: it and start again.
    overlap_samples: int = 0
    #: Times delivery did not continue where it left off, whatever the cause.
    discontinuities: int = 0
    #: The causes, by name, so "how many gaps" can be answered by *what* broke.
    #: Bounded by construction: one of seven fixed names.
    gap_causes: dict = field(default_factory=dict)
    #: Samples that never arrived between two accepted blocks. Distinct from a
    #: skip and from an overrun, and distinct again from a rejected block, whose
    #: samples did arrive and are counted as rejected.
    missing_samples: int = 0
    #: Drain hand-overs that dropped a backlog on purpose.
    capped_skips: int = 0
    capped_skip_samples: int = 0
    #: Ring overruns -- bytes destroyed before any consumer read them.
    ring_overruns: int = 0
    ring_overrun_samples: int = 0
    #: Uninterrupted segments started. One after the first block, one after every
    #: discontinuity.
    segments: int = 0

    # -- decode accounting ---------------------------------------------------

    #: Demodulator output samples produced. One uninterrupted segment of N IQ
    #: yields exactly N-1 of them: the first sample of a segment has no
    #: predecessor, and every later block contributes its own length by way of
    #: the boundary step that joins it to the one before.
    fm_samples: int = 0
    #: Times the demodulator's state was thrown away -- a reset, or a decode that
    #: raised. A non-zero value is normal after a retune and alarming otherwise.
    fm_resets: int = 0
    #: Boundary steps computed across a proven-adjacent block boundary, and
    #: blocks refused as mixtures. The two sum to the number of times the
    #: decoder had to be told where a stream began.
    boundary_steps: int = 0
    #: Times sync detection ran, and how many of those the gate accepted.
    sync_attempts: int = 0
    sync_accepted: int = 0
    #: Fields the rasteriser drew, and how many pushes produced none.
    raster_accepts: int = 0
    raster_rejects: int = 0
    #: Rows placed on the fitted grid rather than at a detection, summed.
    grid_rows: int = 0
    #: Published frames, and how many of those were *new*. The difference is
    #: frames that repeated an interval already published -- which the rolling
    #: window makes possible, since it still holds the field it last drew.
    unique_frames: int = 0
    dup_frames: int = 0
    #: Absolute sample interval ``[first, last)`` of the newest published picture.
    frame_interval: tuple = (-1, -1)

    def frame_rate(self) -> float:
        el = max(1e-6, time.monotonic() - self.started_at)
        return self.frames / el

    def miss_rate(self) -> float:
        total = self.locked_frames + self.fields_missed
        return (self.fields_missed / total) if total else 0.0

    def note_gap(self, cause: str) -> None:
        """Count one discontinuity by cause."""
        self.discontinuities += 1
        self.gap_causes[cause] = self.gap_causes.get(cause, 0) + 1


@dataclass
class VideoFrame:
    """One published field, ready to paint."""

    image: np.ndarray            # uint8 (H, W)
    timestamp: float
    line_rate_hz: float
    sync_quality: float
    standard: str
    locked: bool
    lines_from_grid: int = 0
    #: Which decoder epoch produced this frame. Bumped by
    #: :meth:`DecodeWorker.reset`, so a consumer can tell a frame that survived a
    #: retune from one produced afterwards; appended last so that positional
    #: construction of a frame is unchanged for anything building one directly.
    generation: int = 0
    #: Absolute half-open sample interval ``[iq_first, iq_last)`` of the stream
    #: this picture was drawn from, or ``(-1, -1)`` when the source could not say
    #: where its samples were. This is what makes "is this a new picture"
    #: answerable: two frames of the same field carry the same interval, so
    #: re-publishing one as new can be detected rather than guessed at from a
    #: timestamp that is fresh by construction.
    iq_first: int = -1
    iq_last: int = -1
    #: Uninterrupted stretch of stream this belongs to, counted from 0 as the
    #: decoder's segments begin. Frames from different segments can never be
    #: joined, so this is the number to compare before treating two pictures as
    #: consecutive.
    segment: int = 0

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def located(self) -> bool:
        """Whether the absolute interval of this frame is known."""
        return self.iq_first >= 0 and self.iq_last > self.iq_first


def _sync_capture(sync: dsp.SyncResult) -> dict:
    """TEMP DIAG: one sync result as a JSON-ready dict, both polarities included.

    The losing polarity is the point. "Which extreme did it pick" is a question
    with a short answer; "what did the other extreme look like, and would it have
    been better" is the question the live-lock dispute is actually about, and it
    cannot be answered from a record that only kept the winner.
    """
    sd = sync.diag
    out: dict = {
        "selected_polarity": int(sync.polarity),
        "threshold": float(sync.threshold),
        "quality": float(sync.quality),
        "line_samples": float(sync.line_samples),
        "line_rate_hz": float(sync.line_rate_hz),
        "standard": str(sync.spec.name),
        "pulses": int(sync.pulse_starts.size),
    }
    if sd is not None:
        out["both"] = sd.as_dict()
    return out


@dataclass
class Continuity:
    """Where the decoder's accepted samples end, and what produced them.

    This is the *worker's* side of the continuity contract. The decoder keeps its
    own predecessor sample; what it cannot know on its own is whether the block
    now in hand continues the block before it. That answer needs the absolute
    ranges and the producer identities of both, so it is kept here, beside the
    drain, and is only ever updated under the decoder lock -- together with the
    decode that it describes, so the two cannot disagree.

    ``end`` is the exclusive end of the last block the decoder took, which is what
    the next block's ``first_sample`` is compared against. ``open`` marks a span
    whose samples were *not* decoded -- a refused block, a stale generation, a
    reset. Delivery may well continue from there, and ``end`` says exactly where,
    but the next block is not adjacent to anything the decoder holds and has to
    start a new segment. Keeping ``end`` correct even then is what stops a
    refused block's samples from being counted twice: once as rejected and again
    as missing.
    """

    end: int = -1
    stream_id: int = -1
    tune_id: int = -1
    segment: int = 0
    have: bool = False
    open: bool = False

    @property
    def identity(self) -> tuple[int, int]:
        return (self.stream_id, self.tune_id)


class VideoDecoder:
    """Stateful, incremental composite-video decoder.

    Holds the rolling demodulator buffer and the rasteriser. Not thread-safe by
    itself; :class:`DecodeWorker` owns one instance on its own thread.
    """

    def __init__(
        self,
        sample_rate: int,
        width: int = 160,
        out_h: int | None = None,
        window: int | None = None,
        iq_ledger: "diag.IqLedger | None" = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        # A duration, scaled to the source's rate. 42 ms at 10 MS/s is the
        # reference; a source at 20 MS/s gets 42 ms of samples too, which is the
        # whole point of expressing it in time rather than in samples.
        self.window = (int(window) if window is not None
                       else window_samples(self.sample_rate))
        self.raster = dsp.Rasteriser(width=width, out_h=out_h)
        self._demod = np.zeros(0, dtype=np.float32)
        #: Absolute index, in source samples, of ``_demod[0]``. ``-1`` when the
        #: caller could not place the samples in the stream. Tracked as an
        #: absolute number and advanced by whatever leaves the head of the
        #: window, so a picture can always be traced back to the samples it came
        #: from even after the buffer has turned over many times.
        self._demod_origin = -1
        #: Final IQ sample of the last accepted block, the discriminator's
        #: predecessor. ``None`` after a reset, which is what makes a reset
        #: unforgeable: there is no predecessor, so the next block's first step
        #: cannot be a continuation of anything.
        self._last_iq: np.complex64 | None = None
        #: Demodulator output produced since the decoder was built, so the
        #: invariant "N IQ in one segment gives N-1 FM samples out" is a number
        #: that can be checked rather than a claim.
        self.fm_samples = 0
        self.fm_resets = 0
        self.boundary_steps = 0
        #: Sync-gate and rasteriser accounting, kept on the decoder because only
        #: the decoder can tell whether detection ran at all. They used to be
        #: visible only through the frames that came out of them, which made the
        #: interesting case -- sync found on every push and no frame produced --
        #: report nothing. Mirrored into :class:`DecodeStats` each iteration.
        self.sync_attempts = 0
        self.sync_accepted = 0
        self.raster_accepts = 0
        self.raster_rejects = 0
        self.grid_rows = 0
        self.brightness = 1.0
        self.contrast = 1.0
        self.invert = False
        self.line_filter_hz: float | None = None
        # TEMP DIAG: see fpv_rf/diag.py. Counts only; the array statistics it
        # also gathers run only when FPV_RF_DIAG is set.
        self._diag = diag.section("dec")
        # TEMP DIAG: the bounded ledger of accepted blocks behind the current
        # window. Passed in rather than created here so the capture can be
        # switched off from outside without the decoder having to own the policy;
        # ``None`` when it is off, and every use below is behind that check.
        self.ledger = iq_ledger
        #: TEMP DIAG: the sync result and rasteriser frame of the newest
        #: published picture, held so a capture can describe the *decision* that
        #: produced it rather than re-deriving one. Both are single small objects
        #: and the previous ones are replaced, not accumulated.
        self._pub_sync: "dsp.SyncResult | None" = None
        self._pub_raster = None

    def reset(self) -> None:
        """Forget everything: window, predecessor, rasteriser, provenance.

        Called when the stream is not continuous with what came before, and on an
        explicit reset. The predecessor goes because that is the state the FM
        discriminator would otherwise carry across a gap: keeping it would make
        the first step after a discontinuity read as a phase step across it,
        which is the exact artefact -- a sync pulse that is not in the signal --
        that a decoder must never manufacture.
        """
        self._demod = np.zeros(0, dtype=np.float32)
        self._demod_origin = -1
        self._last_iq = None
        self.raster = dsp.Rasteriser(
            width=self.raster.width, out_h=self.raster.out_h
        )
        # TEMP DIAG: the ledger is dropped with everything else. It describes
        # samples the decoder no longer holds, and a capture that reached back
        # into them would pair a picture with a window that is not the one it was
        # drawn from.
        if self.ledger is not None:
            self.ledger.clear()
        self._pub_sync = None
        self._pub_raster = None
        # Counted here rather than at the call site, because the decoder is the
        # only thing that can say its own state was thrown away -- and a test
        # double with a no-op ``reset`` should not be able to make the counter
        # lie either way.
        self.fm_resets += 1
        self._diag.bump("decoder_resets")        # TEMP DIAG

    def _diag_sync(self, sync: dsp.SyncResult) -> None:
        """TEMP DIAG: record the sync decision whether or not a frame follows.

        The existing counters only advanced when a frame was published, so a
        decoder that detected sync on every line and then refused to rasterise
        reported nothing at all -- and that is precisely the case where the
        question is why. This is called on both exits from the sync gate.
        """
        d = self._diag
        d.gauge("sync_pulses", int(sync.pulse_starts.size))
        d.gauge("sync_quality", float(sync.quality))
        d.gauge("sync_line_samples", float(sync.line_samples))
        d.gauge("sync_line_rate_hz", float(sync.line_rate_hz))
        d.gauge("sync_polarity", int(sync.polarity))
        d.gauge("sync_threshold", float(sync.threshold))
        sd = sync.diag
        if sd is not None:
            d.put("sync", sd.flatten())

    @property
    def last_iq(self) -> np.complex64 | None:
        """Final IQ sample of the last accepted block, or None after a reset."""
        return self._last_iq

    # -- TEMP DIAG: publication capture -------------------------------------

    def publication(self, frame: VideoFrame) -> "diag.PubCap | None":
        """Everything one publication needs beside it, as one snapshot.

        Called at the publication decision and **outside** both locks that matter:
        the worker holds neither ``_decoder_lock`` nor the ring's while this runs,
        because a 2.5 MB copy under either would hold up the next field for longer
        than a field lasts. The copy is safe without them precisely because this
        runs on the decode thread between two pushes: nothing else mutates the
        decoder's window, and the array handed back is a copy, so the next push
        cannot change it underfoot.

        The IQ half is taken from the ledger by absolute range rather than from
        the ring, because the ring's contents at this instant are *not* the samples
        the picture came from -- the decoder is behind the drain by at least one
        block, and a capture taken from the newest bytes would pair a picture with
        a different part of the stream.
        """
        sync = self._pub_sync
        raster = self._pub_raster
        if sync is None or raster is None or self.ledger is None:
            return None
        cap = diag.PubCap()
        cap.frame = frame
        cap.image = frame.image
        # The window's *last* output needs the one sample after it, which is the
        # final sample of the newest block -- already inside the slice below.
        cap.demod = self._demod.copy()
        cap.demod_origin = int(self._demod_origin)
        first = int(self._demod_origin)
        count = int(self._demod.size)
        cap.spans = tuple(self.ledger.slice(first, count))
        cap.sync = _sync_capture(sync)
        cap.positions = raster.line_positions
        cap.lines_used = int(raster.lines_used)
        cap.lines_detected = int(raster.lines_detected)
        cap.lines_from_grid = int(raster.lines_from_grid)
        cap.sample_rate = int(self.sample_rate)
        cap.line_filter_hz = self.line_filter_hz
        return cap

    @property
    def demod_origin(self) -> int:
        """Absolute sample index of ``_demod[0]``, or ``-1`` if unplaced."""
        return self._demod_origin

    def push_iq(
        self,
        iq: np.ndarray,
        *,
        continuous: bool = False,
        first_sample: int | None = None,
        provenance: dict | None = None,
    ) -> VideoFrame | None:
        """Demodulate newly arrived samples and emit a frame if one is ready.

        ``continuous`` is the caller's *proof* that these samples continue the
        previous block -- the same samples, in order, with nothing missing
        between them -- and never an assumption. The decoder does not infer it:
        :class:`DecodeWorker` has the ranges and the producer identities and is
        the only thing that can establish it. That matters because the boundary
        step is the one piece of arithmetic that cannot be recovered later: it is
        the phase difference across the join, and without the sample before the
        join there is nothing to compute it from. So one uninterrupted segment of
        N IQ samples yields exactly N-1 demodulator outputs -- the first sample of
        a segment has no predecessor and contributes none, and every block after
        it contributes its whole length by way of the step that joins it to the
        one before.

        ``first_sample`` is where this block starts in the source, in absolute
        samples. It is provenance, not control: it is what lets a published frame
        say which part of the stream it came from, and it is ignored for anything
        the samples themselves decide.

        ``provenance`` (TEMP DIAG) is the rest of what the caller knows about the
        block -- stream and tuning id, delivery number, whether this block starts
        a segment, why a reset happened, how much the drain skipped. It is
        recorded on the ledger and nothing else reads it: the decoder's decisions
        are made from the samples and the join proof alone, and widening that
        input is exactly the change that could make a capture describe a different
        decode from the one that ran.
        """
        d = self._diag
        d.bump("pushes")
        iq = np.asarray(iq)
        if iq.size < 8 or not np.iscomplexobj(iq):
            # Too short for the discriminator (which needs a pair at all) or not
            # IQ. Either way nothing is accepted, and in particular the
            # predecessor is left alone: a block that was not decoded cannot be
            # the block the next one follows.
            d.bump("push_too_short")
            return None
        new = dsp.fm_demodulate(iq)               # n-1 steps inside the block
        d.bump("samples_in", iq.size)
        # TEMP DIAG: whether a boundary step was actually taken, as opposed to
        # whether one was asked for. Read by the ledger below, and it is the
        # distinction that decides whether this block's first demodulator output
        # is reproducible from the capture alone.
        stepped = continuous and self._last_iq is not None
        if stepped:
            # The step across the join, which is the one output a per-block
            # discriminator cannot produce. Prepended, so the block contributes
            # exactly ``n`` and the segment's total stays N-1.
            #
            # Computed by the *same expression* ``dsp.fm_demodulate`` uses --
            # ``angle(pair[1:] * conj(pair[:-1]))``, applied to the two-sample
            # pair -- rather than as a differently spelled copy. That is not
            # tidiness. The copy that was here had the arguments the other way
            # round, ``angle(prev * conj(first))``, which is the *negative* of
            # the step the one-shot discriminator produces at that position, so
            # every block boundary contributed one sample of the right magnitude
            # and the wrong sign; the incremental decode was not equal to the
            # monolithic one it exists to reproduce, and the rendered picture
            # differed from it by hundreds of pixels per field. Written this
            # way it is not merely close to the one-shot value, it is the same
            # float32: the pair goes through the same array path, so the result
            # is bit-identical rather than one unit in the last place away.
            #
            # ``tools/check_stream_continuity.py`` pins that equality bit for
            # bit, so the two spellings cannot drift apart again unnoticed.
            pair = np.asarray([self._last_iq, iq[0]])
            head = np.angle(pair[1:] * np.conj(pair[:-1])).astype(np.float32)
            new = np.concatenate((head, new))
            self.boundary_steps += 1
            d.bump("boundary_steps")              # TEMP DIAG
        if self.ledger is not None:
            # TEMP DIAG: record the block as accepted, before anything downstream
            # can decline to publish a frame from it. The samples are in the
            # window from here on whether or not a picture comes out.
            #
            # ``_last_iq`` is the sample the boundary step above was computed
            # from, and it is kept with the block: without it the first
            # demodulator output of this block cannot be reproduced, and a replay
            # of the capture would differ from the live decode by one wrong
            # sample per block boundary -- precisely the small, plausible, wrong
            # difference this capture exists to rule out. It is recorded *only*
            # when a boundary step was actually taken. When the block starts a
            # segment the predecessor describes the previous stream and belongs
            # with it: carrying it over would manufacture a continuity the
            # continuity contract had already refused to assert.
            prov = provenance or {}
            self.ledger.add(diag.IqSpan(
                iq=np.asarray(iq, dtype=np.complex64),
                first_sample=(-1 if first_sample is None
                              else int(first_sample)),
                stream_id=int(prov.get("stream_id", -1)),
                tune_id=int(prov.get("tune_id", -1)),
                segment=int(prov.get("segment", 0)),
                delivery=int(prov.get("delivery", -1)),
                starts_segment=bool(prov.get("starts_segment", False)),
                predecessor=(np.complex64(self._last_iq) if stepped else None),
                reset_reason=str(prov.get("reset_reason", "")),
                skipped_samples=int(prov.get("skipped_samples", 0)),
            ))
        if self._demod.size:
            self._demod = np.concatenate((self._demod, new))
        else:
            self._demod = new
            placed = first_sample if first_sample is not None and first_sample >= 0 else -1
            if placed >= 0:
                # The first output of a segment is the step starting at the
                # segment's first sample; with a boundary step in front of it,
                # that is the sample *before* the block's own first one.
                self._demod_origin = placed - 1 if continuous else placed
        self.fm_samples += int(new.size)
        d.bump("fm_samples", new.size)
        if self._demod.size > self.window:
            # The rolling buffer is the concatenation of everything decoded so
            # far, and dropping its head is the only place a sample can vanish
            # between the ring and the rasteriser. Counted, not reasoned about.
            excess = self._demod.size - self.window
            d.bump("window_dropped_samples", excess)
            self._demod = self._demod[excess:]
            if self._demod_origin >= 0:
                self._demod_origin += excess
        d.gauge("demod_window", int(self._demod.size))
        # The predecessor is this block's last sample, whatever happens next:
        # the samples have been accepted into the window, and whether a frame
        # comes out of them is a separate question that does not un-accept them.
        self._last_iq = np.complex64(iq[-1])
        if diag.enabled():
            # TEMP DIAG: min/max/mean/std and the near-pi share of the *new*
            # block (not the window), so a wrapping or saturated input shows up
            # before the rolling average hides it.
            d.put("demod", diag.signal_stats(new))
            d.put("iq", diag.iq_stats(iq))
        if diag.dump_enabled():
            diag.dump_add("demod", new, {
                "sample_rate": self.sample_rate,
                "iq_samples": int(iq.size),
                "demod_window": int(self._demod.size),
            })

        if self._demod.size < 32_000:
            d.bump("window_too_short")
            return None

        sig = self._demod
        if self.line_filter_hz:
            sig = dsp.limit_video_bandwidth(self._demod, self.sample_rate, self.line_filter_hz)
        # ``sig`` is filtered into a *new* array and ``_demod`` is never
        # overwritten, so the window the publication capture copies is the
        # discriminator's own output. That is deliberate and it is the honest
        # choice: the filter is a function of ``line_filter_hz`` and the rate,
        # both recorded, so the filtered window the decoder actually read can be
        # recomputed from the capture. Capturing the filtered window instead would
        # make a replay depend on a filter's implementation details to reproduce
        # a picture, and would hide what the discriminator produced.

        # Where the window this picture is drawn from starts in the source, or
        # ``-1`` when the caller could not place the samples -- passed straight
        # through, because "I cannot say where these samples are" and "these
        # samples start the stream" must not look alike.
        self.sync_attempts += 1
        sync = dsp.detect_sync(
            sig, self.sample_rate, polarity="auto",
            # TEMP DIAG: the publication capture needs both polarities' numbers,
            # so it asks for them whether or not the once-a-second aggregate line
            # is wanted. It does not change what comes back -- ``collect_diag``
            # fills a record and honours the live early-break -- which is pinned
            # by tools/check_decode_diag.py, so asking for it cannot alter the
            # decode it describes.
            collect_diag=diag.enabled() or self.ledger is not None,
        )
        self._diag_sync(sync)
        if (sync.pulse_starts.size < 8 or sync.quality < 0.60
                or not .55 <= sync.pulse_starts.size * sync.line_samples / sig.size <= 1.25):
            d.bump("sync_rejected")
            d.bump(
                "sync_rejected_few_pulses" if sync.pulse_starts.size < 8
                else "sync_rejected_quality"
            )
            return None
        self.sync_accepted += 1
        d.bump("sync_accepted")
        fr = self.raster.push(
            sig,
            sync,
            brightness=self.brightness,
            contrast=self.contrast,
            invert=self.invert,
            origin=self._demod_origin,
        )
        if fr is None:
            self.raster_rejects += 1
            d.bump("raster_none")
            return None
        self.raster_accepts += 1
        self.grid_rows += int(fr.lines_from_grid)
        d.bump("frames_decoded")
        d.gauge("lines_from_grid", int(fr.lines_from_grid))
        # TEMP DIAG: hold the decision, not just the picture. The capture has to
        # describe the sync that passed the gate and the line geometry the
        # rasteriser chose, and by the time the worker publishes, both of these
        # are local variables in a frame that has gone. Kept as references to
        # objects the decoder already holds -- nothing is copied here, and the
        # previous pair is replaced rather than accumulated.
        self._pub_sync = sync
        self._pub_raster = fr
        return VideoFrame(
            image=fr.image,
            timestamp=time.monotonic(),
            line_rate_hz=fr.line_samples and self.sample_rate / fr.line_samples or 0.0,
            sync_quality=sync.quality,
            standard=fr.spec.name,
            locked=True,
            lines_from_grid=fr.lines_from_grid,
            iq_first=fr.first_sample,
            iq_last=fr.last_sample,
        )


class DecodeWorker:
    """Background thread: drain IQ, decode, publish the newest frame."""

    def __init__(self, source: IQSource, width: int = 160, out_h: int | None = None) -> None:
        self.source = source
        # Decode cadence, and the one that used to set the chunk cap as well.
        # Pacing at the *field* rate while capping at 42 ms of IQ made the two
        # disagree: a PAL stream has 2.1 fields in every chunk and 1.5 fields
        # arrive per second-slice at 30 Hz, so the drain spent part of each
        # iteration throwing away samples it had just taken, and a 42 ms chunk is
        # not a field either. One number now sets both: 42 ms of stream per
        # iteration, every 1/30 s, which is 1.26 fields' worth at 10 MS/s --
        # enough to assemble a picture, not enough to grow a backlog.
        self._max_chunk = int(source.sample_rate * WINDOW_S)
        # TEMP DIAG: the ledger exists only while the publication capture is on,
        # and is handed to the decoder rather than created inside it, so switching
        # the capture off is a decision taken once here instead of a test the
        # decoder has to make on every field. Sized against the same two numbers
        # the decoder and the drain use, so it covers the window the decoder
        # actually holds -- see ``diag.pubcap_ledger``.
        self.ledger = (diag.pubcap_ledger(window_samples(source.sample_rate),
                                          self._max_chunk)
                       if diag.pubcap_enabled() else None)
        self.decoder = VideoDecoder(source.sample_rate, width=width, out_h=out_h,
                                    iq_ledger=self.ledger)
        self.stats = DecodeStats()
        self._lock = threading.Lock()
        self._frame: VideoFrame | None = None
        self._published_at = 0.0
        #: Serialises access to the decoder itself, which both the decode loop and
        #: any reset from the GUI thread mutate. Separate from ``_lock``, which
        #: only guards the published frame.
        self._decoder_lock = threading.Lock()
        #: Bumped on every reset. The decode loop checks it inside
        #: ``_decoder_lock`` before feeding samples in and again before
        #: publishing, so a reset that lands mid-iteration cannot have its work
        #: published afterwards. Without the first check the freshly reset
        #: decoder was fed a block captured before the retune -- the exact
        #: two-transmitter picture :meth:`reset` exists to prevent -- and without
        #: the second, a frame built from the old frequency was published after
        #: ``_frame`` had been cleared, leaving the old picture on screen again.
        self._generation = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_decode = 0.0
        # Decode cadence, and the one that used to set the chunk cap as well.
        # Pacing at the *field* rate while capping at 42 ms of IQ made the two
        # disagree: a PAL stream has 2.1 fields in every chunk and 1.5 fields
        # arrive per second-slice at 30 Hz, so the drain spent part of each
        # iteration throwing away samples it had just taken, and a 42 ms chunk is
        # not a field either. One number now sets both: 42 ms of stream per
        # iteration, every 1/30 s, which is 1.26 fields' worth at 10 MS/s --
        # enough to assemble a picture, not enough to grow a backlog.
        self._preview_interval = 1.0 / PREVIEW_HZ
        self._max_chunk = int(source.sample_rate * WINDOW_S)
        #: Where the decoder's accepted samples end, and what produced them.
        #: Updated only under ``_decoder_lock``, together with the decode it
        #: describes, so the two cannot disagree about which samples the decoder
        #: holds.
        self._cont = Continuity()
        #: Absolute end of the newest published picture, or ``-1``. A frame
        #: whose interval ends at or before this has been shown already: the
        #: rolling window still holds the field it last drew, so the rasteriser
        #: can legitimately produce it again, and publishing it as new would
        #: inflate the frame rate with pictures nobody has waited for.
        self._published_end = -1
        # TEMP DIAG: see fpv_rf/diag.py. ``_diag`` drives the one-per-second
        # aggregate line, so this loop is the clock the whole thing runs on.
        self._diag = diag.section("worker")

    # -- frame access (safe from any thread) --------------------------------

    def latest(self) -> VideoFrame | None:
        """Newest frame, or None if it has gone stale.

        Staleness is a decision made here rather than at paint time, so that
        every consumer gets the same answer. The previous version returned the
        last frame unconditionally, which meant a receiver that stopped or a
        transmitter that vanished left the last picture on screen indefinitely
        while the status line still said "locked" -- the two together read as a
        live picture of a channel that was no longer there.

        The frame itself is kept, so a brief drop-out (a retune, a field lost to
        a scheduling hiccup) does not flash the panel black; what changes is that
        the *lock* claim expires. The panel is told the age, so it can mark a
        frozen image rather than silently presenting it as current.
        """
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        if self.frame_age_s() > STALE_FRAME_S:
            return None
        return frame

    def frame_age_s(self) -> float:
        """Seconds since a frame was last published, or infinity if none ever was."""
        with self._lock:
            if self._frame is None:
                return float("inf")
            return max(0.0, time.monotonic() - self._published_at)

    @property
    def locked(self) -> bool:
        """Whether video is *currently* locked, judged on recency.

        Not "has ever locked". That was a cumulative counter, so the answer could
        never go back to False: switch channels, unplug the radio, and the app
        kept insisting it had a lock. Lock is a statement about the last fraction
        of a second, and it has to be able to lapse.
        """
        return self.frame_age_s() <= STALE_FRAME_S

    def reset_stats(self) -> None:
        """Start the counters again, and the picture with them.

        Distinct from :meth:`reset` in what it forgets: that one is a statement
        about the *frequency* and keeps the record of how far the stream got,
        this one is a statement about *the measurement* and forgets the record
        too, so the next hand-over is a first hand-over rather than a gap.
        """
        self.stats = DecodeStats()
        self._reset_decoder()
        self._cont = Continuity()
        self._published_end = -1

    def reset(self) -> None:
        """Forget the current frequency's picture.

        Called after a retune. Without it the panel keeps showing the previous
        channel until the new one locks, and a half-decoded frame straddling the
        two frequencies is published in the meantime -- a picture made of two
        different transmitters, which looks like a decoder fault rather than a
        channel change.

        The decoder is reset *through* :meth:`_reset_decoder`, which serialises on
        the same lock the decode loop takes around ``push_iq``. Resetting from the
        GUI thread while the decode thread was inside the decoder was a data race
        on the demodulator's internal state -- the frame lock only ever guarded
        the published frame, never the decoder, so it was not protecting the
        thing that needed protecting.

        The generation is bumped under ``_lock`` *before* the decoder is touched,
        so a decode already in flight sees a mismatch and throws its result away
        rather than publishing a frame built from the frequency we just left.

        What is *not* forgotten is where the stream had got to. The continuity
        record is left open rather than cleared, which does two things: the next
        block is forced to start a new segment instead of computing a boundary
        step across a retune, and the samples the retune discarded are still
        counted as missing when the next block arrives past them. Clearing it
        would have made every retune look like a fresh start and silently lost
        the size of every discard.
        """
        with self._lock:
            self._generation += 1
        self._reset_decoder()
        if self._cont.have:
            self._cont.open = True
        self._published_end = -1
        with self._lock:
            self._frame = None
            self._published_at = 0.0
        self.stats.locked_frames = 0
        self.stats.sync_quality = 0.0

    def _reset_decoder(self) -> None:
        """Run ``decoder.reset()`` with exclusive use of the decoder.

        Takes ``_decoder_lock``, so it must **not** be called while holding it:
        a plain ``Lock`` is not reentrant, and a decode thread that reset the
        decoder from inside its own critical section deadlocked on the next
        iteration's drain -- the whole picture, silently, after one field.

        Deliberately leaves the continuity record alone. That record says how far
        the *stream* got, which no reset un-knows: only the picture is discarded,
        and the next block still has to be measured against where the last one
        ended. :meth:`reset` and :meth:`reset_stats` move it themselves.
        """
        with self._decoder_lock:
            self.decoder.reset()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="decode", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- main loop ----------------------------------------------------------

    #: Decoder counters mirrored into :class:`DecodeStats`. By name, and read
    #: with ``getattr``, because a decoder can be replaced wholesale and a
    #: diagnostic counter is not a reason to fail a decode.
    _DECODE_COUNTERS = (
        "fm_samples",
        "fm_resets",
        "boundary_steps",
        "sync_attempts",
        "sync_accepted",
        "raster_accepts",
        "raster_rejects",
        "grid_rows",
    )

    #: Why a block could not be joined to the samples before it: eight fixed
    #: names, because these are what :attr:`DecodeStats.gap_causes` groups by and
    #: a name carrying the size of the break would make the counter unbounded.
    #: The join has two *shapes* as well -- refused, or accepted as the start of
    #: a new segment -- and the two are not the same event, so they are counted
    #: separately rather than as one number with a flag.
    GAP_FIRST = "first"          # nothing before it to be continuous with
    GAP_GAP = "gap"              # samples missing between two blocks
    GAP_OVERLAP = "overlap"      # block reaches back into decoded samples
    GAP_STREAM = "stream"        # the producing process was replaced
    GAP_TUNE = "tune"            # the receiver was retuned
    GAP_RANGE = "range"          # a range that cannot be true
    GAP_UNPLACED = "unplaced"    # the source cannot say where these samples are
    GAP_TINY = "tiny"            # too few samples to demodulate at all
    GAP_REFUSED = "after_refusal"  # delivery continued past undecoded samples

    def _run(self) -> None:
        # ``t0`` is the start of the *whole* iteration, not of the decode alone.
        # Pacing against the decode time alone quietly adds the drain on top, so
        # the loop ran at 20 ms/field instead of 16.7 and published 50 fps from
        # a 16.7 ms budget it was already meeting.
        t0 = time.perf_counter()
        deadline = t0
        while not self._stop.is_set():
            # TEMP DIAG: once per iteration, rate-limited inside to one line per
            # second. Called first so every exit from the loop is covered by the
            # same accounting.
            diag.maybe_log()
            self.step()
            # Pace the whole iteration to the decode cadence; decoding faster
            # than real time is pointless, and the wait is what keeps the ring
            # from growing a backlog.
            #
            # Paced against an *absolute* schedule rather than "sleep the
            # remainder after each pass". A wait overshoots by about a
            # millisecond even at 1 ms timer resolution, and paid on every pass
            # that error never washes out. Advancing a fixed deadline and simply
            # not sleeping when it has already passed makes the long-run average
            # right, so a late iteration is followed by a slightly early one
            # instead of a permanently late stream.
            deadline += self._preview_interval
            now = time.perf_counter()
            if deadline < now - self._preview_interval:
                deadline = now                    # fell behind: resync
            slack = deadline - now
            if slack > 0.0005:
                self._stop.wait(slack)
            now = time.perf_counter()
            self.stats.cycle_ms = self.stats.cycle_ms * 0.9 + (now - t0) * 1000.0 * 0.1
            t0 = now

    def step(self) -> bool:
        """One iteration. Returns whether a frame was published.

        Split out of :meth:`_run` because everything interesting about continuity
        is a timing accident on a live stream -- a gap is a dropped USB packet,
        an overlap is a cursor that moved backwards -- and cannot be provoked on
        demand. As a plain method with no pacing wait it can be: the caller
        supplies the bytes, and the join, the refusal and the publication
        decisions become reproducible.

        Everything except the wait is here, including the accounting, so a
        caller driving this by hand measures exactly what the thread does.
        """
        try:
            return self._step()
        finally:
            for name in self._DECODE_COUNTERS:
                setattr(self.stats, name, int(getattr(self.decoder, name, 0)))

    def _step(self) -> bool:
        wd = self._diag
        wd.bump("iterations")
        # Snapshot of the epoch this iteration belongs to. Read once, here, so
        # every check below compares against the same value: a reset in the
        # middle of an iteration invalidates the whole of it.
        with self._lock:
            generation = self._generation
        wd.gauge("generation", generation)
        wd.gauge("locked", int(self.locked))
        wd.gauge("frame_age_ms", self.frame_age_s() * 1000.0)

        t_drain = time.perf_counter()
        block = self._acquire()
        verdict, carry, block = self._join(block)
        t_dec = time.perf_counter()
        self.stats.drain_ms = (
            self.stats.drain_ms * 0.9 + (t_dec - t_drain) * 1000.0 * 0.1
        )
        self.stats.dropped_iq_samples = self.source.ring.dropped_bytes // 2
        self.stats.ring_overrun_samples = self.stats.dropped_iq_samples

        if verdict == "idle":
            # Nothing arrived. This is the "receiver stopped" path, and it used
            # to be the one path that did nothing: it skipped the no-lock
            # bookkeeping entirely, so a source that delivered nothing left the
            # previous frame and the previous "locked" status standing. The
            # accounting below is what makes a dead receiver look dead, so it
            # has to run here too.
            wd.bump("buffers_empty")
            self._count_missed_fields()
            return False
        if verdict == "refuse":
            self._reset_decoder()
            return False

        frame = None
        undecoded = 0
        try:
            with self._decoder_lock:
                if self._generation != generation:
                    # A reset landed between the drain and the decoder lock.
                    # These samples predate it, so feeding them in would rebuild
                    # exactly the picture the reset exists to clear.
                    wd.bump("decode_skipped_stale")   # TEMP DIAG
                    undecoded = int(block.samples)
                else:
                    if not carry:
                        # A new segment: the rolling window describes a stretch
                        # of stream the decoder is no longer holding, so it goes
                        # before this block is appended to it. The reset is
                        # *inside* the decoder lock and takes no lock of its own:
                        # calling _reset_decoder() here would re-enter the very
                        # lock held, and a plain Lock is not reentrant -- the
                        # first block of every segment would deadlock the thread
                        # against the next iteration's drain.
                        self.decoder.reset()
                    # TEMP DIAG: what the drain knows about this block that the
                    # samples cannot say themselves. Built only while the
                    # capture is on, so the ordinary decode path allocates
                    # nothing for it, and passed in one keyword so the decoder's
                    # signature cannot grow an input it might start reading.
                    provenance = (self._block_provenance(block, carry)
                                  if self.ledger is not None else None)
                    frame = self.decoder.push_iq(
                        block.iq,
                        continuous=carry,
                        first_sample=int(block.first_sample),
                        provenance=provenance,
                    )
        except Exception as exc:  # keep the pipeline alive on a bad chunk
            # Swallowed silently before, which turned a repeating programming
            # error into indistinguishable "loss of video": the frame counter
            # simply stopped, and the app looked like a receiver fault. Now it
            # is counted and the first few are recorded, so a decode bug is
            # visible as a decode bug.
            self.stats.decode_errors += 1
            wd.bump("decode_errors")                # TEMP DIAG
            if self.stats.decode_errors <= 3:
                self.stats.last_decode_error = f"{type(exc).__name__}: {exc}"
            frame = None
            undecoded = int(block.samples)
        self.stats.decode_ms = (
            self.stats.decode_ms * 0.9 + (time.perf_counter() - t_dec) * 1000.0 * 0.1
        )
        if undecoded:
            self._poison(undecoded)

        if frame is not None and frame.located and frame.iq_last == self._published_end:
            # The window still holds the field this picture was drawn from and
            # the rasteriser drew it again. Counting it would inflate the frame
            # rate with pictures nobody has waited for, and would keep a dead
            # transmitter looking live. An interval that ends *earlier* than the
            # last published one is a different field and is published: refusing
            # to go backwards would pin the panel to a stale picture.
            self.stats.dup_frames += 1
            wd.bump("frames_duplicate")            # TEMP DIAG
            frame = None
        if frame is not None:
            frame.segment = self._cont.segment

        if frame is not None:
            with self._lock:
                if self._generation == generation:
                    # Published only if no reset has happened while this field
                    # was being decoded. Publishing an epoch-stale frame would
                    # put the old frequency's picture back on screen after the
                    # panel had been cleared for it.
                    self._frame = frame
                    self._published_at = time.monotonic()
                else:
                    frame = None
            if frame is None:
                wd.bump("publish_skipped_stale")   # TEMP DIAG
                return False
            wd.bump("frames_published")            # TEMP DIAG
            # TEMP DIAG: how old the picture on screen was at the moment this one
            # replaced it. The new frame's own age is zero by definition, so
            # recording that would say nothing; this is the number that grows if
            # publication stalls.
            wd.gauge(
                "published_frame_age_ms",
                (self._published_at - self.stats.last_frame_at) * 1000.0
                if self.stats.last_frame_at
                else -1.0,
            )
            self.stats.frames += 1
            self.stats.locked_frames += 1
            self.stats.unique_frames += 1
            self.stats.last_frame_at = time.monotonic()
            self.stats.line_rate_hz = frame.line_rate_hz
            self.stats.sync_quality = frame.sync_quality
            self.stats.standard = frame.standard
            self.stats.lines_from_grid = frame.lines_from_grid
            self.stats.frame_interval = (int(frame.iq_first), int(frame.iq_last))
            if frame.located:
                self._published_end = int(frame.iq_last)
            self._last_decode = time.monotonic()
            # TEMP DIAG: the capture is taken here, at the publication decision
            # and not one line earlier or later. Earlier is before the duplicate
            # and stale-generation checks, so it would capture frames that were
            # never shown; later is on the next iteration, by which time the
            # decoder has moved on. Both locks are released at this point --
            # ``_lock`` closed above and ``_decoder_lock`` released with the
            # decode -- which is what lets a multi-megabyte write happen without
            # holding up the field that is already arriving.
            self._capture_publication(frame, generation)
            return True

        # Nothing publishable this iteration: count the fields spent without one.
        self._count_missed_fields()
        return False

    def _block_provenance(self, block: IQBlock, carry: bool) -> dict:
        """TEMP DIAG: the drain's facts about one block the decoder accepted.

        Who produced the samples, which hand-over they arrived in, whether they
        begin a segment, why the previous span ended and how much the drain
        skipped. None of it can be recovered from the samples, and all of it is
        needed to say whether a captured window is one continuous stretch of one
        stream -- so it is carried beside the samples rather than reconstructed
        afterwards, when the block that knew would be gone.

        Read under ``_decoder_lock``, like the decode it describes. Nothing
        outside the capture reads any of it.
        """
        return {
            "stream_id": int(block.first_stream_id),
            "tune_id": int(block.first_tune_id),
            "segment": int(self._cont.segment),
            "delivery": int(block.delivery),
            "starts_segment": not carry,
            "reset_reason": str(block.reset_reason),
            "skipped_samples": int(block.skipped_samples),
        }

    def _capture_publication(self, frame: VideoFrame, generation: int) -> None:
        """TEMP DIAG: write one publication-matched capture. Never raises.

        Wrapped in its own try because this is the only code on the decode thread
        that touches the filesystem, and a diagnostic that can take the decoder
        down with it is worse than no diagnostic at all. The failure is counted
        where the rest of the section's numbers are counted, so a capture that
        stopped working is visible as such.
        """
        if self.ledger is None:
            return
        wd = self._diag
        try:
            cap = self.decoder.publication(frame)
            if cap is None:
                wd.bump("pubcap_skipped")      # TEMP DIAG
                return
            frame.generation = int(generation)
            cap.generation = int(generation)
            cap.segment = int(self._cont.segment)
            cap.encoding = getattr(self.source.encoding, "value", "signed-int8")
            cap.source_kind = str(getattr(self.source, "kind", ""))
            cap.frequency_hz = int(getattr(self.source, "applied_frequency_hz", 0) or 0)
            st = self.stats
            cap.stats = {
                "frames": st.frames,
                "unique_frames": st.unique_frames,
                "dup_frames": st.dup_frames,
                "segments": st.segments,
                "discontinuities": st.discontinuities,
                "missing_samples": st.missing_samples,
                "overlap_samples": st.overlap_samples,
                "iq_written": st.iq_written,
                "iq_delivered": st.iq_delivered,
                "iq_accepted": st.iq_accepted,
                "iq_rejected": st.iq_rejected,
                "fm_samples": st.fm_samples,
                "fm_resets": st.fm_resets,
                "boundary_steps": st.boundary_steps,
                "sync_attempts": st.sync_attempts,
                "sync_accepted": st.sync_accepted,
                "raster_accepts": st.raster_accepts,
                "raster_rejects": st.raster_rejects,
                "grid_rows": st.grid_rows,
            }
            if diag.pubcap_write(cap):
                wd.bump("pubcap_written")      # TEMP DIAG
        except Exception as exc:
            wd.bump("pubcap_errors")          # TEMP DIAG
            wd.gauge("pubcap_error", f"{type(exc).__name__}: {exc}")

    def _acquire(self) -> IQBlock:
        """One hand-over from the source, as a record whatever the source offers.

        :meth:`IQSource.drain_block` is the primitive and carries the absolute
        range and the two identities that decide continuity. A source that only
        implements the older samples-only :meth:`~IQSource.drain_iq` -- a test
        double, a plugin -- is still usable, but nothing about it can be proven,
        so its samples are wrapped with no position and no identity and every
        hand-over starts a new segment. That costs one field per drain, which is
        the price of not inventing a continuity that was never shown.
        """
        drain = getattr(self.source, "drain_block", None)
        if callable(drain):
            return drain(self._max_chunk)
        iq = self.source.drain_iq(self._max_chunk)
        return IQBlock.orphan(
            iq,
            sample_rate=int(getattr(self.source, "sample_rate", 0)),
            encoding=getattr(self.source, "encoding", None),
        )

    def _join(self, block: IQBlock) -> tuple[str, bool, IQBlock]:
        """Decide what to do with ``block``, and account for it.

        Returns ``(verdict, carry, block)``: *verdict* is ``"idle"`` (nothing
        arrived), ``"refuse"`` (delivered, but the samples cannot be part of a
        stream) or ``"accept"``; *carry* says whether the decoder may keep its
        demodulator predecessor across this join; and *block* is the block that
        verdict applies to, which is **not** always the one that arrived.

        That third value is load-bearing rather than a convenience. The join is
        the only place that knows how much of a delivery has already been
        demodulated, so it is the only place a repeat can be removed -- and the
        result has to travel back to the caller, because the caller is what feeds
        the decoder. Returning only the verdict while keeping the trimmed block
        in a local left the decoder demodulating the *untrimmed* samples with
        ``continuous=True``: a second pass over the same samples went into the
        window, and the join's boundary step was computed from a sample the
        decoder had already moved past. The picture looked roughly right, the
        counters said an overlap had been trimmed, and the boundary artefact the
        trimming exists to prevent was manufactured by the trimming itself.

        The difference between refusing and starting a new segment is the point.
        A block that straddles a retune is refused, because it is a mixture: its
        halves are each fine and together they are a picture assembled from two
        different pictures. A block that merely *follows* a gap is accepted as the
        start of a segment, because its samples are good and refusing them would
        throw away a field of real video over a hole that is already counted.

        A repeated head is removed the same way, and then the question ``carry``
        answers is not about the block but about the run the decoder is holding:
        how much of this delivery has already been demodulated says nothing about
        what the decoder still holds. That is decided by :attr:`Continuity.open`,
        which a refusal and a decode that raised both leave set.

        An empty delivery is neither, and must not be treated as either. The
        cursor did not move, so the next block is still adjacent to whatever came
        before it; poisoning the join on an empty read is how one starved poll at
        30 Hz would cost a field.
        """
        st = self.stats
        c = self._cont
        wd = self._diag
        n = int(block.samples)
        if block.known and int(block.last_sample) > st.iq_written:
            # The drain's cursor *is* the ring's write position read under the
            # same lock, so this is everything written so far -- including
            # whatever the drain skipped on purpose.
            st.iq_written = int(block.last_sample)
        st.chunk_samples = n
        wd.bump("buffers")
        wd.bump("samples_in", n)
        wd.observe("chunk_samples", n)
        wd.gauge("chunk_last", n)
        wd.gauge("chunk_cap", self._max_chunk)
        if not n:
            return ("idle", False, block)
        st.iq_delivered += n
        if block.skipped_samples:
            st.capped_skips += 1
            st.capped_skip_samples += int(block.skipped_samples)
            wd.bump("capped_skips")                # TEMP DIAG
            wd.hist("capped_kib", int(block.skipped_samples) >> 11)
        if block.reset_reason:
            wd.bump("reset_reason_blocks")         # TEMP DIAG
            wd.gauge("reset_reason", block.reset_reason)

        # How far this block starts from where the accepted samples end, and
        # what the samples in between cost. Counted before any decision is taken
        # on it, because the size of a hole is a fact about the stream and the
        # reason it is there is a separate question.
        delta = 0
        if c.have and block.known and c.end >= 0:
            delta = int(block.first_sample) - int(c.end)
            if delta > 0:
                st.missing_samples += delta
                wd.bump("gap_samples", delta)       # TEMP DIAG
                wd.hist("gap_kib", delta >> 11)
            elif delta < 0:
                st.overlap_samples += -delta
                wd.bump("overlap_samples", -delta)  # TEMP DIAG

        seam = block.seam
        if seam:
            cause = seam                            # "stream" or "tune"
        elif block.known and block.last_sample < block.first_sample:
            cause = self.GAP_RANGE
        elif not block.known:
            cause = self.GAP_UNPLACED
        elif n < MIN_BLOCK_SAMPLES:
            cause = self.GAP_TINY
        elif not c.have:
            cause = self.GAP_FIRST
        elif block.first_stream_id != c.stream_id or block.last_stream_id != c.stream_id:
            # Named ahead of the range, because a producer that was replaced is
            # the *reason* the range moved: on a retune the samples between the
            # two ends were discarded on purpose, and calling that a gap would
            # report a hole whose cause is a restart.
            cause = self.GAP_STREAM
        elif block.first_tune_id != c.tune_id or block.last_tune_id != c.tune_id:
            cause = self.GAP_TUNE
        elif delta > 0:
            cause = self.GAP_GAP
        elif delta < 0:
            cause = self.GAP_OVERLAP
        elif c.open:
            # Delivery continued from a point the decoder holds nothing of. The
            # stream is continuous; the decoder's samples are not.
            cause = self.GAP_REFUSED
        else:
            cause = ""                              # proven adjacent

        if cause in (self.GAP_STREAM, self.GAP_TUNE):
            st.seam_blocks += 1
            wd.bump("seams_dropped")                # TEMP DIAG
            wd.observe("seam_tune_id", float(block.last_tune_id))
            if cause == self.GAP_TUNE:
                st.tune_seams += 1
                st.tune_seams_internal += 1
        elif cause == self.GAP_RANGE:
            st.oob_samples += n
        elif cause == self.GAP_TINY:
            wd.bump("blocks_tiny")                  # TEMP DIAG
        elif cause == self.GAP_UNPLACED:
            wd.bump("blocks_unplaced")              # TEMP DIAG
        if cause and cause not in (self.GAP_FIRST, self.GAP_REFUSED):
            # ``first`` is not a break in anything -- there was nothing before it
            # -- and ``after_refusal`` is the consequence of a decision already
            # counted. Neither is a discontinuity of the stream.
            st.note_gap(cause)
            wd.bump("discontinuities")              # TEMP DIAG
            wd.gauge("gap_cause", cause)

        if cause in (self.GAP_STREAM, self.GAP_TUNE, self.GAP_RANGE, self.GAP_TINY):
            self._refuse(block)
            return ("refuse", False, block)

        if cause == self.GAP_OVERLAP:
            # The head of this block has already been demodulated. Feeding it
            # again would put a second pass over the same samples into the
            # window and compute a boundary step from a sample the decoder has
            # already moved past, so the repeated part is dropped and counted as
            # refused -- delivered, and not used.
            trim = min(n, int(c.end) - int(block.first_sample))
            block = block.drop_head(trim)
            st.iq_rejected += trim
            wd.bump("overlap_trimmed", trim)        # TEMP DIAG
            if block.empty:
                # Wholly inside what has already been decoded: nothing left to
                # decode, and the next block is still adjacent to this one. The
                # continuity record is left alone, because ``c.end`` is the end
                # of the accepted samples and this block ends at or before it.
                return ("idle", False, block)
            if int(block.samples) < MIN_BLOCK_SAMPLES:
                # What is left after the repeat is removed is itself too small
                # to demodulate, and that is a fresh version of the same
                # question ``tiny`` answers elsewhere: below the minimum the
                # discriminator's output is mostly the boundary step. It is
                # refused rather than decoded, and refused *after* the trim --
                # the size that decides is the size of what is new, not of what
                # arrived. The span is left open, so the next block starts a
                # segment rather than stepping across samples nobody decoded.
                wd.bump("blocks_tiny")              # TEMP DIAG
                wd.gauge("gap_cause", self.GAP_TINY)
                st.note_gap(self.GAP_TINY)
                wd.bump("discontinuities")          # TEMP DIAG
                self._refuse(block)
                return ("refuse", False, block)
            # What is new is accepted either way; whether the predecessor is
            # carried across the trim is a question about the accepted *run*, not
            # about this block. ``open`` answers it: a refusal and a decode that
            # raised both reset the decoder, so nothing it holds reaches these
            # samples any more. Stepping across the refused span on the strength
            # of the trim alone claimed a predecessor that was gone -- the origin
            # landed a sample early, because the boundary step that offset assumes
            # is the one never taken -- and stamped the segment with the one
            # before it.
            carry = not c.open
            self._advance(block, carry=carry, segment=not carry)
            return ("accept", carry, block)

        carry = cause == ""
        self._advance(block, carry=carry, segment=not carry)
        return ("accept", carry, block)

    def _refuse(self, block: IQBlock) -> None:
        """Account for a block refused whole, and leave the span open.

        ``block`` is the block as the verdict applies to it -- which, after an
        overlap has been trimmed, is shorter than the one that arrived. The
        trimmed part was counted as refused at the moment it was removed, so
        counting what is left here is what keeps
        ``delivered == accepted + rejected``.

        ``end`` still moves to where the cursor is, so the next block is
        recognised as *following* this one rather than as a second, larger gap --
        but the span is left open, because those samples were never decoded and
        so nothing the decoder holds reaches them.
        """
        c = self._cont
        self.stats.iq_rejected += int(block.samples)
        c.end = int(block.last_sample) if block.known else -1
        c.stream_id = int(block.last_stream_id)
        c.tune_id = int(block.last_tune_id)
        c.have = True
        c.open = True
        self._diag.bump("blocks_refused")             # TEMP DIAG

    def _advance(self, block: IQBlock, carry: bool, segment: bool = False) -> None:
        """Record what the decoder now holds, having accepted ``block``.

        ``iq_accepted`` is counted here rather than after the decode, because
        this is the point where the samples have been taken into the decoder's
        segment. Whether a picture comes out of them is a separate question that
        does not un-accept them, and a decode that raises is *reclassified* by
        :meth:`_poison` rather than counted twice.
        """
        c = self._cont
        self.stats.iq_accepted += int(block.samples)
        if not carry and segment:
            # A new segment begins. The published-end mark belongs to the old
            # one, and comparing a first frame after a gap against it would
            # refuse exactly the frames that prove the gap was survived.
            self._published_end = -1
            c.segment = self.stats.segments
            self.stats.segments += 1
            wd = self._diag
            wd.bump("segments_started")             # TEMP DIAG
            wd.gauge("segment", c.segment)
        c.end = int(block.last_sample) if block.known else -1
        c.stream_id = int(block.last_stream_id)
        c.tune_id = int(block.last_tune_id)
        c.have = True
        c.open = False

    def _poison(self, samples: int) -> None:
        """Account for samples that arrived but were not decoded.

        Counted as refused, because they were delivered: they are neither
        accepted by the demodulator nor missing from the stream, and an
        accounting with no bucket for them would quietly lose them.

        A block reaches here *after* :meth:`_join`, so it has already been
        counted as accepted; this is a move between two buckets and not a second
        count. Leaving the accepted total alone would put the same samples on
        both sides of ``delivered == accepted + rejected`` and quietly inflate
        it by whatever the decoder refused.

        The span is left *open*, so the next block starts a new segment instead of
        computing a boundary step across samples the decoder never saw.
        """
        n = int(samples)
        if n <= 0:
            return
        self.stats.iq_accepted = max(0, self.stats.iq_accepted - n)
        self.stats.iq_rejected += n
        self._cont.open = True
        self._diag.bump("poisoned_samples", n)      # TEMP DIAG

    @property
    def _field_interval(self) -> float:
        """The field interval missed fields are counted against.

        Set by the *standard*, and deliberately not by the decode cadence. At
        30 Hz every second field is "missed" by construction, which makes the
        counter a property of this program's pacing rather than of the
        transmitter -- a number that is always small, always growing, and means
        nothing. Before a lock there is no standard to go by and NTSC is
        assumed, which is the wrong answer 20% of the time for PAL and is bounded
        by one field interval of a 50 Hz count.
        """
        if self.stats.standard == "PAL":
            return PAL_FIELD_INTERVAL_S
        return NTSC_FIELD_INTERVAL_S

    def _count_missed_fields(self) -> None:
        """Account for a stretch during which no field could be locked.

        Shared by the "no samples", "samples but no lock" and "nothing
        publishable" paths, because all three mean the same thing -- nothing
        usable arrived -- and the "no samples" path used to skip this entirely,
        which is what let a dead receiver keep reporting healthy-looking counters
        and a "locked" status.
        """
        now = time.monotonic()
        if now - self._last_decode <= 0.5:
            return
        elapsed = now - max(self._last_decode, self.stats.started_at)
        self.stats.fields_missed += max(
            0, int(elapsed / self._field_interval) - self.stats.locked_frames
        )
        self._last_decode = now
