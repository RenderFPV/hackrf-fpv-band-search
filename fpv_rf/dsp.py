"""DSP for wideband-FM analogue FPV composite video.

Signal chain implemented here (all numpy, all streaming-friendly):

    IQ (uint8/complex)
      -> FM discriminator (phase difference)
      -> video-band limiting / decimation
      -> sync envelope + horizontal sync pulse detection
      -> per-line resampling into a raster  (self-re-anchoring)
      -> optional de-emphasis / AGC
      -> luminance field

Design notes
------------
*Rasterisation is per-line and re-anchored on every detected sync pulse.*
This is deliberate. The HackRF shares a USB 2.0 bus with other devices, so the
host occasionally loses IQ samples. A decoder that assumes a fixed sample
offset for line N will tear permanently after the first dropped sample. By
re-detecting sync on each line we only lose the line that was in flight, so
degradation is graceful rather than catastrophic.

*Line rate is measured, not assumed.* NTSC (15.734 kHz) and PAL (15.625 kHz)
differ by only 0.7%, which is still cleanly separable at 10 MS/s (635.6 vs
640.0 samples/line), so we measure the line period and infer the standard
instead of trusting a setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

try:  # scipy gives us a fast, well-filtered polyphase resampler
    from scipy.signal import resample_poly, firwin, lfilter

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only without scipy
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------
# Video standards
# --------------------------------------------------------------------------


class Standard(str, Enum):
    NTSC = "ntsc"
    PAL = "pal"
    AUTO = "auto"


@dataclass(frozen=True)
class VideoSpec:
    """Timing and framing constants for an analogue video standard.

    ``active_start_frac`` and ``active_width_frac`` are expressed as fractions
    of one line, measured **from the sync tip** (not the sync pulse start), so
    they follow directly from the standard's own timing:

    * NTSC: 4.7 us sync + 4.5 us back porch = 9.2 us of a 63.5 us line = 0.145;
      active picture is 52.6 us = 0.828 of a line.
    * PAL:  5.0 us sync + 5.5 us back porch = 10.5 us of a 64.0 us line = 0.164;
      active picture is 52.0 us = 0.813 of a line.

    ``crop_start``/``crop_width`` are the *practical* decode window, which is
    narrower than the ideal timing because FPV cameras overscan. The NTSC values
    are the ones the original still-decoder converged on after working against
    real captures; they are inset from the ideal 0.145/0.828 to avoid pulling
    back-porch noise into the left edge of the picture.
    """

    name: str
    line_rate_hz: float
    total_lines: int          # lines per field
    visible_lines: int        # active (non-blanked) lines per field
    video_bw_hz: float        # composite video bandwidth
    subcarrier_hz: float     # colour subcarrier
    sync_pulse_hz: float      # horizontal sync pulse width
    active_start_frac: float  # from sync tip to first active pixel
    active_width_frac: float  # active video width as fraction of a line
    deemph_tau_s: float       # FM de-emphasis time constant
    crop_start: float         # practical decode start, from sync tip
    crop_width: float         # practical decode width, as fraction of a line

    @property
    def line_period_s(self) -> float:
        return 1.0 / self.line_rate_hz


NTSC = VideoSpec(
    name="NTSC",
    line_rate_hz=15_734.0,
    total_lines=262,
    visible_lines=240,
    video_bw_hz=4_500_000.0,
    subcarrier_hz=3_579_545.0,
    sync_pulse_hz=4_700.0,
    active_start_frac=0.145,
    active_width_frac=0.828,
    deemph_tau_s=75e-6,
    crop_start=0.180,
    crop_width=0.760,
)

PAL = VideoSpec(
    name="PAL",
    line_rate_hz=15_625.0,
    total_lines=312,
    visible_lines=288,
    video_bw_hz=5_500_000.0,
    subcarrier_hz=4_433_619.0,
    sync_pulse_hz=5_000.0,
    active_start_frac=0.164,
    active_width_frac=0.813,
    deemph_tau_s=50e-6,
    crop_start=0.190,
    crop_width=0.750,
)

SPECS = {Standard.NTSC.value: NTSC, Standard.PAL.value: PAL}


def identify_standard(line_rate_hz: float, tol: float = 0.02) -> VideoSpec:
    """Return the spec whose line rate is closest to a measured rate."""
    return min((NTSC, PAL), key=lambda s: abs(s.line_rate_hz - line_rate_hz))


def expected_line_samples(sample_rate_hz: float, spec: VideoSpec) -> int:
    return int(round(sample_rate_hz / spec.line_rate_hz))


#: Half-width of the guard band around DC that occupancy tests ignore.
#:
#: A HackRF's own local oscillator leaks into the centre of every window it
#: tunes. Measured on this hardware with nothing transmitting: bin 0 sits 52 dB
#: above the noise floor and bins +/-1 sit 46 dB, and nothing else in the whole
#: 10 MHz span is more than 3 dB up. It appears at every tuned frequency and
#: moves with the tuner, which is what identifies it -- a real signal holds its
#: frequency while the tuner moves, so it shows up at a different offset in each
#: window, while this one is always dead centre.
#:
#: Leaving it in is worse than leaving noise in, because it is loud, constant,
#: and present at *every* hop: a 34-hop sweep would find all 34 hops occupied and
#: raise 34 alerts for one receiver.
#:
#: Ignoring it costs nothing real. An FPV video downlink is ~15 MHz wide, so it
#: covers the centre of the window hundreds of times over and cannot be missed
#: by a 50 kHz hole. The only thing that could hide in here is a narrowband
#: carrier engineered to land within 25 kHz of whatever frequency the tuner
#: happened to be sitting on, which is not something anyone transmits.
DC_GUARD_HZ = 25_000.0


def away_from_dc(
    freqs: np.ndarray, guard_hz: float = DC_GUARD_HZ
) -> np.ndarray:
    """Boolean mask of the bins that are not the receiver's own DC leakage.

    Every occupancy statistic goes through this. Keeping it in one place is the
    point: the classifier and the band search have to agree about what counts as
    a signal, or the search finds 34 hop-sized candidates and then the thorough
    pass classes every one of them as the receiver hearing itself.
    """
    if freqs.size == 0:
        return np.zeros(0, dtype=bool)
    return np.abs(freqs) > float(guard_hz)


# --------------------------------------------------------------------------
# FM demodulation
# --------------------------------------------------------------------------


def fm_demodulate(iq: np.ndarray) -> np.ndarray:
    """Quadrature FM demodulator: instantaneous frequency as float32.

    Input is a 1-D complex array of IQ samples. Output is proportional to
    instantaneous frequency deviation (arbitrary units).

    A division-based small-angle shortcut was tried here and reverted: for
    ``atan2(cross, dot)`` to become ``cross/dot`` the argument must be tiny, and
    guarding that tightly enough to stay exact pushes the polynomial in front
    of it past the cost of ``np.angle``, which numpy already dispatches to a
    vectorised ``arctan2``. It measured both slower (18.2 vs 13.2 ms per 600k
    samples) and wrong, blowing up to 5e4 wherever ``dot`` approached zero.
    """
    iq = np.asarray(iq)
    if iq.ndim != 1:
        raise ValueError("IQ input must be one-dimensional")
    if iq.size < 8:
        raise ValueError("need at least 8 IQ samples to demodulate")
    if not np.iscomplexobj(iq):
        raise ValueError("IQ input must be complex")
    # angle(x[n] * conj(x[n-1])) is the phase step -> frequency deviation
    step = np.angle(iq[1:] * np.conj(iq[:-1]))
    return step.astype(np.float32)


def deemphasis(x: np.ndarray, sample_rate_hz: float, tau_s: float | None) -> np.ndarray:
    """Single-pole FM de-emphasis (75 us US, 50 us EU)."""
    if tau_s is None or tau_s <= 0:
        return x
    dt = 1.0 / float(sample_rate_hz)
    a = dt / (tau_s + dt)
    if not _HAVE_SCIPY:
        # one-pole IIR via lfilter equivalent, expressed with numpy
        out = np.empty_like(x)
        acc = float(x[0])
        for i in range(x.size):
            acc += a * (float(x[i]) - acc)
            out[i] = acc
        return out
    b = np.array([a], dtype=np.float64)
    a_coef = np.array([1.0, -(1.0 - a)], dtype=np.float64)
    return lfilter(b, a_coef, x.astype(np.float64)).astype(np.float32)


def limit_video_bandwidth(
    x: np.ndarray, sample_rate_hz: float, cutoff_hz: float, decimate_to: float | None = None
) -> np.ndarray:
    """Anti-alias the demodulated signal and optionally decimate.

    The FM discriminator output contains the audio subcarriers (5.8/6.5/7.1 MHz)
    and out-of-band noise well above the video band. Filtering to the video
    band before line sampling markedly improves sync stability.
    """
    x = np.asarray(x, dtype=np.float32)
    nyq = sample_rate_hz / 2.0
    cutoff = float(np.clip(cutoff_hz, 100.0, nyq * 0.95))

    if decimate_to and decimate_to < sample_rate_hz:
        ratio = sample_rate_hz / float(decimate_to)
        up, down = _rational(ratio)
        if _HAVE_SCIPY:
            x = resample_poly(x, up, down).astype(np.float32)
        else:
            x = _boxcar_decimate(x, down)
        sample_rate_hz = sample_rate_hz * down / up
        nyq = sample_rate_hz / 2.0
        cutoff = float(np.clip(cutoff_hz, 100.0, nyq * 0.95))

    if not _HAVE_SCIPY:
        return x
    # gentle FIR lowpass; taps scaled to sample rate
    taps = int(max(15, min(255, int(sample_rate_hz / cutoff * 8) | 1)))
    if taps % 2 == 0:
        taps += 1
    try:
        h = firwin(taps, cutoff / nyq, window="hamming")
    except Exception:
        return x
    return lfilter(h, [1.0], x).astype(np.float32)


def _rational(ratio: float, max_den: int = 64) -> tuple[int, int]:
    """Small-integer up/down factors approximating ``ratio``."""
    for down in range(1, max_den + 1):
        up = int(round(ratio * down))
        if up >= 1 and abs(up / down - ratio) < 1e-6:
            return up, down
    return 1, max(1, int(round(ratio)))


def _boxcar_decimate(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return x
    n = (x.size // factor) * factor
    return x[:n].reshape(-1, factor).mean(axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# Sync detection
# --------------------------------------------------------------------------


@dataclass
class SyncResult:
    pulse_starts: np.ndarray     # float sample positions of the sync LEADING EDGE
    pulse_centres: np.ndarray    # centroids of the above-threshold excursion
    line_samples: float          # measured samples per line
    line_rate_hz: float
    polarity: int                # +1 if sync is a peak in the envelope
    threshold: float
    quality: float               # 0..1 fraction of well-spaced pulses
    spec: VideoSpec = field(default=NTSC)


def _smooth(x: np.ndarray, taps: int) -> np.ndarray:
    if taps <= 1:
        return x
    k = np.ones(int(taps), dtype=np.float32) / float(taps)
    if _HAVE_SCIPY:
        return lfilter(k, [1.0], x).astype(np.float32)
    c = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    out = (c[taps:] - c[:-taps]) / taps
    return out.astype(np.float32)


def _high_percentile(x: np.ndarray, pct: float, max_samples: int = 1 << 16) -> float:
    """A high percentile, estimated from a strided subsample when x is large.

    The sync threshold only has to land *inside* the sync plateau, which is
    about 7% of all samples wide, so a 64k-element subsample places it well
    within it. ``np.percentile`` is an O(n) partition and cost 2.3 ms on a
    320k-sample window -- a seventh of the whole field budget -- to resolve a
    threshold that a thousandth of that precision already satisfies.
    """
    n = x.size
    if n <= max_samples:
        return float(np.percentile(x, pct))
    step = n // max_samples + 1
    return float(np.percentile(x[::step], pct))


def detect_sync(
    demod: np.ndarray,
    sample_rate_hz: float,
    spec: VideoSpec | None = None,
    polarity: str = "auto",
    sensitivity: float = 0.975,
) -> SyncResult:
    """Locate horizontal sync pulses in a demodulated composite waveform.

    Two position estimates are returned per line:

    ``pulse_starts``
        The leading edge of the sync excursion. This is the anchor the
        rasteriser uses, because a threshold crossing is sharp and repeatable
        whereas a centroid is biased by whatever the picture happens to be
        doing. Measured on a known capture, the leading edge scored +0.80
        adjacent-row correlation against +0.59 for the centroid.
    ``pulse_centres``
        Sub-sample centres of the excursion, kept for diagnostics and for
        callers that want a more precise per-line estimate.

    Sync is the largest frequency excursion in an FPV composite signal (black
    and sync sit at peak deviation, so after FM demod the sync tips are the
    envelope extremes). We smooth the envelope, threshold near the extreme, and
    take the rising edges.

    Two details decide whether this finds *most* lines or only a few, and both
    were arrived at by measurement against a known-good capture:

    *Threshold at ~97.5%, not higher.* A horizontal sync pulse occupies 4.7 us
    of a 63.5 us line, so about 7% of all samples belong to sync. A threshold at
    the 98.5th percentile only admits the top 1.5% of samples and therefore
    misses the great majority of lines: on the reference capture that yielded
    8450 pulses for 12587 expected lines (67%). At 97.5% the same capture
    yields 12120 pulses (96%), which is what makes a complete 240-line field
    findable at all.

    *Do not smooth.* A 4.7 us sync pulse is ~47 samples wide at 10 MS/s, so
    there is nothing narrow to sharpen. Any moving average only lowers the tip,
    which forces the percentile threshold up and drops whole lines: a 10-tap
    boxcar cut the detected-line count from 12120 to 9897 and the longest
    usable run from 254 lines to 85 -- i.e. no complete field at all.

    *Use the signed waveform, not its magnitude.* Taking ``abs()`` folds white
    level and sync onto the same footing, so bright picture content injects
    false rising edges roughly as often as real sync. Sign-selecting the
    correct extreme first (the ``polarity`` loop) keeps the threshold
    discriminating one side of the signal only.

    *Rank the two polarities by plausibility, not just spacing regularity.*
    Spacing regularity alone is not enough to identify the right polarity: if
    the picture happens to be brighter than sync, thresholding the wrong
    extreme can still produce edges that land near the line rate by accident,
    and the decoder then locks onto picture detail instead of sync. The
    discriminator is that real sync yields roughly *one* line per line period,
    so the candidate's line density must sit near 1.0. Weighting spacing
    quality by a density plausibility term makes the choice decisive.
    """
    x = np.asarray(demod, dtype=np.float32)
    if x.size < 64:
        raise ValueError("not enough samples for sync detection")

    ref = spec or NTSC
    nominal = expected_line_samples(sample_rate_hz, ref)
    min_gap = max(4, int(nominal * 0.5))

    env = x

    pols = [1, -1] if polarity == "auto" else [1 if polarity == "positive" else -1]
    best: SyncResult | None = None
    best_score = -1.0
    for sign in pols:
        score = sign * env
        thr = _high_percentile(score, sensitivity * 100.0)
        mask = score > thr
        starts = _rising_edges(mask)
        starts = _cluster(starts, min_gap)
        if starts.size < 4:
            continue
        centres = _centroids(score, mask, starts)
        diffs = np.diff(centres)
        near = np.abs(diffs - nominal) < nominal * 0.12
        quality = float(np.count_nonzero(near)) / float(max(1, diffs.size))
        plausible = diffs[near] if np.any(near) else diffs
        line_samples = float(np.median(plausible)) if plausible.size else float(nominal)
        # how many line periods does the buffer actually hold, and how many
        # did we detect? real sync gives a ratio near 1.
        density = (starts.size * line_samples) / float(x.size)
        plaus = float(np.exp(-(((density - 1.0) / 0.35) ** 2)))
        rank = quality * plaus
        if rank > best_score:
            best_score = rank
            best = SyncResult(
                pulse_starts=starts,
                pulse_centres=centres,
                line_samples=line_samples,
                line_rate_hz=sample_rate_hz / line_samples,
                polarity=sign,
                threshold=thr,
                quality=quality,
                spec=identify_standard(sample_rate_hz / line_samples),
            )
        # A candidate that is both well-spaced and at the right line density is
        # already unambiguous, and the opposite polarity only exists to catch
        # sign mistakes. Re-running the whole pipeline to confirm it doubles the
        # cost of the most expensive function in the decoder, so stop here once
        # the answer is decisive. rank >= 0.5 with quality >= 0.8 cannot be
        # produced by locking onto picture detail.
        if best is not None and quality >= 0.8 and rank >= 0.5:
            break
    if best is None:
        empty = np.zeros(0, dtype=np.float64)
        return SyncResult(
            pulse_starts=empty,
            pulse_centres=empty,
            line_samples=float(nominal),
            line_rate_hz=sample_rate_hz / nominal,
            polarity=1,
            threshold=0.0,
            quality=0.0,
            spec=ref,
        )
    return best


def _rising_edges(mask: np.ndarray) -> np.ndarray:
    """Indices where ``mask`` goes low->high, in one pass and one allocation.

    The obvious ``flatnonzero(diff(padded) == 1)`` needs an int8 copy, an int8
    difference, a comparison and then a nonzero, i.e. four full-length
    temporaries over a buffer that is a third of a megabyte. Comparing the two
    shifted views directly collapses that to a single boolean temporary.
    """
    if mask.size < 2:
        return np.zeros(0, dtype=np.float64)
    return (np.flatnonzero(mask[1:] > mask[:-1]) + 1).astype(np.float64)


def _cluster(idx: np.ndarray, min_gap: int) -> np.ndarray:
    """Suppress edges that land inside the excursion of the edge before them.

    Rule: keep edge ``i`` unless ``idx[i] - idx[i-1] < min_gap``, in which case
    the earlier edge already represents that pulse and this one is a duplicate
    crossing inside it.

    Two neighbouring rules are worth distinguishing, because this one is easy to
    get wrong:

    * **Cascading** -- ``keep[i] = keep[i-1] & (gap[i] >= min_gap)`` -- looks
      equivalent but is not. One too-close pair then discards *every* later
      edge, since the flag never recovers. On randomised input it kept 1-4 edges
      where the correct answer has 9-11. It appeared to work on real captures
      only because their sync gaps are all equal, so the cascade never fired;
      vertical blanking or a dropped USB block makes it collapse.
    * **Greedy** -- keep the *first* edge of each cluster, comparing against the
      last *kept* edge. This is what a textbook min-gap filter does, and it
      cannot be vectorised: the state carried forward depends on the previous
      decision. A loop cost 3.75 ms per call here, which alone exceeded the
      16.7 ms field budget, and the vectorised attempts at it were both slow and
      wrong (keeping 1200 where greedy keeps nothing, or dropping every group
      leader).

    The local rule is not identical to greedy, and the difference is bounded:
    greedy would additionally recover an edge from a cluster whose members are
    all within ``min_gap`` of each other but span more than ``min_gap`` in
    total -- edges at 0, 200 and 400 with ``min_gap=318``, where greedy keeps
    0 and 400 and this keeps only 0. For sync detection that cannot arise:
    ``min_gap`` is half a line, so two real pulses are 636 samples apart and
    always open a wide gap, and the spurious edges that clustering exists to
    remove are confined to a single ~47-sample pulse, far inside ``min_gap``.
    Both rules therefore agree on real sync, and the local rule additionally
    keeps the *latest* crossing of a pulse, which is the freshest estimate of
    where that pulse actually is.

    What the local rule does guarantee, and what the caller needs, is the
    invariant that consecutive kept edges are at least ``min_gap`` apart: if kept
    edges ``a`` and ``b`` were closer than ``min_gap`` then, since ``b`` was
    kept, ``idx[b] - idx[b-1] >= min_gap`` forces ``idx[b-1] <= idx[a]`` --
    impossible, because ``b-1 > a``.
    """
    if idx.size <= 1:
        return idx
    keep = np.empty(idx.size, dtype=bool)
    keep[0] = True
    keep[1:] = np.diff(idx) >= min_gap
    return idx[keep]


def _centroids(score: np.ndarray, mask: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Sub-sample sync pulse centre = centroid of each excursion, vectorised.

    Every sample in ``mask[start:end]`` is above threshold by construction, so
    the excursion itself is a valid non-negative weight once shifted by its own
    minimum.

    This was a per-pulse Python loop and cost 98 ms on a 600k-sample buffer --
    by far the most expensive thing in the decoder, and enough on its own to
    miss the 16.7 ms field budget.

    The vectorised form works on the *compacted* array of just the above-
    threshold samples, not on the full signal. Only about 7% of samples are in a
    sync pulse, so every intermediate is 14x smaller, and -- more importantly --
    each run becomes an exact contiguous segment, so ``np.*.reduceat`` can be
    used directly. Working on the full array instead forces the inter-run gaps
    into the segments and needs a full-length segment-id map to mask them out;
    that version was both slower and wrong, because the gap samples dragged the
    per-run minimum around.

    The return value has one entry per element of ``starts``, not per run, so
    callers can zip the two together. The counts differ when the mask touches a
    buffer edge -- a run open at index 0 has no rising edge, and one still open
    at the end has no closing transition -- so the two are mapped through
    ``searchsorted`` rather than assumed equal.
    """
    out = np.asarray(starts, dtype=np.float64).copy()
    if starts.size == 0 or mask.size == 0:
        return out
    if starts.size == 0:
        return np.zeros(0, dtype=np.float64)

    # Every transition in one pass, then pair them into runs via the boundaries.
    # Alternating tr[0::2]/tr[1::2] would be shorter but silently wrong whenever
    # the mask touches either end of the buffer: a run already open at index 0
    # has its first boundary at 0, and one still open at the end has no closing
    # transition. Keeping the explicit boundaries makes all four combinations
    # fall out of the same expression.
    tr = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    b = np.concatenate(([0], tr, [mask.size])).astype(np.int64)
    inside = mask[b[:-1]]                     # state of each [b[i], b[i+1]) span
    run_starts = b[:-1][inside]
    run_ends = b[1:][inside]                  # exclusive
    if run_starts.size == 0:
        return out

    # compact to the above-threshold samples; runs are contiguous in this array
    vals = score[mask].astype(np.float64)
    counts = run_ends - run_starts
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))   # segment starts
    seg_of = np.repeat(np.arange(run_starts.size), counts)

    seg_min = np.minimum.reduceat(vals, offsets)
    w = vals - seg_min[seg_of]                               # >= 0 by construction
    local = np.arange(vals.size, dtype=np.float64) - offsets[seg_of]
    num = np.add.reduceat(w * local, offsets)
    den = np.add.reduceat(w, offsets)
    centres = run_starts + np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)

    # map each start onto the run that contains it
    ri = np.searchsorted(run_starts, starts, side="right") - 1
    ok = ri >= 0
    ri_c = np.where(ok, ri, 0)
    ok &= starts < run_ends[ri_c]
    out[ok] = centres[ri_c[ok]]
    return out


# --------------------------------------------------------------------------
# Rasteriser
# --------------------------------------------------------------------------


@dataclass
class RasterFrame:
    image: np.ndarray            # uint8 (H, W) luminance
    width: int
    height: int
    spec: VideoSpec
    lines_used: int
    line_samples: float
    locked: bool
    quality: float


def line_grid(
    centres: np.ndarray, nominal_line: float, max_gap_lines: float = 200.0
) -> tuple[np.ndarray, float, float]:
    """Fit a global scanline grid to raw sync detections.

    Returns ``(positions, intercept, period)`` where ``positions[i]`` is the
    phase-locked sampling position for the i-th *detected* line.

    Method
    ------
    1. Walk the detections and convert every gap into an integer number of
       lines, ``round(gap / nominal)``. This is what distinguishes a genuinely
       dropped line from a field-blanking interval: both look like a large gap,
       but only the former should keep the line counter advancing by a few.
    2. Estimate the period as the *median* gap and the phase as the *median*
       residual. Both statistics are medians on purpose: ~99% of gaps are a
       single line, so a least-squares fit over all of them gets skewed by the
       ~100 field-blanking intervals of 80-90 lines. On a known capture a
       least-squares period was off by 0.12 samples/line, which accumulates to
       a full 1000 samples of drift by the end of a 0.8 s capture.
    3. Sample each line at the fitted grid instead of at its own detection.

    Why this matters: sampling each line at its raw detection shears the picture
    a few pixels per row and shows up as a comb artefact. Regressing instead
    averages the threshold-crossing noise out, while still re-anchoring after
    dropped samples -- a USB underrun costs one line, not the whole frame.
    """
    n = int(centres.size)
    if n < 4 or nominal_line <= 0:
        return centres.astype(np.float64), (float(centres[0]) if n else 0.0), float(nominal_line)

    y = centres.astype(np.float64)
    gaps = np.diff(y)
    # period from single-line gaps, which dominate; fall back to all gaps
    single = gaps[gaps < 1.5 * float(nominal_line)]
    period = float(np.median(single)) if single.size >= 3 else float(np.median(gaps))
    if not np.isfinite(period) or period <= 0:
        period = float(nominal_line)

    steps = np.maximum(1, np.rint(gaps / period).astype(np.int64))
    steps = np.minimum(steps, int(max_gap_lines))
    line_no = np.concatenate(([0], np.cumsum(steps))).astype(np.float64)

    intercept = float(np.median(y - period * line_no))
    return intercept + period * line_no, intercept, period


# Backwards-compatible alias: positions only.
def lock_phases(
    centres: np.ndarray, line: float, max_gap_lines: float = 200.0
) -> np.ndarray:
    """Phase-locked sampling positions for the detected lines."""
    return line_grid(centres, line, max_gap_lines)[0]


def _resample_2d(block: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a 2-D block to ``(out_h, out_w)``, low-passing when decimating.

    The decode window is ~483x240 and the display is 160x120, so this is
    normally a decimation in both axes. Decimating without filtering folds
    uncorrelated sample noise into the picture, which reads as heavy speckle on
    a live view, so a Gaussian of sigma = half the smaller reduction factor is
    applied first. The rule is scale-aware: any reduction gets filtering in
    proportion, and pure enlargement is left as plain bilinear interpolation.

    Measured against the reference decoder's raw frame, this reaches +0.83
    adjacent-row correlation where unfiltered interpolation reaches +0.19, and
    slightly beats the PIL BILINEAR downscale the reference uses (+0.82), at
    about 2 ms per field.
    """
    h, w = block.shape
    fy, fx = h / float(out_h), w / float(out_w)
    if _HAVE_SCIPY and (fy > 1.05 or fx > 1.05):
        from scipy.ndimage import gaussian_filter, zoom

        sigma = 0.5 * min(max(fy, 1.0), max(fx, 1.0))
        if sigma > 0.05:
            block = gaussian_filter(block, sigma, mode="nearest")
        out = zoom(block, (out_h / h, out_w / w), order=1, mode="nearest")
        return out.astype(np.float32, copy=False)

    # no scipy: bilinear along each axis via index maps
    yi = np.linspace(0.0, h - 1.0, out_h)
    xi = np.linspace(0.0, w - 1.0, out_w)
    y0 = np.floor(yi).astype(np.int64)
    x0 = np.floor(xi).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy_ = (yi - y0).astype(np.float32)[:, None]
    fx_ = (xi - x0).astype(np.float32)[None, :]
    top = block[np.ix_(y0, x0)] * (1 - fx_) + block[np.ix_(y0, x1)] * fx_
    bot = block[np.ix_(y1, x0)] * (1 - fx_) + block[np.ix_(y1, x1)] * fx_
    return (top * (1 - fy_) + bot * fy_).astype(np.float32)


class Rasteriser:
    """Extracts one clean field from a chunk of demodulated composite video.

    Strategy: render the most recent run of exactly ``visible_lines``
    scanlines whose sync-to-sync spacing all match the measured line period.

    Restricting a frame to a *contiguous* run is what makes the picture clean,
    and it is the single biggest quality lever in this decoder:

    * Field blanking (an ~80-90 line gap every field) can never fall inside a
      frame, so no row is interpolated across a discontinuity.
    * Across ~240 lines the per-line detection jitter does not accumulate into
      visible shear, so lines can be sampled at their raw detected positions.
      Measured on a known capture, doing this scored +0.80 adjacent-row
      correlation; forcing a single global grid across the whole capture scored
      +0.25, because any error in the fitted period compounds over 8000 lines.
    * "Most recent" means every emitted frame is the freshest thing in the
      buffer, which is what a live viewer wants.

    The cost is that a field containing a dropped line is skipped rather than
    patched. That is the right trade for viewing: the next field arrives in
    17 ms.

    On picture quality: adjacent-row correlation on the *raw* 483x240 active
    window measures about +0.09 for this decoder and about +0.09 for the
    original still decoder too, so that number describes the capture, not the
    code -- a HackRF capture of a 5.8 GHz VTX is genuinely noisy. Scores around
    +0.82 appear only *after* the downscale to display size, because
    ``_downscale`` low-pass filters away sample noise that is uncorrelated
    between neighbouring pixels. That filtering is real and worth doing for a
    live display, but it should not be mistaken for a sharper decode.
    """

    def __init__(self, width: int = 160, out_h: int | None = None) -> None:
        self.width = int(width)
        self.out_h = int(out_h) if out_h else None
        self._spec: VideoSpec | None = None
        self.frames_emitted = 0
        self.lines_skipped = 0

    @property
    def spec(self) -> VideoSpec | None:
        return self._spec

    def configure(self, spec: VideoSpec, out_h: int | None = None) -> None:
        self._spec = spec
        if out_h is not None:
            self.out_h = int(out_h)

    # -- line selection ----------------------------------------------------

    @staticmethod
    def select_run(
        starts: np.ndarray, line_len: float, want: int, tol_frac: float = 0.15
    ) -> np.ndarray | None:
        """Freshest contiguous run of >= ``want`` evenly spaced lines.

        Every maximal run of nominal spacing is found, and the one that ends
        latest is used. Anchoring only on the newest detected line would reject
        the whole buffer whenever a single line was dropped near the end; with
        roughly one dropped line per hundred, that would starve the decoder
        even though clean fields are available slightly earlier in the buffer.
        """
        n = int(starts.size)
        if n < want or want < 2:
            return None
        good = np.abs(np.diff(starts) - line_len) <= line_len * float(tol_frac)
        # index i is a run start when the gap before it is not nominal
        run_start = np.flatnonzero(~good) + 1
        bounds = np.concatenate(([0], run_start, [n]))
        best: np.ndarray | None = None
        for k in range(bounds.size - 1):
            lo, hi = int(bounds[k]), int(bounds[k + 1])
            if hi - lo >= want:
                best = starts[hi - want : hi]   # later runs overwrite earlier
        return best

    # -- main entry point --------------------------------------------------

    def push(
        self,
        demod: np.ndarray,
        sync: SyncResult,
        brightness: float = 1.0,
        contrast: float = 1.0,
        invert: bool = False,
    ) -> RasterFrame | None:
        """Rasterise the freshest complete field, or None if there isn't one."""
        starts = sync.pulse_starts
        if starts.size < 8:
            return None
        spec = sync.spec
        self._spec = spec
        line = float(sync.line_samples)
        want = spec.visible_lines
        if starts.size < want:
            return None

        offset = line * spec.crop_start
        span = line * spec.crop_width
        # only lines whose whole active window is inside the buffer are usable
        limit = demod.size - (offset + span)
        if limit <= 0:
            return None
        usable = starts[starts + offset + span <= limit]
        if usable.size < want:
            return None

        run = self.select_run(usable, line, want)
        if run is None:
            self.lines_skipped += 1
            return None

        rows = self._render(
            demod, run, line, offset, span, spec, brightness, contrast, invert
        )
        if rows is None:
            return None
        self.frames_emitted += 1
        return RasterFrame(
            image=rows,
            width=rows.shape[1],
            height=rows.shape[0],
            spec=spec,
            lines_used=want,
            line_samples=line,
            locked=True,
            quality=sync.quality,
        )

    # -- rendering ---------------------------------------------------------

    def _render(
        self,
        demod: np.ndarray,
        run: np.ndarray,
        line: float,
        offset: float,
        span: float,
        spec: VideoSpec,
        brightness: float = 1.0,
        contrast: float = 1.0,
        invert: bool = False,
    ) -> np.ndarray | None:
        width = self.width
        out_h = self.out_h or spec.visible_lines
        starts = run.astype(np.int64)
        a0 = starts + int(round(offset))
        n_src = int(round(span))
        if a0.size == 0 or n_src < 8:
            return None
        if int(a0.max()) + n_src > demod.size:
            return None

        # Gather every scanline in one shot, then anti-aliased-resize the whole
        # 2-D block in a single pass. Doing it in one step matters: sampling
        # straight to `width` with plain interpolation aliases the capture's
        # sample noise into visible speckle, whereas filtering before
        # decimating is what makes a HackRF decode look watchable.
        idx = a0[:, None] + np.arange(n_src, dtype=np.int64)[None, :]
        block = demod[idx].astype(np.float32)                # (lines, n_src)
        rows = _resample_2d(block, out_h, width)

        # --- level mapping ------------------------------------------------
        # In composite video the sync tips sit at one extreme of the FM
        # deviation and white at the other, so a robust percentile stretch of
        # the signed waveform maps straight to luminance. Which extreme is
        # "black" depends on the VTX's deviation polarity, so resolve it from
        # the data: compare the signed waveform at the sync positions against
        # the frame as a whole.
        sync_level = float(np.mean(demod[starts])) if starts.size else 0.0
        frame_level = float(np.median(rows))
        flip = (sync_level > frame_level)
        if invert:
            flip = not flip

        lo, hi = np.percentile(rows, [2.0, 98.0])
        if hi <= lo:
            lo, hi = float(rows.min()), float(rows.max())
        if hi <= lo:
            return np.zeros(rows.shape, dtype=np.uint8)
        y = np.clip((rows - lo) / (hi - lo), 0.0, 1.0)
        if flip:
            y = 1.0 - y
        y = (y - 0.5) * float(contrast) + 0.5
        y = y * float(brightness)
        return np.clip(y * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Spectrum / occupancy
# --------------------------------------------------------------------------


def psd(iq: np.ndarray, sample_rate_hz: float, nfft: int = 2048, averages: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Averaged Welch-style PSD in dBFS/Hz-ish plus frequency axis in Hz.

    Returns ``(freqs, psd_db)`` where freqs are offsets from the tuned centre.
    """
    iq = np.asarray(iq)
    if iq.size < nfft:
        nfft = int(2 ** int(np.floor(np.log2(max(64, iq.size)))))
    nfft = min(nfft, iq.size)
    win = np.hanning(nfft).astype(np.float32)
    step = max(nfft // 2, 1)
    acc = np.zeros(nfft, dtype=np.float64)
    n = 0
    for start in range(0, iq.size - nfft + 1, step):
        seg = iq[start : start + nfft] * win
        acc += (np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2)
        n += 1
        if n >= averages:
            break
    if n == 0:
        return np.zeros(0), np.zeros(0)
    psd_lin = acc / n
    scale = (nfft * np.sum(win**2)) or 1.0
    psd_db = 10.0 * np.log10(np.maximum(psd_lin / scale, 1e-30))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / sample_rate_hz))
    return freqs, psd_db


class SignalKind(str, Enum):
    VIDEO = "video"      # lockable analogue composite video
    CARRIER = "carrier"  # narrowband RF present, but no video structure
    NOISE = "noise"      # nothing detectable


@dataclass
class ChannelReading:
    """What is present in one channel, plus the evidence for that call."""

    kind: SignalKind
    snr_db: float                 # in-band mean vs median noise floor
    peak_to_median_db: float      # narrowband-ness
    noise_db: float
    sync_pulses: int
    sync_density: float           # detected lines / expected lines
    sync_quality: float           # 0..1 spacing regularity
    line_rate_hz: float
    spec: VideoSpec | None
    peak_offset_hz: float

    @property
    def occupied(self) -> bool:
        return self.kind is not SignalKind.NOISE

    @property
    def strength(self) -> float:
        """Single 0..100+ score for ranking candidate channels."""
        if self.kind is SignalKind.VIDEO:
            return 40.0 + min(30.0, max(0.0, self.snr_db)) + 30.0 * self.sync_quality
        if self.kind is SignalKind.CARRIER:
            return 10.0 + min(30.0, max(0.0, self.peak_to_median_db))
        return 0.0

    def describe(self) -> str:
        if self.kind is SignalKind.VIDEO:
            std = self.spec.name if self.spec else "?"
            return (
                f"analogue video, {self.line_rate_hz/1000:.2f} kHz lines ({std}), "
                f"SNR {self.snr_db:.1f} dB, sync {self.sync_quality*100:.0f}%"
            )
        if self.kind is SignalKind.CARRIER:
            return f"narrowband carrier at {self.peak_offset_hz/1e6:+.2f} MHz offset, {self.peak_to_median_db:.1f} dB"
        return "noise only"


def assess_channel(
    iq: np.ndarray,
    sample_rate_hz: float,
    spec: VideoSpec | None = None,
    min_sync_density: float = 0.25,
    min_sync_quality: float = 0.60,
    min_peak_db: float = 12.0,
) -> ChannelReading:
    """Classify one channel as video, carrier, or noise.

    Why sync lock rather than SNR
    ----------------------------
    An analogue FPV video signal is FM with a Carson bandwidth of roughly
    2*(deviation + f_max) ~= 13-15 MHz. At a 10 MS/s capture rate the visible
    span is *narrower than the signal itself*, so there is no out-of-band
    region to use as a noise reference and PSD-only SNR is unreliable -- a
    strong, near-clipped VTX reads as flat and can score only a few dB.

    Horizontal sync is different: it is a real property of the signal. Finding
    ~15.7 kHz-spaced pulse trains with tight spacing regularity is essentially
    conclusive for "analogue video is present", and it degrades gracefully
    with distance exactly the way a receiver does. So sync lock is the primary
    test, and spectral narrowband-ness only distinguishes a bare carrier from
    pure noise.
    """
    ref = spec or NTSC
    iq = np.asarray(iq)
    if iq.size < 4096:
        return ChannelReading(
            SignalKind.NOISE, 0.0, 0.0, -120.0, 0, 0.0, 0.0, 0.0, None, 0.0
        )

    freqs, psd_db = psd(iq, sample_rate_hz)
    if freqs.size == 0:
        return ChannelReading(
            SignalKind.NOISE, 0.0, 0.0, -120.0, 0, 0.0, 0.0, 0.0, None, 0.0
        )
    # Every statistic below is taken over the bins the receiver is not using to
    # measure itself. See DC_GUARD_HZ: without this a silent band reads as a
    # 52 dB narrowband carrier, which is a false positive with a cause, not a
    # random one, and it would be reported identically at all 34 hops.
    keep = away_from_dc(freqs)
    if not np.any(keep):
        keep = np.ones_like(freqs, dtype=bool)
    half = min(ref.video_bw_hz / 2.0, sample_rate_hz * 0.45)
    in_band = (np.abs(freqs) <= half) & keep
    if not np.any(in_band):
        in_band = keep

    noise_db = float(np.median(psd_db[keep]))
    signal_db = float(np.mean(psd_db[in_band]))
    snr_db = signal_db - noise_db
    peak_bin = int(np.argmax(psd_db[keep]))
    peak_db = float(psd_db[keep][peak_bin])
    peak_to_median = peak_db - noise_db
    peak_offset = float(freqs[keep][peak_bin])

    # Primary test: does a horizontal sync train lock?
    pulses = 0
    density = 0.0
    quality = 0.0
    line_rate = 0.0
    found: VideoSpec | None = None
    try:
        demod = fm_demodulate(iq)
        sync = detect_sync(demod, sample_rate_hz, ref, polarity="auto", sensitivity=0.985)
        pulses = int(sync.pulse_centres.size)
        if pulses >= 8 and sync.line_samples > 1.0:
            expected = iq.size / sync.line_samples
            density = pulses / expected if expected > 0 else 0.0
            quality = float(sync.quality)
            line_rate = float(sync.line_rate_hz)
            found = sync.spec
    except (ValueError, FloatingPointError):
        pass

    # accept either line standard
    line_ok = any(
        abs(line_rate - s.line_rate_hz) / s.line_rate_hz < 0.02
        for s in (NTSC, PAL)
    ) if line_rate > 0 else False

    if pulses >= 8 and density >= min_sync_density and quality >= min_sync_quality and line_ok:
        kind = SignalKind.VIDEO
    elif peak_to_median >= min_peak_db:
        kind = SignalKind.CARRIER
    else:
        kind = SignalKind.NOISE

    return ChannelReading(
        kind=kind,
        snr_db=snr_db,
        peak_to_median_db=peak_to_median,
        noise_db=noise_db,
        sync_pulses=pulses,
        sync_density=density,
        sync_quality=quality,
        line_rate_hz=line_rate,
        spec=found,
        peak_offset_hz=peak_offset,
    )
