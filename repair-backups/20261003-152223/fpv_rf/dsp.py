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

# TEMP DIAG: counters and optional statistics for the decode-failure
# investigation. Off unless FPV_RF_DIAG is set; delete this import and every
# TEMP DIAG block below together with fpv_rf/diag.py.
from . import diag

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

    No caller passes ``decimate_to`` -- the decode path keeps the full rate,
    because line sampling needs the sample count to match the line rate. It is
    here for offline work, and it is fixed rather than left broken: asked to go
    from 10 MS/s to 5, it used to compute ``ratio = 2``, hand that to
    ``resample_poly`` as ``up=2, down=1`` and *double* the array, then claim the
    result was running at 20 MS/s. Two things wrong, in the same line: the
    factors name the wrong direction, and they are the sample rates rather than
    the ratio between them. 10 -> 5 is one sample out for every one kept, which
    is ``resample_poly(x, 1, 2)``.
    """
    x = np.asarray(x, dtype=np.float32)
    nyq = sample_rate_hz / 2.0
    cutoff = float(np.clip(cutoff_hz, 100.0, nyq * 0.95))

    if decimate_to and decimate_to < sample_rate_hz:
        # ``up/down`` is how much the length changes: down > up to shorten.
        # _rational gives the smallest integers that express the ratio, so the
        # achieved rate is reported from those factors rather than from
        # ``decimate_to``, which they only approximate.
        up, down = _rational(float(sample_rate_hz) / float(decimate_to))
        if _HAVE_SCIPY:
            x = resample_poly(x, down, up).astype(np.float32)
            achieved = sample_rate_hz * down / up
        else:
            # No polyphase filter available, so the only decimation that can be
            # done exactly is by a whole number of samples, and the achieved
            # rate is that whole-number one rather than the requested one.
            factor = max(1, int(round(up / down)))
            x = _boxcar_decimate(x, factor)
            achieved = sample_rate_hz / factor
        sample_rate_hz = achieved
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
    #: TEMP DIAG: per-polarity detail, present only when ``detect_sync`` was
    #: called with ``collect_diag=True``. None otherwise, and none of the
    #: fields above are affected by its presence.
    diag: "SyncDiag | None" = None


# --------------------------------------------------------------------------
# TEMP DIAG: sync diagnostics. Delete with fpv_rf/diag.py.
# --------------------------------------------------------------------------


@dataclass
class PolarityDiag:
    """Everything one sign of the envelope produced, whether or not it won."""

    polarity: int
    threshold: float = 0.0
    raw_edges: int = 0            # rising edges before clustering
    clustered: int = 0            # edges that survived min_gap
    quality: float = 0.0
    near_count: int = 0           # gaps inside the +-12% tolerance band
    diffs: int = 0                # gaps considered
    density: float = 0.0          # clustered * line_samples / buffer
    plaus: float = 0.0
    rank: float = 0.0
    line_samples: float = 0.0
    spacing_min: float = 0.0
    spacing_max: float = 0.0
    spacing_mean: float = 0.0
    spacing_median: float = 0.0
    excursion_min: float = 0.0    # widths of complete above-threshold runs
    excursion_median: float = 0.0
    excursion_max: float = 0.0
    chosen: bool = False

    def flatten(self, prefix: str) -> dict[str, float]:
        out = {
            "thr": self.threshold,
            "raw": self.raw_edges,
            "clus": self.clustered,
            "qual": self.quality,
            "near": self.near_count,
            "gaps": self.diffs,
            "dens": self.density,
            "plaus": self.plaus,
            "rank": self.rank,
            "line": self.line_samples,
            "sp_min": self.spacing_min,
            "sp_max": self.spacing_max,
            "sp_mean": self.spacing_mean,
            "sp_med": self.spacing_median,
            "exc_min": self.excursion_min,
            "exc_med": self.excursion_median,
            "exc_max": self.excursion_max,
            "chosen": int(self.chosen),
        }
        return {f"{prefix}_{k}": float(v) for k, v in out.items()}

    # TEMP DIAG: the same numbers under their own names, for a JSON capture
    # rather than a one-line log. Same source either way -- both are read out of
    # the call, never re-derived -- so a capture and the per-second aggregate
    # cannot disagree about the numbers behind a decision.
    def as_dict(self) -> dict:
        return {
            "polarity": int(self.polarity),
            "threshold": float(self.threshold),
            "raw_edges": int(self.raw_edges),
            "clustered": int(self.clustered),
            "quality": float(self.quality),
            "near_count": int(self.near_count),
            "gaps_considered": int(self.diffs),
            "density": float(self.density),
            "plausibility": float(self.plaus),
            "rank": float(self.rank),
            "line_samples": float(self.line_samples),
            "spacing_min": float(self.spacing_min),
            "spacing_max": float(self.spacing_max),
            "spacing_mean": float(self.spacing_mean),
            "spacing_median": float(self.spacing_median),
            "excursion_min": float(self.excursion_min),
            "excursion_median": float(self.excursion_median),
            "excursion_max": float(self.excursion_max),
            "selected": bool(self.chosen),
        }


@dataclass
class SyncDiag:
    """The two-polarity decision, with the numbers behind it.

    The thresholds here are the live ones -- read out of the call, not
    re-derived -- so this describes the detector that actually ran rather than
    a re-implementation of it.
    """

    sensitivity: float
    nominal_line: float
    min_gap: int
    n_samples: int
    early_break: bool = False     # would the live loop have stopped after one?
    decided_polarity: int = 0     # the one the live loop would have stopped at
    positive: PolarityDiag | None = None
    negative: PolarityDiag | None = None

    def flatten(self) -> dict[str, float]:
        out: dict[str, float] = {
            "sens": self.sensitivity,
            "nominal": self.nominal_line,
            "min_gap": float(self.min_gap),
            "n": float(self.n_samples),
            "early_break": float(self.early_break),
            "decided_polarity": float(self.decided_polarity),
        }
        for name, side in (("pos", self.positive), ("neg", self.negative)):
            if side is not None:
                out.update(side.flatten(name))
        return out

    # TEMP DIAG: as_dict(), for a capture file. Both polarities are always
    # present here -- the whole point of a publication-matched capture is that
    # the polarity which lost is recorded beside the one that won, because
    # "the decoder picked the wrong extreme" is only diagnosable from the pair.
    def as_dict(self) -> dict:
        return {
            "sensitivity": float(self.sensitivity),
            "nominal_line_samples": float(self.nominal_line),
            "min_gap": int(self.min_gap),
            "n_samples": int(self.n_samples),
            "early_break": bool(self.early_break),
            "decided_polarity": int(self.decided_polarity),
            "positive": self.positive.as_dict() if self.positive else None,
            "negative": self.negative.as_dict() if self.negative else None,
        }


def sync_diagnostics(
    demod: np.ndarray,
    sample_rate_hz: float,
    spec: VideoSpec | None = None,
    polarity: str = "auto",
    sensitivity: float = 0.975,
) -> SyncDiag:
    """TEMP DIAG: :func:`detect_sync`'s per-polarity statistics, on demand.

    A thin wrapper, so a tool reads the same numbers the live decoder does
    instead of re-implementing the detector and disagreeing with it. Returns an
    empty-ish result if detection raised or found nothing, never raises.
    """
    try:
        got = detect_sync(
            demod, sample_rate_hz, spec=spec, polarity=polarity,
            sensitivity=sensitivity, collect_diag=True,
        )
    except Exception:
        return SyncDiag(
            sensitivity, expected_line_samples(sample_rate_hz, spec or NTSC), 0,
            int(np.asarray(demod).size),
        )
    if got.diag is not None:
        return got.diag
    return SyncDiag(
        sensitivity,
        expected_line_samples(sample_rate_hz, spec or NTSC),
        0,
        int(np.asarray(demod).size),
    )


def _diag_record(sdiag: "SyncDiag | None", side: "PolarityDiag | None") -> None:
    """TEMP DIAG: file one polarity's result under its own name."""
    if sdiag is None or side is None:
        return
    if side.polarity > 0:
        sdiag.positive = side
    else:
        sdiag.negative = side


def _run_widths(mask: np.ndarray) -> np.ndarray:
    """TEMP DIAG: widths of the *complete* above-threshold runs in ``mask``.

    Read this as "how wide is the excursion the live threshold admits", not as
    "how wide is a sync pulse". The threshold is a high percentile of the
    envelope, so it normally cuts the top off a plateau instead of isolating it,
    and a median of a few samples is what a healthy 10 MS/s signal reports. What
    matters is the contrast between the two polarities and between a working
    capture and a failing one: widths in the tens of samples, or a minimum far
    below the median, mean the threshold is sitting somewhere else entirely.

    Runs touching either end of the buffer are dropped rather than closed
    artificially -- their width is not measurable, and inventing one would make
    the minimum look better than it is.

    Trimming has to be done before pairing. Transitions alternate, so a mask
    that is already true at index 0 opens with a *falling* edge and one that is
    still true at the end closes with a *rising* edge; zipping the two lists
    anyway pairs an edge with one before it and reports negative widths, which
    is how this first spelled it.
    """
    if mask.size < 2:
        return np.zeros(0, dtype=np.int64)
    up = np.flatnonzero(mask[1:] > mask[:-1]) + 1
    down = np.flatnonzero(mask[1:] < mask[:-1]) + 1
    if bool(mask[0]):
        down = down[1:]        # that run opened before the buffer did
    if bool(mask[-1]):
        up = up[:-1]           # this one never closes inside it
    n = int(min(up.size, down.size))
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.abs(down[:n] - up[:n]) + 1


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
    collect_diag: bool = False,
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

    ``collect_diag`` (TEMP DIAG) additionally measures the polarity the live
    loop would have skipped. It does not change the answer: once the loop would
    have broken out, the winning candidate is locked in and the second polarity
    can only report. Nothing here runs unless asked for.
    """
    x = np.asarray(demod, dtype=np.float32)
    if x.size < 64:
        raise ValueError("not enough samples for sync detection")

    ref = spec or NTSC
    nominal = expected_line_samples(sample_rate_hz, ref)
    min_gap = max(4, int(nominal * 0.5))

    env = x

    # TEMP DIAG: read the live thresholds into the record instead of restating
    # them, so this cannot drift from what the detector actually used.
    sdiag = SyncDiag(sensitivity, float(nominal), min_gap, int(x.size)) if collect_diag else None
    locked = False                       # TEMP DIAG: the live early-break, honoured

    pols = [1, -1] if polarity == "auto" else [1 if polarity == "positive" else -1]
    best: SyncResult | None = None
    best_score = -1.0
    for sign in pols:
        score = sign * env
        thr = _high_percentile(score, sensitivity * 100.0)
        mask = score > thr
        raw_edges = _rising_edges(mask)
        starts = _cluster(raw_edges, min_gap)
        side = PolarityDiag(sign, thr, int(raw_edges.size), int(starts.size)) if sdiag else None
        if starts.size < 4:
            _diag_record(sdiag, side)
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
        if side is not None:
            side.quality = quality
            side.near_count = int(np.count_nonzero(near))
            side.diffs = int(diffs.size)
            side.density = density
            side.plaus = plaus
            side.rank = rank
            side.line_samples = line_samples
            if diffs.size:
                side.spacing_min = float(diffs.min())
                side.spacing_max = float(diffs.max())
                side.spacing_mean = float(diffs.mean())
                side.spacing_median = float(np.median(diffs))
            widths = _run_widths(mask)
            if widths.size:
                side.excursion_min = float(widths.min())
                side.excursion_median = float(np.median(widths))
                side.excursion_max = float(widths.max())
        if rank > best_score and not locked:
            best_score = rank
            if side is not None:
                side.chosen = True
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
        if sdiag is not None:
            _diag_record(sdiag, side)
        # A candidate that is both well-spaced and at the right line density is
        # already unambiguous, and the opposite polarity only exists to catch
        # sign mistakes. Re-running the whole pipeline to confirm it doubles the
        # cost of the most expensive function in the decoder, so stop here once
        # the answer is decisive. rank >= 0.5 with quality >= 0.8 cannot be
        # produced by locking onto picture detail.
        if best is not None and quality >= 0.8 and rank >= 0.5:
            if sdiag is None:
                break                        # the live path, unchanged
            # TEMP DIAG: with a record to fill, keep going so the polarity that
            # was skipped gets measured too -- but only the early break can
            # change `best`, which is what `locked` is for.
            if not sdiag.early_break:
                sdiag.early_break = True
                sdiag.decided_polarity = sign
            locked = True
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
            diag=sdiag,                    # TEMP DIAG
        )
    best.diag = sdiag                      # TEMP DIAG
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
    #: How many of ``lines_used`` rows were placed on the fitted line grid rather
    #: than at a detected sync pulse. The *samples* behind those rows are real
    #: video -- it is the line's position that was reconstructed, not its
    #: content -- but a frame that needed 40 of 240 positions reconstructed is a
    #: different claim from one that needed none, and the count is carried so the
    #: caller can say which one it is showing.
    lines_from_grid: int = 0
    #: Absolute half-open sample interval ``[first_sample, last_sample)`` of the
    #: scanlines this picture was assembled from, when the caller said where its
    #: demodulator window starts. Pure provenance: it names which part of the
    #: stream a picture came from, so a consumer can tell a new picture from one
    #: that re-used lines it has already been shown. ``(-1, -1)`` when the caller
    #: could not place the window, which is a real answer -- an offline decode of
    #: a bare array has no stream position -- and not the same as ``(0, 0)``.
    first_sample: int = -1
    last_sample: int = -1
    #: TEMP DIAG: see fpv_rf/diag.py. How many of ``lines_used`` rows came from a
    #: real sync detection rather than from :func:`line_grid`. It is the other
    #: half of :attr:`lines_from_grid` -- ``detected + from_grid == lines_used`` --
    #: and it is what says whether a frame was mostly *found* or mostly
    #: reconstructed, which ``from_grid`` alone cannot distinguish from a window
    #: that only ever needed filling.
    lines_detected: int = 0
    #: TEMP DIAG: the sample positions actually sampled for this frame, in the
    #: order :meth:`_render` gathered them -- detected and grid-filled rows
    #: alike. Kept so a published picture can be written out with the line
    #: geometry it was drawn from, which is otherwise gone the moment ``push``
    #: returns. ``want`` entries when a frame was produced; ``None`` otherwise.
    #: The rasteriser is rebuilt on every reset, so this does not accumulate.
    line_positions: "np.ndarray | None" = None


@dataclass
class LineRun:
    """A chosen window of ``want`` scanlines, in sample positions.

    ``positions`` always has exactly ``want`` entries. ``from_grid`` is how many
    of them were placed on the fitted grid rather than at a real detection.
    """

    positions: np.ndarray
    detected: int
    from_grid: int


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

    Strategy: render the most recent run of exactly ``visible_lines`` scanlines,
    sampled at their measured positions where the sync detector found them and on
    a locally fitted grid where it did not.

    Restricting a frame to a *contiguous* run of line numbers is what makes the
    picture clean, and it is the single biggest quality lever in this decoder:

    * Field blanking (22.5 lines in NTSC, 24.5 in PAL) can never fall inside a
      frame, so no row is interpolated across a discontinuity. :attr:`max_fill`
      is what keeps the two apart; see :meth:`select_run` for the measurement.
    * Across ~240 lines the per-line detection jitter does not accumulate into
      visible shear, so detected lines are sampled at their raw positions rather
      than on one global grid. Measured on a known capture, doing this scored
      +0.80 adjacent-row correlation; forcing a single global grid across the
      whole capture scored +0.25, because any error in the fitted period compounds
      over 8000 lines.
    * "Most recent" means every emitted frame is the freshest thing in the
      buffer, which is what a live viewer wants.

    The cost is that a field needing a *reconstructed* row position reports it
    (:attr:`RasterFrame.lines_from_grid`) rather than passing it off as fully
    detected, and a dropout larger than :attr:`max_fill` ends the run.

    On picture quality: adjacent-row correlation on the *raw* 483x240 active
    window measures about +0.09 for this decoder and about +0.09 for the
    original still decoder too, so that number describes the capture, not the
    code -- a HackRF capture of a 5.8 GHz VTX is genuinely noisy. Scores around
    +0.82 appear only *after* the downscale to display size, because
    ``_downscale`` low-pass filters away sample noise that is uncorrelated
    between neighbouring pixels. That filtering is real and worth doing for a
    live display, but it should not be mistaken for a sharper decode.
    """

    def __init__(self, width: int = 160, out_h: int | None = None,
                 max_fill: int = 20) -> None:
        self.width = int(width)
        self.out_h = int(out_h) if out_h else None
        #: Largest signal dropout :meth:`select_run` bridges rather than treats as
        #: the end of a field. 20 is measured, not chosen: the real 5802 MHz
        #: capture's dropouts run 5-20 lines and field blanking is 22.5 (NTSC) /
        #: 24.5 (PAL), so a separating integer sits between them. This bridges
        #: every dropout observed and no blanking interval, which is what keeps
        #: the no-interpolation-across-a-discontinuity property intact.
        self.max_fill = int(max_fill)
        self._spec: VideoSpec | None = None
        self.frames_emitted = 0
        self.lines_skipped = 0
        # TEMP DIAG: which stage refused, and what it was looking at. See
        # fpv_rf/diag.py; counts only, no effect on the search.
        self._diag = diag.section("raster")

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
        starts: np.ndarray,
        line_len: float,
        want: int,
        tol_frac: float = 0.15,
        max_fill: int = 20,
        diag_out: dict | None = None,
    ) -> LineRun | None:
        """Freshest run of ``want`` lines, bridging the signal's dropouts.

        **Requiring ``want`` consecutive *detected* lines made real signals
        undecodable, and this was the reason.** Measured on a genuine 5802 MHz
        HackRF capture of an analog FPV VTX, the sync detector fires on 82.7% of
        lines -- but the missing 17.3% are not scattered. The gap histogram is
        399 gaps of one line, 9 of two lines, and then 8 chunks of 5, 6, 6, 8, 8,
        8, 17 and 20 lines. So the losses are signal dropouts, not jitter, and
        because the largest runs of clean detections were 112 lines against the
        240 a field needs, the decoder emitted no frame at all from a real VTX
        while the sync detector simultaneously reported quality 0.90. It claimed
        a lock it could not deliver; :meth:`push` now reconciles the two.

        The rule is therefore: a gap of up to ``max_fill`` line periods is
        *bridged* rather than ending the run, and larger gaps still end it. That
        second half is what preserves the property this class was written for --
        field blanking is never spanned. The two are told apart by size, which is
        only safe because the numbers are integers and widely separated:

        * NTSC blanking is 22.5 lines and PAL's is 24.5, so any gap rounds to 22
          or more.
        * The largest dropout in the real capture is 20 lines, which rounds to 20.

        A default of 20 therefore bridges every dropout measured and no blanking
        interval, with the separating integer in between. Verified by sweeping the
        parameter over the real capture: at 8 the mean adjacent-row correlation is
        +0.8954, at 12 +0.8984, at 20 +0.9020, and it stops improving past 20
        (+0.9032 at 32) -- i.e. bridging blanking buys nothing measurable, so there
        is no reason to do it. Against 2, which bridged nothing useful, no frame
        was produced at all.

        Note what "bridged" means: the missing rows are placed on the grid fitted
        to the surrounding detections, and the *samples* at those positions are
        real video. Only the lines' positions are reconstructed. Detected lines
        still get their own measured positions, and the number of reconstructed
        ones is reported on the frame rather than quietly absorbed.

        ``diag_out`` (TEMP DIAG) receives a flat dict naming *which stage*
        rejected: ``degenerate``, ``no_runs``, ``short_segment``, ``span``,
        ``older``, ``no_positions``, ``coverage`` or ``none``. The gap
        histogram, the inferred period and the longest run seen are in it too.
        Nothing about the search itself changes.
        """
        n = int(starts.size)
        d = diag_out if diag_out is not None else {}
        d.clear()
        d["want"] = int(want)
        d["detections"] = n
        d["max_fill"] = int(max_fill)
        d["reason"] = "none"
        d["max_span"] = 0
        d["runs"] = 0
        d["period"] = float(line_len)
        if n < 2 or want < 2 or line_len <= 0:
            d["reason"] = "degenerate"
            return None

        # Walk the detections, turning each gap into an integer number of lines.
        # A gap of ~22 lines (field blanking) therefore advances the counter by
        # 22, and the consecutive-run search below rejects it, while a gap of one
        # or two lines stays consecutive and is fillable.
        single = np.diff(starts)
        single = single[single < 1.5 * float(line_len)]
        period = float(np.median(single)) if single.size >= 3 else float(line_len)
        if not np.isfinite(period) or period <= 0:
            period = float(line_len)
        d["period"] = period
        steps = np.maximum(1, np.rint(np.diff(starts) / period).astype(np.int64))
        line_no = np.concatenate(([0], np.cumsum(steps))).astype(np.int64)

        # TEMP DIAG: what the missing lines actually look like. A cluster of
        # one-line gaps is jitter; a handful of large ones is the signal
        # dropping out, and the two call for different fixes.
        missed = steps[steps > 1] - 1
        if missed.size:
            hist: dict[int, int] = {}
            for m in missed.tolist():
                hist[int(m)] = hist.get(int(m), 0) + 1
            d["gap_hist"] = ",".join(f"{k}:{hist[k]}" for k in sorted(hist)[:12])
            d["gap_max"] = int(missed.max())
            d["gap_total"] = int(missed.sum())
        else:
            d["gap_hist"] = ""
            d["gap_max"] = 0
            d["gap_total"] = 0

        # Runs of consecutive line numbers, allowing a bounded fill.
        fillable = (steps - 1) <= int(max_fill)
        breaks = np.flatnonzero(~fillable) + 1
        bounds = np.concatenate(([0], breaks, [n]))
        d["runs"] = int(bounds.size - 1)

        best: LineRun | None = None
        best_end = -1
        reason = "no_runs"
        for k in range(bounds.size - 1):
            lo, hi = int(bounds[k]), int(bounds[k + 1])
            if hi - lo < 2:
                reason = "short_segment"
                continue
            span = int(line_no[hi - 1] - line_no[lo]) + 1
            if span > int(d["max_span"]):
                d["max_span"] = span
            if span < want:
                reason = "span"
                continue
            end = int(line_no[hi - 1])
            if end <= best_end:
                reason = "older"
                continue          # an older run cannot beat a fresher one
            # Freshest window inside this run: the last `want` lines.
            first = end - want + 1
            sel = (line_no >= first) & (line_no <= end)
            got = starts[sel]
            if got.size == 0:
                reason = "no_positions"
                continue
            sel_no = line_no[sel]

            # Fit the grid to the *global* line numbering, not to a local one.
            # line_grid() returns positions for detected lines only, which is
            # useless here: with ~18% of lines missing it can never produce
            # `want` positions, and the size guard below rejected every frame.
            # What is needed is the grid for every integer line in the window,
            # with the measured positions laid back on top where they exist.
            _, _, per2 = line_grid(got, period, max_gap_lines=float(max_fill) + 2.0)
            if not np.isfinite(per2) or per2 <= 0:
                per2 = period
            phase = float(np.median(got - per2 * sel_no.astype(np.float64)))
            full = phase + per2 * np.arange(first, end + 1, dtype=np.float64)

            # Every detected line keeps its own measured position; only the
            # genuinely missing ones are invented. This is the accuracy the
            # class was built around, and the count of what was invented is
            # reported rather than hidden.
            chosen = full.copy()
            slot = sel_no - first
            hit = (slot >= 0) & (slot < full.size)
            chosen[slot[hit]] = got[hit]
            detected = int(hit.sum())
            from_grid = int(want - detected)
            if chosen.size != want or detected == 0:
                reason = "coverage"
                continue
            best = LineRun(chosen, detected, from_grid)
            best_end = end
        if best is None:
            d["reason"] = reason
        else:
            d["detected"] = best.detected
            d["from_grid"] = best.from_grid
        return best
    # -- main entry point --------------------------------------------------

    def push(
        self,
        demod: np.ndarray,
        sync: SyncResult,
        brightness: float = 1.0,
        contrast: float = 1.0,
        invert: bool = False,
        origin: int = -1,
    ) -> RasterFrame | None:
        """Rasterise the freshest complete field, or None if there isn't one.

        ``origin`` is the absolute source-sample index of ``demod[0]``, when the
        caller knows it, and travels straight into the frame's
        ``first_sample``/``last_sample``. It is the shortest possible statement
        of "which part of the stream is this picture", and it has to be taken
        here rather than reconstructed by the caller: the interval is a property
        of the lines this call actually sampled, which after
        :func:`limit_video_bandwidth` is not by inspection the buffer that was
        decoded.
        """
        rd = self._diag
        starts = sync.pulse_starts
        if starts.size < 8:
            rd.bump("reject_few_starts")
            rd.gauge("reject", "few_starts")
            return None
        spec = sync.spec
        self._spec = spec
        line = float(sync.line_samples)
        want = spec.visible_lines

        offset = line * spec.crop_start
        span = line * spec.crop_width
        # A line is usable when its whole active window lies inside the buffer,
        # and the window is addressed here exactly as :meth:`_render` addresses
        # it: ``int(round(offset))`` samples in, ``int(round(span))`` samples
        # wide, the start itself truncated to an int64. Eligibility looser or
        # stricter than the render that follows it is a bug in itself, and this
        # bound used to be both.
        #
        # It compared ``start + (offset + span)`` against
        # ``size - (offset + span)`` -- two windows of padding where one is
        # needed -- so the last row of an exactly-fitting field was thrown away.
        # Measured: an NTSC field of 240 lines at 640 samples/line, offset
        # 115.2 and span 486.4, whose last row ends at sample 153561, admits
        # only 239 of its 240 rows; select_run then cannot find the 240-line run
        # it requires and push refuses a field the renderer could have drawn.
        # 288 PAL lines at the same period fail the same way. Deriving the bound
        # in floats rather than in the render's own integers costs the same
        # thing again, one sample at a time, wherever the float end of a row
        # lands just past an integer sample count.
        off_i = int(round(offset))
        n_src = int(round(span))
        rd.gauge("buffer_samples", int(demod.size))
        rd.gauge("window_need", int(off_i + n_src))
        if n_src < 8 or demod.size - off_i < n_src:
            rd.bump("reject_buffer_short")
            rd.gauge("reject", "buffer_short")
            return None
        row0 = starts.astype(np.int64) + off_i
        # ``row0 >= 0`` as well as the upper end: a negative crop_start puts the
        # window's first samples before the buffer, and a negative index does not
        # fail there, it wraps round to the far end of the buffer and renders a
        # row of the wrong picture.
        usable = starts[(row0 >= 0) & (row0 + n_src <= demod.size)]

        # The floor on detections is the pigeonhole bound, not a tolerance knob:
        # :meth:`select_run` tolerates at most ``max_fill`` consecutive misses, so
        # 240 rows cannot be filled from fewer than 80 detections. Requiring
        # ``want`` detections instead -- which is what this used to do -- is what
        # made real captures undecodable, and the check has no meaning now that a
        # run may be partly reconstructed.
        floor = max(8, want // (self.max_fill + 1))
        rd.gauge("usable", int(usable.size))
        rd.gauge("floor", int(floor))
        rd.gauge("want", int(want))
        if usable.size < floor:
            rd.bump("reject_too_few_usable")
            rd.gauge("reject", "too_few_usable")
            return None

        run_info: dict = {}
        run = self.select_run(
            usable, line, want, max_fill=self.max_fill, diag_out=run_info
        )
        # TEMP DIAG: the search's own account of itself, promoted to gauges so it
        # reaches the once-a-second line instead of dying with the call.
        for k, v in run_info.items():
            rd.gauge(f"run_{k}", v)
        rd.bump("select_run_calls")
        if run is None:
            self.lines_skipped += 1
            rd.bump("reject_no_run")
            rd.gauge("reject", f"run_{run_info.get('reason', 'none')}")
            return None
        rd.bump("runs_ok")

        rows = self._render(
            demod, run.positions, line, offset, span, spec, brightness, contrast, invert
        )
        if rows is None:
            rd.bump("reject_render")
            rd.gauge("reject", "render")
            return None
        self.frames_emitted += 1
        rd.bump("frames")
        rd.gauge("from_grid", int(run.from_grid))
        rd.gauge("reject", "none")
        if int(origin) >= 0:
            # The interval of the *picture*: the first and last scanlines it was
            # assembled from, in absolute samples. Not the interval of the whole
            # demodulator window -- that grows to the window's length and then
            # stops moving, because the window is full, so every field after the
            # first few would carry the same numbers and no consumer could tell
            # a new picture from the same field drawn again. These move by one
            # field per field, which is what makes "is this a new picture"
            # answerable rather than guessable.
            #
            # Derived from :meth:`_render`'s own integer addresses --
            # ``int(run.positions[k]) + int(round(offset))``, then
            # ``n_src`` samples wide -- because those are the samples actually
            # gathered. The positions alone are the *sync detections*, which sit
            # ``int(round(offset))`` samples before the picture does: NTSC's
            # crop is 0.18 of a line, about 115 samples, so an interval quoted
            # from the detections describes a stretch of stream 115 samples
            # before the one the picture came from, at both ends, on every field.
            # It read as a plausible number and it was wrong.
            #
            # ``last_sample`` is exclusive, as everywhere else in this codebase:
            # demodulator output ``j`` is the phase step between IQ samples ``j``
            # and ``j+1``, so the picture's samples are ``[lo, hi)`` and the
            # second IQ sample supporting its final step is the one just outside.
            lo = int(origin) + int(run.positions[0]) + off_i
            hi = int(origin) + int(run.positions[-1]) + off_i + n_src
        else:
            lo = hi = -1
        return RasterFrame(
            image=rows,
            width=rows.shape[1],
            height=rows.shape[0],
            spec=spec,
            lines_used=want,
            line_samples=line,
            locked=True,
            quality=sync.quality,
            lines_from_grid=run.from_grid,
            first_sample=lo,
            last_sample=hi,
            # TEMP DIAG: the line geometry, for a capture written beside the
            # picture. Pure addition -- nothing below reads either field.
            lines_detected=run.detected,
            line_positions=run.positions,
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
