"""Automatic band search and channel selection.

A HackRF at 10 MS/s sees 10 MHz at a time, so an FPV band of 125 MHz has to be
searched in hops. Doing that at full quality would mean a field of samples per
hop -- 16.7 ms plus a retune each, tens of seconds for one sweep. Instead the
search runs in two passes, cheap-then-thorough, and the difference matters
because *the cheap pass decides where to look carefully*.

    pass 1  coarse    ~1.6 ms of IQ per hop, 9 MHz steps. Count the spectrum
                       bins that sit well above the median. Measured on this
                       machine at that dwell: gaussian noise 0 such bins on 20
                       out of 20 blocks, a sim VTX 191, a real HackRF capture
                       162. Thermal noise never produces a broad hump, and a
                       VTX fills the whole span because its Carson bandwidth is
                       13-15 MHz. This pass costs ~1.6 ms per 9 MHz.

    pass 2  thorough  one field of IQ at each hop that passed, fully demodulated
                       and classified as video / carrier / noise, then snapped
                       to the nearest chart channel.

Two things about the coarse pass are worth stating because they are not obvious.

*The bin count, not the peak height.* The obvious "is there a peak" statistic --
highest bin minus median -- separates badly: measured 5.1 dB for noise against
16.7 dB for a real capture, an 11 dB gap that has to hold across every gain
setting, temperature and interferer. Counting bins above median+6 dB separates
0 from 162. A count is also the statistic that matches the alert condition
("energy above the noise floor") rather than a proxy for it. Peak-minus-median
is kept as a second gate, because a bare unmodulated carrier occupies too few
bins to trip the count.

*Coarse hops are 9 MHz and chart channels are 19-37 MHz apart.* So a hop that
clears the gate pins the transmitter to within +-4.5 MHz, comfortably less than
half the minimum channel spacing. That is what licenses snapping the hop centre
to the nearest chart channel, and it is also the honest limit of the frequency
estimate: a 10 MS/s tuner simply cannot resolve better than that without
narrowing its span. Every candidate therefore carries the uncertainty rather
than pretending to a precision it does not have.

The alert condition the app is built around -- "RF energy above the noise
floor" -- is the pass 1 gate, and it is deliberately *not* the same test as
"decodable video". A VTX that is too far away, or a video downlink on a band
this app cannot lock, is still an alert: the operator wants to know something is
transmitting, and the classification says which.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from . import bands, dsp
from .sdr import DEFAULT_SAMPLE_RATE, IQSource

#: A bin counts as "signal" if it is this far above the spectrum's median. The
#: median is the floor because it is the value most bins report, so it is
#: unmoved by a signal. Six dB is far outside the spread of a thermal-noise bin.
COARSE_BIN_GATE_DB = 6.0

#: Bins needed to call a hop occupied. Noise measured 0 (max 0 of 20 blocks);
#: a signal measured 162-191. Three is a very wide margin in both directions.
COARSE_BIN_COUNT = 3

#: Second gate, for a bare carrier. An unmodulated or drifting carrier can
#: occupy only a bin or two, so peak height catches what the count cannot.
#: Measured noise ceiling is 4.9 dB, so 12 dB does not clip it.
COARSE_PEAK_GATE_DB = 12.0

#: Samples for the coarse pass. 16384 at 10 MS/s is 1.6 ms, and it is exactly
#: 8 averaged 1024-point Welch segments -- 9.8 kHz bins, which resolves a VTX
#: against noise while staying short enough to sweep a whole band quickly.
COARSE_SAMPLES = 16_384
COARSE_NFFT = 1024
COARSE_AVERAGES = 8

#: Samples for the thorough pass. One NTSC field is 166.7k, so 200k is the
#: smallest useful capture that can contain a complete 240-line picture -- and
#: the only size at which "is there video here" is a real question rather than a
#: guess from a partial sync train.
THOROUGH_SAMPLES = 200_000

#: Width over which two findings are the same transmitter. Analogue FPV video
#: runs about 6.5 MHz of deviation, so Carson puts it near 15 MHz wide, and a
#: fine hop grid sees that one VTX across seven or eight consecutive hops.
#: Without this, one transmitter produces one alert per hop -- eight banners
#: saying the same thing, which is worse than useless because the operator
#: cannot tell it from eight transmitters.
VTX_WIDTH_HZ = 15_000_000


@dataclass
class CoarseHit:
    """What one coarse hop saw. Cheap, wide, and not very precise."""

    hop_hz: int
    bins_above: int
    peak_db: float          # peak bin minus median bin
    peak_offset_hz: float   # where the strongest bin sits, from the hop centre
    floor_db: float = -120.0  # the spectrum median: this hop's noise floor

    @property
    def occupied(self) -> bool:
        return self.bins_above >= COARSE_BIN_COUNT or self.peak_db >= COARSE_PEAK_GATE_DB

    @property
    def score(self) -> float:
        """Ranking weight for a hop: broad energy first, then peak height."""
        return self.bins_above + self.peak_db

    def describe(self) -> str:
        return (
            f"{self.bins_above} bins > +{COARSE_BIN_GATE_DB:.0f} dB, "
            f"peak +{self.peak_db:.1f} dB @ {self.peak_offset_hz/1e6:+.2f} MHz"
        )


@dataclass
class Candidate:
    """One frequency found to carry something, with the evidence."""

    frequency_hz: int
    reading: dsp.ChannelReading
    match: bands.Match | None
    #: Full width of the frequency window this was measured over, i.e. 2x the
    #: half-width. Zero when the operator named the frequency directly.
    uncertainty_hz: int
    coarse: CoarseHit | None = None
    #: True when no chart channel is within the hop's resolution, so the label
    #: is explicitly off-chart rather than a silently-wrong channel number.
    off_chart: bool = False
    #: When this candidate was merged with neighbours, the width of the group it
    #: represents -- an estimate of how wide the transmitter actually is.
    span_hz: int = 0
    #: How many findings were merged into this one.
    merged: int = 1

    @property
    def half_width_hz(self) -> int:
        return self.uncertainty_hz // 2

    @property
    def on_chart(self) -> bool:
        """True when a chart channel is within the hop's frequency resolution."""
        return not self.off_chart and self.match is not None

    @property
    def band(self) -> str:
        return self.match.band.key if self.match else "?"

    @property
    def channel(self) -> int | None:
        return self.match.channel if self.match else None

    @property
    def strength(self) -> float:
        return self.reading.strength

    @property
    def decodable(self) -> bool:
        return self.reading.kind is dsp.SignalKind.VIDEO

    def label(self) -> str:
        """Band + channel, the measured frequency, and the resolution.

        The tolerance is not decoration. A 10 MS/s tuner pins a transmitter to
        about +-half a hop, so a channel number is a *nearest* match, and
        saying so is the difference between a label an operator can trust on a
        band where channels sit 2 MHz apart and one they cannot.
        """
        tol = self.half_width_hz
        where = bands.format_mhz(self.frequency_hz)
        if self.off_chart or self.match is None:
            near = f" (nearest {self.match.label})" if self.match else ""
            return f"off-chart {where}{near}"
        if tol < 500_000:
            return f"{self.match.label} @ {where}"
        return f"{self.match.label} @ {where} +-{tol/1e6:.1f} MHz"

    def describe(self) -> str:
        wide = ""
        if self.merged > 1:
            wide = f", {self.merged} finds over {self.span_hz/1e6:.0f} MHz merged"
        return f"{self.label()} -- {self.reading.describe()}{wide}"


@dataclass
class ScanResult:
    """Everything one pass over a span found."""

    lo_hz: int
    hi_hz: int
    candidates: list[Candidate] = field(default_factory=list)
    hits: list[CoarseHit] = field(default_factory=list)
    #: Hops that cleared the coarse gate but held nothing when examined properly.
    #: The coarse pass looks at 1.6 ms and the thorough pass at a full field, so
    #: some of these are expected; they are counted rather than reported, because
    #: "something was here and it turned out to be nothing" is not a finding.
    false_positives: int = 0
    hops: int = 0
    gate_bins: int = COARSE_BIN_COUNT
    gate_peak_db: float = COARSE_PEAK_GATE_DB
    elapsed_s: float = 0.0
    noise_floor_db: float = -120.0
    stopped_early: bool = False
    error: str = ""

    @property
    def occupied(self) -> list[Candidate]:
        return [c for c in self.candidates if c.reading.occupied]

    @property
    def decodable(self) -> list[Candidate]:
        return [c for c in self.candidates if c.decodable]

    def best(self) -> Candidate | None:
        """The channel to listen to: decodable video first, then strength.

        Video is preferred over a bare carrier of the same strength because a
        bare carrier is usually a VTX that is powered but not streaming, or an
        unrelated narrowband interferer -- neither is what the operator wants to
        see. Within a kind, strength decides.
        """
        pool = self.decodable or self.occupied
        if not pool:
            return None
        return max(pool, key=lambda c: c.strength)

    @property
    def peak_noise_db(self) -> float:
        """Largest peak-over-floor seen on a hop that found nothing.

        This is the useful "how much margin is left" number: on a quiet band it
        says how far the gate is from firing, and a band sitting at 10 dB here is
        one interferer away from an alert.
        """
        quiet = [h.peak_db for h in self.hits if h.bins_above < self.gate_bins]
        return max(quiet) if quiet else 0.0

    def describe(self) -> str:
        span = f"{bands.format_mhz(self.lo_hz)}..{bands.format_mhz(self.hi_hz)}"
        head = f"scanned {span} in {self.hops} hops, {self.elapsed_s:.1f} s"
        # Report the per-hop cost. On hardware the retune, not the 1.6 ms of
        # samples, is the whole sweep, so "how long did a sweep take" is not
        # actionable without "how long did one hop take". A sweep that is slow
        # for a reason other than retuning should look different here.
        if self.hops > 1:
            head += f", {self.elapsed_s / self.hops * 1000.0:.0f} ms/hop"
        if self.error:
            return f"{head}: {self.error}"
        tail = (
            f" ({self.false_positives} hop(s) cleared the gate but held noise)"
            if self.false_positives else ""
        )
        if not self.candidates:
            return (
                f"{head}: nothing above the noise floor{tail} "
                f"(loudest quiet hop peaked {self.peak_noise_db:.1f} dB over it, "
                f"gate {self.gate_peak_db:.0f} dB)"
            )
        best = self.best()
        got = f" -- best {best.describe()}" if best else ""
        return f"{head}: {len(self.candidates)} candidate(s){got}{tail}"


class BandScanner:
    """Two-pass band search plus auto channel selection over a live source."""

    def __init__(
        self,
        source: IQSource,
        coarse_samples: int = COARSE_SAMPLES,
        thorough_samples: int = THOROUGH_SAMPLES,
        hop_frac: float = 0.9,
        settle_timeout: float = 2.0,
    ) -> None:
        self.source = source
        self.coarse_samples = int(coarse_samples)
        self.thorough_samples = int(thorough_samples)
        # 0.9 of the sample rate per hop, so consecutive hops overlap rather than
        # abut. Chart channels are 19-37 MHz apart, so a 9 MHz hop still resolves
        # them comfortably -- and overlapping means a signal that straddles a hop
        # boundary is seen whole by at least one of the two hops.
        self.hop_hz = max(100_000, int(source.sample_rate * hop_frac))
        self.settle_timeout = float(settle_timeout)
        self._stop = threading.Event()

    # -- control -----------------------------------------------------------

    def cancel(self) -> None:
        self._stop.set()

    def reset(self) -> None:
        self._stop.clear()

    def __enter__(self) -> "BandScanner":
        self.reset()
        return self

    def __exit__(self, *exc: object) -> None:
        self.cancel()

    # -- searching ---------------------------------------------------------

    def hop_centres(self, lo_hz: int, hi_hz: int) -> list[int]:
        """Tuned centre for each coarse hop across ``[lo, hi]``.

        Centres are placed so the last hop still covers ``hi``: a naive
        ``range(lo, hi, hop)`` leaves the top of the band unseen, which is where
        F8/F9 and the RaceBand F-lite channels live.
        """
        step = self.hop_hz
        span = int(hi_hz) - int(lo_hz)
        if span <= 0:
            return []
        half = step // 2
        first = int(lo_hz) + half
        last = int(hi_hz) - half
        if last < first:                       # span narrower than one hop
            return [(int(lo_hz) + int(hi_hz)) // 2]
        out = list(range(first, last + 1, step))
        if out[-1] != last:
            out.append(last)
        return out

    def _coarse_hit(self, hop_hz: int, iq: np.ndarray) -> CoarseHit:
        """Classify one hop's spectrum: occupied, or just noise?

        Counts spectrum bins sitting more than :data:`COARSE_BIN_GATE_DB` above
        the median, and keeps peak-over-floor as a second opinion. Both are
        measured against the *median* rather than a mean, because the median is
        the value most bins report and a VTX occupying the whole span cannot
        move it.

        Both statistics are taken over the bins outside the receiver's own DC
        leakage (:func:`dsp.away_from_dc`). This is not a refinement, it is the
        difference between the search working on hardware and not: the HackRF's
        LO leak is 52 dB at the centre of every window, so ``peak_db`` would
        clear the bare-carrier gate at all 34 hops and every sweep would report
        the whole band occupied.
        """
        if iq.size < COARSE_NFFT * 2:
            return CoarseHit(hop_hz, 0, 0.0, 0.0)
        freqs, psd_db = dsp.psd(
            iq, self.source.sample_rate, nfft=COARSE_NFFT, averages=COARSE_AVERAGES
        )
        if freqs.size == 0:
            return CoarseHit(hop_hz, 0, 0.0, 0.0)
        keep = dsp.away_from_dc(freqs)
        if not np.any(keep):
            keep = np.ones_like(freqs, dtype=bool)
        floor = float(np.median(psd_db[keep]))
        peak_bin = int(np.argmax(psd_db[keep]))
        peak = float(psd_db[keep][peak_bin])
        above = int(np.count_nonzero(psd_db[keep] > floor + COARSE_BIN_GATE_DB))
        return CoarseHit(
            hop_hz=hop_hz,
            bins_above=above,
            peak_db=peak - floor,
            peak_offset_hz=float(freqs[keep][peak_bin]),
            floor_db=floor,
        )

    def _target_for(self, hit: CoarseHit) -> tuple[int, int, bool, bands.Match | None]:
        """Where to listen for one occupied hop, and how sure we are.

        A hop is a coarse measurement: it says the transmitter is within
        ``hop/2`` of the hop centre, and nothing more. Chart channels are at
        least 19 MHz apart, so if a channel falls inside that window the channel
        *is* the best available answer and is what the operator would tune to.

        If no channel falls inside the window, the transmitter is provably
        further from the chart than this sweep can resolve, and the hop centre
        is reported as off-chart rather than being forced onto a channel number
        that is not where the signal is.
        """
        centre = hit.hop_hz
        half = self.hop_hz // 2
        m = bands.match_frequency(centre)
        if m is not None and abs(centre - m.frequency_hz) <= half:
            return m.frequency_hz, self.hop_hz, False, m
        return centre, self.hop_hz, True, m

    def scan(
        self,
        lo_hz: int | None = None,
        hi_hz: int | None = None,
        region: bands.Region = bands.Region.US,
        progress: Callable[[str, float], None] | None = None,
    ) -> ScanResult:
        """Sweep ``[lo, hi]`` (default: the region's whole band) and rank it."""
        if lo_hz is None or hi_hz is None:
            d_lo, d_hi = bands.default_scan_span(region)
            lo_hz = d_lo if lo_hz is None else lo_hz
            hi_hz = d_hi if hi_hz is None else hi_hz
        lo_hz, hi_hz = int(lo_hz), int(hi_hz)

        centres = self.hop_centres(lo_hz, hi_hz)
        res = ScanResult(lo_hz=lo_hz, hi_hz=hi_hz, hops=len(centres))
        if not centres:
            res.error = "empty span"
            return res
        if not self.source.retunable:
            # A recording cannot be swept. Assess what it holds, once, and say so
            # rather than reporting the same signal at every hop.
            res.error = "source cannot retune; assessed the recording in place"
            res.hops = 1
            if self.source.frequency_hz:
                cand = self.assess_frequency(self.source.frequency_hz, snap=False)
                if cand is not None:
                    res.candidates.append(cand)
                    res.noise_floor_db = cand.reading.noise_db
                    res.hits.append(CoarseHit(cand.frequency_hz, 0,
                                              cand.reading.peak_to_median_db, 0.0))
            else:
                res.error += "; recording has no recorded frequency, so it has " \
                              "no band to name"
            return res

        t_start = time.monotonic()

        # -- pass 1: coarse -------------------------------------------------
        total = len(centres)
        for i, centre in enumerate(centres):
            if self._stop.is_set():
                res.stopped_early = True
                break
            iq = self.source.retune_settled(
                centre, self.coarse_samples, self.settle_timeout
            )
            res.hits.append(self._coarse_hit(centre, iq))
            if progress is not None:
                progress(f"coarse {i + 1}/{total}", (i + 1) / total)

        floors = [h.floor_db for h in res.hits]
        if floors:
            res.noise_floor_db = float(np.median(floors))

        # -- pass 2: thorough, only where pass 1 found something ------------
        targets = self._refine_targets(res.hits)
        verified: list[Candidate] = []
        for i, (freq, unc, off_chart, match, hit) in enumerate(targets):
            if self._stop.is_set():
                res.stopped_early = True
                break
            iq = self.source.retune_settled(
                freq, self.thorough_samples, self.settle_timeout
            )
            if iq.size < 4096:
                continue
            cand = Candidate(
                frequency_hz=freq,
                reading=dsp.assess_channel(iq, self.source.sample_rate),
                match=match,
                uncertainty_hz=unc,
                coarse=hit,
                off_chart=off_chart,
            )
            # The coarse pass sees 1.6 ms; the thorough pass sees a whole field
            # and tries to lock a sync train on it. A hop that looked occupied
            # and then holds nothing is a gate false positive, not a channel.
            # Reporting it would put a channel on the alert list that does not
            # exist, which is the one failure an operator cannot forgive.
            if cand.reading.occupied:
                verified.append(cand)
            else:
                res.false_positives += 1
            if progress is not None:
                progress(
                    f"verify {i + 1}/{len(targets)} {bands.format_mhz(freq)}",
                    (i + 1) / len(targets),
                )

        res.elapsed_s = time.monotonic() - t_start
        res.candidates = self._merge(verified, VTX_WIDTH_HZ)
        res.candidates.sort(key=lambda c: -c.strength)
        return res

    @staticmethod
    def _merge(
        cands: list[Candidate], width_hz: int = VTX_WIDTH_HZ
    ) -> list[Candidate]:
        """Collapse neighbouring findings into one transmitter.

        A VTX is about 15 MHz wide and a hop grid is 9 MHz, so the same drone is
        normally found by two or three consecutive hops -- and on a fine grid by
        eight. Every one of those is a *correct* observation and a *useless*
        alert, because the operator has to decide whether they are looking at one
        transmitter or two, and the only way to know is to compare frequencies
        by eye.

        Grouping is by frequency proximity, single-link, so two genuinely
        different transmitters on the same band stay separate: only findings
        within ``width_hz`` of each other join, and a group is exactly as wide as
        the signal that produced it. The strongest member represents the group,
        and the group's width is reported, which turns a nuisance into a real
        measurement -- a 15 MHz-wide finding is a video downlink, a 2 MHz one is
        a beacon or a Wi-Fi interferer.
        """
        if len(cands) < 2:
            return list(cands)
        ordered = sorted(cands, key=lambda c: c.frequency_hz)
        groups: list[list[Candidate]] = [[ordered[0]]]
        for c in ordered[1:]:
            if c.frequency_hz - groups[-1][-1].frequency_hz <= width_hz:
                groups[-1].append(c)
            else:
                groups.append([c])
        out: list[Candidate] = []
        for g in groups:
            best = max(g, key=lambda c: c.strength)
            if len(g) > 1:
                best.span_hz = g[-1].frequency_hz - g[0].frequency_hz
                best.merged = len(g)
            out.append(best)
        return out

    def _refine_targets(
        self, hits: Iterable[CoarseHit]
    ) -> list[tuple[int, int, bool, bands.Match | None, CoarseHit]]:
        """Turn occupied hops into a deduplicated list of frequencies to verify.

        Neighbouring hops can both clear the gate on one VTX -- a 15 MHz signal
        is wider than a 9 MHz hop -- and would otherwise produce two candidates
        that are really the same transmitter. Keying on the target frequency
        collapses those, keeping the strongest evidence.
        """
        picks: dict[int, tuple[int, bool, bands.Match | None, CoarseHit, float]] = {}
        for hit in hits:
            if not hit.occupied:
                continue
            freq, unc, off, m = self._target_for(hit)
            prev = picks.get(freq)
            if prev is None or hit.score > prev[4]:
                picks[freq] = (unc, off, m, hit, hit.score)
        return [(f, p[0], p[1], p[2], p[3]) for f, p in sorted(picks.items())]

    def scan_band(
        self, band: bands.Band, progress: Callable[[str, float], None] | None = None
    ) -> ScanResult:
        """Sweep exactly one band plan, e.g. Raceband or FatShark F."""
        lo, hi = band.frequency_span()
        return self.scan(lo_hz=lo, hi_hz=hi, progress=progress)

    # -- single-frequency assessment ---------------------------------------

    def assess_frequency(
        self, frequency_hz: int, snap: bool = True
    ) -> Candidate | None:
        """Tune to one frequency, capture a field, and classify it.

        Used by the manual channel buttons and by the initial channel check, so
        that a hand-picked channel is verified the same way the search verifies
        its own choices. With ``snap=False`` the frequency is used verbatim, which
        is what a recording needs: the operator already knows where it was taken.
        """
        freq = int(frequency_hz)
        match = bands.match_frequency(freq)
        if snap and match is not None:
            freq = match.frequency_hz
        iq = self.source.retune_settled(
            freq, self.thorough_samples, self.settle_timeout
        )
        if iq.size < 4096:
            return None
        return Candidate(
            frequency_hz=freq,
            reading=dsp.assess_channel(iq, self.source.sample_rate),
            match=bands.match_frequency(freq) if snap is False else match,
            uncertainty_hz=0,      # an explicit request: no hop grid involved
        )

    def suggest(self, result: ScanResult, band: bands.Band | None = None) -> Candidate | None:
        """Auto channel select: the strongest decodable channel, else nothing.

        Returning ``None`` when only noise is present is the point. A search that
        always names a channel teaches the operator to ignore it.

        ``band`` scopes the choice to one band plan -- what the operator gets
        after clicking a row on the chart. Scoping is strict on purpose: someone
        who picked Raceband to search their own quad does not want to be handed
        an F4 from a different plan because it scored a decibel higher. If the
        chosen band genuinely has nothing on it, the answer is still nothing.
        """
        if band is None:
            return result.best()
        for cand in result.decodable:
            if cand.band == band.key:
                return cand
        return None
