"""VTX frequency charts, regional legality, and band/channel lookup.

Three jobs:

1. **Charts** -- the common 5.8 GHz band plans (A/B/E/F/R/L/U/H/O/D) as plain
   data, so the UI can draw a band chart without hardcoding anything.
2. **Legality** -- which parts of the spectrum may be transmitted in a given
   region. This is advisory information for a receive-and-display tool, not
   legal advice, and it is deliberately conservative.
3. **Lookup / snapping** -- turn a measured frequency into a band+channel
   label, and snap an off-table frequency to the nearest chart channel.

Why snapping matters here
-------------------------
A great many VTXs, and most of them in the field, are *not* sitting exactly on
a chart frequency -- firmware revisions, temperature drift and the "bind
frequency" scratch pad all move the carrier. The reference unit for this
project transmits at 5802 MHz, which is not any channel on any common chart.
So the scanner does a fine spectral sweep rather than only visiting chart
frequencies, and then labels what it found as both the measured frequency and
the nearest chart channel plus the offset. Presenting only the chart channel
would be actively misleading.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Region(str, Enum):
    US = "us"
    CE = "ce"
    AU = "au"
    JP = "jp"


@dataclass(frozen=True)
class RegionPlan:
    """Permitted transmit span for analogue FPV video, in Hz."""

    region: Region
    label: str
    low_hz: int
    high_hz: int
    note: str


# Analogue FPV video occupies roughly 20 MHz of spectrum around the carrier, so
# these limits are stated for the *carrier*.
REGIONS: dict[Region, RegionPlan] = {
    Region.US: RegionPlan(
        Region.US, "United States (5.65-5.95 GHz)", 5_650_000_000, 5_950_000_000,
        "Amateur 5.65-5.95 GHz allocation. Transmit only where you are licensed "
        "to operate, and never at power levels that cause interference.",
    ),
    Region.CE: RegionPlan(
        Region.CE, "Europe / CE (5725-5875 MHz)", 5_725_000_000, 5_875_000_000,
        "ISM band shared with other services. 25 mW is the commonly cited "
        "ceiling for handheld/FPV use in much of the EU.",
    ),
    Region.AU: RegionPlan(
        Region.AU, "Australia (5730-5875 MHz)", 5_730_000_000, 5_875_000_000,
        "Follow ACMA low-power short-range device limits.",
    ),
    Region.JP: RegionPlan(
        Region.JP, "Japan (5725-5875 MHz)", 5_725_000_000, 5_875_000_000,
        "Low-power short-range device band; check current MIC guidance.",
    ),
}


@dataclass(frozen=True)
class Band:
    """One VTX band plan: a name and its channel frequencies in Hz."""

    key: str
    label: str
    channels: tuple[int, ...]
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.channels)

    def channel_freq(self, index: int) -> int | None:
        """Frequency of channel ``index`` (1-based, as printed on VTXs)."""
        if 1 <= index <= self.n:
            return self.channels[index - 1]
        return None

    def frequency_span(self) -> tuple[int, int]:
        return min(self.channels), max(self.channels)


def _b(key: str, label: str, freqs_mhz: list[int], note: str = "") -> Band:
    return Band(key, label, tuple(int(round(f * 1_000_000)) for f in freqs_mhz), note)


# The widely used 5.8 GHz analogue band plans. Frequencies are in MHz in this
# table purely for readability and converted to Hz in _b().
BANDS_58: tuple[Band, ...] = (
    _b("A", "A (Boscam/IRC)", [5865, 5845, 5825, 5805, 5785, 5765, 5745, 5725],
       "Common low-cost 25 mW band."),
    _b("B", "B", [5733, 5752, 5771, 5790, 5809, 5828, 5847, 5866],
       "Boscam B, 19 MHz channel spacing."),
    _b("E", "E (Boscam)", [5705, 5685, 5665, 5645, 5885, 5905, 5925, 5945],
       "200 mW band; note the split around the middle."),
    _b("F", "F (ImmersionRC)", [5740, 5760, 5780, 5800, 5820, 5840, 5860, 5880],
       "FatShark/ImmersionRC F, 20 MHz spacing."),
    _b("R", "R (Raceband)", [5658, 5695, 5732, 5769, 5806, 5843, 5880, 5917],
       "Raceband, 37 MHz spacing. R5 is 5806 MHz."),
    _b("L", "L (B DVR / Loco)", [5867, 5847, 5827, 5807, 5787, 5767, 5747, 5727],
       "Loco/B DVR band."),
    _b("U", "U (Unicity)", [5745, 5765, 5785, 5805, 5825, 5845, 5865, 5885]),
    _b("H", "H (Himax)", [5745, 5765, 5785, 5805, 5825, 5845, 5865, 5885],
       "Himax 2-4 W band."),
    _b("O", "O (Octagon/Pro)", [5658, 5695, 5732, 5769, 5806, 5843, 5880, 5917],
       "Octagon/Pro band; same spacing as Raceband."),
    _b("D", "D (Droneband)", [5865, 5845, 5825, 5805, 5785, 5765, 5745, 5725],
       "Droneband."),
)

BANDS_BY_KEY: dict[str, Band] = {b.key: b for b in BANDS_58}
ALL_BANDS: tuple[Band, ...] = BANDS_58


@dataclass(frozen=True)
class Match:
    """The nearest chart channel to a measured frequency."""

    band: Band
    channel: int
    frequency_hz: int
    offset_hz: int
    exact: bool

    @property
    def label(self) -> str:
        return f"{self.band.key}{self.channel}"

    @property
    def is_exact(self) -> bool:
        """True when the carrier sits on the chart frequency.

        Within 250 kHz counts as on-frequency: that is well inside the sample
        rate / 4000 of frequency resolution, so anything closer cannot be
        distinguished from being exactly on the channel.
        """
        return self.exact

    def describe(self) -> str:
        if self.exact:
            return f"{self.label} @ {self.frequency_hz/1e6:.0f} MHz"
        off = self.offset_hz / 1e6
        sign = "+" if off >= 0 else "-"
        return (
            f"{self.label} @ {self.frequency_hz/1e6:.0f} MHz "
            f"({sign}{abs(off):.0f} MHz from measured)"
        )


# How close a measured carrier must be to a chart channel to be called exact.
EXACT_TOL_HZ = 250_000


def match_frequency(freq_hz: float, bands: tuple[Band, ...] = BANDS_58) -> Match | None:
    """Snap ``freq_hz`` to the nearest channel across all ``bands``."""
    best: Match | None = None
    for band in bands:
        for i, f in enumerate(band.channels, start=1):
            off = int(round(freq_hz - f))
            if best is None or abs(off) < abs(best.offset_hz):
                best = Match(band, i, f, off, abs(off) <= EXACT_TOL_HZ)
    return best


def is_legal(freq_hz: float, region: Region = Region.US) -> bool:
    """True if ``freq_hz`` is inside ``region``'s permitted carrier span."""
    plan = REGIONS[Region(region)]
    return plan.low_hz <= freq_hz <= plan.high_hz


def clamp_to_region(freq_hz: float, region: Region = Region.US) -> float:
    plan = REGIONS[Region(region)]
    return float(min(max(freq_hz, plan.low_hz), plan.high_hz))


def band_of(freq_hz: float) -> Band | None:
    """The band whose *span* contains ``freq_hz``, if any."""
    for b in ALL_BANDS:
        lo, hi = b.frequency_span()
        if lo - 3_000_000 <= freq_hz <= hi + 3_000_000:
            return b
    return None


def band_plan_frequencies(band: Band) -> list[int]:
    return list(band.channels)


def default_scan_span(region: Region = Region.US) -> tuple[int, int]:
    """Start/stop frequency for a full auto band search in ``region``."""
    plan = REGIONS[Region(region)]
    return plan.low_hz, plan.high_hz


def format_mhz(hz: float) -> str:
    return f"{hz/1e6:.3f}".rstrip("0").rstrip(".") + " MHz"
