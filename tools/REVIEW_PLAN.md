# Review remediation plan — 14 findings

All 14 findings verified by direct source inspection against `b63cec6`. None was
invalid. Finding 1 additionally proven empirically (see "IQ format" below).

Fixing finding 1 turned out to expose two further defects that the review could
not have found, because neither is visible without real samples: the app had
never decoded a real signal, and the status line reported a sync score beside a
"no lock" that made the failure look like a healthy noisy lock. See "Beyond the
14" below.

## IQ format — settled empirically, not assumed

`tools/probe_iq_format.py` + `tools/probe_iq_layout.py` against real 5802 MHz
captures from the same Mayhem toolchain this project drives:

- 100.0% of bytes fall outside [64,192); 0.0% inside [108,147).
  Unsigned-offset data would be almost entirely in that middle band.
- No stride structure: all four byte positions show identical statistics
  (75/72/75/72 distinct values). A float32 file would have the exponent byte
  at stride 3 stand out sharply. It does not.
- Even/odd parity means identical (132.1 / 126.2), the I,Q,I,Q signature.
- Read as int8: I mean -2.83, Q mean +0.17, sd ~20/128, amplitude median 29
  with 74% sample-to-sample swing — a zero-centred constant-envelope FM carrier.

Conclusion: **signed int8 interleaved**. The existing `_u8_to_iq` sign-flips the
waveform and adds a full-scale DC offset on the hardware path.

### The detector cannot always decide, and says so

`detect_iq_encoding` returns `EncodingVerdict(encoding | None, share, reason)`.
The two encodings differ by a fold about 128, so they are separable only while
the signal is *quiet*: a loud recording reaches the ends of the byte axis either
way, and a wrapped loud signal is just a loud signal again. So the verdict is
`None` above 90%/below 10% of extreme bytes and undecidable in between, and
`FileSource` falls back to signed — the format the radio produces — while saying
in `describe()` that the data did not settle it. A file loaded with an explicit
`encoding=` skips detection entirely.

The fixtures are generated at amplitude 0.20 to match the real capture
(mean |x| 0.22). A unit-magnitude fixture sits at 33.5% extreme bytes and is
*correctly* reported undecidable; testing detection against a fixture whose
amplitude is chosen to defeat it would test nothing.

### A second, smaller bug in the same function

The offset-binary branch scaled by 1/127.5 and then subtracted 1.0, assuming
128/127.5 == 1. It is 1.0039. So byte 128 — the exact centre of the encoding,
i.e. a zero sample — decoded to +0.0039, and every offset-binary sample carried
a 0.4%-of-full-scale DC offset. The bias is now removed *before* scaling, which
makes the two encodings of one sample bit-identical rather than 1 LSB apart.

## Priority order

| # | Finding | File | Status |
|---|---------|------|--------|
| 1 | signed IQ read as unsigned | sdr.py | done |
| 2 | fallback runs before reclaim | fastscan.py | done |
| 3 | retune timeout returns stale samples | sdr.py | done |
| 10 | drain_iq re-returns consumed samples | sdr.py | done |
| 4 | gain/amp never reach hardware | sdr.py, ui.py | done |
| 5 | stale video shown as locked | video.py, ui.py | done |
| 6 | quiet manual check raises FOUND | ui.py, alerts.py | done |
| 7 | cancelled scan emits false LOST | alerts.py, ui.py | done |
| 8 | scoped auto-select rejects sweep results | scan.py | done |
| 9 | shared frequencies rejected | bands.py, scan.py | done |
| 11 | invert checkbox sets brightness | ui.py | done |
| 12 | failed tune corrupts state | sdr.py, ui.py | done |
| 13 | recording can be relabelled | sdr.py, ui.py | done |
| 14 | scan leaves receiver off-channel | scan.py, ui.py | done |

## Beyond the 14 — two defects only real samples could reveal

### A. The decoder had never decoded a real signal

`Rasteriser.select_run` required `want` *consecutive detected* scanlines. On the
real 5802 MHz capture the sync detector fires on 82.7% of lines, but the losses
are not scattered: the gap histogram is 399 gaps of 1 line, 9 of 2 lines, then 8
chunks of 5, 6, 6, 8, 8, 8, 17 and 20 lines. The longest clean run was 112
lines against the 240 a field needs, so **not one frame was ever produced from a
real VTX**. The class docstring's assumption of "roughly one dropped line per
hundred" was wrong by a factor of 18, and in the wrong *shape*: dropouts, not
jitter.

Fixed by bridging dropouts of up to `max_fill` lines, which is 20 by measurement:
the largest observed dropout is 20 and field blanking is 22.5 (NTSC) / 24.5
(PAL), so a separating integer sits between them and the "never interpolate
across a discontinuity" property survives. Missing rows are placed on a grid
fitted to the surrounding detections — the *samples* there are real video, only
the line positions are reconstructed — and the count is reported on the frame as
`lines_from_grid` rather than absorbed.

Swept on the real capture, mean adjacent-row correlation: max_fill 2 and below
produce no frame at all; 8 gives +0.8954, 12 +0.8984, 20 +0.9020, 32 +0.9032.
Flat past 20, so bridging blanking buys nothing and is not done.

### B. The status line reported a lock score for a decoder that had no lock

`no lock  sync 90%` is what the app displayed, continuously, on this capture.
The sync detector is right about the sync train and the rasteriser was refusing
to assemble a field from it, and the UI presented the first number without the
second. The sync figure now appears only when there is an actual lock to justify
it, and `lines_from_grid` is shown beside it.

### What this corrects about earlier conclusions

The previous session's "5.8 GHz is empty" was read off the *misinterpreted* IQ.
The band is not empty: the capture is a constant-envelope FM carrier at ~6.4 MHz
peak deviation with an NTSC line rate within 0.05% of 15,734.264 Hz. Likewise
`check_fastscan`'s 34-hop/2006 ms-per-hop figures describe energy detection over
one pre-scan capture, not a sweep. The standalone `check_sweep` numbers stand.

### Now verified, and worth stating precisely

`tools/probe_real_decode.py` and case 1d of `tools/check_review_fixes.py` decode
the real capture end to end: 77 frames over 2 s, NTSC 160x240, line rate
15,724.7 Hz (0.061% off), adjacent-row correlation +0.906 against a white-noise
null of -0.002. That is real FPV video from a real HackRF recording, with no
simulator in the path.

This is a **decode**, not a live-hardware test: no radio was plugged in, and no
drone transmitted during this work. What is unproven is the live path
(`hackrf_transfer` handover, USB pacing, gain settings against a real VTX) and
whether the picture is watchable on screen rather than merely coherent.

## Closely related problems fixed alongside

- `DecodeWorker` swallows decode exceptions with no logging → counted, first few
  recorded, surfaced in the status line (video.py)
- decoder reset races the decode thread → `_decoder_lock` serialises `reset()`
  against `push_iq` (video.py)
- `closeEvent` ignores the 3 s scan wait result → 10 s, and checked (ui.py)
- `SimSource.describe` defined twice, later silently wins → removed (sdr.py)
- `Beeper.pattern` short-circuit skips the first beep → fixed (alerts.py)
- `tune()` reported success for subclasses that short-circuit `_apply_tune` →
  `_applied_hz` tracks what is actually receiving, in the base class (sdr.py)
- `FileSource.tune` stored a new frequency it could not apply → refuses (sdr.py)
- comments promise behaviour the code does not deliver → corrected throughout

## Fixtures and gates

`tools/make_iq_fixtures.py` writes `tools/fixtures/`:

| file | bytes | what it is |
|------|-------|------------|
| `iq_reference.f32` | 1,526,400 | complex64 ground truth |
| `iq_signed_int8.u8` | 381,600 | the same samples, two's complement |
| `iq_unsigned_offset.u8.u8` | 381,600 | the same samples, offset-binary |
| `real_5802_signed.u8` | 2,000,000 | verbatim excerpt of a real recording |

The 2 MB excerpt is committed deliberately. It is the only way CI can prove the
decoder still works on real samples; the 38 MB original stays out of git under
`/samples/`.

`tools/check_review_fixes.py` is 24 cases covering all 14 findings plus the two
above, driving real widget signals and ordering-recording mocks. Registered in
`.github/workflows/checks.yml` as a gating check alongside the 8 existing ones.
