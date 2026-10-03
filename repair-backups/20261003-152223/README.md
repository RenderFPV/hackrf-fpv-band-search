# FPV RF: band search, channel select, and alerting

Decodes analogue FPV video straight out of HackRF IQ — no capture card, no
driver — searches the 5.8 GHz band for energy above the noise floor, names what
it finds on the VTX band chart, and alerts with **band + channel + frequency**.

```
python main.py                                     # sim source, works with no hardware
python main.py --mode hackrf --frequency 5802      # a real HackRF
python main.py --mode file --file capture.u8 --capture-frequency 5802
python main.py --check                             # verify a source and exit
```

MIT licensed. See [LICENSE](LICENSE).

## Requirements

Python 3.12 or newer, and:

```
pip install -r requirements.txt
```

That is three packages — numpy, scipy, PySide6 — and it is deliberately all the
**application** needs. `fpv_rf` never imports Pillow: the picture downscale is
`scipy.ndimage.zoom` rather than a resize, so the packaged `.exe` does not carry
an imaging library it would not otherwise use.

To run the checks as well, which need two more:

```
pip install -r requirements-dev.txt
```

Five scripts under `tools/` use Pillow, and only to write a PNG of a decoded
frame so a human can look at it. Nothing in the decode path touches it.
`check_workflow.py` uses PyYAML to parse the CI workflow before pushing it.
Both are development-only, and neither reaches the packaged `.exe`.

## The .exe app

**You do not need to build this.** Download it from the
[v1.0.1 release](https://github.com/RenderFPV/hackrf-fpv-band-search/releases/tag/v1.0.1)
- `FPV-RF-v1.0.1-windows-x64.zip`, 88 MB - extract it, and double-click
`FPV-RF.exe`. No Python, no pip, no drivers. The first launch takes about
fifteen seconds while Windows Defender scans the freshly-extracted files; the
second takes about one. Windows will warn you the app is unsigned; that is the
missing signature, not the program, and *More info → Run anyway* is the correct
response.

To build it yourself instead:

```
python build_exe.py
```

produces **`dist\FPV-RF\FPV-RF.exe`** — double-click it, no console, no Python
needed. The name has no spaces in it on purpose: the first build shipped to
`dist\FPV RF\FPV RF.exe`, and a path with a space in both the folder and the
file has to be quoted everywhere and is awkward to type into a Run box. The
build output should be the one thing in this project that is hard to misplace.

If you would rather not dig into `dist\`, double-click **`FPV-RF.cmd`** in the
project root. It starts the app where it is and forwards any arguments, so
`FPV-RF.cmd --check` works too. The exe cannot simply be moved up there — it
needs its own folder beside it, for the reason below — and copying 212 MB to
make the shortcut real would only leave the same app in two places.
`build_exe.py` writes that file on every build, so it cannot go stale.

`hackrf_transfer.exe` is copied in beside it, and the app looks for it there
*before* `PATH`, so the copy shipped with the app is the one that runs rather
than some other version found elsewhere on the machine.

The packaged app defaults to `--mode hackrf`, because a double-clicked app
should use the radio plugged into it rather than opening a convincing fake
picture. `--mode sim` still selects the simulator.

It is a **folder, not a single file**, on purpose. A one-file build unpacks its
whole payload to a temp directory on every launch and deletes it afterwards;
with Qt and SciPy that is several hundred megabytes copied to disk each time,
turning a one-second start into a thirty-second one. The folder form starts in
about a second and is a normal application directory you can copy or delete.

Three things a windowed build needs that running from source does not:

- **`find_hackrf_transfer` is frozen-aware.** `__file__` points inside the
  bundle in a packaged build, so the app resolves paths against the directory
  holding the executable.
- **Everything is logged to `fpv-rf.log`** beside the exe. A GUI-subsystem build
  has no console, so `sys.stdout` is `None` and every `print` would vanish —
  which is exactly when diagnostics matter, because a frozen build that
  misbehaves otherwise gives you nothing to look at. stdout and stderr are
  replaced with a tee, so no existing call site changed.
- **Results you would have read off the console come back as a message box.**
  `--check` on the packaged app pops up its report instead of exiting silently,
  which is otherwise indistinguishable from crashing.

### Verifying the packaged build

```
"dist\FPV-RF\FPV-RF.exe" --check          # the radio, in a dialog, then exit
"dist\FPV-RF\FPV-RF.exe" --selftest       # build the real window and test it
```

`--selftest` exists because a packaged build fails in ways a source-tree run
cannot: a missing Qt platform plugin, a data file the packager left out. The
symptom in both cases is a window that never appears, and there is no debugger
attached to an `.exe`. `--selftest` runs the real window offscreen *inside the
frozen build* — the same 16 assertions `check_ui.py` makes — so "the exe is
broken" and "the app is broken" become distinguishable. `build_exe.py` runs it
automatically and will not report success without it passing.

`build_exe.py` runs the self test **first**, before anything opens the radio,
and that ordering is load-bearing rather than cosmetic. The self test measures
sustained publish throughput, so it fails if something else is competing for the
CPU, and tearing a HackRF down immediately before it is exactly that. The same
kind of flakiness is why `bench_sim` is a measurement and not an assertion.

The throughput gate is `0.7 * PREVIEW_HZ` — **21 fps** against the 30 Hz decode
cadence — written that way in `check_ui.py` rather than as a literal, so a change
of cadence moves the gate with it. It used to be a flat 45 fps, written when the
decoder ran at the field rate and never revised when the cadence was cut to half
of it: a 30 Hz decoder publishing at 30 Hz could not pass a 45 fps gate, so the
number was not a throughput assertion at all, it was a contradiction. It is one
now rather than merely "some frames appeared": 0.7 leaves a third of the cadence
as headroom for a loaded machine and the first iteration's catch-up, while a
decode that cannot complete an iteration inside 1/30 s publishes slower than that
forever. The check prints the cadence and the gate beside the measured rate so the
three can never be compared by mistake.

The **50–59 fps** figure recorded for the packaged build in earlier revisions
dates from when the cadence was the 60 Hz field rate and is not comparable to
today's 30 Hz decoder; at 30 Hz the equivalent statement is "close to 30 fps".
It has not been re-measured against a fresh 30 Hz build, and the packaged-build
claim is not one this repository can establish on its own. What a source-tree run
does establish is `check_ui.py`'s 16/16, which is the same assertion list
`--selftest` makes inside the frozen build. The live radio correctly reports
`noise only` when nothing is transmitting.

![panel](docs/ui_2_alert.png)

---

## What it does

1. **Reads IQ** from a HackRF at 10 MS/s, or replays a capture, or synthesises a
   transmitter.
2. **Decodes FM composite video** — demodulate, de-emphasis, find the sync train,
   rasterise lines into frames. Runs at the field rate.
3. **Searches the band** on a coarse grid, then verifies what it finds properly:
   a full field of samples, a sync-lock test, a line-rate measurement.
4. **Names the channel** from the band chart, and says how sure it is.
5. **Alerts** when energy appears above the noise floor: banner, beep, log.
6. **Tunes it for you.** A search that finds something tunes it and shows it,
   rather than making you press a second button.

## Working the chart

The chart is one row per band plan, and the three gestures on it are:

| Gesture | What it does |
|---|---|
| Click a **tick** | Tune that exact channel |
| Click anywhere else in a **row** | Make that band plan the **active band** |
| **Shift**-click a row | Sweep that band plan and nothing else |
| Click **empty space** below the rows | Clear the active band, search everything |

The active band scopes both the search and auto-select, and the scan button
shows the width it will sweep. Scoping to Raceband's eight channels is roughly a
fifth of the US region span.

Two controls sit beside the scan button, and both exist because of how the two
search engines differ:

- **passes** (1, 4 or 10) is how many sweeps to combine, and it only applies to
  the sweep engine. One is the default, because FPV video transmits continuously
  and one pass sees it as well as ten. See *Passes* for when more is worth it,
  and for why the combine is a maximum rather than an average.
- **The engine is not a choice.** The app uses the fast sweep when it can get
  the radio and falls back to the hop walk when it cannot, and the status line
  says which one answered. The `hop` control next to the chart only applies to
  the hop walk, and its tooltip says so rather than leaving you to wonder why it
  did nothing.

Scoping is strict on purpose. If you pick Raceband and the strongest signal on
the band is an F4 from a different plan, auto-select says there is nothing
rather than handing you someone else's channel — a selector that always
produces an answer is worse than none, because you cannot tell it apart from
having found your quad.

## The amplifier

There is an **RF amp** checkbox beside the LNA gain, and it is a real bypass,
not a gain reduction: the transfer helper takes `-a 0` and the amplifier is off.
That is a different thing from asking for 0 dB of LNA gain, which leaves the LNA
powered and merely stops it amplifying. The first version of this control did
the 0 dB thing and was wrong.

It is worth having as its own control because of what it diagnoses. Amplified
noise floor looks exactly like a weak transmitter, so switch the amplifier off:
if the reading survives, the energy is really out there.

Two caveats, both in the tooltip:

- **It needs a helper that has the flag.** `-a` is a Mayhem extension. The stock
  Great Scott build does not have it, and it treats an unknown option as a
  usage error, so passing it unconditionally would turn a working install into
  a dead radio. The app therefore asks the binary which flags it understands
  (`hackrf_transfer -h`) and omits `-a` when it is not there, leaving the
  checkbox disabled with an explanation rather than silently doing nothing.
- **It costs a retune**, like any change to the gains: the flags are
  command-line arguments, so they can only change when the process is replaced.
  Not a control to flick mid-sweep.

## If the radio does not appear

`hackrf_open() failed: HackRF not found (-5)` means one of three quite different
things, and the app's own report does not distinguish them:

| Meaning | Usual cause |
|---|---|
| The device is there but busy | Another program holds it — SDR#, Mayhem, a browser tab with a WebUSB app |
| The device is there, driver wrong | WinUSB or the libusb filter is not bound to `USB\VID_1D50&PID_6089` |
| **The device is not enumerating at all** | Cable, port, or power |

The third is worth checking directly, because nothing in the app's output can
tell you about it, and it is the one that looks identical to the other two from
the outside. In PowerShell:

```powershell
Get-PnpDevice -PresentOnly | Where-Object InstanceId -match 'VID_1D50'
```

An empty result means Windows cannot see the radio at all, and no amount of
retrying in software will help. Check in this order: a different port (a USB 2
port directly on the machine, not through a hub — a HackRF draws about 250 mA
and some ports will not enumerate it), a different cable (charge-only USB-C
cables are common and carry no data), and that the radio's own LED is lit.

A `Get-PnpDevice` entry that exists but reports `IsPresent=False` is a *ghost* —
a leftover from a previous session, not proof that anything is plugged in now.


## The alert

> alert with band + channel + frequency

```
03:18:00  F4 @ 5800 MHz -- decodable NTSC video, sync 100%, SNR 9.7 dB
03:18:00  R1 @ 5658 MHz -- decodable NTSC video, sync 100%, SNR 8.1 dB
03:18:41  LOST  F4 @ 5800 MHz
```

Two things make that readable rather than noise:

**A finding is identified by proximity, not by a label or a frequency grid.** A
9 MHz hop grid can legitimately report the same VTX three megahertz higher on
the next sweep, and 5803 is nearer A4 than F4. Keyed on the label or on a
rounded frequency, one drone becomes two alerts. Instead a new finding adopts the
identity of a known one within 15 MHz — the same width the scanner uses to merge
one wideband transmitter seen across several hops.

**Repeats are counted, not raised.** The first sighting alerts. A sighting
inside the cooldown bumps a counter that the banner shows ("seen 47x"). A
finding that has been absent for longer than the forget window raises a `LOST`
event, and only then is it forgotten — so a drone that lands and takes off again
is a new alert, and a drone that has gone is something you are told rather than
something you have to notice.

A cancelled or empty scan is *not* treated as evidence. `update(None)` is an
incomplete observation, and reporting every channel you did not reach as "lost"
would be a lie.

## The picture is noisy on a real capture, and that is correct

The reference decoder's own output scores +0.8241 against its own ideal, and
that number is reproducible: a bilinear resize of its `raw_frame.png` scores
exactly the same. It is a resampling artefact, not a quality score, and
chasing it wastes time.

The real 5802 MHz capture scores +0.8335 — and it is genuinely noisy. Measured
levels: sync at 3.1243 ± 0.0099, picture p5/50/95/99.5 at
−1.7504 / −1.5452 / 3.0665 / 3.1367, σ 1.4328. Only about 0.5% of picture
samples reach sync level, so there is very little margin between "sync" and
"the brightest thing in the picture", and that is a property of a 5802 MHz
capture taken with a handheld antenna, not a decoder fault.

## Frequency resolution is the hop size, and the app says so

A 10 MS/s tuner knows a transmitter to about half a hop. On the default 9 MHz
grid that is ±4.5 MHz, which is wider than the gap between several chart
channels, so the app reports both:

```
F4 @ 5800 MHz ±4.5 MHz
off-chart 5813 MHz (nearest B5)
```

Chart channels are 19–37 MHz apart, which is why snapping a hop that cleared the
gate to the nearest chart channel is sound — and why a transmitter further from
the chart than the grid can resolve is reported as *off-chart* instead of being
quietly relabelled as a channel number it is not on. The grid is a dropdown:
9 MHz (fast), 5 MHz, or 2 MHz (±1 MHz, four and a half times the work).

A **file** source knows its own frequency, so it reports `F4 @ 5802 MHz` with no
tolerance. That is the difference between replaying a recording and sweeping.

## A HackRF retune is a process restart

`hackrf_transfer` has no in-process retune, so every hop is: kill the process,
start a new one, wait for the tuner to settle. A measured 34-hop sweep of the US
span takes **12.5 s** - 367 ms per hop, nearly all of it the retune. Budget for
that if you are on this path, and for a minute or so on the 2 MHz grid. The sim
source retunes instantly, which is why the scan in the screenshot above reports
0.6 s.

This is the engine's worst property, and it is what the next section is about.
The cost is in re-opening the USB device, so it is not a cost that cleverer
Python can reduce. There is another tool that does not pay it, and the app uses
that one whenever it can.

Terminating the old process does *not* wait for it to release the USB device, so
the first spawn after a retune fails outright with `hackrf_open() failed: HackRF
not found (-5)`. That is the normal state, not a fault, and it is retried until
the device appears. Retrying it matters more than it sounds: unretried, that hop
reads an empty band because no samples arrived inside the dwell, and the sweep
silently reports the wrong frequency.

### The retry backoff was most of the sweep

That retry used to wait a flat 350 ms before trying again, and it was charged to
every hop — 30 hops across the US span is over ten seconds of deliberately
sitting still. The device is nearly always free within a few milliseconds of
the outgoing process exiting, and `_kill` has already waited for that exit, so
the backoff now starts at 15 ms and grows, rather than starting at 350 ms and
never growing. The ceiling and the attempt count are unchanged, so a radio that
really is still held gets exactly as much patience as before.

**Re-measured on hardware since.** The hop walk takes 12.5 s for the US
span, 367 ms per hop, so the backoff change bought about 0.2 s out of a figure
still dominated by the USB re-open. On a *degraded* USB bus the same walk takes
68 s, 2006 ms per hop, which is worth knowing: the cost is the device being
asked to do something, so it moves with the device's mood. The simulator reports
14 ms per hop because there is no device to re-open, and the sweep prints its
own per-hop cost so the number can be read straight off a sweep rather than
inferred:

```
scan: scanned 5650 MHz..5950 MHz in 34 hops, 0.5 s, 14 ms/hop: 1 candidate(s)
```

A sweep that is slow for some reason *other* than retuning will show a
per-hop figure that does not explain the total, which is the point of printing
it.

### And the answer: don't retune at all

Everything above assumes the scan has to re-open the device to move frequency.
The Mayhem bundle ships another tool that does not work that way:
`hackrf_sweep` keeps the radio open across a whole sweep and retunes it with a
control transfer, so a hop costs about a millisecond instead of 367.

It is used when it is available, and the difference is not marginal. The same
5650-5950 MHz span:

| engine | span | time | what it can tell you |
| --- | --- | --- | --- |
| `hackrf_sweep` | 5640-5960 MHz, 320 bins | **0.14 s** | energy, 1 MHz bins |
| hop walk | 5650-5950 MHz, 34 hops | 12.5 s | energy, and whether the signal decodes as video |

`fpv_rf/sweep.py` is the engine and `fpv_rf/fastscan.py` chooses between them. A
sweep cannot tell you a signal is decodable analogue video: it measures energy
and never looks at a line rate. So `ScanResult` records which engine answered and
the app says so, rather than letting a coarse answer be read as a precise one.
That is also why the hop walk is kept rather than replaced.

It needs one thing the app cannot assume: `libfftw3f-3.dll`, which the Mayhem
bundle does not ship, and without which the tool exits with
`STATUS_DLL_NOT_FOUND` and no output at all. `find_fftw()` looks for a
single-precision FFTW on the machine and **verifies it by reading the PE export
table**. A double-precision build has the same file name and none of the `fftwf_`
symbols, so finding a file is not the same as finding the right one. What it
finds is copied to `%LOCALAPPDATA%/FPV-RF/sweep-tool/`; the source directory is
only ever read. FFTW is GPL, which is why it is staged at runtime rather than
committed to this MIT repository.

### Why it does not always win, and what happens when it cannot

Handing the radio to `hackrf_sweep` means stopping a `hackrf_transfer` that is
mid-transfer. On a HackRF One that leaves the device refusing the next open for
a period that is not predictable: measured here from 0 s to over 30 s, not fixed
by waiting longer, and not fixed by stopping politely, because the tool ignores
Ctrl-C and has to be killed. `tools/probe_handover.py` reproduces it: five
handovers out of five fail with the raw sample stream running, three out of
three succeed without it.

So the app tries the sweep, gives it a bounded wait, and **falls back to the hop
walk when the tool cannot have the radio**. The fast path is usually fast; the
answer is always real and always arrives, and the status line says which engine
ran and why. The wait is bounded because a scan that hangs for a minute is
indistinguishable from a broken app.

A search that failed is never reported as a quiet band. An exit-0 sweep that
produced no parseable data used to report "nothing above the noise floor", which
reads as a claim about the band when in fact nothing was measured. That is the
one failure mode worth being pedantic about, since the whole value of a scan is
that its silence means something.

### Passes

The `passes` control sets how many sweeps to combine: 1, 4 or 10. One is the
default and it is the right answer for FPV, because the video link transmits
continuously, so one pass sees it as well as ten would, at 0.15 s against 0.53 s.

More passes only help against a transmitter that is *intermittent*. The combine
statistic is the **max** across passes, not the mean, and that is not a
preference: three consecutive single-pass scans of the 2.4 GHz band showed
spreads of 13, 42 and 15 dB, with one pass catching a real WiFi signal at
+42 dB that the other two missed entirely. Averaging a burst seen in one pass out
of three divides its amplitude by three and hides it.

The max-of-N runs high on pure noise, so the gate is priced for it,
`4.34 * log10(N)` dB, because otherwise sweeping ten times would raise every bin
about 4.3 dB and manufacture findings out of an empty band. That is exactly how
a scan whose job is to prove a band is quiet ends up reporting transmitters in
it.

### What a sweep counts as a transmitter

Detection runs on a **15 MHz window** rather than on raw 1 MHz bins, because
adjacent bins come from one FFT and are strongly correlated: a lone bin that
happens to land high looks as significant as its distance above the floor
suggests, and a 1 MHz "signal" is not a 25 MHz VTX. The window matches
`scan.VTX_WIDTH_HZ`, and it responds properly to a real signal whose skirts are
weaker than its centre.

The gate is 14 dB over the floor, set from the two numbers that bound it rather
than tuned until a test passed. The widest excursion a genuinely empty band has
produced on this machine is +8.8 dB, and a real transmitter measures +29 dB in
the same window, +40 dB when a burst lands in the pass being read. Anything below
about 11 dB reports phantom transmitters on a degraded bus, and a pilot cannot
tell a phantom from a real one.

`tools/check_sweep.py` checks this against hardware, using 2.4 GHz as the
known-answer band because it has traffic in it and 5.8 GHz does not. A scanner
that reports transmitters in an empty band is worse than no scanner.

The scan runs on its own thread, so the picture keeps updating while it runs.
That is not a nicety: on hardware the scan is the only thing taking seconds, and
freezing the window for the duration means the operator watches a dead panel
exactly when they are waiting to find out whether their drone is up.

## The HackRF's own DC leakage, and why it is the one thing that had to be measured

With nothing transmitting anywhere in the band, the live HackRF reports a
**52 dB narrowband carrier at +0.00 MHz offset**. At every tuned frequency. It
is the receiver's own local oscillator, and the giveaway is that it does not
move: a real signal holds its frequency while the tuner moves, so it appears at
a different offset in each window, whereas this is dead centre every time.
Measured detail — bin 0 at +52 dB over the floor, bins ±1 at +46 dB, and
nothing else in the entire 10 MHz span more than 3 dB up.

Left in, this is worse than noise, because it is loud, constant, and present at
*every* hop. The coarse pass's secondary test is a bare-carrier peak gate at
12 dB, so all 34 hops would clear it: one sweep would report the whole band
occupied and raise 34 alerts for one receiver.

So every occupancy statistic — the classifier's peak and floor, and the coarse
pass's peak and bin count — is taken over the bins outside a 25 kHz guard band
around DC (`dsp.away_from_dc`). Ignoring it costs nothing real: an FPV downlink
is ~15 MHz wide, so it covers the centre of the window hundreds of times over
and cannot be missed by a 50 kHz hole. The only thing that could hide there is
a carrier engineered to land within 25 kHz of whatever frequency the tuner
happened to be sitting on.

`python tools\check_hardware_scan.py` is the check that matters here, and it is
the one the simulator cannot stand in for — the sim has no LO leak, so its
0-false-positive result said nothing about the real radio. On hardware, with the
band empty:

```
   5650.0          0       2.3      -2.81  quiet
   ...
   5803.0          0       3.8      -4.51  quiet
   ...
   5947.0          0       3.4      -0.82  quiet
34 hops, 0 read as occupied

scanned 5650 MHz..5950 MHz in 34 hops, 12.7 s: nothing above the noise floor
  (loudest quiet hop peaked 5.2 dB over it, gate 12 dB)
candidates: 0, false positives: 0
```

The block above is the hop walk, captured as it ran. The sweep engine has its
own equivalent, and it is the one that matters now, because 34 hops in 12.7 s is
not the number an operator waits for:

```
swept 5640 MHz..5960 MHz in 320 bins, 0.14 s: nothing above the noise floor
  (loudest quiet bin peaked 0.0 dB over it, gate 12 dB)
```
```

## What the sim is and is not

`SimSource` produces a real FM composite video signal: sync pulses, blanking,
gamma, a per-line ramp, at the right line rate. It sustains 1.00× real time on
one core, so a decoder that cannot keep up with real time is visibly starved
rather than quietly working on a backlog.

It is **deliberately narrowband** — hundreds of kHz of deviation, not the
15 MHz Carson width of a real VTX. That is the one thing it does not model, and
it matters in exactly one place: how far from a transmitter a tuner can still
see it. On the sim, signal exists within ±5 MHz; on hardware, within the full
15 MHz. So a merged finding on the sim spans 10 MHz where hardware would span
15. Everything else — sync detection, rasterising, frame rate, level mapping — is
exercised honestly.

Its quiet-band noise is flat across the span, generated as a first *difference*
of white noise. That is not decoration: a random walk in the deviation has a
Lorentzian spectrum with a huge hump at DC, and a band search finds that hump
every time. Making it flat dropped the coarse gate's false-positive rate to
**0 in 200 independent blocks**.

## Occupancy is decided by sync, not by SNR

Video SNR as measured in the 10 MHz span is a bad test, because a 6.5 MHz
deviation FM signal occupies ~15 MHz — wider than the window it is being
measured in. What separates a video downlink from noise is not how loud it is but
whether there is a **sync train in it**: a few thousand regularly spaced pulses,
lockable, at a line rate that matches a standard.

`dsp.assess_channel` therefore classifies by trying to lock, and reports
`video` / `carrier` / `noise`. Bare carriers are a separate class because a
powered-but-not-streaming VTX and a narrowband interferer are both real and
neither is what you want to watch.

Sync detection is the **97.5th percentile of the signed waveform, unsmoothed**.
This is load-bearing: a 10-tap moving average before the threshold dropped
12120 detected pulses to 9897 and the longest clean run from 254 lines to 85.
Smoothed sync detection is the single most common way to get a plausible-looking
picture that is actually a different picture every frame.

## Layout

```
main.py                 argument parsing, source selection, --check, logging
build_exe.py            package the app as a double-clickable .exe
FPV-RF.cmd              start that build from the project root
tools/make_release.py   zip the build, checksum it, attach it to a release
fpv_rf/dsp.py           demod, de-emphasis, sync, rasteriser, classification
fpv_rf/bands.py         the band plans, frequency -> band/channel, legality
fpv_rf/sdr.py           sources: hackrf_transfer, file replay, simulator
fpv_rf/video.py         frame assembly and the decode thread
fpv_rf/scan.py          the hop-walk search, merging, auto-select
fpv_rf/sweep.py         the fast sweep engine: tool, FFTW, trace, detection
fpv_rf/fastscan.py      picks an engine, and hands the radio over and back
fpv_rf/alerts.py        identity, dedupe, cooldown, log, beep
fpv_rf/ui.py            PySide6 window, band chart widget, alert panel
tools/                  the checks below
tools/_paths.py         resolves the optional real capture, no hardcoded paths
tools/check_workflow.py validates .github/workflows/checks.yml before pushing
tools/check_sweep.py    the sweep engine, on hardware (needs a HackRF)
tools/check_fastscan.py the engine choice and the radio handover
tools/probe_handover.py why the handover is unreliable; wedges the radio on purpose
tools/archive/          development scaffolding; nothing runs it automatically
docs/                   the screenshots this README shows
docs/releases/          release notes, versioned alongside the code
```

The 212 MB build is deliberately **not** in git. A binary that size would be
fetched on every clone forever and cannot be reviewed in a diff, so the source
lives here and the build is published as a release asset.

```
python build_exe.py              # build, and verify the build
python tools\make_release.py --dry-run   # zip + checksum, publish nothing
python tools\make_release.py --tag v1.0.1  # zip, checksum, attach to a release
```

## Checks

```
python tools\check_sources.py     # sim / file / real-capture end to end
python tools\check_scan.py        # band search, merging, off-chart, empty band
python tools\check_alerts.py      # dedupe, cooldown, lost, identity
python tools\check_ui.py          # the real Qt window, offscreen, with screenshots
python tools\check_layout.py      # nothing clipped, nothing squeezed
python tools\check_worker.py      # streaming frame rate
python tools\check_workflow.py    # the CI workflow is valid and self-consistent
python tools\check_hardware_scan.py  # a real sweep of a real, empty band
python tools\bench_sim.py         # the simulator sustains real time
python tools\bench_coarse.py      # the coarse gate's false-positive rate
python tools\diag_edges.py        # proves sync detection locks onto SYNC
python tools\diag_spurs.py        # is that carrier the band, or the radio?
python tools\diag_peak_bin.py     # names the exact bin a "carrier" sits in
python build_exe.py               # build the .exe, then verify it did not break
```

The first nine run with no hardware and no capture. The three after
`check_hardware_scan.py` all need the HackRF plugged in.

### Continuous integration

`.github/workflows/checks.yml` runs the nine on every push and pull request, on
both Linux and Windows, because none of them need a radio. The Windows leg is
not redundant: a missing directory, a hardcoded path and an undeclared
dependency all behave identically on either platform, and the shipping target
with the Windows-only timer, subprocess and `winsound` paths is Windows.

Two things are deliberately **not** gates, and the workflow says so in its own
header rather than leaving it to be inferred:

- `bench_sim.py` measures a wall-clock deadline. On a shared runner any
  co-tenant taking CPU fails the build for a change that did not cause it, so it
  is reported and ignored.
- `check_sources.py` and `check_frame_structure.py` assert no threshold. They
  exist to catch a crash in the end-to-end path, and a non-zero exit still fails
  the step. `check_frame_structure.py` reads frames that other checks wrote, so
  on its own it says what it examined rather than printing nothing.

`check_hardware_scan.py` needs the HackRF and is the only one that needs nothing
transmitting to be meaningful — it asserts the *absence* of a finding. It is
also the one that would have caught the DC leak, and the sim would not have.

`check_worker.py` reports about **29–30 pictures/s** against the 30 Hz decode
cadence (`PREVIEW_HZ`), which is a 33.3 ms budget: drain 2.5–3.6 ms, decode
16–21 ms, cycle 32.8–33.2 ms, 1.00× real time from the source. The exact
milliseconds move with the machine and it is the ratio that matters — the whole
iteration fits inside the budget, so the decoder is keeping up. Decode is
deliberately half the NTSC field rate, so the panel's 60 Hz repaint and the
decoder's 30 Hz are different numbers for different reasons and neither is the
other's bottleneck.

Its **pass**, though, is a liveness verdict and not proof of a sustained rate.
It waits for `MIN_FRAMES` — two cadences' worth — and gives up after 30 s, so a
decoder managing 2 fps clears the same bar, only later, and the rate it prints
alongside is reported rather than gated, because any threshold chosen there
would be a statement about the machine it ran on. The gate on throughput is
`check_ui.py`'s `MIN_FRAME_RATE`; `measure_stream_delivery.py` below is the
measurement of the same path with the real worker thread and a real capture.

### Delivery and continuity

Two tools cover what `check_worker.py` does not: that every sample is accounted
for, and what the pipeline actually costs on this machine.

```
python tools\check_stream_continuity.py   # seams, gaps, caps, overruns, identity
python tools\measure_stream_delivery.py --realtime --seconds 2   # the same path, timed
```

`check_stream_continuity.py` pins the ring and drain contract — a capped skip
discards the *oldest* samples, an overrun destroys what nobody read, a block that
straddles a retune is refused whole, a block straddling a process restart is
refused the same way, and `delivered == accepted + rejected` always holds. It also
pins the three seams that used to pass silently: an overlap trim is *decoded*
rather than counted and dropped — and one reaching into a span that was refused
or poisoned starts a fresh segment, because that run continues nothing, so it
takes no boundary step across it — a ring wrap keeps the history that spans it,
and an oversized write reports the bytes it could not keep as loss instead of
manufacturing a contiguous past it never had. It runs on the simulator and, if a
capture is present, repeats the demodulation against the recorded bytes and
requires it to be bit-identical.

`measure_stream_delivery.py` runs the real worker thread over a real capture,
wrapped at the demodulator, the sync detector and the rasteriser, and prints the
phase timings, the publish rate against the 30 Hz cadence, stream lag and the
delivery counters. It takes no gate on those numbers: it is a measurement, and
what is fast is a property of the machine rather than of the decoder. Without
`--realtime` it drains as fast as it can, which saturates the cap and shows the
backlog path instead of the steady-state one.

### Decode diagnostics (temporary)

`fpv_rf/diag.py` is instrumentation for one question — why does a real 5.8 GHz
downlink produce no picture — and it is meant to be deleted when that question
is answered. It counts, once a second, what each stage of the decode chain last
saw: buffers drained and bytes deliberately skipped, sync candidates and the
threshold that produced them, the rasteriser's longest usable run and which stage
refused it, frames started, completed and published. It counts whether or not a
frame comes out, which is the case that matters and the one the statistics in
`DecodeStats` are silent about by construction.

Two independent switches, both off unless set, both environment variables so
nothing has to be threaded through the UI:

```
set FPV_RF_DIAG=1                  # one aggregate DIAG line per second on stdout
set FPV_RF_DIAG_DUMP=D:\cap\seg    # ... and bounded segments on disk
```

The app tees stdout to `fpv-rf.log`, so a live session needs the variable and a
look at that file. A dump writes `<prefix>_NN_iq.u8` (raw interleaved bytes
exactly as received, before scaling), `<prefix>_NN_demod.f32` and
`<prefix>_NN.json` (sample rate, encoding, frequency, the tuning id at **each
end** of the range, spawn epoch, absolute write range and the counters at that
moment). It is bounded by `FPV_RF_DIAG_DUMP_SAMPLES` (1M complex samples, i.e.
2 MB — half the 4 MiB ring), `FPV_RF_DIAG_DUMP_FILES` (4) and
`FPV_RF_DIAG_DUMP_FLUSH_S` (5), so an enabled dump cannot fill a disk.

The raw half is a **snapshot of the IQ ring** taken at the flush deadline, not a
concatenation of drain hand-overs: a hand-over is capped, skips a backlog and
begins wherever the drain cursor was, so several of them make a file whose bytes
are in order but are not adjacent in the stream — gaps the metadata would not
mention. The snapshot is taken without consuming, so reading the dump does not
change what the decoder is credited with having received. If a snapshot straddles
a retune it holds two frequencies, and since nothing in the samples says so, the
JSON records `"spans_tune": true`, `"continuous": false` and both tuning ids; a
segment following a transfer-process restart records
`"after_process_restart"` the same way.

```
python tools\check_decode_diag.py   # the instrumentation is sound, and says so
```

That check is deliberately **not** in CI, because neither is the code it checks:
it asserts the decode is pixel-identical with the variables unset and set, that
counters move on a signal which never locks, that every drained byte is accounted
for as delivered, skipped or lost in a range gap, that the dump is off by default
and bounded when asked for, that the raw half of a segment is one contiguous
non-consuming ring snapshot with its range and retune seams labelled, and that
`dsp.detect_sync(collect_diag=True)` reports both polarities without changing the
`SyncResult` it returns.

To remove all of it: delete `fpv_rf/diag.py`, `tools\check_decode_diag.py`,
every block marked `TEMP DIAG` in `dsp.py`, `sdr.py` and `video.py`, and this
subsection. Two checks outside `tools\check_decode_diag.py` touch it and need
their lines as well: `tools\check_stream_continuity.py` drives `diag.dump_tick`
in its snapshot-neutrality case, and `tools\check_raster_window.py` calls
`diag.reset()` and reads `Rasteriser._diag.gauge_value("reject")` to tell a
refusal reason from a missing frame. Nothing else refers to any of it.

## The optional real capture

Several checks and diagnostics are more convincing against a real HackRF
recording than against the simulator. A capture is 38 MB of IQ, so it is not in
this repository, and `tools/_paths.py` resolves where one is instead of any
tool hardcoding a path. In order:

1. `$FPV_CAPTURE` — a full path to a `.u8` file, or the folder containing it.
2. `samples/r5_5802_g16.u8` in the repository. That folder is gitignored, so
   dropping a capture there is safe.
3. The same filename found by a shallow search of your home directory.

**A fresh clone has none, and that is the normal case.** Everything that needs
one reports `SKIP` and continues; nothing fails. `check_scan.py`, `validate_dsp.py`
and `bench_coarse.py` all exercise the simulator instead, so the suite is
meaningful with no radio and no capture at all.

To record your own with a HackRF on a quiet band:

```
hackrf_transfer -r my.u8 -f 5802000000 -s 10000000 -n 20000000 -l 32 -g 16
$env:FPV_CAPTURE = "C:\path\to\my.u8"
python tools\validate_dsp.py
```

A further handful of scripts in `tools/archive/` need that same capture. They are
development scaffolding: nothing runs them, and `tools/archive/README.md`
explains what each one is for. The repository is self-contained — nothing in it
depends on a project that isn't here.

## Hardware

Tested against a Mayhem firmware build (`n_260505`, API 1.11), serial
`0000000000000000a09867dc391638A3`.

```
hackrf_transfer -r file.u8 -f 5802000000 -s 10000000 -n 20000000 -l 32 -g 16
```

LNA 32, VGA 16, 10 MS/s. `pyhackrf2` is not used: it hardcodes
`libhackrf.so.0` and this machine has no `hackrf.dll`, and the Mayhem build is
statically linked. Streaming is a `hackrf_transfer -r -` subprocess with the
`-r` flag first, because that is where it has to be, and the child's stderr
gets its own thread because on a `bufsize=0` pipe a read on it would otherwise
stall the only path to the samples.

10 MS/s, not 20: NTSC at 60 fields/s needs 10.014 and PAL at 50 needs 9.98, and
20 MS/s costs USB bus contention that shows up as dropped samples. Measured
sustained on the live radio: 20 MB/s, exactly 10 MS/s.
