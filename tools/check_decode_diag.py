"""Focused check of the TEMP DIAG decode instrumentation (see fpv_rf/diag.py).

The numbers are only worth reading if the instrumentation itself is sound, and
there are five ways it could be quietly worthless:

  1. It changed the decode. The same IQ must produce the same pixels with
     FPV_RF_DIAG unset and set -- checked on a real simulator capture, fed to
     two VideoDecoders.
  2. It is silent when there is no picture, which is the case it exists for.
     Noise that never locks must still move counters.
  3. The drain accounting does not add up. Every byte must be accounted for as
     delivered, skipped, or lost in a range gap.
  4. The optional extras are optional: no file without the environment variable,
     bounded files with it, and sync statistics that report both polarities
     without changing the SyncResult that comes back.
  5. The captured bytes are not the stream they claim to be. The raw half of a
     segment is snapshotted out of the IQ ring, so it must be one contiguous
     run -- not several drain hand-overs stitched together, which are in order
     but not adjacent and still look like one recording. It must also not count
     as consumption, or the decoder's loss accounting improves because a
     diagnostic looked at the data, and a snapshot across a retune must say so.
  6. The publication-matched capture describes the picture it was taken at. Its
     IQ must cover the whole decoder window, its PNG must be the frame that was
     published rather than merely a frame, a discontinuity inside the window must
     be recorded as one, and with it switched off the decode must be identical
     and the file count must be zero.

Run:  python tools\\check_decode_diag.py
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv_rf import diag, dsp, sdr, video  # noqa: E402

OUT = Path("_validate_out/diag_dump")
SNAP_OUT = Path("_validate_out/diag_snapshot")
CAP_OUT = Path("_validate_out/diag_pubcap")
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def capture_sim_iq(n_samples: int = 420_000) -> np.ndarray:
    """One deterministic chunk of real simulator IQ, for both decode passes."""
    src = sdr.SimSource(video=True)
    src.start()
    end = time.monotonic() + 2.0
    while src.ring.available() < n_samples * 2 and time.monotonic() < end:
        time.sleep(0.02)
    iq = src.take_iq(n_samples)
    src.stop()
    return iq


def decode_chunks(iq: np.ndarray, chunk: int = 175_000) -> list[np.ndarray]:
    dec = video.VideoDecoder(10_000_000, width=160)
    frames = []
    for i in range(0, iq.size, chunk):
        fr = dec.push_iq(iq[i : i + chunk])
        if fr is not None:
            frames.append(fr.image.copy())
    return frames


def case_no_behaviour_change(iq: np.ndarray) -> None:
    print("\n1. the instrumentation does not change the decode")
    os.environ.pop(diag.ENV_DIAG, None)
    plain = decode_chunks(iq)
    os.environ[diag.ENV_DIAG] = "1"
    diag.reset()
    traced = decode_chunks(iq)
    os.environ.pop(diag.ENV_DIAG, None)
    check("frames produced in both passes", bool(plain) and len(plain) == len(traced),
          f"{len(plain)} vs {len(traced)}")
    same = len(plain) == len(traced) and all(
        np.array_equal(a, b) for a, b in zip(plain, traced)
    )
    check("every frame is pixel-identical", same)
    check("diag off prints nothing", diag.maybe_log() is None)
    os.environ[diag.ENV_DIAG] = "1"
    check("diag on emits one line", diag.maybe_log() is not None)
    check("second call inside the second is rate-limited", diag.maybe_log() is None)
    os.environ.pop(diag.ENV_DIAG, None)
    r = diag.section("raster")
    check("rasteriser recorded its own account",
          r.value("runs_ok") > 0 and r.gauge_value("reject") == "none"
          and int(r.gauge_value("usable") or 0) >= int(r.gauge_value("floor") or 1),
          f"runs_ok={r.value('runs_ok'):.0f} usable={r.gauge_value('usable')} "
          f"floor={r.gauge_value('floor')}")


def case_counters_without_frames() -> None:
    print("\n2. counters move even when nothing is published")
    diag.reset()
    os.environ.pop(diag.ENV_DIAG, None)
    rng = np.random.default_rng(7)
    noise = (rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000)).astype(
        np.complex64
    ) * np.float32(0.3)
    dec = video.VideoDecoder(10_000_000, width=160)
    frames = 0
    for i in range(0, noise.size, 175_000):
        frames += dec.push_iq(noise[i : i + 175_000]) is not None
    d = diag.section("dec")
    r = diag.section("raster")
    check("no frame from noise", frames == 0)
    check("pushes counted", d.value("pushes") >= 2, f"{d.value('pushes'):.0f}")
    check("samples counted", d.value("samples_in") >= noise.size,
          f"{d.value('samples_in'):.0f}")
    check("sync rejections counted",
          d.value("sync_rejected") > 0 and d.value("sync_rejected_few_pulses") > 0,
          f"rejected={d.value('sync_rejected'):.0f} "
          f"few={d.value('sync_rejected_few_pulses'):.0f}")
    check("sync state still recorded pre-raster", d.gauge_value("sync_pulses") is not None)
    check("the rasteriser was never reached", r.value("select_run_calls") == 0,
          f"select_run_calls={r.value('select_run_calls'):.0f}")
    check("no frame counted", d.value("frames_decoded") == 0)

    # ... and the worker's own accounting, on a source that never locks.
    diag.reset()
    src = sdr.SimSource(video=False)
    src.start()
    w = video.DecodeWorker(src, width=160)
    w.start()
    end = time.monotonic() + 1.5
    while time.monotonic() < end:
        time.sleep(0.05)
    w.stop()
    src.stop()
    wd = diag.section("worker")
    check("worker counted buffers", wd.value("buffers") > 10, f"{wd.value('buffers'):.0f}")
    check("worker counted samples", wd.value("samples_in") > 1_000_000)
    check("worker saw the sync gate refuse", diag.section("dec").value("sync_rejected") > 0)
    check("no frames published", w.stats.frames == 0)
    check("published-frame age is visible",
          wd.gauge_value("frame_age_ms") is not None)


def case_drain_accounting() -> None:
    print("\n3. every byte of a hand-over is accounted for")
    diag.reset()
    src = sdr.SimSource()
    d = diag.section("iq")
    src.ring.write(bytes(2 * 1_000_000), src._tune_id)      # 1M samples pending
    first = src.drain_iq(100_000)
    check("delivered the capped tail", first.size == 100_000, f"{first.size} samples")
    check("pending counted", d.value("pending_samples") == 1_000_000)
    check("delivered counted", d.value("delivered_samples") == 100_000)
    check("silent skip counted", d.value("skipped_samples") == 900_000,
          f"{d.value('skipped_samples'):.0f}")
    check("skip events counted", d.value("skip_events") == 1)
    check("read-size histogram populated", bool(d.snapshot()["histograms"].get("read_kib")))
    check("the ring did not call it loss", src.ring.dropped_bytes == 0,
          "a deliberate skip is not an overrun")

    src.ring.write(bytes(2 * 50_000), src._tune_id)         # 50k more, contiguous
    second = src.drain_iq(100_000)
    check("next hand-over is contiguous", second.size == 50_000 and
          d.value("range_gap_events") == 0)
    check("tune id stamped on the hand-over", d.gauge_value("tune_id_last") == 0)
    check("ring seam list readable", src.ring.diag_tags() == ((0, 0),))

    src.ring.write(bytes(2 * 1000), 1)                     # a retune, mid-stream
    src.drain_iq(100_000)
    check("a seam is visible in the ring's tags",
          len(src.ring.diag_tags()) == 2 and src.ring.diag_tags()[-1][1] == 1,
          str(src.ring.diag_tags()))

    src.ring.write(bytes(2 * 1000), 1)                     # discarded by the reset
    src.reset_drain()
    check("reset counted", d.value("drain_resets") == 1)
    check("reset discard counted", d.value("reset_discarded_samples") == 1000,
          f"{d.value('reset_discarded_samples'):.0f}")

    src.tune(5_815_000_000)
    check("applied tune counted", d.value("tune_applied") == 1 and
          d.gauge_value("tune_id") == 1)


def case_dump() -> None:
    print("\n4. the dump hook is off by default and bounded when asked for")
    for stale in OUT.glob("*"):
        stale.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    diag.reset()
    src = sdr.SimSource(video=True)
    block = bytes(2 * 40_000)
    src.ring.write(block, 0)
    src.drain_iq(20_000)
    src.drain_iq(20_000)
    check("nothing written without the variable", not list(OUT.glob("*")),
          f"{len(list(OUT.glob('*')))} files")

    prefix = str(OUT / "seg")
    os.environ[diag.ENV_DUMP] = prefix
    os.environ[diag.ENV_DUMP_SAMPLES] = "100000"
    os.environ[diag.ENV_DUMP_FILES] = "2"
    os.environ[diag.ENV_DUMP_FLUSH_S] = "1"
    diag._DUMP.reset()
    src.reset_drain()
    dec = video.VideoDecoder(10_000_000, width=160)
    end = time.monotonic() + 1.4
    while time.monotonic() < end:
        src.ring.write(block, 0)
        iq = src.drain_iq(20_000)
        dec.push_iq(iq)
        time.sleep(0.01)
    files = sorted(p.name for p in OUT.glob("*"))
    check("dump wrote a segment", any(n.endswith(".json") for n in files), str(files[:4]))
    meta_path = sorted(OUT.glob("*.json"))[0]
    meta = json.loads(meta_path.read_text())
    iq_file = Path(str(meta_path).replace(".json", "_iq.u8"))
    demod_file = Path(str(meta_path).replace(".json", "_demod.f32"))
    check("raw IQ saved", iq_file.exists() and
          iq_file.stat().st_size == 2 * meta["iq_samples"],
          f"{iq_file.name} {iq_file.stat().st_size if iq_file.exists() else 0} B")
    check("demod saved as float32", demod_file.exists() and
          demod_file.stat().st_size == 4 * meta["demod_samples"],
          f"{demod_file.name} {demod_file.stat().st_size if demod_file.exists() else 0} B")
    check("window bounded", meta["iq_samples"] <= 100_000 and meta["demod_samples"] <= 100_000,
          f"iq={meta['iq_samples']} demod={meta['demod_samples']}")
    for key in ("sample_rate", "encoding", "frequency_hz", "tune_id", "epoch",
                "first_byte", "last_byte", "counts"):
        check(f"meta records {key}", key in meta)
    check("file budget respected", len(list(OUT.glob("*.json"))) <= 2,
          f"{len(list(OUT.glob('*.json')))} files")
    for name in (diag.ENV_DUMP, diag.ENV_DUMP_SAMPLES, diag.ENV_DUMP_FILES,
                 diag.ENV_DUMP_FLUSH_S):
        os.environ.pop(name, None)
    diag._DUMP.reset()
    check("dumping stops when switched off", diag.dump_enabled() is False)


def case_sync_diagnostics(iq: np.ndarray) -> None:
    print("\n6. sync diagnostics report both polarities and change nothing")
    demod = dsp.fm_demodulate(iq)
    os.environ.pop(diag.ENV_DIAG, None)
    plain = dsp.detect_sync(demod, 10_000_000, polarity="auto")
    traced = dsp.detect_sync(demod, 10_000_000, polarity="auto", collect_diag=True)
    check("no diag unless asked", plain.diag is None and traced.diag is not None)
    check("same pulses", np.array_equal(plain.pulse_starts, traced.pulse_starts))
    check("same centres", np.array_equal(plain.pulse_centres, traced.pulse_centres))
    check("same line samples and polarity",
          plain.line_samples == traced.line_samples and plain.polarity == traced.polarity
          and plain.threshold == traced.threshold and plain.quality == traced.quality
          and plain.spec is traced.spec)
    sd = traced.diag
    check("both polarities measured", sd.positive is not None and sd.negative is not None)
    check("live thresholds recorded",
          sd.sensitivity == 0.975 and sd.nominal_line > 0
          and sd.min_gap == int(sd.nominal_line * 0.5),
          f"sens={sd.sensitivity} nominal={sd.nominal_line} min_gap={sd.min_gap}")
    check("early break honoured", sd.early_break and sd.decided_polarity == plain.polarity,
          f"early_break={sd.early_break} decided={sd.decided_polarity} "
          f"chosen={plain.polarity}")
    for side, name in ((sd.positive, "pos"), (sd.negative, "neg")):
        check(f"{name}: raw edges exceed clustered",
              side.raw_edges >= side.clustered and side.clustered > 0,
              f"raw={side.raw_edges} clustered={side.clustered}")
        check(f"{name}: spacing recorded",
              side.spacing_min > 0 and side.spacing_median > 0 and side.spacing_max > 0,
              f"{side.spacing_min:.0f}/{side.spacing_median:.0f}/{side.spacing_max:.0f}")
        check(f"{name}: excursion widths recorded",
              0 < side.excursion_min <= side.excursion_median <= side.excursion_max,
              f"{side.excursion_min:.0f}/{side.excursion_median:.0f}/"
              f"{side.excursion_max:.0f} samples")
    flat = sd.flatten()
    check("flatten is gauge-ready",
          all(isinstance(v, float) for v in flat.values())
          and any(k.startswith("pos_") for k in flat) and "pos_thr" in flat,
          f"{len(flat)} keys")
    check("wrapper agrees with the call", dsp.sync_diagnostics(demod, 10_000_000).n_samples
          == demod.size)

    # select_run: same answer, plus the account of how it got there.
    starts = traced.pulse_starts
    line = traced.line_samples
    want = traced.spec.visible_lines
    out: dict = {}
    with_out = dsp.Rasteriser.select_run(starts, line, want, diag_out=out)
    without = dsp.Rasteriser.select_run(starts, line, want)
    check("select_run unchanged", (with_out is None) == (without is None)
          and (with_out is None or np.array_equal(with_out.positions, without.positions)))
    check("select_run reports its search", out.get("reason") is not None and
          out.get("runs", 0) >= 1 and out.get("period", 0) > 0, str(out.get("reason")))
    check("select_run reports the longest run", int(out.get("max_span", 0)) >= want,
          f"{out.get('max_span')} of {want}")
    short: dict = {}
    check("too few detections is named",
          dsp.Rasteriser.select_run(starts[:3], line, want, diag_out=short) is None
          and short.get("reason") == "span", str(short.get("reason")))
    tiny: dict = {}
    dsp.Rasteriser.select_run(starts[:1], line, want, diag_out=tiny)
    check("degenerate input is named", tiny.get("reason") == "degenerate",
          str(tiny.get("reason")))


def case_iq_snapshot() -> None:
    print("\n5. the raw half is one contiguous ring snapshot, not a collage")
    # A pattern that identifies every byte's absolute position, so a snapshot
    # that stitched two stretches together could not pass for one run.
    stream = bytes((i * 7 + 3) & 0xFF for i in range(8192))

    # (a) Atomic and in order across the wrap, with the range it covers.
    ring = sdr.IQRing(size=1024)
    for i in range(0, 3000, 200):                  # 200 B at a time: wraps, no truncation
        ring.write(stream[i : i + 200], 0)
    snap = ring.snapshot_newest(512)
    check("snapshot is contiguous and self-describing",
          snap.first_byte == 3000 - 512 and snap.last_byte == 3000
          and snap.last_byte - snap.first_byte == len(snap.data)
          and snap.data == stream[2488:3000],
          f"range [{snap.first_byte}, {snap.last_byte}) of {len(snap.data)} B")
    whole = ring.snapshot_newest(99_999)
    check("a snapshot larger than the ring returns the ring, once",
          len(whole.data) == 1024 and whole.data == stream[1976:3000],
          f"{len(whole.data)} B == ring size")

    # (b) Non-consuming. Were the snapshot to move the loss cursor, an overrun
    # past it would read as "somebody had this" and the loss would vanish --
    # which is the decoder's accounting being quietly improved by a diagnostic.
    def overrun(read: str) -> int:
        r = sdr.IQRing(size=1024)
        r.write(stream[:400], 0)
        if read == "snapshot":
            r.snapshot_newest(400)
        elif read == "consuming":
            r.read_newest(400)
        for i in range(400, 3000, 200):            # overrun the ring, in order
            r.write(stream[i : i + 200], 0)
        return r.dropped_bytes

    with_snap = overrun("snapshot")
    without = overrun("none")
    consumed = overrun("consuming")
    check("a snapshot does not count as consumption",
          with_snap == without and with_snap > 0,
          f"{with_snap} B lost with the snapshot, {without} B without")
    check("read_newest, unlike the snapshot, does count as consumption",
          consumed < with_snap,
          f"read_newest loses {consumed} B where a snapshot loses {with_snap} B")

    # (c) A snapshot across a retune says so, in the metadata beside it.
    ring4 = sdr.IQRing(size=1024)
    ring4.write(stream[:600], 0)
    ring4.write(stream[600:1000], 1)
    seam = ring4.snapshot_newest(800)
    check("a snapshot spanning a retune is identified",
          seam.spans_tune and seam.first_tune_id == 0 and seam.last_tune_id == 1,
          f"tune {seam.first_tune_id} -> {seam.last_tune_id}")
    same = ring4.snapshot_newest(200)
    check("a snapshot inside one tuning is not",
          not same.spans_tune and same.first_tune_id == same.last_tune_id == 1,
          f"tune {same.first_tune_id} -> {same.last_tune_id}")

    # (d) End to end, with the deadline forced rather than waited for: the file
    # is the snapshot, and one that straddles the retune is labelled rather than
    # passed off as a continuous capture.
    for stale in SNAP_OUT.glob("*"):
        stale.unlink()
    SNAP_OUT.mkdir(parents=True, exist_ok=True)
    src = sdr.SimSource(video=True)
    src.ring.write(bytes(2 * 60_000), 0)
    src.ring.write(bytes(2 * 60_000), 1)            # the seam is inside the range
    os.environ[diag.ENV_DUMP] = str(SNAP_OUT / "snap")
    os.environ[diag.ENV_DUMP_SAMPLES] = "100000"
    os.environ[diag.ENV_DUMP_FILES] = "4"
    os.environ[diag.ENV_DUMP_FLUSH_S] = "1"
    diag._DUMP.reset()
    diag._DUMP._next = 0.0                          # the deadline is due now
    diag.dump_tick(src._dump_snapshot)
    src.ring.write(bytes(2 * 200_000), 1)           # well past the seam
    diag._DUMP._next = 0.0
    diag.dump_tick(src._dump_snapshot)
    src.process_epoch += 1                          # a transfer process restarted
    src.ring.write(bytes(2 * 200_000), 1)
    diag._DUMP._next = 0.0
    diag.dump_tick(src._dump_snapshot)
    metas = sorted(SNAP_OUT.glob("*.json"))
    check("every segment was written", len(metas) == 3,
          f"{[p.name for p in SNAP_OUT.glob('*')]}")
    for path in metas:
        meta = json.loads(path.read_text())
        iq_file = Path(str(path).replace(".json", "_iq.u8"))
        tag = path.stem
        check(f"{tag}: the file is the snapshot, sample for sample",
              iq_file.exists() and iq_file.stat().st_size == 2 * meta["iq_samples"]
              and meta["last_byte"] - meta["first_byte"] == iq_file.stat().st_size
              and meta["iq_source"] == "ring_snapshot",
              f"{iq_file.stat().st_size if iq_file.exists() else 0} B, range "
              f"[{meta.get('first_byte')}, {meta.get('last_byte')}), "
              f"source={meta.get('iq_source')}")
        spanning = bool(meta.get("spans_tune"))
        restarted = "after_process_restart" in meta
        # continuous is False for either kind of discontinuity, and absent for a
        # segment with neither -- which is the third case, not a fourth.
        want = False if (spanning or restarted) else None
        check(f"{tag}: continuity is labelled as it is",
              meta.get("continuous", None) is want,
              f"spans_tune={spanning} restarted={restarted} "
              f"continuous={meta.get('continuous')} "
              f"tune {meta.get('first_tune_id')} -> {meta.get('last_tune_id')}")
    if len(metas) == 3:
        first, second, third = (json.loads(p.read_text()) for p in metas)
        check("only the segment that crosses the seam is marked",
              first.get("spans_tune") is True and second.get("spans_tune") is False
              and third.get("spans_tune") is False
              and first["first_tune_id"] == 0 and second["first_tune_id"] == 1,
              f"[{first['first_byte']}, {first['last_byte']}) spans="
              f"{first.get('spans_tune')}, [{second['first_byte']}, "
              f"{second['last_byte']}) spans={second.get('spans_tune')}")
        check("a segment after a process restart is labelled as one",
              third.get("continuous") is False
              and third.get("after_process_restart") == {"previous_epoch": second["epoch"],
                                                         "epoch": third["epoch"]}
              and third["epoch"] == second["epoch"] + 1,
              f"epoch {second['epoch']} -> {third['epoch']}, "
              f"continuous={third.get('continuous')}")
    for name in (diag.ENV_DUMP, diag.ENV_DUMP_SAMPLES, diag.ENV_DUMP_FILES,
                 diag.ENV_DUMP_FLUSH_S):
        os.environ.pop(name, None)
    diag._DUMP.reset()
    check("dumping stops when switched off", diag.dump_enabled() is False)


def png_pixels(raw: bytes) -> np.ndarray:
    """Decode the greyscale PNG ``diag.png_bytes`` writes, with stdlib only.

    Deliberately not Pillow: the point is that the file is a *real* PNG a viewer
    can open, so every chunk is walked and its CRC verified rather than handed to
    a library that would quietly tolerate a malformed one. The app itself never
    imports Pillow and the packaged build excludes it, so a capture that needed it
    would not exist in the thing the capture is diagnosing.
    """
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos, idat, hdr = 8, b"", None
    while pos + 8 <= len(raw):
        (length,) = struct.unpack(">I", raw[pos : pos + 4])
        tag = raw[pos + 4 : pos + 8]
        payload = raw[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", raw[pos + 8 + length : pos + 12 + length])
        if crc != zlib.crc32(tag + payload) & 0xFFFFFFFF:
            raise ValueError(f"bad CRC in {tag!r}")
        if tag == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", payload)
        elif tag == b"IDAT":
            idat += payload
        pos += 12 + length
    if hdr is None:
        raise ValueError("no IHDR")
    width, height, depth, colour = hdr[0], hdr[1], hdr[2], hdr[3]
    if (depth, colour) != (8, 0):
        raise ValueError(f"not 8-bit greyscale: depth={depth} colour={colour}")
    flat = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(height, width + 1)
    if not (flat[:, 0] == 0).all():
        raise ValueError("a scanline used a filter the writer does not emit")
    return flat[:, 1:].copy()


def to_u8(iq: np.ndarray, signed: bool = False) -> bytes:
    """Interleaved IQ bytes, spelled here independently of :mod:`fpv_rf.diag`.

    The simulator's own encoding is ``round(sample * 127) + 128`` into offset
    binary; this is the inverse of what ``sdr.u8_to_iq`` does, so the worker under
    test receives exactly the samples this function encodes. Written out rather
    than borrowed so that the check's round trip through ``diag.iq_to_u8`` is not
    a tautology.
    """
    scaled = np.rint(np.column_stack((iq.real, iq.imag)) * 127.5)
    if signed:
        return np.clip(scaled, -128, 127).astype(np.int8).tobytes()
    return np.clip(scaled + 128.0, 0, 255).astype(np.uint8).tobytes()


def feed_worker(w: video.DecodeWorker, iq: np.ndarray, per_step: int,
                tune_id: int = 0, stream_id: int = 1) -> list[np.ndarray]:
    """Push ``iq`` through a worker one chunk per step.

    Returns the images it published, in order, so a capture can be matched to the
    picture it claims to be rather than merely to "a" picture.
    """
    raw = to_u8(iq)
    published: list[np.ndarray] = []
    step_bytes = 2 * per_step
    for off in range(0, len(raw), step_bytes):
        w.source.ring.write(raw[off : off + step_bytes], tune_id, stream_id)
        if w.step():
            frame = w.latest()
            if frame is not None:
                published.append(frame.image.copy())
    return published


def capture_for_pubcap(n_samples: int = 2_000_000, seconds: float = 10.0) -> np.ndarray:
    """Simulator IQ long enough for several publications, captured in one go."""
    src = sdr.SimSource(video=True)
    src.start()
    end = time.monotonic() + seconds
    while src.ring.available() < n_samples * 2 and time.monotonic() < end:
        time.sleep(0.02)
    iq = src.take_iq(n_samples)
    src.stop()
    return iq


def case_publication_capture(iq: np.ndarray) -> None:
    print("\n7. the publication capture: off by default, matched to the frame, bounded")
    for stale in CAP_OUT.glob("*"):
        stale.unlink()
    CAP_OUT.mkdir(parents=True, exist_ok=True)

    # (a) Off unless asked for. Nothing on disk, no ledger, no budget spent.
    diag.reset()
    diag._DUMP.reset()
    diag.pubcap_reset()
    for name in (diag.ENV_DIAG, diag.ENV_PUBCAP, diag.ENV_DUMP,
                 diag.ENV_DUMP_FILES, diag.ENV_DUMP_SAMPLES, diag.ENV_DUMP_FLUSH_S):
        os.environ.pop(name, None)
    check("off unless the switch is set", diag.pubcap_enabled() is False)
    src = sdr.SimSource(video=True)
    plain_worker = video.DecodeWorker(src, width=160)
    check("and no ledger is built for it", plain_worker.ledger is None
          and plain_worker.decoder.ledger is None)
    plain = feed_worker(plain_worker, iq, 175_000)
    check("pictures are published without it", len(plain) > 0, f"{len(plain)}")
    check("nothing written while off", not list(CAP_OUT.glob("*")),
          f"{len(list(CAP_OUT.glob('*')))} files")
    check("and nothing spent of the budget", diag._DUMP._spent == 0)

    # (b) The decode is bit-identical with the capture on. Two decoders, the same
    # samples, one carrying a ledger: any difference is the capture's doing.
    # Pushed with placements, because the capture is about *placed* windows and an
    # unplaced one has nothing to match the samples against.
    diag.reset()
    ledger = diag.pubcap_ledger(video.window_samples(10_000_000), 420_000)
    dec_off = video.VideoDecoder(10_000_000, width=160)
    dec_on = video.VideoDecoder(10_000_000, width=160, iq_ledger=ledger)
    off_frames, on_frames, last_on = [], [], None
    for k, off in enumerate(range(0, iq.size, 175_000)):
        chunk = iq[off : off + 175_000]
        a = dec_off.push_iq(chunk, continuous=bool(k), first_sample=off)
        b = dec_on.push_iq(chunk, continuous=bool(k), first_sample=off)
        if a is not None:
            off_frames.append((int(dec_off._demod_origin), int(dec_off._demod.size),
                               a.image.copy()))
        if b is not None:
            last_on = b
            on_frames.append((int(dec_on._demod_origin), int(dec_on._demod.size),
                              b.image.copy()))
    check("both passes publish the same pictures",
          len(off_frames) == len(on_frames) and bool(off_frames),
          f"{len(off_frames)} vs {len(on_frames)}")
    check("every picture is pixel-identical",
          all(np.array_equal(a[2], b[2]) for a, b in zip(off_frames, on_frames)))
    check("and each is drawn from the same window",
          all(a[:2] == b[:2] for a, b in zip(off_frames, on_frames)))

    # (c) The ledger covers the whole window the picture was drawn from. This is
    # the property the margin in ``pubcap_ledger`` exists for: every sample of the
    # window has to be there, including the one supporting its last step, or the
    # oldest lines of the picture are missing from the capture.
    cap = dec_on.publication(last_on) if last_on is not None else None
    window_end = int(dec_on._demod_origin) + int(dec_on._demod.size)
    check("the ledger covers the window the picture was drawn from",
          cap is not None and int(cap.demod.size) == int(dec_on._demod.size)
          and bool(cap.spans)
          and int(cap.spans[0].first_sample) <= int(dec_on._demod_origin)
          and int(cap.spans[-1].last_sample) >= window_end,
          f"demod [{dec_on._demod_origin}, {window_end}) in "
          f"{len(cap.spans) if cap else 0} span(s) from "
          f"{cap.spans[0].first_sample if cap else 0} to "
          f"{cap.spans[-1].last_sample if cap else 0}")
    check("and the capture says it is one continuous stretch",
          cap is not None and cap.as_dict()["continuous"] is True
          and not cap.as_dict()["iq_gaps"])
    check("the capture's ranges reach the frame's own interval",
          cap is not None
          and int(cap.spans[0].first_sample) <= int(last_on.iq_first)
          and int(cap.spans[-1].last_sample) >= int(last_on.iq_last),
          f"frame [{last_on.iq_first}, {last_on.iq_last})" if last_on else "")
    check("the ledger stays bounded while it fills",
          ledger.samples <= ledger.limit, f"{ledger.samples} of {ledger.limit}")

    # (d) ``iq_to_u8`` is the exact inverse of ``sdr.u8_to_iq``, so a replay of the
    # capture decodes the samples the decoder held rather than samples near them.
    # The input is built the way ``sdr`` builds one -- an integer byte times
    # float32 ``1/127.5`` -- because only samples already on that grid survive a
    # byte. Rounding an arbitrary float onto the grid is a quantisation, not an
    # inverse, and testing the inverse with an input it cannot represent would
    # report a defect that is not there.
    rng = np.random.default_rng(99)
    scale = np.float32(1.0 / 127.5)
    ints = rng.integers(-127, 128, size=(4096, 2)).astype(np.float32)
    probe = ((ints[:, 0] * scale) + 1j * (ints[:, 1] * scale)).astype(np.complex64)
    for encoding, enum in (("signed-int8", sdr.IQEncoding.SIGNED_INT8),
                           ("unsigned-offset", sdr.IQEncoding.UNSIGNED_OFFSET)):
        back = sdr.u8_to_iq(diag.iq_to_u8(probe, encoding), enum)
        check(f"iq_to_u8 round-trips exactly ({encoding})",
              back.size == probe.size and np.array_equal(back, probe),
              f"{int((back != probe).sum())} of {probe.size} samples differ, "
              f"max |diff| = {float(np.abs(back - probe).max()) if back.size else -1}")
    # Both encodings scale by the same factor, so the same samples must come back
    # from the same amplitudes whatever the byte convention was.
    check("the two encodings carry the same amplitudes",
          np.array_equal(
              sdr.u8_to_iq(diag.iq_to_u8(probe, "unsigned-offset"),
                           sdr.IQEncoding.UNSIGNED_OFFSET),
              sdr.u8_to_iq(diag.iq_to_u8(probe, "signed-int8"),
                           sdr.IQEncoding.SIGNED_INT8)))
    # And the shift must not fold the top of the range: an unsigned clip at 127
    # instead of 255 would turn every sample above -1/127.5 into one value.
    loud = (np.array([0.99, 0.5, 0.0, -0.5, -0.99], dtype=np.float32)
            * (1 + 0j)).astype(np.complex64)
    un = sdr.u8_to_iq(diag.iq_to_u8(loud, "unsigned-offset"),
                      sdr.IQEncoding.UNSIGNED_OFFSET)
    check("the unsigned shift reaches the top of the byte range",
          np.array_equal(np.rint(un.real * 127.5), np.rint(loud.real * 127.5)),
          f"{[round(float(v) * 127.5) for v in un.real]} vs "
          f"{[round(float(v) * 127.5) for v in loud.real]}")
    check("the bytes are one u8 pair per sample",
          len(diag.iq_to_u8(probe, "signed-int8")) == 2 * probe.size)
    check("an empty window encodes to nothing",
          diag.iq_to_u8(np.zeros(0), "signed-int8") == b"")
    try:
        diag.iq_to_u8(probe, "nonsense")
        unknown_ok = False
    except ValueError:
        unknown_ok = True
    check("an unknown encoding is refused rather than guessed", unknown_ok)

    # (e) A gap inside the window is recorded, not smoothed over. Driven here
    # rather than end to end because the decoder resets its window at every
    # discontinuity -- so this is the case that must never arise from the live
    # path, and a check that could only ever see the easy answer would prove
    # nothing about the reporting.
    holed = diag.IqLedger(1 << 20)
    holed.add(diag.IqSpan(iq=probe[:100], first_sample=1_000_000, stream_id=4,
                          tune_id=1, segment=2, delivery=9, starts_segment=True))
    holed.add(diag.IqSpan(iq=probe[100:200], first_sample=1_000_400, stream_id=4,
                          tune_id=1, segment=2, delivery=10))
    gapcap = diag.PubCap()
    gapcap.spans = holed.spans()
    gapcap.demod = np.zeros(199, dtype=np.float32)
    _, gaps = gapcap.ranges()
    info = gapcap.as_dict()
    check("a hole in the window is reported as a hole",
          len(gaps) == 1 and gaps[0]["missing_samples"] == 300
          and gaps[0]["reason"] == "gap" and info["continuous"] is False,
          f"{gaps}")
    check("and the samples are still counted", info["iq_samples"] == 200)

# (f) End to end through the worker, at the publication decision. Long
    # enough for several publications, because the bounds below are only
    # meaningful once the file set has had to roll.
    long_iq = capture_for_pubcap()
    print(f"    {long_iq.size} more simulator samples for the end-to-end drive")
    prefix = str(CAP_OUT / "pub")
    os.environ[diag.ENV_DUMP] = prefix
    os.environ[diag.ENV_PUBCAP] = "1"
    os.environ[diag.ENV_DUMP_FILES] = str(diag.DUMP_FILES)
    os.environ[diag.ENV_DUMP_FLUSH_S] = "3600"        # the periodic dump stays out
    diag.reset()
    diag._DUMP.reset()
    diag.pubcap_reset()
    src = sdr.SimSource(video=True)
    w = video.DecodeWorker(src, width=160)
    check("the ledger is built when the switch is on",
          w.ledger is not None and w.decoder.ledger is w.ledger
          and w.ledger.limit >= 2 * 420_000,
          f"limit={w.ledger.limit if w.ledger else 0}")
    pictures = feed_worker(w, long_iq, 175_000)
    check("several publications happened", len(pictures) >= 4, f"{len(pictures)}")
    metas = sorted(CAP_OUT.glob("*_pub*.json"))
    check("one json per captured publication, and no more than the file set",
          0 < len(metas) <= diag.PUBCAP_FILES,
          f"{[p.name for p in CAP_OUT.glob('*')]}")
    check("the file count is bounded at four per publication",
          len(list(CAP_OUT.glob("*"))) <= 4 * diag.PUBCAP_FILES,
          f"{len(list(CAP_OUT.glob('*')))} files")
    check("the slots roll rather than accumulate",
          len({json.loads(p.read_text())["slot"] for p in metas}) == len(metas)
          and len(pictures) > len(metas),
          f"{len(pictures)} publications, slots "
          f"{sorted({json.loads(p.read_text())['slot'] for p in metas})}")
    check("nothing periodic was written alongside",
          not list(CAP_OUT.glob("*_[0-9][0-9].json")),
          f"{[p.name for p in CAP_OUT.glob('*_[0-9][0-9].json')]}")

    for path in metas:
        meta = json.loads(path.read_text())
        stem = str(path)[: -len(".json")]
        iq_file = Path(stem + "_iq.u8")
        demod_file = Path(stem + "_demod.f32")
        png_file = Path(stem + ".png")
        tag = path.stem
        spans = meta["iq_spans"]
        check(f"{tag}: all four files are named in the json and exist",
              set(meta["files"]) == {"iq", "demod", "png", "json"}
              and all(Path(v).exists() for v in meta["files"].values()),
              str(meta["files"]))
        check(f"{tag}: the sizes agree with the counts",
              iq_file.stat().st_size == 2 * meta["iq_samples"]
              and demod_file.stat().st_size == 4 * meta["demod_samples"],
              f"{iq_file.stat().st_size} B iq, {demod_file.stat().st_size} B demod, "
              f"declared {meta['iq_samples']}/{meta['demod_samples']}")
        check(f"{tag}: the iq file decodes back to that many samples",
              sdr.u8_to_iq(iq_file.read_bytes(),
                           sdr.IQEncoding(meta["encoding"])).size == meta["iq_samples"])
        check(f"{tag}: the window is one continuous stretch",
              meta["continuous"] is True and not meta["iq_gaps"]
              and meta["iq_unplaced"] is False and meta["iq_unmatched"] is False,
              f"{len(spans)} span(s), gaps={meta['iq_gaps']}")
        check(f"{tag}: every span is placed, adjacent and identified",
              bool(spans)
              and all(s["first_sample"] >= 0 for s in spans)
              and all(b["first_sample"] == a["last_sample"]
                      for a, b in zip(spans, spans[1:]))
              and len({s["stream_id"] for s in spans}) == 1
              and len({s["tune_id"] for s in spans}) == 1,
              f"stream {sorted({s['stream_id'] for s in spans})}, "
              f"tune {sorted({s['tune_id'] for s in spans})}")
        check(f"{tag}: a predecessor is kept for every block that stepped over one",
              all(s["has_predecessor"] != s["starts_segment"] for s in spans)
              and sum(1 for s in spans if s["starts_segment"]) <= 1,
              f"{sum(1 for s in spans if s['has_predecessor'])} with a "
              f"predecessor, {sum(1 for s in spans if s['starts_segment'])} "
              f"starting a segment of {len(spans)}")
        check(f"{tag}: deliveries are distinct and increasing",
              [s["delivery"] for s in spans] == sorted({s["delivery"] for s in spans}))
        check(f"{tag}: the segment and the range agree with the frame",
              meta["frame"]["iq_first"] >= spans[0]["first_sample"]
              and meta["frame"]["iq_last"] <= spans[-1]["last_sample"]
              and meta["segment"] == meta["frame"]["segment"],
              f"frame [{meta['frame']['iq_first']}, {meta['frame']['iq_last']}) in "
              f"[{spans[0]['first_sample']}, {spans[-1]['last_sample']}), "
              f"segment {meta['segment']}")
        check(f"{tag}: the generation is recorded",
              meta["generation"] >= 0
              and meta["frame"]["generation"] == meta["generation"],
              f"generation {meta['generation']}")
        both = meta["sync"].get("both") or {}
        pos, neg = both.get("positive"), both.get("negative")
        check(f"{tag}: both sync polarities are in the capture",
              pos is not None and neg is not None
              and {pos["polarity"], neg["polarity"]} == {1, -1},
              f"pos raw={pos['raw_edges'] if pos else None} "
              f"neg raw={neg['raw_edges'] if neg else None}")
        if pos is None or neg is None:
            continue
        check(f"{tag}: the selected polarity is marked, and it is the one chosen",
              pos["selected"] != neg["selected"]
              and (pos if pos["selected"] else neg)["polarity"]
              == meta["sync"]["selected_polarity"],
              f"selected={meta['sync']['selected_polarity']}")
        for side, name in ((pos, "pos"), (neg, "neg")):
            check(f"{tag}: sync {name} records threshold, density and spacing",
                  side["threshold"] > 0 and side["density"] > 0
                  and 0 < side["spacing_min"] <= side["spacing_median"] <= side["spacing_max"],
                  f"thr={side['threshold']:.3f} dens={side['density']:.4f} "
                  f"spacing {side['spacing_min']:.0f}/{side['spacing_median']:.0f}/"
                  f"{side['spacing_max']:.0f}")
        check(f"{tag}: the sync record is the live one",
              meta["sync"]["threshold"] > 0 and meta["sync"]["line_samples"] > 0
              and abs(meta["sync"]["line_rate_hz"]
                      - 10_000_000.0 / meta["sync"]["line_samples"]) < 1e-6
              and meta["sync"]["pulses"] > 0,
              f"{meta['sync']['pulses']} pulses, "
              f"{meta['sync']['line_samples']:.1f} samples per line")
        check(f"{tag}: the line geometry is recorded",
              len(meta["line_positions"]) == meta["lines_used"] > 0
              and meta["lines_detected"] + meta["lines_from_grid"] == meta["lines_used"]
              and meta["frame"]["line_rate_hz"] > 0,
              f"used={meta['lines_used']} detected={meta['lines_detected']} "
              f"grid={meta['lines_from_grid']}")
        check(f"{tag}: the positions are inside the captured window",
              all(spans[0]["first_sample"] <= p <= spans[-1]["last_sample"]
                  for p in meta["line_positions"]))
        check(f"{tag}: the demod file is the float32 window, unfiltered",
              np.fromfile(demod_file, dtype=np.float32).size == meta["demod_samples"]
              and (meta["line_filter_hz"] is None
                   or 0 < float(meta["line_filter_hz"]) < 10_000_000.0),
              f"filter={meta['line_filter_hz']}")
        # The claim the whole capture rests on: replaying the demod file through
        # the same detector reproduces the decision that was made from it. If the
        # file were offset, truncated or filtered, this is where it would show.
        replay = dsp.detect_sync(
            np.fromfile(demod_file, dtype=np.float32), float(meta["sample_rate"]),
            polarity="auto",
        )
        check(f"{tag}: replaying the demod file reproduces the sync decision",
              int(replay.polarity) == int(meta["sync"]["selected_polarity"])
              and replay.line_samples == meta["sync"]["line_samples"]
              and abs(replay.quality - meta["sync"]["quality"]) < 1e-9
              and int(replay.pulse_starts.size) == int(meta["sync"]["pulses"]),
              f"{int(replay.pulse_starts.size)} pulses, "
              f"{replay.line_samples:.1f} samples/line, quality "
              f"{replay.quality:.4f} vs {meta['sync']['quality']:.4f}")
        try:
            pixels = png_pixels(png_file.read_bytes())
            png_ok = (pixels.shape == (meta["image"]["height"], meta["image"]["width"])
                      and meta["image"]["dtype"] == "uint8")
            png_detail = f"{pixels.shape} {pixels.min()}..{pixels.max()}"
        except Exception as exc:
            pixels, png_ok = None, False
            png_detail = f"{type(exc).__name__}: {exc}"
        check(f"{tag}: the png is a real 8-bit greyscale png of the right size",
              png_ok, png_detail)
        # The picture itself: the PNG has to be *that* publication's frame, not
        # merely a frame. Publication indices count from zero and every
        # publication is captured, so the index picks the picture directly.
        index = meta["publication_index"]
        check(f"{tag}: the png is the picture that was published",
              pixels is not None and 0 <= index < len(pictures)
              and np.array_equal(pixels, pictures[index]),
              f"publication {index} of {len(pictures)}"
              + ("" if pixels is not None else " (no png)"))

    recent = diag.pubcap_recent()
    check("the recent-publications summary is kept, metadata only",
          0 < len(recent) <= diag.PUBCAP_FILES
          and all(isinstance(s["line_positions"], int) and "image" not in s
                  for s in recent),
          f"publications {[s['publication_index'] for s in recent]}")

    # (g) One budget, shared. Captures spend it, and the periodic dump stops when
    # it is gone -- which is the claim the documentation makes about both.
    for stale in CAP_OUT.glob("*"):
        stale.unlink()
    os.environ[diag.ENV_DUMP_FILES] = "2"
    diag._DUMP.reset()
    diag.pubcap_reset()
    src = sdr.SimSource(video=True)
    w = video.DecodeWorker(src, width=160)
    feed_worker(w, long_iq, 175_000)
    jsons = sorted(CAP_OUT.glob("*_pub*.json"))
    check("the budget stops the captures", len(jsons) == 2,
          f"{len(jsons)} file sets of a budget of 2, spent={diag._DUMP._spent}")
    check("and it is the same counter the periodic dump reads",
          diag._DUMP._done is True and diag._PUB._done is True)
    src.ring.write(to_u8(long_iq[:175_000]), 0, 1)
    diag.dump_add("demod", np.zeros(200_000, dtype=np.float32), {})
    diag._DUMP._next = 0.0
    diag.dump_tick(src._dump_snapshot)
    periodic = [p for p in CAP_OUT.glob("*_[0-9][0-9].json")]
    check("so the periodic dump does not write past it either", not periodic,
          f"{[p.name for p in periodic]}")
    check("nothing on disk grew without bound",
          len(list(CAP_OUT.glob("*"))) <= 2 * 4,
          f"{len(list(CAP_OUT.glob('*')))} files")
    check("the publications are numbered from zero",
          sorted(json.loads(p.read_text())["publication_index"] for p in jsons) == [0, 1],
          f"{sorted(json.loads(p.read_text())['publication_index'] for p in jsons)}")

    for name in (diag.ENV_PUBCAP, diag.ENV_DUMP, diag.ENV_DUMP_FILES,
                 diag.ENV_DUMP_SAMPLES, diag.ENV_DUMP_FLUSH_S):
        os.environ.pop(name, None)
    diag._DUMP.reset()
    diag.pubcap_reset()


def main() -> int:
    print("TEMP DIAG instrumentation check")
    iq = capture_sim_iq()
    print(f"  captured {iq.size} simulator samples")
    case_no_behaviour_change(iq)
    case_counters_without_frames()
    case_drain_accounting()
    case_dump()
    case_iq_snapshot()
    case_sync_diagnostics(iq)
    case_publication_capture(iq)
    print("\n" + ("RESULT: FAIL -- " + ", ".join(FAILURES) if FAILURES else "RESULT: PASS"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())