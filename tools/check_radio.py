"""Is the radio there, and if so how fast is a sweep?

One command, because the question "is it plugged in?" and the question "how long
does a sweep take?" are the same question in practice: you only care about the
second one once the answer to the first is yes.

    python tools/check_radio.py              # diagnose, then sweep if possible
    python tools/check_radio.py --sweep-only # skip straight to the timing

Why this exists as a script rather than a flag on ``--check``. When a HackRF
does not enumerate, every layer above it reports the same unhelpful thing --
``hackrf_open() failed: HackRF not found (-5)`` -- and it is genuinely
ambiguous between three causes:

* the device is present but another program holds it
* the device is present and the bound driver is wrong
* the device is not on the bus at all

Only the third can be settled without the hardware in hand, and it is settled by
asking Windows rather than the radio. A ``Get-PnpDevice`` entry with
``IsPresent=False`` is a *ghost* left in the registry by a previous session; it
is not evidence that anything is plugged in now. That distinction has already
cost one false positive here.

The third cause is also the one that produces no event log entry whatsoever,
which is the giveaway: a device that attaches and fails to enumerate is loud,
and a device that never attaches is silent.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpv_rf import bands, sdr  # noqa: E402

#: HackRF One on the bus. The bootloader shows up as 1d50:6018 instead, and a
#: radio stuck in bootloader mode will not open either -- but for a different
#: reason, so it is reported separately rather than lumped in.
HACKRF_VID_PID = "VID_1D50&PID_6089"
BOOTLOADER_VID_PID = "VID_1D50&PID_6018"


def _pnp(*args: str) -> str:
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command", *args],
        capture_output=True, text=True, timeout=60,
    )
    return (r.stdout or "").strip()


def device_state() -> tuple[bool, bool, str]:
    """(present, in_bootloader, human summary) for the HackRF."""
    out = _pnp(
        "Get-PnpDevice | Where-Object { $_.InstanceId -match 'VID_1D50' } | "
        "ForEach-Object { "
        "  $p = (Get-PnpDeviceProperty -InstanceId $_.InstanceId "
        "        -KeyName 'DEVPKEY_Device_IsPresent' "
        "        -ErrorAction SilentlyContinue).Data; "
        "  \"$($_.InstanceId)|$p|$($_.Status)\" }"
    )
    present = False
    bootloader = False
    lines = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        inst, is_present, status = (p.strip() for p in line.split("|", 2))
        is_present = is_present.lower() == "true"
        if is_present:
            present = True
        if BOOTLOADER_VID_PID in inst and is_present:
            bootloader = True
        state = "present" if is_present else "GHOST (not plugged in now)"
        lines.append(f"    {inst.split(chr(92))[-1]}: {state}, status={status}")
    summary = "\n".join(lines) if lines else "    no VID_1D50 device at all"
    return present, bootloader, summary


def diagnose() -> bool:
    """Print why the radio is or is not usable. Returns True if it looks usable."""
    print("== is the radio on the bus? ==")
    present, bootloader, summary = device_state()
    print(summary)

    if bootloader and not present:
        print(
            "\n  The radio is enumerating as PID_6018, which is the BOOTLOADER,\n"
            "  not a HackRF One. It will not open until it is reflashed with the\n"
            "  Mayhem firmware, or reset so the normal firmware runs."
        )
        return False
    if not present:
        print(
            "\n  Windows cannot see the radio. No driver fix will help at this\n"
            "  level -- the device is not reaching the bus. Check, in order:\n"
            "\n"
            "    1. The cable. A charge-only USB-C cable carries no data and\n"
            "       looks exactly like 'not plugged in'. Try a different one.\n"
            "    2. The port. A HackRF draws about 250 mA; some USB-C ports will\n"
            "       not enumerate it. A USB-A port on the machine itself is the\n"
            "       safe choice, not a hub.\n"
            "    3. The radio's own LED.\n"
            "\n"
            "  The giveaway that it is this and not a driver problem: an event\n"
            "  log with nothing in it. A device that attaches and fails to\n"
            "  enumerate is loud; one that never attaches is silent."
        )
        return False

    print("\n  The radio is enumerating.")
    return True


def try_open() -> tuple[bool, str]:
    """Actually open it, which presence alone does not promise."""
    exe = sdr.find_hackrf_transfer()
    if not exe:
        return False, "hackrf_transfer not found (set HACKRF_TRANSFER)"
    out = Path(ROOT) / "_validate_out" / "radio_probe.u8"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [exe, "-r", str(out), "-f", "5802000000", "-s", "10000000",
             "-n", "400000", "-l", "32", "-g", "16"],
            capture_output=True, text=True, timeout=90,
        )
    except subprocess.TimeoutExpired:
        return False, "timed out after 90 s"
    text = f"{r.stdout or ''}{r.stderr or ''}"
    if "hackrf_open" in text and ("not found" in text or "(-5)" in text):
        got = out.stat().st_size if out.exists() else 0
        out.unlink(missing_ok=True)
        return False, (
            "the device is on the bus but hackrf_open failed -- another program "
            "is holding it (SDR#, Mayhem, a WebUSB tab), or the bound driver is "
            f"wrong. helper said: {text.strip().splitlines()[0][:90]}"
        )
    got = out.stat().st_size if out.exists() else 0
    if got < 100_000:
        out.unlink(missing_ok=True)
        return False, f"opened but produced only {got} bytes"
    out.unlink(missing_ok=True)
    return True, f"opened, {got:,} bytes at 10 MS/s"


def sweep(lo_hz: int, hi_hz: int) -> int:
    """Time a real sweep and report the per-hop cost. Returns a process exit code."""
    exe = sdr.find_hackrf_transfer()
    if not exe:
        print("hackrf_transfer not found")
        return 2
    print(f"\n== timing a real sweep of {bands.format_mhz(lo_hz)}-"
          f"{bands.format_mhz(hi_hz)} ==")
    src = sdr.HackrfSource(lo_hz, executable=exe)
    src.start()
    try:
        from fpv_rf import scan as scan_mod

        scanner = scan_mod.BandScanner(src)
        hops_done = 0

        def progress(message: str, frac: float) -> None:
            nonlocal hops_done
            hops_done += 1

        t0 = time.monotonic()
        res = scanner.scan(lo_hz=lo_hz, hi_hz=hi_hz, progress=progress)
        elapsed = time.monotonic() - t0
    finally:
        src.stop()

    hops = max(1, res.hops)
    print(f"  hops            : {res.hops}")
    print(f"  total           : {elapsed:.2f} s")
    print(f"  per hop         : {elapsed / hops * 1000.0:.0f} ms")
    print(f"  retunes         : {src.stats.tune_count}")
    print(f"  mean retune     : {src.stats.tune_mean_ms:.0f} ms")
    print(f"  open retries    : {src.stats.open_retries}")
    print(f"  dropped bytes   : {src.ring.dropped_bytes}")
    print(f"  result          : {res.describe()}")

    # The number that decides whether a target is reachable, stated plainly
    # rather than left for the reader to derive.
    span_mhz = (hi_hz - lo_hz) / 1e6
    per_hop_ms = elapsed / hops * 1000.0
    for target_s in (2.0, 1.0):
        hops_needed = span_mhz / 9.0
        best_ms = target_s * 1000.0 / hops_needed
        verdict = "reachable" if per_hop_ms <= best_ms else "NOT reachable"
        print(f"  {target_s:.0f} s target : needs {best_ms:.0f} ms/hop over "
              f"{hops_needed:.0f} hops -- {verdict} at {per_hop_ms:.0f} ms/hop")
    return 0 if res is not None else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep-only", action="store_true",
                    help="skip the diagnosis and go straight to timing")
    ap.add_argument("--region", default="us", help="region key for the span")
    args = ap.parse_args()

    if not args.sweep_only:
        if not diagnose():
            return 2
        ok, detail = try_open()
        print(f"\n== can it be opened? ==\n  {detail}")
        if not ok:
            return 2

    region = bands.Region(args.region) if hasattr(bands, "Region") else None
    lo, hi = bands.default_scan_span(region)
    return sweep(lo, hi)


if __name__ == "__main__":
    raise SystemExit(main())
