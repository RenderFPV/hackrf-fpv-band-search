"""Why the fast scan path sometimes declines, on a real HackRF.

The sweep engine is fast because it keeps the radio open across a whole sweep.
That requires a handover: ``hackrf_transfer`` has to be stopped before
``hackrf_sweep`` will open the device. On a HackRF One that handover is not
reliable, and this is what it looks like when you measure it.

The finding, so the next person does not have to repeat the afternoon:

* Stopping a ``hackrf_transfer`` that is streaming raw samples with ``-r -``
  leaves the device refusing the next open for a period that is not
  predictable. Measured anywhere from 0 s to over 30 s on this machine, and
  waiting longer does not reliably clear it.
* It is specifically the raw sample stream. The same tool, same gains, same
  stop, without ``-r -``, hands the radio over cleanly every time.
* Asking politely does not help. ``hackrf_sweep`` advertises "Stop with
  Ctrl-C", so ``hackrf_transfer`` plausibly handles it too -- it ignores
  Ctrl-C entirely and has to be killed.
* The device is not broken, and neither tool is at fault. The app reopens the
  radio fine afterwards, and the sweep tool works perfectly on a radio nobody
  has been streaming from.

So the app does not wait this out and does not pretend the handover is safe. It
tries the sweep, and if the tool cannot have the radio it falls back to the hop
walk, which drives the source it already holds. See ``fpv_rf/fastscan.py``.

Not part of the check suite, and not something to run casually: it deliberately
wedges the radio. ``tools/check_sweep.py`` measures the fast path's actual
speed with no handover involved, and is the check to run instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpv_rf import sdr, sweep  # noqa: E402

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CSV = Path(os.environ["TEMP"]) / "probe_handover.csv"

BASE = ["-f", "5800000000", "-s", "10000000"]
RAW = ["-r", "-"]


def handover(args: list[str], settle: float = 0.0) -> tuple[bool, float]:
    """Stream with ``args``, stop hard, wait ``settle``, then sweep.

    Returns (the sweep worked, how long it took).
    """
    exe = sdr.find_hackrf_transfer()
    if exe is None:
        raise SystemExit("hackrf_transfer not found")
    tool = sweep.runnable_tool()
    if tool is None:
        raise SystemExit(f"no sweep tool: {sweep.unavailable_reason()}")

    tx = subprocess.Popen([exe] + args, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL,
                          creationflags=CREATE_NO_WINDOW)
    time.sleep(1.2)                       # let it get the device
    tx.terminate()
    try:
        tx.wait(timeout=5)
    except Exception:
        tx.kill()
    if settle:
        time.sleep(settle)

    p = subprocess.Popen(
        [str(tool), "-f", "5800:5820", "-w", "1000000", "-N", "1", "-1",
         "-r", str(CSV)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=str(tool.parent), creationflags=CREATE_NO_WINDOW,
    )
    t0 = time.monotonic()
    try:
        p.wait(timeout=8)
    except Exception:
        p.kill()
        return False, time.monotonic() - t0
    worked = p.returncode == 0 and CSV.exists() and CSV.stat().st_size > 0
    return worked, time.monotonic() - t0


def row(label: str, args: list[str], settle: float = 0.0) -> bool:
    worked, el = handover(args, settle)
    print(f"  {label:32s} -> {'ok  ' if worked else 'no radio'}"
          f"   sweep took {el:5.2f}s")
    return worked


def main() -> int:
    print("Each row: start hackrf_transfer, stop it, then sweep 5800-5820.")
    print()
    print("Without the raw sample stream:")
    for _ in range(2):
        row("no -r", BASE + ["-a", "1"])
    print()
    print("With it:")
    for _ in range(3):
        row("-r -", RAW + BASE + ["-a", "1"])
    print()
    print("Same again after waiting, to show that waiting is not the answer:")
    row("-r -, after 2 s", RAW + BASE + ["-a", "1"], settle=2.0)
    row("-r -, after 8 s", RAW + BASE + ["-a", "1"], settle=8.0)
    print()
    print("And the tool itself, on a radio nobody was streaming from:")
    row("quiet radio", BASE + ["-a", "1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
