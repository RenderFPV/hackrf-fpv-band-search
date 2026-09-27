"""Where the optional real-hardware capture lives.

Several checks and diagnostics are more convincing against a real HackRF
recording than against the simulator, but a 38 MB capture of someone else's
band does not belong in this repository. So the path is resolved rather than
hardcoded, and every caller that uses it already handles "not found" by
reporting a skip instead of failing.

Resolution order:

1. ``$FPV_CAPTURE`` -- an explicit full path, or a folder containing
   ``r5_5802_g16.u8``.
2. ``samples/r5_5802_g16.u8`` inside the repository, if you have dropped one
   there. It is gitignored, so putting it there is safe.
3. The same filename anywhere under your home directory, found by a shallow
   search. This is what makes the original location -- a sibling project called
   ``hackrf-fpv-still-decoder`` -- keep working without this file containing a
   single absolute path or anybody's username.

If none of those find it, ``CAPTURE`` points at the repository-relative default
anyway. It simply will not exist, and that is the normal case for a fresh
clone: run ``python tools/record_sample.py`` to make one with your own radio.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CAPTURE_NAME = "r5_5802_g16.u8"

#: Where it would be if you kept it in the repository. Gitignored.
LOCAL = REPO / "samples" / CAPTURE_NAME

#: What to look for, relative to whatever the user pointed us at.
_ENV = "FPV_CAPTURE"


def _from_env() -> Path | None:
    raw = os.environ.get(_ENV)
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_dir():
        p = p / CAPTURE_NAME
    return p if p.exists() else None


def _from_home() -> Path | None:
    """A shallow search of the home directory, so no absolute path is baked in.

    One walk with a depth cutoff, not a walk per depth level: an unrestricted
    search of a home directory is slow enough to be noticeable and can descend
    into multi-gigabyte trees.
    """
    home = Path.home()
    max_depth = 3
    skip = {"AppData", "node_modules", ".git", ".venv", "venv",
            "site-packages", "OneDrive", "Downloads", "Pictures", "Music",
            "Movies"}
    try:
        for dirpath, dirnames, filenames in os.walk(home):
            here = Path(dirpath)
            if len(here.relative_to(home).parts) >= max_depth:
                dirnames[:] = []          # do not descend any further
                continue
            dirnames[:] = [d for d in dirnames if d not in skip]
            if CAPTURE_NAME in filenames:
                return here / CAPTURE_NAME
    except (OSError, ValueError):
        return None
    return None


def find_capture() -> Path | None:
    """The first capture that actually exists, or None if there isn't one."""
    for candidate in (_from_env(), LOCAL, _from_home()):
        if candidate is not None and candidate.exists():
            return candidate
    return None


#: Always a path, never None, so callers keep their existing ``.exists()``
#: skip logic unchanged. Unresolvable simply means "not present".
CAPTURE = find_capture() or LOCAL


#: There used to be a second resolver here, for a separate project called
#: ``hackrf-fpv-still-decoder`` that a handful of comparison scripts imported
#: from. Those scripts were removed, because a repository that needs a project
#: it does not contain is not self-contained, and nothing referenced it any
#: more. This file resolves one path now, and it resolves it in one place.
