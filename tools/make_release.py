"""Package the built app and publish it as a GitHub release asset.

The packaged app is a 212 MB folder, which does not belong in git history: a
binary that size would be fetched on every clone forever, and it cannot be
reviewed in a diff. So the source lives in the repository and the build is
published as a release asset, which is what a visitor downloads.

This is the second piece of the release, after ``build_exe.py``. Nothing here
rebuilds: it takes the already-verified ``dist/FPV-RF`` folder, zips it, checksums
it, and attaches it to a release. Run it after ``build_exe.py`` has passed its own
``--selftest``.

    python tools/make_release.py --dry-run     # package and checksum, publish nothing
    python tools/make_release.py --tag v1.0.0   # package, checksum, and publish

Why a zip and not the raw folder: GitHub release assets are single files, and
because the build is a folder rather than one executable, that folder has to be
compressed to be attachable at all. Everything lands under a single ``FPV-RF/``
directory inside the archive, so extracting it does not scatter 300 files into
the Downloads folder.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "FPV-RF"
RELEASES = ROOT / "dist" / "releases"

#: Directory name inside the archive, so extraction produces one clean folder.
INNER = "FPV-RF"

#: Runtime debris that must not be published. The log is written by the app
#: itself on every launch, so a build that has been run once always has one.
EXCLUDE_NAMES = {"fpv-rf.log"}
EXCLUDE_SUFFIXES = {".log", ".pdb", ".pyc"}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def collect(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name in EXCLUDE_NAMES or p.suffix.lower() in EXCLUDE_SUFFIXES:
            continue
        out.append(p)
    return out


def build_zip(files: list[Path], root: Path, dest: Path) -> None:
    print(f"compressing {len(files)} files -> {dest.name}")
    # ZIP_DEFLATED: the payload is 123 .pyd plus 56 .dll, and DLLs compress
    # poorly, so this saves far less than the naive "half the size" expectation
    # and costs real time. It is still the right choice; storing would be
    # barely smaller and universally readable either way.
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for i, p in enumerate(files, 1):
            z.write(p, f"{INNER}/{p.relative_to(root).as_posix()}")
            if i % 60 == 0:
                print(f"  {i}/{len(files)}")
    print(f"  wrote {dest.name}  {human(dest.stat().st_size)}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", help="release tag, e.g. v1.0.0")
    ap.add_argument("--notes-file", help="release notes; defaults to docs/releases/<tag>.md")
    ap.add_argument("--dry-run", action="store_true",
                    help="package and checksum, but do not publish")
    args = ap.parse_args()

    exe = DIST / "FPV-RF.exe"
    if not exe.exists():
        print(f"no build at {DIST}. Run:  python build_exe.py")
        return 2
    if not (DIST / "hackrf_transfer.exe").exists():
        print("hackrf_transfer.exe is missing from the build.")
        print("The app looks for it beside the executable before PATH, so a build")
        print("without it silently falls back to whatever else is installed.")
        return 2
    if not (DIST / "hackrf_sweep.exe").exists():
        # Same reasoning, and the reason this is a gate rather than a warning: a
        # build without the sweep tool still works, searches 5650-5950 MHz in
        # 12.5 s instead of 0.14 s, and reports nothing wrong. Publishing that
        # as a release is how a fast feature ships looking absent.
        print("hackrf_sweep.exe is missing from the build.")
        print("Without it every user gets the 12.5 s hop walk and no way to know")
        print("why. The app is not broken, which is the problem: it is quietly slow.")
        return 2

    RELEASES.mkdir(parents=True, exist_ok=True)
    tag = args.tag or "v0.0.0-dev"
    stem = f"FPV-RF-{tag}-windows-x64"
    zip_path = RELEASES / f"{stem}.zip"

    files = collect(DIST)
    raw = sum(p.stat().st_size for p in files)
    print(f"source   : {DIST}")
    print(f"  {len(files)} files, {human(raw)} uncompressed")
    build_zip(files, DIST, zip_path)

    digest = sha256(zip_path)
    sums = RELEASES / f"{stem}.zip.sha256"
    # The bare-digest format is what sha256sum -c and friends expect.
    sums.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    print(f"\nsha256   : {digest}")
    print(f"wrote    : {sums.name}")

    if args.dry_run or not args.tag:
        print("\n--dry-run: nothing published.")
        if not args.tag:
            print("Pass --tag v1.0.0 to publish.")
        return 0

    print(f"\npublishing {tag} ...")
    # Release notes are versioned in the repository rather than kept beside the
    # build output, so they are reviewable in a diff and survive dist/ being
    # wiped. docs/releases/<tag>.md, or --notes-file.
    notes_path = Path(args.notes_file) if args.notes_file else (
        ROOT / "docs" / "releases" / f"{args.tag}.md")
    body = ""
    if notes_path.exists():
        body = notes_path.read_text(encoding="utf-8").strip()
        print(f"  notes    : {notes_path.relative_to(ROOT).as_posix()}")
    else:
        print(f"  notes    : none found at {notes_path}, using a placeholder")

    cmd = ["gh", "release", "create", args.tag, str(zip_path), str(sums),
           "--title", f"FPV RF {args.tag}", "--target", "main",
           "--notes", body or "See the README."]
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print("\ngh release create failed.")
        print("If the error was 'HTTP 403: Resource not accessible by", file=sys.stderr)
        print("integration', the cause is the kind of token, not the repository.", file=sys.stderr)
        print("", file=sys.stderr)
        print("  A ghu_ token is a GitHub App user-to-server token. It is limited", file=sys.stderr)
        print("  to what that app's installation permits, and the CLI's app cannot", file=sys.stderr)
        print("  create releases. A classic gho_ or ghp_ token with repo scope can.", file=sys.stderr)
        print("", file=sys.stderr)
        print("  The archive is already built and checksummed, so nothing above", file=sys.stderr)
        print("  this line needs redoing. Publish it with:", file=sys.stderr)
        print("", file=sys.stderr)
        print("    $env:GH_TOKEN = (git credential fill <<<", file=sys.stderr)
        print("      'protocol=https^`n^`nhost=github.com^`n^`n' |", file=sys.stderr)
        print("      Select-String '^password=') -replace '^password=',''", file=sys.stderr)
        print("    gh release create %s %s %s --title 'FPV RF %s' --target main \\" % (
            args.tag, zip_path.name, sums.name, args.tag), file=sys.stderr)
        print("      --notes-file docs/releases/%s.md" % args.tag, file=sys.stderr)
        return r.returncode
    print(f"\npublished {args.tag}")
    print(f"  https://github.com/RenderFPV/hackrf-fpv-band-search/releases/tag/{args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
