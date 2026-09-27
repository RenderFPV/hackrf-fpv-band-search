"""Validate .github/workflows/checks.yml before it is pushed.

A malformed workflow is not caught locally by anything else, and it fails on
GitHub rather than here. This parses the YAML, then checks the things that
would only fail once the runner is already executing: that every script named
in a run step exists, that the steps install a requirements file that exists,
and that the hardware-only tools are not in the gate.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows" / "checks.yml"

#: Tools that need a real HackRF plugged in. They must never be in a run step.
HARDWARE_ONLY = {"check_hardware_scan", "diag_spurs", "diag_peak_bin"}

#: Import name -> distribution name on PyPI. They differ often enough that
#: comparing them directly produces a false "undeclared dependency" for PIL,
#: which is the whole thing this check exists to catch.
DIST = {"pil": "pillow", "numpy": "numpy", "scipy": "scipy", "pyside6": "pyside6"}

problems: list[str] = []


def main() -> int:
    raw = WF.read_text(encoding="utf-8")
    doc = yaml.safe_load(raw)
    print(f"parsed {WF.relative_to(ROOT)}")

    triggers = doc[True] if True in doc else doc["on"]
    print(f"  triggers  : {sorted(triggers)}")
    print(f"  jobs      : {list(doc['jobs'])}")

    job = doc["jobs"]["checks"]
    print(f"  os matrix : {job['strategy']['matrix']['os']}")
    print(f"  steps     : {len(job['steps'])}")

    if job["permissions"] if "permissions" in job else doc.get("permissions"):
        pass
    if doc.get("permissions") != {"contents": "read"}:
        problems.append("permissions should be exactly {'contents': 'read'}")

    # Every script a run step executes must exist. The steps loop over a shell
    # variable ("python tools/$check.py"), so the names appear in a `for check
    # in \` list rather than as literal paths, and both forms have to be read.
    executed: set[str] = set()
    for step in job["steps"]:
        run = step.get("run", "")
        if "tools/" not in run and "tools\\" not in run:
            continue
        executed |= set(re.findall(r"tools[/\\](\w+)\.py", run))
        executed |= set(re.findall(r"^\s*((?:check|diag|bench)_\w+)", run, re.M))
        executed |= set(re.findall(r"[\s\\]((?:check|diag|bench)_\w+)", run))
    # Names appearing only in comments document the hardware-only exclusions and
    # are not executed. Drop anything not reachable from a run step.
    in_run = set()
    for step in job["steps"]:
        run = step.get("run", "")
        in_run |= set(re.findall(r"tools[/\\](\w+)\.py", run))
        in_run |= set(re.findall(r"(?:check|diag|bench)_\w+", run))
    executed = in_run

    print(f"\n  scripts executed by run steps: {len(executed)}")
    for name in sorted(executed):
        if name in HARDWARE_ONLY:
            problems.append(f"run step executes {name}.py, which needs a real HackRF")
            print(f"    HARDWARE {name}.py  <-- must not be in CI")
        elif (ROOT / "tools" / f"{name}.py").exists():
            print(f"    ok        {name}.py")
        else:
            problems.append(f"workflow runs tools/{name}.py, which does not exist")
            print(f"    MISSING   {name}.py")

    documented = set(re.findall(r"(?:check|diag|bench)_\w+", raw))
    undocumented = documented - executed - HARDWARE_ONLY
    if undocumented:
        print(f"\n  named but never run and not hardware-only: {sorted(undocumented)}")

    for step in job["steps"]:
        run = step.get("run", "")
        for name in HARDWARE_ONLY:
            if re.search(rf"tools[/\\]{name}\.py", run):
                problems.append(
                    f"step {step.get('name', '?')!r} runs {name}.py, which needs hardware"
                )
    print(f"  hardware-only tools absent from run steps: "
          f"{'yes' if not any('needs a real HackRF' in p for p in problems) else 'NO'}")

    # The requirements files the workflow installs must exist.
    for req in re.findall(r"-r (\S+)", raw):
        if not (ROOT / req).exists():
            problems.append(f"workflow installs {req}, which does not exist")
        else:
            print(f"  installs  {req}  (exists)")

    # Every dependency any tools/ script imports must be declared somewhere.
    declared = set()
    for req in ("requirements.txt", "requirements-dev.txt"):
        for line in (ROOT / req).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-")):
                declared.add(re.split(r"[><=\[]", line)[0].strip().lower())
    imported: set[str] = set()
    for path in list((ROOT / "fpv_rf").glob("*.py")) + list((ROOT / "tools").glob("*.py")) + [ROOT / "main.py"]:
        for mod in re.findall(r"^\s*(?:import|from)\s+(\w+)", path.read_text(encoding="utf-8"), re.M):
            if mod.lower() in DIST:
                imported.add(mod.lower())
    print(f"\n  declared  : {sorted(declared)}")
    print(f"  imported  : {sorted(imported)}")
    undeclared = {DIST[m] for m in imported} - declared
    if undeclared:
        problems.append(f"imported but undeclared in any requirements file: {sorted(undeclared)}")
    else:
        print("  every third-party import is declared")

    print()
    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}")
        print(f"RESULT: FAIL ({len(problems)} problem(s))")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
