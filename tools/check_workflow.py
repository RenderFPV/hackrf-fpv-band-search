"""Validate .github/workflows/checks.yml before it is pushed.

A malformed workflow is not caught locally by anything else, and it fails on
GitHub rather than here. This parses the YAML, then checks the things that
would only fail once the runner is already executing: that every script named
in a run step exists, that the steps install a requirements file that exists,
that no hardware-only tool is in the gate, and that no step containing
bash-only syntax relies on the default shell on a Windows matrix leg.

It also checks the declared dependencies against what the code actually
imports, which is not a self-check: it is aimed at the checks, several of which
used Pillow for years without declaring it.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - the interesting path
    sys.stderr.write(
        "check_workflow.py needs PyYAML to parse the workflow.\n"
        "It is declared in requirements-dev.txt, not requirements.txt, because\n"
        "the application does not use it:\n"
        "\n"
        "    pip install -r requirements-dev.txt\n"
    )
    raise SystemExit(2) from None

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows" / "checks.yml"

#: Tools that need a real HackRF plugged in. They must never be in a run step.
HARDWARE_ONLY = {"check_hardware_scan", "diag_spurs", "diag_peak_bin"}

#: Import name -> distribution name, for the cases where they differ. Anything
#: not listed is assumed to be its own name in lower case, which is right for
#: numpy, scipy and pyside6. This map used to be the *whole* check, listing the
#: four packages that happened to be imported when it was written. That is how
#: PyYAML got through: it was imported by this file, declared nowhere, and a
#: list written from memory had no reason to contain it.
RENAMES = {"pil": "pillow", "yaml": "pyyaml", "cv2": "opencv-python",
           "sklearn": "scikit-learn", "bs4": "beautifulsoup4", "attr": "attrs"}

#: Constructs that only parse under bash. A windows runner runs `run:` with pwsh
#: unless the step says `shell: bash`, and a bash loop then dies at *parse* time
#: -- before executing anything -- so the job fails with a message that points
#: at the workflow rather than at the code. That is exactly what happened to the
#: first version of this workflow: Ubuntu passed, Windows reported
#: "Missing opening '(' after keyword 'for'".
BASH_ONLY = (
    (re.compile(r"^\s*set\s+-[a-z]*o\b", re.M), "set -o"),
    (re.compile(r"^\s*for\s+\w+\s+in\b", re.M), "a for-in loop"),
    (re.compile(r"^\s*do\s*$", re.M), "a do/done loop"),
    (re.compile(r"^\s*done\s*$", re.M), "a do/done loop"),
    (re.compile(r"\\\s*$", re.M), "a backslash line continuation"),
    (re.compile(r"\$\{[A-Za-z_]"), "${VAR} expansion"),
)

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

    # A step whose body only parses under bash must say so, or it breaks on any
    # matrix leg that defaults to pwsh. Steps restricted to Linux are exempt:
    # they never reach a Windows runner.
    matrix_os = job.get("strategy", {}).get("matrix", {}).get("os", [])
    windows_leg = any("windows" in str(o) for o in matrix_os)
    print(f"\n  windows leg: {windows_leg}")
    for step in job["steps"]:
        run = step.get("run", "")
        if not run:
            continue
        cond = str(step.get("if", ""))
        linux_only = "Linux" in cond
        shell = step.get("shell", "(default)")
        found = sorted({label for rx, label in BASH_ONLY if rx.search(run)})
        name = step.get("name", "?")
        if not found:
            print(f"    ok        {name}: no bash-only syntax")
        elif linux_only:
            print(f"    exempt    {name}: {', '.join(found)} (Linux-only step)")
        elif shell == "bash":
            print(f"    ok        {name}: {', '.join(found)} but shell: bash is set")
        else:
            problems.append(
                f"step {name!r} uses {', '.join(found)} but runs with {shell!r}; "
                f"a Windows runner would fail to parse it. Add `shell: bash`."
            )
            print(f"    PROBLEM   {name}: {', '.join(found)} with {shell!r}")

    # Every dependency any tools/ script imports must be declared somewhere.
    declared = set()
    for req in ("requirements.txt", "requirements-dev.txt"):
        for line in (ROOT / req).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-")):
                declared.add(re.split(r"[><=\[]", line)[0].strip().lower())
    # Every import in the tree, classified as stdlib, local, or third-party by
    # asking the interpreter rather than by remembering a list. sys.
    # stdlib_module_names is the authoritative answer and it grows with Python,
    # so this does not rot.
    #
    # Parsed with ast, not a regex. A regex over `^\s*(import|from)\s+(\w+)`
    # also matches English: alerts.py's module docstring contains the sentence
    # "from a drone that was transmitting two minutes ago is itself
    # information", and that reported a third-party package called `a`.
    local = {"fpv_rf", "tools", "main", "build_exe"}
    for path in (ROOT / "tools").glob("*.py"):
        local.add(path.stem)
    third_party: dict[str, set[str]] = {}
    sources = (list((ROOT / "fpv_rf").glob("*.py"))
               + list((ROOT / "tools").glob("*.py"))
               + [ROOT / "main.py", ROOT / "build_exe.py"])
    for path in sources:
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        mods: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import, i.e. this repository's own code
                if node.level == 0 and node.module:
                    mods.add(node.module.split(".")[0])
        for mod in mods:
            low = mod.lower()
            if low in sys.stdlib_module_names or low in local or low == "__future__":
                continue
            third_party.setdefault(low, set()).add(rel)
    imported = sorted(third_party)
    print(f"\n  declared  : {sorted(declared)}")
    print(f"  imported  : {imported}")
    for mod in imported:
        users = sorted(third_party[mod])
        shown = ", ".join(users[:3]) + (f" (+{len(users)-3} more)" if len(users) > 3 else "")
        dist = RENAMES.get(mod, mod)
        mark = "ok      " if dist in declared else "PROBLEM "
        print(f"    {mark} {mod:<10} -> {dist:<10} {shown}")
    undeclared = {RENAMES.get(m, m) for m in imported} - declared
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
