"""The two extra checks run_tests.py --production-ready runs once every test has passed.

Passing the tests shows the implementation is correct on the tests' inputs. Before a result is
called production-ready, two more things are checked, each against a file the run already has:

  defects   Every defect in decision_log.json must be closed, and closed honestly: either the user
            closed it, or the fix is proven by a test that passes the built implementation and
            fails a deliberately broken copy (the "mutant") that has the defect.
  audit     The auditor (an agent that looks for ways the implementation games the benchmark) must
            have looked at this exact implementation: synthesis/audit/completeness.json carries the
            digest of what it covered.

Each check returns 0 when satisfied, 1 when the implementation fails a test, and 2 when the
question cannot be answered (a missing file, a test that could not run). The delivery hook treats
2 like 1: nothing ships that could not be checked.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite  # noqa: E402  (also puts spec/ on sys.path)

# isort: split
from spec.checkpoint import audit_covers  # noqa: E402
from spec.findings import Findings, snapshot_path  # noqa: E402


def _err(msg: str) -> int:
    print(f"check_release: {msg}", file=sys.stderr)
    return 2


# ------------------------------------------------------------------------------------ defects


def defects(log: Path, tests: Path, impl: Path, interface: Optional[Path] = None) -> int:
    """No defect may be open, and no defect may have quietly been relabelled as less than a defect.
    A defect is closed either by the user (answered_by: human) or by a proven fix: its test passes
    the built implementation and fails the deliberately broken copy (mutant) that has the defect."""
    if not log.is_file():
        return _err(f"no decision log at {log}")
    try:
        raw = json.loads(log.read_text(encoding="utf-8"))
        findings = Findings(str(log)).all()
    except Exception as e:  # noqa: BLE001
        return _err(f"cannot read the decision log {log}: {e}")
    if not isinstance(raw, list):
        return _err(f"{log} is not a list of findings")
    if len(findings) != len(raw):
        return _err(
            f"{log} has {len(raw) - len(findings)} row(s) that do not parse; one could be an open defect"
        )

    # Severity is the field an escape goes through: relabel a defect 'advisory' and it leaves every
    # check at once. The findings CLI writes each row's first severity into the snapshot beside the
    # log as it writes the row, so the snapshot is the only record of what a severity was. Status is
    # the field that may change (a fix, or the user setting a finding aside); severity may not.
    snap_path = snapshot_path(log)
    now = {
        str(r.get("id")): str(r.get("severity", ""))
        for r in raw
        if isinstance(r, dict) and r.get("id")
    }
    was: Optional[Dict[str, str]] = None
    if snap_path.is_file():
        try:
            doc = json.loads(snap_path.read_text(encoding="utf-8"))
            if isinstance(doc, dict) and isinstance(doc.get("severities"), dict):
                was = doc["severities"]
        except (OSError, json.JSONDecodeError):
            was = None

    # No snapshot, no check: it is the only evidence of what each severity was, so its absence or
    # emptying is refused whether or not the log still holds rows. The findings CLI writes it beside
    # the log with every row, so a present log (even one emptied to []) without a snapshot, or with an
    # emptied {} one, was hand-written or had its snapshot erased to hide a defect -- emptying the log
    # and neutering the snapshot is otherwise the cheapest way past the check below. A run that
    # recorded nothing never writes a row, so a genuine run always reaches here with a non-empty one.
    if not was:
        return _err(
            f"no severity snapshot at {snap_path} beside {log}; the findings CLI writes it with "
            "every row, so the log was edited by hand or the snapshot was deleted, and a defect "
            "could have been erased or relabelled unnoticed"
        )

    # An emptied log skips the per-row check below, so consult the snapshot: if it still records a
    # defect, the rows were cleared to hide it. A snapshot naming no defect clears an emptied log.
    if not findings:
        wiped = [rid for rid, sev in was.items() if sev == "defect"]
        if wiped:
            print(
                f"RELEASE BLOCKED: {log} carries no findings but its snapshot still records "
                f"{len(wiped)} defect(s) ({', '.join(wiped)}); the log was cleared to hide them.",
                file=sys.stderr,
            )
            return 2
        print(f"DEFECTS PASSED: {log} has no findings.")
        return 0
    downgraded = [
        f"{rid}: was a defect, now {now.get(rid, 'absent')!r}"
        for rid, sev in was.items()
        if sev == "defect" and now.get(rid) != "defect"
    ]
    if downgraded:
        print(
            f"RELEASE BLOCKED: {len(downgraded)} defect(s) changed severity since the snapshot:\n  "
            + "\n  ".join(downgraded)
            + "\n  A severity never changes. A defect closes by a fix (findings set-status --status waived "
            "--who ai --test <test> --mutant <mutant>) or the user setting it aside (decisions <run> drop <row> --by human).",
            file=sys.stderr,
        )
        return 2

    open_ = [f for f in findings if f.is_open_defect]
    if open_:
        rows = "\n  ".join(f"{f.id} [{f.status}] {f.title}" for f in open_)
        print(f"RELEASE BLOCKED: {len(open_)} open defect(s) in {log}:\n  {rows}", file=sys.stderr)
        return 2

    closed = [f for f in findings if f.severity == "defect" and f.status == "waived"]
    by_user = [f for f in closed if f.answered_by == "human"]
    fixed = [f for f in closed if f.answered_by != "human"]
    if by_user:
        print("DEFECTS CLOSED BY THE USER: " + ", ".join(f.id for f in by_user))
    if not fixed:
        print(f"DEFECTS PASSED: no open defect in {log}.")
        return 0
    # A fix is proven by its test: it must pass the built implementation and fail the recorded
    # mutant, so an always-green test cannot be cited for any defect.
    unproven: List[str] = []
    for f in fixed:
        test = (f.test or "").strip()
        if not test:
            unproven.append(f"{f.id}: no test recorded for the fix")
            continue
        name = Path(test).name
        if not (tests / name).is_file():
            unproven.append(f"{f.id}: test {name} not found in {tests}")
            continue
        res = suite.run(tests, impl, interface, [name], timeout=suite.SLOW_SECS)
        if not res.passed:
            unproven.append(
                f"{f.id}: test {name} fails the built implementation ({res.why()}); the fix is not in the build"
            )
            continue
        mutant = (f.mutant or "").strip()
        if not mutant:
            unproven.append(
                f"{f.id}: test {name} passes, but no mutant is recorded to show it catches this defect"
            )
            continue
        mp = Path(mutant) if os.path.isabs(mutant) else log.parent / mutant  # relative to the run
        if not mp.exists():
            unproven.append(f"{f.id}: recorded mutant {mutant} not found (looked in {mp})")
            continue
        mres = suite.run(tests, mp, interface, [name], timeout=suite.SLOW_SECS)
        if not mres.failed:
            why = "it timed out" if mres.timed_out else "the test passed it"
            unproven.append(f"{f.id}: test {name} does not catch mutant {mutant} ({why})")
    if unproven:
        print(
            "RELEASE BLOCKED: closed defect(s) whose fix is not proven in the build:\n  "
            + "\n  ".join(unproven),
            file=sys.stderr,
        )
        return 2
    print(
        f"DEFECTS PASSED: {len(fixed)} fixed defect(s), each test passes the build and catches its mutant."
    )
    return 0


# -------------------------------------------------------------------------------------- audit


def audit(artifact: Path, audit_dir: Path) -> int:
    """The auditor records what it looked at with `spec.checkpoint stamp-audit`, which
    writes completeness.json with the implementation's digest and a findings list (empty when it
    found nothing). The stamp must cover the bytes being delivered: synthesis/impl/ itself, or the
    published copy of it (`best/artifact/`, the same bytes with the interface folded in) that
    `run finish` checks. `audit_dir` is the run's synthesis/audit/, so the run is its grandparent.
    """
    stamp = audit_dir / "completeness.json"
    why = audit_covers(stamp, artifact, audit_dir.parent.parent)
    if why:
        return _err(f"audit stamp {why}; run the auditor on this candidate and stamp it")
    print("AUDIT PASSED: the auditor covered the delivered implementation.")
    return 0
