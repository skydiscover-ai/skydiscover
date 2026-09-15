#!/usr/bin/env python3
"""Run the kept tests against a candidate implementation. The delivery hook calls this.

    run_tests.py --run <run dir> [--impl <path>] [--production-ready]
    run_tests.py --suite <dir> --impl <path> [--interface <dir>] [--test <file>...]

The suite is a directory with a test.sh (suite.py); this runs it with SKYDISCOVER_IMPL pointing at
the candidate and reads the exit code. With --run, everything is found inside the run directory
(spec/paths.py): the suite under synthesis/tests/, the interface, and the candidate under
synthesis/impl/ (or --impl when several sit there). --production-ready adds the release checks
(check_release.py).

Exit 0 every test passed, 1 the implementation fails a test, 2 the tests could not be run (no
implementation, no test.sh, out of time). The hook treats 2 like 1: nothing ships unchecked.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite  # noqa: E402  (also puts spec/ on sys.path)

# isort: split
import check_release  # noqa: E402
from spec.paths import Run  # noqa: E402


def run_suite(tests: Path, impl: Path, interface: Optional[Path], names: Sequence[str] = ()) -> int:
    """Every test must pass the implementation; `names` narrows the run to those test files."""
    if not impl.exists():
        print(f"run_tests: implementation not found: {impl}", file=sys.stderr)
        return 2
    if not suite.script(tests).is_file():
        print(f"run_tests: no {suite.TEST_SCRIPT} in {tests}; nothing to run", file=sys.stderr)
        return 2
    missing = [n for n in names if not (tests / n).is_file()]
    if missing:
        print(f"run_tests: no such test in {tests}: {', '.join(missing)}", file=sys.stderr)
        return 2
    res = suite.run(tests, impl, interface, names, timeout=suite.SLOW_SECS)
    what = " ".join([suite.TEST_SCRIPT, *names])
    if res.passed:
        print(f"TESTS PASSED: {what} exited 0 for {impl}.")
        return 0
    suite.report(res)
    if res.timed_out:
        print(f"run_tests: {what} did not finish in {suite.SLOW_SECS}s", file=sys.stderr)
        return 2
    print(f"TESTS FAILED: {what} {res.why()} for {impl}.", file=sys.stderr)
    return 1


def _entry(run: Run, named: str, ap: argparse.ArgumentParser) -> Path:
    entry = run.entry_impl(named)
    if entry is None:
        ap.error(
            f"--run: no implementation to test ({named or run.impl}); "
            "put the candidate under synthesis/impl/ or name it with --impl"
        )
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], prog="run_tests.py")
    ap.add_argument(
        "--run", default="", help="the run directory; everything else is derived from its layout"
    )
    ap.add_argument(
        "--impl", default="", help="the implementation (required with --run only if several)"
    )
    ap.add_argument("--interface", default="", help="the interface directory")
    ap.add_argument("--suite", default="", help="the directory holding test.sh and the tests")
    ap.add_argument(
        "--test",
        action="append",
        default=[],
        help="run only this test file of the suite (repeatable); default: every test",
    )
    ap.add_argument(
        "--production-ready",
        action="store_true",
        help="with --run: also require no open defect and a current audit",
    )
    ap.add_argument(
        "--check-defects", action="store_true", help="also check recorded fixes and open defects"
    )
    ap.add_argument(
        "--wall-secs",
        type=int,
        default=0,
        help="give up with exit 2 after this many seconds in total (0: no limit); the delivery "
        "hook sets it below its own timeout so a slow check is never mistaken for a pass",
    )
    args = ap.parse_args(argv)
    if args.check_defects and not args.run:
        ap.error("--check-defects needs --run")
    if args.production_ready and not args.run:
        ap.error("--production-ready needs --run")
    if args.wall_secs > 0:
        _stop_after(args.wall_secs)

    run: Optional[Run] = None
    if args.run:
        run = Run(args.run)
        # Tests that need the run's own tree (its frozen benchmark, its reference) find it through
        # SKYDISCOVER_RUN; with --run the caller need not export it.
        os.environ.setdefault("SKYDISCOVER_RUN", str(run.path.resolve()))
        tests = Path(args.suite) if args.suite else run.tests
        interface: Optional[Path] = Path(args.interface) if args.interface else run.interface
        impl = _entry(run, args.impl, ap)
    else:
        if not (args.suite and args.impl):
            ap.error("--run, or --suite with --impl")
        tests, impl = Path(args.suite), Path(args.impl)
        interface = Path(args.interface) if args.interface else None

    rc = run_suite(tests, impl, interface, args.test)
    if not rc and run is not None and args.check_defects:
        rc = _defects(run, tests, impl, interface)
    if rc or not args.production_ready:
        return rc
    assert run is not None
    if not run.decision_log.exists():
        print(f"RELEASE BLOCKED: no decision log at {run.decision_log}", file=sys.stderr)
        return 2
    # Both run, so one pass reports every blocker instead of one per attempt.
    return max(
        check_release.defects(run.decision_log, tests, impl, interface),
        check_release.audit(_audited(run, args.impl), run.audit),
    )


def _defects(run: Run, tests: Path, impl: Path, interface: Optional[Path]) -> int:
    if not (run.decision_log.exists() or run.severity.exists()):
        return 0
    return check_release.defects(run.decision_log, tests, impl, interface)


def _stop_after(secs: int) -> None:
    """Exit 2 when the whole check outlives `secs`; the subprocess in flight is killed with it."""

    def expired(_signum, _frame):
        print(
            f"run_tests: not finished after {secs}s; the check is incomplete, nothing is shipped "
            "unchecked (raise the hook timeout and SKYDISCOVER_DELIVERY_SECS for a longer suite)",
            file=sys.stderr,
        )
        sys.exit(2)

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(secs)


def _audited(run: Run, impl: str) -> Path:
    """The tree the audit stamp must cover: synthesis/impl/ when the candidate lives there (the
    stamp covers the whole directory), otherwise exactly the bytes named by --impl."""
    if not impl:
        return run.impl
    candidate = Path(impl).resolve()
    root = run.impl.resolve()
    return run.impl if candidate == root or root in candidate.parents else Path(impl)


if __name__ == "__main__":
    raise SystemExit(main())
