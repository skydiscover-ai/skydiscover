#!/usr/bin/env python3
"""Decide whether a new test is kept.

    validate_test.py --test <suite>/<file> --reference <impl> --mutant <impl> [--mutant ...]
    validate_test.py --test <suite>/<file> --reference <impl> --seed

The test is run through its suite's test.sh (`bash test.sh <file>`, see suite.py). It is kept if it
passes the trusted reference and fails at least one mutant, a copy of the reference broken in one
way. A test the task shipped (--seed) needs no mutant; a person already vouched that it
discriminates. Exit 0 keep, 1 reject, 2 could not decide (the reference itself failed here, or
nothing could run); a 2 is never a pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], prog="validate_test.py")
    ap.add_argument("--test", required=True, help="the test file, inside its suite directory")
    ap.add_argument("--reference", required=True, help="the trusted reference implementation")
    ap.add_argument(
        "--mutant", action="append", default=[], help="a deliberately broken copy (repeatable)"
    )
    ap.add_argument(
        "--seed",
        action="store_true",
        help="the test came with the task; check it against the reference only (no --mutant)",
    )
    ap.add_argument("--interface", default="", help="the interface directory, if the run has one")
    ap.add_argument(
        "--timeout",
        type=int,
        default=0,
        help=f"seconds one run may take (default $SKYDISCOVER_TEST_MAX_SECS, {suite.TEST_MAX_SECS})",
    )
    args = ap.parse_args(argv)

    if args.seed and args.mutant:
        print(
            "validate_test: --seed takes no --mutant; the task vouches for the test",
            file=sys.stderr,
        )
        return 2
    if not args.seed and not args.mutant:
        print(
            "validate_test: need at least one --mutant to show the test discriminates "
            "(or --seed for a test the task shipped)",
            file=sys.stderr,
        )
        return 2
    test = Path(args.test)
    if not test.is_file():
        print(f"validate_test: no such test: {test}", file=sys.stderr)
        return 2
    tests_dir, name = test.parent, test.name
    if not suite.script(tests_dir).is_file():
        print(
            f"validate_test: no {suite.TEST_SCRIPT} beside {name} in {tests_dir}", file=sys.stderr
        )
        return 2
    interface = Path(args.interface) if args.interface else None
    timeout = args.timeout or suite.TEST_MAX_SECS

    ref = suite.run(tests_dir, Path(args.reference), interface, [name], timeout=timeout)
    if ref.timed_out:
        print(
            f"REJECT: {name} took over {timeout}s on the reference; a test runs every iteration, "
            "so it must be quick on a plain correct implementation.",
            file=sys.stderr,
        )
        return 1
    if not ref.passed:
        print(
            f"CANNOT VALIDATE: the reference {ref.why()} on {name}; a test must pass a correct "
            f"implementation before it can reject anything.\n{suite.tail(ref.log, 800)}",
            file=sys.stderr,
        )
        return 2

    if args.seed:
        print(f"SEED TEST: {name} passes the reference in {ref.secs:.1f}s; kept on the task's word")
        return 0

    # A mutant that only times out is not caught: a hang cannot be told from a slow machine.
    caught = []
    for m in args.mutant:
        res = suite.run(tests_dir, Path(m), interface, [name], timeout=timeout)
        if res.failed:
            caught.append(Path(m).name)
    if not caught:
        print(
            f"REJECT: {name} passes the reference but catches no mutant; it does not discriminate.",
            file=sys.stderr,
        )
        return 1
    print(
        f"SOUND TEST: {name} passes the reference, catches {len(caught)}/{len(args.mutant)} "
        f"mutant(s): {', '.join(caught)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
