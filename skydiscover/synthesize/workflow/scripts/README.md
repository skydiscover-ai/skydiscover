# Workflow Scripts

The agents and the delivery hook run these; you do not. `validate_test.py` and `run_tests.py` have
`--help`; the rest are libraries they import.

| Script | What it does |
|---|---|
| `validate_test.py` | keep or reject one new test: it must pass the reference implementation and fail a broken one; `--seed` for a test the task ships: reference only |
| `run_tests.py` | run the suite against a candidate; `--production-ready` adds the release checks |
| `check_release.py` | the release checks: no defect is open, the auditor saw this code |
| `suite.py` | run a suite's `test.sh` against one implementation |
| `kb/` | `kbtool.py` reads and checks the knowledge base wiki ([README](kb/README.md)) |
| `unattended/` | `supervisor.sh <tmux-session> <run-dir>` keeps a multi-hour Claude Code session going; optional |

Exit codes: 0 pass, 1 fail, 2 could not decide (never a pass).

## The suite

A suite is a directory with a `test.sh` at its top and one file per test beside it, in any
language. The scripts run `bash test.sh` (every test) or `bash test.sh <file>...` (only those)
from that directory with:

| Variable | Meaning |
|---|---|
| `SKYDISCOVER_IMPL` | the implementation under test: a file or a directory |
| `SKYDISCOVER_INTERFACE` | the interface directory, when the run has one |
| `SKYDISCOVER_RUN` | the run directory, when there is one |

and read the exit code: 0, every test passed; anything else, the implementation failed. How the
tests are built and run is `test.sh`'s business. `examples/single-machine-kvstore/evaluator/tests/`
is one for C++ tests compiled against a `.cc` implementation.

## Settings

| Setting | Meaning |
|---|---|
| `SKYDISCOVER_TEST_MAX_SECS` | how long one test may take on the reference at validation (default 30) |
| `SKYDISCOVER_SLOW_SECS` | time budget for one suite run: the final checks at `run finish`, the release checks (default 600) |
