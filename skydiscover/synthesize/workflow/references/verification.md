# Verification

How the Evaluator decides. Correctness is a prerequisite for scoring: a model's opinion, a code
review, or a benchmark number never substitutes for an executable check.

## The Test Suite

The suite is the directory `<run>/synthesis/tests/`. It holds one file per test, in any language,
and a `test.sh` that knows how to build and run them:

```bash
bash test.sh              # run every test
bash test.sh <file>...    # run only the named tests
```

The harness sets `SKYDISCOVER_IMPL` to the implementation under test (a file or a directory) and
`SKYDISCOVER_INTERFACE` to the interface directory, runs `test.sh` from the suite directory, and
reads the exit code: 0 means every test passed, anything else means the implementation failed.
That is the whole contract. How a test is compiled, imported, or launched is `test.sh`'s business;
the evaluator writes it with the first test and extends it as the suite grows. A task that ships
tests ships its own `test.sh` beside them.

## Correctness Tests

A test checks behavior the interface promises. It is kept only if it:

1. passes the **trusted reference**, a simple and obviously correct implementation;
2. fails at least one **mutant**, a copy of the reference deliberately broken in one way; and
3. `workflow/scripts/validate_test.py` confirms both.

A test the task ships (`task.md` points at it) skips the mutant: run `validate_test.py --seed`
and it is kept if it passes the reference. A person already vouched that it discriminates.

The coding agent never writes the trusted reference. An evaluator in correctness mode writes it from the resolved
specification, so the implementation being optimized never also defines what "correct" means.

Before writing a test, reuse a test the domain's knowledge base already holds for the same
requirement (`<kb>/tests/`). Even matching IDs need a meaning check and fresh validation:

```bash
python3 -m skydiscover.synthesize.spec.kept_tests lookup "<domain>" --props <id...>
```

For each uncovered property, prefer a real test collected during discovery. The **finder** of a
requirement (the spec-builder, the auditor in a final review mode) supplies the failure
condition, the driving input or injection seam, and the oracle. The evaluator (correctness mode) writes
the tests for spec-derived and review-derived requirements; the auditor writes the
tests for the reward hacks it finds.

Every kept test, `<id>.<ext>`, named after the property it checks, drives the public interface
into one failure condition and asserts one property. A **fault-injection test** is a test whose failure condition only appears
under a fault the environment injects (a crash between two writes, a dropped message, a full disk,
a slow peer): it drives that fault through a public or documented injection seam, never by
editing the implementation's files, and asserts the property still holds. It is the test to write
when the trusted reference shares the failure, since a plain test against that reference would
then fail validation. An **operating-point test** is either kind run at the load the score was
measured at (the requirements card's `operating_point`); every suite has at least one.

### Authoring Rules

- Test only behavior the public interface promises, never an internal file layout or mechanism.
  A test may create its own fixtures under a scratch directory; it may not open the
  implementation's files or reach into its private state. A test that does would fail a correct
  implementation built differently.
- Assert exactly the resolved requirement. Do not add ordering, timing, capacity, concurrency, or
  lifecycle assumptions the contract leaves open.
- For resource bounds, use an independent observable, not the implementation's own counters.
- A test depends only on `$SKYDISCOVER_IMPL`, `$SKYDISCOVER_INTERFACE`, the suite directory, and
  fixtures it creates itself; never on the project tree or on the test file's own location. The
  suite is copied to `best/tests/` at delivery and run from there, so a test that imports a
  project file by relative path fails the final check.
- Keep tests deterministic, bounded by work rather than wall-clock time, and under
  `SKYDISCOVER_TEST_MAX_SECS` (`validate_test.py` rejects a test that exceeds it).
- For a fault requirement, drive a documented public or injection seam. Never modify
  implementation-owned files from a test.
- If a requirement cannot be tested through an observable interface, record it as advisory with a
  concrete explanation instead of inventing a test.

### Checking a Test

```bash
python3 <scripts>/validate_test.py --test <run>/synthesis/tests/<file> \
  --reference <reference> --mutant <broken-impl> [--interface <interface-dir>]
```

It runs `test.sh <file>` against the reference, then against each mutant. For a test the task
ships, `--seed` replaces `--mutant`. Exit 0 keeps the test; 1 rejects it; 2 means the reference or
the environment is broken.

Before saving, check that a correct implementation cannot fail the test through a free choice, a
race, a timing assumption, or an unrelated assertion; loosen an over-strict test while keeping its
ability to catch the mutant.

The delivery hook runs every kept test against every delivered candidate. `run finish` copies
the final suite into the knowledge base (`<kb>/tests/`) for later runs.

## Operating-Point Coverage

Small tests can miss the code paths the headline configuration uses. If the requirements card
declares an `operating_point`, `{"params": {"<name>": {"min"|"max"|"eq": N}, ...}}` (a bare number
as a param means at least that), the suite needs at least one test that drives every declared
parameter onto the scored side of its bound and exercises the scored code path there. For native
code, run it under a sanitizer (a compiler mode that reports memory errors, undefined behavior, or
data races at run time) as well; a concurrent native target needs a completed race-detector run.
It is checked like every other test.

## Performance Measurement

- Bind every number to the exact candidate and configuration that produced it. Capture
  `spec.checkpoint inputs <run>` before measuring and store it as `input_digests` with `impl`,
  `config`, `metrics`, and `objective`; the evaluator brief defines the fields.
- The headline is the median of clean samples on the full declared workload, never a proxy or a
  stale run.
- Headline measurements run alone on the evaluation host. Exclusivity is binary: if anything else
  touched the measured resource during the run (another GPU tenant, a peer benchmark), the whole
  measurement is void and is rerun alone. Never salvage a contaminated run by dropping samples,
  and never clear the machine by stopping processes the run did not start: wait, or report the
  measurement as provisional.
- Proxy runs are for iteration feedback only. Calibrate at full scale early and recheck as the
  design changes.
- Before `--became-best` is recorded, the evaluator checks the candidate on a held-out seed,
  slice, or nearby workload the coding agent did not see. A win that vanishes on the held-out draw is
  an `overfit` finding, not a best.
- Profile the scored configuration and name its measured bottleneck before choosing the next change.
- Never score or report a candidate that fails a test.

## Release Claims

`production-ready` is a mechanically supported claim, not a label. It requires all six:

1. a full-scale score, measured from the selected candidate after its last change, that beats the
   declared baseline (the same comparison the loop uses for `--became-best`);
2. no open defect in the decision log, whatever its kind (a reward hack or an overfit logged as a
   defect blocks; one logged as advisory does not);
3. a kept test for every required correctness, safety, and operational class;
4. an Auditor completeness pass whose stamp covers the delivered bytes, requirements, tests,
   evaluator, and decisions and found
   nothing new;
5. a test that runs at the operating point; and
6. the Auditor's attack pass (`review/attack.md`) against the acknowledged guarantees, which found
   no reproducible violation.

Universal coverage includes result fidelity, memory and concurrency safety where applicable, visible
failures, bounded resources under sustained operation, and resistance to benchmark gaming. Stateful
systems also need durability, recovery, and concurrency semantics when the resolved contract
requires them. Domain-specific requirements come from discovery and the specification cards, not
from a fixed checklist.

Enforce the release checks with:

```bash
python3 <scripts>/run_tests.py --run <run> --production-ready
# <scripts>: .claude/skills/skysynth/scripts in a project set up by `skydiscover init`;
#            skydiscover/synthesize/workflow/scripts in a source checkout.
```

The command checks the test suite, recorded defects, and audit stamp; property coverage, the
operating-point test, baseline validity, and the final attack-review outcome still require review.
A successful `run finish` alone is not a production-readiness claim.

After every test passes, this asks two more questions (`scripts/check_release.py`), each answered
by a file the run already has:

- Is every defect in `decision_log.json` closed, either by the user or by a fix whose test passes
  the built implementation and fails the broken copy that has the defect?
- Has the auditor looked at this exact implementation? Its
  `synthesis/audit/completeness.json` (written with `spec.checkpoint stamp-audit`) must carry the
  digests of the implementation and current inputs being delivered.

`run finish --export-to . [--production-ready]` also runs the suite against a temporary copy of the
actual result before deleting working files. Saved inputs and logs
live in `best/.verification/`; these checks use the installed framework and host toolchains, not an
OS sandbox or a self-contained runtime. Inspect property coverage and external dependencies yourself.

Anything short of all six is a scoped result with known limitations. A reproducible contract
violation is fixed and covered by a test; it is never parked as documentation to protect a number.
