"""Publish a run's result under outputs/synthesize/<slug>_<timestamp>/ (`paths.outputs()` moves the root).

    checkpoints/checkpoint_<N>/   artifact/, score.json, tests.json (the tests it was scored against)
    best/                         artifact/, score.json, tests/, spec.md
    history.json                  written by `run finish`: one row per checkpoint, with what it fails today

score.json is filled from the leaderboard (the score and the baselines) plus `became_best`, the
loop's call at the time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .paths import Run, outputs, project_of, test_files
from .render import render_spec

_TRANSIENT = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".DS_Store",
}


def _is_dot_lock(name: str) -> bool:
    """A bookkeeping lock sidecar (`.index.lock`); a dependency lockfile (`cache.lock`) is not one."""
    return name.startswith(".") and name.endswith(".lock")


def _transient(relative: Path, *, ignored=()) -> bool:
    """What neither the copy nor the digest counts: runtime caches, the run's own bookkeeping and
    its lock sidecars. Everything else, hidden or not (a task's `.data/`), is part of the candidate;
    the copy and the digest apply this one rule so a copy always re-digests to the same value."""
    parts = relative.parts
    return (
        bool(set(parts) & (_TRANSIENT | set(ignored)))
        or relative.suffix in {".pyc", ".pyo"}
        or any(_is_dot_lock(part) for part in parts)
    )


def _copy_ignore(directory, names):
    return {n for n in names if _transient(Path(n))}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _tree_digest(root: Path, *, ignored=()) -> str:
    """Hash file names and bytes, including hidden dependencies but excluding runtime caches."""
    if root.is_symlink():
        raise ValueError(f"Record real dependency files instead of symlinks: {root}")
    digest = hashlib.sha256()
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        relative = Path(path.name) if path == root else path.relative_to(root)
        if _transient(relative, ignored=ignored):
            continue
        if path.is_symlink():
            raise ValueError(f"Record real dependency files instead of symlinks: {path}")
        if path.is_file():
            digest.update(relative.as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return f"sha256:{digest.hexdigest()}"


def _copy_mismatch(source: Path, copy: Path, limit: int = 3) -> list[str]:
    """The first files that differ between a tree and its copy (missing, extra or changed), so a
    refused snapshot names what it could not carry over."""

    def files(root: Path) -> dict:
        found = {}
        for path in sorted(root.rglob("*")) if root.is_dir() else [root]:
            relative = Path(path.name) if path == root else path.relative_to(root)
            if path.is_file() and not _transient(relative):
                found[relative.as_posix()] = path
        return found

    left, right = files(source), files(copy)
    names = sorted(set(left) | set(right))
    out = []
    for name in names:
        a, b = left.get(name), right.get(name)
        if a is None or b is None or a.read_bytes() != b.read_bytes():
            out.append(name)
        if len(out) >= limit:
            break
    return out


# The reference and the mutants admit tests (validate_test.py); the benchmark never reads them, so
# the evaluator adding a mutant mid-run does not make an earlier measurement incomparable. The audit
# digest keeps them: the auditor reviews what the tests were validated against.
_TEST_ONLY = ("reference", "mutants")


def input_digests(run_dir: Path, *, audit=False) -> dict:
    """Capture these BEFORE measuring; changing inputs requires a fresh measurement."""
    run = Run(run_dir)
    inputs = {
        "implementation": _tree_digest(run.impl),
        "task": _tree_digest(run.task),
        "specification": _tree_digest(
            run.specification,
            ignored=("sources", "profiling", *(() if audit else ("failure_patterns.json",))),
        ),
        "tests": _tree_digest(run.tests),
        "evaluator": _tree_digest(run.evaluator, ignored=() if audit else _TEST_ONLY),
    }
    if audit:
        inputs["decisions"] = _tree_digest(run.decision_log)
    return inputs


def read_record(checkpoint: Path) -> dict:
    record = _read_json(checkpoint / ".verification/record.json", {})
    if not isinstance(record, dict) or not isinstance(record.get("inputs"), dict):
        return {}
    entry = record.get("entry")
    if (
        not isinstance(entry, str)
        or not entry
        or Path(entry).is_absolute()
        or ".." in Path(entry).parts
    ):
        return {}
    return record


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _merge_missing(src: Path, dst: Path) -> None:
    """Copy source files from `src` into `dst` without overwriting any. Markdown is not source."""
    if not src.is_dir():
        return
    for path in src.rglob("*"):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix in {".pyc", ".pyo", ".md"}
            or path.name.startswith(".")
        ):
            continue
        target = dst / path.relative_to(src)
        if target.exists():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        except OSError:
            continue


def _replace_dir(staged: Path, destination: Path) -> None:
    """Swap `staged` into place of `destination` atomically."""
    backup = destination.with_name(f".{destination.name}.old")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _unique_output(root: Path, slug: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root / f"{slug}_{stamp}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    return candidate


def published_output(run_dir: Path) -> Optional[Path]:
    """Return the output directory the run's pointer names, if it exists. A pointer inside the
    project is stored relative to it, so a moved or copied project still finds its result."""
    pointer = Run(run_dir).output_pointer
    if not pointer.is_file():
        return None
    raw = pointer.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    target = Path(raw)
    if not target.is_absolute():
        target = (project_of(run_dir) or Path.cwd()) / raw
    return target if target.is_dir() else None


def output_for_run(run_dir: Path, export_root: Path) -> Path:
    """Return the run's output directory, creating it on first use."""
    existing = published_output(run_dir)
    if existing is not None:
        return existing
    project = project_of(run_dir)
    root = export_root.resolve()  # CLI paths are relative to the caller, not a relocated runs root.
    out = _unique_output(root / outputs(), run_dir.name)
    (out / "checkpoints").mkdir(parents=True)
    (out / "best").mkdir()
    resolved = out.resolve()
    record = str(resolved)
    if project is not None:  # relative to the project when there is one, so the project can move
        try:
            record = str(resolved.relative_to(project))
        except ValueError:
            pass
    Run(run_dir).output_pointer.write_text(record + "\n", encoding="utf-8")
    return out


def _checkpoint_numbers(output_dir: Path) -> list[int]:
    values: list[int] = []
    for path in (output_dir / "checkpoints").glob("checkpoint_*"):
        try:
            values.append(int(path.name.removeprefix("checkpoint_")))
        except ValueError:
            continue
    return values


def _checkpoint(output_dir: Path, number: int) -> Path:
    return output_dir / "checkpoints" / f"checkpoint_{number}"


def checkpoint_count(output_dir: Path) -> int:
    """Return how many checkpoints the result has."""
    return len(_checkpoint_numbers(output_dir))


def artifact_digest(checkpoint: Path) -> str:
    """Return the digest of a checkpoint's artifact/."""
    return _tree_digest(checkpoint / "artifact")


def read_score(checkpoint: Path) -> dict[str, Any]:
    """Return a checkpoint's score.json, or {} if missing or malformed."""
    doc = _read_json(checkpoint / "score.json", {})
    return doc if isinstance(doc, dict) else {}


def read_tests(checkpoint: Path) -> Optional[list[str]]:
    """Return a checkpoint's tests.json: the test files it was scored against. None if absent."""
    if not (checkpoint / "tests.json").is_file():
        return None
    names = _read_json(checkpoint / "tests.json", [])
    return [str(n) for n in names] if isinstance(names, list) else []


def history(
    output_dir: Path,
    fails: Optional[dict[int, Optional[list[str]]]] = None,
    best: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Return one row per checkpoint, read from the checkpoints themselves.

    Each row: `checkpoint`, `created_at`, `score`, `became_best`, `tests` (how many; null when the
    checkpoint has no tests.json). With
    `run finish`'s verdicts: `fails` (the tests it fails today, null if they could not be run)
    and `published: true` on the one in best/.
    """
    rows = []
    for number in sorted(_checkpoint_numbers(output_dir)):
        checkpoint = _checkpoint(output_dir, number)
        score = read_score(checkpoint)
        row: dict[str, Any] = {
            "checkpoint": number,
            "created_at": score.get("created_at"),
            "score": score.get("score") or {},
            "became_best": bool(score.get("became_best")),
            "tests": None if (names := read_tests(checkpoint)) is None else len(names),
        }
        if fails is not None:
            row["fails"] = fails.get(number)
        if number == best:
            row["published"] = True
        rows.append(row)
    return rows


def write_history(
    output_dir: Path, fails: dict[int, Optional[list[str]]], best: Optional[int]
) -> list[dict[str, Any]]:
    """Write history.json at the end of the run and return its rows."""
    rows = history(output_dir, fails, best)
    _write_json(output_dir / "history.json", rows)
    return rows


def entry_in(checkpoint: Path, run_dir: Path) -> Optional[Path]:
    """Return the checkpoint's copy of the run's entry file, if it is there."""
    run = Run(run_dir)
    recorded = read_record(checkpoint).get("entry")
    entry = run.impl / recorded if isinstance(recorded, str) else None
    if entry is None:
        return None
    try:
        rel = entry.resolve().relative_to(run.impl.resolve())
    except ValueError:
        return None
    path = checkpoint / "artifact" / rel
    return path if path.exists() else None


def _stage_artifact(artifact: Path, contract: Optional[Path], into: Path) -> Path:
    """Copy the candidate to `into`/artifact and fold the interface's source files in."""
    staged = into / "artifact"
    shutil.copytree(artifact, staged, ignore=_copy_ignore)
    if contract is not None:
        _merge_missing(contract, staged)
        if (contract / "__init__.py").is_file():
            package = staged / "interface"
            if package.exists() and _tree_digest(package) != _tree_digest(contract):
                raise ValueError(
                    "The implementation and contract both define different interface packages."
                )
            if not package.exists():
                shutil.copytree(contract, package, ignore=_copy_ignore)
    return staged


def latest_checkpoint_for(run_dir: Path, output_dir: Path) -> Optional[Path]:
    """Return the newest matching artifact and entry, including a restored earlier candidate."""
    numbers = _checkpoint_numbers(output_dir)
    run = Run(run_dir)
    if not numbers or not run.impl.is_dir():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        staged = _stage_artifact(run.impl, run.interface, Path(tmp))
        entry = run.entry_impl()
        relative = entry.resolve().relative_to(run.impl.resolve()).as_posix() if entry else "."
        for number in sorted(numbers, reverse=True):
            saved = _checkpoint(output_dir, number)
            if read_record(saved).get("entry") == relative and _tree_digest(
                staged
            ) == artifact_digest(saved):
                return saved
    return None


def best_lineage(output_dir: Path) -> list[Path]:
    """Return the checkpoints to publish from, in order: best/'s own first, then newest first."""
    picks: list[Path] = []
    selected = read_score(output_dir / "best").get("checkpoint")
    if isinstance(selected, int):
        checkpoint = _checkpoint(output_dir, selected)
        if (checkpoint / "artifact").is_dir() and (checkpoint / "score.json").is_file():
            picks.append(checkpoint)
    for number in sorted(_checkpoint_numbers(output_dir), reverse=True):
        checkpoint = _checkpoint(output_dir, number)
        if checkpoint not in picks:
            picks.append(checkpoint)
    return picks


def selected_best_checkpoint(output_dir: Path) -> Optional[Path]:
    """Return the checkpoint best/ was made from, if it still exists."""
    selected = read_score(output_dir / "best").get("checkpoint")
    if not isinstance(selected, int):
        return None
    checkpoint = _checkpoint(output_dir, selected)
    return checkpoint if (checkpoint / "score.json").is_file() else None


def _measured_row(board: Any, inputs: dict) -> dict:
    """The newest scored candidate measured against these exact inputs."""
    return (
        next(
            (
                r
                for r in reversed(board)
                if isinstance(r, dict)
                and r.get("role", "candidate") == "candidate"
                and r.get("draw", "scored") == "scored"
                and r.get("input_digests") == inputs
            ),
            {},
        )
        if isinstance(board, list)
        else {}
    )


# What makes two measurements comparable: the same task and specification (the workload), the
# same evaluator (the benchmark), the same configuration. New tests, mutants, or decisions do not
# move a score.
_BENCHMARK_INPUTS = ("task", "specification", "evaluator")


def score_from_run(run_dir: Path) -> dict[str, Any]:
    """Use a measurement of the current inputs and every baseline comparable to it (same inputs
    and configuration), the one the candidate names first."""
    run = Run(run_dir)
    doc: dict[str, Any] = {"checkpoint": None, "created_at": _utc_now(), "score": {}}
    if run.is_proof_run():
        return doc
    board = _read_json(run.leaderboard, [])
    inputs = input_digests(run_dir)
    row = _measured_row(board, inputs)
    if not row:
        raise ValueError(
            "No leaderboard measurement matches these inputs. Capture 'checkpoint inputs' before measuring, then record it with the score."
        )
    if not isinstance(row.get("config"), dict):
        raise ValueError("Record the benchmark configuration as an object, even when empty.")
    if row.get("direction", "max") not in ("min", "max"):
        raise ValueError("Score direction must be 'min' or 'max'.")
    metrics = row.get("metrics")
    if (
        not isinstance(metrics, dict)
        or not metrics
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values())
    ):
        raise ValueError("Metrics must be finite numbers.")
    objective = row.get("objective", next(iter(metrics)) if len(metrics) == 1 else None)
    if not isinstance(objective, str) or objective not in metrics:
        raise ValueError("Name the objective when recording several metrics.")
    entry = run.entry_impl()
    named = row.get("impl")
    if not isinstance(named, str) or entry is None:
        raise ValueError("The measurement must name its impl file or directory.")
    measured = Path(named) if Path(named).is_absolute() else run.path / named
    if measured.resolve() != entry.resolve():
        raise ValueError("The measured entry differs from the selected implementation.")
    doc["score"] = {objective: metrics[objective]}
    requested = row.get("baseline")
    # One entry per baseline: the measurement taken beside this candidate (same implementation
    # digest) when there is one, else the newest. A row without a name goes by the name another
    # row gave the same impl, else by the impl itself.
    rows = [
        base
        for base in board
        if isinstance(base, dict)
        and base.get("role") == "baseline"
        and base.get("draw", "scored") == "scored"
    ]
    names = {
        base["impl"]: base["name"]
        for base in rows
        if isinstance(base.get("impl"), str) and isinstance(base.get("name"), str) and base["name"]
    }
    chosen: dict[str, tuple[bool, dict[str, Any]]] = {}
    for base in rows:
        hashes = base.get("input_digests", {})
        if (
            not isinstance(hashes, dict)
            or any(hashes.get(k) != inputs.get(k) for k in _BENCHMARK_INPUTS)
            or base.get("config") != row.get("config")
        ):
            continue  # measured against another benchmark, workload, or configuration: no comparison
        name = base.get("name") or names.get(base.get("impl"), base.get("impl"))
        value = (
            base.get("metrics", {}).get(objective)
            if isinstance(base.get("metrics"), dict)
            else None
        )
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            continue  # nothing to compare against
        beside = hashes.get("implementation") == inputs.get("implementation")
        if name not in chosen or beside >= chosen[name][0]:
            chosen[name] = (beside, {"name": name, "score": {objective: value}})
    baselines = [entry for _, entry in chosen.values()]
    if requested is not None:
        if requested not in chosen:
            # The margin is informational, never a check: the score stands, spec.md just shows no margin.
            print(
                f"note: baseline {requested!r} was measured against another benchmark, workload, or "
                "configuration; no margin against it is recorded (re-measure it to get one).",
                file=sys.stderr,
            )
        baselines.sort(key=lambda b: b["name"] != requested)  # the headline comparison first
    if baselines:
        doc["baselines"] = baselines
    return doc


def write_checkpoint(
    output_dir: Path,
    artifact: Path,
    score: dict[str, Any],
    *,
    contract: Optional[Path] = None,
    tests: Optional[Path] = None,
    became_best: bool = False,
    run_dir: Optional[Path] = None,
    inputs: Optional[dict] = None,
) -> Path:
    """Write the next checkpoint and return its path."""
    if not artifact.is_dir():
        raise FileNotFoundError(f"artifact directory not found: {artifact}")
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    number = max(_checkpoint_numbers(output_dir), default=0) + 1
    destination = _checkpoint(output_dir, number)
    staged = Path(tempfile.mkdtemp(prefix=f".checkpoint_{number}.", dir=destination.parent))
    try:
        _stage_artifact(artifact, contract, staged)
        _write_json(staged / "score.json", dict(score, checkpoint=number, became_best=became_best))
        names = [p.name for p in test_files(tests)] if tests and tests.is_dir() else []
        _write_json(staged / "tests.json", names)
        if run_dir is not None:
            run = Run(run_dir)
            entry = run.entry_impl()
            if entry is None:
                raise ValueError("No implementation entry to checkpoint.")
            private = staged / ".verification"
            private.mkdir()
            measured = _measured_row(_read_json(run.leaderboard, []), inputs or {})
            _write_json(
                private / "record.json",
                {
                    "entry": entry.resolve().relative_to(run.impl.resolve()).as_posix(),
                    "inputs": inputs,
                    "config": measured.get("config"),
                    "direction": measured.get("direction", "max"),
                },
            )
            shutil.copytree(run.impl, private / "source", ignore=_copy_ignore)
            if _tree_digest(private / "source") != input_digests(run_dir)["implementation"]:
                differing = _copy_mismatch(run.impl, private / "source")
                raise ValueError(
                    "The source copy changed or left out a dependency"
                    + (f" ({', '.join(differing)})" if differing else "")
                    + "; keep dependencies in publishable files."
                )
            if run.evaluator.is_dir():
                shutil.copytree(run.evaluator, private / "evaluator", ignore=_copy_ignore)
                if (
                    _tree_digest(private / "evaluator", ignored=_TEST_ONLY)
                    != input_digests(run_dir)["evaluator"]
                ):
                    raise ValueError("The evaluator copy changed or left out a dependency.")
            if inputs != input_digests(run_dir):
                raise ValueError(
                    "Inputs changed while saving the checkpoint; measure the stable version again."
                )
        os.replace(staged, destination)
    except BaseException:
        if staged.exists():
            shutil.rmtree(staged)
        raise
    return destination


def materialize_best(
    output_dir: Path,
    checkpoint: Path,
    *,
    tests: Optional[Path] = None,
    spec: str = "",
) -> Path:
    """Replace best/ with a checkpoint's artifact and score, plus `tests` and `spec`."""
    artifact = checkpoint / "artifact"
    if not artifact.is_dir() or not (checkpoint / "score.json").is_file():
        raise ValueError(f"not a complete checkpoint: {checkpoint}")
    destination = output_dir / "best"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".best.", dir=destination.parent))
    try:
        shutil.copytree(artifact, staged / "artifact", ignore=_copy_ignore)
        shutil.copy2(checkpoint / "score.json", staged / "score.json")
        if (checkpoint / ".verification").is_dir():
            shutil.copytree(
                checkpoint / ".verification", staged / ".verification", ignore=_copy_ignore
            )
        if tests and tests.is_dir():
            shutil.copytree(tests, staged / "tests", ignore=_copy_ignore)
        (staged / "spec.md").write_text(spec, encoding="utf-8")
        _replace_dir(staged, destination)
    except BaseException:
        if staged.exists():
            shutil.rmtree(staged)
        raise
    return destination


def publish_best(
    output_dir: Path,
    checkpoint: Path,
    run_dir: Path,
    rows: Optional[list[dict[str, Any]]] = None,
) -> Path:
    """Publish `checkpoint` as best/, with the run's tests and a rendered spec.md.

    `rows` is the history to render; by default the checkpoints as they stand, without verdicts.
    """
    run = Run(run_dir)
    record = read_record(checkpoint)
    spec = render_spec(
        run_dir,
        read_score(checkpoint),
        history(output_dir) if rows is None else rows,
        measured_config=record.get("config"),
        direction=record.get("direction"),
    )
    return materialize_best(
        output_dir, checkpoint, tests=run.tests if run.tests.is_dir() else None, spec=spec
    )


def audit_covers(stamp: Path, artifact: Path, run_dir: Optional[Path] = None) -> Optional[str]:
    """Return why the audit stamp does not cover `artifact`, or None if it does.

    The stamp must carry the artifact's digest; a stamp without one is refused. `artifact` is
    synthesis/impl/, the tree the auditor read, or its published form: the same bytes with the
    interface folded in, which is what `run finish` checks in `best/artifact/`. `run_dir` is the run
    the stamp belongs to; by default the run holding `artifact` as its synthesis/impl/.
    """
    if not stamp.is_file():
        return f"{stamp} is missing"
    data = _read_json(stamp, None)
    if not isinstance(data, dict) or not isinstance(data.get("findings"), list):
        return f"{stamp} is malformed (need an object with a 'findings' list)"
    if not isinstance(data.get("artifact_id"), str):
        return f"{stamp} carries no artifact_id; re-stamp with stamp-audit to bind it to the bytes"
    run = Run(run_dir if run_dir is not None else artifact.parent.parent)
    if data["artifact_id"] != _tree_digest(artifact) and not _is_published_form(
        artifact, run, data["artifact_id"]
    ):
        return f"{stamp} was written for different implementation bytes"
    if data.get("input_digests") != input_digests(run.path, audit=True):
        return "the audit does not cover the current requirements, tests, evaluator and decisions"
    return None


def _is_published_form(artifact: Path, run: Run, artifact_id: str) -> bool:
    """Whether `artifact` is the audited candidate as published: the stamp covers synthesis/impl/
    and `artifact` is byte-for-byte what staging that candidate with its interface produces."""
    if not run.impl.is_dir() or artifact.resolve() == run.impl.resolve():
        return False
    if artifact_id != _tree_digest(run.impl):
        return False
    with tempfile.TemporaryDirectory() as tmp:
        try:
            staged = _stage_artifact(run.impl, run.interface, Path(tmp))
        except ValueError:
            return False
        return _tree_digest(staged) == _tree_digest(artifact)


def stamp_audit(run_dir: Path, findings: list[str]) -> Path:
    """Record that the auditor covered the current candidate, and which findings it produced."""
    run = Run(run_dir)
    if not run.impl.is_dir():
        raise FileNotFoundError(f"no candidate at {run.impl}")
    run.audit.mkdir(parents=True, exist_ok=True)
    doc = {
        "candidate": run.impl.relative_to(run.path).as_posix(),
        "artifact_id": _tree_digest(run.impl),
        "input_digests": input_digests(run_dir, audit=True),
        "timestamp": int(datetime.now(timezone.utc).timestamp()),
        "findings": list(findings),
    }
    run.completeness.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return run.completeness


def audit_staleness_notice(run_dir: Path) -> str:
    """Return a reminder if the current candidate has no covering audit stamp, else ""."""
    run = Run(run_dir)
    if not run.impl.is_dir():
        return ""
    why = audit_covers(run.completeness, run.impl)
    if why:
        return (
            f"checkpoint: new best not covered by the audit ({why}). "
            "Run the auditor before the next iteration; the delivery check requires it."
        )
    return ""


def snapshot_run(
    run_dir: Path,
    export_root: Path,
    *,
    became_best: bool = False,
    require_evaluation: bool = True,
) -> tuple[Path, Path]:
    """Checkpoint the current candidate; refuse one with no score unless told otherwise."""
    run = Run(run_dir)
    if not run.impl.is_dir():
        raise FileNotFoundError(f"no generated artifact at {run.impl}")
    inputs = input_digests(run_dir)
    score = score_from_run(run_dir)
    if require_evaluation and not score["score"]:
        raise ValueError(
            f"refusing to checkpoint an unevaluated artifact: {run.leaderboard} "
            "has no entry with metrics for this candidate. A checkpoint is one evaluated iteration; "
            "run the scored benchmark and append the leaderboard entry first."
        )
    output_dir = output_for_run(run_dir, export_root)
    if became_best:
        # `--became-best` is checked, not taken on trust: against the current best under the
        # leaderboard's objective and direction, a worse score is not a new best. A tie is allowed
        # (the same bytes re-measured); a different objective is incomparable and not judged here.
        current = selected_best_checkpoint(output_dir)
        objective = next(iter(score["score"]), None)
        previous = read_score(current).get("score", {}) if current is not None else {}
        if current is not None and objective is not None and objective in previous:
            new, old = score["score"][objective], previous[objective]
            direction = _measured_row(_read_json(run.leaderboard, []), inputs).get(
                "direction", "max"
            )
            worse = new < old if direction == "max" else new > old
            if worse:
                raise ValueError(
                    f"not a new best: {objective} {new} is worse than {current.name}'s {old} "
                    f"(direction {direction}); snapshot it without --became-best."
                )
    checkpoint = write_checkpoint(
        output_dir,
        run.impl,
        score,
        contract=run.interface,
        tests=run.tests,
        became_best=became_best,
        run_dir=run_dir,
        inputs=inputs,
    )
    if became_best:
        publish_best(output_dir, checkpoint, run_dir)
    return output_dir, checkpoint


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m skydiscover.synthesize.spec.checkpoint",
        description="Snapshot one evaluated iteration of a run into its published output.",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    inputs = sub.add_parser("inputs", help="capture inputs before a benchmark run")
    inputs.add_argument("run_dir")
    snap = sub.add_parser(
        "snapshot", help="write checkpoint_<n>/ and, with --became-best, refresh best/"
    )
    snap.add_argument("run_dir")
    snap.add_argument(
        "--export-root",
        default=None,
        help="where outputs/synthesize/ lives (default: the project holding the run, else here)",
    )
    snap.add_argument(
        "--became-best",
        action="store_true",
        help="this candidate beats the incumbent and passes every current test",
    )
    stamp = sub.add_parser(
        "stamp-audit",
        help="record that the auditor covered the current candidate (synthesis/audit/completeness.json)",
    )
    stamp.add_argument("run_dir")
    stamp.add_argument(
        "--finding",
        action="append",
        default=[],
        help="a decision-log id this round produced (repeatable)",
    )
    args = ap.parse_args(argv)

    if args.command == "inputs":
        print(json.dumps(input_digests(Path(args.run_dir)), indent=2))
        return 0
    if args.command == "stamp-audit":
        print(stamp_audit(Path(args.run_dir), args.finding))
        return 0

    run_dir = Path(args.run_dir)
    export_root = (  # the run's project, not the caller's cwd, decides where the result lives
        Path(args.export_root) if args.export_root else (project_of(run_dir) or Path("."))
    )
    _, out = snapshot_run(run_dir, export_root, became_best=args.became_best)
    if args.became_best:
        notice = audit_staleness_notice(Path(args.run_dir))
        if notice:
            print(notice, file=sys.stderr)
    print(out)
    return 0


if __name__ == "__main__":
    from .paths import run_cli

    raise SystemExit(run_cli(main, "checkpoint"))
