"""Check a run before the build loop starts, and save what it learned when it ends.

    run check  <run_dir>                          discovery produced its files? exit 0 = ready to build
    run finish <run_dir> [domain] [--export-to <path>] [--keep-run] [--refresh-tests] [--production-ready]
                                                  save new tests and the user's answers; publish the result

The domain names the knowledge base folder (~/.skydiscover/<domain>/); by default it is the `domain:` line
of the run's task.md front matter.

Both exit 1 on a problem and 2 when the run dir does not exist; finish is safe to repeat.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

from . import checkpoint as artifact_store
from . import kept_tests
from .decisions import kept_decisions, save
from .findings import Findings
from .paths import TEST_SCRIPT, Domain, Run, home, project_of, shared_wiki, test_files
from .render import passed_final_tests, render_spec

# check

# Present but carries nothing: a file or folder made only of these is empty.
_PLACEHOLDERS = {".gitkeep", ".keep", ".gitignore", ".DS_Store"}
_EMPTY_JSON = {"", "{}", "[]", "null"}

# Run attributes the synthesis loop cannot start without; each must exist and be non-empty.
_REQUIRED = ("reference_tests", "reference_tests_index", "skeleton")
# Run attributes only some runs produce; reported, never required.
_OPTIONAL = ("workload_card", "environment_card")

_ARTIFACTS_DOC = "skydiscover/synthesize/workflow/references/artifacts.md"


def _has_real_clone(run: Run) -> bool:
    """True if sources/ holds at least one git clone (a symlink into the shared clone cache counts)."""
    if not run.sources.is_dir():
        return False
    return any((d / ".git").exists() for d in run.sources.iterdir() if d.is_dir())


def _wiki_roots(run: Run) -> List[Path]:
    """Every wiki folder a reuse may come from: the run's own domain first, then the rest of the
    knowledge base, then the shared wiki. The clone-reuse hook refuses a `git clone` when ANY
    domain's wiki covers the repo at its current commit and tells the run to use the wiki
    instead, so the reuse it forces must be recognizable here wherever the page lives."""
    roots: List[Path] = []
    slug = run.domain()
    if slug:
        own = Domain(slug).wiki
        if own.is_dir():
            roots.append(own)
    base = home()
    if base.is_dir():
        for d in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")):
            wiki = d / "wiki"
            if wiki.is_dir() and wiki not in roots:
                roots.append(wiki)
    shared = shared_wiki()
    if shared.is_dir() and shared not in roots:
        roots.append(shared)
    return roots


def _is_source_page(page: Path) -> bool:
    """True if a wiki page pins a repository: its front matter carries both a url and a sha."""
    try:
        text = page.read_text(encoding="utf-8")
    except OSError:
        return False
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return False
    fm = m.group(1)
    return bool(re.search(r"^url:\s*\S", fm, re.M) and re.search(r"^sha:\s*\S", fm, re.M))


def _safe_rel_page(page) -> bool:
    """A wiki_page from acquisitions.json names one file under the wiki: no absolute path, no
    `..`, no glob characters."""
    if not isinstance(page, str) or not page or os.path.isabs(page):
        return False
    if any(ch in page for ch in "*?["):
        return False
    return ".." not in Path(page).parts


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _has_wiki_reuse(run: Run) -> bool:
    """True when acquisitions.json names a source reused from the wiki and that page exists with a
    url and sha, in any domain's wiki (the same scope the clone-reuse hook searches when it
    refuses the clone)."""
    acq = _read_json(run.acquisitions)
    if not isinstance(acq, dict):
        return False
    roots = _wiki_roots(run)
    if not roots:
        return False
    for src in acq.get("reused_from_wiki") or []:
        page = src.get("wiki_page") if isinstance(src, dict) else None
        if not _safe_rel_page(page):
            continue
        for wiki in roots:
            hit = wiki / Path(page)
            if _within(hit, wiki) and hit.is_file() and _is_source_page(hit):
                return True
    return False


def _reference_specs(run: Run) -> Tuple[List[str], List[str]]:
    """(names whose spec.json has {source, axes}, names whose spec.json has neither)."""
    good: List[str] = []
    bad: List[str] = []
    if not run.references.is_dir():
        return good, bad
    for d in sorted(
        p for p in run.references.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        if not (d / "spec.json").is_file():
            continue
        spec = _read_json(d / "spec.json")
        if isinstance(spec, dict) and (spec.get("axes") or spec.get("source")):
            good.append(d.name)
        else:
            bad.append(d.name)
    return good, bad


def _is_empty(path: Path) -> bool:
    """A folder is empty unless it holds a non-empty real file; a file is empty if it holds nothing
    (or nothing but {} / [])."""
    if path.is_dir():
        for p in path.rglob("*"):
            if p.is_file() and p.name not in _PLACEHOLDERS and not p.name.startswith("."):
                if "__pycache__" not in p.parts and p.stat().st_size > 0:
                    return False
        return True
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip() in _EMPTY_JSON
    except OSError:
        return True


def _read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def check(run_dir: Path) -> Tuple[bool, List[str], List[str], List[str]]:
    """(ok, report lines, problems, warnings). ok is False when a required file is missing or
    empty; warnings never change ok."""
    run = Run(run_dir)
    lines: List[str] = []
    problems: List[str] = []
    warnings: List[str] = []

    def rel(p: Path) -> str:
        return p.relative_to(run.path).as_posix() + ("/" if p.suffix == "" else "")

    spec_label = f"{rel(run.references)}*/spec.json"
    verify_label = f"{rel(run.references)}*/verification.json"
    names = [rel(getattr(run, a)) for a in _REQUIRED + _OPTIONAL] + [
        rel(run.sources),
        spec_label,
        verify_label,
    ]
    width = max(len(n) for n in names)

    def row(tag: str, name: str, note: str) -> None:
        lines.append(f"  [{tag.ljust(7)}] {name.ljust(width)}  {note}")

    for attr in _REQUIRED:
        path = getattr(run, attr)
        name = rel(path)
        if path.exists() and not _is_empty(path):
            row("ok", name, "present")
        elif path.exists():
            row("EMPTY", name, "exists but holds nothing")
            problems.append(f"{name}: empty")
        else:
            row("MISSING", name, "not found")
            problems.append(f"{name}: missing")

    for attr in _OPTIONAL:
        path = getattr(run, attr)
        name = rel(path)
        if path.exists() and not _is_empty(path):
            row("ok", name, "present")
        else:
            row("-", name, "absent (optional)")

    # A reference system was read this run: a git clone under sources/, or a source reused from the
    # a wiki page that pins url + sha (references/acquisitions.json).
    sources = rel(run.sources)
    if _has_real_clone(run):
        row("ok", sources, "git clone present")
    elif _has_wiki_reuse(run):
        row("ok", sources, "source reused from the wiki")
    else:
        row("MISSING", sources, "no git clone, no source reused from the wiki")
        problems.append(
            f"{sources}: no git clone under it and no source reused from the wiki "
            f"({rel(run.acquisitions)})"
        )

    # At least one reference spec with {source, axes}.
    good, bad = _reference_specs(run)
    if good:
        row("ok", spec_label, ", ".join(good))
        if bad:
            row("WARN", spec_label, f"without {{source, axes}}: {', '.join(bad)}")
            warnings.append(f"{spec_label}: without {{source, axes}}: {', '.join(bad)}")
    else:
        row("MISSING", spec_label, "no spec.json with {source, axes}")
        problems.append(
            f"{spec_label}: no spec.json with {{source, axes}}"
            + (f" (without them: {', '.join(bad)})" if bad else "")
        )

    # Every reference spec's citations were re-checked.
    unverified = [n for n in good if _is_empty(run.references / n / "verification.json")]
    if good and not unverified:
        row("ok", verify_label, ", ".join(good))
    elif unverified:
        row("MISSING", verify_label, ", ".join(unverified))
        problems.append(f"{verify_label}: missing for {', '.join(unverified)}")

    header = [f"RUN CHECK  {run_dir}", ""]
    return not problems, header + lines, problems, warnings


def main_check(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="run check",
        description="Are the specification files there? Exit 0 = the synthesis loop may start.",
    )
    ap.add_argument("run_dir", help="the run directory, e.g. .skydiscover/<slug>")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"run check: no run directory at {run_dir}", file=sys.stderr)
        return 2

    ok, lines, problems, warnings = check(run_dir)
    print("\n".join(lines))
    for w in warnings:
        print(f"\nWARNING: {w}")
    print(f"\nWhat each file is and who writes it: {_ARTIFACTS_DOC}")
    if ok:
        print("\nPASSED")
        return 0
    print(f"\nFAILED: {len(problems)} required file(s) missing or empty")
    for i, p in enumerate(problems, 1):
        print(f"  {i}. {p}")
    return 1


# finish

_RUN_TESTS = Path(__file__).resolve().parent.parent / "workflow" / "scripts" / "run_tests.py"


def _run_one(checkpoint: Path, entry: Path, suite: Path, name: str) -> Optional[int]:
    """`test.sh <name>` against a checkpoint's bytes: run_tests' exit code, None if it never ran."""
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(_RUN_TESTS),
                "--interface",
                str(checkpoint / ".verification/evaluator/interface"),
                "--impl",
                str(entry),
                "--suite",
                str(suite),
                "--test",
                name,
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=int(os.environ.get("SKYDISCOVER_SLOW_SECS", "600") or "600"),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode not in (0, 1):
        print(
            f"  checks: run_tests gave no verdict for {checkpoint.name} on {name}:\n"
            + "\n".join("    " + line for line in proc.stderr.strip().splitlines()[-6:]),
            file=sys.stderr,
        )
    return proc.returncode


def _failing_tests(checkpoint: Path, run_dir: Path) -> Optional[List[str]]:
    """The current tests a checkpoint fails, every test run once, all at the same time. [] means
    it passes. None means no verdict (no suite, or a test that could not run), so the caller
    keeps its selection."""
    suite = Run(run_dir).tests
    entry = artifact_store.entry_in(checkpoint, run_dir)
    names = [p.name for p in test_files(suite)] if (suite / TEST_SCRIPT).is_file() else []
    if not names or entry is None:
        return None
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        codes = list(pool.map(lambda n: _run_one(checkpoint, entry, suite, n), names))
    if any(code not in (0, 1) for code in codes):
        return None
    return [name for name, code in zip(names, codes) if code == 1]


def _number(checkpoint: Path) -> int:
    return int(checkpoint.name.removeprefix("checkpoint_"))


def _checkpoint_verdicts(out: Path, run_dir: Path) -> dict[int, Optional[List[str]]]:
    """Re-test every checkpoint against the current tests; identical bytes are run once."""
    verdicts: dict[int, Optional[List[str]]] = {}
    by_digest: dict[str, Optional[List[str]]] = {}
    for cp in artifact_store.best_lineage(out):
        digest = (
            artifact_store.artifact_digest(cp)
            + artifact_store._tree_digest(cp / ".verification/evaluator")
            + str(artifact_store.read_record(cp).get("entry"))
        )
        if digest not in by_digest:
            try:
                by_digest[digest] = _failing_tests(cp, run_dir)
            except (OSError, subprocess.SubprocessError):
                by_digest[digest] = None
        verdicts[_number(cp)] = by_digest[digest]
    return verdicts


def _dir_bytes(path: Path) -> int:
    """Total size of the regular files under path, not following symlinks; 0 on any error."""
    try:
        return sum(f.lstat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())
    except OSError:
        return 0


def _reclaim_transient(run: Run) -> List[str]:
    """Delete the run's transient acquisition data now that the run is finished.

    The reference systems under specification/sources/ exist only for Phase 1 discovery:
    references/ holds what the synthesis loop actually reads, and the published result never
    contains them (see export_deliverable). Left in place they are hundreds of MB of .git history
    per clone, accumulating across runs. Symlinks into the shared clone cache are unlinked, never
    followed. Also drop any __pycache__ the run wrote into its dir."""
    lines: List[str] = []
    freed = 0
    if run.sources.is_dir():
        freed += _dir_bytes(run.sources)
        shutil.rmtree(run.sources, ignore_errors=True)
    for cache in list(run.path.rglob("__pycache__")):
        if cache.is_dir() and not cache.is_symlink():
            freed += _dir_bytes(cache)
            shutil.rmtree(cache, ignore_errors=True)
    if freed:
        lines.append(
            f"  reclaimed: freed {freed // (1024 * 1024)} MiB (specification/sources/, __pycache__)"
        )
    return lines


def finish(
    run_dir: Path,
    domain: Optional[str] = None,
    refresh_tests: bool = False,
    *,
    cleanup: bool = True,
) -> tuple[bool, List[str]]:
    """(ok, report lines). ok is False only if a new test could not be recorded, or no domain is
    known. cleanup deletes the run's cloned sources and bytecode; off when the run dir is about to
    be deleted whole."""
    run = Run(run_dir)
    domain = domain or run.domain()
    if not domain:
        return False, [
            f"run finish: no domain for {run_dir}. Put `domain: <name>` in the front matter of "
            f"{run.task}, or pass it: run finish <run> <domain>."
        ]
    store = Domain(domain)
    lines = [
        f"run finish: save what outlives the run for {run_dir}",
        f"domain: {domain!r} -> {store.path}",
        "",
    ]

    synced = kept_tests.sync(run_dir, domain, refresh_bodies=refresh_tests)
    suite = synced["suite"]
    recorded = synced["recorded"]
    refreshed = synced["refreshed"]
    held = synced["held"]
    present = synced["skipped"]
    missing: List[str] = []
    if not suite:
        # A completed test-driven run always earns tests. Zero means test authoring never ran or the
        # tests/ dir is misplaced -- saving "nothing" and printing PASSED would hide that, so fail.
        lines.append(f"FAILED: no tests found in {run.tests}; nothing to save.")
        return False, lines
    for gid in recorded:
        lines.append(f"  [record] {gid}  -> {store.tests}")
    for gid in refreshed:
        lines.append(f"  [refresh] {gid}  -> {store.tests}")
    for gid in held:
        lines.append(
            f"  [held   ] {gid}: the run's body differs from the knowledge base's validated "
            "copy; kept that one. Re-validate the new body, then `run finish --refresh-tests` "
            "to replace it deliberately."
        )
    # Verify every new test is now in the knowledge base.
    have = {g["id"] for g in kept_tests.tests_for(domain)}
    missing = [g for g in suite if g not in have]
    lines.append(
        f"  kept tests: {len(recorded)} newly recorded, {len(refreshed)} refreshed, "
        f"{len(held)} held back, {len(present)} already present, {len(suite)} in the run's suite"
    )

    if run.decision_log.is_file():
        target = kept_decisions(run_dir, domain)
        written, picked = save(Findings(str(run.decision_log)), target)
        lines.append(
            f"  decisions: saved {written}/{picked} row(s) that outlive the run -> {target.path}"
        )
    else:
        lines.append("  decisions: no decision log in the run dir; nothing to save")

    if missing:
        lines.append("")
        lines.append("FAILED: new tests missing from the knowledge base after sync:")
        lines += [f"  - {g}" for g in missing]
        return False, lines
    if cleanup:
        lines += _reclaim_transient(run)
    lines.append("")
    lines.append("PASSED: tests saved; later runs can reuse them after validation.")
    return True, lines


def _scored_candidate_count(run_dir: Path) -> Optional[int]:
    """Return the number of scored candidates in the leaderboard, or None if unreadable."""
    path = Run(run_dir).leaderboard
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, list):
        return None
    rows = [
        row
        for row in doc
        if isinstance(row, dict)
        and row.get("role", "candidate") == "candidate"
        and row.get("draw", "scored") == "scored"
    ]
    # One candidate may be measured several times (a re-measure after restoring a checkpoint's
    # source); count the distinct implementations, not the rows, so a checkpoint per candidate is
    # not reported as a skipped snapshot. Rows without a digest (older runs) count one each.
    digests = set()
    undigested = 0
    for row in rows:
        inputs = row.get("input_digests")
        if isinstance(inputs, dict) and inputs.get("implementation"):
            digests.add(inputs["implementation"])
        else:
            undigested += 1
    return len(digests) + undigested


def _provenance_lines(run_dir: Path, out: Path, checkpoints_before_publish: int) -> List[str]:
    """Warn when the result has fewer checkpoints than the leaderboard has scored candidates."""
    lines: List[str] = []
    if checkpoints_before_publish == 0:
        lines.append(
            "  checkpoints: WARNING: no checkpoints existed before publish; the build loop never "
            "ran `spec.checkpoint snapshot`. Per-iteration history survives only in the run dir's "
            "decision log, which is gitignored."
        )
    scored = _scored_candidate_count(run_dir)
    total = artifact_store.checkpoint_count(out)
    unscored = [
        cp.name
        for cp in sorted((out / "checkpoints").glob("checkpoint_*"))
        if not artifact_store.read_score(cp).get("score")
    ]
    if unscored:
        lines.append(
            "  checkpoints: WARNING: checkpoint(s) without a score: "
            f"{', '.join(unscored)}. A checkpoint is one evaluated iteration; its score.json "
            "carries the metrics of the leaderboard entry for these artifact bytes."
        )
    if scored is None:
        lines.append(
            "  checkpoints: the leaderboard is missing or unreadable; cannot verify that "
            f"checkpoints cover the scored iterations ({total} checkpoint(s) published)."
        )
    elif total < scored:
        lines.append(
            f"  checkpoints: WARNING: leaderboard records {scored} scored candidate(s) but the "
            f"result has only {total} checkpoint(s); per-iteration snapshots were skipped."
        )
    else:
        lines.append(
            f"  checkpoints: {total} checkpoint(s) cover the {scored} scored candidate(s)."
        )
    return lines


def _verify_export(best: Path, run: Run, *, production_ready=False) -> None:
    """Check the published copy of the result in a separate directory with its own suite."""
    private = best / ".verification"
    record = artifact_store.read_record(best)
    if record.get("inputs", {}).get("implementation") != artifact_store._tree_digest(run.impl):
        raise ValueError(
            "Restore the selected checkpoint's .verification/source into synthesis/impl, then measure and audit it again."
        )
    with tempfile.TemporaryDirectory() as tmp:
        staged = artifact_store._stage_artifact(run.impl, run.interface, Path(tmp))
        if artifact_store._tree_digest(staged) != artifact_store.artifact_digest(best):
            raise ValueError(
                "The selected artifact differs from the measured code or interface; checkpoint the measured version again."
            )
    entry = run.entry_impl()
    if (
        entry is not None
        and entry.resolve().relative_to(run.impl.resolve()).as_posix() != record["entry"]
    ):
        raise ValueError(
            "The selected entry differs from the measured entry; select its checkpoint explicitly."
        )
    why = artifact_store.audit_covers(run.completeness, run.impl)
    if why:
        raise ValueError(f"Final audit required: {why}")
    # Preserve the current verification inputs, not a second copy of the public presentation.
    bundle = Run(private / "run").create()
    for source, target in (
        (run.task, bundle.task),
        (run.specification, bundle.specification),
        (run.impl, bundle.impl),
        (run.evaluator, bundle.evaluator),
        (run.tests, bundle.tests),
        (run.audit, bundle.audit),
        (run.review, bundle.review),
        (run.leaderboard, bundle.leaderboard),
        (run.decision_log, bundle.decision_log),
        (run.report, bundle.report),
        (run.path / ".severity_snapshot.json", bundle.path / ".severity_snapshot.json"),
    ):
        if source.exists():
            # A symlink that escapes the run is refused before anything is copied.
            artifact_store._tree_digest(source, ignored=("sources", "profiling"))
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(
                    source,
                    target,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(
                        "__pycache__",
                        ".git",
                        ".pytest_cache",
                        "sources",
                        "profiling",
                        "*.pyc",
                        ".*.lock",  # bookkeeping lock sidecars, never a dependency lockfile
                    ),
                )
            else:
                shutil.copy2(source, target)
    expected = artifact_store.input_digests(run.path, audit=True)
    if artifact_store.input_digests(bundle.path, audit=True) != expected:
        raise ValueError(
            "Verification inputs changed while being copied; finish again after writers stop."
        )
    # The checkpoint's own measurement is the published score; history.json and spec.md read the
    # same file, so the three never disagree. Only a checkpoint saved before it was scored takes
    # the run's current measurement.
    current_score = artifact_store.score_from_run(run.path)
    score = artifact_store.read_score(best)
    if not score.get("score"):
        score["score"] = current_score["score"]
    if not score.get("baselines") and current_score.get("baselines"):
        score["baselines"] = current_score["baselines"]
    artifact_store._write_json(best / "score.json", score)
    # This catches literal dependencies on the old project; relocation is not an OS sandbox. The
    # project is the directory holding the runs folder (`.skydiscover/` by default); runs kept at
    # an absolute SKYDISCOVER_RUNS have no project path to match, and a filesystem root would
    # match every source file.
    project = project_of(run.path)
    if project is not None and project != Path(project.anchor):
        # Every delivered file, in any language. The copies under .verification/ (logs, decision
        # log, evaluator) may name the project; they record where things came from, not what a user takes away.
        for source in best.rglob("*"):
            if not source.is_file() or private in source.parents:
                continue
            if str(project) in source.read_text(encoding="utf-8", errors="replace"):
                raise ValueError(
                    f"Exported file refers to the original project: {source.relative_to(best)}"
                )
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "PYTHONPATH",
            "PYTHONHOME",
            "SKYDISCOVER_RUN",
            "SKYDISCOVER_IMPL",
            "SKYDISCOVER_HOME",
            "SKYDISCOVER_RUNS",
        }
    }
    with tempfile.TemporaryDirectory(prefix="synthesize-check-") as tmp:
        copied = Path(tmp) / "best"
        shutil.copytree(best, copied)
        command = [
            sys.executable,
            str(_RUN_TESTS),
            "--run",
            str(copied / ".verification/run"),
            "--check-defects",
            "--impl",
            str(copied / "artifact" / record["entry"]),
            "--suite",
            str(copied / "tests"),
        ]
        if production_ready:
            command.append("--production-ready")
        try:
            checked = subprocess.run(
                command,
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=int(os.environ.get("SKYDISCOVER_SLOW_SECS", "600") or "600"),
            )
        except subprocess.TimeoutExpired as exc:
            parts = [exc.stdout or b"", exc.stderr or b""]
            (private / "checks.log").write_text(
                "".join(p.decode(errors="replace") if isinstance(p, bytes) else p for p in parts)
                + "\nFinal checks timed out; working files kept.\n"
            )
            raise ValueError("Final checks timed out") from exc
        (private / "checks.log").write_text(checked.stdout + checked.stderr)
        if checked.returncode:
            raise ValueError("Exported checks failed")
        if artifact_store.artifact_digest(copied) != artifact_store.artifact_digest(
            best
        ) or artifact_store._tree_digest(copied / "tests") != artifact_store._tree_digest(
            best / "tests"
        ):
            raise ValueError("A check modified the result or its tests; nothing was published.")
        if artifact_store.input_digests(copied / ".verification/run", audit=True) != expected:
            raise ValueError("A check modified its verification inputs; nothing was published.")
    if artifact_store.input_digests(run.path, audit=True) != expected:
        raise ValueError(
            "Working inputs changed during verification; finish again after writers stop."
        )
    inputs = artifact_store.input_digests(run.path)
    measured = artifact_store._measured_row(artifact_store._read_json(run.leaderboard, []), inputs)
    artifact_store._write_json(
        private / "record.json",
        {
            **record,
            "inputs": inputs,
            "config": measured.get("config"),
            "direction": measured.get("direction", "max"),
        },
    )


def _unverified_history(out: Path, run_dir: Path, verdicts: dict) -> None:
    """Keep the page and machine-readable history consistent after a failed finish."""
    rows = artifact_store.write_history(out, verdicts, None)
    if (out / "best").is_dir():
        page = "Checks: not yet verified for final delivery.\n"
        try:
            record = artifact_store.read_record(out / "best")
            page = render_spec(
                run_dir,
                artifact_store.read_score(out / "best"),
                rows,
                measured_config=record.get("config"),
                direction=record.get("direction"),
            )
        finally:
            (out / "best/spec.md").write_text(page, encoding="utf-8")


def export_deliverable(run_dir: Path, dest: Path, *, production_ready=False) -> List[str]:
    """Publish the final checkpoint and best/ under `dest`."""
    run = Run(run_dir)
    artifact = run.impl
    if not artifact.is_dir():
        raise ValueError(f"No generated artifact in {artifact}; nothing was published.")

    out = artifact_store.output_for_run(run_dir, dest)
    if (out / "history.json").exists():
        _unverified_history(out, run_dir, {})
    checkpoints_before_publish = artifact_store.checkpoint_count(out)
    selected = artifact_store.selected_best_checkpoint(out)
    # finish is idempotent: do not create another checkpoint for identical artifact bytes.
    checkpoint = artifact_store.latest_checkpoint_for(run_dir, out)
    if checkpoint is None:
        # An unscored working copy must not be silently published.
        _, checkpoint = artifact_store.snapshot_run(
            run_dir, dest, became_best=selected is None, require_evaluation=not run.is_proof_run()
        )
    selected = artifact_store.selected_best_checkpoint(out) or checkpoint
    verdicts = _checkpoint_verdicts(out, run_dir)
    revalidation: List[str] = []
    if verdicts.get(_number(selected)) != []:
        passing = [cp for cp in artifact_store.best_lineage(out) if verdicts.get(_number(cp)) == []]
        original = artifact_store.read_record(selected)
        objective = next(iter(artifact_store.read_score(selected).get("score", {})), None)
        comparable = []
        for candidate in passing:
            record = artifact_store.read_record(candidate)
            value = artifact_store.read_score(candidate).get("score", {}).get(objective)
            if (
                not isinstance(record.get("config"), dict)
                or not isinstance(original.get("config"), dict)
                or record.get("config") != original.get("config")
                or record.get("direction", "max") != original.get("direction", "max")
                or any(
                    record.get("inputs", {}).get(k) != original.get("inputs", {}).get(k)
                    for k in ("task", "specification", "evaluator")
                )
                or type(value) not in (int, float)
                or not math.isfinite(value)
            ):
                _unverified_history(out, run_dir, verdicts)
                raise ValueError(
                    "Passing checkpoints have incomparable scores; remeasure them with one objective and configuration before selecting best."
                )
            comparable.append((value, _number(candidate), candidate))
        ranked = sorted(
            comparable,
            key=lambda x: (x[0] if original.get("direction", "max") == "max" else -x[0], x[1]),
            reverse=True,
        )
        replacement = ranked[0][2] if ranked else None
        if replacement is not None:
            revalidation.append(
                f"  export: WARNING: selected best {selected.name} fails the current test suite or could not be checked; "
                f"selected {replacement.name}, the best-scoring checkpoint that still passes."
            )
            selected = replacement
        else:
            _unverified_history(out, run_dir, verdicts)
            raise ValueError(
                f"No checkpoint passed the current tests; {selected.name} failed or could not be checked. "
                f"Working files were kept. See {out / 'history.json'}."
            )
    # Do not stamp history as successful until the actual staged copy passes.
    rows = artifact_store.history(out, verdicts, _number(selected))
    with tempfile.TemporaryDirectory(prefix=".finish-", dir=out) as tmp:
        best = artifact_store.publish_best(Path(tmp), selected, run_dir, rows)
        try:
            _verify_export(best, run, production_ready=production_ready)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            private = out / ".verification"
            private.mkdir(exist_ok=True)
            log = best / ".verification/checks.log"
            if log.exists():
                shutil.copy2(log, private / "checks.log")
            _unverified_history(out, run_dir, verdicts)
            # Say why another checkpoint was selected before saying what to do about it.
            why = " ".join(line.strip().removeprefix("export: ") for line in revalidation)
            raise ValueError(
                f"{why + ' ' if why else ''}{str(exc).rstrip('.')}. Working files kept; "
                f"selected checkpoint: {selected.name}; final logs: {private}"
            ) from exc
        record = artifact_store.read_record(best)
        (best / "spec.md").write_text(
            render_spec(
                run_dir,
                artifact_store.read_score(best),
                rows,
                measured_config=record.get("config"),
                direction=record.get("direction"),
            ),
            encoding="utf-8",
        )
        artifact_store._replace_dir(best, out / "best")
    artifact_store.write_history(out, verdicts, _number(selected))
    best = out / "best"
    return [
        f"  export: checkpoint -> {checkpoint}",
        f"  export: best -> {best}",
        *revalidation,
        *_provenance_lines(run_dir, out, checkpoints_before_publish),
    ]


def main_finish(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="run finish",
        description="Save what outlives a finished run (its new tests, the user's answers) into the "
        "domain's knowledge base. Run it at the end of every run.",
    )
    ap.add_argument("run_dir", help="the run's working dir, e.g. .skydiscover/<slug>")
    ap.add_argument(
        "domain",
        nargs="?",
        default=None,
        help="the knowledge base folder to save into; default: the `domain:` front matter of task.md",
    )
    ap.add_argument(
        "--export-to",
        default="",
        help="publish checkpoints/ + best/ beneath <path>/outputs/synthesize (or $SKYDISCOVER_OUTPUTS). Default off.",
    )
    ap.add_argument(
        "--refresh-tests",
        action="store_true",
        help="replace a kept test whose body changed in this run; without it the changed body is "
        "held back and the knowledge base keeps its validated copy",
    )
    ap.add_argument(
        "--keep-run",
        action="store_true",
        help="keep the run's working directory after publishing; by default it is deleted, and "
        ".skydiscover/<slug>.done records where the result went",
    )
    ap.add_argument(
        "--production-ready",
        action="store_true",
        help="also run the configured release checks on the export",
    )
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        marker = _done_marker(run_dir)
        if marker.is_file():
            raw = Path(marker.read_text().strip())
            result = raw if raw.is_absolute() else run_dir.resolve().parent.parent / raw
            rows = artifact_store._read_json(result / "history.json", [])
            if (
                isinstance(rows, list)
                and any(passed_final_tests(row) for row in rows)
                and (result / "best/artifact").is_dir()
            ):
                print(f"run finish: already finished; the result is at {result}")
                return 0
            print(f"run finish: saved result is missing or unverified: {result}", file=sys.stderr)
            return 2
        print(f"run finish: no run dir at {run_dir}; nothing to save.", file=sys.stderr)
        return 2

    # Validate the export before saving tests or decisions to the shared knowledge base.
    # A missing domain is reported by finish without publishing anything first.
    if args.export_to and (args.domain or Run(run_dir).domain()):
        try:
            print(
                "\n".join(
                    export_deliverable(
                        run_dir, Path(args.export_to), production_ready=args.production_ready
                    )
                )
            )
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            print(f"run finish: --export-to failed ({e}).", file=sys.stderr)
            return 1
    keep_run = not args.export_to or args.keep_run or Run(run_dir).is_proof_run()
    ok, lines = finish(run_dir, args.domain, refresh_tests=args.refresh_tests, cleanup=keep_run)
    print("\n".join(lines))
    if not ok:
        print(
            "\nRe-run once the tests are recordable; do not consider the run closed until it exits 0."
        )
        return 1
    if not args.export_to:
        print("  run dir: kept; nothing was published (pass --export-to to publish and clean up)")
        return 0
    if keep_run:
        print(f"  run dir: kept at {run_dir} (--keep-run or proof replay files)")
    else:
        print(delete_run(run_dir))
    return 0


def _done_marker(run_dir: Path) -> Path:
    return run_dir.parent / f"{run_dir.name}.done"


def delete_run(run_dir: Path) -> str:
    """Delete a published run's working directory; leave `<slug>.done` naming the result."""
    result = artifact_store.published_output(run_dir)
    if not Run(run_dir).task.is_file() or result is None:
        return f"  run dir: kept; {run_dir} has no task.md or no published result to point at"
    if not any(
        passed_final_tests(row) for row in artifact_store._read_json(result / "history.json", [])
    ):
        return f"  run dir: kept; no checkpoint passed the final tests"
    if result.resolve() == run_dir.resolve() or run_dir.resolve() in result.resolve().parents:
        raise ValueError("The result is inside the run directory; it cannot be deleted safely.")
    marker = _done_marker(run_dir)
    project = run_dir.resolve().parent.parent
    record = os.path.relpath(result, project) if result.is_relative_to(project) else str(result)
    marker.write_text(record + "\n", encoding="utf-8")
    shutil.rmtree(run_dir)
    return f"  run dir: deleted {run_dir}; the result is at {result} (recorded in {marker})"


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "check":
        return main_check(args[1:])
    if args and args[0] == "finish":
        return main_finish(args[1:])
    usage = (
        "usage: run check <run_dir> | run finish <run_dir> [domain] [--export-to <path>] "
        "[--keep-run] [--refresh-tests] [--production-ready]"
    )
    if args and args[0] in ("-h", "--help"):
        print(usage)
        return 0
    print(usage, file=sys.stderr)
    return 2


if __name__ == "__main__":
    from .paths import run_cli

    raise SystemExit(run_cli(main, "run"))
