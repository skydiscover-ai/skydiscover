from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from skydiscover.synthesize.spec import checkpoint as artifacts
from skydiscover.synthesize.spec import run as spec_run
from skydiscover.synthesize.spec.paths import Run


def _record_measurement(run_dir):
    run = Run(run_dir)
    rows = json.loads(run.leaderboard.read_text())
    inputs = artifacts.input_digests(run_dir)
    for row in rows:
        row["input_digests"] = inputs
        row.setdefault("config", {})
        row.setdefault("objective", next(iter(row["metrics"])))
        if row.get("role", "candidate") == "candidate":
            row["impl"] = run.entry_impl().relative_to(run.path).as_posix()
    run.leaderboard.write_text(json.dumps(rows))
    artifacts.stamp_audit(run_dir, [])


def _snapshot(run_dir, *args, **kwargs):
    _record_measurement(run_dir)
    return artifacts.snapshot_run(run_dir, *args, **kwargs)


def _write_run(tmp_path: Path, *, runnable: bool = False) -> Run:
    run = Run(tmp_path / "scratch" / "kv-store").create()
    (run.impl / "src").mkdir(parents=True)
    (run.impl / "src" / "store.cc").write_text("int value = 1;\n", encoding="utf-8")
    run.tests.mkdir()
    if runnable:
        run.reference.mkdir(parents=True)
        run.mutants.mkdir()
        (run.reference / "store.cc").write_text("int value = 1;\n")
        (run.mutants / "wrong.cc").write_text("int value = 0;\n")
        (run.tests / "check_roundtrip.cc").write_text(
            "extern int value; int main() { return value == 1 ? 0 : 1; }\n"
        )
        # a toolchain-free test.sh: the candidate passes iff its source still says value = 1
        run.test_script.write_text('grep -rq "value = 1" "$SKYDISCOVER_IMPL"\n', encoding="utf-8")
    run.bench.mkdir()
    run.leaderboard.write_text(
        json.dumps([{"metrics": {"throughput": 12.5}, "config": {"threads": 4}}]),
        encoding="utf-8",
    )
    return run


def test_a_checkpoint_is_the_artifact_and_its_score(tmp_path):
    run = _write_run(tmp_path)
    out, checkpoint = _snapshot(run.path, tmp_path, became_best=True)

    assert {path.name for path in out.iterdir()} == {
        "checkpoints",
        "best",
    }  # history.json: at finish
    assert {path.name for path in checkpoint.iterdir() if not path.name.startswith(".")} == {
        "artifact",
        "score.json",
        "tests.json",
    }
    assert (checkpoint / "artifact" / "src" / "store.cc").is_file()
    score = json.loads((checkpoint / "score.json").read_text())
    assert list(score) == ["checkpoint", "created_at", "score", "became_best"]
    assert score["checkpoint"] == 1
    assert score["score"] == {"throughput": 12.5}  # the first metric is the objective
    assert score["became_best"] is True


def test_snapshot_publishes_into_the_runs_project_whatever_the_cwd(tmp_path, monkeypatch):
    """The first snapshot pins where the result lives for the rest of the run, so the caller's
    working directory must not decide it: a run under <project>/.skydiscover/ publishes into that
    project; an explicit --export-root still wins."""
    monkeypatch.delenv("SKYDISCOVER_RUNS", raising=False)
    project = tmp_path / "project"
    run = Run(project / ".skydiscover" / "demo").create()
    (run.impl / "src").mkdir(parents=True)
    (run.impl / "src" / "store.cc").write_text("int value = 1;\n", encoding="utf-8")
    run.tests.mkdir()
    run.bench.mkdir()
    run.leaderboard.write_text(json.dumps([{"metrics": {"throughput": 1.0}}]), encoding="utf-8")
    _record_measurement(run.path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert artifacts.main(["snapshot", str(run.path)]) == 0
    out = artifacts.published_output(run.path)
    assert (
        out is not None and out.resolve().parent == (project / "outputs" / "synthesize").resolve()
    )
    assert not (elsewhere / "outputs").exists()
    assert not Path(run.output_pointer.read_text().strip()).is_absolute()  # project-relative

    other = Run(project / ".skydiscover" / "other").create()
    shutil.copytree(run.synthesis, other.synthesis, dirs_exist_ok=True)
    assert artifacts.main(["snapshot", str(other.path), "--export-root", str(elsewhere)]) == 0
    assert (
        artifacts.published_output(other.path).resolve().parent
        == (elsewhere / "outputs" / "synthesize").resolve()
    )


def test_each_checkpoint_lists_the_tests_it_was_scored_against(tmp_path):
    run = _write_run(tmp_path)
    (run.tests / "check_roundtrip.cc").write_text("// test\n", encoding="utf-8")
    out, first = _snapshot(run.path, tmp_path, became_best=True)
    (run.tests / "check_no_tombstone_leak.cc").write_text("// added later\n", encoding="utf-8")
    (run.impl / "src" / "store.cc").write_text("int value = 2;\n", encoding="utf-8")
    _, second = _snapshot(run.path, tmp_path)

    assert json.loads((first / "tests.json").read_text()) == ["check_roundtrip.cc"]
    assert json.loads((second / "tests.json").read_text()) == [
        "check_no_tombstone_leak.cc",
        "check_roundtrip.cc",
    ]
    rows = artifacts.history(out)  # what `run finish` writes to history.json, before its verdicts
    assert [row["checkpoint"] for row in rows] == [1, 2]
    assert list(rows[0]) == ["checkpoint", "created_at", "score", "became_best", "tests"]
    assert rows[0]["score"] == {"throughput": 12.5}
    assert (rows[0]["became_best"], rows[1]["became_best"]) == (True, False)
    assert (rows[0]["tests"], rows[1]["tests"]) == (1, 2)
    assert not (out / "history.json").exists()


def test_best_is_the_artifact_tests_score_and_spec_page(tmp_path):
    run = _write_run(tmp_path)
    (run.tests / "check_roundtrip.cc").write_text("// test\n", encoding="utf-8")
    out, _ = _snapshot(run.path, tmp_path, became_best=True)

    best = out / "best"
    assert {path.name for path in best.iterdir() if not path.name.startswith(".")} == {
        "artifact",
        "tests",
        "score.json",
        "spec.md",
    }
    assert (best / "tests" / "check_roundtrip.cc").is_file()
    assert json.loads((best / "score.json").read_text())["checkpoint"] == 1
    page = (best / "spec.md").read_text()
    assert page.startswith("# kv-store\n")  # no card yet: the run's slug names the page
    assert "**12.5 throughput**" in page
    assert "| Tests | 1 in `tests/` |" in page


def test_the_score_pairs_the_scored_candidate_with_the_scored_baseline(tmp_path):
    run = _write_run(tmp_path)
    run.leaderboard.write_text(
        json.dumps(
            [
                {"role": "baseline", "name": "stock", "metrics": {"throughput": 10.0, "p99_ms": 3}},
                {
                    "role": "baseline",
                    "name": "stock",
                    "draw": "held-out",
                    "metrics": {"throughput": 9.0},
                },
                {"role": "baseline", "name": "tuned", "metrics": {"throughput": 12.0}},
                {
                    "role": "candidate",
                    "baseline": "tuned",
                    "metrics": {"throughput": 12.5, "p99_ms": 2},
                },
                {"role": "candidate", "draw": "held-out", "metrics": {"throughput": 11.0}},
            ]
        ),
        encoding="utf-8",
    )
    out, checkpoint = _snapshot(run.path, tmp_path, became_best=True)

    score = json.loads((checkpoint / "score.json").read_text())
    assert score["score"] == {"throughput": 12.5}  # the scored draw, not the held-out check
    # Every comparable baseline, the one the candidate names first; the held-out draw is not one.
    assert score["baselines"] == [
        {"name": "tuned", "score": {"throughput": 12.0}},
        {"name": "stock", "score": {"throughput": 10.0}},
    ]
    assert list(score) == ["checkpoint", "created_at", "score", "baselines", "became_best"]
    assert "1.04× tuned (12), 1.25× stock (10)" in (out / "best" / "spec.md").read_text()


def test_the_score_uses_the_declared_numeric_objective(tmp_path):
    run = _write_run(tmp_path)
    run.leaderboard.write_text(
        json.dumps(
            [
                {
                    "metrics": {
                        "throughput": 12.5,
                    }
                }
            ]
        ),
        encoding="utf-8",
    )
    _, checkpoint = _snapshot(run.path, tmp_path)
    assert json.loads((checkpoint / "score.json").read_text())["score"] == {"throughput": 12.5}


def test_snapshot_makes_artifact_self_contained_and_ships_nothing_else(tmp_path):
    run = _write_run(tmp_path)
    # The candidate #includes the public contract, which lives outside impl/.
    (run.impl / "src" / "store.cc").write_text(
        '#include "store.h"\nint value = 1;\n', encoding="utf-8"
    )
    run.interface.mkdir(parents=True)
    (run.interface / "store.h").write_text("struct S {};\n", encoding="utf-8")
    (run.interface / "README.md").write_text("# the contract\n", encoding="utf-8")
    run.benchmark.mkdir(parents=True)
    (run.benchmark / "bench.cc").write_text("int main(){}\n", encoding="utf-8")

    out, checkpoint = _snapshot(run.path, tmp_path, became_best=True)

    # The contract is folded into artifact/ so it compiles standalone; the benchmark, the decision
    # log, and the audit stay in the run directory.
    assert (checkpoint / "artifact" / "src" / "store.cc").is_file()
    assert (checkpoint / "artifact" / "store.h").is_file()
    assert (out / "best" / "artifact" / "store.h").is_file()
    assert not (checkpoint / "artifact" / "README.md").exists()  # prose about the contract stays
    assert not (checkpoint / "harness").exists()
    assert not (out / "best" / "harness").exists()


def test_run_finish_recognises_an_already_checkpointed_candidate(tmp_path):
    run = _write_run(tmp_path)
    run.interface.mkdir(parents=True)
    (run.interface / "store.h").write_text("struct S {};\n", encoding="utf-8")

    out, checkpoint = _snapshot(run.path, tmp_path, became_best=True)

    assert (checkpoint / "artifact" / "store.h").is_file()  # the contract is folded in
    assert artifacts.latest_checkpoint_for(run.path, out) == checkpoint
    (run.impl / "src" / "store.cc").write_text("int value = 2;\n", encoding="utf-8")
    assert artifacts.latest_checkpoint_for(run.path, out) is None


def test_snapshot_contract_never_overwrites_a_builder_file(tmp_path):
    run = _write_run(tmp_path)
    run.interface.mkdir(parents=True)
    # A name clash: the coding agent's own file must win.
    (run.impl / "src" / "store.cc").write_text("CANDIDATE\n", encoding="utf-8")
    (run.interface / "src").mkdir(parents=True)
    (run.interface / "src" / "store.cc").write_text("CONTRACT\n", encoding="utf-8")

    _out, checkpoint = _snapshot(run.path, tmp_path, became_best=False)

    assert (checkpoint / "artifact" / "src" / "store.cc").read_text() == "CANDIDATE\n"


def test_snapshot_excludes_runtime_cache_files(tmp_path):
    run = _write_run(tmp_path)
    cache = run.impl / "__pycache__"
    cache.mkdir()
    (cache / "store.cpython-314.pyc").write_bytes(b"runtime cache")

    out, checkpoint = _snapshot(run.path, tmp_path, became_best=True)

    assert not (checkpoint / "artifact" / "__pycache__").exists()
    assert not (out / "best" / "artifact" / "__pycache__").exists()


def test_published_best_carries_no_bookkeeping_dotfiles_or_locks(tmp_path):
    # Lock sidecars are the run's own bookkeeping, never part of the deliverable.
    run = _write_run(tmp_path)
    (run.tests / "check_roundtrip.cc").write_text("// test\n", encoding="utf-8")
    (run.tests / ".index.lock").write_text("", encoding="utf-8")
    (run.tests / ".build.lock").write_text("", encoding="utf-8")
    (run.impl / "__pycache__").mkdir()
    (run.impl / "__pycache__/secret.pyc").write_bytes(b"x")
    (run.impl / "src" / "cache.lock").write_text("x", encoding="utf-8")

    out, checkpoint = _snapshot(run.path, tmp_path, became_best=True)

    best = out / "best"
    assert (best / "tests" / "check_roundtrip.cc").is_file()
    assert {p.name for p in (best / "tests").iterdir()} == {"check_roundtrip.cc"}
    assert not (best / "artifact" / ".secret").exists()
    assert (best / "artifact" / "src" / "cache.lock").exists()  # dependency lockfiles survive
    assert not (checkpoint / "artifact" / ".secret").exists()


def test_a_second_checkpoint_lands_in_the_same_result(tmp_path):
    run = _write_run(tmp_path)
    out, first = _snapshot(run.path, tmp_path)
    (run.impl / "src" / "store.cc").write_text("int value = 2;\n", encoding="utf-8")
    out2, second = _snapshot(run.path, tmp_path)

    assert out2.resolve() == out.resolve()
    assert (first.name, second.name) == ("checkpoint_1", "checkpoint_2")
    assert json.loads((second / "score.json").read_text())["checkpoint"] == 2


def test_best_is_materialized_from_an_immutable_checkpoint(tmp_path):
    run = _write_run(tmp_path)
    test = run.tests / "check_roundtrip.cc"
    test.write_text("// test\n", encoding="utf-8")
    out, checkpoint = _snapshot(run.path, tmp_path, became_best=True)

    best = artifacts.publish_best(out, checkpoint, run.path)
    assert (best / "artifact" / "src" / "store.cc").read_text() == "int value = 1;\n"
    assert (best / "tests" / test.name).is_file()
    assert (best / "score.json").read_text() == (checkpoint / "score.json").read_text()


def test_the_run_report_is_not_part_of_the_result(tmp_path):
    # Phase 3 writes <run>/report.md for the lead; the result speaks through spec.md and score.json.
    run = _write_run(tmp_path)
    run.report.write_text("# Final\n", encoding="utf-8")
    out, _ = _snapshot(run.path, tmp_path, became_best=True)

    assert not (out / "best" / "report.md").exists()
    assert not (out / "best" / "metadata.json").exists()


def test_postflight_export_uses_minimal_layout_and_is_idempotent(tmp_path):
    run = _write_run(tmp_path, runnable=True)

    _record_measurement(run.path)
    first = spec_run.export_deliverable(run.path, tmp_path / "published")
    second = spec_run.export_deliverable(run.path, tmp_path / "published")
    output = Path(first[0].split(" -> ", 1)[1]).parents[1]

    assert {path.name for path in output.iterdir()} == {"checkpoints", "best", "history.json"}
    assert len(list((output / "checkpoints").glob("checkpoint_*"))) == 1
    assert first[0] == second[0]
    assert (output / "best" / "artifact" / "src" / "store.cc").is_file()
    assert run.output_pointer.is_file()
    # the pointer is relative to the project, so a moved or copied project still finds its result
    raw = run.output_pointer.read_text().strip()
    assert not Path(raw).is_absolute() and artifacts.published_output(run.path) == output
    moved = tmp_path.parent / (tmp_path.name + "_moved")
    shutil.copytree(tmp_path, moved)
    assert artifacts.published_output(moved / "scratch" / "kv-store") == moved / raw
    shutil.rmtree(moved)


def test_run_finish_deletes_the_run_dir_and_leaves_a_done_marker(tmp_path, capsys):
    run = _write_run(tmp_path, runnable=True)
    run.task.write_text("---\ndomain: kv\n---\n# store\n", encoding="utf-8")
    home = tmp_path / "home"

    _record_measurement(run.path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SKYDISCOVER_HOME", str(home))
        assert spec_run.main_finish([str(run.path), "kv", "--export-to", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert not run.path.exists()
        marker = run.path.parent / "kv-store.done"
        result = tmp_path / marker.read_text().strip()
        assert (result / "best" / "spec.md").is_file() and (result / "history.json").is_file()
        assert "run dir: deleted" in out
        # finishing again is a no-op that points at the result
        assert spec_run.main_finish([str(run.path), "kv", "--export-to", str(tmp_path)]) == 0
        assert str(result) in capsys.readouterr().out


def test_run_finish_keep_run_and_no_export_leave_the_run_dir(tmp_path, capsys):
    run = _write_run(tmp_path, runnable=True)
    run.task.write_text("---\ndomain: kv\n---\n# store\n", encoding="utf-8")

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SKYDISCOVER_HOME", str(tmp_path / "home"))
        assert spec_run.main_finish([str(run.path), "kv"]) == 0
        assert run.path.is_dir() and "nothing was published" in capsys.readouterr().out
        _record_measurement(run.path)
        args = [str(run.path), "kv", "--export-to", str(tmp_path), "--keep-run"]
        assert spec_run.main_finish(args) == 0
        assert run.path.is_dir() and "--keep-run" in capsys.readouterr().out


def test_postflight_preserves_an_earlier_best_checkpoint(tmp_path):
    run = _write_run(tmp_path, runnable=True)
    out, first = _snapshot(run.path, tmp_path, became_best=True)
    (run.impl / "src" / "store.cc").write_text("int value = 2;\n", encoding="utf-8")
    _, second = _snapshot(run.path, tmp_path, became_best=False)

    (run.impl / "src/store.cc").write_text("int value = 1;\n")
    _record_measurement(run.path)
    spec_run.export_deliverable(run.path, tmp_path)

    assert first.name == "checkpoint_1"
    assert second.name == "checkpoint_2"
    assert (out / "best" / "artifact" / "src" / "store.cc").read_text() == "int value = 1;\n"
    assert json.loads((out / "best" / "score.json").read_text(encoding="utf-8"))["checkpoint"] == 1


needs_gpp = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not available")

_COUNT_H = "int count();\n"
_TEST_SH = """#!/usr/bin/env bash
cd "$(dirname "$0")"
tests=("$@"); [ ${#tests[@]} -eq 0 ] && tests=(*.cc)
out=$(mktemp -d); trap 'rm -rf "$out"' EXIT
rc=0
for t in "${tests[@]}"; do
  g++ -std=c++17 -I"$SKYDISCOVER_INTERFACE" "$t" "$SKYDISCOVER_IMPL" -o "$out/t" && "$out/t" || { echo "FAIL $t"; rc=1; }
done
exit $rc
"""
_GOOD = '#include "count.h"\nint count() { return 8; }\n'  # passes the floor test
_FAST_BROKEN = (
    '#include "count.h"\nint count() { return 4; }\n'  # faster-looking, fails a later test
)
_CHECK_POSITIVE = '#include "count.h"\nint main() { return count() > 0 ? 0 : 1; }\n'
_CHECK_FLOOR = '#include "count.h"\nint main() { return count() >= 8 ? 0 : 1; }\n'


def _write_compilable_run(tmp_path: Path, impl_src: str) -> Run:
    run = Run(tmp_path / "scratch" / "counter").create()
    run.impl.mkdir(parents=True)
    (run.impl / "counter.cc").write_text(impl_src, encoding="utf-8")
    run.interface.mkdir(parents=True)
    (run.interface / "count.h").write_text(_COUNT_H, encoding="utf-8")
    run.reference.mkdir(parents=True)
    run.mutants.mkdir()
    (run.reference / "counter.cc").write_text(_GOOD)
    (run.mutants / "wrong.cc").write_text('#include "count.h"\nint count() { return 0; }\n')
    run.tests.mkdir()
    run.test_script.write_text(_TEST_SH, encoding="utf-8")
    run.bench.mkdir()
    run.leaderboard.write_text(json.dumps([{"metrics": {"throughput": 1.0}}]), encoding="utf-8")
    return run


@needs_gpp
def test_export_demotes_a_best_that_a_later_gate_invalidates(tmp_path):
    """A candidate marked best when scored can be caught by a test validated later. Export re-runs the
    current suite against the selected best and demotes it to the newest earlier best that still
    passes, proven by compiling and running the tests."""
    run = _write_compilable_run(tmp_path, _GOOD)
    (run.tests / "check_positive.cc").write_text(_CHECK_POSITIVE, encoding="utf-8")
    _snapshot(run.path, tmp_path, became_best=True)  # checkpoint_1

    (run.impl / "counter.cc").write_text(_FAST_BROKEN, encoding="utf-8")
    _snapshot(run.path, tmp_path, became_best=True)  # checkpoint_2

    (run.tests / "check_floor.cc").write_text(_CHECK_FLOOR, encoding="utf-8")  # validated later
    (run.impl / "counter.cc").write_text(_GOOD)
    _record_measurement(run.path)
    lines = spec_run.export_deliverable(run.path, tmp_path)

    out = artifacts.output_for_run(run.path, tmp_path)
    assert json.loads((out / "best" / "score.json").read_text(encoding="utf-8"))["checkpoint"] == 1
    assert (out / "best" / "artifact" / "counter.cc").read_text() == _GOOD
    assert any("fails the current test suite" in line and "checkpoint_2" in line for line in lines)

    # history.json records the verdict for every checkpoint, so the plot can show what was flagged.
    rows = {row["checkpoint"]: row for row in json.loads((out / "history.json").read_text())}
    assert rows[1]["fails"] == [] and rows[1]["published"] is True
    assert rows[2]["fails"] == ["check_floor.cc"] and "best" not in rows[2]
    manifest = json.loads((out / "checkpoints" / "checkpoint_2" / "tests.json").read_text())
    assert manifest == ["check_positive.cc"]  # check_floor.cc did not exist yet
    page = (out / "best" / "spec.md").read_text(encoding="utf-8")
    assert "## Checkpoints" in page
    assert "| 1 | 1 | 1 | **best** |" in page
    assert "| 2 | 1 | 1 | fails `check_floor.cc` |" in page


def test_the_re_test_runs_each_test_by_name_and_reports_the_failing_ones(tmp_path):
    run = _write_run(tmp_path)
    run.test_script.write_text(
        'cd "$(dirname "$0")"\nfor t in "$@"; do grep -q "$(cat "$t")" "$SKYDISCOVER_IMPL" || exit 1; done\n',
        encoding="utf-8",
    )
    (run.tests / "one.t").write_text("value = 1")  # in the candidate
    (run.tests / "two.t").write_text("value = 2")  # not
    (run.tests / "sub").mkdir()
    (run.tests / "sub" / "helper.t").write_text("value = 9")  # a helper, not a test
    _, checkpoint = _snapshot(run.path, tmp_path)
    entry = artifacts.entry_in(checkpoint, run.path)
    assert spec_run._failing_tests(checkpoint, run.path) == ["two.t"]
    (run.tests / "two.t").write_text("value = 1")
    assert spec_run._failing_tests(checkpoint, run.path) == []
    run.test_script.unlink()
    assert spec_run._failing_tests(checkpoint, run.path) is None  # no test.sh: no verdict


def test_entry_in_a_checkpoint_follows_the_runs_entry_rule(tmp_path):
    # a dead end left beside the selected candidate in impl/: the newest leaderboard row decides.
    run = Run(tmp_path / "scratch" / "cache").create()
    run.impl.mkdir(parents=True)
    (run.impl / "tinylfu_lite.py").write_text("def create_cache(c): ...\n", encoding="utf-8")
    (run.impl / "two_q_lite.py").write_text("def create_cache(c): ...\n", encoding="utf-8")
    run.bench.mkdir()
    run.leaderboard.write_text(
        json.dumps(
            [
                {"impl": "synthesis/impl/tinylfu_lite.py", "metrics": {"app_hit_rate": 0.79}},
                {"impl": "synthesis/impl/two_q_lite.py", "metrics": {"app_hit_rate": 0.90}},
            ]
        ),
        encoding="utf-8",
    )
    _, checkpoint = _snapshot(run.path, tmp_path)
    assert artifacts.entry_in(checkpoint, run.path) == checkpoint / "artifact" / "two_q_lite.py"

    (run.impl / "two_q_lite.py").unlink()  # the sole remaining source is the entry
    assert artifacts.entry_in(checkpoint, run.path) == checkpoint / "artifact" / "two_q_lite.py"


def test_snapshot_carries_a_dotted_dependency_directory_and_skips_lock_sidecars(tmp_path):
    # A task's own `.data/` (the llm-router example keeps its traces there) is part of the
    # candidate: the copy and the digest apply the same rule, so the snapshot is not refused with
    # "omitted a dependency". A dotted lock sidecar is bookkeeping and is neither copied nor hashed.
    run = _write_run(tmp_path)
    (run.impl / ".data").mkdir()
    (run.impl / ".data" / "trace.jsonl").write_text('{"op": "get"}\n', encoding="utf-8")
    (run.impl / ".index.lock").write_text("", encoding="utf-8")

    _out, checkpoint = _snapshot(run.path, tmp_path, became_best=True)

    assert (checkpoint / "artifact" / ".data" / "trace.jsonl").read_text() == '{"op": "get"}\n'
    assert not (checkpoint / "artifact" / ".index.lock").exists()
    source = checkpoint / ".verification" / "source"
    assert artifacts._tree_digest(source) == artifacts._tree_digest(run.impl)


def test_copy_mismatch_names_the_file_that_differs(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "kept.py").write_text("x", encoding="utf-8")
    (src / "lost.bin").write_bytes(b"1")
    shutil.copytree(src, dst)
    (dst / "lost.bin").unlink()
    assert artifacts._copy_mismatch(src, dst) == ["lost.bin"]


def _set_measurement(run, value, *, direction=None):
    rows = json.loads(run.leaderboard.read_text())
    rows[0]["metrics"] = {"throughput": value}
    if direction:
        rows[0]["direction"] = direction
    run.leaderboard.write_text(json.dumps(rows), encoding="utf-8")


def test_became_best_is_checked_against_the_current_best(tmp_path):
    # The live llm-router run flagged a checkpoint scoring 0.3295 as best over one at 0.3419: the
    # flag was taken on trust. A worse score under the leaderboard's objective is refused; a tie
    # (the same bytes re-measured) and a better score are accepted.
    run = _write_run(tmp_path)
    out, first = _snapshot(run.path, tmp_path, became_best=True)
    assert artifacts.read_score(out / "best")["checkpoint"] == 1

    (run.impl / "src" / "store.cc").write_text("int value = 2;\n", encoding="utf-8")
    _set_measurement(run, 5.0)
    with pytest.raises(
        ValueError, match="not a new best: throughput 5.0 is worse than checkpoint_1's 12.5"
    ):
        _snapshot(run.path, tmp_path, became_best=True)
    assert artifacts.read_score(out / "best")["checkpoint"] == 1
    _out, second = _snapshot(run.path, tmp_path)  # without the flag it is recorded, not published
    assert second.name == "checkpoint_2"

    (run.impl / "src" / "store.cc").write_text("int value = 3;\n", encoding="utf-8")
    _set_measurement(run, 12.5)
    _out, third = _snapshot(run.path, tmp_path, became_best=True)  # a tie is allowed
    assert artifacts.read_score(out / "best")["checkpoint"] == 3

    (run.impl / "src" / "store.cc").write_text("int value = 4;\n", encoding="utf-8")
    _set_measurement(run, 20.0)
    _snapshot(run.path, tmp_path, became_best=True)
    assert artifacts.read_score(out / "best")["score"] == {"throughput": 20.0}


def test_became_best_follows_a_lower_is_better_objective(tmp_path):
    run = _write_run(tmp_path)
    _set_measurement(run, 12.5, direction="min")
    out, _first = _snapshot(run.path, tmp_path, became_best=True)
    (run.impl / "src" / "store.cc").write_text("int value = 2;\n", encoding="utf-8")
    _set_measurement(run, 20.0, direction="min")
    with pytest.raises(ValueError, match="direction min"):
        _snapshot(run.path, tmp_path, became_best=True)
    _set_measurement(run, 5.0, direction="min")
    _snapshot(run.path, tmp_path, became_best=True)
    assert artifacts.read_score(out / "best")["score"] == {"throughput": 5.0}


def test_a_baseline_stays_comparable_when_only_tests_or_decisions_changed(tmp_path, capsys):
    """Adding a test or a decision-log row does not move a benchmark score, so a baseline measured
    before it is still comparable; one measured against another evaluator is not, and a named
    baseline that is not comparable drops the margin instead of blocking the checkpoint."""
    run = _write_run(tmp_path)
    before = artifacts.input_digests(run.path)
    audited_before = artifacts.input_digests(run.path, audit=True)
    (run.tests / "check_more.py").write_text("def test_more():\n    pass\n", encoding="utf-8")
    (run.mutants / "m1").mkdir(parents=True)
    (run.mutants / "m1" / "impl.py").write_text("broken\n", encoding="utf-8")
    (run.evaluator / "reference").mkdir(exist_ok=True)
    (run.evaluator / "reference" / "impl.py").write_text("trusted\n", encoding="utf-8")
    after = artifacts.input_digests(run.path)
    assert after["tests"] != before["tests"] and after["evaluator"] == before["evaluator"]
    # the audit digest still sees the reference and the mutants: the auditor reviews them
    audited_after = artifacts.input_digests(run.path, audit=True)
    assert audited_after["evaluator"] != audited_before["evaluator"]
    (run.evaluator / "benchmark.py").write_text("changed\n", encoding="utf-8")
    assert artifacts.input_digests(run.path)["evaluator"] != after["evaluator"]
    (run.evaluator / "benchmark.py").unlink()
    impl = run.entry_impl().relative_to(run.path).as_posix()
    run.leaderboard.write_text(
        json.dumps(
            [
                {"role": "baseline", "name": "old-tests", "metrics": {"throughput": 10.0},
                 "objective": "throughput", "config": {}, "input_digests": before},
                {"role": "baseline", "name": "other-bench", "metrics": {"throughput": 1.0},
                 "objective": "throughput", "config": {},
                 "input_digests": {**after, "evaluator": "somethingelse"}},
                {"role": "candidate", "baseline": "other-bench", "metrics": {"throughput": 12.5},
                 "objective": "throughput", "config": {}, "input_digests": after, "impl": impl},
            ]
        ),
        encoding="utf-8",
    )  # fmt: skip
    artifacts.stamp_audit(run.path, [])
    _, checkpoint = artifacts.snapshot_run(run.path, tmp_path, became_best=True)
    score = json.loads((checkpoint / "score.json").read_text())
    assert score["baselines"] == [{"name": "old-tests", "score": {"throughput": 10.0}}]
    assert "other-bench" in capsys.readouterr().err


def test_one_entry_per_baseline_and_an_unnamed_row_goes_by_its_impl(tmp_path):
    """The evaluator re-measures the baselines every iteration, so many rows are comparable; the
    score keeps one per name: the row measured beside this candidate, else the newest. A row
    without a name takes the name another row gave the same impl (else the impl) rather than
    being refused."""
    run = _write_run(tmp_path)
    inputs = artifacts.input_digests(run.path)
    impl = run.entry_impl().relative_to(run.path).as_posix()
    other = {**inputs, "implementation": "an-earlier-candidate"}
    run.leaderboard.write_text(
        json.dumps(
            [
                {"role": "baseline", "name": "FIFO", "metrics": {"throughput": 9.0},
                 "objective": "throughput", "config": {}, "input_digests": other},
                {"role": "baseline", "name": "FIFO", "metrics": {"throughput": 10.0},
                 "objective": "throughput", "config": {}, "input_digests": inputs},
                {"role": "baseline", "name": "FIFO", "metrics": {"throughput": 9.5},
                 "objective": "throughput", "config": {}, "input_digests": other},
                {"role": "baseline", "name": "LRU", "impl": "workload.LRU",
                 "metrics": {"throughput": 7.0},
                 "objective": "throughput", "config": {}, "input_digests": other},
                {"role": "baseline", "impl": "workload.LRU", "metrics": {"throughput": 8.0},
                 "objective": "throughput", "config": {}, "input_digests": other},
                {"role": "baseline", "metrics": {"throughput": 1.0},
                 "objective": "throughput", "config": {}, "input_digests": inputs},
                {"role": "candidate", "baseline": "FIFO", "metrics": {"throughput": 12.5},
                 "objective": "throughput", "config": {}, "input_digests": inputs, "impl": impl},
            ]
        ),
        encoding="utf-8",
    )  # fmt: skip
    artifacts.stamp_audit(run.path, [])
    _, checkpoint = artifacts.snapshot_run(run.path, tmp_path, became_best=True)
    score = json.loads((checkpoint / "score.json").read_text())
    assert score["baselines"] == [
        {"name": "FIFO", "score": {"throughput": 10.0}},  # measured beside this candidate
        {"name": "LRU", "score": {"throughput": 8.0}},  # the newest; unnamed re-measure, same impl
    ]
