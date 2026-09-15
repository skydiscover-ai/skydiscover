"""The kept tests (spec/kept_tests.py) and `run finish`.

- the index is keyed by the domain slug, so spelling variants resolve to one domain;
- a record survives a concurrent writer (locked read-modify-write, per-writer temp file);
- an empty requirement set reports 100% reuse (nothing to miss), not 0%;
- a corrupt index is a hard error, never read as empty;
- a test-driven run that earned no tests fails `run finish` instead of passing vacuously.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from skydiscover.synthesize.spec import kept_tests
from skydiscover.synthesize.spec import run as spec_run
from skydiscover.synthesize.spec.paths import Domain, Run


@pytest.fixture
def lib(tmp_path, monkeypatch):
    """Isolated knowledge base root (~/.skydiscover stand-in) so tests never touch the real one."""
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setenv("SKYDISCOVER_HOME", str(root))
    return root


def _tests_dir(domain: str) -> Path:
    return Domain(domain).tests


def _seed_test(lib_root: Path, domain: str, gid: str) -> None:
    """Copy a real test file into the domain's tests/ and record it, the way sync would."""
    dest = _tests_dir(domain)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"check_{gid}.cc").write_text("// test\nint main(){return 0;}\n", encoding="utf-8")
    assert kept_tests.record(
        domain, {"id": gid, "property": "p", "keywords": [], "kind": "check", "catches": []}
    )


def _run_with_tests(tmp_path: Path, domain: str = "") -> Run:
    run = Run(tmp_path / "run").create()
    run.tests.mkdir(parents=True, exist_ok=True)
    if domain:
        run.task.write_text(f"---\ndomain: {domain}\n---\n# Task\n", encoding="utf-8")
    return run


def test_b1_slug_keying_resolves_spelling_variants(lib):
    # Save under the spaced spelling...
    _seed_test(lib, "example domain", "capacity_bound")
    # ...and every natural spelling must see it.
    for spelling in ("example domain", "example-domain", "Example Domain", "  example   domain "):
        assert len(kept_tests.tests_for(spelling)) == 1, spelling
        req = [{"id": "capacity_bound", "text": "x"}]
        assert [c["match"] for c in kept_tests.lookup(spelling, req)["candidates"]] == ["id"]
    # And the index physically lives under ONE folder, the slug, keeping the human spelling as label.
    assert sorted(p.name for p in lib.iterdir()) == ["example-domain"]
    idx = json.loads(Domain("example domain").tests_index.read_text())
    assert idx["label"] == "example domain"
    assert [g["source"] for g in idx["tests"]] == ["check_capacity_bound.cc"]  # bare filename


def test_empty_requirements_have_no_gaps(lib):
    assert kept_tests.lookup("anything", []) == {"domain": "anything", "candidates": [], "gaps": []}


def test_min14_corrupt_index_is_hard_error_not_empty(lib):
    _seed_test(lib, "example domain", "capacity_bound")
    Domain("example domain").tests_index.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(ValueError):
        kept_tests.tests_for("example domain")


def test_missing_index_is_empty_library(lib):
    assert kept_tests.tests_for("example domain") == []


def test_record_appends_without_dropping_existing_rows(lib):
    # record() re-reads the index under the lock and appends, so a second test into a domain that
    # already holds one keeps BOTH rows -- it never overwrites the file with only its own row.
    _seed_test(lib, "other domain", "a")
    # place file for the second test so record won't refuse it as dangling
    dest = _tests_dir("other domain")
    (dest / "check_b.cc").write_text("// test\nint main(){return 0;}\n", encoding="utf-8")
    assert kept_tests.record(
        "other domain", {"id": "b", "property": "p", "keywords": [], "kind": "check", "catches": []}
    )
    ids = {g["id"] for g in kept_tests.tests_for("other domain")}
    assert ids == {"a", "b"}  # neither row lost


def test_postflight_m9_empty_suite_fails(tmp_path, lib):
    run = _run_with_tests(tmp_path)  # tests/ exists but holds no check_*
    ok, lines = spec_run.finish(run.path, "example domain")
    assert ok is False
    assert any("no tests found" in ln.lower() for ln in lines)


def test_finish_reads_the_domain_from_task_front_matter(tmp_path, lib):
    run = _run_with_tests(tmp_path, domain="Example Domain")
    (run.tests / "check_capacity_bound.cc").write_text("// test\n", encoding="utf-8")
    ok, lines = spec_run.finish(run.path)
    assert ok is True, lines
    assert len(kept_tests.tests_for("example domain")) == 1
    assert (lib / "example-domain" / "tests" / "check_capacity_bound.cc").is_file()


def test_finish_without_a_domain_fails_with_a_clear_message(tmp_path, lib):
    run = _run_with_tests(tmp_path)
    (run.tests / "check_capacity_bound.cc").write_text("// test\n", encoding="utf-8")
    ok, lines = spec_run.finish(run.path)
    assert ok is False
    assert "no domain" in lines[0] and "task.md" in lines[0]
    assert list(lib.iterdir()) == []  # nothing saved anywhere


def test_finish_saves_a_real_suite(tmp_path, lib):
    run = _run_with_tests(tmp_path)
    run_dir = run.path
    (run.tests / "check_capacity_bound.cc").write_text(
        "// test\nint main(){return 0;}\n", encoding="utf-8"
    )
    ok, lines = spec_run.finish(run_dir, "example domain")
    assert ok is True
    assert len(kept_tests.tests_for("example domain")) == 1
    # idempotent: a second run records nothing new and still passes
    ok2, _ = spec_run.finish(run_dir, "example domain")
    assert ok2 is True
    assert len(kept_tests.tests_for("example domain")) == 1


def test_finish_reclaims_transient_sources_and_pycache(tmp_path, lib):
    # A finished run's reference systems (specification/sources/) and stray __pycache__ are
    # transient weight: references/ holds what the synthesis loop reads and the result never
    # contains sources/. finish prunes them; references/ and tests/ survive; finish still passes.
    run = _run_with_tests(tmp_path)
    (run.tests / "check_lru.py").write_text(
        "# Exact LRU eviction is preserved.\n", encoding="utf-8"
    )
    repo_git = run.sources / "caffeine" / ".git"
    repo_git.mkdir(parents=True)
    (repo_git / "pack").write_text("x" * 1024, encoding="utf-8")
    # a source reused from the shared clone cache is a symlink: unlinked, never followed
    cache_clone = tmp_path / "cache" / "guava"
    cache_clone.mkdir(parents=True)
    (cache_clone / "kept.java").write_text("// shared\n", encoding="utf-8")
    (run.sources / "guava").symlink_to(cache_clone)
    (run.references / "caffeine").mkdir(parents=True)
    (run.references / "caffeine" / "spec.json").write_text("{}\n", encoding="utf-8")
    (run.impl / "__pycache__").mkdir(parents=True)
    (run.impl / "__pycache__" / "m.pyc").write_text("bytecode", encoding="utf-8")

    ok, lines = spec_run.finish(run.path, "python-lru")

    assert ok, lines
    assert not run.sources.exists()  # cloned reference systems reclaimed
    assert (cache_clone / "kept.java").exists()  # the shared cache behind the symlink is untouched
    assert not (run.impl / "__pycache__").exists()  # stray bytecode reclaimed
    assert (run.references / "caffeine" / "spec.json").exists()  # what the loop reads survives
    assert (run.tests / "check_lru.py").exists()  # tests survive
    assert any("reclaimed" in ln for ln in lines)
    # idempotent: a second finish with sources/ already gone still passes
    ok2, _ = spec_run.finish(run.path, "python-lru")
    assert ok2


def _fake_clone(path: Path, url: str) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "remote", "add", "origin", url], check=True)
    (path / "pack").write_text("x" * 4096, encoding="utf-8")
    return path


def test_finish_keeps_shared_clones_that_other_runs_may_use(tmp_path, lib):
    """The shared clone cache is reused across runs and may be in use by another run right now;
    finish never touches it, pinned by a wiki page or not."""
    run = _run_with_tests(tmp_path)
    (run.tests / "check_lru.py").write_text("# Exact LRU eviction.\n", encoding="utf-8")
    cache = lib / ".cache" / "sources"
    unpinned = _fake_clone(cache / "redis-redis", "https://github.com/redis/redis.git")
    pinned = _fake_clone(cache / "ben-manes-caffeine", "git@github.com:ben-manes/caffeine.git")
    page = lib / "python-lru" / "wiki" / "sources" / "repo-caffeine.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\nkind: repo\nid: repo-caffeine\nurl: https://github.com/ben-manes/caffeine\n"
        "sha: abc123\n---\n",
        encoding="utf-8",
    )

    ok, lines = spec_run.finish(run.path, "python-lru")
    assert ok, lines
    assert unpinned.exists() and pinned.exists()


def test_finish_saves_python_gates(tmp_path, lib):
    run = _run_with_tests(tmp_path)
    (run.tests / "check_lru.py").write_text(
        "# Exact LRU eviction is preserved.\n", encoding="utf-8"
    )

    ok, lines = spec_run.finish(run.path, "python-lru")

    assert ok, lines
    assert len(kept_tests.tests_for("python-lru")) == 1


def test_sync_copies_the_test_script_with_the_tests(tmp_path, lib):
    run = _run_with_tests(tmp_path)
    (run.tests / "lru.py").write_text("# Exact LRU eviction is preserved.\n", encoding="utf-8")
    run.test_script.write_text("exit 0\n", encoding="utf-8")

    first = kept_tests.sync(run.path, "python-lru")
    run.test_script.write_text("exit 1\n", encoding="utf-8")  # the latest run's script wins
    second = kept_tests.sync(run.path, "python-lru")

    assert first["recorded"] == ["lru"] and second["skipped"] == ["lru"]
    kept = Domain("python-lru").tests
    assert (kept / "test.sh").read_text() == "exit 1\n"
    assert [g["id"] for g in kept_tests.tests_for("python-lru")] == ["lru"], "test.sh is not a test"


def test_sync_copies_helper_subdirectories_but_does_not_record_them_as_tests(tmp_path, lib):
    run = _run_with_tests(tmp_path)
    (run.tests / "lru.py").write_text("# Exact LRU eviction is preserved.\n", encoding="utf-8")
    run.test_script.write_text("exit 0\n", encoding="utf-8")
    (run.tests / "helpers").mkdir()
    (run.tests / "helpers" / "fixture.py").write_text("def make(): pass\n", encoding="utf-8")
    (run.tests / "helpers" / "__pycache__").mkdir()
    (run.tests / "helpers" / "__pycache__" / "fixture.pyc").write_bytes(b"\x00")

    kept_tests.sync(run.path, "python-lru")
    (run.tests / "helpers" / "fixture.py").write_text("def make(): return 1\n", encoding="utf-8")
    kept_tests.sync(run.path, "python-lru")  # the latest run's helpers win, like test.sh

    kept = Domain("python-lru").tests
    assert (kept / "helpers" / "fixture.py").read_text() == "def make(): return 1\n"
    assert not (kept / "helpers" / "__pycache__").exists()
    assert [g["id"] for g in kept_tests.tests_for("python-lru")] == ["lru"]


def test_lookup_suggests_exact_id_but_requires_validation(lib):
    # A matching ID suggests a test; it does not prove the current requirement.
    _seed_test(lib, "example domain", "capacity_bound")
    required = [
        {"id": "capacity_bound", "text": "the structure never exceeds its declared bound"},
        {"id": "frobnicate", "text": "the quux must wibble the wobble"},  # nothing like it
    ]
    r = kept_tests.lookup("example domain", required)
    assert [(c["requirement"]["id"], c["match"]) for c in r["candidates"]] == [
        ("capacity_bound", "id")
    ]
    assert {g["id"] for g in r["gaps"]} == {"capacity_bound", "frobnicate"}


def test_record_refuses_a_row_without_its_file_and_is_idempotent(lib, tmp_path):
    row = {
        "id": "no_torn_read",
        "property": "no torn value on read",
        "keywords": [],
        "kind": "check",
        "catches": [],
    }
    assert (
        kept_tests.record("example domain", row) is False
    )  # no file in the kept tests for this id
    dest = _tests_dir("example domain")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "check_no_torn_read.cc").write_text(
        "// test\nint main(){return 0;}\n", encoding="utf-8"
    )
    assert kept_tests.record("example domain", row) is True
    r = kept_tests.lookup(
        "example domain", [{"id": "no_torn_read", "text": "no torn value on read"}]
    )
    assert r["candidates"][0]["requirement"]["id"] == "no_torn_read"
    kept_tests.record(
        "example domain", {**row, "property": "dup"}
    )  # a re-record does not duplicate
    assert len(kept_tests.tests_for("example domain")) == 1


def test_sync_holds_back_a_changed_test_body(tmp_path, lib):
    """A run cannot silently replace a validated kept test with a weaker body: the
    library keeps its copy, the change is reported, and lookup still serves the validated test."""
    run = _run_with_tests(tmp_path)
    test = run.tests / "check_lru.py"
    test.write_text("# Exact LRU eviction is preserved.\nassert order == expected\n")
    first = kept_tests.sync(run.path, "python-lru")
    assert first["recorded"] == ["lru"]

    test.write_text("# Exact LRU eviction is preserved.\nassert True\n")  # the weakening
    second = kept_tests.sync(run.path, "python-lru")
    assert second["held"] == ["lru"] and second["refreshed"] == []
    lib_body = (kept_tests._store("python-lru").tests / "check_lru.py").read_text()
    assert "order == expected" in lib_body  # the validated body survived


def test_refresh_tests_flag_replaces_a_body_deliberately(tmp_path, lib):
    """The explicit path still exists: after re-validation, finish --refresh-tests replaces the
    library body."""
    run = _run_with_tests(tmp_path)
    test = run.tests / "check_lru.py"
    test.write_text("# Exact LRU eviction is preserved.\nassert order == expected\n")
    kept_tests.sync(run.path, "python-lru")

    test.write_text("# Exact LRU eviction is preserved.\nassert order == tuple(expected)\n")
    ok, lines = spec_run.finish(run.path, "python-lru", refresh_tests=True)
    assert ok, lines
    lib_body = (kept_tests._store("python-lru").tests / "check_lru.py").read_text()
    assert "tuple(expected)" in lib_body


def test_finish_reports_a_held_test_loudly(tmp_path, lib):
    run = _run_with_tests(tmp_path)
    (run.tests / "check_lru.py").write_text("# Exact LRU eviction is preserved.\nassert a\n")
    kept_tests.sync(run.path, "python-lru")
    (run.tests / "check_lru.py").write_text("# Exact LRU eviction is preserved.\nassert True\n")

    ok, lines = spec_run.finish(run.path, "python-lru")
    assert ok, lines
    held_line = next(ln for ln in lines if "[held" in ln)
    assert "--refresh-tests" in held_line


def test_scored_candidate_count_counts_implementations_not_measurements(tmp_path):
    # A checkpoint's source restored and re-measured (what a demotion refusal asks for) adds a
    # leaderboard row for bytes that already have a checkpoint; provenance must not report that
    # as a skipped snapshot.
    run = Run(tmp_path / "project" / ".skydiscover" / "demo").create()
    run.bench.mkdir(parents=True, exist_ok=True)
    rows = [
        {"impl": "impl", "metrics": {"ops": 8.0}, "input_digests": {"implementation": "sha256:a"}},
        {"impl": "impl", "metrics": {"ops": 9.0}, "input_digests": {"implementation": "sha256:b"}},
        {"impl": "impl", "metrics": {"ops": 7.5}, "input_digests": {"implementation": "sha256:a"}},
        {"impl": "impl", "metrics": {"ops": 1.0}},  # an older row without digests counts once
        {
            "impl": "impl",
            "metrics": {},
            "role": "baseline",
            "input_digests": {"implementation": "x"},
        },
    ]
    run.leaderboard.write_text(json.dumps(rows), encoding="utf-8")
    assert spec_run._scored_candidate_count(run.path) == 3


def test_run_help_exits_zero_and_a_bad_subcommand_exits_two(capsys):
    """`--help` is a successful call; an unknown subcommand is not. Both print the same usage."""
    assert spec_run.main(["--help"]) == 0
    assert "usage: run check" in capsys.readouterr().out
    assert spec_run.main(["bogus"]) == 2
    assert "usage: run check" in capsys.readouterr().err
