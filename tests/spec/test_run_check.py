"""Tests for `run check` (spec/run.py): the check that runs before the synthesis loop.

It fails when a required specification file is missing or empty and names each one; optional
cards are reported, never required.
"""

import json
import shutil

import pytest

from skydiscover.synthesize.spec import run as spec_run
from skydiscover.synthesize.spec.paths import Domain, Run

# Labels exactly as `run check` prints them (paths relative to the run directory).
TESTS, TESTS_JSON, SKELETON, ENVIRONMENT, WORKLOAD = (
    "specification/references/tests/",
    "specification/references/tests.json",
    "specification/references/skeleton.json",
    "specification/cards/environment.json",
    "specification/cards/workload.json",
)
SPEC_LABEL = "specification/references/*/spec.json"
VERIFY_LABEL = "specification/references/*/verification.json"


def _grounded_run(tmp_path, *, domain="demo") -> Run:
    """A run dir with every required file present and non-empty."""
    run = Run(tmp_path / ".skydiscover" / "demo").create()
    (run.sources / "reference-system" / ".git").mkdir(parents=True)
    run.reference_tests.mkdir(parents=True)
    (run.reference_tests / "reference_property_test.cc").write_text(
        "TEST(Property, HoldsUnderReplay)"
    )
    run.reference_tests_index.write_text(
        json.dumps(
            {"a-property": [{"system": "reference-system", "path": "src/prop_test.cc", "line": 1}]}
        )
    )
    run.skeleton.write_text(json.dumps({"shared": [{"id": "core", "label": "core"}]}))
    _write_extraction(
        run,
        "reference-system",
        {"source": "o/reference-system @ c1", "axes": {"a-property": {"property": "p"}}},
    )
    run.task.write_text(f"---\ndomain: {domain}\n---\n# Task\n")
    return run


def _write_extraction(run: Run, name, spec, *, verify=True):
    d = run.references / name
    d.mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps(spec))
    if verify:
        (d / "verification.json").write_text(json.dumps({"confirmed": 1, "method": "grep"}))


def _strip_clones(run: Run):
    for d in run.sources.iterdir():
        (d / ".git").rmdir()


def _cli(argv, capsys):
    code = spec_run.main_check(argv)
    return code, capsys.readouterr().out


# the run dir itself


def test_missing_run_dir_is_a_usage_error(tmp_path, capsys):
    assert spec_run.main_check([str(tmp_path / "nope")]) == 2
    assert "no run directory" in capsys.readouterr().err


def test_run_dir_that_is_a_file_is_a_usage_error(tmp_path, capsys):
    f = tmp_path / "afile"
    f.write_text("x")
    assert spec_run.main_check([str(f)]) == 2


# missing / empty


def test_empty_run_dir_fails_and_names_every_missing_file(tmp_path, capsys):
    run = Run(tmp_path / ".skydiscover" / "demo").create()
    code, out = _cli([str(run.path)], capsys)
    assert code == 1
    assert "FAILED" in out
    for artifact in (TESTS, TESTS_JSON, SKELETON):
        assert f"{artifact}: missing" in out
    assert "artifacts.md" in out  # the one pointer to what each file is


def test_grounded_run_passes_and_reports_the_optional_cards(tmp_path, capsys):
    code, out = _cli([str(_grounded_run(tmp_path).path)], capsys)
    assert code == 0
    assert "PASSED" in out
    for card in (ENVIRONMENT, WORKLOAD):
        assert card in out and "absent (optional)" in out
    assert "WARNING" not in out


def test_optional_cards_are_reported_present_when_written(tmp_path, capsys):
    run = _grounded_run(tmp_path)
    run.cards.mkdir(exist_ok=True)
    run.environment_card.write_text(json.dumps({"device": "x", "bottleneck": "bandwidth"}))
    code, out = _cli([str(run.path)], capsys)
    assert code == 0
    line = next(l for l in out.splitlines() if ENVIRONMENT in l)
    assert "present" in line


def test_no_real_clone_fails_even_when_otherwise_grounded(tmp_path, capsys):
    """Every other file present, but no git clone under sources/ and no reused source: a source list
    written from memory is not a reference system."""
    run = _grounded_run(tmp_path)
    _strip_clones(run)
    code, out = _cli([str(run.path)], capsys)
    assert code == 1
    assert "no git clone" in out


def _page_under(domain: str, name: str) -> None:
    """A repo page filed under *domain*, as an earlier run in that domain would leave it."""
    page = Domain(domain).wiki / "sources" / f"repo-{name}.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        f"---\nkind: repo\nurl: https://github.com/o/{name}\nsha: 0d17caa4\n---\nbody\n"
    )


def _claim_reuse(run: Run, page: str, name: str = "refsys") -> None:
    run.acquisitions.write_text(
        json.dumps({"reused_from_wiki": [{"name": name, "wiki_page": page}]})
    )


@pytest.mark.parametrize(
    "page_domain, task_domain",
    [
        ("demo", "demo"),  # the page is in this run's own domain
        (
            "kvstore",
            "demo",
        ),  # the clone hook refuses a download when ANY domain's wiki covers the repo
        ("kvstore", None),  # the domain line in task.md is optional
    ],
)
def test_wiki_reuse_counts_as_acquisition(tmp_path, capsys, monkeypatch, page_domain, task_domain):
    """A warm domain has no fresh clone; a source reused from the wiki, whose page pins url + sha,
    passes instead, wherever an earlier run filed the page."""
    run = _grounded_run(tmp_path)
    _strip_clones(run)
    if task_domain is None:
        run.task.write_text("# Task\n")
    monkeypatch.setenv("SKYDISCOVER_HOME", str(tmp_path / "home"))
    _page_under(page_domain, "refsys")
    _claim_reuse(run, "sources/repo-refsys.md")
    code, out = _cli([str(run.path)], capsys)
    assert code == 0
    assert "source reused from the wiki" in out


def test_wiki_reuse_without_the_page_on_disk_still_fails(tmp_path, capsys, monkeypatch):
    run = _grounded_run(tmp_path)
    _strip_clones(run)
    monkeypatch.setenv("SKYDISCOVER_HOME", str(tmp_path / "empty-home"))
    _claim_reuse(run, "sources/repo-refsys.md")
    code, out = _cli([str(run.path)], capsys)
    assert code == 1
    assert "no git clone" in out


@pytest.mark.parametrize("page_domain", ["demo", "kvstore"])
def test_wiki_reuse_cannot_be_faked_with_a_crafted_page_path(
    tmp_path, capsys, monkeypatch, page_domain
):
    """`wiki_page` is agent-controlled: a traversal path, an absolute path, or a glob must not match
    a file outside the wiki, and scanning every domain's wiki must not weaken that."""
    run = _grounded_run(tmp_path)
    _strip_clones(run)
    monkeypatch.setenv("SKYDISCOVER_HOME", str(tmp_path / "home"))
    _page_under(page_domain, "real")
    outside = tmp_path / "home" / "outside.md"
    outside.write_text("---\nurl: https://github.com/o/x\nsha: deadbeef\n---\n")
    for crafted in ("../outside.md", str(outside), "**/*.md", "sources/*.md"):
        _claim_reuse(run, crafted, name="x")
        code, out = _cli([str(run.path)], capsys)
        assert code == 1, crafted
        assert "no git clone" in out


def _only_a_placeholder(run: Run) -> None:
    for p in run.reference_tests.iterdir():
        p.unlink()
    (run.reference_tests / ".gitkeep").write_text("")


@pytest.mark.parametrize(
    "empty, label",
    [
        (_only_a_placeholder, TESTS),
        (lambda run: run.reference_tests_index.write_text(""), TESTS_JSON),
        (lambda run: run.skeleton.write_text("{}\n"), SKELETON),
    ],
    ids=["placeholder-only dir", "zero-byte file", "empty json"],
)
def test_an_empty_file_or_dir_is_reported_empty(tmp_path, capsys, empty, label):
    run = _grounded_run(tmp_path)
    empty(run)
    code, out = _cli([str(run.path)], capsys)
    assert code == 1
    assert f"{label}: empty" in out


def test_files_in_the_wrong_place_are_not_found(tmp_path, capsys):
    """There is exactly one place for each file; a file at the run root is not it."""
    run = _grounded_run(tmp_path)
    run.reference_tests_index.rename(run.specification / "tests.json")
    code, out = _cli([str(run.path)], capsys)
    assert code == 1
    assert f"{TESTS_JSON}: missing" in out


# reference specs


def test_a_reference_spec_with_source_and_axes_passes(tmp_path, capsys):
    run = _grounded_run(tmp_path)
    _write_extraction(
        run, "refsys", {"source": "o/refsys @ c1", "axes": {"consistency": {"property": "p"}}}
    )
    code, out = _cli([str(run.path)], capsys)
    assert code == 0
    assert SPEC_LABEL in out and "refsys" in out


def test_a_run_with_no_reference_spec_fails(tmp_path, capsys):
    run = _grounded_run(tmp_path)
    shutil.rmtree(run.references)
    code, out = _cli([str(run.path)], capsys)
    assert code == 1
    assert f"{SPEC_LABEL}: no spec.json with {{source, axes}}" in out


def test_a_spec_without_source_or_axes_does_not_count(tmp_path, capsys):
    run = _grounded_run(tmp_path)
    shutil.rmtree(run.references)
    _write_extraction(
        run,
        "libbloom",
        {"system": "libbloom", "repo": "o/libbloom", "properties": {}},
        verify=False,
    )
    code, out = _cli([str(run.path)], capsys)
    assert code == 1
    assert f"{SPEC_LABEL}: no spec.json with {{source, axes}}" in out
    assert "libbloom" in out


def test_a_spec_without_source_or_axes_beside_a_good_one_only_warns(tmp_path, capsys):
    run = _grounded_run(tmp_path)
    _write_extraction(
        run,
        "libbloom",
        {"system": "libbloom", "repo": "o/libbloom", "properties": {}},
        verify=False,
    )
    code, out = _cli([str(run.path)], capsys)
    assert code == 0
    assert "WARNING" in out and "libbloom" in out
    assert "PASSED" in out


def test_an_unreadable_spec_does_not_count(tmp_path, capsys):
    run = _grounded_run(tmp_path)
    shutil.rmtree(run.references)
    (run.references / "broken").mkdir(parents=True)
    (run.references / "broken" / "spec.json").write_text("{ not json")
    code, out = _cli([str(run.path)], capsys)
    assert code == 1
    assert f"{SPEC_LABEL}: no spec.json" in out and "broken" in out


def test_a_spec_without_verification_fails(tmp_path, capsys):
    run = _grounded_run(tmp_path)
    (run.references / "reference-system" / "verification.json").unlink()
    code, out = _cli([str(run.path)], capsys)
    assert code == 1
    assert f"{VERIFY_LABEL}: missing for reference-system" in out


# the function


def test_check_separates_problems_from_warnings(tmp_path):
    run = _grounded_run(tmp_path)
    ok, lines, problems, warnings = spec_run.check(run.path)
    assert ok and problems == [] and warnings == []

    run.reference_tests_index.unlink()
    ok, lines, problems, warnings = spec_run.check(run.path)
    assert not ok
    assert problems == [f"{TESTS_JSON}: missing"]
