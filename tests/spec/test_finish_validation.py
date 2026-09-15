"""Finalization validates results before updating reusable knowledge."""

import json

import pytest

from skydiscover.synthesize.spec import checkpoint, kept_tests
from skydiscover.synthesize.spec import run as delivery
from skydiscover.synthesize.spec.paths import Domain, Run


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYDISCOVER_HOME", str(tmp_path / "knowledge"))
    run = Run(tmp_path / "project" / ".skydiscover" / "value").create()
    run.task.write_text("---\ndomain: integer-provider\n---\n# Integer provider\n")
    for directory in (
        run.impl,
        run.tests,
        run.interface,
        run.reference,
        run.mutants,
        run.bench,
        run.cards,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (run.impl / "value.py").write_text("VALUE = 7\n")
    (run.reference / "value.py").write_text("VALUE = 7\n")
    (run.mutants / "wrong.py").write_text("VALUE = 0\n")
    (run.tests / "check_value.py").write_text("from value import VALUE\nassert VALUE == 7\n")
    run.test_script.write_text(
        '#!/usr/bin/env bash\ncd "$(dirname "$0")"\n'
        'impl="$SKYDISCOVER_IMPL"; [ -d "$impl" ] || impl="$(dirname "$impl")"\n'
        'for t in *.py; do PYTHONPATH="$impl" python3 "$t" || exit 1; done\n'
    )
    return run


def _measure(run, config=None):
    run.leaderboard.write_text(
        json.dumps(
            [
                {
                    "impl": "synthesis/impl/value.py",
                    "metrics": {"throughput": 7},
                    "config": {"threads": 1} if config is None else config,
                    "input_digests": checkpoint.input_digests(run.path),
                }
            ]
        )
    )
    checkpoint.stamp_audit(run.path, [])


def test_failed_export_preserves_knowledge(run, tmp_path):
    kept_tests.sync(run.path, run.domain())
    domain = Domain(run.domain())
    original_test = (domain.tests / "check_value.py").read_bytes()
    original_index = domain.tests_index.read_bytes()
    (run.tests / "check_value.py").write_text("assert False\n")  # the candidate now fails a test
    _measure(run)

    assert (
        delivery.main_finish([str(run.path), "--export-to", str(tmp_path), "--refresh-tests"]) == 1
    )

    assert (domain.tests / "check_value.py").read_bytes() == original_test
    assert domain.tests_index.read_bytes() == original_index
    assert run.path.is_dir()
    assert not delivery._done_marker(run.path).exists()


@pytest.mark.parametrize("contents", ["{ broken json", "[]"])
def test_invalid_requirement_card_blocks_export(run, tmp_path, contents, capsys):
    run.requirements_card.write_text(contents)
    _measure(run)

    assert delivery.main_finish([str(run.path), "--export-to", str(tmp_path)]) == 1
    assert "requirements.json" in capsys.readouterr().err
    assert run.path.is_dir()
    assert not Domain(run.domain()).tests_index.exists()
    assert not delivery._done_marker(run.path).exists()


@pytest.mark.parametrize("config", [{"threads": 1}, {}])
def test_published_workload_uses_measured_configuration(run, tmp_path, config):
    run.workload_card.write_text(json.dumps({"scored_configuration": {"threads": 64}}))
    _measure(run, {"threads": 2})
    out, _ = checkpoint.snapshot_run(run.path, tmp_path, became_best=True)
    preview = (out / "best/spec.md").read_text()
    assert "| Threads | 64 |" not in preview
    assert "| Threads | 2 |" in preview
    _measure(run, config)

    assert delivery.main_finish([str(run.path), "--export-to", str(tmp_path)]) == 0

    page = (out / "best/spec.md").read_text()
    assert "Verified: every test in `tests/` passes against `artifact/`." in page
    assert "| Threads | 64 |" not in page
    assert ("| Threads | 1 |" in page) == bool(config)
    assert Domain("integer-provider").tests_index.exists()
    assert not run.path.exists()
    assert delivery._done_marker(run.path).is_file()


def test_invalid_card_on_retry_clears_previous_success(run, tmp_path):
    _measure(run)
    delivery.export_deliverable(run.path, tmp_path)
    out = checkpoint.published_output(run.path)
    assert (
        "Verified: every test in `tests/` passes against `artifact/`."
        in (out / "best/spec.md").read_text()
    )
    run.requirements_card.write_text("{ broken json")

    assert delivery.main_finish([str(run.path), "--export-to", str(tmp_path)]) == 1

    assert (
        "Verified: every test in `tests/` passes against `artifact/`."
        not in (out / "best/spec.md").read_text()
    )
    assert not any(row.get("published") for row in json.loads((out / "history.json").read_text()))
    assert not Domain(run.domain()).tests_index.exists()
    assert run.path.is_dir()


def test_missing_domain_does_not_publish(run, tmp_path):
    run.task.write_text("# Integer provider\n")
    _measure(run)
    assert delivery.main_finish([str(run.path), "--export-to", str(tmp_path)]) == 1
    assert not run.output_pointer.exists()
    assert run.path.is_dir()
