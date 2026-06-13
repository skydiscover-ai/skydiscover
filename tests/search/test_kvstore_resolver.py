"""Tests for the kvstore benchmark family: the resolver and the task config.

These run anywhere — they don't build C++ or run the evaluator (that needs the
shared _harness + a build box). They lock in that a checked-in task resolves to
its files and that the task config parses against the real SkyDiscover schema.
"""

from pathlib import Path

import pytest

from skydiscover.config import Config, JitsKitConfig

_FAMILY = Path(__file__).resolve().parents[1].parent / "benchmarks" / "kvstore"
_TASK = "0001_ycsb50_zipf_8gb"


def _resolver():
    from benchmarks.kvstore.resolver import resolver

    return resolver


def test_resolver_resolves_checked_in_task(tmp_path):
    res = _resolver().resolve({"task": _TASK}, output_dir=tmp_path)
    assert res.initial_program_path.endswith(f"{_TASK}/initial_program.cc")
    assert Path(res.initial_program_path).exists()
    assert res.evaluator_path.endswith(f"{_TASK}/evaluator")
    assert Path(res.evaluator_path).is_dir()


def test_resolver_forwards_spec_as_env(tmp_path):
    res = _resolver().resolve(
        {"task": _TASK, "workload": "50:50", "distribution": "zipf(0.99)", "value_size": 100},
        output_dir=tmp_path,
    )
    assert res.evaluator_env_vars["SKYKV_WORKLOAD"] == "50:50"
    assert res.evaluator_env_vars["SKYKV_DISTRIBUTION"] == "zipf(0.99)"
    assert res.evaluator_env_vars["SKYKV_VALUE_SIZE"] == "100"


def test_resolver_requires_task_id(tmp_path):
    with pytest.raises(ValueError, match="task"):
        _resolver().resolve({}, output_dir=tmp_path)


def test_resolver_rejects_unknown_task(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolver().resolve({"task": "9999_does_not_exist"}, output_dir=tmp_path)


def test_task_config_parses_against_real_schema():
    """The task config.yaml must load and route jitskit knobs to search.database."""
    cfg = Config.from_yaml(str(_FAMILY / _TASK / "config.yaml"))
    assert cfg.language == "cpp"
    assert cfg.search.type == "jitskit"
    db = cfg.search.database
    assert isinstance(db, JitsKitConfig)
    assert db.workload == "50:50"
    assert db.distribution == "zipf(0.99)"
    assert db.mem_budget_gb == [8]
    assert db.threads == [16]
    # the resolver + spec live under benchmark
    assert cfg.benchmark.resolver == "benchmarks.kvstore.resolver"
    assert cfg.benchmark.params["task"] == _TASK


def test_task_seed_has_evolve_markers_and_factory():
    seed = (_FAMILY / _TASK / "initial_program.cc").read_text()
    assert "// EVOLVE-BLOCK-START" in seed
    assert "// EVOLVE-BLOCK-END" in seed
    assert "create_kvstore()" in seed  # implements the IKVStore factory
