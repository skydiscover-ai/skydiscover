"""A config key SkyDiscover does not understand must be reported, not ignored.

Unknown keys used to split two ways, both bad: a nested one raised a bare
`TypeError: __init__() got an unexpected keyword argument`, and a top-level one
was dropped silently -- so a config could ask for a seed or for human feedback
and simply not get it. Ten shipped configs were doing exactly that.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
import yaml

from skydiscover.optimize.config import Config, ConfigError

ROOT = Path(__file__).resolve().parents[2]


def _write(tmp_path, payload) -> str:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(payload))
    return str(path)


@pytest.mark.parametrize(
    "payload,needle",
    [
        ({"llm": {"primary_model": "gpt-5"}}, "primary_model"),
        ({"evaluator": {"use_llm_feedback": True}}, "use_llm_feedback"),
        ({"monitor": {"hostname": "x"}}, "hostname"),
        ({"search": {"typo": 1}}, "typo"),
        ({"agentic": {"nope": 1}}, "nope"),
        ({"prompt": {"nope": 1}}, "nope"),
    ],
)
def test_unknown_nested_key_is_reported(payload, needle, tmp_path):
    with pytest.raises(ConfigError) as exc:
        Config.from_yaml(_write(tmp_path, payload))
    assert needle in str(exc.value)
    assert "Valid keys" in str(exc.value)


def test_unknown_top_level_key_is_not_silently_dropped(tmp_path):
    with pytest.raises(ConfigError) as exc:
        Config.from_yaml(_write(tmp_path, {"hil_enabled": True}))
    assert "hil_enabled" in str(exc.value)


def test_near_misses_get_a_suggestion(tmp_path):
    with pytest.raises(ConfigError) as exc:
        Config.from_yaml(_write(tmp_path, {"max_iteration": 10}))
    assert "did you mean 'max_iterations'?" in str(exc.value)


def test_sections_that_accept_extras_still_do():
    """Algorithm-specific database keys and benchmark params are passthrough by design."""
    config = Config.from_dict(
        {
            "search": {"type": "adaevolve", "database": {"some_engine_specific_knob": 3}},
            "benchmark": {"name": "b", "resolver": "r", "arbitrary_param": 7},
        }
    )
    assert config.search.database.some_engine_specific_knob == 3
    assert config.benchmark.params["arbitrary_param"] == 7


def test_every_config_shipped_in_the_repo_loads():
    """The strict check is only safe if nothing we ship trips it."""
    files = sorted(
        set(glob.glob(str(ROOT / "skydiscover/optimize/configs/*.yaml")))
        | set(glob.glob(str(ROOT / "benchmarks/**/config*.yaml"), recursive=True))
        | set(glob.glob(str(ROOT / "skydiscover/optimize/examples/**/config.yaml"), recursive=True))
    )
    assert len(files) > 30
    failures = []
    for path in files:
        try:
            Config.from_yaml(path)
        except Exception as exc:  # noqa: BLE001 - report every offender at once
            failures.append(f"{Path(path).relative_to(ROOT)}: {type(exc).__name__}")
    assert not failures, "shipped configs rejected by their own loader:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("field", ["models", "evaluator_models", "guide_models"])
def test_unknown_key_in_a_model_entry_is_reported(field, tmp_path):
    """Model entries are the most-edited part of a config; they get indexed errors."""
    payload = {"llm": {field: [{"name": "gpt-5"}, {"name": "gpt-5", "weght": 1.0}]}}
    with pytest.raises(ConfigError) as exc:
        Config.from_yaml(_write(tmp_path, payload))
    message = str(exc.value)
    assert f"llm.{field}[1]" in message, "the message must point at the offending entry"
    assert "did you mean 'weight'?" in message


def test_valid_model_lists_still_load(tmp_path):
    config = Config.from_yaml(
        _write(tmp_path, {"llm": {"models": [{"name": "gpt-5", "weight": 1.0}]}})
    )
    assert [m.name for m in config.llm.models] == ["gpt-5"]


def test_no_config_section_raises_a_bare_type_error(tmp_path):
    """Every dict-to-dataclass path must go through ConfigError, not TypeError."""
    probes = [
        {"llm": {"zzz": 1}},
        {"llm": {"models": [{"zzz": 1}]}},
        {"evaluator": {"zzz": 1}},
        {"agentic": {"zzz": 1}},
        {"monitor": {"zzz": 1}},
        {"prompt": {"zzz": 1}},
        {"search": {"zzz": 1}},
        {"zzz": 1},
    ]
    for payload in probes:
        with pytest.raises(ConfigError):
            Config.from_yaml(_write(tmp_path, payload))
