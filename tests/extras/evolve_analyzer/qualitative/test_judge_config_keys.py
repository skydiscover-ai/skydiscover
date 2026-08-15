"""
Regression test for judge config key names in QualitativeAnalyzer.run().

Background
----------
Judges A (stagnation_root_cause) and B (artifact_clustering) were previously
read from the wrong config keys ("stagnation" and "artifact_clusters"), causing
them to ignore the user's explicit enable/disable settings.

What is tested
--------------
For every supported source (jsonl, skydiscover, shinkaevolve, openevolve):
  1. Setting ``judges.stagnation_root_cause = False`` suppresses judge A.
  2. Setting ``judges.artifact_clustering = False`` suppresses judge B.
  3. The old (wrong) key names have no effect on the dispatching logic.
  4. All six judge keys are read correctly when set to True/False.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from skydiscover.extras.evolve_analyzer.qualitative.qualitative_analyzer import QualitativeAnalyzer
from skydiscover.extras.evolve_analyzer.quantitative.bundle import QuantitativeBundle, StagnationPeriod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_SOURCES = ["jsonl", "skydiscover", "shinkaevolve", "openevolve"]

_JUDGE_METHODS = {
    "stagnation_root_cause": "_eval_stagnation_periods",
    "artifact_clustering":   "_eval_artifact_clusters",
    "mutation_quality":      "_eval_mutation_quality",
    "semantic_compliance":   "_eval_semantic_compliance",
    "exploration_structure": "_eval_exploration_structure",
    "meta_quality":          "_eval_meta_quality",
}


def _make_bundle() -> QuantitativeBundle:
    """Minimal QuantitativeBundle with an empty DataFrame and no alert periods."""
    bundle = QuantitativeBundle(df=pd.DataFrame())
    bundle.stagnation_periods = []
    return bundle


def _make_analyzer(judges_cfg: dict) -> QualitativeAnalyzer:
    config = {"judges": judges_cfg}
    return QualitativeAnalyzer(llm_client=MagicMock(), config=config)


def _run_with_all_patched(analyzer: QualitativeAnalyzer) -> dict[str, bool]:
    """
    Run analyzer.run() with every _eval_* method patched to a sentinel.
    Returns a dict mapping judge key → whether its method was called.
    """
    bundle = _make_bundle()
    patches = {}
    called = {}

    for key, method_name in _JUDGE_METHODS.items():
        mock = MagicMock(return_value=[])
        patches[key] = patch.object(analyzer, method_name, mock)
        called[key] = mock

    ctx_managers = list(patches.values())
    # Enter all patches
    for ctx in ctx_managers:
        ctx.__enter__()
    try:
        analyzer.run(bundle)
    finally:
        for ctx in ctx_managers:
            ctx.__exit__(None, None, None)

    return {key: called[key].called for key in _JUDGE_METHODS}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", ALL_SOURCES)
def test_stagnation_root_cause_key_disables_judge_a(source):
    """Setting stagnation_root_cause=False must suppress judge A."""
    analyzer = _make_analyzer({"stagnation_root_cause": False})
    was_called = _run_with_all_patched(analyzer)
    assert not was_called["stagnation_root_cause"], (
        f"[source={source}] Judge A ran despite stagnation_root_cause=False. "
        "Wrong config key used (old key was 'stagnation')."
    )


@pytest.mark.parametrize("source", ALL_SOURCES)
def test_artifact_clustering_key_disables_judge_b(source):
    """Setting artifact_clustering=False must suppress judge B."""
    analyzer = _make_analyzer({"artifact_clustering": False})
    was_called = _run_with_all_patched(analyzer)
    assert not was_called["artifact_clustering"], (
        f"[source={source}] Judge B ran despite artifact_clustering=False. "
        "Wrong config key used (old key was 'artifact_clusters')."
    )


@pytest.mark.parametrize("source", ALL_SOURCES)
def test_old_stagnation_key_has_no_effect(source):
    """The legacy key 'stagnation' must not control judge A dispatch."""
    # Set old wrong key to False, correct key to True → judge A should still run
    analyzer = _make_analyzer({"stagnation": False, "stagnation_root_cause": True})
    was_called = _run_with_all_patched(analyzer)
    assert was_called["stagnation_root_cause"], (
        f"[source={source}] Judge A was suppressed by the legacy 'stagnation' key — "
        "the code is still reading the wrong config key."
    )


@pytest.mark.parametrize("source", ALL_SOURCES)
def test_old_artifact_clusters_key_has_no_effect(source):
    """The legacy key 'artifact_clusters' must not control judge B dispatch."""
    analyzer = _make_analyzer({"artifact_clusters": False, "artifact_clustering": True})
    was_called = _run_with_all_patched(analyzer)
    assert was_called["artifact_clustering"], (
        f"[source={source}] Judge B was suppressed by the legacy 'artifact_clusters' key — "
        "the code is still reading the wrong config key."
    )


@pytest.mark.parametrize("source", ALL_SOURCES)
@pytest.mark.parametrize("judge_key,method_name", list(_JUDGE_METHODS.items()))
def test_each_judge_key_enables_its_method(source, judge_key, method_name):
    """Every judge key set to True causes exactly its method to be dispatched."""
    # Enable only this one judge
    cfg = {k: False for k in _JUDGE_METHODS}
    cfg[judge_key] = True
    analyzer = _make_analyzer(cfg)
    was_called = _run_with_all_patched(analyzer)
    assert was_called[judge_key], (
        f"[source={source}] Judge '{judge_key}' was not dispatched despite being enabled. "
        f"Expected method '{method_name}' to be called."
    )
    # All other judges must be suppressed
    for other_key in _JUDGE_METHODS:
        if other_key != judge_key:
            assert not was_called[other_key], (
                f"[source={source}] Judge '{other_key}' ran but should be disabled."
            )
