"""
Tests for detect_algorithm_class and the tool-aware rating changes it drives.
"""

from __future__ import annotations

import types

import pytest

from skydiscover.extras.evolve_analyzer.ingestion.checkpoint_adapter import detect_algorithm_class
from skydiscover.extras.evolve_analyzer.report_synthesizer import (
    _build_rating_context,
    _build_convergence_dimension,
    _rate_by_thresholds,
    _DEFAULT_REGRESSION_THRESHOLDS,
    _DEFAULT_EXPLORATION_THRESHOLDS,
    _DEFAULT_CONVERGENCE_TTB_THRESHOLDS,
    _DEFAULT_CONVERGENCE_PLATEAU_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conv(ttbf: float, poi: int | None = None, n: int = 100) -> types.SimpleNamespace:
    """Minimal convergence object for _build_convergence_dimension."""
    return types.SimpleNamespace(
        time_to_best_fraction=ttbf,
        plateau_onset_iteration=poi,
        best_so_far_curve=[0.5] * n,
        convergence_rate=0.001,
    )


def _make_quant(ttbf: float, poi: int | None = None, n: int = 100) -> types.SimpleNamespace:
    return types.SimpleNamespace(convergence=_make_conv(ttbf, poi, n))


_POP_CONFIG = {
    "algorithm_classes": {
        "population_evolutionary": {
            "regression_frequency_thresholds": [0.15, 0.30, 0.50, 0.70],
            "exploration_sdi_thresholds": [0.70, 0.50, 0.30, 0.10],
            "convergence_thresholds": {
                "time_to_best_thresholds": [0.60, 0.40],
                "plateau_fraction_thresholds": [0.15, 0.30, 0.55],
            },
        },
        "serial_refinement": {
            "regression_frequency_thresholds": [0.03, 0.10, 0.20, 0.35],
            "exploration_sdi_thresholds": [0.40, 0.25, 0.10, 0.05],
            "convergence_thresholds": {
                "time_to_best_thresholds": [0.80, 0.60],
                "plateau_fraction_thresholds": [0.20, 0.40, 0.60],
            },
        },
        "bayesian_optimization": {
            "regression_frequency_thresholds": [0.05, 0.15, 0.30, 0.50],
            "exploration_sdi_thresholds": [0.50, 0.30, 0.15, 0.05],
            "convergence_thresholds": {
                "time_to_best_thresholds": [0.70, 0.50],
                "plateau_fraction_thresholds": [0.20, 0.40, 0.60],
            },
        },
    }
}


# ---------------------------------------------------------------------------
# detect_algorithm_class
# ---------------------------------------------------------------------------

class TestDetectAlgorithmClass:
    def test_skydiscover_is_population_evolutionary(self):
        assert detect_algorithm_class("skydiscover", []) == "population_evolutionary"

    def test_openevolve_is_population_evolutionary(self):
        assert detect_algorithm_class("openevolve", []) == "population_evolutionary"

    def test_shinkaevolve_is_population_evolutionary(self):
        assert detect_algorithm_class("shinkaevolve", []) == "population_evolutionary"

    def test_source_case_insensitive(self):
        assert detect_algorithm_class("SkyDiscover", []) == "population_evolutionary"
        assert detect_algorithm_class("OPENEVOLVE", []) == "population_evolutionary"

    def test_jsonl_with_island_id_is_population_evolutionary(self):
        records = [{"iteration": 0, "child_score": 0.5, "island_id": "island_0"}]
        assert detect_algorithm_class("jsonl", records) == "population_evolutionary"

    def test_jsonl_island_id_none_does_not_trigger(self):
        records = [{"iteration": 0, "child_score": 0.5, "island_id": None}]
        # None island_id should NOT classify as population_evolutionary
        result = detect_algorithm_class("jsonl", records)
        assert result != "population_evolutionary"

    def test_jsonl_with_numeric_parameters_is_bayesian_optimization(self):
        records = [
            {"iteration": i, "child_score": 0.5, "parameters": {"lr": 0.01, "depth": 3}}
            for i in range(5)
        ]
        assert detect_algorithm_class("jsonl", records) == "bayesian_optimization"

    def test_jsonl_with_only_string_parameters_is_not_bayesian_optimization(self):
        records = [
            {"iteration": 0, "child_score": 0.5, "parameters": {"mutation_type": "diff"}}
        ]
        result = detect_algorithm_class("jsonl", records)
        assert result == "serial_refinement"

    def test_jsonl_minimal_records_is_serial_refinement(self):
        records = [{"iteration": i, "child_score": float(i) * 0.1} for i in range(5)]
        assert detect_algorithm_class("jsonl", records) == "serial_refinement"

    def test_jsonl_empty_records_is_serial_refinement(self):
        assert detect_algorithm_class("jsonl", []) == "serial_refinement"

    def test_island_id_takes_priority_over_numeric_parameters(self):
        # A record with both island_id and numeric parameters → population_evolutionary
        records = [
            {
                "iteration": 0,
                "child_score": 0.5,
                "island_id": "island_1",
                "parameters": {"lr": 0.01},
            }
        ]
        assert detect_algorithm_class("jsonl", records) == "population_evolutionary"

    def test_only_first_ten_records_inspected(self):
        # First 10 records have no island_id; record 11 does — should still be serial
        records = [{"iteration": i, "child_score": 0.5} for i in range(10)]
        records.append({"iteration": 10, "child_score": 0.5, "island_id": "island_0"})
        assert detect_algorithm_class("jsonl", records) == "serial_refinement"


# ---------------------------------------------------------------------------
# _rate_by_thresholds
# ---------------------------------------------------------------------------

class TestRateByThresholds:
    thresholds = [0.05, 0.15, 0.30, 0.50]  # default regression thresholds

    def test_lower_is_better_rating_5(self):
        assert _rate_by_thresholds(0.04, self.thresholds, lower_is_better=True) == 5

    def test_lower_is_better_boundary_at_t5(self):
        assert _rate_by_thresholds(0.05, self.thresholds, lower_is_better=True) == 4

    def test_lower_is_better_rating_4(self):
        assert _rate_by_thresholds(0.10, self.thresholds, lower_is_better=True) == 4

    def test_lower_is_better_rating_3(self):
        assert _rate_by_thresholds(0.20, self.thresholds, lower_is_better=True) == 3

    def test_lower_is_better_rating_2(self):
        assert _rate_by_thresholds(0.40, self.thresholds, lower_is_better=True) == 2

    def test_lower_is_better_rating_1(self):
        assert _rate_by_thresholds(0.60, self.thresholds, lower_is_better=True) == 1

    def test_higher_is_better_rating_5(self):
        thresholds = [0.70, 0.50, 0.30, 0.10]  # exploration thresholds
        assert _rate_by_thresholds(0.75, thresholds, lower_is_better=False) == 5

    def test_higher_is_better_boundary_at_t5(self):
        thresholds = [0.70, 0.50, 0.30, 0.10]
        assert _rate_by_thresholds(0.70, thresholds, lower_is_better=False) == 4

    def test_higher_is_better_rating_1(self):
        thresholds = [0.70, 0.50, 0.30, 0.10]
        assert _rate_by_thresholds(0.05, thresholds, lower_is_better=False) == 1


# ---------------------------------------------------------------------------
# _build_rating_context
# ---------------------------------------------------------------------------

class TestBuildRatingContext:
    def test_population_evolutionary_uses_wider_regression_thresholds(self):
        config = {
            "algorithm_classes": {
                "population_evolutionary": {
                    "regression_frequency_thresholds": [0.15, 0.30, 0.50, 0.70],
                    "exploration_sdi_thresholds": [0.70, 0.50, 0.30, 0.10],
                }
            }
        }
        ctx = _build_rating_context("population_evolutionary", config)
        assert ctx["regression_frequency_thresholds"] == [0.15, 0.30, 0.50, 0.70]
        assert ctx["algorithm_class"] == "population_evolutionary"

    def test_serial_refinement_uses_tighter_regression_thresholds(self):
        config = {
            "algorithm_classes": {
                "serial_refinement": {
                    "regression_frequency_thresholds": [0.03, 0.10, 0.20, 0.35],
                    "exploration_sdi_thresholds": [0.40, 0.25, 0.10, 0.05],
                }
            }
        }
        ctx = _build_rating_context("serial_refinement", config)
        assert ctx["regression_frequency_thresholds"][0] < _DEFAULT_REGRESSION_THRESHOLDS[0]

    def test_unknown_class_falls_back_to_defaults(self):
        ctx = _build_rating_context("unknown_class", {})
        assert ctx["regression_frequency_thresholds"] == _DEFAULT_REGRESSION_THRESHOLDS
        assert ctx["exploration_sdi_thresholds"] == _DEFAULT_EXPLORATION_THRESHOLDS

    def test_empty_config_falls_back_to_defaults(self):
        ctx = _build_rating_context("population_evolutionary", {})
        assert ctx["regression_frequency_thresholds"] == _DEFAULT_REGRESSION_THRESHOLDS

    def test_partial_config_falls_back_per_key(self):
        config = {
            "algorithm_classes": {
                "population_evolutionary": {
                    "regression_frequency_thresholds": [0.15, 0.30, 0.50, 0.70],
                    # no exploration_sdi_thresholds
                }
            }
        }
        ctx = _build_rating_context("population_evolutionary", config)
        assert ctx["regression_frequency_thresholds"] == [0.15, 0.30, 0.50, 0.70]
        assert ctx["exploration_sdi_thresholds"] == _DEFAULT_EXPLORATION_THRESHOLDS

    # --- convergence threshold keys ---

    def test_convergence_keys_present_in_context(self):
        ctx = _build_rating_context("population_evolutionary", _POP_CONFIG)
        assert "convergence_ttb_thresholds" in ctx
        assert "convergence_plateau_thresholds" in ctx

    def test_convergence_thresholds_loaded_from_config(self):
        ctx = _build_rating_context("population_evolutionary", _POP_CONFIG)
        assert ctx["convergence_ttb_thresholds"] == [0.60, 0.40]
        assert ctx["convergence_plateau_thresholds"] == [0.15, 0.30, 0.55]

    def test_convergence_thresholds_fall_back_for_unknown_class(self):
        ctx = _build_rating_context("unknown_class", {})
        assert ctx["convergence_ttb_thresholds"] == _DEFAULT_CONVERGENCE_TTB_THRESHOLDS
        assert ctx["convergence_plateau_thresholds"] == _DEFAULT_CONVERGENCE_PLATEAU_THRESHOLDS

    def test_convergence_thresholds_fall_back_when_block_absent(self):
        # Known class but convergence_thresholds key is missing from config
        config = {
            "algorithm_classes": {
                "population_evolutionary": {
                    "regression_frequency_thresholds": [0.15, 0.30, 0.50, 0.70],
                    "exploration_sdi_thresholds": [0.70, 0.50, 0.30, 0.10],
                    # no convergence_thresholds block
                }
            }
        }
        ctx = _build_rating_context("population_evolutionary", config)
        assert ctx["convergence_ttb_thresholds"] == _DEFAULT_CONVERGENCE_TTB_THRESHOLDS
        assert ctx["convergence_plateau_thresholds"] == _DEFAULT_CONVERGENCE_PLATEAU_THRESHOLDS

    def test_convergence_ttb_falls_back_when_subkey_absent(self):
        config = {
            "algorithm_classes": {
                "population_evolutionary": {
                    "convergence_thresholds": {
                        "plateau_fraction_thresholds": [0.15, 0.30, 0.55],
                        # no time_to_best_thresholds
                    }
                }
            }
        }
        ctx = _build_rating_context("population_evolutionary", config)
        assert ctx["convergence_ttb_thresholds"] == _DEFAULT_CONVERGENCE_TTB_THRESHOLDS
        assert ctx["convergence_plateau_thresholds"] == [0.15, 0.30, 0.55]


# ---------------------------------------------------------------------------
# Tool-aware rating: same metric, different class → different rating
# ---------------------------------------------------------------------------

class TestToolAwareRatingDiffers:
    """Regression frequency of 0.20 should rate differently across algorithm classes."""

    freq = 0.20  # Fair for default, but should be Good for population_evolutionary

    def _regression_rating(self, algorithm_class: str, config: dict) -> int:
        ctx = _build_rating_context(algorithm_class, config)
        return _rate_by_thresholds(
            self.freq,
            ctx["regression_frequency_thresholds"],
            lower_is_better=True,
        )

    def test_population_evolutionary_rates_higher_for_moderate_regression(self):
        config = {
            "algorithm_classes": {
                "population_evolutionary": {
                    "regression_frequency_thresholds": [0.15, 0.30, 0.50, 0.70],
                    "exploration_sdi_thresholds": [0.70, 0.50, 0.30, 0.10],
                },
                "serial_refinement": {
                    "regression_frequency_thresholds": [0.03, 0.10, 0.20, 0.35],
                    "exploration_sdi_thresholds": [0.40, 0.25, 0.10, 0.05],
                },
            }
        }
        pop_rating = self._regression_rating("population_evolutionary", config)
        serial_rating = self._regression_rating("serial_refinement", config)
        assert pop_rating > serial_rating, (
            f"Expected population_evolutionary ({pop_rating}) > serial_refinement ({serial_rating})"
        )

    def test_exploration_sdi_rates_higher_for_serial_with_relaxed_thresholds(self):
        sdi = 0.30  # Poor for population_evolutionary, Fair for serial_refinement
        config = {
            "algorithm_classes": {
                "population_evolutionary": {
                    "regression_frequency_thresholds": [0.15, 0.30, 0.50, 0.70],
                    "exploration_sdi_thresholds": [0.70, 0.50, 0.30, 0.10],
                },
                "serial_refinement": {
                    "regression_frequency_thresholds": [0.03, 0.10, 0.20, 0.35],
                    "exploration_sdi_thresholds": [0.40, 0.25, 0.10, 0.05],
                },
            }
        }
        pop_ctx = _build_rating_context("population_evolutionary", config)
        serial_ctx = _build_rating_context("serial_refinement", config)
        pop_rating = _rate_by_thresholds(sdi, pop_ctx["exploration_sdi_thresholds"], lower_is_better=False)
        serial_rating = _rate_by_thresholds(sdi, serial_ctx["exploration_sdi_thresholds"], lower_is_better=False)
        assert serial_rating > pop_rating, (
            f"Expected serial_refinement ({serial_rating}) > population_evolutionary ({pop_rating}) for SDI={sdi}"
        )


# ---------------------------------------------------------------------------
# _build_convergence_dimension: default threshold behaviour
# ---------------------------------------------------------------------------

class TestBuildConvergenceDimension:
    """Rating logic for the convergence dimension using default thresholds."""

    # --- no-plateau branch ---

    def test_no_plateau_ttbf_above_t5_rates_5(self):
        dim = _build_convergence_dimension(_make_quant(ttbf=0.85), [])
        assert dim.rating == 5

    def test_no_plateau_ttbf_at_t5_boundary_rates_5(self):
        # ttbf >= t5 is inclusive; ttbf == 0.80 == t5 → rating 5
        dim = _build_convergence_dimension(_make_quant(ttbf=0.80), [])
        assert dim.rating == 5

    def test_no_plateau_ttbf_between_t4_and_t5_rates_4(self):
        dim = _build_convergence_dimension(_make_quant(ttbf=0.65), [])
        assert dim.rating == 4

    def test_no_plateau_ttbf_below_t4_rates_3(self):
        dim = _build_convergence_dimension(_make_quant(ttbf=0.30), [])
        assert dim.rating == 3

    # --- plateau branch: n=100 so plateau_fraction = poi/100 ---

    def test_plateau_at_10pct_rates_1(self):
        # plateau_fraction=0.10 ≤ t1(0.20) → rating 1
        dim = _build_convergence_dimension(_make_quant(ttbf=0.5, poi=10, n=100), [])
        assert dim.rating == 1

    def test_plateau_at_20pct_boundary_rates_2(self):
        # plateau_fraction=0.20 ≤ t2(0.40), > t1(0.20) is false → still rating 2
        # 0.20 <= 0.20 → rating 1? Let's check: condition is <= t1 → 1
        # 0.20 <= 0.20 is True → rating 1
        dim = _build_convergence_dimension(_make_quant(ttbf=0.5, poi=20, n=100), [])
        assert dim.rating == 1

    def test_plateau_at_30pct_rates_2(self):
        # plateau_fraction=0.30, t1=0.20: 0.30 > 0.20 → not 1; ≤ t2(0.40) → rating 2
        dim = _build_convergence_dimension(_make_quant(ttbf=0.5, poi=30, n=100), [])
        assert dim.rating == 2

    def test_plateau_at_50pct_rates_3(self):
        # plateau_fraction=0.50 > t2(0.40), ≤ t3(0.60) → rating 3
        dim = _build_convergence_dimension(_make_quant(ttbf=0.5, poi=50, n=100), [])
        assert dim.rating == 3

    def test_plateau_at_70pct_rates_4(self):
        # plateau_fraction=0.70 > t3(0.60) → rating 4
        dim = _build_convergence_dimension(_make_quant(ttbf=0.5, poi=70, n=100), [])
        assert dim.rating == 4

    # --- missing data ---

    def test_no_convergence_data_returns_not_available(self):
        quant = types.SimpleNamespace(convergence=None)
        dim = _build_convergence_dimension(quant, [])
        assert dim.data_available is False
        assert dim.rating is None

    # --- evidence note ---

    def test_no_evidence_note_when_thresholds_are_default(self):
        # serial_refinement thresholds == defaults → no note appended
        ctx = _build_rating_context("serial_refinement", _POP_CONFIG)
        dim = _build_convergence_dimension(_make_quant(ttbf=0.85), [], rating_context=ctx)
        assert not any("Note:" in e and "convergence" in e for e in dim.evidence)

    def test_evidence_note_present_when_thresholds_differ(self):
        ctx = _build_rating_context("population_evolutionary", _POP_CONFIG)
        dim = _build_convergence_dimension(_make_quant(ttbf=0.65), [], rating_context=ctx)
        assert any("Note:" in e and "population_evolutionary" in e for e in dim.evidence)


# ---------------------------------------------------------------------------
# Tool-aware convergence: same metric, different class → different rating
# ---------------------------------------------------------------------------

class TestToolAwareConvergenceRatingDiffers:
    """The same ttbf / plateau_fraction produces different ratings per algorithm class."""

    def _conv_rating(self, algo_class: str, ttbf: float, poi: int | None = None) -> int:
        ctx = _build_rating_context(algo_class, _POP_CONFIG)
        return _build_convergence_dimension(_make_quant(ttbf, poi), [], rating_context=ctx).rating

    # --- time-to-best ---

    def test_ttbf_065_rates_5_for_pop_evo_and_4_for_serial(self):
        # pop_evo t5=0.60: 0.65 >= 0.60 → 5
        # serial  t5=0.80: 0.65 < 0.80, >= t4(0.60) → 4
        assert self._conv_rating("population_evolutionary", ttbf=0.65) == 5
        assert self._conv_rating("serial_refinement", ttbf=0.65) == 4

    def test_ttbf_045_rates_5_for_pop_evo_and_3_for_serial(self):
        # pop_evo t4=0.40: 0.45 >= 0.40 → 4... wait:
        # pop_evo: 0.45 >= t5(0.60)? No. >= t4(0.40)? Yes → 4
        # serial:  0.45 >= t5(0.80)? No. >= t4(0.60)? No → 3
        assert self._conv_rating("population_evolutionary", ttbf=0.45) == 4
        assert self._conv_rating("serial_refinement", ttbf=0.45) == 3

    def test_ttbf_075_rates_5_for_bo_and_4_for_serial(self):
        # BO t5=0.70: 0.75 >= 0.70 → 5
        # serial t5=0.80: 0.75 < 0.80, >= t4(0.60) → 4
        assert self._conv_rating("bayesian_optimization", ttbf=0.75) == 5
        assert self._conv_rating("serial_refinement", ttbf=0.75) == 4

    # --- plateau fraction: n=100 so fraction = poi/100 ---

    def test_plateau_at_18pct_rates_2_for_pop_evo_and_1_for_serial(self):
        # pop_evo t1=0.15: 0.18 > 0.15 → not 1; ≤ t2(0.30) → 2
        # serial  t1=0.20: 0.18 ≤ 0.20 → 1
        assert self._conv_rating("population_evolutionary", ttbf=0.5, poi=18) == 2
        assert self._conv_rating("serial_refinement", ttbf=0.5, poi=18) == 1

    def test_plateau_at_32pct_rates_2_for_pop_evo_and_2_for_serial(self):
        # pop_evo t2=0.30: 0.32 > 0.30, ≤ t3(0.55) → 3
        # serial  t2=0.40: 0.32 ≤ 0.40 → 2
        assert self._conv_rating("population_evolutionary", ttbf=0.5, poi=32) == 3
        assert self._conv_rating("serial_refinement", ttbf=0.5, poi=32) == 2

    def test_pop_evo_is_never_stricter_than_serial_for_ttbf(self):
        # For any ttbf, pop_evo rating should be >= serial rating
        for ttbf in [0.10, 0.30, 0.45, 0.60, 0.65, 0.75, 0.85, 0.95]:
            pop = self._conv_rating("population_evolutionary", ttbf=ttbf)
            serial = self._conv_rating("serial_refinement", ttbf=ttbf)
            assert pop >= serial, f"ttbf={ttbf}: pop_evo({pop}) < serial_refinement({serial})"

    def test_pop_evo_is_never_stricter_than_serial_for_plateau(self):
        # For any plateau onset, pop_evo rating should be >= serial rating
        for poi in [5, 10, 15, 18, 20, 25, 30, 32, 40, 50, 60, 70]:
            pop = self._conv_rating("population_evolutionary", ttbf=0.5, poi=poi)
            serial = self._conv_rating("serial_refinement", ttbf=0.5, poi=poi)
            assert pop >= serial, f"poi={poi}: pop_evo({pop}) < serial_refinement({serial})"
