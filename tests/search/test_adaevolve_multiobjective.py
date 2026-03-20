"""Tests for AdaEvolve's explicit multiobjective mode."""

import math

import pytest

from skydiscover.config import AdaEvolveDatabaseConfig, Config
from skydiscover.context_builder.adaevolve import AdaEvolveContextBuilder
from skydiscover.search.adaevolve.archive.unified_archive import ArchiveConfig, UnifiedArchive
from skydiscover.search.adaevolve.database import AdaEvolveDatabase
from skydiscover.search.adaevolve.paradigm.generator import ParadigmGenerator
from skydiscover.search.base_database import Program
from skydiscover.utils.metrics import normalize_metric_value


def _make_program(program_id: str, **metrics) -> Program:
    return Program(
        id=program_id,
        solution=f"def solve():\n    return '{program_id}'\n",
        metrics=metrics,
    )


def _pareto_db(
    pareto_objectives=None,
    higher_is_better=None,
    fitness_key="accuracy",
    num_islands=2,
    **extra,
):
    """Shorthand for creating a Pareto-enabled database."""
    config = AdaEvolveDatabaseConfig(
        population_size=10,
        num_islands=num_islands,
        use_dynamic_islands=False,
        use_paradigm_breakthrough=False,
        pareto_objectives=pareto_objectives or ["accuracy", "latency"],
        higher_is_better=higher_is_better or {"accuracy": True, "latency": False},
        fitness_key=fitness_key,
        pareto_objectives_weight=0.4,
        **extra,
    )
    return AdaEvolveDatabase("test", config)


def _scalar_db(num_islands=1, **extra):
    """Shorthand for creating a scalar-mode database."""
    config = AdaEvolveDatabaseConfig(
        population_size=10,
        num_islands=num_islands,
        use_dynamic_islands=False,
        use_paradigm_breakthrough=False,
        **extra,
    )
    return AdaEvolveDatabase("test", config)


# =========================================================================
# 1. Core Pareto front logic
# =========================================================================


class TestAdaEvolveMultiobjectiveDatabase:
    def test_global_pareto_front_and_representative_best(self):
        db = _pareto_db()

        high_accuracy = _make_program("p1", accuracy=0.95, latency=90.0)
        low_latency = _make_program("p2", accuracy=0.90, latency=10.0)
        dominated = _make_program("p3", accuracy=0.80, latency=120.0)

        db.add(high_accuracy, target_island=0)
        db.add(low_latency, target_island=1)
        db.add(dominated, target_island=0)

        pareto_ids = {program.id for program in db.get_pareto_front()}
        assert pareto_ids == {"p1", "p2"}
        assert {program.id for program in db.get_pareto_front(0)} == {"p1"}

        best = db.get_best_program()
        assert best is not None
        assert best.id == "p1"
        assert db.best_program_id == "p1"

        top_ids = [program.id for program in db.get_top_programs(2)]
        assert top_ids == ["p1", "p2"]

    def test_scalar_mode_remains_backward_compatible(self):
        db = _scalar_db()

        worse = _make_program("p1", combined_score=0.1, accuracy=0.9)
        better = _make_program("p2", combined_score=0.9, accuracy=0.1)

        db.add(worse, target_island=0)
        db.add(better, target_island=0)

        best = db.get_best_program()
        assert best is not None
        assert best.id == "p2"
        assert db.get_pareto_front(0)[0].id == "p2"
        assert [program.id for program in db.get_top_programs(2)] == ["p2", "p1"]

    def test_global_top_context_prefers_pareto_front_then_proxy_score(self):
        db = _pareto_db()

        pareto_a = _make_program("p1", accuracy=0.95, latency=90.0)
        pareto_b = _make_program("p2", accuracy=0.90, latency=10.0)
        dominated_but_high_proxy = _make_program("p3", accuracy=0.92, latency=120.0)
        dominated_low_proxy = _make_program("p4", accuracy=0.70, latency=150.0)

        db.add(pareto_a, target_island=0)
        db.add(pareto_b, target_island=1)
        db.add(dominated_but_high_proxy, target_island=0)
        db.add(dominated_low_proxy, target_island=1)

        selected = db._sample_global_top(exclude_id="missing", n=3)
        assert [program.id for program in selected] == ["p1", "p2", "p3"]


# =========================================================================
# 2. Pareto front caching
# =========================================================================


class TestGlobalParetoCaching:
    def test_cache_is_reused_across_calls(self):
        db = _pareto_db()
        db.add(_make_program("p1", accuracy=0.95, latency=90.0), target_island=0)

        front_a = db.get_global_pareto_front()
        front_b = db.get_global_pareto_front()
        # Same list contents (cached), not recomputed.
        assert [p.id for p in front_a] == [p.id for p in front_b]
        assert db._global_pareto_cache_valid is True

    def test_cache_invalidated_on_add(self):
        db = _pareto_db()
        db.add(_make_program("p1", accuracy=0.95, latency=90.0), target_island=0)

        # Warm cache
        front_before = db.get_global_pareto_front()
        assert len(front_before) == 1

        # Add a dominating program — cache should invalidate and rebuild
        db.add(_make_program("p2", accuracy=0.96, latency=5.0), target_island=1)
        front_after = db.get_global_pareto_front()

        # p2 dominates p1, so only p2 is on the front
        assert [p.id for p in front_after] == ["p2"]

    def test_stale_cache_used_for_previous_front_in_update_best(self):
        """The _update_best_program method should detect when a new program
        enters the Pareto front by comparing the stale cache (pre-add) with
        the freshly computed front (post-add)."""
        db = _pareto_db()

        db.add(_make_program("p1", accuracy=0.95, latency=90.0), target_island=0)
        db.add(_make_program("p2", accuracy=0.90, latency=10.0), target_island=1)

        # Warm the cache so the stale snapshot is available
        db.get_global_pareto_front()
        front_ids_before = {p.id for p in db.get_global_pareto_front()}
        assert front_ids_before == {"p1", "p2"}

        # Add a non-dominated program that should enter the front
        new_prog = _make_program("p3", accuracy=0.93, latency=50.0)
        # Manually test _update_best_program by simulating the add() flow:
        db.archives[0].add(new_prog)
        db.programs[new_prog.id] = new_prog
        db._invalidate_global_pareto_cache()
        result = db._update_best_program(new_prog)

        # Should detect a change (p3 entered the front)
        assert result is True
        new_front = db.get_global_pareto_front()
        assert "p3" in {p.id for p in new_front}

    def test_update_best_returns_false_for_dominated_addition(self):
        """Adding a dominated program should NOT trigger a best change."""
        db = _pareto_db()

        db.add(_make_program("p1", accuracy=0.95, latency=10.0), target_island=0)
        # Warm cache
        db.get_global_pareto_front()

        dominated = _make_program("p_dom", accuracy=0.80, latency=120.0)
        db.archives[0].add(dominated)
        db.programs[dominated.id] = dominated
        db._invalidate_global_pareto_cache()
        result = db._update_best_program(dominated)

        # p1 is still dominant, representative unchanged
        assert result is False
        assert db.best_program_id == "p1"


# =========================================================================
# 3. Dominance logic edge cases
# =========================================================================


class TestDominanceLogic:
    def test_dominates_basic(self):
        assert AdaEvolveDatabase._dominates([1.0, 1.0], [0.0, 0.0]) is True
        assert AdaEvolveDatabase._dominates([0.0, 0.0], [1.0, 1.0]) is False

    def test_equal_vectors_do_not_dominate(self):
        assert AdaEvolveDatabase._dominates([0.5, 0.5], [0.5, 0.5]) is False

    def test_partial_improvement_does_not_dominate(self):
        # Better on first, worse on second
        assert AdaEvolveDatabase._dominates([1.0, 0.0], [0.0, 1.0]) is False

    def test_single_objective_dominance(self):
        assert AdaEvolveDatabase._dominates([1.0], [0.5]) is True
        assert AdaEvolveDatabase._dominates([0.5], [1.0]) is False
        assert AdaEvolveDatabase._dominates([1.0], [1.0]) is False

    def test_mismatched_lengths_raises_error(self):
        with pytest.raises(ValueError, match="equal length"):
            AdaEvolveDatabase._dominates([1.0, 2.0], [1.0])

    def test_three_objective_dominance(self):
        assert AdaEvolveDatabase._dominates([1.0, 1.0, 1.0], [0.5, 0.5, 0.5]) is True
        # Equal on one axis: still dominates if strictly better on at least one
        assert AdaEvolveDatabase._dominates([1.0, 0.5, 1.0], [0.5, 0.5, 0.5]) is True
        # Worse on one axis: not dominant
        assert AdaEvolveDatabase._dominates([1.0, 0.4, 1.0], [0.5, 0.5, 0.5]) is False


# =========================================================================
# 4. Island-level Pareto front
# =========================================================================


class TestIslandPareto:
    def test_pareto_front_per_island_with_archives(self):
        db = _pareto_db(num_islands=2)

        db.add(_make_program("a1", accuracy=0.95, latency=50.0), target_island=0)
        db.add(_make_program("a2", accuracy=0.90, latency=10.0), target_island=0)
        db.add(_make_program("a3", accuracy=0.80, latency=100.0), target_island=0)  # dominated

        db.add(_make_program("b1", accuracy=0.85, latency=15.0), target_island=1)

        front_0 = db.get_pareto_front(0)
        front_0_ids = {p.id for p in front_0}
        assert "a1" in front_0_ids
        assert "a2" in front_0_ids
        assert "a3" not in front_0_ids

        front_1 = db.get_pareto_front(1)
        assert {p.id for p in front_1} == {"b1"}

    def test_global_front_spans_islands(self):
        db = _pareto_db(num_islands=2)

        db.add(_make_program("a1", accuracy=0.95, latency=50.0), target_island=0)
        db.add(_make_program("b1", accuracy=0.85, latency=15.0), target_island=1)

        global_front = db.get_pareto_front()  # island_idx=None → global
        global_ids = {p.id for p in global_front}
        assert global_ids == {"a1", "b1"}

    def test_out_of_bounds_island_returns_empty(self):
        db = _pareto_db(num_islands=2)
        assert db.get_pareto_front(99) == []


# =========================================================================
# 5. Proxy score and fitness key fallbacks
# =========================================================================


class TestProxyScoreFallbacks:
    def test_fitness_key_used_as_proxy(self):
        db = _pareto_db(fitness_key="accuracy")
        p = _make_program("p1", accuracy=0.95, latency=50.0)
        assert db.get_program_proxy_score(p) == 0.95

    def test_fitness_key_none_returns_neg_inf(self):
        db = _pareto_db(fitness_key="accuracy")
        assert db.get_program_proxy_score(None) == float("-inf")

    def test_fitness_key_missing_falls_back_to_combined_score(self):
        db = _pareto_db(fitness_key="nonexistent")
        p = _make_program("p1", combined_score=0.7, accuracy=0.95)
        # nonexistent key not in metrics → fall back to combined_score
        assert db.get_program_proxy_score(p) == 0.7

    def test_minimization_objective_negated_in_proxy(self):
        db = _pareto_db(fitness_key="latency")
        p = _make_program("p1", accuracy=0.95, latency=50.0)
        # latency is higher_is_better=False → negated
        assert db.get_program_proxy_score(p) == -50.0

    def test_no_fitness_key_averages_objectives(self):
        db = _pareto_db(fitness_key=None)
        p = _make_program("p1", accuracy=0.80, latency=20.0)
        # accuracy(0.80) + (-latency(-20.0)) → average = (0.80 + (-20.0)) / 2
        expected = (0.80 + (-20.0)) / 2
        assert abs(db.get_program_proxy_score(p) - expected) < 1e-9

    def test_empty_metrics_returns_neg_inf(self):
        db = _pareto_db()
        p = _make_program("p1")  # no metrics
        assert db.get_program_proxy_score(p) == float("-inf")

    def test_scalar_mode_uses_combined_score(self):
        db = _scalar_db()
        p = _make_program("p1", combined_score=0.42, accuracy=0.99)
        assert db.get_program_proxy_score(p) == 0.42

    def test_get_top_programs_with_specific_metric(self):
        """get_top_programs(metric=...) should sort by that metric, not proxy."""
        db = _pareto_db(fitness_key="accuracy")

        # p1 has highest accuracy (proxy), p2 has lowest latency (best if minimised)
        db.add(_make_program("p1", accuracy=0.95, latency=100.0), target_island=0)
        db.add(_make_program("p2", accuracy=0.80, latency=10.0), target_island=0)
        db.add(_make_program("p3", accuracy=0.85, latency=50.0), target_island=1)

        # Sort by latency — p2 (10) is best because higher_is_better=False → negated
        top_by_latency = db.get_top_programs(n=3, metric="latency")
        assert top_by_latency[0].id == "p2"  # lowest latency = best

        # Sort by accuracy — p1 (0.95) is best
        top_by_accuracy = db.get_top_programs(n=3, metric="accuracy")
        assert top_by_accuracy[0].id == "p1"

    def test_representative_prefers_newer_on_tie(self):
        """When proxy score and other signals are equal, newer programs should win.

        Programs are placed on separate islands so their archive-level elite
        scores are symmetric and the iteration tie-breaker is decisive.
        """
        db = _pareto_db(fitness_key="accuracy")

        old = _make_program("aaa_old", accuracy=0.90, latency=10.0)
        new = _make_program("zzz_new", accuracy=0.90, latency=10.0)

        # Separate islands → each is the only program in its archive,
        # so crowding distance and elite score are symmetric.
        db.add(old, iteration=1, target_island=0)
        db.add(new, iteration=5, target_island=1)

        # Both on global front (identical metrics), newer iteration should win
        best = db.get_best_program()
        assert best is not None
        assert best.id == "zzz_new"


# =========================================================================
# 6. Shared normalize_metric_value utility
# =========================================================================


class TestNormalizeMetricValue:
    def test_maximize_keeps_value(self):
        assert normalize_metric_value("acc", 0.9, {"acc": True}) == 0.9

    def test_minimize_negates_value(self):
        assert normalize_metric_value("latency", 50.0, {"latency": False}) == -50.0

    def test_missing_key_defaults_to_maximize(self):
        assert normalize_metric_value("unknown", 1.0, {}) == 1.0

    def test_non_numeric_returns_none(self):
        assert normalize_metric_value("acc", "high", {"acc": True}) is None
        assert normalize_metric_value("acc", None, {}) is None

    def test_integer_values(self):
        assert normalize_metric_value("count", 5, {"count": True}) == 5.0
        assert normalize_metric_value("errors", 3, {"errors": False}) == -3.0

    def test_boolean_values_excluded(self):
        """bool is a subclass of int in Python; must not be treated as numeric."""
        assert normalize_metric_value("timeout", True, {}) is None
        assert normalize_metric_value("success", False, {}) is None

    def test_nan_returns_none(self):
        """NaN breaks comparison semantics and must not enter objective vectors."""
        assert normalize_metric_value("acc", float("nan"), {"acc": True}) is None
        assert normalize_metric_value("latency", float("nan"), {"latency": False}) is None


# =========================================================================
# 7. Unified archive fitness fallbacks
# =========================================================================


class TestUnifiedArchiveFitnessFallbacks:
    def test_combined_score_is_preferred_over_accuracy_without_fitness_key(self):
        archive = UnifiedArchive(config=ArchiveConfig())
        p1 = _make_program("p1", combined_score=0.1, accuracy=0.9)
        p2 = _make_program("p2", combined_score=0.9, accuracy=0.1)

        archive.add(p1)
        archive.add(p2)

        assert [program.id for program in archive.get_top_programs(2)] == ["p2", "p1"]

    def test_fitness_key_respects_higher_is_better_for_minimization(self):
        archive = UnifiedArchive(
            config=ArchiveConfig(
                fitness_key="latency",
                higher_is_better={"latency": False},
            )
        )
        slow = _make_program("slow", latency=120.0, combined_score=0.9)
        fast = _make_program("fast", latency=10.0, combined_score=0.1)

        archive.add(slow)
        archive.add(fast)

        assert archive.get_best().id == "fast"
        assert [program.id for program in archive.get_top_programs(2)] == ["fast", "slow"]

    def test_archive_normalize_delegates_to_shared_utility(self):
        """Verify archive's _normalize_metric_value uses the shared function."""
        archive = UnifiedArchive(
            config=ArchiveConfig(higher_is_better={"loss": False})
        )
        assert archive._normalize_metric_value("loss", 5.0) == -5.0
        assert archive._normalize_metric_value("acc", 0.9) == 0.9
        assert archive._normalize_metric_value("acc", "string") is None


# =========================================================================
# 8. Prompt builder — Pareto vs scalar mode
# =========================================================================


class TestAdaEvolveMultiobjectivePrompts:
    def _pareto_builder(self):
        config = Config.from_dict(
            {
                "language": "python",
                "search": {
                    "type": "adaevolve",
                    "database": {
                        "pareto_objectives": ["accuracy", "latency"],
                        "higher_is_better": {"accuracy": True, "latency": False},
                        "fitness_key": "accuracy",
                        "use_dynamic_islands": False,
                        "use_paradigm_breakthrough": False,
                    },
                },
            }
        )
        return AdaEvolveContextBuilder(config)

    def _scalar_builder(self):
        config = Config.from_dict(
            {
                "language": "python",
                "search": {
                    "type": "adaevolve",
                    "database": {
                        "use_dynamic_islands": False,
                        "use_paradigm_breakthrough": False,
                    },
                },
            }
        )
        return AdaEvolveContextBuilder(config)

    def test_context_builder_uses_pareto_language(self):
        builder = self._pareto_builder()
        current = _make_program("parent", accuracy=0.91, latency=25.0)
        previous = _make_program("child", accuracy=0.89, latency=20.0)

        prompt = builder.build_prompt(
            current,
            {
                "program_metrics": current.metrics,
                "previous_programs": [previous],
            },
        )

        assert "Pareto trade-offs across: accuracy (maximize), latency (minimize)." in prompt["user"]
        assert "Pareto proxy" in prompt["user"]
        assert "COMBINED_SCORE" not in prompt["user"]

    def test_scalar_builder_uses_combined_score_language(self):
        builder = self._scalar_builder()
        current = _make_program("parent", combined_score=0.5)

        prompt = builder.build_prompt(
            current,
            {"program_metrics": current.metrics, "previous_programs": []},
        )

        assert "COMBINED_SCORE" in prompt["user"]
        assert "Pareto" not in prompt["user"]

    def test_paradigm_generator_mentions_objectives(self):
        generator = ParadigmGenerator(
            llm_pool=None,
            system_message="Improve the solver.",
            evaluator_code="def evaluate(path): return {}",
            objective_names=["accuracy", "latency"],
            higher_is_better={"accuracy": True, "latency": False},
            fitness_key="accuracy",
        )

        prompt = generator._build_prompt(
            program_solution="def solve(): pass",
            best_score=0.95,
            previously_tried=[],
        )

        assert "Optimize the Pareto trade-offs across: accuracy (maximize), latency (minimize)." in prompt
        assert '"what_to_optimize": "accuracy, latency"' in prompt
        assert "combined_score" not in prompt

    def test_paradigm_generator_scalar_mode(self):
        generator = ParadigmGenerator(
            llm_pool=None,
            system_message="Improve the solver.",
            evaluator_code="def evaluate(path): return {}",
        )

        prompt = generator._build_prompt(
            program_solution="def solve(): pass",
            best_score=0.5,
            previously_tried=[],
        )

        assert "Optimize the primary scalar score" in prompt
        assert "score 0.500000" in prompt or "score: 0.500000" in prompt


# =========================================================================
# 9. Builder progress score edge cases
# =========================================================================


class TestBuilderProgressScore:
    def _builder(self):
        config = Config.from_dict(
            {
                "language": "python",
                "search": {
                    "type": "adaevolve",
                    "database": {
                        "pareto_objectives": ["accuracy", "latency"],
                        "higher_is_better": {"accuracy": True, "latency": False},
                        "fitness_key": "accuracy",
                        "use_dynamic_islands": False,
                        "use_paradigm_breakthrough": False,
                    },
                },
            }
        )
        return AdaEvolveContextBuilder(config)

    def test_empty_metrics_returns_missing_sentinel(self):
        builder = self._builder()
        score = builder._get_progress_score({})
        assert score == builder._PROGRESS_SCORE_MISSING
        assert math.isinf(score) and score < 0

    def test_fitness_key_used_when_present(self):
        builder = self._builder()
        assert builder._get_progress_score({"accuracy": 0.9, "latency": 10.0}) == 0.9

    def test_combined_score_fallback(self):
        builder = self._builder()
        # fitness_key="accuracy" but it's not in metrics; combined_score is.
        assert builder._get_progress_score({"combined_score": 0.42}) == 0.42

    def test_improvement_areas_with_empty_previous_metrics(self):
        builder = self._builder()
        current = _make_program("c", accuracy=0.9, latency=10.0)
        previous_empty = _make_program("prev")  # no metrics

        result = builder._identify_improvement_areas(
            current.solution,
            current.metrics,
            [previous_empty],
        )

        # Should not crash, should show "first measurement" or skip delta
        assert "Pareto" in result
        assert "inf" not in result  # must not show -inf in prompt text

    def test_determine_outcome_with_missing_metrics(self):
        builder = self._builder()
        result = builder._determine_outcome({"accuracy": 0.9}, {})
        assert "Insufficient" in result

    def test_sibling_context_with_missing_metrics(self):
        builder = self._builder()
        parent = _make_program("parent", accuracy=0.9, latency=10.0)
        empty_child = _make_program("child")  # no metrics

        result = builder._format_sibling_context([empty_child], parent)
        assert result is not None
        assert "unavailable" in result
        assert "inf" not in result


# =========================================================================
# 10. Format previous attempts in Pareto mode
# =========================================================================


class TestFormatPreviousAttempts:
    def _builder(self):
        config = Config.from_dict(
            {
                "language": "python",
                "search": {
                    "type": "adaevolve",
                    "database": {
                        "pareto_objectives": ["accuracy", "latency"],
                        "higher_is_better": {"accuracy": True, "latency": False},
                        "fitness_key": "accuracy",
                        "use_dynamic_islands": False,
                        "use_paradigm_breakthrough": False,
                    },
                },
            }
        )
        return AdaEvolveContextBuilder(config)

    def test_format_previous_attempts_pareto_mode(self):
        builder = self._builder()
        programs = [
            _make_program("p1", accuracy=0.80, latency=50.0),
            _make_program("p2", accuracy=0.90, latency=30.0),
            _make_program("p3", accuracy=0.85, latency=20.0),
        ]

        result = builder._format_previous_attempts(programs, num_previous_attempts=3)
        assert "accuracy" in result
        assert "latency" in result
        # Should contain attempt formatting
        assert "Attempt" in result

    def test_empty_previous_programs(self):
        builder = self._builder()
        result = builder._format_previous_attempts([], num_previous_attempts=3)
        assert "No previous attempts" in result

    def test_previous_attempts_sorted_by_proxy_score(self):
        builder = self._builder()
        programs = [
            _make_program("low", accuracy=0.50, latency=50.0),
            _make_program("high", accuracy=0.99, latency=50.0),
            _make_program("mid", accuracy=0.75, latency=50.0),
        ]

        result = builder._format_previous_attempts(programs, num_previous_attempts=2)
        # Best 2 by proxy (accuracy) should be selected: 0.99 and 0.75, not 0.50
        assert "0.9900" in result  # high accuracy present
        assert "0.7500" in result  # mid accuracy present
        assert "0.5000" not in result  # low accuracy excluded


# =========================================================================
# 11. Comprehensive iteration stats in Pareto mode
# =========================================================================


class TestComprehensiveStats:
    def test_pareto_stats_included(self):
        db = _pareto_db()
        db.add(_make_program("p1", accuracy=0.95, latency=90.0), target_island=0)
        db.add(_make_program("p2", accuracy=0.90, latency=10.0), target_island=1)

        stats = db.get_comprehensive_iteration_stats(iteration=1)
        global_stats = stats["global"]

        assert global_stats["optimization_mode"] == "pareto"
        assert global_stats["pareto_objectives"] == ["accuracy", "latency"]
        assert global_stats["global_pareto_front_size"] == 2
        assert set(global_stats["global_pareto_front_ids"]) == {"p1", "p2"}
        assert global_stats["fitness_proxy_key"] == "accuracy"

    def test_scalar_stats(self):
        db = _scalar_db()
        db.add(_make_program("p1", combined_score=0.5), target_island=0)

        stats = db.get_comprehensive_iteration_stats(iteration=1)
        global_stats = stats["global"]

        assert global_stats["optimization_mode"] == "scalar"
        assert global_stats["pareto_objectives"] == []
        assert global_stats["global_pareto_front_size"] == 0


# =========================================================================
# 12. End-to-end multiobjective flow
# =========================================================================


class TestEndToEndMultiobjective:
    def test_minimization_objective_sorts_correctly(self):
        """latency with higher_is_better=False should prefer lower values."""
        db = _pareto_db(fitness_key="latency")

        fast = _make_program("fast", accuracy=0.80, latency=10.0)
        slow = _make_program("slow", accuracy=0.80, latency=100.0)

        db.add(fast, target_island=0)
        db.add(slow, target_island=0)

        # Both on front (different accuracy? no, same accuracy but different latency)
        # fast dominates slow: same accuracy, lower latency
        front = db.get_pareto_front()
        assert {p.id for p in front} == {"fast"}

        # Proxy score uses latency (negated) → fast = -10, slow = -100
        assert db.get_program_proxy_score(fast) == -10.0
        assert db.get_program_proxy_score(slow) == -100.0

    def test_three_way_pareto_front(self):
        """Three mutually non-dominating solutions."""
        db = _pareto_db(
            pareto_objectives=["a", "b", "c"],
            higher_is_better={"a": True, "b": True, "c": True},
            fitness_key="a",
        )

        # Each excels on exactly one objective
        db.add(_make_program("p1", a=1.0, b=0.0, c=0.0), target_island=0)
        db.add(_make_program("p2", a=0.0, b=1.0, c=0.0), target_island=0)
        db.add(_make_program("p3", a=0.0, b=0.0, c=1.0), target_island=1)

        front = db.get_global_pareto_front()
        assert {p.id for p in front} == {"p1", "p2", "p3"}

    def test_adding_dominating_program_evicts_from_front(self):
        db = _pareto_db()

        db.add(_make_program("p1", accuracy=0.80, latency=50.0), target_island=0)
        assert {p.id for p in db.get_global_pareto_front()} == {"p1"}

        # p2 dominates p1 on both objectives
        db.add(_make_program("p2", accuracy=0.90, latency=40.0), target_island=0)
        assert {p.id for p in db.get_global_pareto_front()} == {"p2"}

    def test_missing_objective_metric_treated_as_worst(self):
        """Programs with missing objective values get -inf for that dimension,
        preventing them from accidentally dominating fully-evaluated programs."""
        db = _pareto_db()

        complete = _make_program("complete", accuracy=0.5, latency=50.0)
        partial = _make_program("partial", accuracy=0.6)  # missing latency

        db.add(complete, target_island=0)
        db.add(partial, target_island=0)

        vec_complete = db._get_objective_vector(complete)
        vec_partial = db._get_objective_vector(partial)

        assert vec_complete == [0.5, -50.0]  # latency negated
        assert vec_partial[0] == 0.6
        assert vec_partial[1] == float("-inf")  # missing latency → worst possible

        # complete has latency=-50 > -inf, so partial does NOT dominate complete.
        # partial has accuracy=0.6 > 0.5, so complete does NOT dominate partial.
        # Both are on the front (trade-off: complete has real latency, partial has better accuracy).
        front = db.get_global_pareto_front()
        assert {p.id for p in front} == {"complete", "partial"}

    def test_sample_global_top_excludes_id(self):
        db = _pareto_db()
        db.add(_make_program("p1", accuracy=0.9, latency=10.0), target_island=0)
        db.add(_make_program("p2", accuracy=0.8, latency=20.0), target_island=1)

        selected = db._sample_global_top(exclude_id="p1", n=10)
        assert all(p.id != "p1" for p in selected)

    def test_scalar_global_pareto_front_returns_empty(self):
        """Scalar mode should return empty from get_global_pareto_front."""
        db = _scalar_db()
        db.add(_make_program("p1", combined_score=0.5), target_island=0)
        assert db.get_global_pareto_front() == []
        assert db.is_multiobjective_enabled() is False

    def test_nan_metric_does_not_dominate_real_solutions(self):
        """A program with NaN metrics must not enter the Pareto front over real solutions."""
        db = _pareto_db()

        good = _make_program("good", accuracy=0.9, latency=10.0)
        nan_prog = _make_program("nan_prog", accuracy=float("nan"), latency=5.0)

        db.add(good, target_island=0)
        db.add(nan_prog, target_island=1)

        # NaN accuracy → -inf in objective vector, so nan_prog cannot dominate good
        front = db.get_global_pareto_front()
        assert "good" in {p.id for p in front}
