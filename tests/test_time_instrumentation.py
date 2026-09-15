"""Tests for LLM generation time and evaluation time instrumentation."""

from skydiscover.optimize.search.utils.discovery_utils import SerializableResult


class TestSerializableResultTimeFields:
    """Verify time instrumentation fields on SerializableResult."""

    def test_default_values(self):
        result = SerializableResult()
        assert result.iteration_time == 0.0
        assert result.llm_generation_time == 0.0
        assert result.eval_time == 0.0

    def test_explicit_values(self):
        result = SerializableResult(
            iteration_time=10.5,
            llm_generation_time=8.2,
            eval_time=2.1,
        )
        assert result.iteration_time == 10.5
        assert result.llm_generation_time == 8.2
        assert result.eval_time == 2.1

    def test_error_before_eval(self):
        """LLM succeeds but evaluation never runs."""
        result = SerializableResult(
            llm_generation_time=5.0,
            error="Parse failed",
        )
        assert result.llm_generation_time == 5.0
        assert result.eval_time == 0.0
        assert result.iteration_time == 0.0

    def test_error_before_llm(self):
        """Iteration fails before LLM call (e.g. prompt building)."""
        result = SerializableResult(
            error="No parent found",
        )
        assert result.llm_generation_time == 0.0
        assert result.eval_time == 0.0
        assert result.iteration_time == 0.0

    def test_iteration_time_without_breakdown(self):
        """Old-style result with only iteration_time set."""
        result = SerializableResult(
            iteration_time=12.0,
        )
        assert result.iteration_time == 12.0
        assert result.llm_generation_time == 0.0
        assert result.eval_time == 0.0
