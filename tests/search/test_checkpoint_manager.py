"""Tests for CheckpointManager.save() — evolution_trace.json output."""

import json
import time

import pytest

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program
from skydiscover.search.utils.checkpoint_manager import CheckpointManager


def _make_program(id_: str, iteration: int, score: float, parent_id=None) -> Program:
    return Program(
        id=id_,
        solution=f"def solve(): return {score}",
        language="python",
        metrics={"combined_score": score},
        iteration_found=iteration,
        generation=iteration,
        parent_id=parent_id,
        timestamp=time.time(),
    )


@pytest.fixture
def manager(tmp_path):
    config = DatabaseConfig(db_path=str(tmp_path))
    return CheckpointManager(config)


class TestEvolutionTrace:
    def test_trace_file_is_created(self, manager, tmp_path):
        programs = {"a": _make_program("a", 0, 0.5)}
        manager.save(programs, None, "a", 0)
        assert (tmp_path / "evolution_trace.json").exists()

    def test_trace_top_level_fields(self, manager, tmp_path):
        programs = {
            "a": _make_program("a", 0, 0.5),
            "b": _make_program("b", 1, 0.8, parent_id="a"),
        }
        manager.save(programs, None, "b", 1)
        trace = json.loads((tmp_path / "evolution_trace.json").read_text())

        assert trace["last_iteration"] == 1
        assert trace["best_program_id"] == "b"
        assert trace["total_programs"] == 2
        assert len(trace["programs"]) == 2

    def test_trace_sorted_by_iteration(self, manager, tmp_path):
        programs = {
            "c": _make_program("c", 2, 0.9),
            "a": _make_program("a", 0, 0.5),
            "b": _make_program("b", 1, 0.7),
        }
        manager.save(programs, None, "c", 2)
        trace = json.loads((tmp_path / "evolution_trace.json").read_text())

        iterations = [p["iteration_found"] for p in trace["programs"]]
        assert iterations == sorted(iterations)

    def test_trace_entry_fields(self, manager, tmp_path):
        prog = _make_program("a", 0, 0.5)
        manager.save({"a": prog}, None, "a", 0)
        trace = json.loads((tmp_path / "evolution_trace.json").read_text())

        entry = trace["programs"][0]
        assert entry["id"] == "a"
        assert entry["score"] == pytest.approx(0.5)
        assert entry["metrics"] == {"combined_score": 0.5}
        assert entry["parent_id"] is None
        assert entry["solution"] == prog.solution
        assert "timestamp" in entry

    def test_trace_preserves_parent_id(self, manager, tmp_path):
        programs = {
            "a": _make_program("a", 0, 0.5),
            "b": _make_program("b", 1, 0.8, parent_id="a"),
        }
        manager.save(programs, None, "b", 1)
        trace = json.loads((tmp_path / "evolution_trace.json").read_text())

        by_id = {p["id"]: p for p in trace["programs"]}
        assert by_id["a"]["parent_id"] is None
        assert by_id["b"]["parent_id"] == "a"

    def test_trace_no_metrics_score_is_none(self, manager, tmp_path):
        prog = Program(id="x", solution="pass", metrics={})
        manager.save({"x": prog}, None, None, 0)
        trace = json.loads((tmp_path / "evolution_trace.json").read_text())
        assert trace["programs"][0]["score"] is None

    def test_trace_overwritten_on_subsequent_save(self, manager, tmp_path):
        manager.save({"a": _make_program("a", 0, 0.5)}, None, "a", 0)
        manager.save(
            {
                "a": _make_program("a", 0, 0.5),
                "b": _make_program("b", 1, 0.9),
            },
            None,
            "b",
            1,
        )
        trace = json.loads((tmp_path / "evolution_trace.json").read_text())
        assert trace["total_programs"] == 2
        assert trace["last_iteration"] == 1
