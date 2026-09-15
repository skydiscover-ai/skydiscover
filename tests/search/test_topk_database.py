"""Regression tests for Top-K context selection.

A zero context count fetched only the parent and triggered the single-program
fallback, incorrectly returning the parent as context.
"""

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import Program
from skydiscover.optimize.search.topk.database import TopKDatabase


def _make_database(num_programs: int) -> TopKDatabase:
    db = TopKDatabase("topk", DatabaseConfig())
    for i in range(num_programs):
        db.add(Program(id=f"p{i}", solution=f"code {i}", metrics={"combined_score": float(i)}))
    return db


def test_no_context_requested():
    parent, context = _make_database(3).sample(num_context_programs=0)
    assert parent.id == "p2"
    assert context == []


def test_context_requested_with_one_program_falls_back_to_parent():
    parent, context = _make_database(1).sample(num_context_programs=1)
    assert parent.id == "p0"
    assert context == [parent]


def test_context_contains_next_best_programs():
    parent, context = _make_database(4).sample(num_context_programs=2)
    assert parent.id == "p3"
    assert [p.id for p in context] == ["p2", "p1"]
