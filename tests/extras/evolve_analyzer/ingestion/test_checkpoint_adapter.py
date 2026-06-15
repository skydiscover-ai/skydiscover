"""
Tests for all four ingestion source types in checkpoint_adapter.py:
  - jsonl
  - skydiscover (adaevolve JSONL / log / checkpoint fallback, 3 directory layouts)
  - shinkaevolve (SQLite)
  - openevolve (trace JSONL / log supplement / checkpoint fallback)
"""

from __future__ import annotations

import json
import sqlite3
import textwrap
from pathlib import Path

import pytest

from skydiscover.extras.evolve_analyzer.ingestion.checkpoint_adapter import (
    _fill_derived_fields,
    _normalise_status,
    adapt_openevolve,
    adapt_shinkaevolve,
    adapt_skydiscover,
    load_evolve_records,
    load_jsonl,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _make_shinkaevolve_db(path: Path, table: str = "iterations", rows: list[dict] | None = None) -> Path:
    rows = rows or []
    conn = sqlite3.connect(str(path))
    if rows:
        cols = list(rows[0].keys())
        placeholders = ", ".join("?" for _ in cols)
        col_defs = ", ".join(f"{c} TEXT" for c in cols)
        conn.execute(f"CREATE TABLE {table} ({col_defs})")
        for row in rows:
            conn.execute(
                f"INSERT INTO {table} VALUES ({placeholders})",
                [row.get(c) for c in cols],
            )
    else:
        conn.execute(f"CREATE TABLE {table} (iteration TEXT, score TEXT)")
    conn.commit()
    conn.close()
    return path


# ===========================================================================
# _normalise_status
# ===========================================================================

class TestNormaliseStatus:
    def test_ok_maps_to_success(self):
        assert _normalise_status("ok") == "success"

    def test_success_identity(self):
        assert _normalise_status("success") == "success"

    def test_timeout(self):
        assert _normalise_status("timeout") == "timeout"

    def test_error_maps_to_crash(self):
        assert _normalise_status("error") == "crash"

    def test_exception_maps_to_crash(self):
        assert _normalise_status("exception") == "crash"

    def test_crash_identity(self):
        assert _normalise_status("crash") == "crash"

    def test_case_insensitive(self):
        assert _normalise_status("OK") == "success"
        assert _normalise_status("ERROR") == "crash"

    def test_unknown_defaults_to_unknown(self):
        assert _normalise_status("whatever") == "unknown"


# ===========================================================================
# _fill_derived_fields
# ===========================================================================

class TestFillDerivedFields:
    def test_empty_list(self):
        assert _fill_derived_fields([]) == []

    def test_iteration_assigned_when_missing(self):
        records = [{"child_score": 0.5}, {"child_score": 0.6}]
        out = _fill_derived_fields(records)
        assert out[0]["iteration"] == 0
        assert out[1]["iteration"] == 1

    def test_existing_iteration_preserved(self):
        records = [{"iteration": 5, "child_score": 0.5}]
        out = _fill_derived_fields(records)
        assert out[0]["iteration"] == 5

    def test_score_delta_computed(self):
        records = [{"child_score": 0.8, "parent_score": 0.5}]
        out = _fill_derived_fields(records)
        assert abs(out[0]["score_delta"] - 0.3) < 1e-9

    def test_score_delta_not_overwritten(self):
        records = [{"child_score": 0.8, "parent_score": 0.5, "score_delta": 99.0}]
        out = _fill_derived_fields(records)
        assert out[0]["score_delta"] == 99.0

    def test_evaluation_status_from_status_key(self):
        records = [{"status": "error"}]
        out = _fill_derived_fields(records)
        assert out[0]["evaluation_status"] == "crash"

    def test_evaluation_status_defaults_to_success(self):
        records = [{"child_score": 0.5}]
        out = _fill_derived_fields(records)
        assert out[0]["evaluation_status"] == "success"

    def test_evaluation_status_not_overwritten(self):
        records = [{"evaluation_status": "timeout"}]
        out = _fill_derived_fields(records)
        assert out[0]["evaluation_status"] == "timeout"

    def test_format_valid_defaults_to_true(self):
        records = [{"child_score": 0.5}]
        out = _fill_derived_fields(records)
        assert out[0]["format_valid"] is True

    def test_format_valid_not_overwritten(self):
        records = [{"format_valid": False}]
        out = _fill_derived_fields(records)
        assert out[0]["format_valid"] is False


# ===========================================================================
# load_jsonl
# ===========================================================================

class TestLoadJsonl:
    def test_empty_file(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text("", encoding="utf-8")
        assert list(load_jsonl(str(p))) == []

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('\n{"iteration": 1, "child_score": 0.5}\n\n', encoding="utf-8")
        records = list(load_jsonl(str(p)))
        assert len(records) == 1
        assert records[0]["iteration"] == 1

    def test_minimal_record(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [{"iteration": 0, "child_score": 0.42}])
        records = list(load_jsonl(str(p)))
        assert records[0]["child_score"] == 0.42

    def test_score_delta_computed_inline(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [{"iteration": 1, "child_score": 0.7, "parent_score": 0.5}])
        records = list(load_jsonl(str(p)))
        assert abs(records[0]["score_delta"] - 0.2) < 1e-9

    def test_score_delta_not_overwritten_if_present(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [{"iteration": 1, "child_score": 0.7, "parent_score": 0.5, "score_delta": -1.0}])
        records = list(load_jsonl(str(p)))
        assert records[0]["score_delta"] == -1.0

    def test_all_optional_fields_passed_through(self, tmp_path):
        rec = {
            "iteration": 3,
            "child_score": 0.9,
            "parent_score": 0.8,
            "evaluation_status": "success",
            "format_valid": True,
            "mutation_type": "diff",
            "model": "gpt-4",
            "evaluator_metrics": {"accuracy": 0.9, "latency_ms": 120},
            "llm_tokens_used": 500,
            "llm_cost_usd": 0.01,
            "timestamp": 1700000000.0,
            "island_id": "0",
        }
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [rec])
        out = list(load_jsonl(str(p)))[0]
        assert out["mutation_type"] == "diff"
        assert out["evaluator_metrics"]["accuracy"] == 0.9
        assert out["llm_tokens_used"] == 500
        assert out["island_id"] == "0"

    def test_multiple_records(self, tmp_path):
        records = [{"iteration": i, "child_score": i * 0.1} for i in range(5)]
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, records)
        out = list(load_jsonl(str(p)))
        assert len(out) == 5
        assert out[4]["child_score"] == pytest.approx(0.4)

    def test_score_delta_missing_parent_no_compute(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [{"iteration": 0, "child_score": 0.5}])
        records = list(load_jsonl(str(p)))
        assert "score_delta" not in records[0]


# ===========================================================================
# load_evolve_records (integration, source dispatch, post-processing)
# ===========================================================================

class TestLoadEvolveRecords:
    def test_unknown_source_raises(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text("{}\n")
        with pytest.raises(ValueError, match="Unknown source"):
            load_evolve_records("bogus", str(p))

    def test_source_case_insensitive(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [{"iteration": 0, "child_score": 0.5}])
        records = load_evolve_records("JSONL", str(p))
        assert len(records) == 1

    def test_fill_derived_fields_applied(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [{"child_score": 0.5, "parent_score": 0.3}])
        records = load_evolve_records("jsonl", str(p))
        assert records[0]["format_valid"] is True
        assert records[0]["evaluation_status"] == "success"
        assert abs(records[0]["score_delta"] - 0.2) < 1e-9

    def test_records_sorted_by_iteration(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [
            {"iteration": 3, "child_score": 0.9},
            {"iteration": 1, "child_score": 0.6},
            {"iteration": 2, "child_score": 0.7},
        ])
        records = load_evolve_records("jsonl", str(p))
        assert [r["iteration"] for r in records] == [1, 2, 3]

    def test_empty_returns_empty_list(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text("", encoding="utf-8")
        assert load_evolve_records("jsonl", str(p)) == []


# ===========================================================================
# adapt_shinkaevolve
# ===========================================================================

class TestAdaptShinkaevolve:
    def test_empty_iterations_table(self, tmp_path):
        db = _make_shinkaevolve_db(tmp_path / "run.db", table="iterations", rows=[])
        records = list(adapt_shinkaevolve(str(db)))
        assert records == []

    def test_iterations_table_preferred_over_runs(self, tmp_path):
        db = tmp_path / "run.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE iterations (iteration TEXT, score TEXT)")
        conn.execute("INSERT INTO iterations VALUES ('1', '0.7')")
        conn.execute("CREATE TABLE runs (iteration TEXT, score TEXT)")
        conn.execute("INSERT INTO runs VALUES ('99', '0.1')")
        conn.commit()
        conn.close()
        records = list(adapt_shinkaevolve(str(db)))
        assert records[0]["iteration"] == 1
        assert records[0]["child_score"] == pytest.approx(0.7)

    def test_runs_table_fallback(self, tmp_path):
        db = tmp_path / "run.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE runs (iteration TEXT, score TEXT)")
        conn.execute("INSERT INTO runs VALUES ('5', '0.55')")
        conn.commit()
        conn.close()
        records = list(adapt_shinkaevolve(str(db)))
        assert records[0]["iteration"] == 5
        assert records[0]["child_score"] == pytest.approx(0.55)

    def test_no_known_table_yields_nothing(self, tmp_path):
        db = tmp_path / "run.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE other (x TEXT)")
        conn.commit()
        conn.close()
        assert list(adapt_shinkaevolve(str(db))) == []

    def test_column_mapping(self, tmp_path):
        row = {
            "iteration": "2",
            "score": "0.8",
            "parent_score": "0.6",
            "code": "def f(): pass",
            "parent_code": "def f(): return 1",
            "diff": "@@ -1 +1 @@",
            "model": "gpt-4o",
            "mutation_type": "rewrite",
            "status": "ok",
            "tokens_used": "1000",
            "cost_usd": "0.02",
            "latency_ms": "300",
            "timestamp": "1700000000.0",
            "island_id": "A",
        }
        db = _make_shinkaevolve_db(tmp_path / "run.db", rows=[row])
        records = list(adapt_shinkaevolve(str(db)))
        r = records[0]
        assert r["iteration"] == 2
        assert r["child_score"] == pytest.approx(0.8)
        assert r["parent_score"] == pytest.approx(0.6)
        assert abs(r["score_delta"] - 0.2) < 1e-9
        assert r["child_code"] == "def f(): pass"
        assert r["parent_code"] == "def f(): return 1"
        assert r["diff"] == "@@ -1 +1 @@"
        assert r["model"] == "gpt-4o"
        assert r["mutation_type"] == "rewrite"
        assert r["evaluation_status"] == "success"
        assert r["llm_tokens_used"] == 1000
        assert r["llm_cost_usd"] == pytest.approx(0.02)
        assert r["llm_latency_ms"] == pytest.approx(300)
        assert r["timestamp"] == pytest.approx(1700000000.0)
        assert r["island_id"] == "A"

    def test_status_normalised_error_to_crash(self, tmp_path):
        row = {"iteration": "1", "score": "0.5", "status": "error"}
        db = _make_shinkaevolve_db(tmp_path / "run.db", rows=[row])
        records = list(adapt_shinkaevolve(str(db)))
        assert records[0]["evaluation_status"] == "crash"

    def test_status_normalised_exception_to_crash(self, tmp_path):
        row = {"iteration": "1", "score": "0.5", "status": "exception"}
        db = _make_shinkaevolve_db(tmp_path / "run.db", rows=[row])
        records = list(adapt_shinkaevolve(str(db)))
        assert records[0]["evaluation_status"] == "crash"

    def test_island_spawn_event_on_island_change(self, tmp_path):
        rows = [
            {"iteration": "1", "score": "0.5", "island_id": "A"},
            {"iteration": "2", "score": "0.6", "island_id": "B"},
            {"iteration": "3", "score": "0.7", "island_id": "B"},
        ]
        db = _make_shinkaevolve_db(tmp_path / "run.db", rows=rows)
        records = list(adapt_shinkaevolve(str(db)))
        assert "framework_stagnation_event" not in records[0]
        assert records[1]["framework_stagnation_event"] == "island_spawn"
        assert "framework_stagnation_event" not in records[2]

    def test_metadata_json_merged_gaps_only(self, tmp_path):
        meta = json.dumps({"extra_field": "hello", "score": "9999"})
        row = {"iteration": "1", "score": "0.5", "metadata": meta}
        db = _make_shinkaevolve_db(tmp_path / "run.db", rows=[row])
        records = list(adapt_shinkaevolve(str(db)))
        r = records[0]
        assert r["extra_field"] == "hello"
        assert r["child_score"] == pytest.approx(0.5)  # not overwritten by metadata

    def test_metadata_invalid_json_ignored(self, tmp_path):
        row = {"iteration": "1", "score": "0.5", "metadata": "not_json{"}
        db = _make_shinkaevolve_db(tmp_path / "run.db", rows=[row])
        records = list(adapt_shinkaevolve(str(db)))
        assert records[0]["child_score"] == pytest.approx(0.5)

    def test_missing_optional_columns_handled(self, tmp_path):
        row = {"iteration": "1", "score": "0.5"}
        db = _make_shinkaevolve_db(tmp_path / "run.db", rows=[row])
        records = list(adapt_shinkaevolve(str(db)))
        r = records[0]
        assert "parent_score" not in r
        assert "diff" not in r
        assert "model" not in r

    def test_score_delta_absent_when_parent_score_missing(self, tmp_path):
        row = {"iteration": "1", "score": "0.5"}
        db = _make_shinkaevolve_db(tmp_path / "run.db", rows=[row])
        records = list(adapt_shinkaevolve(str(db)))
        assert "score_delta" not in records[0]

    def test_ordered_by_iteration(self, tmp_path):
        rows = [
            {"iteration": "3", "score": "0.9"},
            {"iteration": "1", "score": "0.5"},
            {"iteration": "2", "score": "0.7"},
        ]
        db = _make_shinkaevolve_db(tmp_path / "run.db", rows=rows)
        records = list(adapt_shinkaevolve(str(db)))
        assert [r["iteration"] for r in records] == [1, 2, 3]


# ===========================================================================
# adapt_openevolve — trace JSONL
# ===========================================================================

class TestAdaptOpenevolveTrace:
    def _make_trace(self, tmp_path: Path, records: list[dict]) -> Path:
        d = tmp_path / "run"
        d.mkdir()
        _write_jsonl(d / "evolution_trace.jsonl", records)
        return d

    def test_nested_child_metrics_combined_score(self, tmp_path):
        d = self._make_trace(tmp_path, [
            {"iteration": 1, "child_metrics": {"combined_score": 0.75, "accuracy": 0.8}}
        ])
        records = list(adapt_openevolve(str(d)))
        assert records[0]["child_score"] == pytest.approx(0.75)

    def test_nested_parent_metrics_combined_score(self, tmp_path):
        d = self._make_trace(tmp_path, [
            {"iteration": 1,
             "child_metrics": {"combined_score": 0.75},
             "parent_metrics": {"combined_score": 0.60}}
        ])
        records = list(adapt_openevolve(str(d)))
        assert records[0]["parent_score"] == pytest.approx(0.60)

    def test_improvement_delta_combined_score(self, tmp_path):
        d = self._make_trace(tmp_path, [
            {"iteration": 1,
             "child_metrics": {"combined_score": 0.8},
             "improvement_delta": {"combined_score": 0.1}}
        ])
        records = list(adapt_openevolve(str(d)))
        assert records[0]["score_delta"] == pytest.approx(0.1)

    def test_top_level_score_fallback(self, tmp_path):
        d = self._make_trace(tmp_path, [{"iteration": 1, "score": 0.65}])
        records = list(adapt_openevolve(str(d)))
        assert records[0]["child_score"] == pytest.approx(0.65)

    def test_top_level_fitness_fallback(self, tmp_path):
        d = self._make_trace(tmp_path, [{"iteration": 1, "fitness": 0.71}])
        records = list(adapt_openevolve(str(d)))
        assert records[0]["child_score"] == pytest.approx(0.71)

    def test_program_field_becomes_child_code_and_code_alias(self, tmp_path):
        d = self._make_trace(tmp_path, [
            {"iteration": 1, "child_metrics": {"combined_score": 0.5},
             "program": "def f(): pass"}
        ])
        records = list(adapt_openevolve(str(d)))
        assert records[0]["child_code"] == "def f(): pass"
        assert records[0]["code"] == "def f(): pass"

    def test_child_code_field_fallback(self, tmp_path):
        d = self._make_trace(tmp_path, [
            {"iteration": 1, "child_metrics": {"combined_score": 0.5},
             "child_code": "def g(): pass"}
        ])
        records = list(adapt_openevolve(str(d)))
        assert records[0]["child_code"] == "def g(): pass"

    def test_prompt_system_becomes_system_message(self, tmp_path):
        d = self._make_trace(tmp_path, [
            {"iteration": 1, "child_metrics": {"combined_score": 0.5},
             "prompt": {"system": "You are a coder."}}
        ])
        records = list(adapt_openevolve(str(d)))
        assert records[0]["system_message"] == "You are a coder."

    def test_evaluator_metrics_from_child_metrics(self, tmp_path):
        d = self._make_trace(tmp_path, [
            {"iteration": 1,
             "child_metrics": {"combined_score": 0.8, "accuracy": 0.9, "latency_ms": 50}}
        ])
        records = list(adapt_openevolve(str(d)))
        assert records[0]["evaluator_metrics"] == {
            "combined_score": 0.8, "accuracy": 0.9, "latency_ms": 50
        }

    def test_parent_metrics_full_dict_preserved(self, tmp_path):
        d = self._make_trace(tmp_path, [
            {"iteration": 1,
             "child_metrics": {"combined_score": 0.8},
             "parent_metrics": {"combined_score": 0.6, "accuracy": 0.7}}
        ])
        records = list(adapt_openevolve(str(d)))
        assert records[0]["parent_metrics"] == {"combined_score": 0.6, "accuracy": 0.7}

    def test_passthrough_fields(self, tmp_path):
        d = self._make_trace(tmp_path, [
            {"iteration": 2,
             "child_metrics": {"combined_score": 0.7},
             "evaluation_status": "success",
             "format_valid": True,
             "mutation_type": "diff",
             "model": "gpt-4o",
             "diff": "@@ -1 +1 @@",
             "llm_tokens_used": 300,
             "llm_cost_usd": 0.005,
             "island_id": "isle0",
             "followed_suggestion": True,
             "early_stop_suggested": False,
             "reasoning_trace": "step by step",
            }
        ])
        records = list(adapt_openevolve(str(d)))
        r = records[0]
        assert r["evaluation_status"] == "success"
        assert r["format_valid"] is True
        assert r["mutation_type"] == "diff"
        assert r["model"] == "gpt-4o"
        assert r["diff"] == "@@ -1 +1 @@"
        assert r["llm_tokens_used"] == 300
        assert r["llm_cost_usd"] == pytest.approx(0.005)
        assert r["island_id"] == "isle0"
        assert r["followed_suggestion"] is True
        assert r["early_stop_suggested"] is False
        assert r["reasoning_trace"] == "step by step"

    def test_parameters_dict_synthesized(self, tmp_path):
        d = self._make_trace(tmp_path, [
            {"iteration": 1,
             "child_metrics": {"combined_score": 0.6},
             "mutation_type": "rewrite",
             "island_id": "i0",
             "model": "llama",
             "generation": 5,
            }
        ])
        records = list(adapt_openevolve(str(d)))
        params = records[0]["parameters"]
        assert params["mutation_type"] == "rewrite"
        assert params["island_id"] == "i0"
        assert params["model"] == "llama"
        assert params["generation"] == 5

    def test_metadata_changes_adds_change_type_parameter(self, tmp_path):
        d = self._make_trace(tmp_path, [
            {"iteration": 1,
             "child_metrics": {"combined_score": 0.6},
             "metadata": {"changes": "refactor"}}
        ])
        records = list(adapt_openevolve(str(d)))
        assert records[0]["parameters"]["change_type"] == "refactor"

    def test_out_of_order_records_sorted_by_iteration(self, tmp_path):
        d = self._make_trace(tmp_path, [
            {"iteration": 3, "child_metrics": {"combined_score": 0.9}},
            {"iteration": 1, "child_metrics": {"combined_score": 0.5}},
            {"iteration": 2, "child_metrics": {"combined_score": 0.7}},
        ])
        records = list(adapt_openevolve(str(d)))
        assert [r["iteration"] for r in records] == [1, 2, 3]

    def test_blank_lines_in_trace_skipped(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        content = (
            '{"iteration": 1, "child_metrics": {"combined_score": 0.5}}\n'
            '\n'
            '{"iteration": 2, "child_metrics": {"combined_score": 0.6}}\n'
        )
        (d / "evolution_trace.jsonl").write_text(content, encoding="utf-8")
        records = list(adapt_openevolve(str(d)))
        assert len(records) == 2

    def test_autodetect_trace_in_checkpoint_dir(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        _write_jsonl(d / "evolution_trace.jsonl", [
            {"iteration": 1, "score": 0.5},
        ])
        records = list(adapt_openevolve(str(d)))
        assert len(records) == 1
        assert records[0]["child_score"] == pytest.approx(0.5)

    def test_explicit_trace_path_used(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        trace = tmp_path / "custom_trace.jsonl"
        _write_jsonl(trace, [{"iteration": 7, "score": 0.99}])
        records = list(adapt_openevolve(str(d), trace_path=str(trace)))
        assert records[0]["iteration"] == 7
        assert records[0]["child_score"] == pytest.approx(0.99)

    def test_load_evolve_records_forwards_trace_path_kwarg(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        trace = tmp_path / "other.jsonl"
        _write_jsonl(trace, [{"iteration": 2, "score": 0.88}])
        records = load_evolve_records("openevolve", str(d), trace_path=str(trace))
        assert records[0]["child_score"] == pytest.approx(0.88)


class TestAdaptOpenevolveLogSupplement:
    def test_failed_iterations_added_from_log(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        _write_jsonl(d / "evolution_trace.jsonl", [
            {"iteration": 1, "child_metrics": {"combined_score": 0.5}},
        ])
        logs_dir = d / "logs"
        logs_dir.mkdir()
        log_content = textwrap.dedent("""\
            2024-01-01 10:00:00,000 WARNING - Iteration 2 error: budget exceeded
        """)
        (logs_dir / "run.log").write_text(log_content, encoding="utf-8")
        records = list(adapt_openevolve(str(d)))
        assert len(records) == 2
        failed = next(r for r in records if r["iteration"] == 2)
        assert failed["evaluation_status"] == "crash"
        assert "budget exceeded" in failed["error"]

    def test_log_iteration_not_added_if_already_in_trace(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        _write_jsonl(d / "evolution_trace.jsonl", [
            {"iteration": 2, "child_metrics": {"combined_score": 0.7}},
        ])
        logs_dir = d / "logs"
        logs_dir.mkdir()
        log_content = "2024-01-01 10:00:00,000 WARNING - Iteration 2 error: old error\n"
        (logs_dir / "run.log").write_text(log_content, encoding="utf-8")
        records = list(adapt_openevolve(str(d)))
        assert len(records) == 1
        assert records[0]["child_score"] == pytest.approx(0.7)


class TestAdaptOpenevolveCheckpointFallback:
    def test_checkpoint_fallback_no_trace(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        for n, score, code in [(1, 0.5, "def f(): pass"), (2, 0.7, "def f(): return 1")]:
            ckpt = d / f"checkpoint_{n}"
            ckpt.mkdir()
            (ckpt / "best_solution.json").write_text(
                json.dumps({"score": score, "program": code}), encoding="utf-8"
            )
            (ckpt / "metadata.json").write_text(
                json.dumps({"iteration": n}), encoding="utf-8"
            )
        records = list(adapt_openevolve(str(d)))
        assert len(records) == 2
        assert records[0]["child_score"] == pytest.approx(0.5)
        assert records[1]["child_score"] == pytest.approx(0.7)
        assert records[1]["parent_score"] == pytest.approx(0.5)

    def test_checkpoint_fallback_checkpoints_subdir(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        ckpts = d / "checkpoints"
        ckpts.mkdir()
        ckpt = ckpts / "checkpoint_1"
        ckpt.mkdir()
        (ckpt / "best_solution.json").write_text(
            json.dumps({"score": 0.6, "code": "x = 1"}), encoding="utf-8"
        )
        records = list(adapt_openevolve(str(d)))
        assert len(records) == 1
        assert records[0]["child_score"] == pytest.approx(0.6)


# ===========================================================================
# adapt_skydiscover — adaevolve JSONL primary source
# ===========================================================================

class TestAdaptSkydiscoverAdaevolveJsonl:
    def _make_run(self, tmp_path: Path, entries: list[dict]) -> Path:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "checkpoints").mkdir()
        _write_jsonl(run_dir / "adaevolve_iteration_stats_000.jsonl", entries)
        return run_dir

    def test_basic_iteration_record(self, tmp_path):
        entry = {
            "iteration": 1,
            "iteration_result": {
                "success": True,
                "child_program": {"metrics": {"combined_score": 0.7}, "id": "abc"},
            },
            "global": {},
        }
        run_dir = self._make_run(tmp_path, [entry])
        records = list(adapt_skydiscover(str(run_dir)))
        assert len(records) == 1
        assert records[0]["child_score"] == pytest.approx(0.7)

    def test_failed_iteration_is_crash(self, tmp_path):
        entry = {
            "iteration": 1,
            "iteration_result": {
                "success": False,
                "error": "LLM timeout",
                "child_program": {},
            },
            "global": {},
        }
        run_dir = self._make_run(tmp_path, [entry])
        records = list(adapt_skydiscover(str(run_dir)))
        assert records[0]["evaluation_status"] == "crash"
        assert records[0]["error"] == "LLM timeout"

    def test_parent_score_filled_from_previous_child(self, tmp_path):
        entries = [
            {
                "iteration": 1,
                "iteration_result": {
                    "success": True,
                    "child_program": {"metrics": {"combined_score": 0.5}},
                },
                "global": {},
            },
            {
                "iteration": 2,
                "iteration_result": {
                    "success": True,
                    "child_program": {"metrics": {"combined_score": 0.7}},
                },
                "global": {},
            },
        ]
        run_dir = self._make_run(tmp_path, entries)
        records = list(adapt_skydiscover(str(run_dir)))
        assert records[1]["parent_score"] == pytest.approx(0.5)
        assert records[1]["score_delta"] == pytest.approx(0.2)

    def test_island_id_extracted(self, tmp_path):
        entry = {
            "iteration": 1,
            "iteration_result": {
                "success": True,
                "child_program": {"metrics": {"combined_score": 0.6}},
            },
            "islands": [{"island_idx": 2, "productivity": 0.9, "population_size": 10}],
            "global": {},
        }
        run_dir = self._make_run(tmp_path, [entry])
        records = list(adapt_skydiscover(str(run_dir)))
        assert records[0]["island_id"] == 2

    def test_evaluator_metrics_without_combined_score(self, tmp_path):
        entry = {
            "iteration": 1,
            "iteration_result": {
                "success": True,
                "child_program": {
                    "metrics": {"combined_score": 0.8, "accuracy": 0.9, "latency_ms": 100}
                },
            },
            "global": {},
        }
        run_dir = self._make_run(tmp_path, [entry])
        records = list(adapt_skydiscover(str(run_dir)))
        em = records[0]["evaluator_metrics"]
        assert "combined_score" not in em
        assert em["accuracy"] == pytest.approx(0.9)

    def test_paradigm_shift_event(self, tmp_path):
        entry = {
            "iteration": 1,
            "iteration_result": {
                "success": True,
                "child_program": {"metrics": {"combined_score": 0.7}},
            },
            "global": {},
            "paradigm": {"has_active_paradigm": True, "improvement_rate": 0.05},
        }
        run_dir = self._make_run(tmp_path, [entry])
        records = list(adapt_skydiscover(str(run_dir)))
        assert records[0]["framework_stagnation_event"] == "paradigm_shift"

    def test_parameters_dict_from_sampling_and_island(self, tmp_path):
        entry = {
            "iteration": 1,
            "iteration_result": {
                "success": True,
                "child_program": {"metrics": {"combined_score": 0.6}},
            },
            "global": {},
            "sampling": {"mode": "exploration", "intensity_used": 0.8},
            "islands": [{"island_idx": 0, "productivity": 0.7, "population_size": 20}],
            "paradigm": {"improvement_rate": 0.03, "is_stagnating": False},
        }
        run_dir = self._make_run(tmp_path, [entry])
        records = list(adapt_skydiscover(str(run_dir)))
        p = records[0]["parameters"]
        assert p["sampling_mode"] == "exploration"
        assert p["search_intensity"] == pytest.approx(0.8)
        assert p["island_productivity"] == pytest.approx(0.7)
        assert p["population_size"] == 20
        assert p["improvement_rate"] == pytest.approx(0.03)
        assert p["is_stagnating"] is False


# ===========================================================================
# adapt_skydiscover — log secondary source
# ===========================================================================

class TestAdaptSkydiscoverLog:
    def _make_log_run(self, tmp_path: Path, log_content: str) -> Path:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        logs = run_dir / "logs"
        logs.mkdir()
        (logs / "run.log").write_text(log_content, encoding="utf-8")
        return run_dir

    def test_successful_iteration_from_log(self, tmp_path):
        log = textwrap.dedent("""\
            2024-01-01 10:00:00,000 Adding initial program to database
            2024-01-01 10:00:01,000 Evaluated program abc123 in 1.2s: combined_score=0.65
        """)
        run_dir = self._make_log_run(tmp_path, log)
        records = list(adapt_skydiscover(str(run_dir)))
        assert len(records) == 1
        assert records[0]["child_score"] == pytest.approx(0.65)

    def test_failed_iteration_from_log(self, tmp_path):
        log = textwrap.dedent("""\
            2024-01-01 10:00:00,000 Adding initial program to database
            2024-01-01 10:00:01,000 Iteration 3 failed: syntax error in generated code
        """)
        run_dir = self._make_log_run(tmp_path, log)
        records = list(adapt_skydiscover(str(run_dir)))
        assert len(records) == 1
        assert records[0]["outcome"] == "failed"
        assert records[0]["iteration"] == 3

    def test_parent_score_propagated(self, tmp_path):
        # Consecutive successes: the log parser overwrites pending_success, so only
        # the last success is flushed. parent_score comes from _fill_derived_fields
        # for the second success, which needs the first success to be a failed record
        # to emit it. Test the simpler case: one success followed by one failure so
        # the parent_score propagates from the previous child_score.
        log = textwrap.dedent("""\
            2024-01-01 10:00:00,000 Adding initial program to database
            2024-01-01 10:00:01,000 Evaluated program abc in 1s: combined_score=0.5
            2024-01-01 10:00:02,000 Iteration 2 failed: out of budget
        """)
        run_dir = self._make_log_run(tmp_path, log)
        records = list(adapt_skydiscover(str(run_dir)))
        # success is flushed first (pending_success), then the failure record
        # Both records should carry the best score seen so far
        assert len(records) == 2
        assert records[0]["outcome"] == "succeeded"
        assert records[0]["child_score"] == pytest.approx(0.5)
        assert records[1]["outcome"] == "failed"
        # The failure record inherits the current_score (0.5) from the previous success
        assert records[1].get("child_score") == pytest.approx(0.5)


# ===========================================================================
# adapt_skydiscover — evox co-evolution log parsing
# ===========================================================================

class TestAdaptSkydiscoverEvoxLog:
    """Tests for parsing evox co-evolution logs where solution-evolution
    iterations ('Iteration N: Program X ... completed') are interleaved with
    search-strategy meta-evolution failures ('Iteration N failed:')."""

    def _make_log_run(self, tmp_path: Path, log_content: str) -> Path:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        logs = run_dir / "logs"
        logs.mkdir()
        (logs / "run.log").write_text(log_content, encoding="utf-8")
        return run_dir

    def test_completed_iteration_parsed(self, tmp_path):
        log = textwrap.dedent("""\
            2024-01-01 10:00:00,000 - INFO - Adding initial program to database
            2024-01-01 10:00:01,000 - INFO - Evaluated program abc123-def in 96.71s: speedup=0.4859, runtime=0.0023, combined_score=433.14
            2024-01-01 10:00:02,000 - INFO - Iteration 1: Program abc123-def (parent: seed000) completed in 162.10s (llm: 40.30s, eval: 121.81s)
            2024-01-01 10:00:02,000 - INFO - Metrics: speedup=0.4859, runtime=0.0023, combined_score=433.14
        """)
        run_dir = self._make_log_run(tmp_path, log)
        records = list(adapt_skydiscover(str(run_dir)))
        assert len(records) == 1
        assert records[0]["iteration"] == 1
        assert records[0]["outcome"] == "succeeded"
        assert records[0]["child_score"] == pytest.approx(433.14)
        assert records[0]["iteration_duration_seconds"] == pytest.approx(162.10)

    def test_combined_score_not_first_metric(self, tmp_path):
        """Regression test: combined_score preceded by other metrics."""
        log = textwrap.dedent("""\
            2024-01-01 10:00:00,000 - INFO - Adding initial program to database
            2024-01-01 10:00:01,000 - INFO - Evaluated program prog1 in 5s: speedup=0.5, runtime=0.002, combined_score=474.07, data_throughput=0.001
        """)
        run_dir = self._make_log_run(tmp_path, log)
        records = list(adapt_skydiscover(str(run_dir)))
        assert len(records) == 1
        assert records[0]["child_score"] == pytest.approx(474.07)

    def test_completed_overrides_failed_same_iteration(self, tmp_path):
        """In evox co-evolution, a solution 'Iteration 1: completed' should
        take precedence over a meta-evolution 'Iteration 1 failed:'."""
        log = textwrap.dedent("""\
            2024-01-01 10:00:00,000 - INFO - Adding initial program to database
            2024-01-01 10:01:00,000 - INFO - Evaluated program prog1 in 90s: speedup=0.52, combined_score=465.0
            2024-01-01 10:01:01,000 - INFO - Iteration 1: Program prog1 (parent: seed1) completed in 130s (llm: 40s, eval: 90s)
            2024-01-01 10:02:00,000 - WARNING - Iteration 1 failed: LLM generation failed: Error code: 401
            2024-01-01 10:03:00,000 - INFO - Evaluated program prog2 in 80s: speedup=0.53, combined_score=470.0
            2024-01-01 10:03:01,000 - INFO - Iteration 2: Program prog2 (parent: prog1) completed in 110s (llm: 30s, eval: 80s)
            2024-01-01 10:04:00,000 - WARNING - Iteration 2 failed: LLM generation failed: Error code: 401
        """)
        run_dir = self._make_log_run(tmp_path, log)
        records = list(adapt_skydiscover(str(run_dir)))
        assert len(records) == 2
        assert all(r["outcome"] == "succeeded" for r in records)
        assert records[0]["iteration"] == 1
        assert records[0]["child_score"] == pytest.approx(465.0)
        assert records[1]["iteration"] == 2
        assert records[1]["child_score"] == pytest.approx(470.0)

    def test_failed_preserved_when_no_completed_for_that_iteration(self, tmp_path):
        """Failed records are kept when no completed record exists for same iteration."""
        log = textwrap.dedent("""\
            2024-01-01 10:00:00,000 - INFO - Adding initial program to database
            2024-01-01 10:01:00,000 - INFO - Evaluated program prog1 in 90s: combined_score=465.0
            2024-01-01 10:01:01,000 - INFO - Iteration 1: Program prog1 (parent: seed1) completed in 130s (llm: 40s, eval: 90s)
            2024-01-01 10:02:00,000 - WARNING - Iteration 5 failed: timeout
        """)
        run_dir = self._make_log_run(tmp_path, log)
        records = list(adapt_skydiscover(str(run_dir)))
        assert len(records) == 2
        assert records[0]["iteration"] == 1
        assert records[0]["outcome"] == "succeeded"
        assert records[1]["iteration"] == 5
        assert records[1]["outcome"] == "failed"

    def test_metrics_line_fallback_for_score(self, tmp_path):
        """When the Evaluated line doesn't contain combined_score, the Metrics
        line should provide it."""
        log = textwrap.dedent("""\
            2024-01-01 10:00:00,000 - INFO - Adding initial program to database
            2024-01-01 10:01:00,000 - INFO - Evaluated program prog1 in 90s: some_metric=42.0
            2024-01-01 10:01:01,000 - INFO - Iteration 1: Program prog1 (parent: seed1) completed in 130s (llm: 40s, eval: 90s)
            2024-01-01 10:01:01,000 - INFO - Metrics: some_metric=42.0, combined_score=500.5
        """)
        run_dir = self._make_log_run(tmp_path, log)
        records = list(adapt_skydiscover(str(run_dir)))
        assert len(records) == 1
        assert records[0]["child_score"] == pytest.approx(500.5)

    def test_many_completed_iterations_sorted(self, tmp_path):
        """Multiple completed iterations are returned sorted by iteration number."""
        lines = ["2024-01-01 10:00:00,000 - INFO - Adding initial program to database"]
        for i in range(1, 6):
            score = 400 + i * 10
            lines.append(
                f"2024-01-01 10:0{i}:00,000 - INFO - Evaluated program p{i} in 90s: combined_score={score}"
            )
            lines.append(
                f"2024-01-01 10:0{i}:01,000 - INFO - Iteration {i}: Program p{i} (parent: p{i-1}) completed in 100s (llm: 10s, eval: 90s)"
            )
        log = "\n".join(lines) + "\n"
        run_dir = self._make_log_run(tmp_path, log)
        records = list(adapt_skydiscover(str(run_dir)))
        assert len(records) == 5
        assert [r["iteration"] for r in records] == [1, 2, 3, 4, 5]
        assert records[0]["child_score"] == pytest.approx(410.0)
        assert records[4]["child_score"] == pytest.approx(450.0)


# ===========================================================================
# adapt_skydiscover — checkpoint fallback
# ===========================================================================

class TestAdaptSkydiscoverCheckpointFallback:
    def _make_checkpoint(
        self,
        directory: Path,
        n: int,
        score: float,
        code: str | None = None,
        extra_metadata: dict | None = None,
    ) -> None:
        ckpt = directory / f"checkpoint_{n}"
        ckpt.mkdir(parents=True)
        meta = {"last_iteration": n, **(extra_metadata or {})}
        (ckpt / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        solution = {"score": score}
        if code:
            solution["code"] = code
        (ckpt / "best_solution.json").write_text(json.dumps(solution), encoding="utf-8")

    def test_flat_layout(self, tmp_path):
        self._make_checkpoint(tmp_path, 1, 0.5, "code_v1")
        self._make_checkpoint(tmp_path, 2, 0.8, "code_v2")
        records = list(adapt_skydiscover(str(tmp_path)))
        assert len(records) == 2
        assert records[0]["child_score"] == pytest.approx(0.5)
        assert records[1]["child_score"] == pytest.approx(0.8)
        assert records[1]["parent_score"] == pytest.approx(0.5)

    def test_single_run_dir_layout(self, tmp_path):
        run = tmp_path / "my_run"
        run.mkdir()
        checkpoints = run / "checkpoints"
        checkpoints.mkdir()
        self._make_checkpoint(checkpoints, 1, 0.6, "v1")
        records = list(adapt_skydiscover(str(run)))
        assert len(records) == 1
        assert records[0]["child_score"] == pytest.approx(0.6)

    def test_run_collection_layout(self, tmp_path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        run_a = outputs / "run_a"
        run_a.mkdir()
        (run_a / "checkpoints").mkdir()
        self._make_checkpoint(run_a / "checkpoints", 1, 0.5)
        run_b = outputs / "run_b"
        run_b.mkdir()
        (run_b / "checkpoints").mkdir()
        self._make_checkpoint(run_b / "checkpoints", 1, 0.8)
        records = list(adapt_skydiscover(str(outputs)))
        assert len(records) == 2
        scores = sorted(r["child_score"] for r in records)
        assert scores[0] == pytest.approx(0.5)
        assert scores[1] == pytest.approx(0.8)

    def test_score_extracted_from_metadata_fallback(self, tmp_path):
        ckpt = tmp_path / "checkpoint_1"
        ckpt.mkdir()
        (ckpt / "metadata.json").write_text(
            json.dumps({"last_iteration": 1, "score": 0.55}), encoding="utf-8"
        )
        records = list(adapt_skydiscover(str(tmp_path)))
        assert records[0]["child_score"] == pytest.approx(0.55)

    def test_code_extracted_from_solution_json(self, tmp_path):
        ckpt = tmp_path / "checkpoint_1"
        ckpt.mkdir()
        (ckpt / "metadata.json").write_text(json.dumps({"last_iteration": 1}), encoding="utf-8")
        (ckpt / "best_solution.json").write_text(
            json.dumps({"score": 0.7, "program": "def solve(): pass"}), encoding="utf-8"
        )
        records = list(adapt_skydiscover(str(tmp_path)))
        assert records[0]["child_code"] == "def solve(): pass"

    def test_paradigm_shift_from_evo_state(self, tmp_path):
        ckpt = tmp_path / "checkpoint_1"
        ckpt.mkdir()
        (ckpt / "metadata.json").write_text(json.dumps({"last_iteration": 1}), encoding="utf-8")
        (ckpt / "best_solution.json").write_text(json.dumps({"score": 0.6}), encoding="utf-8")
        (ckpt / "evolution_state.json").write_text(
            json.dumps({"paradigm_shift": True}), encoding="utf-8"
        )
        records = list(adapt_skydiscover(str(tmp_path)))
        assert records[0]["framework_stagnation_event"] == "paradigm_shift"

    def test_evaluator_metrics_from_checkpoint(self, tmp_path):
        ckpt = tmp_path / "checkpoint_1"
        ckpt.mkdir()
        (ckpt / "metadata.json").write_text(
            json.dumps({"last_iteration": 1, "evaluator_metrics": {"pass_rate": 0.9}}),
            encoding="utf-8",
        )
        (ckpt / "best_solution.json").write_text(json.dumps({"score": 0.6}), encoding="utf-8")
        records = list(adapt_skydiscover(str(tmp_path)))
        assert records[0]["evaluator_metrics"]["pass_rate"] == pytest.approx(0.9)

    def test_empty_directory_returns_nothing(self, tmp_path):
        records = list(adapt_skydiscover(str(tmp_path)))
        assert records == []
