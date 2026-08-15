"""
Tests for seed program parent_metrics injection across all adapters.

The bug: when a framework (e.g. skydiscover/adaevolve) evaluates the seed
program but does not log its metrics in the primary stats file, the
sub_metric_analyzer cannot compute improvement_vs_seed for any metric and
primary_driver is reported as 'unknown'.

The fix: each adapter now recovers seed metrics from secondary sources
(checkpoint program JSONs, run log) and injects them as parent_metrics on
the first iteration record.  _seed_metrics() in sub_metric_analyzer also
gains a fallback for an explicit iteration-0 record.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from textwrap import dedent

import pytest

from skydiscover.extras.evolve_analyzer.ingestion.checkpoint_adapter import (
    _extract_seed_metrics_from_log,
    _find_seed_program_in_checkpoints,
    _parse_log_metrics_string,
    adapt_openevolve,
    adapt_shinkaevolve,
    adapt_skydiscover,
    load_evolve_records,
    load_jsonl,
)
from skydiscover.extras.evolve_analyzer.quantitative.sub_metric_analyzer import _seed_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _make_adaevolve_jsonl(run_dir: Path, iterations: list[dict]) -> Path:
    """Create a minimal adaevolve_iteration_stats_*.jsonl file."""
    p = run_dir / "adaevolve_iteration_stats_20260101_000000.jsonl"
    lines = []
    for it in iterations:
        lines.append(json.dumps(it))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _make_program_json(prog_dir: Path, program_id: str, metrics: dict,
                       generation: int = 0, parent_id: object = None) -> Path:
    prog_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "id": program_id,
        "metrics": metrics,
        "generation": generation,
        "parent_id": parent_id,
    }
    p = prog_dir / f"{program_id}.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _make_log_with_seed(log_path: Path, seed_metrics_str: str) -> None:
    """Write a minimal skydiscover log containing a seed evaluation."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(dedent(f"""\
        2026-01-01 00:00:01,000 - main - INFO - Adding initial program to database
        2026-01-01 00:00:01,001 - root - INFO - Evaluating program seed-id-001
        2026-01-01 00:09:00,000 - evaluator - INFO - Evaluated program seed-id-001 in 539.0s: {seed_metrics_str}
        2026-01-01 00:09:01,000 - root - INFO - Evaluating program child-id-001
        2026-01-01 00:18:00,000 - evaluator - INFO - Evaluated program child-id-001 in 539.0s: combined_score=0.95
    """), encoding="utf-8")


# ===========================================================================
# _parse_log_metrics_string
# ===========================================================================

class TestParseLogMetricsString:
    def test_flat_key_values(self):
        s = "combined_score=0.9175, cpu_hit_rate=0.8313, evictions=691.0000"
        result = _parse_log_metrics_string(s)
        assert result["combined_score"] == pytest.approx(0.9175)
        assert result["cpu_hit_rate"] == pytest.approx(0.8313)
        assert result["evictions"] == pytest.approx(691.0)

    def test_nested_dict_preserved(self):
        s = "combined_score=0.92, metrics={'request_throughput': 1.47, 'mean_ttft_ms': 79.24}"
        result = _parse_log_metrics_string(s)
        assert result["combined_score"] == pytest.approx(0.92)
        assert isinstance(result["metrics"], dict)
        assert result["metrics"]["request_throughput"] == pytest.approx(1.47)
        assert result["metrics"]["mean_ttft_ms"] == pytest.approx(79.24)

    def test_nested_keys_not_duplicated_at_top_level(self):
        # Keys inside the nested dict must NOT appear twice at top level
        s = "x=1.0, metrics={'y': 2.0}"
        result = _parse_log_metrics_string(s)
        assert set(result.keys()) == {"x", "metrics"}

    def test_empty_string_returns_empty(self):
        assert _parse_log_metrics_string("") == {}

    def test_real_skydiscover_line(self):
        s = (
            "combined_score=0.9175, cpu_hit_rate=0.8313, ttft_ratio=1.0770, "
            "throughput_ratio=0.9967, evictions=691.0000, eviction_rate=0.0719, "
            "metrics={'request_throughput': 1.47, 'output_token_throughput': 329.19}"
        )
        result = _parse_log_metrics_string(s)
        assert result["cpu_hit_rate"] == pytest.approx(0.8313)
        assert result["eviction_rate"] == pytest.approx(0.0719)
        assert result["metrics"]["request_throughput"] == pytest.approx(1.47)


# ===========================================================================
# _extract_seed_metrics_from_log
# ===========================================================================

class TestExtractSeedMetricsFromLog:
    def test_returns_empty_when_no_file(self, tmp_path):
        result = _extract_seed_metrics_from_log(tmp_path / "nonexistent.log")
        assert result == {}

    def test_returns_empty_when_no_initial_program_line(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text(
            "2026-01-01 00:00:01,000 - evaluator - INFO - Evaluated program x in 1s: combined_score=0.9\n",
            encoding="utf-8",
        )
        assert _extract_seed_metrics_from_log(log) == {}

    def test_extracts_metrics_after_initial_program_line(self, tmp_path):
        log = tmp_path / "run.log"
        _make_log_with_seed(
            log,
            "combined_score=0.9175, cpu_hit_rate=0.8313, eviction_rate=0.0719",
        )
        result = _extract_seed_metrics_from_log(log)
        assert result["combined_score"] == pytest.approx(0.9175)
        assert result["cpu_hit_rate"] == pytest.approx(0.8313)
        assert result["eviction_rate"] == pytest.approx(0.0719)

    def test_extracts_nested_metrics(self, tmp_path):
        log = tmp_path / "run.log"
        _make_log_with_seed(
            log,
            "combined_score=0.92, metrics={'request_throughput': 1.47}",
        )
        result = _extract_seed_metrics_from_log(log)
        assert result["metrics"]["request_throughput"] == pytest.approx(1.47)


# ===========================================================================
# _find_seed_program_in_checkpoints
# ===========================================================================

class TestFindSeedProgramInCheckpoints:
    def test_returns_empty_when_no_checkpoints_dir(self, tmp_path):
        result = _find_seed_program_in_checkpoints(tmp_path)
        assert result == {}

    def test_finds_by_explicit_seed_id(self, tmp_path):
        prog_dir = tmp_path / "checkpoints" / "checkpoint_5" / "programs"
        metrics = {"combined_score": 0.917, "cpu_hit_rate": 0.83}
        _make_program_json(prog_dir, "seed-abc", metrics, generation=0)
        result = _find_seed_program_in_checkpoints(tmp_path, seed_id="seed-abc")
        assert result["metrics"]["cpu_hit_rate"] == pytest.approx(0.83)

    def test_finds_by_generation_zero(self, tmp_path):
        prog_dir = tmp_path / "checkpoints" / "checkpoint_5" / "programs"
        metrics = {"combined_score": 0.917}
        _make_program_json(prog_dir, "seed-gen0", metrics, generation=0, parent_id=None)
        result = _find_seed_program_in_checkpoints(tmp_path)
        assert result.get("id") == "seed-gen0"

    def test_finds_by_null_parent_id(self, tmp_path):
        prog_dir = tmp_path / "checkpoints" / "checkpoint_5" / "programs"
        metrics = {"combined_score": 0.917}
        _make_program_json(prog_dir, "seed-nop", metrics, generation=1, parent_id=None)
        result = _find_seed_program_in_checkpoints(tmp_path)
        assert result.get("id") == "seed-nop"

    def test_returns_empty_when_no_matching_program(self, tmp_path):
        prog_dir = tmp_path / "checkpoints" / "checkpoint_5" / "programs"
        # generation=1 AND parent_id set → not a seed
        _make_program_json(prog_dir, "child-001", {"combined_score": 0.9},
                           generation=1, parent_id="something")
        result = _find_seed_program_in_checkpoints(tmp_path)
        assert result == {}


# ===========================================================================
# adapt_skydiscover — JSONL path: seed from checkpoints
# ===========================================================================

class TestSkydiscoverSeedFromCheckpoints:
    """Seed metrics injected via checkpoint program JSON (primary JSONL path)."""

    def _make_run_dir(self, tmp_path: Path,
                      seed_id: str, seed_metrics: dict,
                      child_id: str = "child-001") -> Path:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Seed program in checkpoint programs dir
        prog_dir = run_dir / "checkpoints" / "checkpoint_5" / "programs"
        _make_program_json(prog_dir, seed_id, seed_metrics, generation=0, parent_id=None)

        # adaevolve JSONL: first iteration's child_program has parent_id = seed_id
        _make_adaevolve_jsonl(run_dir, [
            {
                "iteration": 1,
                "timestamp": "2026-01-01T00:00:01",
                "global": {"best_program": {"metrics": {"combined_score": 0.95}}},
                "iteration_result": {
                    "success": True,
                    "child_program": {
                        "id": child_id,
                        "parent_id": seed_id,
                        "metrics": {
                            "combined_score": 0.95,
                            "cpu_hit_rate": 0.85,
                        },
                    },
                },
                "islands": [],
                "sampling": {},
                "paradigm": {},
                "dynamic_islands": {},
            }
        ])
        return run_dir

    def test_parent_metrics_injected_from_checkpoint(self, tmp_path):
        seed_metrics = {"combined_score": 0.85, "cpu_hit_rate": 0.78}
        run_dir = self._make_run_dir(tmp_path, "seed-xyz", seed_metrics)
        records = list(adapt_skydiscover(str(run_dir)))
        assert len(records) == 1
        assert "parent_metrics" in records[0], "parent_metrics must be set on first record"
        assert records[0]["parent_metrics"]["cpu_hit_rate"] == pytest.approx(0.78)

    def test_parent_program_id_temp_field_cleaned_up(self, tmp_path):
        seed_metrics = {"combined_score": 0.85}
        run_dir = self._make_run_dir(tmp_path, "seed-xyz", seed_metrics)
        records = list(adapt_skydiscover(str(run_dir)))
        assert "_parent_program_id" not in records[0]

    def test_seed_not_found_no_error(self, tmp_path):
        """If seed isn't in checkpoints, first record simply has no parent_metrics."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _make_adaevolve_jsonl(run_dir, [
            {
                "iteration": 1,
                "timestamp": "2026-01-01T00:00:01",
                "global": {},
                "iteration_result": {
                    "success": True,
                    "child_program": {
                        "id": "child-001",
                        "parent_id": "nonexistent-seed",
                        "metrics": {"combined_score": 0.9},
                    },
                },
                "islands": [],
                "sampling": {},
                "paradigm": {},
                "dynamic_islands": {},
            }
        ])
        records = list(adapt_skydiscover(str(run_dir)))
        # Must not raise; parent_metrics may or may not be present
        assert len(records) == 1

    def test_sub_metric_seed_values_populated(self, tmp_path):
        """End-to-end: after adapter fix, seed values appear in sub_metric stats."""
        seed_metrics = {
            "combined_score": 0.85,
            "cpu_hit_rate": 0.70,
            "ttft_ratio": 1.0,
        }
        run_dir = self._make_run_dir(tmp_path, "seed-xyz", seed_metrics)
        records = load_evolve_records("skydiscover", str(run_dir))
        assert records[0].get("parent_metrics", {}).get("cpu_hit_rate") == pytest.approx(0.70)


# ===========================================================================
# adapt_skydiscover — log path: seed from log file
# ===========================================================================

class TestSkydiscoverSeedFromLog:
    """Seed metrics injected from log when there is no adaevolve JSONL."""

    def _make_log_only_run(self, tmp_path: Path, seed_metrics_str: str) -> Path:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "checkpoints").mkdir()
        _make_log_with_seed(run_dir / "logs" / "run.log", seed_metrics_str)
        return run_dir

    def test_parent_metrics_injected_from_log(self, tmp_path):
        run_dir = self._make_log_only_run(
            tmp_path, "combined_score=0.9175, cpu_hit_rate=0.8313, eviction_rate=0.0719"
        )
        # The log path only triggers when there's a checkpoint so the log parser
        # can return at least one record.  Add a minimal checkpoint.
        ckpt = run_dir / "checkpoints" / "checkpoint_1"
        ckpt.mkdir()
        (ckpt / "metadata.json").write_text(
            json.dumps({"last_iteration": 1, "score": 0.95}), encoding="utf-8"
        )
        records = list(adapt_skydiscover(str(run_dir)))
        # log_records are produced; first should have parent_metrics
        first = records[0]
        assert "parent_metrics" in first
        assert first["parent_metrics"]["cpu_hit_rate"] == pytest.approx(0.8313)
        assert first["parent_metrics"]["eviction_rate"] == pytest.approx(0.0719)


# ===========================================================================
# adapt_skydiscover — checkpoint fallback: seed from programs dir
# ===========================================================================

class TestSkydiscoverSeedFromCheckpointFallback:
    """Seed metrics injected via checkpoint program scan (no JSONL, no log)."""

    def test_parent_metrics_injected_in_checkpoint_fallback(self, tmp_path):
        # Checkpoint with a best program
        ckpt = tmp_path / "checkpoint_1"
        ckpt.mkdir()
        (ckpt / "metadata.json").write_text(
            json.dumps({"last_iteration": 1}), encoding="utf-8"
        )
        (ckpt / "best_solution.json").write_text(
            json.dumps({"score": 0.95, "program": "def solve(): pass"}),
            encoding="utf-8",
        )
        # Seed program with generation=0
        seed_metrics = {"combined_score": 0.85, "cpu_hit_rate": 0.78}
        _make_program_json(
            ckpt / "programs", "seed-gen0", seed_metrics, generation=0, parent_id=None
        )
        records = list(adapt_skydiscover(str(tmp_path)))
        assert len(records) == 1
        assert "parent_metrics" in records[0]
        assert records[0]["parent_metrics"]["cpu_hit_rate"] == pytest.approx(0.78)


# ===========================================================================
# adapt_shinkaevolve — parent_metrics column
# ===========================================================================

class TestShinkaevolveSeedParentMetrics:
    def _make_db(self, tmp_path: Path, rows: list[dict], table: str = "iterations") -> Path:
        db = tmp_path / "run.db"
        conn = sqlite3.connect(str(db))
        if rows:
            cols = list(rows[0].keys())
            col_defs = ", ".join(f"{c} TEXT" for c in cols)
            conn.execute(f"CREATE TABLE {table} ({col_defs})")
            placeholders = ", ".join("?" for _ in cols)
            for row in rows:
                conn.execute(
                    f"INSERT INTO {table} VALUES ({placeholders})",
                    [row.get(c) for c in cols],
                )
        else:
            conn.execute(f"CREATE TABLE {table} (iteration TEXT, score TEXT)")
        conn.commit()
        conn.close()
        return db

    def test_parent_metrics_column_parsed(self, tmp_path):
        pm = {"cpu_hit_rate": 0.78, "combined_score": 0.85}
        db = self._make_db(tmp_path, [
            {
                "iteration": "1",
                "score": "0.92",
                "parent_metrics": json.dumps(pm),
            }
        ])
        records = list(adapt_shinkaevolve(str(db)))
        assert "parent_metrics" in records[0]
        assert records[0]["parent_metrics"]["cpu_hit_rate"] == pytest.approx(0.78)

    def test_parent_metrics_in_metadata_json(self, tmp_path):
        """parent_metrics embedded inside the metadata JSON blob."""
        pm = {"cpu_hit_rate": 0.78}
        meta = {"parent_metrics": pm, "extra": "info"}
        db = self._make_db(tmp_path, [
            {
                "iteration": "1",
                "score": "0.92",
                "metadata": json.dumps(meta),
            }
        ])
        records = list(adapt_shinkaevolve(str(db)))
        assert records[0].get("parent_metrics", {}).get("cpu_hit_rate") == pytest.approx(0.78)

    def test_no_parent_metrics_no_error(self, tmp_path):
        db = self._make_db(tmp_path, [{"iteration": "1", "score": "0.9"}])
        records = list(adapt_shinkaevolve(str(db)))
        assert len(records) == 1
        assert "parent_metrics" not in records[0]


# ===========================================================================
# _seed_metrics fallback — iteration-0 record
# ===========================================================================

class TestSeedMetricsFallback:
    def test_uses_parent_metrics_of_earliest_iteration(self):
        records = [
            {
                "iteration": 2,
                "evaluator_metrics": {"accuracy": 0.95},
                "parent_metrics": {"accuracy": 0.70},
            },
            {
                "iteration": 1,
                "evaluator_metrics": {"accuracy": 0.80},
                "parent_metrics": {"accuracy": 0.60},
            },
        ]
        seed = _seed_metrics(records)
        # Earliest iteration is 1; its parent_metrics has accuracy=0.60
        assert seed["accuracy"] == pytest.approx(0.60)

    def test_fallback_to_iteration_zero_evaluator_metrics(self):
        records = [
            {"iteration": 0, "evaluator_metrics": {"accuracy": 0.55}, "child_score": 0.55},
            {"iteration": 1, "evaluator_metrics": {"accuracy": 0.80}, "child_score": 0.80},
        ]
        seed = _seed_metrics(records)
        assert seed["accuracy"] == pytest.approx(0.55)

    def test_parent_metrics_takes_priority_over_iteration_zero(self):
        records = [
            {
                "iteration": 0,
                "evaluator_metrics": {"accuracy": 0.55},
                "child_score": 0.55,
            },
            {
                "iteration": 1,
                "evaluator_metrics": {"accuracy": 0.80},
                "parent_metrics": {"accuracy": 0.60},  # ← should win
                "child_score": 0.80,
            },
        ]
        seed = _seed_metrics(records)
        assert seed["accuracy"] == pytest.approx(0.60)

    def test_empty_records_returns_empty(self):
        assert _seed_metrics([]) == {}

    def test_no_parent_metrics_no_iteration_zero_returns_empty(self):
        records = [
            {"iteration": 1, "evaluator_metrics": {"accuracy": 0.80}, "child_score": 0.80},
        ]
        assert _seed_metrics(records) == {}
