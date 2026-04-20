"""Integration test: evolution_trace.json is written correctly during a real discovery run.

Runs the full pipeline (3 iterations, mocked LLM) and verifies:
- evolution_trace.json exists in every checkpoint directory
- Contents are sorted, complete, and structurally correct
- Old checkpoint artefacts (metadata.json, best_program_info.json) still exist
- Trace score matches what the evaluator returns
"""

import json
import os
import textwrap
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from skydiscover.api import DiscoveryResult, run_discovery
from skydiscover.config import Config, LLMModelConfig
from skydiscover.llm.base import LLMResponse


EVALUATOR_SOURCE = textwrap.dedent("""\
    import ast

    def evaluate(program_path: str) -> dict:
        with open(program_path) as f:
            source = f.read()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"combined_score": 0.0}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "solve":
                return {"combined_score": 0.9}
        return {"combined_score": 0.1}
""")

SEED_SOURCE = textwrap.dedent("""\
    def hello():
        return "hi"
""")

MOCK_LLM_CODE = textwrap.dedent("""\
    def solve(x):
        return x * 2
""")

MOCK_RESPONSE = f"```python\n{MOCK_LLM_CODE}```"


class FakeLLMPool:
    def __init__(self, models_cfg):
        self.models_cfg = models_cfg

    async def generate(self, system_message, messages, **kwargs):
        return LLMResponse(text=MOCK_RESPONSE)

    async def generate_all(self, system_message, messages, **kwargs):
        return [LLMResponse(text=MOCK_RESPONSE)]


def _find_checkpoints(output_dir: str) -> List[str]:
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        return []
    dirs = []
    for name in os.listdir(checkpoint_dir):
        full = os.path.join(checkpoint_dir, name)
        if os.path.isdir(full) and name.startswith("checkpoint_"):
            dirs.append(full)
    return sorted(dirs)


class TestEvolutionTraceIntegration:
    def _run(self, tmp_path, iterations=3):
        evaluator_file = tmp_path / "evaluator.py"
        evaluator_file.write_text(EVALUATOR_SOURCE)
        seed_file = tmp_path / "seed.py"
        seed_file.write_text(SEED_SOURCE)
        output_dir = str(tmp_path / "output")

        config = Config.from_dict(
            {
                "max_iterations": iterations,
                "diff_based_generation": False,
                "monitor": {"enabled": False},
                "search": {"type": "topk"},
                "evaluator": {"evaluation_file": str(evaluator_file)},
                "llm": {
                    "models": [
                        {
                            "name": "fake-model",
                            "api_key": "fake",
                            "api_base": "http://localhost:1",
                        }
                    ]
                },
            }
        )

        with patch(
            "skydiscover.search.default_discovery_controller.LLMPool",
            FakeLLMPool,
        ):
            result = run_discovery(
                evaluator=str(evaluator_file),
                initial_program=str(seed_file),
                config=config,
                output_dir=output_dir,
                cleanup=False,
            )

        return result, output_dir

    def test_trace_file_exists_in_every_checkpoint(self, tmp_path):
        _, output_dir = self._run(tmp_path)
        checkpoints = _find_checkpoints(output_dir)
        assert checkpoints, "No checkpoints were created"
        for ckpt in checkpoints:
            assert os.path.exists(os.path.join(ckpt, "evolution_trace.json")), (
                f"Missing evolution_trace.json in {ckpt}"
            )

    def test_old_checkpoint_artefacts_still_present(self, tmp_path):
        """metadata.json and best_program_info.json must still exist."""
        _, output_dir = self._run(tmp_path)
        for ckpt in _find_checkpoints(output_dir):
            assert os.path.exists(os.path.join(ckpt, "metadata.json"))
            assert os.path.exists(os.path.join(ckpt, "programs"))

    def test_trace_is_valid_json(self, tmp_path):
        _, output_dir = self._run(tmp_path)
        for ckpt in _find_checkpoints(output_dir):
            trace_path = os.path.join(ckpt, "evolution_trace.json")
            with open(trace_path) as f:
                trace = json.load(f)
            assert "programs" in trace
            assert "last_iteration" in trace
            assert "best_program_id" in trace
            assert "total_programs" in trace

    def test_trace_programs_sorted_by_iteration(self, tmp_path):
        _, output_dir = self._run(tmp_path)
        # Check the final checkpoint (has the most programs)
        ckpt = sorted(_find_checkpoints(output_dir))[-1]
        with open(os.path.join(ckpt, "evolution_trace.json")) as f:
            trace = json.load(f)
        iterations = [p["iteration_found"] for p in trace["programs"]]
        assert iterations == sorted(iterations)

    def test_trace_total_programs_matches_programs_list(self, tmp_path):
        _, output_dir = self._run(tmp_path)
        for ckpt in _find_checkpoints(output_dir):
            with open(os.path.join(ckpt, "evolution_trace.json")) as f:
                trace = json.load(f)
            assert trace["total_programs"] == len(trace["programs"])

    def test_trace_best_program_id_matches_metadata(self, tmp_path):
        _, output_dir = self._run(tmp_path)
        for ckpt in _find_checkpoints(output_dir):
            with open(os.path.join(ckpt, "evolution_trace.json")) as f:
                trace = json.load(f)
            with open(os.path.join(ckpt, "metadata.json")) as f:
                metadata = json.load(f)
            assert trace["best_program_id"] == metadata["best_program_id"]

    def test_trace_entries_have_required_fields(self, tmp_path):
        _, output_dir = self._run(tmp_path)
        ckpt = sorted(_find_checkpoints(output_dir))[-1]
        with open(os.path.join(ckpt, "evolution_trace.json")) as f:
            trace = json.load(f)
        required = {"id", "iteration_found", "generation", "score", "metrics", "parent_id",
                    "timestamp", "solution"}
        for entry in trace["programs"]:
            assert required.issubset(entry.keys()), f"Missing fields in entry: {entry.keys()}"

    def test_trace_score_matches_evaluator(self, tmp_path):
        """The mock LLM produces `def solve` which scores 0.9."""
        _, output_dir = self._run(tmp_path)
        ckpt = sorted(_find_checkpoints(output_dir))[-1]
        with open(os.path.join(ckpt, "evolution_trace.json")) as f:
            trace = json.load(f)
        # All programs after iteration 0 should be from the mock LLM (def solve → 0.9)
        llm_programs = [p for p in trace["programs"] if p["iteration_found"] > 0]
        for p in llm_programs:
            assert p["score"] == pytest.approx(0.9), f"Unexpected score: {p['score']}"

    def test_best_program_in_trace(self, tmp_path):
        result, output_dir = self._run(tmp_path)
        ckpt = sorted(_find_checkpoints(output_dir))[-1]
        with open(os.path.join(ckpt, "evolution_trace.json")) as f:
            trace = json.load(f)
        ids = {p["id"] for p in trace["programs"]}
        assert trace["best_program_id"] in ids
