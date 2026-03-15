"""Smoke tests for the end-to-end discovery pipeline and unit guards for recent bug fixes."""

import os
import textwrap
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from skydiscover.api import DiscoveryResult, run_discovery
from skydiscover.config import Config, LLMModelConfig
from skydiscover.evaluation.evaluator import Evaluator, EvaluatorConfig
from skydiscover.llm.base import LLMResponse
from skydiscover.llm.llm_pool import LLMPool

# ---------------------------------------------------------------------------
# Inline evaluator source — scores programs with `def solve` higher
# ---------------------------------------------------------------------------
EVALUATOR_SOURCE = textwrap.dedent("""\
    import ast

    def evaluate(program_path: str) -> dict:
        with open(program_path, "r") as f:
            source = f.read()

        score = 0.1  # baseline for any non-empty program
        try:
            tree = ast.parse(source)
            # reward programs that define a `solve` function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "solve":
                    score = 0.8
                    break
        except SyntaxError:
            score = 0.0

        return {"combined_score": score}
    """)

# ---------------------------------------------------------------------------
# Inline seed program — intentionally does NOT define `solve` so it scores low
# ---------------------------------------------------------------------------
SEED_SOURCE = textwrap.dedent("""\
    def hello():
        return "hi"
    """)

# ---------------------------------------------------------------------------
# Mock LLM response — a full-rewrite code block containing `def solve`
# ---------------------------------------------------------------------------
MOCK_LLM_CODE = textwrap.dedent("""\
    def solve(x):
        return x ** 2 + 1
    """)

MOCK_RESPONSE_TEXT = f"```python\n{MOCK_LLM_CODE}```"


# ---------------------------------------------------------------------------
# FakeLLMPool — replaces the real LLMPool so no OpenAI client is created
# ---------------------------------------------------------------------------
class FakeLLMPool:
    """Drop-in replacement for LLMPool that returns a canned response."""

    def __init__(self, models_cfg: List[LLMModelConfig]):
        # Intentionally do NOT create real clients.
        self.models_cfg = models_cfg

    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        return LLMResponse(text=MOCK_RESPONSE_TEXT)

    async def generate_all(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> List[LLMResponse]:
        return [LLMResponse(text=MOCK_RESPONSE_TEXT)]


# ===========================================================================
# Smoke test: end-to-end pipeline with mocked LLM
# ===========================================================================
class TestSmokePipeline:
    def test_run_discovery_returns_result(self, tmp_path):
        """run_discovery completes 2 iterations and returns a valid DiscoveryResult."""

        # Write evaluator and seed program to tmp_path
        evaluator_file = tmp_path / "evaluator.py"
        evaluator_file.write_text(EVALUATOR_SOURCE)

        seed_file = tmp_path / "seed.py"
        seed_file.write_text(SEED_SOURCE)

        output_dir = str(tmp_path / "output")

        config = Config.from_dict(
            {
                "max_iterations": 2,
                "diff_based_generation": False,
                "monitor": {"enabled": False},
                "search": {"type": "topk"},
                "evaluator": {"evaluation_file": str(evaluator_file)},
                "llm": {
                    "models": [
                        {"name": "fake-model", "api_key": "fake", "api_base": "http://localhost:1"}
                    ],
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

        # Assertions
        assert isinstance(result, DiscoveryResult)
        assert result.best_score >= 0.8  # mock LLM produces `def solve` → scored 0.8
        assert result.best_solution  # non-empty string
        assert os.path.isdir(output_dir)


# ===========================================================================
# Unit guards for recent bug fixes
# ===========================================================================
class TestBugFixGuards:
    def test_llm_pool_raises_on_zero_weights(self):
        """LLMPool must raise ValueError when all model weights are zero."""
        cfgs = [
            LLMModelConfig(name="m1", weight=0.0, api_key="k", api_base="http://x"),
            LLMModelConfig(name="m2", weight=0.0, api_key="k", api_base="http://x"),
        ]
        with pytest.raises(ValueError, match="weights must not all be zero"):
            # Patch OpenAILLM so we never touch a real client
            with patch("skydiscover.llm.llm_pool.OpenAILLM"):
                LLMPool(cfgs)

    def test_evaluator_unique_module_names(self, tmp_path):
        """Two Evaluator instances for the same file must get distinct _module_name values."""
        eval_file = tmp_path / "eval.py"
        eval_file.write_text("def evaluate(program_path):\n    return {'combined_score': 1.0}\n")

        cfg = EvaluatorConfig(evaluation_file=str(eval_file))
        ev1 = Evaluator(config=cfg)
        ev2 = Evaluator(config=cfg)

        assert ev1._module_name != ev2._module_name
