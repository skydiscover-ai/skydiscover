"""Unit tests for the AlphaEvolve evaluator adapter and result extraction."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from skydiscover.extras.external.alphaevolve_backend import (
    _extract_best_program,
    _make_alphaevolve_evaluator,
)

# Evaluator that records the path it was handed, so tests can assert on the
# temp-file extension the adapter chose.
_PATH_ECHO_EVALUATOR = """
import os

_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seen.txt")


def evaluate(program_path):
    with open(_LOG, "a") as fh:
        fh.write(program_path + "\\n")
    with open(program_path) as src:
        return {"combined_score": float(len(src.read()))}
"""

_FAILING_EVALUATOR = """
def evaluate(program_path):
    raise RuntimeError("boom")
"""


def _write_evaluator(tmp_path: Path, source: str) -> str:
    path = tmp_path / "evaluator.py"
    path.write_text(source)
    return str(path)


def _seen_paths(tmp_path: Path) -> List[str]:
    log = tmp_path / "seen.txt"
    return log.read_text().split() if log.exists() else []


def _candidate(path: str, content: str = "x = 1") -> Dict[str, Any]:
    return {"content": {"files": [{"path": path, "content": content}]}}


@pytest.fixture
def isolated_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point tempfile at a private dir so leftovers are attributable."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    return scratch


class TestEvaluatorSuffix:
    """The candidate temp file must carry the run's language extension."""

    def test_uses_candidate_file_extension(self, tmp_path: Path) -> None:
        evaluator = _make_alphaevolve_evaluator(
            _write_evaluator(tmp_path, _PATH_ECHO_EVALUATOR),
            file_suffix=".py",
        )

        evaluator(_candidate("kernel.cu", "__global__ void k() {}"))

        assert _seen_paths(tmp_path)[0].endswith(".cu")

    def test_falls_back_to_configured_suffix(self, tmp_path: Path) -> None:
        evaluator = _make_alphaevolve_evaluator(
            _write_evaluator(tmp_path, _PATH_ECHO_EVALUATOR),
            file_suffix=".cpp",
        )

        # No path on the candidate -> the configured file_suffix wins.
        evaluator(_candidate("", "int main() { return 0; }"))

        assert _seen_paths(tmp_path)[0].endswith(".cpp")

    def test_python_default(self, tmp_path: Path) -> None:
        evaluator = _make_alphaevolve_evaluator(
            _write_evaluator(tmp_path, _PATH_ECHO_EVALUATOR)
        )

        evaluator(_candidate("seed.py"))

        assert _seen_paths(tmp_path)[0].endswith(".py")


class TestEvaluatorAdapter:
    """Behaviour of the wrapped evaluator closure."""

    def test_scores_are_pre_wrapped(self, tmp_path: Path) -> None:
        evaluator = _make_alphaevolve_evaluator(
            _write_evaluator(tmp_path, _PATH_ECHO_EVALUATOR)
        )

        result = evaluator(_candidate("seed.py", "abc"))

        assert result["scores"]["scores"] == [
            {"metric": "combined_score", "score": 3.0}
        ]

    def test_monitor_callback_gets_run_language(self, tmp_path: Path) -> None:
        seen: List[Any] = []

        evaluator = _make_alphaevolve_evaluator(
            _write_evaluator(tmp_path, _PATH_ECHO_EVALUATOR),
            monitor_callback=lambda prog, it: seen.append(prog),
            file_suffix=".cu",
            language="cuda",
        )

        evaluator(_candidate("kernel.cu", "__global__ void k() {}"))

        assert len(seen) == 1
        assert seen[0].language == "cuda"

    def test_evaluator_error_returns_insight(self, tmp_path: Path) -> None:
        evaluator = _make_alphaevolve_evaluator(
            _write_evaluator(tmp_path, _FAILING_EVALUATOR)
        )

        result = evaluator(_candidate("seed.py"))

        assert result["scores"]["scores"] == []
        assert "boom" in result["insights"]["error"]

    def test_no_files_returns_insight(self, tmp_path: Path) -> None:
        evaluator = _make_alphaevolve_evaluator(
            _write_evaluator(tmp_path, _PATH_ECHO_EVALUATOR)
        )

        result = evaluator({"content": {"files": []}})

        assert result["scores"]["scores"] == []
        assert "error" in result["insights"]

    def test_temp_file_is_cleaned_up(
        self, tmp_path: Path, isolated_tmpdir: Path
    ) -> None:
        evaluator = _make_alphaevolve_evaluator(
            _write_evaluator(tmp_path, _PATH_ECHO_EVALUATOR)
        )

        evaluator(_candidate("seed.py"))

        assert os.listdir(isolated_tmpdir) == []

    def test_temp_file_is_cleaned_up_on_evaluator_error(
        self, tmp_path: Path, isolated_tmpdir: Path
    ) -> None:
        evaluator = _make_alphaevolve_evaluator(
            _write_evaluator(tmp_path, _FAILING_EVALUATOR)
        )

        evaluator(_candidate("seed.py"))

        assert os.listdir(isolated_tmpdir) == []


class _FakeExperiment:
    def __init__(self, programs: List[Dict[str, Any]]) -> None:
        self._programs = programs

    def list_programs(self, params: Any = None) -> Dict[str, Any]:
        return {"alphaEvolvePrograms": self._programs}


def _program(code: str, score: float) -> Dict[str, Any]:
    return {
        "content": {"files": [{"path": "p.py", "content": code}]},
        "evaluation": {
            "scores": {"scores": [{"metric": "combined_score", "score": score}]}
        },
    }


class TestExtractBestProgram:
    """The API's ordering is not trusted, the best score must win."""

    def test_picks_highest_score_when_unordered(self) -> None:
        experiment = _FakeExperiment(
            [_program("low", 0.1), _program("high", 0.9), _program("mid", 0.5)]
        )

        code, metrics = _extract_best_program(experiment)

        assert code == "high"
        assert metrics["combined_score"] == 0.9

    def test_empty_response(self) -> None:
        assert _extract_best_program(_FakeExperiment([])) == ("", {})

    def test_program_without_scores_is_not_preferred(self) -> None:
        experiment = _FakeExperiment(
            [
                {"content": {"files": [{"content": "unscored"}]}},
                _program("ok", 0.2),
            ]
        )

        code, metrics = _extract_best_program(experiment)

        assert code == "ok"
        assert metrics["combined_score"] == 0.2
