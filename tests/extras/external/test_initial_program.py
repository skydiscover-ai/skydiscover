"""Unit tests for AlphaEvolve initial program construction."""

import os
from pathlib import Path

import pytest

from skydiscover.extras.external.alphaevolve_backend import (
    _EVOLVE_BLOCK_END,
    _EVOLVE_BLOCK_MARKER_END,
    _EVOLVE_BLOCK_MARKER_START,
    _EVOLVE_BLOCK_START,
    _build_initial_program,
    _comment_prefix_for_path,
    _has_evolve_block_markers,
)


class TestBuildInitialProgram:
    """Tests for _build_initial_program."""

    def test_auto_wraps_evolve_block(self, tmp_path: Path) -> None:
        seed = tmp_path / "seed.py"
        seed.write_text("def solve(): return 42")

        result = _build_initial_program(str(seed))

        content = result["content"]["files"][0]["content"]
        assert content.startswith(_EVOLVE_BLOCK_START)
        assert content.endswith(_EVOLVE_BLOCK_END)

    def test_passes_through_existing_markers(self, tmp_path: Path) -> None:
        code = f"{_EVOLVE_BLOCK_START}\ndef solve(): return 42\n{_EVOLVE_BLOCK_END}"
        seed = tmp_path / "seed.py"
        seed.write_text(code)

        result = _build_initial_program(str(seed))

        content = result["content"]["files"][0]["content"]
        assert content.count(_EVOLVE_BLOCK_START) == 1

    def test_missing_path_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            _build_initial_program("/nonexistent/path.py")

    def test_empty_path_raises(self) -> None:
        with pytest.raises(ValueError, match="required"):
            _build_initial_program("")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        seed = tmp_path / "empty.py"
        seed.write_text("   ")

        with pytest.raises(ValueError, match="empty"):
            _build_initial_program(str(seed))

    def test_payload_structure(self, tmp_path: Path) -> None:
        seed = tmp_path / "prog.py"
        seed.write_text("x = 1")

        result = _build_initial_program(str(seed))

        assert "content" in result
        assert "files" in result["content"]
        files = result["content"]["files"]
        assert len(files) == 1
        assert files[0]["path"] == "prog.py"
        assert "evaluation" in result


class TestCommentPrefixForPath:
    """Tests for _comment_prefix_for_path helper."""

    @pytest.mark.parametrize(
        "ext",
        [".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"],
    )
    def test_cpp_extensions(self, ext: str) -> None:
        assert _comment_prefix_for_path(f"foo{ext}") == "//"

    @pytest.mark.parametrize(
        "ext",
        [".py", ".yaml", ".txt", ".rs"],
    )
    def test_python_and_default(self, ext: str) -> None:
        assert _comment_prefix_for_path(f"foo{ext}") == "#"


class TestHasEvolveBlockMarkers:
    """Tests for _has_evolve_block_markers helper."""

    def test_detects_python_style(self) -> None:
        content = "# EVOLVE-BLOCK-START\ncode\n# EVOLVE-BLOCK-END"
        assert _has_evolve_block_markers(content) is True

    def test_detects_cpp_style(self) -> None:
        content = "// EVOLVE-BLOCK-START\ncode\n// EVOLVE-BLOCK-END"
        assert _has_evolve_block_markers(content) is True

    def test_no_markers(self) -> None:
        assert _has_evolve_block_markers("just code") is False


class TestBuildInitialProgramLanguageAware:
    """Tests for language-aware _build_initial_program behaviour."""

    def test_cpp_auto_wraps_with_cpp_markers(self, tmp_path: Path) -> None:
        seed = tmp_path / "prefetcher.cpp"
        seed.write_text("int main() { return 0; }")

        result = _build_initial_program(str(seed))

        content = result["content"]["files"][0]["content"]
        assert content.startswith("// EVOLVE-BLOCK-START")
        assert content.endswith("// EVOLVE-BLOCK-END")

    def test_cpp_passes_through_existing_cpp_markers(
        self, tmp_path: Path
    ) -> None:
        code = "// EVOLVE-BLOCK-START\nint main() { return 0; }\n// EVOLVE-BLOCK-END"
        seed = tmp_path / "prefetcher.cpp"
        seed.write_text(code)

        result = _build_initial_program(str(seed))

        content = result["content"]["files"][0]["content"]
        assert content.count("EVOLVE-BLOCK-START") == 1

    def test_py_auto_wraps_with_python_markers(self, tmp_path: Path) -> None:
        seed = tmp_path / "solver.py"
        seed.write_text("def solve(): return 42")

        result = _build_initial_program(str(seed))

        content = result["content"]["files"][0]["content"]
        assert content.startswith("# EVOLVE-BLOCK-START")
        assert content.endswith("# EVOLVE-BLOCK-END")

    def test_py_passes_through_existing_python_markers(
        self, tmp_path: Path
    ) -> None:
        code = "# EVOLVE-BLOCK-START\ndef solve(): return 42\n# EVOLVE-BLOCK-END"
        seed = tmp_path / "solver.py"
        seed.write_text(code)

        result = _build_initial_program(str(seed))

        content = result["content"]["files"][0]["content"]
        assert content.count("EVOLVE-BLOCK-START") == 1

    def test_header_file_uses_cpp_markers(self, tmp_path: Path) -> None:
        """A .h header file should get // style markers."""
        seed = tmp_path / "prefetcher.h"
        seed.write_text("#pragma once\nvoid prefetch();")

        result = _build_initial_program(str(seed))

        content = result["content"]["files"][0]["content"]
        assert content.startswith("// EVOLVE-BLOCK-START")
        assert content.endswith("// EVOLVE-BLOCK-END")

    def test_cpp_no_double_wrap_python_markers(
        self, tmp_path: Path
    ) -> None:
        """A .cpp file that already has # EVOLVE-BLOCK-START should pass through.

        The bare marker string is detected regardless of comment prefix,
        preventing double-wrapping even when the prefix doesn't match the
        file extension.
        """
        code = "# EVOLVE-BLOCK-START\nint main() { return 0; }\n# EVOLVE-BLOCK-END"
        seed = tmp_path / "prefetcher.cpp"
        seed.write_text(code)

        result = _build_initial_program(str(seed))

        content = result["content"]["files"][0]["content"]
        assert content.count("EVOLVE-BLOCK-START") == 1
