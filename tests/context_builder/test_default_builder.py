"""Tests for DefaultContextBuilder omitting repeated fixed code from context programs (issue #61)."""

import pytest

from skydiscover.optimize.config import Config
from skydiscover.optimize.context_builder.default import DefaultContextBuilder
from skydiscover.optimize.context_builder.default.builder import _OMITTED_FIXED_CODE
from skydiscover.optimize.search.base_database import Program

FIXED_PREFIX = '''"""Constructor-based circle packing for n=26 circles."""

import numpy as np

N_CIRCLES = 26
TOLERANCE = 1e-9

'''

FIXED_SUFFIX = '''

# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor."""
    centers, radii = construct_packing()
    print(f"Sum of radii: {np.sum(radii)}")
    return centers, radii


if __name__ == "__main__":
    run_packing()
'''

PARENT_BODY = "def construct_packing():\n    return ring_layout()"
CONTEXT_BODY = "def construct_packing():\n    return hexagonal_layout()"


def _block(body: str, comment: str = "#") -> str:
    return f"{comment} EVOLVE-BLOCK-START\n{body}\n{comment} EVOLVE-BLOCK-END"


def _solution(
    body: str, prefix: str = FIXED_PREFIX, suffix: str = FIXED_SUFFIX, comment: str = "#"
) -> str:
    return f"{prefix}{_block(body, comment)}{suffix}"


def _program(program_id: str, body: str, **kwargs) -> Program:
    return Program(
        id=program_id, solution=_solution(body, **kwargs), metrics={"combined_score": 0.5}
    )


def _user_message(current_program, context_programs, **config) -> str:
    builder = DefaultContextBuilder(Config.from_dict({"language": "python", **config}))
    context = {"other_context_programs": context_programs}
    return builder.build_prompt(current_program, context)["user"]


@pytest.mark.parametrize("diff_based_generation", [True, False], ids=["diff", "full-rewrite"])
@pytest.mark.parametrize("labelled", [True, False], ids=["labelled", "flat"])
def test_shared_fixed_code_is_shown_only_in_current_solution(diff_based_generation, labelled):
    current = _program("parent", PARENT_BODY)
    others = [
        _program("ctx1", CONTEXT_BODY),
        _program("ctx2", "def construct_packing():\n    return grid_layout()"),
    ]
    context_programs = {"Top Programs": others} if labelled else others

    user = _user_message(current, context_programs, diff_based_generation=diff_based_generation)

    # The current solution stays complete so SEARCH blocks and rewrites can use it.
    assert current.solution in user
    assert user.count("N_CIRCLES = 26") == 1
    assert user.count("def run_packing():") == 1
    placeholder = f"# {_OMITTED_FIXED_CODE}"
    assert f"```python\n{placeholder}\n{_block(CONTEXT_BODY)}\n{placeholder}\n```" in user
    assert "return grid_layout()" in user
    assert user.count(placeholder) == 4


def test_dict_wrapped_current_program_is_used_as_reference():
    current = _program("parent", PARENT_BODY)
    other = _program("ctx", CONTEXT_BODY)

    user = _user_message({"Parent from island 0": current}, [other])

    assert user.count("def run_packing():") == 1
    assert user.count(_OMITTED_FIXED_CODE) == 2


def test_fixed_code_that_differs_from_current_solution_is_kept():
    current = _program("parent", PARENT_BODY)
    edited_suffix = FIXED_SUFFIX.replace("Sum of radii", "Total radius")
    other = _program("ctx", CONTEXT_BODY, suffix=edited_suffix)

    user = _user_message(current, [other])

    assert f"{_block(CONTEXT_BODY)}{edited_suffix}" in user
    assert user.count("N_CIRCLES = 26") == 1
    assert user.count(_OMITTED_FIXED_CODE) == 1


def test_fixed_code_shorter_than_the_placeholder_is_kept():
    prefix = "import numpy as np\n\n"
    current = _program("parent", PARENT_BODY, prefix=prefix)
    other = _program("ctx", CONTEXT_BODY, prefix=prefix)

    user = _user_message(current, [other])

    assert f"```python\n{prefix}{_block(CONTEXT_BODY)}" in user
    assert user.count(_OMITTED_FIXED_CODE) == 1


def test_placeholder_follows_the_marker_comment_syntax():
    fixed = {
        "prefix": (
            "#include <cstdio>\n#include <vector>\n\n"
            "constexpr int kCircles = 26;\nconstexpr double kTolerance = 1e-9;\n\n"
        ),
        "suffix": (
            "\n\nint main() {\n  auto packing = construct_packing();\n"
            '  std::printf("%f\\n", score(packing));\n  return 0;\n}\n'
        ),
        "comment": "//",
    }
    current = _program("parent", "Packing construct_packing() { return ring(); }", **fixed)
    other = _program("ctx", "Packing construct_packing() { return hexagonal(); }", **fixed)

    user = _user_message(current, [other], language="cpp")

    assert user.count(f"// {_OMITTED_FIXED_CODE}") == 2
    assert user.count("int main() {") == 1


@pytest.mark.parametrize(
    "current_solution, context_solution",
    [
        pytest.param(
            FIXED_PREFIX + PARENT_BODY + FIXED_SUFFIX,
            FIXED_PREFIX + CONTEXT_BODY + FIXED_SUFFIX,
            id="no-markers",
        ),
        pytest.param(
            FIXED_PREFIX + PARENT_BODY + FIXED_SUFFIX,
            _solution(CONTEXT_BODY),
            id="current-without-markers",
        ),
        pytest.param(
            _solution(PARENT_BODY),
            FIXED_PREFIX + "# EVOLVE-BLOCK-START\n" + CONTEXT_BODY + FIXED_SUFFIX,
            id="unclosed-block",
        ),
        pytest.param(
            _solution(PARENT_BODY),
            FIXED_PREFIX
            + "# EVOLVE-BLOCK-END\n"
            + CONTEXT_BODY
            + "\n# EVOLVE-BLOCK-START"
            + FIXED_SUFFIX,
            id="reversed-markers",
        ),
        pytest.param(
            _solution(PARENT_BODY),
            _solution(CONTEXT_BODY) + _block("def helper():\n    pass"),
            id="extra-block",
        ),
    ],
)
def test_context_program_is_shown_in_full_when_blocks_do_not_line_up(
    current_solution, context_solution
):
    current = Program(id="parent", solution=current_solution)
    other = Program(id="ctx", solution=context_solution)

    user = _user_message(current, [other])

    assert context_solution in user
    assert _OMITTED_FIXED_CODE not in user


def test_context_programs_are_complete_without_a_current_program():
    other = _program("ctx", CONTEXT_BODY)

    user = _user_message(None, [other])

    assert other.solution in user
    assert _OMITTED_FIXED_CODE not in user
