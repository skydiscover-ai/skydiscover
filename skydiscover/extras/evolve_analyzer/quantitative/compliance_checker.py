"""
compliance_checker.py
---------------------
Deterministic compliance checks for evolutionary code mutations.
Zero LLM cost — uses only Python stdlib (difflib, ast).

Three core checks:
  1. EVOLVE-BLOCK containment   – all diffs stay inside the marked region.
  2. Format validity            – diff / child_code is parseable / non-trivial.
  3. Signature preservation     – public function / class signatures unchanged.
"""
from __future__ import annotations

import ast
import difflib
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BLOCK_START_MARKER = "# EVOLVE-BLOCK-START"
_BLOCK_END_MARKER = "# EVOLVE-BLOCK-END"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_block_line_range(
    lines: List[str],
) -> Optional[Tuple[int, int]]:
    """
    Return the (start_line, end_line) indices (0-based, inclusive) of the
    EVOLVE-BLOCK in *lines*, or ``None`` if no markers are found.

    The start line is the line containing EVOLVE-BLOCK-START itself; the end
    line is the line containing EVOLVE-BLOCK-END.
    """
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None

    for idx, line in enumerate(lines):
        stripped = line.rstrip()
        if start_idx is None and stripped == _BLOCK_START_MARKER:
            start_idx = idx
        elif start_idx is not None and stripped == _BLOCK_END_MARKER:
            end_idx = idx
            break  # first complete block

    if start_idx is None or end_idx is None:
        return None
    return (start_idx, end_idx)


def _changed_line_numbers_in_parent(
    parent_lines: List[str],
    child_lines: List[str],
) -> List[int]:
    """
    Return the 0-based line indices in *parent_lines* that were removed or
    modified (i.e. appear in a ``-`` hunk of a unified diff).

    We use this to check that only lines inside the EVOLVE-BLOCK were touched.
    """
    changed: List[int] = []
    parent_idx = 0

    for group in difflib.SequenceMatcher(
        None, parent_lines, child_lines, autojunk=False
    ).get_grouped_opcodes(n=0):
        for tag, i1, i2, _j1, _j2 in group:
            if tag in ("replace", "delete"):
                changed.extend(range(i1, i2))
            # "insert" has no parent-side line numbers

    return changed


def _extract_python_signatures(
    source: str,
) -> Dict[str, ast.arguments]:
    """
    Parse *source* and return ``{name: arguments_node}`` for every top-level
    (module-level) function and class definition.

    Class signatures are keyed by their class name with an ``arguments`` node
    derived from ``__init__`` if present, otherwise a synthetic empty one.

    Returns an empty dict if *source* cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    sigs: Dict[str, ast.arguments] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sigs[node.name] = node.args
        elif isinstance(node, ast.ClassDef):
            # Look for __init__ to capture constructor signature
            init_args: Optional[ast.arguments] = None
            for child in ast.iter_child_nodes(node):
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == "__init__"
                ):
                    init_args = child.args
                    break
            if init_args is None:
                # Synthetic: no arguments beyond self
                init_args = ast.arguments(
                    posonlyargs=[],
                    args=[],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                )
            sigs[node.name] = init_args

    return sigs


def _args_equal(a: ast.arguments, b: ast.arguments) -> bool:
    """
    Return True when two ``ast.arguments`` nodes represent the same parameter
    list (names and ordering only; default values and annotations are ignored).
    """
    def _names(args: ast.arguments) -> List[str]:
        names: List[str] = []
        names.extend(arg.arg for arg in getattr(args, "posonlyargs", []))
        names.extend(arg.arg for arg in args.args)
        if args.vararg:
            names.append(f"*{args.vararg.arg}")
        names.extend(arg.arg for arg in args.kwonlyargs)
        if args.kwarg:
            names.append(f"**{args.kwarg.arg}")
        return names

    return _names(a) == _names(b)


# ---------------------------------------------------------------------------
# check_evolve_block
# ---------------------------------------------------------------------------

def check_evolve_block(parent_code: str, child_code: str) -> bool:
    """
    Verify that every line changed between *parent_code* and *child_code*
    falls within the ``# EVOLVE-BLOCK-START`` / ``# EVOLVE-BLOCK-END`` region
    defined in *parent_code*.

    Parameters
    ----------
    parent_code:
        Source code of the parent (before mutation).
    child_code:
        Source code of the child (after mutation).

    Returns
    -------
    bool
        ``True`` if no EVOLVE-BLOCK markers exist in *parent_code* (no
        constraint) **or** if all changed lines are within the block.
        ``False`` if any changed line is outside the block boundaries.
    """
    parent_lines = parent_code.splitlines(keepends=True)
    parent_block = _find_block_line_range(parent_lines)

    if parent_block is None:
        # No markers → no constraint
        return True

    parent_start, parent_end = parent_block
    child_lines = child_code.splitlines(keepends=True)
    child_block = _find_block_line_range(child_lines)

    for group in difflib.SequenceMatcher(
        None, parent_lines, child_lines, autojunk=False
    ).get_grouped_opcodes(n=0):
        for tag, i1, i2, j1, j2 in group:
            if tag in ("replace", "delete"):
                for idx in range(i1, i2):
                    if not (parent_start <= idx <= parent_end):
                        return False
            if tag == "insert":
                if child_block is None:
                    return False
                child_start, child_end = child_block
                for idx in range(j1, j2):
                    if not (child_start <= idx <= child_end):
                        return False

    return True


# ---------------------------------------------------------------------------
# check_format_valid
# ---------------------------------------------------------------------------

def check_format_valid(
    diff: Optional[str],
    child_code: Optional[str],
    mutation_type: str,
) -> bool:
    """
    Lightweight format validity check that avoids LLM calls.

    Parameters
    ----------
    diff:
        The diff string produced by the mutation (may be ``None``).
    child_code:
        The full mutated source code (may be ``None``).
    mutation_type:
        The type of mutation; when ``"diff"``, the *diff* string is validated
        as a unified diff.

    Returns
    -------
    bool
        ``True`` if the output appears parseable / non-trivial, ``False``
        otherwise.
    """
    if mutation_type == "diff" and diff is not None:
        # A valid unified diff starts with "---", "@@", or "diff " on any line
        stripped = diff.strip()
        if not stripped:
            return False
        first_line = stripped.splitlines()[0] if stripped else ""
        if first_line.startswith("---") or first_line.startswith("@@") or first_line.startswith("diff "):
            return True
        # Looser: check whether any line begins with "@@" or "---"
        for line in stripped.splitlines():
            if line.startswith("@@") or line.startswith("--- ") or line.startswith("+++ "):
                return True
        return False

    if child_code is not None:
        stripped_child = child_code.strip()
        if not stripped_child:
            return False
        # Non-trivial: either multi-line or more than 10 characters
        return "\n" in stripped_child or len(stripped_child) > 10

    # Neither diff nor child_code available
    return False


# ---------------------------------------------------------------------------
# check_signature_preserved
# ---------------------------------------------------------------------------

def check_signature_preserved(
    parent_code: str,
    child_code: str,
    language: str = "python",
) -> bool:
    """
    Verify that all top-level function / class signatures present in
    *parent_code* still exist unchanged in *child_code*.

    Parameters
    ----------
    parent_code:
        Source of the parent.
    child_code:
        Source of the child.
    language:
        Programming language.  Only ``"python"`` is fully implemented; all
        other languages return ``True`` (assumed preserved).

    Returns
    -------
    bool
        ``True`` if every signature present in the parent also appears in the
        child with the same name and argument list.
    """
    if language.lower() != "python":
        return True

    parent_sigs = _extract_python_signatures(parent_code)
    if not parent_sigs:
        # Either empty parent or unparseable — nothing to enforce
        return True

    child_sigs = _extract_python_signatures(child_code)

    for name, parent_args in parent_sigs.items():
        if name not in child_sigs:
            return False
        if not _args_equal(parent_args, child_sigs[name]):
            return False

    return True


# ---------------------------------------------------------------------------
# analyze_compliance
# ---------------------------------------------------------------------------

def analyze_compliance(
    records: List[dict],
    config: dict,
) -> List[dict]:
    """
    Tag each record with compliance check results where data are available.

    Modifies records **in-place** and returns the list.

    Fields added (when the corresponding config key is ``True`` and the
    required source fields are present):

    ``evolved_block_only``
        Whether all mutations stayed inside the EVOLVE-BLOCK region.
    ``format_valid_checked``
        Deterministic format validity (distinct from the raw ``format_valid``
        field supplied by the evaluator).
    ``signature_preserved``
        Whether all parent-side signatures are intact in the child.

    Parameters
    ----------
    records:
        List of raw record dicts.
    config:
        Dict with boolean keys ``"check_evolve_block"``, ``"check_format"``,
        and ``"check_signature"``.  A missing key is treated as ``False``.

    Returns
    -------
    List[dict]
        The same list with compliance tags added to each element.
    """
    do_block = bool(config.get("check_evolve_block", False))
    do_format = bool(config.get("check_format", False))
    do_sig = bool(config.get("check_signature", False))

    for rec in records:
        parent_code: Optional[str] = rec.get("parent_code")
        child_code: Optional[str] = rec.get("child_code")
        diff: Optional[str] = rec.get("diff")
        mutation_type: str = rec.get("mutation_type") or ""
        language: str = "python"  # default; could be extended via config

        if do_block and parent_code is not None and child_code is not None:
            rec["evolved_block_only"] = check_evolve_block(parent_code, child_code)

        if do_format:
            rec["format_valid_checked"] = check_format_valid(diff, child_code, mutation_type)

        if do_sig and parent_code is not None and child_code is not None:
            rec["signature_preserved"] = check_signature_preserved(
                parent_code, child_code, language
            )

    return records
