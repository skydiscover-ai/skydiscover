"""
Tests for compliance_checker.py
"""
import pytest

from skydiscover.extras.evolve_analyzer.quantitative.compliance_checker import (
    analyze_compliance,
    check_evolve_block,
    check_format_valid,
    check_signature_preserved,
)


# ---------------------------------------------------------------------------
# check_evolve_block
# ---------------------------------------------------------------------------

class TestCheckEvolveBlock:
    PARENT_WITH_BLOCK = """\
def setup():
    pass
# EVOLVE-BLOCK-START
def evolve_me():
    return 1
# EVOLVE-BLOCK-END
def teardown():
    pass
"""

    def test_no_markers_always_true(self):
        parent = "def foo(): pass\n"
        child  = "def foo(): return 42\n"
        assert check_evolve_block(parent, child) is True

    def test_change_inside_block_is_valid(self):
        child = """\
def setup():
    pass
# EVOLVE-BLOCK-START
def evolve_me():
    return 99
# EVOLVE-BLOCK-END
def teardown():
    pass
"""
        assert check_evolve_block(self.PARENT_WITH_BLOCK, child) is True

    def test_change_outside_block_is_invalid(self):
        child = """\
def setup():
    x = 1  # added outside block
# EVOLVE-BLOCK-START
def evolve_me():
    return 1
# EVOLVE-BLOCK-END
def teardown():
    pass
"""
        assert check_evolve_block(self.PARENT_WITH_BLOCK, child) is False

    def test_identical_code_is_valid(self):
        assert check_evolve_block(self.PARENT_WITH_BLOCK, self.PARENT_WITH_BLOCK) is True

    def test_change_in_teardown_is_invalid(self):
        child = """\
def setup():
    pass
# EVOLVE-BLOCK-START
def evolve_me():
    return 1
# EVOLVE-BLOCK-END
def teardown():
    return "modified"
"""
        assert check_evolve_block(self.PARENT_WITH_BLOCK, child) is False

    def test_empty_parent_no_markers(self):
        assert check_evolve_block("", "def new(): pass\n") is True

    def test_only_start_marker_no_end_treated_as_no_constraint(self):
        # Only EVOLVE-BLOCK-START with no end → _find_block_line_range returns None
        parent = "# EVOLVE-BLOCK-START\ndef foo(): pass\n"
        child  = "def bar(): return 1\n"
        assert check_evolve_block(parent, child) is True

    def test_multiple_changes_all_inside_block(self):
        parent = """\
# EVOLVE-BLOCK-START
x = 1
y = 2
# EVOLVE-BLOCK-END
"""
        child = """\
# EVOLVE-BLOCK-START
x = 10
y = 20
# EVOLVE-BLOCK-END
"""
        assert check_evolve_block(parent, child) is True


# ---------------------------------------------------------------------------
# check_format_valid
# ---------------------------------------------------------------------------

class TestCheckFormatValid:
    # --- diff mutation type ---

    def test_diff_starting_with_three_dashes_is_valid(self):
        diff = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n"
        assert check_format_valid(diff, None, "diff") is True

    def test_diff_starting_with_at_signs_is_valid(self):
        diff = "@@ -1,3 +1,3 @@\n context\n"
        assert check_format_valid(diff, None, "diff") is True

    def test_diff_starting_with_diff_keyword_is_valid(self):
        diff = "diff --git a/foo b/foo\n--- a/foo\n+++ b/foo\n"
        assert check_format_valid(diff, None, "diff") is True

    def test_empty_diff_is_invalid(self):
        assert check_format_valid("", None, "diff") is False
        assert check_format_valid("   \n  ", None, "diff") is False

    def test_diff_without_markers_is_invalid(self):
        diff = "some random text\nno diff markers here\n"
        assert check_format_valid(diff, None, "diff") is False

    def test_diff_with_markers_in_later_line_is_valid(self):
        # Looser check: markers anywhere in the diff
        diff = "header info\n--- a/file.py\n+++ b/file.py\n"
        assert check_format_valid(diff, None, "diff") is True

    # --- non-diff mutation types ---

    def test_multiline_child_code_is_valid(self):
        code = "def foo():\n    return 1\n"
        assert check_format_valid(None, code, "replace") is True

    def test_short_single_line_child_code_is_invalid(self):
        # <= 10 characters, no newline
        assert check_format_valid(None, "x = 1", "replace") is False

    def test_long_single_line_child_code_is_valid(self):
        # > 10 characters, no newline
        assert check_format_valid(None, "x = 1234567890123", "replace") is True

    def test_empty_child_code_is_invalid(self):
        assert check_format_valid(None, "", "replace") is False
        assert check_format_valid(None, "   ", "replace") is False

    def test_none_diff_and_none_child_code_is_invalid(self):
        assert check_format_valid(None, None, "diff") is False
        assert check_format_valid(None, None, "replace") is False

    def test_diff_argument_ignored_for_non_diff_mutation(self):
        # mutation_type != "diff" → child_code is checked, diff is ignored
        diff = "--- a\n+++ b\n"
        code = "def foo():\n    return 1\n"
        assert check_format_valid(diff, code, "replace") is True


# ---------------------------------------------------------------------------
# check_signature_preserved
# ---------------------------------------------------------------------------

class TestCheckSignaturePreserved:
    PARENT_PY = """\
def add(a, b):
    return a + b

def subtract(x, y):
    return x - y
"""

    def test_same_code_returns_true(self):
        assert check_signature_preserved(self.PARENT_PY, self.PARENT_PY) is True

    def test_implementation_change_preserves_signatures(self):
        child = """\
def add(a, b):
    return b + a  # swapped

def subtract(x, y):
    return x - y
"""
        assert check_signature_preserved(self.PARENT_PY, child) is True

    def test_removed_function_returns_false(self):
        child = "def add(a, b):\n    return a + b\n"
        assert check_signature_preserved(self.PARENT_PY, child) is False

    def test_renamed_parameter_returns_false(self):
        child = """\
def add(p, q):
    return p + q

def subtract(x, y):
    return x - y
"""
        assert check_signature_preserved(self.PARENT_PY, child) is False

    def test_added_parameter_returns_false(self):
        child = """\
def add(a, b, c=0):
    return a + b + c

def subtract(x, y):
    return x - y
"""
        assert check_signature_preserved(self.PARENT_PY, child) is False

    def test_empty_parent_returns_true(self):
        assert check_signature_preserved("", "def new(): pass\n") is True

    def test_unparseable_parent_returns_true(self):
        assert check_signature_preserved("def (:", "def foo(): pass\n") is True

    def test_non_python_language_always_true(self):
        parent = "fn add(a int, b int) int { return a + b }"
        child  = "fn add(x int) int { return x }"
        assert check_signature_preserved(parent, child, language="go") is True

    def test_class_signature_preserved(self):
        parent = "class Foo:\n    def __init__(self, x, y):\n        pass\n"
        child  = "class Foo:\n    def __init__(self, x, y):\n        self.x = x\n"
        assert check_signature_preserved(parent, child) is True

    def test_class_constructor_signature_changed_returns_false(self):
        parent = "class Foo:\n    def __init__(self, x, y):\n        pass\n"
        child  = "class Foo:\n    def __init__(self, x):\n        pass\n"
        assert check_signature_preserved(parent, child) is False

    def test_class_without_init_against_same_returns_true(self):
        parent = "class Bar:\n    pass\n"
        child  = "class Bar:\n    def method(self):\n        return 1\n"
        assert check_signature_preserved(parent, child) is True

    def test_vararg_and_kwargs_considered(self):
        parent = "def fn(*args, **kwargs): pass\n"
        child_same  = "def fn(*args, **kwargs): return 1\n"
        child_changed = "def fn(*args): return 1\n"
        assert check_signature_preserved(parent, child_same) is True
        assert check_signature_preserved(parent, child_changed) is False

    def test_annotations_ignored(self):
        # Annotations differ but parameter names match → should still be True
        parent = "def fn(a: int, b: str) -> None: pass\n"
        child  = "def fn(a, b): pass\n"
        assert check_signature_preserved(parent, child) is True


# ---------------------------------------------------------------------------
# analyze_compliance (integration)
# ---------------------------------------------------------------------------

class TestAnalyzeCompliance:
    PARENT_CODE = """\
# EVOLVE-BLOCK-START
def target(x):
    return x
# EVOLVE-BLOCK-END
"""
    CHILD_CODE_INSIDE = """\
# EVOLVE-BLOCK-START
def target(x):
    return x * 2
# EVOLVE-BLOCK-END
"""
    CHILD_CODE_OUTSIDE = """\
extra = 1
# EVOLVE-BLOCK-START
def target(x):
    return x
# EVOLVE-BLOCK-END
"""

    def _make_rec(self, parent_code=None, child_code=None, diff=None, mutation_type="replace"):
        r = {"iteration": 1, "mutation_type": mutation_type}
        if parent_code is not None:
            r["parent_code"] = parent_code
        if child_code is not None:
            r["child_code"] = child_code
        if diff is not None:
            r["diff"] = diff
        return r

    def test_all_checks_disabled_adds_no_fields(self):
        rec = self._make_rec(self.PARENT_CODE, self.CHILD_CODE_INSIDE)
        analyze_compliance([rec], {})
        assert "evolved_block_only" not in rec
        assert "format_valid_checked" not in rec
        assert "signature_preserved" not in rec

    def test_check_evolve_block_enabled(self):
        rec_ok = self._make_rec(self.PARENT_CODE, self.CHILD_CODE_INSIDE)
        rec_bad = self._make_rec(self.PARENT_CODE, self.CHILD_CODE_OUTSIDE)
        analyze_compliance([rec_ok, rec_bad], {"check_evolve_block": True})
        assert rec_ok["evolved_block_only"] is True
        assert rec_bad["evolved_block_only"] is False

    def test_check_format_enabled(self):
        rec = self._make_rec(child_code="def foo():\n    return 1\n")
        analyze_compliance([rec], {"check_format": True})
        assert "format_valid_checked" in rec
        assert rec["format_valid_checked"] is True

    def test_check_signature_enabled(self):
        rec = self._make_rec(self.PARENT_CODE, self.CHILD_CODE_INSIDE)
        analyze_compliance([rec], {"check_signature": True})
        assert rec["signature_preserved"] is True

    def test_check_evolve_block_skipped_when_code_missing(self):
        rec = {"iteration": 1, "mutation_type": "replace"}
        analyze_compliance([rec], {"check_evolve_block": True})
        assert "evolved_block_only" not in rec

    def test_modifies_records_in_place_and_returns_same_list(self):
        records = [self._make_rec(self.PARENT_CODE, self.CHILD_CODE_INSIDE)]
        returned = analyze_compliance(records, {"check_evolve_block": True})
        assert returned is records
        assert "evolved_block_only" in records[0]

    def test_empty_records(self):
        result = analyze_compliance([], {"check_evolve_block": True, "check_format": True, "check_signature": True})
        assert result == []

    def test_all_checks_enabled_together(self):
        rec = self._make_rec(
            parent_code=self.PARENT_CODE,
            child_code=self.CHILD_CODE_INSIDE,
            diff="--- a\n+++ b\n@@ -1 +1 @@\n",
            mutation_type="diff",
        )
        analyze_compliance([rec], {"check_evolve_block": True, "check_format": True, "check_signature": True})
        assert "evolved_block_only" in rec
        assert "format_valid_checked" in rec
        assert "signature_preserved" in rec
