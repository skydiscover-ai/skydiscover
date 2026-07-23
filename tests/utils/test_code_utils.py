"""Tests for skydiscover.utils.code_utils — diff parsing, summary generation, and language support."""

import importlib.util
from pathlib import Path

import pytest

# Load code_utils directly from file to avoid pulling in the full skydiscover
# package (which requires openai, yaml, etc. not needed for these unit tests).
_code_utils_path = Path(__file__).resolve().parents[2] / "skydiscover" / "utils" / "code_utils.py"
_spec = importlib.util.spec_from_file_location("code_utils", _code_utils_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_extract_c_comment = _mod._extract_c_comment
_extract_def_info = _mod._extract_def_info
_extract_first_comment = _mod._extract_first_comment
apply_diff = _mod.apply_diff
extract_diffs = _mod.extract_diffs
format_diff_summary = _mod.format_diff_summary


# ---------------------------------------------------------------------------
# _extract_def_info — Python
# ---------------------------------------------------------------------------


class TestExtractDefInfoPython:
    def test_function_with_docstring(self):
        code = '''def solve(x, y):
    """Compute the sum of x and y."""
    return x + y
'''
        result = _extract_def_info(code)
        assert result == ("function", "solve", "Compute the sum of x and y.")

    def test_function_with_comment(self):
        code = """def optimize(params):
    # Gradient descent optimizer
    lr = 0.01
    return params - lr * grad
"""
        result = _extract_def_info(code)
        assert result == ("function", "optimize", "Gradient descent optimizer")

    def test_class_with_docstring(self):
        code = '''class MyModel:
    """A neural network model."""
    pass
'''
        result = _extract_def_info(code)
        assert result == ("class", "MyModel", "A neural network model.")

    def test_function_no_docstring_no_comment(self):
        code = """def bare_func(x):
    return x * 2
"""
        result = _extract_def_info(code)
        assert result == ("function", "bare_func", None)

    def test_star_unpack_not_treated_as_comment(self):
        code = """def k():
    *rest, last = [1, 2, 3]
    return last
"""
        result = _extract_def_info(code)
        assert result == ("function", "k", None)


# ---------------------------------------------------------------------------
# _extract_def_info — CUDA
# ---------------------------------------------------------------------------


class TestExtractDefInfoCUDA:
    def test_global_kernel(self):
        code = """__global__ void my_kernel(float* data, int n) {
    // Process one element per thread
    int tid = threadIdx.x;
}"""
        result = _extract_def_info(code)
        assert result is not None
        kind, name, docstring = result
        assert kind == "kernel"
        assert name == "my_kernel"
        assert docstring == "Process one element per thread"

    def test_launch_bounds_before_return_type(self):
        code = """__global__ __launch_bounds__(256, 4)
void fused_layer_norm(const half* x, half* y, int hidden, float eps) {
    // One block per row; blockDim=256
    const int row = blockIdx.x;
}"""
        result = _extract_def_info(code)
        assert result is not None
        kind, name, docstring = result
        assert kind == "kernel"
        assert name == "fused_layer_norm"
        assert "One block per row" in docstring

    def test_launch_bounds_after_return_type(self):
        code = """__global__ void __launch_bounds__(256, 4) fused_layer_norm(const half* x, half* y) {
    // Alternate placement of launch_bounds
    const int row = blockIdx.x;
}"""
        result = _extract_def_info(code)
        assert result is not None
        kind, name, docstring = result
        assert kind == "kernel"
        assert name == "fused_layer_norm"
        assert "Alternate placement" in docstring

    def test_device_function(self):
        code = """__device__ float warp_reduce_sum(float val) {
    // Butterfly warp reduction
    val += __shfl_down_sync(0xffffffff, val, 16);
    return val;
}"""
        result = _extract_def_info(code)
        assert result is not None
        kind, name, docstring = result
        assert kind == "device function"
        assert name == "warp_reduce_sum"
        assert docstring == "Butterfly warp reduction"

    def test_kernel_with_preceding_block_comment(self):
        code = """/* Fast parallel reduction for SM90 */
__global__ void reduce_kernel(float* data, int n) {
    int tid = threadIdx.x;
}"""
        result = _extract_def_info(code)
        assert result is not None
        kind, name, docstring = result
        assert kind == "kernel"
        assert name == "reduce_kernel"
        assert docstring == "Fast parallel reduction for SM90"

    def test_kernel_with_preceding_line_comments(self):
        code = """// H100-optimized warp shuffle reduction
// Handles hidden=128 case specifically
__global__ void norm_kernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}"""
        result = _extract_def_info(code)
        assert result is not None
        kind, name, docstring = result
        assert kind == "kernel"
        assert name == "norm_kernel"
        assert "H100-optimized warp shuffle reduction" in docstring
        assert "Handles hidden=128 case specifically" in docstring


# ---------------------------------------------------------------------------
# _extract_def_info — C/C++
# ---------------------------------------------------------------------------


class TestExtractDefInfoCCpp:
    def test_void_function(self):
        code = """void process_data(int* buf, size_t len) {
    /* Zero-initialize the buffer */
    memset(buf, 0, len * sizeof(int));
}"""
        result = _extract_def_info(code)
        assert result is not None
        kind, name, docstring = result
        assert kind == "function"
        assert name == "process_data"
        assert docstring == "Zero-initialize the buffer"

    def test_static_inline_function(self):
        code = """static inline float fast_rsqrt(float x) {
    // Quake III fast inverse square root
    return 1.0f / sqrtf(x);
}"""
        result = _extract_def_info(code)
        assert result is not None
        kind, name, docstring = result
        assert kind == "function"
        assert name == "fast_rsqrt"
        assert docstring == "Quake III fast inverse square root"

    def test_const_pointer_function(self):
        code = """const char** get_names() {
    // Return list of names
    return names;
}"""
        result = _extract_def_info(code)
        assert result is not None
        kind, name, docstring = result
        assert kind == "function"
        assert name == "get_names"
        assert docstring == "Return list of names"

    def test_struct(self):
        code = """struct SharedMemLayout {
    float warp_sums[8];
    float mean;
    float inv_std;
};"""
        result = _extract_def_info(code)
        assert result is not None
        kind, name, _ = result
        assert kind == "struct"
        assert name == "SharedMemLayout"


# ---------------------------------------------------------------------------
# _extract_def_info — no match
# ---------------------------------------------------------------------------


class TestExtractDefInfoNoMatch:
    def test_no_recognizable_pattern(self):
        assert _extract_def_info("hello world") is None
        assert _extract_def_info("") is None


# ---------------------------------------------------------------------------
# _extract_first_comment — C-style
# ---------------------------------------------------------------------------


class TestExtractFirstCommentCStyle:
    def test_double_slash_comments(self):
        code = """void func() {
    // First comment line
    // Second comment line
    int x = 0;
}"""
        result = _extract_first_comment(code, 0)
        assert result is not None
        assert "First comment line" in result
        assert "Second comment line" in result

    def test_block_comment_inline(self):
        code = """void func() {
    /* Initialize the accumulator */
    float acc = 0.0f;
}"""
        result = _extract_first_comment(code, 0)
        assert result == "Initialize the accumulator"

    def test_multiline_block_comment_continuation(self):
        code = """void func() {
    /*
     * Step 1: load data
     * Step 2: compute
     */
    int x = 0;
}"""
        result = _extract_first_comment(code, 0)
        assert result is not None
        assert "Step 1: load data" in result


# ---------------------------------------------------------------------------
# _extract_c_comment
# ---------------------------------------------------------------------------


class TestExtractCComment:
    def test_block_comment_before_function(self):
        code = """/* Compute mean and variance in one pass */
void compute_stats(float* data, int n) {
    float sum = 0;
}"""
        func_start = code.index("void")
        result = _extract_c_comment(code, func_start)
        assert result == "Compute mean and variance in one pass"

    def test_line_comments_before_function(self):
        code = """// Fast reciprocal square root
// Optimized for H100 SM90
__global__ void kernel(float* x) {
    int tid = threadIdx.x;
}"""
        func_start = code.index("__global__")
        result = _extract_c_comment(code, func_start)
        assert "Fast reciprocal square root" in result
        assert "Optimized for H100 SM90" in result

    def test_no_comment_falls_back_to_body(self):
        code = """__global__ void kernel(float* x) {
    // Body comment here
    int tid = threadIdx.x;
}"""
        func_start = code.index("__global__")
        result = _extract_c_comment(code, func_start)
        assert result == "Body comment here"

    def test_no_comment_anywhere(self):
        code = """__global__ void kernel(float* x) {
    int tid = threadIdx.x;
}"""
        func_start = code.index("__global__")
        result = _extract_c_comment(code, func_start)
        assert result is None


# ---------------------------------------------------------------------------
# format_diff_summary — CUDA integration
# ---------------------------------------------------------------------------


class TestFormatDiffSummaryCUDA:
    def test_modified_kernel_same_comment(self):
        old = """__global__ void my_kernel(float* x, int n) {
    // Process elements
    int tid = threadIdx.x;
    x[tid] = x[tid] * 2.0f;
}"""
        new = """__global__ void my_kernel(float* x, int n) {
    // Process elements
    int tid = threadIdx.x;
    x[tid] = x[tid] * 3.0f;
    __syncthreads();
}"""
        summary = format_diff_summary([(old, new)])
        assert "my_kernel" in summary
        assert "kernel" in summary.lower()
        # Same docstring -> line count format
        assert "→" in summary or "->" in summary or "lines" in summary

    def test_modified_kernel_different_comment(self):
        old = """__global__ void my_kernel(float* x, int n) {
    // Old approach: scalar loads
    int tid = threadIdx.x;
}"""
        new = """__global__ void my_kernel(float* x, int n) {
    // New approach: vectorized loads
    int tid = threadIdx.x;
}"""
        summary = format_diff_summary([(old, new)])
        assert "my_kernel" in summary
        assert "New approach: vectorized loads" in summary

    def test_renamed_kernel(self):
        old = """__global__ void old_kernel(float* x) {
    int tid = threadIdx.x;
}"""
        new = """__global__ void new_kernel(float* x) {
    int tid = threadIdx.x;
}"""
        summary = format_diff_summary([(old, new)])
        assert "old_kernel" in summary
        assert "new_kernel" in summary
        assert "Renamed" in summary


# ---------------------------------------------------------------------------
# format_diff_summary — Python (preserve existing behavior)
# ---------------------------------------------------------------------------


class TestFormatDiffSummaryPython:
    def test_modified_function_same_docstring(self):
        old = '''def solve(x):
    """Compute solution."""
    return x + 1
'''
        new = '''def solve(x):
    """Compute solution."""
    return x + 2
'''
        summary = format_diff_summary([(old, new)])
        assert "solve" in summary
        assert "function" in summary.lower()

    def test_renamed_function(self):
        old = '''def old_name(x):
    """Do something."""
    return x
'''
        new = '''def new_name(x):
    """Do something."""
    return x
'''
        summary = format_diff_summary([(old, new)])
        assert "old_name" in summary
        assert "new_name" in summary
        assert "Renamed" in summary

    def test_fallback_no_function(self):
        old = "x = 1\ny = 2\nz = 3"
        new = "x = 10\ny = 20\nz = 30"
        summary = format_diff_summary([(old, new)])
        assert "Change 1" in summary


# ---------------------------------------------------------------------------
# extract_diffs and apply_diff (existing functionality)
# ---------------------------------------------------------------------------


class TestExtractDiffs:
    def test_single_block(self):
        text = """Some explanation.

<<<<<<< SEARCH
old line 1
old line 2
=======
new line 1
new line 2
new line 3
>>>>>>> REPLACE

Done."""
        blocks = extract_diffs(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "old line 1\nold line 2"
        assert blocks[0][1] == "new line 1\nnew line 2\nnew line 3"

    def test_multiple_blocks(self):
        text = """<<<<<<< SEARCH
a
=======
b
>>>>>>> REPLACE

<<<<<<< SEARCH
c
=======
d
>>>>>>> REPLACE"""
        blocks = extract_diffs(text)
        assert len(blocks) == 2
        assert blocks[0] == ("a", "b")
        assert blocks[1] == ("c", "d")

    def test_no_blocks(self):
        assert extract_diffs("no diffs here") == []


class TestApplyDiff:
    def test_simple_replacement(self):
        original = "line 1\nline 2\nline 3"
        diff = """<<<<<<< SEARCH
line 2
=======
replaced line 2
>>>>>>> REPLACE"""
        result = apply_diff(original, diff)
        assert result == "line 1\nreplaced line 2\nline 3"

    def test_no_match_returns_original(self):
        original = "line 1\nline 2"
        diff = """<<<<<<< SEARCH
nonexistent
=======
replacement
>>>>>>> REPLACE"""
        result = apply_diff(original, diff)
        assert result == original
