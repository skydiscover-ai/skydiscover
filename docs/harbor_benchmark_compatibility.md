# Harbor Benchmark Compatibility Report

Status of SkyDiscover's HarborEvaluator across Harbor registry benchmarks.

## Tested Benchmarks

| Benchmark | Tasks | Solution Path | Language | Docker Build | Evaluation | Notes |
|-----------|-------|--------------|----------|-------------|------------|-------|
| **algotune** | 154 | `/app/solver.py` | Python | OK | OK | Reference benchmark; optimization scoring |
| **evoeval** | 100 | `/app/solution.py` | Python | OK | OK | Code generation from HumanEval |
| **humanevalfix** | 164 | `/workspace/Python__27.py` (varies) | Python | OK | OK | Code repair; per-task solution filenames |
| **bigcodebench-hard-complete** | 145 | `/workspace/solution.py` | Python | Slow | OK | Heavy Dockerfile (R, gdal, build-essential) |
| **livecodebench** | 100 | `/app/solution.py` | Python | OK | OK | Competitive programming (stdin/stdout) |
| **codepde** | 5 | `/app/solver.py` | Python | OK | OK | PDE solving; downloads HF data at test time |
| **crustbench** | 100 | `/workspace/rbench_reference/src/interfaces/*.rs` | Rust | OK | OK | C-to-Rust transpilation; cargo test |
| **usaco** | 304 | `/app/solution.py` | Python | OK | OK | USACO competition; installs deps at test time |
| **hello-world** | 1 | `hello.txt` (not code) | Shell | OK | N/A | Creates a text file, not code injection |

## Key Issue: Solution Path Extraction

The primary compatibility issue was that `_extract_solution_path()` used regex on `instruction.md` to find the solution file path. This only worked for **2 out of 9** benchmarks (algotune, codepde) because most instruction files don't mention an absolute path explicitly.

### Root Cause

Each Harbor task includes `solution/solve.sh` — the authoritative reference solution script. This file almost always contains a `cat > /path/to/file` pattern that reveals exactly where the solution should be placed. The evaluator was not using this information.

### Solution Paths by Benchmark (from solve.sh)

| Benchmark | Pattern in solve.sh | Path |
|-----------|-------------------|------|
| algotune | `cat > /app/solver.py` | `/app/solver.py` |
| evoeval | `cat > /app/solution.py` | `/app/solution.py` |
| humanevalfix | `cat > /workspace/Python__27.py` | `/workspace/<task>.py` (varies per task) |
| bigcodebench | `cat > /workspace/solution.py` | `/workspace/solution.py` |
| livecodebench | `cat > /app/solution.py` | `/app/solution.py` |
| codepde | `cat > /app/solver.py` | `/app/solver.py` |
| crustbench | `cat > src/interfaces/base122.rs` | Relative path within cargo project |
| usaco | `cat > /app/solution.py` | `/app/solution.py` |
| hello-world | `echo ... > hello.txt` | Not standard code injection |

## Fix Applied

The solution path extraction now uses a three-tier strategy:
1. **Parse `solution/solve.sh`** — most reliable; extracts the target path from `cat >` or redirect patterns
2. **Parse `instruction.md`** — regex fallback for tasks without solve.sh
3. **Default to `/app/solution.py`** — the most common path across Harbor benchmarks

This fixes path extraction for **8 out of 8** code benchmarks tested (hello-world is a shell task, not a code optimization task). End-to-end verification with reference solutions confirmed `combined_score=1.0` for evoeval and humanevalfix.

## Other Observations

- **All benchmarks write rewards to `/logs/verifier/reward.txt`** — the existing reward-reading code works universally
- **Most rewards are binary (0 or 1)** — algotune is the exception with scaled scores
- **Docker build times vary wildly** — bigcodebench installs ~1.6GB of packages
- **Some benchmarks install deps at test time** (usaco installs uv/pytest via apt in test.sh) — this adds latency but works
- **test.sh always uses `bash` or `sh`** — the current `/bin/sh -c` invocation in the evaluator should use `bash` for compatibility with `set -euo pipefail`
