#!/usr/bin/env bash
# Reproduce math benchmarks (17 problems x 2 search methods).
# All benchmarks launch in parallel.
set -euo pipefail

# ── Settings ─────────────────────────────────────────────────────────────────
# Only two things to change:

MODEL="gpt-5"                        # main generation model
# MODEL="gemini/gemini-3.0-pro-preview"  # alternative
ITERATIONS=100

# -m sets all models (main + guide/paradigm) to the same MODEL.
# API keys: export OPENAI_API_KEY="sk-..." (and/or GEMINI_API_KEY for Gemini)

# ── Install ──────────────────────────────────────────────────────────────────

# Resolve the repo root by walking up to the directory holding pyproject.toml, so
# this script keeps working wherever it lives in the tree.
root="$(cd "$(dirname "$0")" && pwd)"
while [[ ! -f "$root/pyproject.toml" && "$root" != "/" ]]; do root="$(dirname "$root")"; done
cd "$root"
uv sync --extra math

# ── Helper ───────────────────────────────────────────────────────────────────

run() {
  local dir=$1 search=$2
  local init="$dir/initial_program.py"
  [[ -f "$dir/initial_program.cpp" ]] && init="$dir/initial_program.cpp"
  [[ -f "$dir/initial_prompt.txt" ]] && init="$dir/initial_prompt.txt"
  local cfg="$dir/config.yaml"
  [[ -f "$dir/config_${search}.yaml" ]] && cfg="$dir/config_${search}.yaml"
  echo "== $search: ${dir#benchmarks/} =="
  uv run skydiscover optimize "$init" "$dir/evaluator.py" \
    -c "$cfg" -s "$search" -m "$MODEL" -i "$ITERATIONS" \
    -o "outputs/reproduce/$search/${dir#benchmarks/}"
}

# ── AdaEvolve ────────────────────────────────────────────────────────────────

run benchmarks/math/circle_packing           adaevolve &
run benchmarks/math/circle_packing_rect      adaevolve &
run benchmarks/math/erdos_min_overlap        adaevolve &
run benchmarks/math/first_autocorr_ineq      adaevolve &
run benchmarks/math/second_autocorr_ineq     adaevolve &
run benchmarks/math/third_autocorr_ineq      adaevolve &
run benchmarks/math/uncertainty_ineq         adaevolve &
run benchmarks/math/hexagon_packing/11       adaevolve &
run benchmarks/math/hexagon_packing/12       adaevolve &
run benchmarks/math/heilbronn_convex/13      adaevolve &
run benchmarks/math/heilbronn_convex/14      adaevolve &
run benchmarks/math/heilbronn_triangle       adaevolve &
run benchmarks/math/minimizing_max_min_dist/2 adaevolve &
run benchmarks/math/minimizing_max_min_dist/3 adaevolve &
run benchmarks/math/matmul                   adaevolve &
run benchmarks/math/signal_processing        adaevolve &
run benchmarks/math/sums_diffs_finite_sets   adaevolve &

# ── EvoX ─────────────────────────────────────────────────────────────────────

run benchmarks/math/circle_packing           evox &
run benchmarks/math/circle_packing_rect      evox &
run benchmarks/math/erdos_min_overlap        evox &
run benchmarks/math/first_autocorr_ineq      evox &
run benchmarks/math/second_autocorr_ineq     evox &
run benchmarks/math/third_autocorr_ineq      evox &
run benchmarks/math/uncertainty_ineq         evox &
run benchmarks/math/hexagon_packing/11       evox &
run benchmarks/math/hexagon_packing/12       evox &
run benchmarks/math/heilbronn_convex/13      evox &
run benchmarks/math/heilbronn_convex/14      evox &
run benchmarks/math/heilbronn_triangle       evox &
run benchmarks/math/minimizing_max_min_dist/2 evox &
run benchmarks/math/minimizing_max_min_dist/3 evox &
run benchmarks/math/matmul                   evox &
run benchmarks/math/signal_processing        evox &
run benchmarks/math/sums_diffs_finite_sets   evox &

wait
echo "math.sh: all 34 runs finished."
