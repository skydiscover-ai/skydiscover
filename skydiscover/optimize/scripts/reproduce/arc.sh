#!/usr/bin/env bash
# Reproduce ARC benchmark (1 problem x 2 search methods).
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
uv sync

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

run benchmarks/arc_benchmark adaevolve &

# ── EvoX ─────────────────────────────────────────────────────────────────────

run benchmarks/arc_benchmark evox &

wait
echo "arc.sh: all 2 runs finished."
