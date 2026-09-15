#!/usr/bin/env bash
# Run all reproduce scripts in parallel.
# Each category launches in the background; we wait for all to finish.
# Tip: set ITERATIONS=2 in each script for a quick smoke test.
set -euo pipefail

DIR="$(dirname "$0")"

bash "$DIR/math.sh"        &
bash "$DIR/adrs.sh"        &
bash "$DIR/ale_bench.sh"   &
bash "$DIR/frontier_cs.sh" &
bash "$DIR/gpu.sh"         &
bash "$DIR/arc.sh"         &
bash "$DIR/prompt_opt.sh"  &

wait
echo "All reproduce scripts finished."
