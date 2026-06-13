#!/bin/bash
# run_feedback_ablation.sh -- Run matched pairs at both feedback levels for
# controlled ablation comparison.
#
# Usage:
#   bash agent-pipeline/scripts/run_feedback_ablation.sh \
#     --backend codex --mode ltm --distribution zipf --setup rmw --iterations 20
#
# This runs TWO sequential experiments with identical parameters except
# --feedback-level: one "minimal" (no gradient) and one "rich" (full gradient).
# Results go to separate run directories for comparison.
#
# All flags are forwarded to run.sh. --feedback-level is overridden.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="$(dirname "$SCRIPT_DIR")/run.sh"

if [[ ! -f "$RUN_SH" ]]; then
    echo "ERROR: cannot find run.sh at $RUN_SH"
    exit 1
fi

# Strip any --feedback-level from forwarded args
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --feedback-level) shift 2 ;;  # skip it + its value
        *) ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
    echo "Usage: $0 [run.sh flags except --feedback-level]"
    echo ""
    echo "Example:"
    echo "  $0 --backend codex --mode ltm --distribution zipf --setup rmw --iterations 20"
    exit 1
fi

echo "================================================================"
echo "  FEEDBACK ABLATION: minimal (no gradient)"
echo "================================================================"
bash "$RUN_SH" "${ARGS[@]}" --feedback-level minimal

echo ""
echo "================================================================"
echo "  FEEDBACK ABLATION: rich (full gradient)"
echo "================================================================"
bash "$RUN_SH" "${ARGS[@]}" --feedback-level rich

echo ""
echo "================================================================"
echo "  ABLATION COMPLETE"
echo "  Compare the two run directories in runs/ to measure"
echo "  convergence speed and final throughput differences."
echo "================================================================"
