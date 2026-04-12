#!/usr/bin/env bash
set -euo pipefail

PROGRAM="$1"

python /benchmark/evaluator.py "$PROGRAM"
