#!/usr/bin/env bash
# Container entrypoint: score a candidate kvstore_impl.cc.
# Prints one JSON object on stdout (combined_score / metrics / artifacts).
set -euo pipefail

PROGRAM="$1" # path to the candidate kvstore_impl.cc
python3 /benchmark/evaluator.py "$PROGRAM"
