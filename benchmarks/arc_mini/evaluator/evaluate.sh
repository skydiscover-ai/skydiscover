#!/usr/bin/env bash
set -euo pipefail

PROGRAM="$1"     # absolute path to candidate program inside container
MODE="$2"        # train | test

case "$MODE" in
    train) DATA_FILE="/benchmark/data/train.json" ;;
    test)  DATA_FILE="/benchmark/data/test.json" ;;
    *)
        echo '{"status":"error","combined_score":0,"metrics":{},"artifacts":{"error":"unknown mode: '"$MODE"'"}}'
        exit 1
        ;;
esac

python /benchmark/run.py "$PROGRAM" "$DATA_FILE"
