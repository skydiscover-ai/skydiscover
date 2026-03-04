#!/usr/bin/env bash
set -euo pipefail

MODE="$1"        # train | val | test
PROGRAM="$2"     # absolute path to candidate program inside container

case "$MODE" in
    train) DATA_FILE="/benchmark/data/train.json" ;;
    val)   DATA_FILE="/benchmark/data/val.json" ;;
    test)  DATA_FILE="/benchmark/data/test.json" ;;
    *)
        echo '{"status":"error","combined_score":0,"metrics":{},"artifacts":{"error":"unknown mode"}}'
        exit 1
        ;;
esac

python /benchmark/run.py "$PROGRAM" "$DATA_FILE"
