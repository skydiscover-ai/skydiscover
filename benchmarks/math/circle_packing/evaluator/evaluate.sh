#!/usr/bin/env bash
set -euo pipefail

# MODE ($1) is accepted but ignored — pure optimization has no data split.
PROGRAM="$2"

echo "[$(date '+%H:%M:%S')] eval start: $PROGRAM" >> /tmp/eval.log
python /benchmark/run.py "$PROGRAM"
