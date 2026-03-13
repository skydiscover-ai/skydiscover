#!/usr/bin/env bash
set -euo pipefail

PROGRAM="$1"
# MODE ($2) is accepted but ignored — pure optimization has no data split.

echo "[$(date '+%H:%M:%S')] eval start: $PROGRAM" >> /tmp/eval.log
python /benchmark/run.py "$PROGRAM"
