#!/usr/bin/env bash
set -euo pipefail

PROGRAM="$1"
# MODE ($2) accepted but ignored — override this file to use train/test splits.

python /benchmark/wrapper.py "$PROGRAM"
