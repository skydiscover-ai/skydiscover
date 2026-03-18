#!/usr/bin/env bash
set -euo pipefail

PROGRAM="$1"
MODE="${2:-train}"

# Start the judge server if not already running
if ! curl -s http://localhost:8081/health > /dev/null 2>&1; then
    echo "Starting Frontier-CS judge server..." >&2
    cd /benchmark/Frontier-CS/algorithmic
    docker compose up -d 2>&1 >&2
    # Wait for judge to be ready
    for i in $(seq 1 30); do
        if curl -s http://localhost:8081/health > /dev/null 2>&1; then
            echo "Judge server ready" >&2
            break
        fi
        sleep 1
    done
    cd /benchmark
fi

python /benchmark/evaluator.py "$PROGRAM"
