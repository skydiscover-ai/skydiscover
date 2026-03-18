#!/usr/bin/env bash
# Download dataset and config files for the Cloudcast benchmark.
#
# Required files:
#   profiles/cost.csv         - Cloud egress cost per region pair ($/GB)
#   profiles/throughput.csv   - Measured throughput per region pair (bps)
#   examples/config/*.json    - Network configurations for evaluation
#
# Usage:
#   cd benchmarks/ADRS/cloudcast
#   bash download_dataset.sh

set -euo pipefail
cd "$(dirname "$0")"

BASE_URL="https://huggingface.co/datasets/f20180301/adrs-data/resolve/main/cloudcast"

echo "Downloading Cloudcast benchmark data..."

# Download profiles
mkdir -p profiles
echo "  Downloading profiles/cost.csv..."
wget -q -O profiles/cost.csv "${BASE_URL}/profiles/cost.csv"
echo "  Downloading profiles/throughput.csv..."
wget -q -O profiles/throughput.csv "${BASE_URL}/profiles/throughput.csv"

# Download example configs
mkdir -p examples/config
for config in intra_aws.json intra_azure.json intra_gcp.json inter_agz.json inter_gaz2.json; do
    echo "  Downloading examples/config/${config}..."
    wget -q -O "examples/config/${config}" "${BASE_URL}/examples/config/${config}"
done

echo ""
echo "Done. Downloaded files:"
ls -lh profiles/*.csv
ls -lh examples/config/*.json
