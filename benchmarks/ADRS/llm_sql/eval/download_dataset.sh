#!/usr/bin/env bash
# Download CSV datasets for the LLM-SQL benchmark.
#
# Required files (placed in datasets/):
#   movies.csv    - Rotten Tomatoes movie reviews (~9 MB)
#   beer.csv      - Beer review dataset (~2.5 MB)
#   BIRD.csv      - BIRD text-to-SQL dataset (~34 MB)
#   PDMX.csv      - PDMX metadata dataset (~7.4 MB)
#   products.csv  - Amazon product catalog (~16 MB)
#
# Usage:
#   cd benchmarks/ADRS/llm_sql
#   bash download_dataset.sh

set -euo pipefail
cd "$(dirname "$0")"

BASE_URL="https://huggingface.co/datasets/f20180301/adrs-data/resolve/main/llm_sql"

echo "Downloading LLM-SQL benchmark datasets..."

mkdir -p datasets
for dataset in movies.csv beer.csv BIRD.csv PDMX.csv products.csv; do
    echo "  Downloading datasets/${dataset}..."
    wget -q --show-progress -O "datasets/${dataset}" "${BASE_URL}/datasets/${dataset}"
done

echo ""
echo "Done. Downloaded files:"
ls -lh datasets/*.csv
