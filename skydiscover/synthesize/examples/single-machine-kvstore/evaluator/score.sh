#!/usr/bin/env bash
# Score one implementation: build it with the harness, run the throughput benchmark (the score) and
# the crash-consistency check. Default IMPL is the reference. Needs cmake, a C++17 compiler, and
# Linux for the full-size run. Override any default inline:
#   IMPL=path/to/candidate.cc WORKLOAD=uniform LOAD=250000000 bash score.sh
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
while [[ ! -f "$root/pyproject.toml" && "$root" != "/" ]]; do root="$(dirname "$root")"; done
cd "$root"

EX="${EX:-$root/skydiscover/synthesize/examples/single-machine-kvstore}"
IMPL="${IMPL:-$EX/evaluator/reference_kvstore.cc}"
WORKLOAD="${WORKLOAD:-zipf}"
LOAD="${LOAD:-1000000}"
RUN="${RUN:-2000000}"
OUT="${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/kvstore.XXXXXX")}"

# Stage the harness, header, and impl out of tree (CMake globs generated/*.cc), then build.
mkdir -p "$OUT/src/generated" "$OUT/traces" "$OUT/storage"
cp "$EX/evaluator/benchmark_harness.cc" "$EX/evaluator/consistency_harness.cc" "$EX/evaluator/CMakeLists.txt" "$EX/evaluator/kvstore_interface.h" "$OUT/src/"
cp "$IMPL" "$OUT/src/generated/"
cmake -S "$OUT/src" -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$OUT/build" -j"$(nproc)" >/dev/null

# Throughput (the score): generate the workload, run kvstore_bench -> Mops/s.
uv run python "$EX/evaluator/generate.py" "$WORKLOAD" --load-count "$LOAD" --run-count "$RUN" --outdir "$OUT/traces"
read -r L R < <(uv run python -c "import json,glob,os;d=json.load(open(max(glob.glob('$OUT/traces/*.meta.json'),key=os.path.getmtime)));print(d['load_file'],d['run_file'])")
"$OUT/build/kvstore_bench" 0 4 "$L" "$R"

# Consistency (the test): the harness expects a 1M-key load; run api_smoke + fuzzy crash recovery.
uv run python "$EX/evaluator/generate.py" uniform --load-count 1000000 --run-count 1000000 --outdir "$OUT/traces" >/dev/null
C="$(ls -t "$OUT/traces"/load_uniform_*_raw.dat | head -1)"
"$OUT/build/consistency_test" 7 4 "$C" 67108864 "$OUT/storage"
"$OUT/build/consistency_test" 1 4 "$C" 67108864 "$OUT/storage"

echo "done. Artifacts under $OUT"
