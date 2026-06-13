#!/bin/bash
# run.sh -- Thin CLI wrapper for the evolution orchestrator.
#
# Parses arguments, resolves config, exports as SKYKV_* env vars,
# and calls agent-pipeline/orchestrator.py to run the evolution loop.
#
# Usage:
#   bash agent-pipeline/run.sh --backend claude --mode inmem --distribution zipf --setup rmw --iterations 100
#   bash agent-pipeline/run.sh --backend codex  --mode ltm  --distribution zipf --setup 50:50 --iterations 100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Parse args ────────────────────────────────────────────────────────────────
BACKEND=""
MODE=""
DISTRIBUTION=""
SETUP=""
TRACE_LOAD=""
TRACE_RUN=""
VALUE_SIZE=""
MEM_BUDGET_GB=""
ITERATIONS=50
MAX_TURNS=50
MODEL=""
THREADS=""
# critique_mode: one of {off, review, audit, full}
#   off    = no critique (default)
#   review = existing review feedback to generator
#   audit  = accumulate tests + run them as a gate (new)
#   full   = both
CRITIQUE_MODE="off"
# Default ON — iteration loops don't need destructor hygiene and the OS
# reclaims memory via _exit() in ms rather than minutes. Use --no-fast-exit
# only for final runs that need clean shutdown (e.g., valgrind).
BENCH_FAST_EXIT=1
# --summary: rotate codex sessions + inject a handoff memo when per-turn
# tokens exceed 200K. Prevents unbounded resume-replay (which triggers TPM
# 429s because codex 0.120.0 does not auto-compact across exec-resume).
SUMMARY=0
# --show-baseline: show "X% of FASTER" in agent feedback. Off by default —
# the agent should optimize from its own trajectory, not anchor to a number.
SHOW_BASELINE=0
# --audit-every: how often the AUDITOR (test-writing agent) fires, in iters.
#   1 = every iteration.
#   N > 1 = queue N iters' artifacts, fire auditor once every N with the full
#           batch. Audit GATE (running accumulated tests pre-bench) still runs
#           every iter — only the test-writing stage is batched.
# Default 15: auditor now has a high bar (null finding is the preferred output
# when no hack meeting the ≥5% Mops/s threshold exists), so writing new tests
# every iter is wasteful. Accumulated gate tests still run every iter, so
# signal strength is unaffected. Set to 1 if you want per-iter audit writes.
AUDIT_EVERY=15
# --seed: start iter 1 from seeds/<name>/ instead of an agent-generated impl.
# Orchestrator clears workspace/interface/generated/ and copies the seed in.
# Agent is SKIPPED on iter 1 (baseline measurement of the seed as-is); iter
# 2+ runs normally and may modify/delete/rewrite anything under generated/
# including vendored dependency trees like faster_src/. Empty = no seed.
SEED=""
PARALLEL_EVAL=0
NUM_WORKERS=""
# --feedback-level: controls how much diagnostic detail the agent sees.
#   minimal = Mops/s + pass/fail only (ablation: no "gradient")
#   rich    = everything: op breakdown, cache hits, memory util, budget
#             guidance, load speed, perf stat counters (default)
FEEDBACK_LEVEL="rich"
# --delete-rate: time-series-only knob. r = delete_ops / insert_ops. r=1.0
# (default) is stationary. Exported as KVSTORE_DELETE_RATE for the harness.
DELETE_RATE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)       BACKEND="$2"; shift 2 ;;
        --mode)          MODE="$2"; shift 2 ;;
        --distribution)  DISTRIBUTION="$2"; shift 2 ;;
        --setup)         SETUP="$2"; shift 2 ;;
        --value-size)    VALUE_SIZE="$2"; shift 2 ;;
        --mem-budget)    MEM_BUDGET_GB="${2//,/ }"; shift 2 ;;
        --iterations)    ITERATIONS="$2"; shift 2 ;;
        --max-turns)     MAX_TURNS="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --threads)       THREADS="$2"; shift 2 ;;
        --delete-rate)   DELETE_RATE="$2"; shift 2 ;;
        --critique)
            # Accept either `--critique` (legacy, means review) or
            # `--critique <mode>` (new, mode in off/review/audit/full).
            if [[ $# -ge 2 && "$2" =~ ^(off|review|audit|full)$ ]]; then
                CRITIQUE_MODE="$2"; shift 2
            else
                CRITIQUE_MODE="review"; shift
            fi ;;
        --no-fast-exit)  BENCH_FAST_EXIT=0; shift ;;
        --fast-exit)     BENCH_FAST_EXIT=1; shift ;;
        --summary)       SUMMARY=1; shift ;;
        --show-baseline) SHOW_BASELINE=1; shift ;;
        --seed)          SEED="$2"; shift 2 ;;
        --trace-load)    TRACE_LOAD="$2"; shift 2 ;;
        --trace-run)     TRACE_RUN="$2"; shift 2 ;;
        --audit-every)   AUDIT_EVERY="$2"; shift 2 ;;
        --feedback-level)
            if [[ $# -ge 2 && "$2" =~ ^(minimal|rich)$ ]]; then
                FEEDBACK_LEVEL="$2"; shift 2
            else
                echo "ERROR: --feedback-level must be 'minimal' or 'rich'"; exit 1
            fi ;;
        --parallel-eval) PARALLEL_EVAL=1; shift ;;
        --num-workers)   NUM_WORKERS="$2"; shift 2 ;;
        --no-planner)    NO_PLANNER=1; shift ;;
        --no-leaderboard) NO_LEADERBOARD=1; shift ;;
        --audit-checks-dir) AUDIT_CHECKS_DIR="$2"; shift 2 ;;
        -h|--help)       head -9 "$0" | tail -8; exit 0 ;;
        *)               echo "Unknown option: $1"; exit 1 ;;
    esac
done

NO_PLANNER="${NO_PLANNER:-0}"
NO_LEADERBOARD="${NO_LEADERBOARD:-0}"
AUDIT_CHECKS_DIR="${AUDIT_CHECKS_DIR:-}"

# ── Validate required args ────────────────────────────────────────────────────
[[ -z "$BACKEND" ]]      && { echo "ERROR: --backend is required (claude or codex)"; exit 1; }
[[ -z "$MODE" ]]         && { echo "ERROR: --mode is required (inmem or ltm)"; exit 1; }
[[ -z "$DISTRIBUTION" && -z "$TRACE_LOAD" ]] && { echo "ERROR: --distribution or --trace-load/--trace-run is required"; exit 1; }
[[ -z "$SETUP" ]]        && { echo "ERROR: --setup is required (rmw, 0:100, 50:50, or 100:0)"; exit 1; }
[[ ! "$AUDIT_EVERY" =~ ^[1-9][0-9]*$ ]] && { echo "ERROR: --audit-every must be a positive integer (got: $AUDIT_EVERY)"; exit 1; }

# ── Handle --setup all: run each setup sequentially ───────────────────────────
# Note: `ts-head` is intentionally EXCLUDED from `--setup all` -- it requires
# `--distribution timeseries` (enforced by the coupling check below) and is
# invoked directly via setups/faster-single-machine/scripts/run_ts_head_delete_*.sh.
if [[ "$SETUP" == "all" ]]; then
    for s in rmw 50:50 100:0 0:100; do
        echo "================================================================"
        echo "  SETUP: $s"
        echo "================================================================"
        bash "$0" --backend "$BACKEND" --mode "$MODE" --distribution "$DISTRIBUTION" \
            --setup "$s" --iterations "$ITERATIONS" --max-turns "$MAX_TURNS" \
            ${MODEL:+--model "$MODEL"} ${VALUE_SIZE:+--value-size "$VALUE_SIZE"} \
            ${MEM_BUDGET_GB:+--mem-budget "$MEM_BUDGET_GB"} ${THREADS:+--threads "$THREADS"} \
            --audit-every "$AUDIT_EVERY" \
            --critique "$CRITIQUE_MODE"
    done
    exit 0
fi

# ── Resolve config ────────────────────────────────────────────────────────────
case "$BACKEND" in
    claude) MODEL="${MODEL:-claude-sonnet-4-6}" ;;
    codex)  MODEL="${MODEL:-gpt-5.4}" ;;
    *)      echo "ERROR: --backend must be 'claude' or 'codex'"; exit 1 ;;
esac

# Custom trace paths override --distribution
if [[ -n "$TRACE_LOAD" && -n "$TRACE_RUN" ]]; then
    LOAD="$TRACE_LOAD"
    RUN="$TRACE_RUN"
    # Derive a unique-per-trace key so STORAGE_DIR doesn't collide across runs
    # (multiple --trace-load invocations would otherwise overwrite each other).
    _base=$(basename "$LOAD" .dat)
    _base=${_base#load_}
    _base=${_base%_raw}
    DIST_KEY="custom_${_base}"
    DIST_DESC="Custom trace ($_base)"
    [[ ! -f "$LOAD" ]] && { echo "ERROR: --trace-load not found: $LOAD"; exit 1; }
    [[ ! -f "$RUN" ]]  && { echo "ERROR: --trace-run not found: $RUN"; exit 1; }
elif [[ -n "$DISTRIBUTION" ]]; then
case "$DISTRIBUTION" in
    zipf|zipfian)
        DIST_KEY="zipf"; DIST_DESC="Zipfian θ=0.99"
        LOAD=/mnt/ssd/ycsb_data/load_zipf_250M_raw.dat
        RUN=/mnt/ssd/ycsb_data/run_zipf_250M_1000M_raw.dat ;;
    uniform)
        DIST_KEY="uniform"; DIST_DESC="Uniform"
        LOAD=/mnt/ssd/ycsb_data/load_uniform_250M_raw.dat
        RUN=/mnt/ssd/ycsb_data/run_uniform_250M_1000M_raw.dat
        [[ ! -f "$LOAD" ]] && { echo "ERROR: uniform trace not found at $LOAD"; exit 1; } ;;
    # ── Adversarial distributions (traces/generate.py) ──────────────────
    # These exploit weaknesses in FASTER's FIFO hybrid-log eviction.
    # Generate key files first:
    #   python3 traces/generate.py --pattern <name> --outdir /mnt/ssd/ycsb_data [--mem-budget-gb 8 --value-size 100]
    scan)
        # Sequential cyclic scan over all 250M keys.  Best with: rmw, 50:50
        DIST_KEY="scan"; DIST_DESC="Sequential scan (adversarial)"
        LOAD=/mnt/ssd/ycsb_data/load_scan_250M_raw.dat
        RUN=/mnt/ssd/ycsb_data/run_scan_250M_1000M_raw.dat
        [[ ! -f "$LOAD" ]] && { echo "ERROR: scan trace not found — run: python3 traces/generate.py scan --outdir /mnt/ssd/ycsb_data"; exit 1; } ;;
    belady)
        # Cycle through (in-memory capacity + 1) keys.  Best with: rmw, 50:50
        DIST_KEY="belady"; DIST_DESC="Capacity+1 cycle — optimal FIFO adversary"
        LOAD=/mnt/ssd/ycsb_data/load_belady_250M_raw.dat
        RUN=/mnt/ssd/ycsb_data/run_belady_250M_1000M_raw.dat
        [[ ! -f "$LOAD" ]] && { echo "ERROR: belady trace not found — run: python3 traces/generate.py belady --working-set 50000001 --outdir /mnt/ssd/ycsb_data"; exit 1; } ;;
    stride)
        # Cycle through 1.1× mutable region — forces 100% RCU copies.  Best with: rmw
        DIST_KEY="stride"; DIST_DESC="Mutable-overflow stride (adversarial)"
        LOAD=/mnt/ssd/ycsb_data/load_stride_250M_raw.dat
        RUN=/mnt/ssd/ycsb_data/run_stride_250M_1000M_raw.dat
        [[ ! -f "$LOAD" ]] && { echo "ERROR: stride trace not found — run: python3 traces/generate.py stride --working-set 45000000 --outdir /mnt/ssd/ycsb_data"; exit 1; } ;;
    # ── Real-world traces (traces/generate.py) ──────────────────────────
    # Unlike the synthetic 250M/1B files above, these have native sizes (Meta
    # KV = 82M unique / 1.64B accesses; Twitter varies per cluster). The
    # harness auto-detects counts from file size — no code changes needed.
    # Generate first:
    #   python3 traces/generate.py --source metakv      --outdir /mnt/ssd/ycsb_data
    #   python3 traces/generate.py --source twitter --cluster 18 --outdir /mnt/ssd/ycsb_data
    metakv)
        # Meta Cachelib KV trace (Berg OSDI'20). Highly skewed KV workload.
        DIST_KEY="metakv"; DIST_DESC="Meta Cachelib KV (Berg OSDI'20)"
        # Glob-match the generated load/run pair — sizes vary by trace version
        # (the `_<N>_` in the filename is derived from the actual trace count).
        # `shopt -s nullglob` + subshell keeps the pipeline happy under `set -e`.
        LOAD=$(shopt -s nullglob; files=(/mnt/ssd/ycsb_data/load_metakv_*_raw.dat); echo "${files[0]:-}")
        RUN=$(shopt -s nullglob;  files=(/mnt/ssd/ycsb_data/run_metakv_*_raw.dat);  echo "${files[0]:-}")
        [[ -z "$LOAD" || -z "$RUN" ]] && { echo "ERROR: metakv trace not found — run: python3 traces/generate.py metakv --outdir /mnt/ssd/ycsb_data"; exit 1; } ;;
    twitter*)
        # Twitter Twemcache cluster (Yang OSDI'20). Pass cluster N as
        # `--distribution twitter18`. Defaults to cluster18 if bare 'twitter'.
        TWITTER_CLUSTER="${DISTRIBUTION#twitter}"
        [[ -z "$TWITTER_CLUSTER" ]] && TWITTER_CLUSTER=18
        [[ ! "$TWITTER_CLUSTER" =~ ^[0-9]+$ ]] && { echo "ERROR: twitter cluster must be numeric, got '$TWITTER_CLUSTER'"; exit 1; }
        DIST_KEY="twitter${TWITTER_CLUSTER}"; DIST_DESC="Twitter cluster ${TWITTER_CLUSTER} (Yang OSDI'20)"
        LOAD=$(shopt -s nullglob; files=(/mnt/ssd/ycsb_data/load_twitter${TWITTER_CLUSTER}_*_raw.dat); echo "${files[0]:-}")
        RUN=$(shopt -s nullglob;  files=(/mnt/ssd/ycsb_data/run_twitter${TWITTER_CLUSTER}_*_raw.dat);  echo "${files[0]:-}")
        [[ -z "$LOAD" || -z "$RUN" ]] && { echo "ERROR: twitter${TWITTER_CLUSTER} trace not found — run: python3 traces/generate.py twitter --cluster ${TWITTER_CLUSTER} --outdir /mnt/ssd/ycsb_data"; exit 1; } ;;
    # ── Synthetic knobs (traces/generators/synthetic.py) ───────────────
    # Named `one_hit` / `bursty` / `hotspot`; filenames use `onehit_*`,
    # `bursty_*`, `hotspot_*`. Generated via traces/generate.py with their
    # own flags; we just glob whichever variant is on disk.
    one_hit|onehit)
        DIST_KEY="onehit"; DIST_DESC="One-hit wonders (synthetic)"
        LOAD=$(shopt -s nullglob; files=(/mnt/ssd/ycsb_data/load_onehit_*_raw.dat); echo "${files[0]:-}")
        RUN=$(shopt -s nullglob;  files=(/mnt/ssd/ycsb_data/run_onehit_*_raw.dat);  echo "${files[0]:-}")
        [[ -z "$LOAD" || -z "$RUN" ]] && { echo "ERROR: onehit trace not found — run: python3 traces/generate.py one_hit --one-hit-ratio 0.1 --outdir /mnt/ssd/ycsb_data"; exit 1; } ;;
    bursty)
        DIST_KEY="bursty"; DIST_DESC="Bursty accesses (synthetic)"
        LOAD=$(shopt -s nullglob; files=(/mnt/ssd/ycsb_data/load_bursty_*_raw.dat); echo "${files[0]:-}")
        RUN=$(shopt -s nullglob;  files=(/mnt/ssd/ycsb_data/run_bursty_*_raw.dat);  echo "${files[0]:-}")
        [[ -z "$LOAD" || -z "$RUN" ]] && { echo "ERROR: bursty trace not found — run: python3 traces/generate.py bursty --burst-size 5 --outdir /mnt/ssd/ycsb_data"; exit 1; } ;;
    hotspot)
        DIST_KEY="hotspot"; DIST_DESC="Hot-key concentration (synthetic)"
        LOAD=$(shopt -s nullglob; files=(/mnt/ssd/ycsb_data/load_hotspot_*_raw.dat); echo "${files[0]:-}")
        RUN=$(shopt -s nullglob;  files=(/mnt/ssd/ycsb_data/run_hotspot_*_raw.dat);  echo "${files[0]:-}")
        [[ -z "$LOAD" || -z "$RUN" ]] && { echo "ERROR: hotspot trace not found — run: python3 traces/generate.py hotspot --outdir /mnt/ssd/ycsb_data"; exit 1; } ;;
    bimodal)
        # Bimodal value sizes by key parity. The key trace is pure zipf; the
        # harness picks per-op value size from KVSTORE_BIMODAL_VALUES=1
        # (exported below for the orchestrator's bench subprocess).
        DIST_KEY="bimodal"; DIST_DESC="Bimodal per-key value size (synthetic)"
        LOAD=$(shopt -s nullglob; files=(/mnt/ssd/ycsb_data/load_bimodal_*_raw.dat); echo "${files[0]:-}")
        RUN=$(shopt -s nullglob;  files=(/mnt/ssd/ycsb_data/run_bimodal_*_raw.dat);  echo "${files[0]:-}")
        [[ -z "$LOAD" || -z "$RUN" ]] && { echo "ERROR: bimodal trace not found — run: python3 traces/generate.py bimodal --outdir /mnt/ssd/ycsb_data"; exit 1; }
        export KVSTORE_BIMODAL_VALUES=1 ;;
    # ── Time-series head-delete (traces/generators/timeseries.py) ──────
    # Inserts append at tail, deletes only remove from head. The run file
    # is a placeholder -- the harness owns monotone head/tail atomics.
    # Must be paired with --setup ts-head; run.sh enforces this below.
    timeseries)
        DIST_KEY="timeseries"; DIST_DESC="Time-series head-delete"
        LOAD=$(shopt -s nullglob; files=(/mnt/ssd/ycsb_data/load_timeseries_*_raw.dat); echo "${files[0]:-}")
        RUN=$(shopt -s nullglob;  files=(/mnt/ssd/ycsb_data/run_timeseries_*_raw.dat);  echo "${files[0]:-}")
        [[ -z "$LOAD" || -z "$RUN" ]] && { echo "ERROR: timeseries trace not found — run: python3 traces/generate.py timeseries --outdir /mnt/ssd/ycsb_data"; exit 1; } ;;
    # ── Real-world large traces (traces/generators/real.py) ─────────────
    # Generate with `--max-load N` for the huge ones (tencent_photo ~1B keys).
    wikimedia)
        DIST_KEY="wikimedia"; DIST_DESC="Wikimedia CDN 2019 (Yang OSDI'20)"
        LOAD=$(shopt -s nullglob; files=(/mnt/ssd/ycsb_data/load_wikimedia_*_raw.dat); echo "${files[0]:-}")
        RUN=$(shopt -s nullglob;  files=(/mnt/ssd/ycsb_data/run_wikimedia_*_raw.dat);  echo "${files[0]:-}")
        [[ -z "$LOAD" || -z "$RUN" ]] && { echo "ERROR: wikimedia trace not found — run: python3 traces/generate.py wikimedia --outdir /mnt/ssd/ycsb_data"; exit 1; } ;;
    tencent_photo)
        DIST_KEY="tencent_photo"; DIST_DESC="Tencent QQPhoto CDN ICS'18"
        LOAD=$(shopt -s nullglob; files=(/mnt/ssd/ycsb_data/load_tencent_photo_*_raw.dat); echo "${files[0]:-}")
        RUN=$(shopt -s nullglob;  files=(/mnt/ssd/ycsb_data/run_tencent_photo_*_raw.dat);  echo "${files[0]:-}")
        [[ -z "$LOAD" || -z "$RUN" ]] && { echo "ERROR: tencent_photo trace not found — run: python3 traces/generate.py tencent_photo --max-load 50000000 --outdir /mnt/ssd/ycsb_data"; exit 1; } ;;
    *) echo "ERROR: --distribution must be one of: zipf, uniform, scan, belady, stride, metakv, twitter<N>, one_hit, bursty, hotspot, bimodal, timeseries, wikimedia, tencent_photo"; exit 1 ;;
esac
else
    echo "ERROR: --distribution or --trace-load/--trace-run is required"; exit 1
fi

case "$SETUP" in
    rmw|RMW)       WL_ID=1; SETUP_KEY="rmw";   SETUP_NAME="RMW";   SETUP_DESC="100% Read-Modify-Write" ;;
    0:100)         WL_ID=4; SETUP_KEY="0_100";  SETUP_NAME="0:100"; SETUP_DESC="100% blind updates" ;;
    50:50|ycsba)   WL_ID=0; SETUP_KEY="50_50";  SETUP_NAME="50:50"; SETUP_DESC="50% Read, 50% Upsert" ;;
    100:0|ycsbc)   WL_ID=3; SETUP_KEY="100_0";  SETUP_NAME="100:0"; SETUP_DESC="100% Read" ;;
    ts-head|timeseries)
        WL_ID=5; SETUP_KEY="ts_head"; SETUP_NAME="ts-head"
        SETUP_DESC="Time-series head-delete (insert at tail, delete from head)" ;;
    *)             echo "ERROR: --setup must be rmw, 0:100, 50:50, 100:0, ts-head, or all"; exit 1 ;;
esac

# ── Time-series coupling: ts-head requires --distribution timeseries ──
if [[ "$SETUP_KEY" == "ts_head" && "$DIST_KEY" != "timeseries" ]]; then
    echo "ERROR: --setup ts-head requires --distribution timeseries (got '$DIST_KEY')"
    exit 1
fi
if [[ "$DIST_KEY" == "timeseries" && "$SETUP_KEY" != "ts_head" ]]; then
    echo "ERROR: --distribution timeseries requires --setup ts-head (got '$SETUP_KEY')"
    exit 1
fi

# Export KVSTORE_DELETE_RATE if --delete-rate was passed (ts-head only)
if [[ -n "$DELETE_RATE" ]]; then
    if [[ "$SETUP_KEY" != "ts_head" ]]; then
        echo "ERROR: --delete-rate only applies to --setup ts-head (got '$SETUP_KEY')"
        exit 1
    fi
    export KVSTORE_DELETE_RATE="$DELETE_RATE"
fi

case "$MODE" in
    inmem)
        VALUE_SIZE="${VALUE_SIZE:-8}"
        MEM_BUDGET_GB="${MEM_BUDGET_GB:-0}"
        THREADS="${THREADS:-1 64}"
        STORAGE_DIR="/mnt/ssd/kvstore_${BACKEND}_${DIST_KEY}_${SETUP_KEY}_inmem" ;;
    ltm)
        VALUE_SIZE="${VALUE_SIZE:-100}"
        MEM_BUDGET_GB="${MEM_BUDGET_GB:-8 32}"
        THREADS="${THREADS:-16}"
        STORAGE_DIR="/mnt/ssd/kvstore_${BACKEND}_${DIST_KEY}_${SETUP_KEY}_ltm_${MEM_BUDGET_GB// /_}GB" ;;
    *) echo "ERROR: --mode must be 'inmem' or 'ltm'"; exit 1 ;;
esac

RUN_KEY="${DIST_KEY}_${SETUP_KEY}_${VALUE_SIZE}B"
[[ "$MEM_BUDGET_GB" != "0" ]] && RUN_KEY="${RUN_KEY}_${MEM_BUDGET_GB// /_}GB"

# ── Real-trace safety caps ───────────────────────────────────────────────────
# Real traces (metakv / twitter / wikimedia / tencent_photo) have native run
# counts of 1.6–5.6 B. Cap the harness's consumed prefix at 1 B so the array
# + budget + cgroup overhead fit our standard 20 GB cgroup. Matches the
# synthetic baseline (250 M load / 1 B run). Respects user override.
case "$DIST_KEY" in
    metakv|twitter*|wikimedia|tencent_photo)
        if [[ -z "${KVSTORE_MAX_TXN_COUNT:-}" ]]; then
            export KVSTORE_MAX_TXN_COUNT=1000000000
        fi ;;
esac

# ── 200 GB LOAD-size safety guard ────────────────────────────────────────────
# Refuse to start if (LOAD keys × --value-size) would exceed 200 GB. Catches
# footguns like `--distribution wikimedia --value-size 16384` (= 4 TB LOAD).
# Escape hatches:
#   SKYKV_BIG_LOAD_OK=1         override guard entirely
#   KVSTORE_MAX_INIT_COUNT=N    cap LOAD keys at runtime (harness honors this)
if [[ -f "$LOAD" && "${SKYKV_BIG_LOAD_OK:-0}" != "1" ]]; then
    LOAD_BYTES=$(stat -c %s "$LOAD" 2>/dev/null || stat -f %z "$LOAD" 2>/dev/null || echo 0)
    LOAD_KEYS=$(( LOAD_BYTES / 8 ))
    # If caller set KVSTORE_MAX_INIT_COUNT, the harness reads at most that
    # many keys — reflect the tighter bound here.
    if [[ -n "${KVSTORE_MAX_INIT_COUNT:-}" && "$KVSTORE_MAX_INIT_COUNT" -gt 0 ]]; then
        (( KVSTORE_MAX_INIT_COUNT < LOAD_KEYS )) && LOAD_KEYS="$KVSTORE_MAX_INIT_COUNT"
    fi
    _FIRST_VALUE_SIZE=${VALUE_SIZE%% *}   # handle multi-value sweeps
    EST_BYTES=$(( LOAD_KEYS * _FIRST_VALUE_SIZE ))
    if (( EST_BYTES > 200 * 1024 * 1024 * 1024 )); then
        EST_GB=$(( EST_BYTES / 1024 / 1024 / 1024 ))
        echo "ERROR: estimated LOAD data = ${LOAD_KEYS} keys × ${_FIRST_VALUE_SIZE} B"
        echo "       ≈ ${EST_GB} GB, exceeds the 200 GB safety cap. Likely one of:"
        echo "         1. Shrink --value-size  (e.g. 100 instead of 4096)"
        echo "         2. Cap LOAD keys at runtime:  KVSTORE_MAX_INIT_COUNT=25000000 bash $0 ..."
        echo "         3. Regenerate trace with --max-load (for real traces)"
        echo "         4. Override this guard:        SKYKV_BIG_LOAD_OK=1 bash $0 ..."
        exit 1
    fi
fi

# ── Export config as SKYKV_* env vars and call orchestrator ───────────────────
export SKYKV_PROJECT_DIR="$PROJECT_DIR"
export SKYKV_BACKEND="$BACKEND"
export SKYKV_MODE="$MODE"
export SKYKV_DIST_KEY="$DIST_KEY"
export SKYKV_DIST_DESC="$DIST_DESC"
export SKYKV_LOAD="$LOAD"
export SKYKV_RUN="$RUN"
export SKYKV_WL_ID="$WL_ID"
export SKYKV_SETUP_KEY="$SETUP_KEY"
export SKYKV_SETUP_NAME="$SETUP_NAME"
export SKYKV_SETUP_DESC="$SETUP_DESC"
export SKYKV_VALUE_SIZE="$VALUE_SIZE"
export SKYKV_MEM_BUDGET_GB="$MEM_BUDGET_GB"
export SKYKV_ITERATIONS="$ITERATIONS"
export SKYKV_MAX_TURNS="$MAX_TURNS"
export SKYKV_MODEL="$MODEL"
export SKYKV_THREADS="$THREADS"
export SKYKV_CARD_SET="faster-single-machine"
export SKYKV_BASELINE_NAME="FASTER"
export SKYKV_STORAGE_DIR="$STORAGE_DIR"
export SKYKV_RUN_KEY="$RUN_KEY"
export SKYKV_CRITIQUE_MODE="$CRITIQUE_MODE"
export SKYKV_BENCH_FAST_EXIT="$BENCH_FAST_EXIT"
export SKYKV_SUMMARY="$SUMMARY"
export SKYKV_SHOW_BASELINE="$SHOW_BASELINE"
export SKYKV_SEED="$SEED"
export SKYKV_AUDIT_EVERY="$AUDIT_EVERY"
export SKYKV_FEEDBACK_LEVEL="$FEEDBACK_LEVEL"
export SKYKV_PARALLEL_EVAL="$PARALLEL_EVAL"
export SKYKV_NUM_WORKERS="${NUM_WORKERS}"
export SKYKV_NO_PLANNER="$NO_PLANNER"
export SKYKV_NO_LEADERBOARD="$NO_LEADERBOARD"
export SKYKV_AUDIT_CHECKS_DIR="$AUDIT_CHECKS_DIR"
# Back-compat: keep the old boolean env var set to 1 when any critique mode
# is active, so code that still reads SKYKV_CRITIQUE_ENABLED behaves correctly.
if [[ "$CRITIQUE_MODE" == "off" ]]; then
    export SKYKV_CRITIQUE_ENABLED=0
else
    export SKYKV_CRITIQUE_ENABLED=1
fi

exec python3 -u "$SCRIPT_DIR/orchestrator.py"
