# ALE-Bench: AtCoder Heuristic Contest Benchmark

10 problems from AtCoder Heuristic Contests (AHC), evaluated via the `ale_bench` package. Programs are written in C++ and scored on 50 public test cases during evolution. A separate private evaluator runs the full hidden test set for final ranking.

## Problems

| Problem | Description |
|---------|-------------|
| `ahc008` | Pet partitioning — place walls to create pet-free areas on a 30×30 grid over 300 turns |
| `ahc011` | AtCoder Heuristic Contest 11 |
| `ahc015` | AtCoder Heuristic Contest 15 |
| `ahc016` | AtCoder Heuristic Contest 16 |
| `ahc024` | AtCoder Heuristic Contest 24 |
| `ahc025` | Balance weighing — use a balance scale to divide N items into D equal-weight sets using Q queries |
| `ahc026` | AtCoder Heuristic Contest 26 |
| `ahc027` | AtCoder Heuristic Contest 27 |
| `ahc039` | AtCoder Heuristic Contest 39 |
| `ahc046` | AtCoder Heuristic Contest 46 |

## Quick Start

Run evolution on a single problem:

```bash
uv run skydiscover-run \
  benchmarks/ale_bench/ale-bench-lite-problems/ahc025/initial_program.cpp \
  benchmarks/ale_bench/ale-bench-lite-problems/ahc025/eval \
  -c benchmarks/ale_bench/ale-bench-lite-problems/ahc025/config.yaml \
  --search evox \
  -i 100
```

## Scoring

During evolution, each iteration runs 50 public test cases:

```
combined_score = overall_absolute_score * optim_factor / num_public_cases
```

`optim_factor` is `+1` for maximize problems and `-1` for minimize problems (so `combined_score` is always higher-is-better).

## Private Evaluation

After evolution, evaluate the best program on the full private test set:

```bash
python benchmarks/ale_bench/private_eval.py \
  --program-path path/to/best_program.cpp \
  --problem-id ahc025
```

This runs 3 independent evaluations and reports the average private rank, performance score, and per-case pass/fail counts.

## Directory Structure

```
ale_bench/
├── ale-bench-lite-problems/
│   └── ahcXXX/
│       ├── initial_program.cpp   # Starting C++ solution
│       ├── eval/             # Containerized evaluator (runs 50 public cases via ale_bench)
│       └── config.yaml           # Search config (cpp, diff-based, 100 iterations)
├── ale_agent_best/
│   └── ahcXXX.cpp               # Best known solutions (reference)
└── private_eval.py              # Full private set evaluation + ranking
```

## Requirements

Requires the `ale_bench` and `ale_bench_eval` packages. These are not in the default `uv sync` — install them separately per the ALE-Bench documentation.

## Config Defaults

All problems share the same base config:

```yaml
language: cpp
diff_based_evolution: true
max_iterations: 100
max_solution_length: 60000
evaluator:
  timeout: 10000
```
