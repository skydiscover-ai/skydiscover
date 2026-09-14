# Hypothesis → experiment

Scientific discovery wedge: evolve an **experiment design** plus a **fitted
hypothesis**, scored against a hidden noisy oracle. Held-out R² on the true
process is the primary metric.

Baseline uses stratified random queries + degree-2 least squares — good enough
to beat chance, bad enough that better designs / models can win.

## Run

```bash
uv run skydiscover-run benchmarks/hypothesis_experiment/initial_program.py \
  benchmarks/hypothesis_experiment/evaluator.py \
  -c benchmarks/hypothesis_experiment/config.yaml -s best_of_n -i 50
```

```bash
python3 benchmarks/hypothesis_experiment/evaluator.py \
  benchmarks/hypothesis_experiment/initial_program.py
```
