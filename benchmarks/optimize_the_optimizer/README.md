# Optimize the optimizer

EvoX²-style benchmark: the discovery object is a **search controller**
(selection + mutation / `ask` policy), scored by how well it optimizes a
portfolio of black-box functions under a fixed eval budget.

Held-out problems (Ackley, Rosenbrock) dominate `combined_score` so the
controller must transfer, not memorize the train suite.

## Run

```bash
uv run skydiscover-run benchmarks/optimize_the_optimizer/initial_program.py \
  benchmarks/optimize_the_optimizer/evaluator.py \
  -c benchmarks/optimize_the_optimizer/config.yaml -s best_of_n -i 50
```

```bash
python3 benchmarks/optimize_the_optimizer/evaluator.py \
  benchmarks/optimize_the_optimizer/initial_program.py
```
