# Program

Create a Bayesian optimization strategy for the benchmark problem in `/workspace/project/benchmark_problem.py`.

First read `/workspace/project/benchmark_problem.py` carefully. Use the observed dataset, benchmark metadata, search space, parameter bounds, metric names, and metric direction from that file. The remote evaluator will run your strategy outside this sandbox for 200 Bayesian optimization steps.

Your goal is to get the best final metric value after those 200 steps. Favor strategies that match the benchmark structure you discover: dimensionality, continuous or discrete parameters, objective direction, apparent noise, and the quality and spread of the initial observations. Balance exploration and exploitation for a sequential 200-step run.

You must use the available file-writing/editing tools to create the required output files. Do not finish by merely explaining the strategy in chat. Before you stop, verify that both `/workspace/output/bo_algorithm.py` and `/workspace/output/summary.md` exist.

Write `/workspace/output/bo_algorithm.py`. It must be importable and must expose one of the following:

- `make_strategy_spec()`, returning either a `lila.kepler.StrategySpec` instance or a plain dict of keyword arguments for `StrategySpec`.
- `STRATEGY_SPEC`, containing either a `StrategySpec` instance or a plain dict of keyword arguments for `StrategySpec`.

Prefer a plain dict so the evaluator can validate it cleanly. Keep `bo_algorithm.py` simple and side-effect free: when the evaluator imports it or calls `make_strategy_spec()`, it should not read or write files, run optimization, call the oracle, install packages, or depend on anything outside the available environment. Example shape:

The evaluator loads plain dicts by running `StrategySpec(**STRATEGY_SPEC)`. Your dict must contain only valid `lila.kepler.StrategySpec` keyword arguments. Valid keys are:

- Required: `strategy`, `n_candidates`
- Optional: `objectives`, `constraints`, `target_metric`, `target_value`, `tolerance`, `punchout_radius`, `eci_radius_sampling_seed`, `ref_point`, `num_fantasies`, `llm`, `exploration_weight`, `candidate_pool_size`, `label`, `feasibility`

Do not include any keys outside this list. Unknown keys cause `TypeError` and make the benchmark fail.

```python
STRATEGY_SPEC = {
    "strategy": "qLogNoisyExpectedImprovement",
    "n_candidates": 1,
    "objectives": "negative_ackley",
    "label": "agent_qlognei",
}


def make_strategy_spec():
    return STRATEGY_SPEC
```

Only write files under `/workspace/output/`. Do not modify `/workspace/project/benchmark_problem.py` or any evaluation harness files. Do not run the optimization loop yourself in `bo_algorithm.py`; the evaluator handles that.

All else being equal, simpler is better. A small expected gain is not worth fragile, opaque, or overfit code. A simple strategy that uses the benchmark metadata correctly is better than a complicated one that may fail to import or generalize.

Also write `/workspace/output/summary.md` with the strategy you selected and a short explanation of how you used the dataset and benchmark metadata. Mention the objective name and direction, the search-space shape, and why the selected strategy is appropriate for the 200-step budget.

Final checklist:

- `/workspace/output/bo_algorithm.py` exists and contains `STRATEGY_SPEC` or `make_strategy_spec()`.
- `/workspace/output/summary.md` exists.
- `STRATEGY_SPEC` contains only valid `StrategySpec` keyword arguments.
- Importing `bo_algorithm.py` will not run optimization, call an oracle, install packages, or access the network.
