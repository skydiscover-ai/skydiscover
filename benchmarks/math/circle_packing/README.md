# Circle Packing

Pack 26 non-overlapping circles in a unit square to maximize the sum of their radii (AlphaEvolve B.12). Target: 2.635.

## Problem

- Pack exactly 26 circles inside a unit square
- No circles may overlap
- Each circle must lie entirely within the square
- Maximize the sum of all radii

## Run

```bash
# From repo root
uv run skydiscover-run \
  benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/eval \
  -c benchmarks/math/circle_packing/config.yaml \
  -s [your_algorithm] \
  -i 100
```

A `codebase/reference/` directory is provided with geometric insights (hex grids, optimization patterns, packing strategies) that can be used with agentic mode (`--agentic`).

## Scoring

- **combined_score**: `sum_of_radii / 2.635` (ratio to AlphaEvolve target)
- Evaluator validates no overlaps and boundary constraints

## Files

| File | Description |
|------|-------------|
| `initial_program.py` | Seed: simple ring-based circle arrangement |
| `eval/` | Containerized evaluator — validates constraints, computes sum-of-radii ratio to target |
| `config.yaml` | LLM and evaluator settings |
| `codebase/reference/` | Geometric reference material for agentic mode |
