# Math Benchmarks

Mathematical optimization and algorithm evolution problems.

## Problems

### Signal processing & geometry (from SkyDiscover demos)

- [signal_processing](signal_processing/) — Real-time adaptive filtering for non-stationary time series
- [circle_packing](circle_packing/) — Pack 26 circles in a unit square to maximize sum of radii (AlphaEvolve B.12)

### AlphaEvolve mathematical problems

12 problems from [AlphaEvolve Appendices A and B](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf). All evaluators are normalized to **maximize** the target metric.

**Appendix A:**
- [matmul](matmul/) — Faster algorithm for matrix multiplication (A)

**Appendix B:**
1. [first_autocorr_ineq](first_autocorr_ineq/) — Upper bound on autoconvolution constant (B.1)
2. [second_autocorr_ineq](second_autocorr_ineq/) — Lower bound on autoconvolution norm constant (B.2)
3. [third_autocorr_ineq](third_autocorr_ineq/) — Upper bound on absolute autoconvolution constant (B.3)
4. [uncertainty_ineq](uncertainty_ineq/) — Upper bound on Fourier uncertainty constant (B.4)
5. [erdos_min_overlap](erdos_min_overlap/) — Upper bound on Erdos minimum overlap constant (B.5)
6. [sums_diffs_finite_sets](sums_diffs_finite_sets/) — Lower bound on sums/differences of finite sets (B.6)
7. [hexagon_packing](hexagon_packing/) — Pack unit hexagons in a regular hexagon, n=11,12 (B.7)
8. [minimizing_max_min_dist](minimizing_max_min_dist/) — Minimize max/min distance ratio, n=16 d=2 and n=14 d=3 (B.8)
9. [heilbronn_triangle](heilbronn_triangle/) — Heilbronn problem for triangles, n=11 (B.9)
10. [heilbronn_convex](heilbronn_convex/) — Heilbronn problem for convex regions, n=13,14 (B.10)
11. [circle_packing_rect](circle_packing_rect/) — Pack circles in a rectangle of perimeter 4 (B.13)

## Run

```bash
uv run skydiscover-run \
  benchmarks/math/signal_processing/initial_program.py \
  benchmarks/math/signal_processing/eval \
  -c benchmarks/math/signal_processing/config.yaml \
  -s [your_algorithm] \
  -i 100
```

Each problem directory contains `initial_program.py`, `eval/` (containerized evaluator), and either `config.yaml` or per-search configs. Some multi-variant problems have numbered subdirectories (e.g., `heilbronn_convex/13/`, `hexagon_packing/11/`).
