# Real-Time Adaptive Signal Processing

Evolve a real-time adaptive filtering algorithm for non-stationary time series data. The algorithm must filter noise while preserving signal dynamics and minimizing computational latency.

## Problem

**Input**: Univariate time series with non-linear dynamics, non-stationary statistics, and rapidly changing spectral characteristics.

**Constraints**: Causal processing (finite sliding window), fixed latency, real-time capability.

**Multi-objective function**:
```
J(theta) = 0.3*S + 0.2*L_recent + 0.2*L_avg + 0.3*R
```
- **S**: Slope change penalty (directional reversals in filtered signal)
- **L_recent**: Instantaneous lag error
- **L_avg**: Average tracking error
- **R**: False reversal penalty (noise-induced trend changes)

The evaluator tests on 5 synthetic signals: sinusoidal, multi-frequency, non-stationary, step changes, and random walk.

## Run

```bash
# From repo root
uv run skydiscover-run \
  benchmarks/math/signal_processing/initial_program.py \
  benchmarks/math/signal_processing/eval \
  -c benchmarks/math/signal_processing/config.yaml \
  -s [your_algorithm] \
  -i 100
```

## Scoring

- **combined_score**: Composite J(theta) metric (higher is better)
- Also reports: slope changes, correlation, lag error, noise reduction, processing time

## Files

| File | Description |
|------|-------------|
| `initial_program.py` | Seed: basic moving average / weighted exponential filters |
| `eval/` | Containerized evaluator — multi-objective evaluation across 5 synthetic test signals |
| `config.yaml` | LLM and evaluator settings |
| `requirements.txt` | Python dependencies |
