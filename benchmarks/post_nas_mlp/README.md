# Post-NAS MLP microbenchmark

Addresses [issue #2](https://github.com/skydiscover-ai/skydiscover/issues/2)
with a mergeable wedge: evolve **model graphs** (MLP layer sequences), not
just Python algorithms.

The evaluator is pure NumPy (no torch) so it runs in the default SkyDiscover
environment. Architectures are trained with SGD on a synthetic nonlinear
3-class task; score = accuracy with a light size penalty.

## Run

```bash
uv run skydiscover-run benchmarks/post_nas_mlp/initial_program.py \
  benchmarks/post_nas_mlp/evaluator.py \
  -c benchmarks/post_nas_mlp/config.yaml -s best_of_n -i 50
```

Offline smoke:

```bash
python3 benchmarks/post_nas_mlp/evaluator.py benchmarks/post_nas_mlp/initial_program.py
```
