# External Backends

SkyDiscover wraps third-party evolution engines so you can swap algorithms with a single flag. Each backend translates SkyDiscover's unified config into the native library's format and streams results back through the common monitor interface.

| Backend | Package | Key Idea |
|---------|---------|----------|
| **OpenEvolve** | `openevolve` | Island-model genetic programming with diff-based generation and feature-based diversity |
| **GEPA** | `gepa[full]` | Reflection-guided evolutionary optimization via `optimize_anything` API |
| **ShinkaEvolve** | `shinka` | Multi-patch evolution with UCB1 dynamic LLM selection and code-embedding deduplication |
| **AlphaEvolve** | `alpha-evolve` | Google's hosted evolution service (Discovery Engine), candidates generated in the cloud and evaluated locally |

## OpenEvolve

Island model with population size 40, 5 islands, and migration. Supports diff-based generation mode with feature-based diversity tracking (score + complexity dimensions).

### Usage

```bash
uv run skydiscover-run initial_program.py evaluator.py \
  --config config.yaml --search openevolve --iterations 100
```

```python
from skydiscover import run_discovery
result = run_discovery(
    initial_program="initial_program.py",
    evaluator="evaluator.py",
    search="openevolve",
    model="gpt-5",
    iterations=100,
)
```

### Config

See `defaults/openevolve_default.yaml` for full template. Key settings:

```yaml
search:
  type: "openevolve"
  population_size: 40
  num_islands: 5
  checkpoint_interval: 10
  llm_temperature: 0.7
  evaluator_timeout: 360
```

---

## GEPA

Delegates directly to GEPA's `optimize_anything` API with reflection-guided optimization. Maps system prompts to GEPA's "background" parameter.

### Usage

```bash
uv run skydiscover-run initial_program.py evaluator.py \
  --config config.yaml --search gepa --iterations 100
```

```python
from skydiscover import run_discovery
result = run_discovery(
    initial_program="initial_program.py",
    evaluator="evaluator.py",
    search="gepa",
    model="gpt-5",
    iterations=100,
)
```

### Config

GEPA maps SkyDiscover config to its own `GEPAConfig` (engine + reflection configs). Key settings:

```yaml
search:
  type: "gepa"
```

---

## ShinkaEvolve

Multi-patch evolution (diff 60%, full 30%, cross 10%) with dynamic LLM selection via UCB1 strategy. Meta-model guidance every 10 generations. Code-embedding deduplication at threshold 0.995.

### Usage

```bash
uv run skydiscover-run initial_program.py evaluator.py \
  --config config.yaml --search shinkaevolve --iterations 100
```

```python
from skydiscover import run_discovery
result = run_discovery(
    initial_program="initial_program.py",
    evaluator="evaluator.py",
    search="shinkaevolve",
    model="gpt-5",
    iterations=100,
)
```

### Config

See `defaults/shinkaevolve_default.yaml` for full template. Key settings:

```yaml
search:
  type: "shinkaevolve"
  num_parallel_jobs: 4
  num_islands: 5
  archive_size: 20
  meta_model_interval: 10
  llm_selection_strategy: "ucb1"
  embedding_model: "text-embedding-3-small"
  code_similarity_threshold: 0.995
```

---

## AlphaEvolve

Wraps Google's AlphaEvolve EAP SDK. Candidate generation happens in Google Cloud (Discovery Engine), while your evaluator still runs locally. The backend writes each candidate to a temp file using the run's `file_suffix` and calls `evaluate(path)` on it.

Requires GCP Application Default Credentials (`gcloud auth application-default login`, `GOOGLE_APPLICATION_CREDENTIALS`, or a GCP VM service account) and a Discovery Engine engine with the AlphaEvolve API enabled.

### Usage

```bash
export ALPHAEVOLVE_PROJECT_ID=your-gcp-project
export ALPHAEVOLVE_ENGINE_ID=your-engine-id

uv run skydiscover-run initial_program.py evaluator.py \
  --config config.yaml --search alphaevolve --iterations 100
```

### Config

See `defaults/alphaevolve_default.yaml` for the full template. Settings live in a top-level `alphaevolve:` section, and `ALPHAEVOLVE_*` environment variables override it.

```yaml
alphaevolve:
  project_id: "your-gcp-project"
  engine_id: "your-engine-id"
  location: "global"
  num_samplers: 1
  num_evaluators: 1        # > 1 runs evaluators in parallel threads
  idle_timeout_s: 120
  # credentials_file: "${HOME}/.config/sa.json"
```

`EVOLVE-BLOCK` markers are added automatically when the seed has none (`#` for Python, `//` for C/C++/CUDA). Only single-file programs are supported, and extra files in a candidate are ignored with a warning.

## Files

| File | Purpose |
|------|---------|
| `openevolve_backend.py` | Wrapper around OpenEvolve's island-model controller |
| `alphaevolve_backend.py` | Wrapper around Google's AlphaEvolve EAP SDK (cloud sampling, local evaluation) |
| `gepa_backend.py` | Wrapper around GEPA's `optimize_anything` API |
| `shinkaevolve_backend.py` | Wrapper around ShinkaEvolve's async evolution runner |
| `__init__.py` | Backend registry, loader, and package-name mapping |
| `defaults/` | Backend-specific YAML configs with tuned hyperparameters |
