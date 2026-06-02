# Evolve Experiment Analyzer

**Diagnostic tool for evolutionary code optimization experiments.**

Part of the [SkyDiscover](../../..) extras. After (or during) an evolve loop run, `evolve_analyzer` answers: *"What worked and what didn't?"* — using the same framing as deep-learning training diagnostics: convergence curve shape, exploration/exploitation balance, cost efficiency, stagnation root cause, and ceiling detection, all applied to evolutionary code optimization.

---

## Concept

Evolutionary code optimization frameworks (SkyDiscover, ShinkaEvolve, OpenEvolve) run an LLM in a loop: generate a mutation → evaluate it → keep the best. Each iteration produces a score. Over hundreds of iterations, patterns emerge that are hard to spot manually:

- **Stagnation** — the LLM is stuck in a local optimum, making the same kinds of changes that aren't working
- **Evaluator noise** — the scoring function is inconsistent, masking real improvement
- **Compliance drift** — mutations violate the EVOLVE-BLOCK contract, wasting iterations
- **Premature plateau** — the run kept going long after gains stopped, burning budget
- **Exploration collapse** — the LLM converged to one strategy and stopped trying others

`evolve_analyzer` runs 12 deterministic analyzers (zero LLM cost) to tag every iteration, then dispatches targeted LLM judges only where they add value — stagnation root cause analysis, evaluator artifact clustering, per-mutation quality assessment.

### How it works

```
Your run output (JSONL / SQLite / checkpoint dirs)
        │
        ▼
[Ingestion]  — normalises SkyDiscover / ShinkaEvolve / OpenEvolve / raw JSONL
        │
        ▼
[Quantitative Analyzers]  — 12 pure-function analyzers, zero LLM cost
  - stagnation
  - convergence
  - regression
  - efficiency
  - exploration
  - compliance
  - evaluator
  - search space
  - meta-analysis
  - ceiling
  - infrastructure
  - sub-metrics (per-component trajectory + primary driver identification)
        │
        ▼
[Qualitative Analyzers]  — targeted LLM judges (optional, cached)
  - Judge A: stagnation root cause
  - Judge B: evaluator artifact clusters
  - Judge C: mutation quality
  - Judge D: semantic compliance
  - Judge E: exploration structure
  - Judge F: reasoning trace coherence
        │
        ▼
[Report]  — per-dimension 1–5 ratings + evidence + recommendations
[Dashboard]  — Streamlit: score curve, stagnation panels, failure budget, ...
```

### Supported frameworks

| Framework | Input format |
|---|---|
| **SkyDiscover** | `checkpoint_N/` directory tree + `adaevolve_iteration_stats_*.jsonl` (preferred when present) |
| **ShinkaEvolve** | SQLite database (`iterations` table) |
| **OpenEvolve** | `evolution_trace.jsonl` or checkpoint dirs |
| **Any framework** | Standard JSONL (one record per iteration) |

---

## Installation

`evolve_analyzer` is bundled as a SkyDiscover extra. From the SkyDiscover repo root:

```bash
uv sync --extra evolve-analyzer
```

Set your LLM API key (only needed for qualitative judge steps):

```bash
export EVOLVE_ANALYZER_API_KEY="sk-..."   # used by all judge steps
```

The key is kept separate from your evolve framework's key so the analyzer runs on its own rate-limit quota.

---

## Getting Started

### Analyse your own run

**SkyDiscover** checkpoint directory:

```bash
uv run run-evolve-analysis postmortem \
    --source skydiscover \
    --path results/circle_packing/checkpoints/ \
    --provider openai \
    --model gpt-4o-mini \
    --stagnation-threshold 10 \
    --output-dir results/analysis/
```

**JSONL** (universal — works with any framework that can export iteration data):

```bash
uv run run-evolve-analysis postmortem \
    --source jsonl \
    --path path/to/my_run.jsonl \
    --provider openai \
    --model gpt-4o-mini \
    --output-dir results/my_run_analysis/
```

**ShinkaEvolve** SQLite database:

```bash
uv run run-evolve-analysis postmortem \
    --source shinkaevolve \
    --path path/to/run.db \
    --no-llm \
    --output-dir results/analysis/
```

**OpenEvolve** checkpoint or trace directory:

```bash
# evolution_trace.jsonl is auto-detected if it sits inside --path
uv run run-evolve-analysis postmortem \
    --source openevolve \
    --path path/to/openevolve_output/ \
    --provider anthropic \
    --model claude-3-5-haiku-20241022 \
    --output-dir results/analysis/

# Pass --trace-path explicitly when the trace lives elsewhere
uv run run-evolve-analysis postmortem \
    --source openevolve \
    --path path/to/openevolve_output/ \
    --trace-path path/to/evolution_trace.jsonl \
    --output-dir results/analysis/
```

**With an external baseline** (compare best score against an independently measured reference):

```bash
uv run run-evolve-analysis postmortem \
    --source openevolve \
    --path path/to/openevolve_output/ \
    --baseline-score 0.8984 \
    --baseline-metrics '{"mean_ttft_ms": 85.34, "p99_ttft_ms": 640.89, "cpu_hit_rate": 0.8307}' \
    --no-llm \
    --output-dir results/analysis/
```

The report will show `+X% vs external baseline` in the executive summary and per-metric `vs baseline` deltas in the Sub-Metrics dimension.

### View the report and launch the dashboard

```bash
# Show the text report
uv run run-evolve-analysis show-report --report-dir output/

# Launch the Streamlit dashboard
uv run dashboard-evolve-analysis --report-dir output/
# Press Ctrl+C in the terminal to stop the dashboard
```

### JSONL record format

If you're exporting from a custom framework, each line should be a JSON object with these fields (all optional except `iteration` and `child_score`):

```jsonc
{
  "iteration":          0,          // required — iteration index
  "child_score":        0.72,       // required — score of the mutated solution
  "parent_score":       0.68,       // score of the parent solution
  "score_delta":        0.04,       // computed automatically if omitted
  "evaluation_status":  "success",  // "success" | "timeout" | "crash"
  "format_valid":       true,       // was the mutation parseable?
  "mutation_type":      "diff",     // "diff" | "rewrite" | "local_search" | ...
  "model":              "gpt-4o-mini",
  "parent_code":        "...",      // enables compliance + exploration analysis
  "child_code":         "...",
  "diff":               "...",      // unified diff

  // Sub-metric analysis: flat or one-level nested dict of component scores.
  // One level of nesting is transparently unwrapped (e.g. metrics.mean_ttft_ms).
  // Metric names ending in _ms, _latency, eviction_rate, etc. are treated as
  // "lower is better"; all others are "higher is better".
  "evaluator_metrics":  {
    "cpu_hit_rate": 0.83,
    "ttft_ratio":   1.12,
    "metrics": { "mean_ttft_ms": 76.7, "p99_ttft_ms": 165.2 }
  },

  // Seed metrics for the sub-metric analyzer: parent_metrics of the first
  // iteration is used as the seed baseline for vs-seed improvement calculations.
  "parent_metrics":     { "cpu_hit_rate": 0.83, "metrics": { "mean_ttft_ms": 77.8 } },

  "evaluator_artifacts":{ "stderr": "...", "llm_feedback": "..." },
  "llm_tokens_used":    1240,       // enables cost efficiency analysis
  "llm_cost_usd":       0.0025,
  "timestamp":          1700000000.0,
  "island_id":          "island_0", // for island-based frameworks
  "reasoning_trace":    "..."       // enables meta-analysis
}
```

### Key CLI options

| Option | Default | Description |
|---|---|---|
| `--source` | — | `jsonl` · `skydiscover` · `shinkaevolve` · `openevolve` |
| `--path` | — | Path to the file or directory |
| `--no-llm` | off | Skip all qualitative judge steps (free, fast) |
| `--provider` | `openai` | LLM provider for judges |
| `--model` | `gpt-4o-mini` | Default model (high-value steps auto-upgrade to `gpt-4o`) |
| `--stagnation-threshold` | `10` | Consecutive non-improving iterations before alert |
| `--min-delta` | `0.001` | Minimum score improvement to count as progress |
| `--baseline-score` | none | External reference score (e.g. 5-run baseline average) for comparison |
| `--baseline-metrics` | none | JSON dict of baseline sub-metrics, e.g. `'{"mean_ttft_ms": 85.34}'` |
| `--trace-path` | none | OpenEvolve: explicit path to `evolution_trace.jsonl` (auto-detected if omitted) |
| `--historical-db` | none | Path to SQLite DB for cross-experiment comparison |
| `--output-dir` | `./evolve_analysis_output` | Where to write `report.json` and `report.txt` |
| `--experiment-id` | auto | Label for this run in the historical DB |
| `--config` | none | Path to a YAML config file (overrides defaults) |

### Output

Every run writes to `--output-dir`:

```
output/
├── report.json    — full structured report (load programmatically or feed to dashboard)
├── report.txt     — human-readable text report
└── report.md      — GitHub-flavored Markdown report
```

The text report looks like:

```
EXECUTIVE SUMMARY
Overall run quality is poor (average dimension rating 2.4/5) across 60 iterations
reaching a best score of 0.9306 (+3.59% vs external baseline of 0.8984). The biggest
concern is the Stagnation dimension (🔴 Critical): ...

AGGREGATE STATS
  Total iterations        : 60
  Successful evaluations  : 36
  Best score              : 0.930612
  Baseline score          : 0.8984
  vs baseline             : +3.59%
  Total duration          : 7.25 h

RUN PARAMETERS
  --source                : openevolve
  --path                  : path/to/openevolve_output/

LLM JUDGE
  Model                   : gpt-4o-mini (openai via https://api.openai.com/v1)
  Status                  : ✅ Connected successfully

Dimension:      Sub-Metrics
Rating:         5/5  ✅
Summary:        Strong sub-metric improvement — primary driver (p99_ttft_ms) improved ≥20% vs seed.
Evidence:
  • Primary driver: p99_ttft_ms (28.6% decrease vs seed)
  • p99_ttft_ms   seed=165.6  best=118.2 (iter 31)  −28.6% vs seed  [↓ better]  baseline=640.9 (−81.6% vs baseline)
  • mean_ttft_ms  seed=77.75  best=75.94 (iter 35)  −2.3% vs seed   [↓ better]  baseline=85.34 (−11.0% vs baseline)
  • cpu_hit_rate  seed=0.832  best=0.834 (iter 1)   +0.2% vs seed   [↑ better]  baseline=0.831 (+0.4% vs baseline)

Dimension:      Stagnation
Rating:         1/5  🔴
Summary:        Critical stagnation — at least one period classified as critical severity.
Evidence:
  • Streak streak_37: iters 37–ongoing, length=24, dominant_failure=crash
Critical stagnation details:
  Streak streak_37: iters 37–ongoing  len=24  failure=crash
    Recommendation: Immediate intervention: critical stagnation suggests systemic failure mode.
Recommendation: Immediate intervention: critical stagnation suggests systemic failure mode.
```

---

## Running with an internal LiteLLM proxy

If your organisation serves LLMs through a self-hosted [LiteLLM](https://github.com/BerriAI/litellm) proxy, use the ready-made config at `skydiscover/extras/evolve_analyzer/config/internal_litellm.yaml`.

A LiteLLM proxy exposes an OpenAI-compatible `/v1/chat/completions` endpoint, so the tool connects to it directly — all org-level access controls, audit logging, and cost attribution enforced by your proxy apply automatically.

### Edit the config

Open `skydiscover/extras/evolve_analyzer/config/internal_litellm.yaml` and set the two lines specific to your deployment:

```yaml
llm:
  model: "gemini-2.5-flash"                              # alias exposed by your proxy
  base_url: "https://litellm.internal.myorg.com/v1"     # your proxy URL (must end with /v1)
  api_key_env: "LITELLM_API_KEY"                         # env var holding your virtual key
```

The `overrides` block lets you route expensive judge steps to a stronger model alias while keeping the cheaper default for high-frequency steps:

```yaml
  overrides:
    stagnation_root_cause: { model: "gemini-2.5-pro" }  # fires once per stagnation period
    artifact_clustering:   { model: "gemini-2.5-pro" }  # fires once per batch of failures
    exploration_structure: { model: "gemini-2.5-pro" }  # fires once per run
    meta_quality:          { model: "gemini-2.5-pro" }  # fires once per run
    # mutation_quality and semantic_compliance use the cheap default
```

### Set your virtual key

```bash
export LITELLM_API_KEY="sk-..."    # your org's LiteLLM virtual key
```

### Run

```bash
uv run run-evolve-analysis postmortem \
    --source skydiscover \
    --path results/my_run/checkpoints/ \
    --config skydiscover/extras/evolve_analyzer/config/internal_litellm.yaml \
    --output-dir output/
```

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | Wrong or missing virtual key | Check `echo $LITELLM_API_KEY` |
| `404 Not Found` | Wrong `base_url` (missing `/v1` suffix) | Ensure URL ends with `/v1` |
| `Model not found` | Alias not registered on the proxy | Ask your LiteLLM admin for available aliases |
| Slow startup probe call | `_validate_temperature_support` warmup | Normal — one call to verify `temperature=0` is accepted |

---

## Configuration

Copy and edit `skydiscover/extras/evolve_analyzer/config/default_config.yaml` to customise thresholds, enable/disable individual judge steps, set per-step model overrides, or point to a historical database:

```yaml
stagnation:
  threshold: 10       # lower to catch stagnation earlier
  min_delta: 0.001

judges:
  stagnation_root_cause: true   # Judge A — infrequent, high-value
  artifact_clustering:  true    # Judge B
  mutation_quality:     false   # Judge C — disable to save cost
  semantic_compliance:  false   # Judge D — disable to save cost
  exploration_structure: true   # Judge E
  meta_quality:         true    # Judge F

llm:
  max_cost_usd: 5.0   # hard budget cap for all judge steps

historical:
  db_path: ~/.evolve_analyzer/history.db   # accumulates across experiments
```

Pass a custom config with `--config path/to/config.yaml`.

For a full explanation of every threshold parameter and how each maps to a 1–5 rating, see [thresholds.md](docs/thresholds.md).

---

## Architecture

```
skydiscover/extras/evolve_analyzer/
├── llm/                   # LLM layer (LiteLLM + diskcache)
│   ├── client.py
│   ├── cache.py
│   └── parallel.py
├── ingestion/             # per-framework adapters
│   └── checkpoint_adapter.py  # SkyDiscover / ShinkaEvolve / OpenEvolve / JSONL
├── quantitative/          # 12 pure-function analyzers + dataclasses
│   ├── bundle.py                  # all dataclasses (SubMetricStats, AggregateStats, …)
│   ├── sub_metric_analyzer.py     # per-component metric trajectories + primary driver
│   ├── convergence_analyzer.py
│   ├── stagnation_detector.py
│   ├── regression_analyzer.py
│   ├── efficiency_analyzer.py
│   ├── exploration_analyzer.py
│   ├── search_space_analyzer.py
│   ├── ceiling_analyzer.py
│   ├── meta_analyzer.py
│   ├── evaluator_analyzer.py
│   ├── infrastructure_analyzer.py
│   └── compliance_checker.py
├── qualitative/           # 6 LLM judge steps
│   └── qualitative_analyzer.py
├── historical_db.py       # SQLite cross-experiment store
├── report_synthesizer.py
├── coordinator.py         # analysis pipeline
├── cli.py                 # Click CLI (run-evolve-analysis / dashboard-evolve-analysis)
├── dashboard.py           # Streamlit dashboard
└── config/
    ├── default_config.yaml
    └── internal_litellm.yaml  # pre-built config for org-hosted LiteLLM proxy
```

Tests live in `tests/extras/evolve_analyzer/`, mirroring this structure.

The quantitative analyzers are **pure functions** on `List[dict]` — no side effects, no LLM calls — making them fast, testable, and safe to run in any context.

---

## Development

### Setup

From the SkyDiscover repo root:

```bash
uv sync --extra evolve-analyzer --extra dev
```

### Running tests

```bash
uv run pytest tests/extras/evolve_analyzer/ -v
```

### Where to add tests

| What you changed | Where to add tests |
|---|---|
| A quantitative analyzer | `tests/extras/evolve_analyzer/quantitative/` |
| A qualitative judge step | `tests/extras/evolve_analyzer/qualitative/` |
| An ingestion adapter | `tests/extras/evolve_analyzer/ingestion/` |

### Adding a new ingestion adapter

1. Add a new `adapt_<framework>(path) -> Iterator[dict]` function in `ingestion/checkpoint_adapter.py`.
2. Register the new `--source` choice in `cli.py` (`postmortem` command).
3. Add at least one test that ingests sample data and checks the record schema.

### Adding a new quantitative analyzer

1. Create `quantitative/<name>_analyzer.py` with a single pure function `analyze(records: list[dict]) -> <ResultDataclass>`.
2. Add the result dataclass to `quantitative/bundle.py`.
3. Call the analyzer and attach the result in `coordinator.py`.
4. Add a dimension entry in `report_synthesizer.py`.
5. Add tests in `tests/extras/evolve_analyzer/quantitative/test_<name>_analyzer.py`.

### Guidelines

- **No LLM calls in tests** — mock the LLM layer (`unittest.mock.patch`) so the suite runs without API keys.
- **Keep the quantitative analyzers pure** — no side effects, no I/O, no LLM calls inside `quantitative/`.
