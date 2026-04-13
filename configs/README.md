# ⚙️ Configuration

All settings are YAML. Environment variables can be referenced with `${VAR}` syntax in any string value.

---

## 📁 Available Config Files

| File | Search | Description |
|------|--------|-------------|
| **default.yaml** | Top-K | Minimal starting template — good for first experiments |
| **adaevolve.yaml** | AdaEvolve | Full multi-island config with adaptive intensity, migration, paradigm breakthroughs, and ablation flags |
| **evox.yaml** | EvoX | Co-evolving solution generation and search strategies |
| **openevolve_native.yaml** | OpenEvolve Native | Native port of OpenEvolve's island-based MAP-Elites search with ring migration |
| **llm_judge.yaml** | - | Demonstrates LLM-as-a-judge evaluation (uses gpt-4o-mini for both generation and judging) |
| **human_in_the_loop.yaml** | Top-K | Enables the live monitor dashboard and human-in-the-loop feedback |

Each file is a ready-to-copy template. Fill in the **system_message** with your problem description and you're good to go.

---

## 🔧 Parameter Reference

### Top-level

```yaml
max_iterations: 100        # total evolution iterations
checkpoint_interval: 10    # save checkpoint every N iterations
log_level: "INFO"          # DEBUG / INFO / WARNING
log_dir: null              # directory for logs (default: outputs/)
random_seed: 42
language: null             # auto-detected from initial program ("python", "cpp", "java", etc.)
                           # set to "image" for image-generation mode
file_suffix: ".py"         # output file extension, auto-set from initial program at runtime
diff_based_generation: true # LLM receives diffs instead of full programs
max_solution_length: 60000 # max characters in a program; longer programs are trimmed
```

### llm

```yaml
llm:
  temperature: 0.7
  top_p: 0.95
  max_tokens: 32000
  timeout: 600           # seconds per LLM request
  retries: 3
  retry_delay: 5         # seconds between retries
  random_seed: null
  reasoning_effort: null # "low" / "medium" / "high" for o-series models
```

**Model specification** — use `provider/model` or a bare name (auto-detected for known prefixes):

| Provider | Format | API key env var |
|----------|--------|-----------------|
| OpenAI | `gpt-5`, `o3-mini` | OPENAI_API_KEY |
| Gemini | `gemini/gemini-2.0-flash` | GEMINI_API_KEY or GOOGLE_API_KEY |
| Anthropic | `claude-sonnet-4-6` or `anthropic/claude-sonnet-4-6` | ANTHROPIC_API_KEY |
| DeepSeek | `deepseek-chat` or `deepseek/deepseek-chat` | DEEPSEEK_API_KEY |
| Mistral | `mistral-large` or `mistral/mistral-large` | MISTRAL_API_KEY |
| Ollama / vLLM | `ollama/llama3`, `vllm/my-model` | — |

<details>
<summary><b>Single model, multi-model pool, separate pools, and API override examples</b></summary>

**Single model (shorthand):**
```yaml
llm:
  primary_model: "gpt-5"
  primary_model_weight: 1.0
```

**Multi-model pool (weighted sampling):**
```yaml
llm:
  models:
    - name: "gpt-5"
      weight: 0.8
    - name: "anthropic/claude-opus-4-6"
      weight: 0.2
```

**Separate model pools** — by default all pools share `models`; override individually:
```yaml
llm:
  models:
    - name: "gpt-5"
  evaluator_models:   # used by LLM-as-a-judge (evaluator.use_llm_feedback)
    - name: "gpt-4o-mini"
  guide_models:       # used for paradigm breakthroughs and variation labels
    - name: "gpt-4o-mini"
```

**Override API base:**
```yaml
llm:
  api_base: "https://my-proxy.example.com/v1"
  api_key: "${MY_API_KEY}"
```

You can also set OPENAI_API_BASE or OPENAI_BASE_URL env vars to override the config globally.

</details>

### search

```yaml
search:
  type: "adaevolve" # evox | openevolve_native | beam_search | best_of_n | topk
  num_context_programs: 4   # context programs shown to LLMs as examples 
```

<details>
<summary><b>topk</b> — no extra settings</summary>

Always picks the single best program as parent and the next K as additional context programs.

</details>

<details>
<summary><b>adaevolve</b> — full settings</summary>

```yaml
search:
  type: "adaevolve"
  database:
    population_size: 20
    num_islands: 2

    # Adaptive intensity
    decay: 0.9              # EMA weight for accumulated signal G
    intensity_min: 0.15     # min intensity (exploitation)
    intensity_max: 0.5      # max intensity (exploration)

    # Migration
    migration_interval: 15  # migrate every N iterations
    migration_count: 5      # top programs to copy between islands

    # Archive diversity
    fitness_weight: 1.0     # fitness contribution to elite score
    novelty_weight: 0.0     # novelty contribution to elite score
    diversity_strategy: "code"  # "code" / "metric" / "hybrid"

    # Dynamic island spawning
    use_dynamic_islands: true
    max_islands: 5
    spawn_productivity_threshold: 0.015
    spawn_cooldown_iterations: 30

    # Paradigm breakthrough
    use_paradigm_breakthrough: true
    paradigm_window_size: 10
    paradigm_improvement_threshold: 0.12
    paradigm_num_to_generate: 3
    paradigm_max_uses: 2
    paradigm_max_tried: 10

    # Error retry
    enable_error_retry: true
    max_error_retries: 2

    # Ablation flags (set false to disable)
    use_adaptive_search: true   # G-based intensity; false → use fixed_intensity
    use_ucb_selection: true     # UCB island selection; false → round-robin
    use_migration: true
    use_unified_archive: true   # quality-diversity archive; false → simple list
```

</details>

<details>
<summary><b>evox</b></summary>

```yaml
search:
  type: "evox"
  database:
    auto_generate_variation_operators: true  # by default generate variation operator once 
```

</details>

<details>
<summary><b>beam_search</b></summary>

```yaml
search:
  type: "beam_search"
  database:
    beam_width: 5
    beam_selection_strategy: "diversity_weighted"  # diversity_weighted / stochastic / round_robin / best
    beam_diversity_weight: 0.3
    beam_temperature: 1.0
    beam_depth_penalty: 0.0
```

</details>

<details>
<summary><b>best_of_n</b></summary>

```yaml
search:
  type: "best_of_n"
  database:
    best_of_n: 5  # reuse the same parent for N iterations, then switch to current best
```

</details>

<details>
<summary><b>openevolve_native</b> — MAP-Elites + island-based evolutionary search</summary>

Native port of [OpenEvolve](https://github.com/codelion/openevolve)'s search algorithm.
Uses MAP-Elites quality-diversity grid per island with ring-topology migration.

```yaml
search:
  type: "openevolve_native"
  num_context_programs: 5
  database:
    num_islands: 5
    population_size: 40
    archive_size: 100
    exploration_ratio: 0.2          # P(explore) — random from current island
    exploitation_ratio: 0.7         # P(exploit) — archive elite, prefer current island
    # remaining 0.1 = P(random)    — any program in population
    elite_selection_ratio: 0.1      # fraction of additional context programs from top elites
    feature_dimensions: ["complexity", "diversity"]
    feature_bins: 10
    diversity_reference_size: 20
    migration_interval: 10          # migrate every N island-generations
    migration_rate: 0.1             # fraction of island to migrate
    random_seed: 42
```

See [`skydiscover/search/openevolve_native/README.md`](../skydiscover/search/openevolve_native/README.md) for architecture details.

</details>

### prompt

```yaml
prompt:
  system_message: |
    You are an expert coder helping to improve programs through evolution.

  # system_message can also be a path to a .txt file (relative to the config):
  # system_message: "system_prompt.txt"

  evaluator_system_message: |   # system message for the LLM judge
    You are a strict code quality judge. ...  # only used when evaluator.use_llm_feedback: true

  suggest_simplification_after_chars: 500  # threshold for program labeling in prompts
```

### evaluator

```yaml
evaluator:
  timeout: 360            # seconds before killing evaluate() subprocess
  max_retries: 3

  # Cascade evaluation: skip expensive full eval on low-scoring programs
  cascade_evaluation: true
  cascade_thresholds: [0.3, 0.6]

  # Prepend evaluator source code (or instruction.md for Harbor tasks)
  # to the LLM system message so the model can see how solutions are scored.
  inject_evaluator_context: false  # default false

  # LLM-as-a-judge
  use_llm_feedback: false
  llm_feedback_weight: 1.0  # relative weight of LLM score in combined_score
```

### agentic

Multi-turn agent that can read files and search the codebase before generating solutions.
Enable via `--agentic` on the CLI or `agentic=True` in `run_discovery()`. The codebase root
defaults to the initial program's directory; override it here if needed.

```yaml
agentic:
  enabled: false
  codebase_root: null      # defaults to initial program's directory when omitted
  max_steps: 5             # max tool-call turns per iteration
  per_step_timeout: 60.0   # seconds per tool call
  overall_timeout: 300.0   # total seconds for one agentic generation
  max_context_chars: 400000
  max_file_chars: 50000
  max_files_read: 20
  max_search_results: 50
```

### monitor

Live dashboard served over WebSocket.

```yaml
monitor:
  enabled: false
  port: 8765
  host: "0.0.0.0"
  max_solution_length: 10000

  # AI-generated run summaries
  summary_model: "gpt-5-mini"
  summary_api_key: null    # falls back to OPENAI_API_KEY
  summary_top_k: 3
  summary_interval: 0      # auto-generate every N programs (0 = manual only)
```

### human_feedback

```yaml
human_feedback_enabled: false
human_feedback_file: null       # path to a file containing feedback text
human_feedback_mode: "append"   # "append" or "replace"
```

---

## 🚀 Getting Started

**1. Pick a template** — copy one of the config files above into your project directory:

```bash
cp configs/evox.yaml my_config.yaml
```

**2. Fill in the system message** — this is the most important field. Tell the LLM what problem it's solving:

```yaml
prompt:
  system_message: |
    You are an expert at optimizing circle packing algorithms.
    Maximize the number of non-overlapping circles in a unit square.
```

**3. Run with your config:**

```bash
uv run skydiscover-run initial_program.py evaluator.py -c my_config.yaml -i 100
```

You can override any config value from the CLI — for example, switch the search algorithm or model without editing the YAML:

```bash
uv run skydiscover-run initial_program.py evaluator.py \
  -c my_config.yaml \
  --model gemini/gemini-3-pro-preview \
  -i 50
```
