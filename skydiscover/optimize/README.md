# SkyDiscover-Optimize

<p align="center">
  <img src="../../assets/architecture.png" width="720" alt="SkyDiscover-Optimize architecture: a control loop of solution selector, context builder, solution generator, and evaluator">
</p>

**Pick anything you can score, an algorithm, a prompt, a GPU kernel, a component of your system,
and SkyDiscover-Optimize evolves it.** You write an evaluator that scores a solution; the control
loop proposes candidates, scores them against it, and keeps what scores better. Each iteration
runs five steps:

`sample → prompt → generate → evaluate → add`

1. **Sample**: the solution selector picks a parent solution (and any useful context) from the solution database.
2. **Prompt**: the context builder turns the parent, problem context, and any human guidance into an LLM prompt.
3. **Generate**: the solution generator produces a candidate solution (code, text, or image).
4. **Evaluate**: your evaluator scores the candidate and returns metadata: score, logs, feedback, artifacts.
5. **Add**: the scored candidate is stored back in the solution database, closing the loop.

This guide covers writing an evaluator, choosing a search algorithm, configuration, and the full
result tables.

← [Back to the SkyDiscover overview](../../README.md) · [Synthesize guide](../synthesize/README.md)

## 🚀 Optimization Quick Start

**Prerequisites:** Python >= 3.10, [uv](https://docs.astral.sh/uv/)

```bash
# Install
uv sync
export OPENAI_API_KEY="<your-key>"

# Try the circle packing benchmark (swap --search adaevolve to try the other SOTA method)
uv sync --extra math
uv run skydiscover optimize benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  --config benchmarks/math/circle_packing/config.yaml \
  --search evox \
  --iterations 100

# Or run on your own problem
# --search: evox, adaevolve, topk, beam_search, best_of_n, openevolve, gepa, shinkaevolve
# (external backends: openevolve, gepa via `uv sync --extra external`; shinkaevolve via manual install, see External backends)
uv run skydiscover optimize initial_program.py evaluator.py \
  --search <algo> \
  --model gpt-5 \
  --iterations 100

# initial_program is optional; omit it to let the LLM start from scratch
uv run skydiscover optimize evaluator.py \
  --search <algo> \
  --model gpt-5 \
  --iterations 100

# Run a Harbor benchmark (e.g. AlgoTune), no seed program needed
uv tool install --python 3.12 harbor   # the Harbor CLI needs Python >= 3.12
harbor datasets download algotune@1.0 -o /tmp/algotune   # one dir per task: /tmp/algotune/<id>/… (<id> is a per-task hash)
uv run skydiscover optimize /tmp/algotune/<id>/algotune-set-cover \
  --model anthropic/claude-sonnet-4-6 \
  --search best_of_n -i 10
```

Or use the Python API:

```python
from skydiscover import run_discovery

result = run_discovery(
    initial_program="initial_program.py",
    evaluator="evaluator.py",
    search="adaevolve",  # or "evox", "topk", "beam_search", "best_of_n", "openevolve", "gepa", "shinkaevolve"
    #                      (openevolve, gepa need: uv sync --extra external; shinkaevolve needs a manual install)
    model="gpt-5",
    iterations=100,
)

print(result.best_score, result.best_solution)
```


## ✏️ What You Write

### Evaluator (required)

SkyDiscover supports three evaluator formats; pick whichever fits your use case:

| Format | When to use | What you point `evaluation_file` at |
|:---|:---|:---|
| **Python function** | Simple tasks, no system deps | `evaluator.py` |
| **Containerized** | Custom deps, data files, isolation | `evaluator/` directory (must contain `Dockerfile` + `evaluate.sh`) |
| **Harbor task** | External benchmark suites (AlgoTune, EvoEval, HumanEvalFix, BigCodeBench, LiveCodeBench, USACO, CRUSTBench, CodePDE, and more) | Task directory (must contain `instruction.md` + `tests/` + `environment/Dockerfile`) |

SkyDiscover auto-detects the format. See [`benchmarks/README.md`](../../benchmarks/README.md#adding-a-benchmark) for full setup instructions.

**Python evaluator**: a file with an `evaluate(program_path)` function:

```python
def evaluate(program_path):
    score = run_and_grade(program_path)
    return {
        "combined_score": score,       # primary optimization target (maximized)
        "artifacts": {                 # optional; stored with the solution for future context
            "feedback": "Off by one in the loop boundary",
        },
    }
```

**Containerized evaluator**: a directory with a `Dockerfile` and `evaluate.sh` that writes JSON to stdout. Runs in Docker, so it can have arbitrary dependencies.

**Harbor task**: a directory following the [Harbor](https://harborframework.com/) format (`instruction.md`, `environment/Dockerfile`, `tests/test.sh`). Works out of the box with 8+ tested benchmark suites (see [benchmarks/README.md](../../benchmarks/README.md#tested-harbor-datasets) for the full list).

- **combined_score** drives evolution. If omitted, SkyDiscover averages all numeric values in the dict.
- **artifacts** is optional; entries are injected into the next LLM prompt as context.

For `search.type: adaevolve`, you can also enable explicit Pareto optimization by configuring `search.database.pareto_objectives` and returning those objective metrics directly from the evaluator. In that mode, `combined_score` becomes optional and is used only as a scalar fallback.

### Starting Solution (optional)

The initial program is **optional**. When omitted, the LLM generates a solution from scratch. If provided, it marks the region to mutate with EVOLVE-BLOCK markers. Everything outside is left untouched.

```python
# EVOLVE-BLOCK-START
def solve(input_data):
    return input_data  # baseline; SkyDiscover will improve this
# EVOLVE-BLOCK-END
```

If no markers are present, the entire file is treated as mutable.


## 🧬 Pick an Algorithm

See [Benchmarks and Results](#-benchmarks-and-results) for a detailed comparison of AdaEvolve and EvoX against other algorithms.

| Algorithm | Flag | Description |
|:---|:---|:---|
| ⭐&nbsp;**AdaEvolve** | `--search adaevolve` | Multi-island adaptive search with UCB, migration, and paradigm breakthroughs |
| 🧠&nbsp;**EvoX** | `--search evox` | Self-evolving paradigm that co-adapts solution generation and experience management |
| 📊&nbsp;**Top-K** | `--search topk` | Selects top-K solutions to refine |
| 🔍&nbsp;**Beam&nbsp;Search** | `--search beam_search` | Breadth-first expansion of a beam of top solutions |
| 🎲&nbsp;**Best-of-N** | `--search best_of_n` | Generates N variants per iteration, keeps the best |
| 🧪&nbsp;**GEPA&nbsp;Native** | `--search gepa_native` | Pareto-efficient search with reflective prompting and LLM-mediated merge |
| 🗺️&nbsp;**OpenEvolve&nbsp;Native** | `--search openevolve_native` | MAP-Elites + island-based evolutionary search |
| 🤖&nbsp;**Coding-Agent&nbsp;Baseline** | `--search claude_code` | Single-agent baseline running a coding-agent CLI in a Docker container |

### External backends

Most install with `uv sync --extra external`, then use the corresponding flag (the table notes the two exceptions: ShinkaEvolve is a manual install, AlphaEvolve needs GCP):

| Backend | Flag | Source |
|:---|:---|:---|
| **OpenEvolve** | `--search openevolve` | [codelion/openevolve](https://github.com/codelion/openevolve) |
| **GEPA** | `--search gepa` | [gepa-ai/gepa](https://github.com/gepa-ai/gepa) |
| **ShinkaEvolve** | `--search shinkaevolve` | [SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (manual install) |
| **AlphaEvolve** | `--search alphaevolve` | [Google Cloud AlphaEvolve](https://github.com/Google-Cloud-AI/alphaevolve-on-googlecloud) (needs GCP, see below) |

<details>
<summary>ShinkaEvolve manual install</summary>

```bash
git clone --depth 1 https://github.com/SakanaAI/ShinkaEvolve.git external_repos/ShinkaEvolve
uv pip install -e external_repos/ShinkaEvolve
```

</details>

<details>
<summary>AlphaEvolve on Google Cloud</summary>

[AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
runs as a hosted service on Google Cloud's Discovery Engine, so it needs a GCP project and an
engine ID rather than a local model. Point it at your instance with environment variables:

```bash
export ALPHAEVOLVE_PROJECT_ID="your-gcp-project"   # Discovery Engine API must be enabled
export ALPHAEVOLVE_ENGINE_ID="your-engine-id"
```

…or an `alphaevolve:` section in your config. Environment variables win over the config file,
which in turn wins over the shipped defaults:

```yaml
search:
  type: "alphaevolve"

alphaevolve:
  project_id: "your-gcp-project"
  engine_id: "your-engine-id"
  location: "global"            # default
  num_samplers: 1               # concurrent sampling workers
  num_evaluators: 1             # raise above 1 for slow compile/benchmark evaluators,
                                # which would otherwise starve the samplers
  # credentials_file: "${HOME}/.config/sa.json"   # overrides ambient ADC
```

Then run:

```bash
uv run skydiscover optimize initial_program.py evaluator.py --search alphaevolve -i 100
```

Credentials resolve from ambient [Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc)
unless `credentials_file` is set. Every option is documented in
[`extras/external/defaults/alphaevolve_default.yaml`](extras/external/defaults/alphaevolve_default.yaml).

</details>


## ⚙️ Configuration

Pass a YAML config with `-c`. See [`configs/`](configs/) for full annotated templates.

```yaml
max_iterations: 100
llm:
  models: [{ name: "gemini/gemini-3-pro-preview", weight: 1.0 }]
search:
  type: "adaevolve"                  # or "evox", "topk", "beam_search", "best_of_n"
prompt:
  system_message: |
    You are an expert at optimizing algorithms.
```

API keys (OPENAI_API_KEY, GEMINI_API_KEY, etc.) are resolved from environment variables automatically.

### 📊 Live Monitor & Human Feedback

Add `monitor: { enabled: true }` to your config. The dashboard URL prints at run start: a scatter plot of all programs, code diffs, metrics, and AI summaries. A **Human Feedback** panel lets you steer evolution in real time.
Replay a completed run:

```bash
uv run skydiscover viewer /path/to/checkpoints/checkpoint_100
```


## 📖 Reference

<details>
<summary><b>CLI flags</b></summary>

```
uv run skydiscover optimize [INITIAL_PROGRAM] EVALUATOR [options]
```

| Flag | Description |
|:---|:---|
| `-c, --config FILE` | Config YAML |
| `-i, --iterations N` | Number of iterations |
| `-m, --model MODEL` | LLM model (overrides config) |
| `-s, --search TYPE` | Search algorithm |
| `-o, --output DIR` | Output directory |
| `--api-base URL` | Override LLM API endpoint |
| `--checkpoint DIR` | Resume from checkpoint |
| `--agentic` | Enable agentic mode (LLM can read your files) |
| `-l, --log-level LEVEL` | DEBUG, INFO, WARNING, ERROR, or CRITICAL |

</details>

<details>
<summary><b>Python API: discover_solution() (convenience wrapper)</b></summary>

`discover_solution()` is a convenience wrapper around `run_discovery()` (shown in [Quick Start](#-optimization-quick-start)) for inline string solutions and callable evaluators:

```python
from skydiscover import discover_solution

result = discover_solution(
    initial_solution="def solve(x): return x",  # optional; omit to start from scratch
    evaluator=lambda path: {"combined_score": run_tests(path)},
    iterations=50,
    search="evox",
)
```

</details>

<details>
<summary><b>Model providers</b></summary>

Any OpenAI-compatible endpoint works. Built-in `provider/model` shortcuts resolve the endpoint
and API-key environment variable for OpenAI, Azure, Gemini, Anthropic, DeepSeek, Mistral, Cohere,
Hugging Face, and local servers (Ollama, vLLM, FreeToken); anything else takes `--api-base`:

```bash
--model gpt-5                                               # OpenAI (default)
--model gemini/gemini-3-pro-preview                          # Gemini
--model anthropic/claude-sonnet-4-6                          # Anthropic
--model ollama/llama3 --api-base http://localhost:11434/v1   # Local (Ollama, vLLM, etc.)
--model freetoken/Qwen3.6-35B-A3B                            # On-device via FreeToken
```

**On-device via [FreeToken](https://github.com/FlashML-org/FreeToken)** runs fully
local, no API key. Serve a model, then use the `freetoken/` prefix; the endpoint
(`http://127.0.0.1:1919/v1`) and key are set for you:

```bash
uv pip install "freetoken[accel]"
ft serve --model Qwen3.6-35B-A3B          # OpenAI-compatible server on 127.0.0.1:1919

uv run skydiscover optimize initial_program.py evaluator.py \
  --model freetoken/Qwen3.6-35B-A3B --search evox --iterations 100
```

Add `--api-base http://<host>:<port>/v1` if the server runs elsewhere.

Multi-model pools with weighted sampling are supported in config:

```yaml
llm:
  models:
    - name: "gpt-5-mini"
      weight: 0.7
    - name: "gemini/gemini-2.0-flash"
      weight: 0.3
```

</details>

<details id="dependency-extras">
<summary><b>Benchmark dependency extras</b></summary>

```bash
uv sync                              # Base install
uv sync --extra math                 # Math benchmarks (SciPy, JAX, PyWavelets, …)
uv sync --extra adrs                 # ADRS systems benchmarks
uv sync --extra frontier-cs          # Frontier-CS benchmark tooling
uv sync --extra external             # OpenEvolve / GEPA backends (ShinkaEvolve is a manual install)
uv sync --extra prompt-optimization  # HotPotQA prompt optimization
```

Combine extras as needed: `uv sync --extra external --extra math`

If a benchmark ships its own `requirements.txt`, also run: `uv pip install -r path/to/requirements.txt`

</details>

## 📊 Benchmarks and Results

SkyDiscover-Optimize has been used across industry including Google and Uber; the algorithms released by the SkyDiscover team, AdaEvolve and EvoX, achieve the strongest open-source results across 200+ optimization benchmarks, matching or exceeding AlphaEvolve and human-designed baselines, and outperforming OpenEvolve, GEPA, and ShinkaEvolve under identical generation budgets.

- **Frontier-CS (172 problems)**: ~34% median score improvement over OpenEvolve, GEPA, and ShinkaEvolve  
- **Math + Systems Optimization (14 tasks evaluated)**: Matches or exceeds AlphaEvolve and human-designed SOTA on 6/6 systems and 6/8 math tasks
- **Real-world systems impact**: 41% lower cross-cloud transfer cost, 14% better GPU load balance for MoE serving, and 29% lower KV-cache pressure via GPU model placement

<details>
<summary><b>📊 Complete results of AdaEvolve and EvoX (100 iterations)</b></summary>

> AdaEvolve and EvoX are **complementary**: AdaEvolve adapts search *parameters* for fast early gains; EvoX evolves the search *strategy itself* for stronger long-horizon gains. Both are built on SkyDiscover.

<p align="center">
  <img src="../../assets/comparison.png" width="900" alt="Main results for systems and math problems">
</p>

</details>

<details>
<summary><b>📈 Scaling behavior of AdaEvolve and EvoX</b></summary>

The scaling behavior of AdaEvolve and EvoX shows a **complementary crossover**. AdaEvolve's per-iteration parameter adaptation yields fast early gains in low-budget runs (T≤50), while EvoX's demand-driven strategy evolution unlocks step-change improvements in longer runs (T≥50).

<p align="center">
  <img src="../../assets/scaling_comparison.png" width="900" alt="Scaling behavior of AdaEvolve vs EvoX across 500 iterations">
  <br><em>Best-so-far score vs. iteration for Signal Processing, Heilbronn Convex, Prism, and Cloudcast (500 iterations, GPT-5).</em>
</p>

</details>

<details>
<summary><b>🔗 Evolving AdaEvolve's policy with EvoX (coming soon)</b></summary>

The two methods are **composable**: EvoX can evolve using AdaEvolve as its starting strategy, achieving the best results on 3 out of 4 benchmarks (100 iterations, GPT-5).

| Benchmark | AdaEvolve | EvoX (Random Init) | EvoX (AdaEvolve Init) |
|:--|--:|--:|--:|
| Signal Proc. (↑) | 0.718 | 0.721 | **0.760** |
| Heilbronn Cvx. (↑) | 0.0290 | 0.0270 | **0.0291** |
| Cloudcast (↓) | 640.5 | 637.1 | **623.4** |
| Prism (↑) | 26.37 | **30.52** | 26.27 |

</details>

<details>
<summary><b>Task breakdown across math, systems, and programming challenges</b></summary>

| | Benchmark | Domain | Tasks | Description |
|-|-----------|--------|------:|-------------|
| 🔢 | [math/](../../benchmarks/math/) | Math | 14 | Circle packing, Erdos problems, geometric optimization |
| 🖥️ | [ADRS/](../../benchmarks/ADRS/) | Systems | 5 | Cloud scheduling, load balancing, MoE expert placement |
| ⚡ | [gpu_mode/](../../benchmarks/gpu_mode/) | Systems | 4 | GPU kernel optimization |
| 🔧 | [kernelbench/](../../benchmarks/kernelbench/) | Systems | 250+ | [KernelBench](https://github.com/ScalingIntelligence/KernelBench) GPU kernel speedup optimization |
| 🧩 | [frontier-cs-eval/](../../benchmarks/frontier-cs-eval/) | Algorithms | 172 | [Frontier-CS](https://frontier-cs.org/) competitive programming |
| 🧠 | [arc_benchmark/](../../benchmarks/arc_benchmark/) | Reasoning | n/a | ARC-AGI visual reasoning |
| 💻 | [ale_bench/](../../benchmarks/ale_bench/) | Algorithms | 10 | Algorithmic programming contests |
| 🎨 | [image_gen/](../../benchmarks/image_gen/) | Creative | 1 | AI image generation evolution |
| 💬 | [prompt_optimization/](../../benchmarks/prompt_optimization/) | NLP | 1 | HotPotQA prompt evolution |

See [Dependency extras](#dependency-extras) for install commands per benchmark.

</details>

## 🛠️ Extending SkyDiscover

- **New benchmark** → [`benchmarks/README.md`](../../benchmarks/README.md#adding-a-benchmark)
- **New search algorithm** → [`skydiscover/optimize/search/README.md`](search/README.md)
- **New context builder** → [`skydiscover/optimize/context_builder/README.md`](context_builder/README.md)

## 🔗 Related Work

SkyDiscover-Optimize is inspired by [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) and incorporates useful code components from open-source efforts such as [OpenEvolve](https://github.com/codelion/openevolve). Its interface is compatible with the [optimize_anything](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/) API.

## ✍️ Citation

If you use SkyDiscover-Optimize, please cite the framework paper:

```bibtex
@inproceedings{liu2026skydiscover,
  author    = {Liu, Shu and Cemri, Mert and Agarwal, Shubham and Krentsel, Alexander and Naren, Ashwin and Mang, Qiuyang and Li, Zhifei and Gupta, Akshat and Maheswaran, Monishwaran and Cheng, Audrey and Pan, Melissa and Boneh, Ethan and Ramchandran, Kannan and Sen, Koushik and Zaharia, Matei and Dimakis, Alexandros G. and Stoica, Ion},
  title     = {SkyDiscover: A Flexible, Adaptive Framework for AI-Driven Scientific and Algorithmic Discovery},
  booktitle = {Proceedings of the ACM Conference on AI and Agentic Systems},
  series    = {CAIS '26},
  year      = {2026},
  pages     = {1223--1227},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3786335.3813221},
  url       = {https://doi.org/10.1145/3786335.3813221}
}
```

<details>
<summary><b>Citations for AdaEvolve and EvoX</b></summary>

If you use the **AdaEvolve** search algorithm:

```bibtex
@misc{cemri2026adaevolve,
  author        = {Mert Cemri and Shubham Agrawal and Akshat Gupta and Shu Liu and Audrey Cheng and Qiuyang Mang and Ashwin Naren and Lutfi Eren Erdogan and Koushik Sen and Matei Zaharia and Alex Dimakis and Ion Stoica},
  title         = {AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization},
  year          = {2026},
  eprint        = {2602.20133},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NE},
  url           = {https://arxiv.org/abs/2602.20133}
}
```

If you use the **EvoX** search algorithm:

```bibtex
@misc{liu2026evox,
  author        = {Shu Liu and Shubham Agarwal and Monishwaran Maheswaran and Mert Cemri and Zhifei Li and Qiuyang Mang and Ashwin Naren and Ethan Boneh and Audrey Cheng and Melissa Z. Pan and Alexander Du and Kurt Keutzer and Alvin Cheung and Alexandros G. Dimakis and Koushik Sen and Matei Zaharia and Ion Stoica},
  title         = {EvoX: Meta-Evolution for Automated Discovery},
  year          = {2026},
  eprint        = {2602.23413},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2602.23413}
}
```

</details>
