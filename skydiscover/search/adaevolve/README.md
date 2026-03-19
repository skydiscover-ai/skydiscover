# AdaEvolve: Adaptive LLM-Driven Zeroth-Order Optimization

Paper: [AdaEvolve](https://arxiv.org/abs/2602.20133)

## Idea

Evolutionary LLM search systems typically use fixed strategies — static exploration rates, rigid temperatures, fixed prompt templates — regardless of how the search is progressing. AdaEvolve treats discovery as a **non-stationary optimization** process and dynamically adapts search behavior based on observed improvement signals, analogous to how Adam/AdaGrad adapt learning rates in gradient-based optimization, but for gradient-free LLM-driven search.

## Three Levels of Adaptation

**Global — island scheduling.** Multiple subpopulations (islands) evolve in parallel. A UCB bandit with decayed rewards selects which island to allocate compute to. Recent breakthroughs boost an island's priority; old wins decay away.

**Local — exploration vs. exploitation.** Each island tracks an accumulated improvement signal `G` (exponential moving average of squared normalized deltas). Search intensity is computed as:

```
intensity = I_min + (I_max - I_min) / (1 + sqrt(G + ε))
```

High `G` (productive island) → low intensity → exploit. Low `G` (stagnating island) → high intensity → explore.

**Meta-guidance — paradigm breakthroughs.** When global improvement rate drops below a threshold, the LLM generates high-level strategy shifts (new algorithmic directions) that are injected into prompts to escape local optima.

## Algorithm

```
for each iteration:
    1. Select island via UCB (decayed rewards + exploration bonus)
    2. Compute search intensity from island's accumulated signal G
    3. Sample parent — high intensity → diversity sampling, low → top-fitness sampling
    4. Sample other relevant context programs (local island archive + global top programs)
    5. Generate candidate via LLM with mode-aware prompting, evaluate, store
    6. Every N iterations: migrate top programs between islands (ring topology)
    7. If globally stagnating: trigger paradigm breakthrough generation
```

## Code Structure

```
adaevolve/
├── controller.py              # AdaEvolveController — discovery loop, mode-aware prompting,
│                           #   paradigm injection, sibling context, error retry
├── database.py             # AdaEvolveDatabase — island population management,
│                           #   UCB-based island selection, migration (ring topology),
│                           #   adaptive sampling, dynamic island spawning
├── adaptation.py           # AdaptiveState — per-island G tracking, intensity formula
│                           # MultiDimensionalAdapter — UCB with decayed magnitude
│                           #   rewards, dual normalization (local for intensity,
│                           #   global for fair cross-island UCB comparison)
├── archive/
│   ├── unified_archive.py  # UnifiedArchive — quality-diversity archive per island,
│   │                       #   elite score = fitness + novelty, deterministic crowding
│   └── diversity.py        # DiversityStrategy — pluggable distance metrics
│                           #   (code-based, metric-based, hybrid)
└── paradigm/
    ├── generator.py        # ParadigmGenerator — LLM-based breakthrough idea generation
    └── tracker.py          # ParadigmTracker — improvement rate monitoring, stagnation
                            #   detection, active paradigm state management
```

## Usage

```bash
uv run skydiscover-run initial_program.py evaluator.py \
  --config config.yaml --search adaevolve --iterations 100
```

```python
from skydiscover import run_discovery
result = run_discovery(
    initial_program="initial_program.py",
    evaluator="evaluator.py",
    search="adaevolve",
    model="gpt-5",
    iterations=100,
)
```

## Config

See `configs/adaevolve.yaml` for the full template. Key settings:

```yaml
search:
  type: "adaevolve"
  database:
    num_islands: 2
    decay: 0.9
    intensity_min: 0.15
    intensity_max: 0.5
    migration_interval: 15
    migration_count: 5
    use_paradigm_breakthrough: true
    use_dynamic_islands: true
```

Ablation flags: `use_adaptive_search`, `use_ucb_selection`, `use_migration`, `use_unified_archive`, `use_paradigm_breakthrough`, `use_dynamic_islands` — set any to `false` to disable.

### Multiobjective Mode

AdaEvolve can run in scalar mode or explicit Pareto mode:

```yaml
search:
  type: "adaevolve"
  database:
    pareto_objectives: ["accuracy", "latency"]
    higher_is_better:
      accuracy: true
      latency: false
    fitness_key: "accuracy"          # optional scalar proxy for adaptive state
    pareto_objectives_weight: 0.4    # archive weight for Pareto percentile
```

When `pareto_objectives` is configured:
- parent selection can exploit the Pareto front directly
- global `best_program_id` becomes a deterministic representative of the current Pareto front
- logs and prompt guidance refer to the configured objectives instead of assuming `combined_score`

When `pareto_objectives` is omitted, AdaEvolve keeps the existing scalar `combined_score` behavior.
