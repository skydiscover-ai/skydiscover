# EvoX: Self-Evolving Search Strategies

Paper: [EvoX](https://arxiv.org/abs/2602.23413)

## Idea

Evolutionary LLM search systems rely on fixed strategies — static explore–exploit ratios, rigid sampling heuristics, predefined selection logic — that remain unchanged as the search landscape shifts. EvoX treats the search strategy itself as a **co-evolving program**, jointly evolving candidate solutions and the search algorithms used to generate them, analogous to learning a learning rate schedule but for the entire search procedure.

## Two Levels of Co-Evolution

**Inner — solution discovery.** A solution database stores candidates and defines sampling (parent + context programs selection). The LLM generates new candidates; the evaluator scores them. The database implementation is not fixed — it gets hot-swapped by the outer loop.

**Outer — search strategy evolution.** When solutions stagnate for `switch_interval` iterations (~10% of total), the LLM generates an entirely new `EvolvedProgramDatabase` class (as Python code) — a new sampling/selection strategy. The new strategy is validated, all programs are migrated, and solution discovery resumes. Search algorithms are scored by:

```
score = improvement * (1 + log(1 + start_score)) / sqrt(horizon)
```

Higher `start_score` → larger weight → algorithms that improve an already-strong solution are rewarded more.

## Algorithm

```
for each iteration:
    1. Sample parent + context programs from current solution database
    2. Generate solution candidate via LLM, evaluate, store
    3. Track stagnation — if improvement < 0.01 for switch_interval consecutive iterations:
        a. Score current search algorithm by solution improvement in its window
        b. Generate new EvolvedProgramDatabase class via LLM
        c. Validate new database (structural + functional checks)
        d. Migrate all programs to new database, resume
    4. If new database fails at runtime: restore previous database, preserve new programs
```

## Code Structure

```
evox/
├── coevolve_controller.py              # CoEvolutionController — solution + search co-evolution,
│                                    #   stagnation detection, database hot-swapping,
│                                    #   fallback/restore on failure
├── database/
│   ├── initial_search_strategy.py   # EvolvedProgramDatabase — seed solution database
│   │                                #   (the code the LLM evolves into new strategies)
│   ├── search_strategy_db.py        # SearchStrategyDatabase — stores evolved search
│   │                                #   strategy programs, samples best for refinement
│   └── search_strategy_evaluator.py # Validates LLM-generated database implementations
│                                    #   (structural checks, metric preservation tests)
├── config/
│   ├── search.yaml                  # Search-side evolution config (loaded automatically)
│   └── evox_search_sys_prompt.txt   # System prompt for search strategy generation
└── utils/
    ├── search_scorer.py             # LogWindowScorer — log-weighted scoring of search
    │                                #   algorithms by solution improvement over horizon
    ├── variation_operator_generator.py # variation operator that determines how to present candidate
    ├── template.py                  # Default variation operator templates
    └── coevolve_logging.py          # Artifact saving (code, metadata, failed attempts)
```

## Usage

```bash
uv run skydiscover-run initial_program.py evaluator.py \
  --config config.yaml --search evox --iterations 100
```

```python
from skydiscover import run_discovery
result = run_discovery(
    initial_program="initial_program.py",
    evaluator="evaluator.py",
    search="evox",
    model="gpt-5",
    iterations=100,
)
```

## Config

See `configs/evox.yaml` for the full template. Key settings:

```yaml
search:
  type: "evox"
  database:
    auto_generate_variation_operators: true
    # Opt-in multiobjective (same knobs as AdaEvolve / issue #42):
    # pareto_objectives: ["accuracy", "latency"]
    # higher_is_better: {accuracy: true, latency: false}
    # fitness_key: accuracy
```

Ablation flag: `auto_generate_variation_operators` — set to `false` to use default templates for variation operator instead of LLM-generated ones.

### Multiobjective mode

When `pareto_objectives` is configured, the solution database:

- Maintains a lazy-cached global Pareto front
- Samples parents preferentially from the front
- Uses `compute_proxy_score` / `fitness_key` only for representative-best and search-window scoring
- Leaves scalar behaviour unchanged when the list is empty
