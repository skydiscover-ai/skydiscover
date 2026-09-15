# OpenEvolve Native

Native port of [OpenEvolve](https://github.com/codelion/openevolve)'s
island-based MAP-Elites search as a SkyDiscover `ProgramDatabase` subclass.

All search logic -- sampling, MAP-Elites, archive, island migration -- is identical
to the reference. The entire algorithm fits inside `sample()` and `add()`.

Named `openevolve_native` (not `openevolve`) to avoid any confusion with the
external `openevolve` pip package used by `skydiscover/optimize/extras/external/`.

## SkyDiscover adaptations

Three minimal changes to fit SkyDiscover's `DiscoveryController` loop:

1. **`sample()` return type** -- returns `(Program, List[Program])`.
   The framework's `DiscoveryController` normalises both plain and dict-wrapped returns.

2. **`Program.solution` instead of `Program.code`** -- field rename, no logic change.

3. **Island rotation + generation + migration in `sample()`/`add()`** -- OpenEvolve's
   controller calls `next_island()`, `increment_island_generation()`, and
   `migrate_programs()` separately. We integrate these into `sample()` and `add()`
   since `DiscoveryController` doesn't call them.

## What is NOT ported

These are non-search concerns that don't affect the algorithm:

- **Novelty rejection** -- embedding-based + LLM-judge filtering. Disabled by default
  in OpenEvolve (no-op when `embedding_model` is unset). Use the external `openevolve`
  backend if needed.
- **`sample_from_island()`** -- thread-safe sampling for parallel workers. Not needed
  in sequential execution.
- **Artifact storage** -- uses SkyDiscover's artifact system instead.
- **Prompt construction** -- uses SkyDiscover's `DefaultContextBuilder`.

## Configuration

| Field | Default | OpenEvolve Default |
|---|---|---|
| `population_size` | 40 | 1000 |
| `archive_size` | 100 | 100 |
| `num_islands` | 5 | 5 |
| `exploration_ratio` | 0.2 | 0.2 |
| `exploitation_ratio` | 0.7 | 0.7 |
| `elite_selection_ratio` | 0.1 | 0.1 |
| `feature_dimensions` | ["complexity", "diversity"] | ["complexity", "diversity"] |
| `feature_bins` | 10 | 10 |
| `diversity_reference_size` | 20 | 20 |
| `migration_interval` | 10 | 50 |
| `migration_rate` | 0.1 | 0.1 |

All values are overridable in YAML config.
