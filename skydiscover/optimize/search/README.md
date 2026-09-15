# Search Algorithms

This is the core of skydiscover — where algorithms decide *which* programs to evolve and *how* to evolve them.

Every algorithm plugs into the same loop: `sample → prompt → LLM → evaluate → add`.
You only implement what changes; everything else is inherited.

There are two levels of customization:

| Level | What you build | When you need it |
|-------|---------------|-----------------|
| **Database only** | `add()` and `sample()` | Different parent selection or storage logic |
| **Database + Controller** | `run_discovery()` loop | Cross-iteration behavior: stagnation response, island rotation, acceptance gating |

---

## Level 1: Database only

Subclass `ProgramDatabase` and implement two methods. The default controller runs the loop unchanged.

```python
from skydiscover.optimize.search.base_database import Program, ProgramDatabase

class MyDatabase(ProgramDatabase):

    def __init__(self, name: str, config):
        # read any custom config attributes here, before super().__init__,
        # if they are needed during load() -> add() on startup
        super().__init__(name, config)

    def add(self, program: Program, iteration=None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)  # required
        return program.id

    def sample(self, num_context_programs=4, **kwargs):
        parent = ...        # pick the program to mutate from
        context = ...       # pick additional programs to show as examples
        return parent, context
```

Register in `route.py`:

```python
from skydiscover.optimize.search.my_algo.database import MyDatabase
register_database("my_algo", MyDatabase)
```

That is all. `--search my_algo` now works.

Examples at this level: `topk/` (56 lines), `best_of_n/` (85 lines), `beam_search/` (527 lines).

### `ProgramDatabase` helpers

| Method / attribute | What it does |
|--------------------|-------------|
| `self.programs` | `dict[str, Program]` storing all programs |
| `self._update_best_program(program)` | Updates the global best. Call this in every `add()`. |
| `self._save_program(program)` | Persists a program to disk (no-op if no db_path) |
| `self.get_top_programs(n)` | Returns top n programs by score |
| `self.get_best_program()` | Returns the highest-scoring program seen |

### `Program` fields

| Field | Type | Notes |
|-------|------|-------|
| `id` | `str` | UUID. Use as dict key. |
| `parent_id` | `str` or None | ID of the program this was mutated from |
| `solution` | `str` | Source code or prompt text |
| `metrics` | `dict` | Evaluation results, includes `combined_score` |
| `iteration_found` | `int` | Iteration that produced this program |
| `metadata` | `dict` | Arbitrary extra data |

---

## Level 2: Database + Controller

Use this when you need behavior that spans across iterations: tracking improvement history, reacting to stagnation, filtering results before they enter the population.

The key point: you do not rewrite the generate-evaluate logic. You call `_run_iteration()`, which runs the full `sample → prompt → LLM → evaluate` cycle, and then decide what to do with the result.

```python
from skydiscover.optimize.search.default_discovery_controller import DiscoveryController, DiscoveryControllerInput

class MyController(DiscoveryController):

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

    async def run_discovery(self, start_iteration, max_iterations, **kwargs):
        for iteration in range(start_iteration, start_iteration + max_iterations):
            if self.shutdown_event.is_set():
                break

            result = await self._run_iteration(iteration)

            if result.error:
                continue

            # optional: filter before storing
            self._process_iteration_result(result, iteration, kwargs.get("checkpoint_callback"))

            # optional: cross-iteration logic here

        return self.database.get_best_program()
```

Register both:

```python
register_database("my_algo", MyDatabase)
register_controller("my_algo", MyController)
```

Examples at this level: `adaevolve/` (multi-island UCB search), `gepa_native/` (acceptance gating + merge), `evox/` (co-evolves the search algorithm itself).

### Controller primitives

| Method | What it does |
|--------|-------------|
| `await self._run_iteration(iteration)` | Full single-step cycle. Returns `SerializableResult`. |
| `self._process_iteration_result(result, iteration, cb)` | Stores to DB, logs, triggers checkpoint. |
| `self.database.get_best_program()` | Returns the best program seen so far. |
| `self.shutdown_event.is_set()` | True when graceful shutdown is requested. |

`SerializableResult` fields: `error` (str or None), `child_program_dict` (dict or None), `parent_id` (str or None).

---

## Config dataclass (optional)

If your algorithm has custom settings, add a dataclass in `skydiscover/optimize/config.py`:

```python
@dataclass
class MyDatabaseConfig(DatabaseConfig):
    my_param: float = 1.0
```

Add it to `_DB_CONFIG_BY_TYPE`:

```python
"my_algo": MyDatabaseConfig,
```

Then users can set it in `config.yaml`:

```yaml
search:
  type: my_algo
  database:
    my_param: 2.0
```

---

## Registration Reference

All registrations happen in `route.py`:

```python
# Level 1: database only (uses default DiscoveryController)
register_database("topk",              TopKDatabase)
register_database("best_of_n",         BestOfNDatabase)
register_database("beam_search",       BeamSearchDatabase)
register_database("openevolve_native", OpenEvolveNativeDatabase)

# Level 2: database + controller
register_database("adaevolve",         AdaEvolveDatabase)
register_controller("adaevolve",       AdaEvolveController)

register_database("gepa_native",       GEPANativeDatabase)
register_controller("gepa_native",     GEPANativeController)

# Level 2: controller + dynamic database
register_controller("evox",            CoEvolutionController)
register_database("evox_meta",         SearchStrategyDatabase)
```

`get_discovery_controller()` in `route.py` looks up the controller registry: if a controller is registered for the search type it is used, otherwise the default `DiscoveryController` is returned.

---

## `ProgramDatabase` API

Override the abstract methods; the rest are inherited.

| Method | Abstract? | Purpose |
|--------|-----------|---------|
| `add(program, iteration)` | Yes | Store a scored program |
| `sample(num_context_programs)` | Yes | Select parent and context |
| `save(path, iteration)` | No | Checkpoint to disk |
| `load(path)` | No | Restore from checkpoint |
| `_update_best_program(program)` | No | Track best program (call from `add`) |
| `get_best_program()` | No | Return highest-scoring program |
| `get_top_programs(n)` | No | Return top N programs by score |
| `get(program_id)` | No | Retrieve by ID |
| `log_status()` | No | Log database summary |
| `get_statistics()` | No | Return stats dict for prompt context |
| `log_prompt(...)` | No | Store prompt and response for a program |

---

## Directory structure

```
search/
  base_database.py                 Program dataclass + ProgramDatabase ABC
  default_discovery_controller.py  Default loop + iteration primitives
  registry.py                      register_database / register_controller / register_program
  route.py                         --search flag to class mapping

  topk/                            Level 1: simplest example
  best_of_n/                       Level 1
  beam_search/                     Level 1

  adaevolve/                       Level 2: database + controller + components
  gepa_native/                     Level 2: database + controller
  evox/                            Level 2: co-evolutionary search

  utils/                           Shared: checkpointing, logging, serialization
```
