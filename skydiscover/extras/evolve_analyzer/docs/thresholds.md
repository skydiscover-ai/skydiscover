# Rating Thresholds Reference

All rating thresholds live in `config/default_config.yaml` under `algorithm_classes`. Each algorithm class overrides the cut-points for dimensions whose healthy operating range differs by algorithm type. There are three independent threshold parameters.

---

## `regression_frequency_thresholds: [t5, t4, t3, t2]`

**What it measures:** fraction of iterations where the best score *dropped* vs the previous iteration.

**Direction:** lower is better.

**How the 4 values map to ratings:**

| Condition | Rating |
|---|---|
| freq < t5 | 5 — Excellent |
| t5 ≤ freq < t4 | 4 — Good |
| t4 ≤ freq < t3 | 3 — Fair |
| t3 ≤ freq < t2 | 2 — Poor |
| freq ≥ t2 | 1 — Critical |

**Why it differs by class:** a 30% regression rate is healthy in a population-based run (selection pressure naturally produces drops) but pathological in serial refinement where every step is supposed to improve.

---

## `exploration_sdi_thresholds: [t5, t4, t3, t2]`

**What it measures:** Simpson's Diversity Index (SDI) of the search — how spread out the explored parameter/code space is. Range 0–1.

**Direction:** higher is better.

**How the 4 values map to ratings:**

| Condition | Rating |
|---|---|
| sdi > t5 | 5 |
| t4 ≤ sdi ≤ t5 | 4 |
| t3 ≤ sdi < t4 | 3 |
| t2 ≤ sdi < t3 | 2 |
| sdi < t2 | 1 |

**Why it differs by class:** population search naturally achieves high diversity (multiple simultaneous lines of descent); serial refinement is a single chain so the same SDI bar would be unfairly punishing.

---

## `convergence_thresholds`

The convergence dimension splits into two cases — whether a plateau was detected or not.

### `time_to_best_thresholds: [t5, t4]`

Used when *no plateau was detected*. Measures `time_to_best_fraction` (ttbf) — how far into the run (0–1) the all-time-best score was found. Higher means the run kept improving later, which is good.

| Condition | Rating |
|---|---|
| ttbf ≥ t5 | 5 |
| ttbf ≥ t4 | 4 |
| ttbf < t4 | 3 |

Example — `population_evolutionary: [0.60, 0.40]` means finding the best in the last 40% of the run is already Excellent (5), because population search distributes improvement across many parallel candidates and rarely has a single late breakthrough. `serial_refinement: [0.80, 0.60]` is stricter — it expects continuous improvement right up to the end.

### `plateau_fraction_thresholds: [t1, t2, t3]`

Used when *a plateau was detected*. Measures when the plateau onset occurs as a fraction of the total run (0 = immediately, 1 = end of run). Earlier plateau = worse.

| Condition | Rating |
|---|---|
| plateau_fraction ≤ t1 | 1 — Critical |
| plateau_fraction ≤ t2 | 2 — Poor |
| plateau_fraction ≤ t3 | 3 — Fair |
| plateau_fraction > t3 | 4 — Good |

`population_evolutionary` uses `[0.15, 0.30, 0.55]` — a plateau in the first 15% is Critical (vs 20% for default), reflecting that population search should maintain diversity longer before stagnating.

---

## Values by algorithm class

| Class | regression t5/t4/t3/t2 | exploration t5/t4/t3/t2 | ttb t5/t4 | plateau t1/t2/t3 |
|---|---|---|---|---|
| `population_evolutionary` | 0.15 / 0.30 / 0.50 / 0.70 | 0.70 / 0.50 / 0.30 / 0.10 | 0.60 / 0.40 | 0.15 / 0.30 / 0.55 |
| `serial_refinement` | 0.03 / 0.10 / 0.20 / 0.35 | 0.40 / 0.25 / 0.10 / 0.05 | 0.80 / 0.60 | 0.20 / 0.40 / 0.60 |
| `bayesian_optimization` | 0.05 / 0.15 / 0.30 / 0.50 | 0.50 / 0.30 / 0.15 / 0.05 | 0.70 / 0.50 | 0.20 / 0.40 / 0.60 |
