# Top-K

The simplest search strategy. At every iteration, the best-scoring program is selected as the parent and the next K highest-scoring programs are provided as other context programs.

## Algorithm

```
sample():
    parent      = programs sorted by score, rank 1
    other context = programs sorted by score, ranks 2..K+1
```

No population cap is enforced. All programs ever generated are stored and sorted by `combined_score` on each call.

## Config

No algorithm-specific fields. Uses base `DatabaseConfig` only:

```yaml
search:
  type: "topk"
  num_context_programs: 4
```

## When to use

Reliable baseline. Deterministic and well-understood. Tends to exploit the current best heavily — useful for verifying that a new benchmark or evaluator works before running adaptive search.
