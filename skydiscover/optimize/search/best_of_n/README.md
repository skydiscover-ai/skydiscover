# Best-of-N

Generates N independent mutations from the same parent before switching parents. This gives the search N attempts to improve a single program, after which it commits to the current global best and repeats.

## Algorithm

```
sample():
    if parent_iteration_count >= N or no current parent:
        parent = argmax combined_score across all programs   # switch to best
        reset counter
    else:
        parent = current_parent                              # reuse
        increment counter
    other context programs = random sample from top programs (excluding parent)
```

Other context programs are re-sampled from the top pool every iteration regardless of the parent reuse cycle.

## Config

```yaml
search:
  type: "best_of_n"
  database:
    best_of_n: 5   # number of iterations to reuse the same parent
```

## When to use

Useful when single-step improvements are noisy and you want to give the model multiple shots at the same problem state before moving on. Larger `best_of_n` increases exploitation depth; smaller values approach Top-K behavior.
