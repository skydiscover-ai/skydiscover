# Beam Search

Maintains a fixed-width beam of the most promising programs. At each iteration the parent is selected from the beam using one of four strategies; the beam is then pruned back to `beam_width` by fitness and optional diversity.

## Algorithm

```
add(program):
    add to beam
    if len(beam) > beam_width:
        prune: keep top beam_width by fitness (+ diversity bonus if beam_diversity_weight > 0)

sample():
    parent = select from beam using beam_selection_strategy
    other context = top programs globally (may overlap with beam)
```

Beam depth is tracked per program via parent linkage and can be penalized in scoring via `beam_depth_penalty`.

## Selection strategies

| Strategy | Behavior |
|----------|----------|
| `diversity_weighted` | Softmax over `(1 - w) * fitness + w * diversity_from_recently_expanded` (default) |
| `stochastic` | Softmax-weighted random sampling by score at temperature `beam_temperature` |
| `round_robin` | Cycles through beam members in score order |
| `best` | Always picks the highest-scoring beam member |

## Config

```yaml
search:
  type: "beam_search"
  database:
    beam_width: 5
    beam_selection_strategy: "diversity_weighted"
    beam_diversity_weight: 0.3    # weight for diversity term (0 = pure fitness)
    beam_temperature: 1.0         # temperature for stochastic / diversity_weighted
    beam_depth_penalty: 0.0       # exponential penalty per depth level (0 = disabled)
```

## When to use

Beam search prevents the population from collapsing to a single lineage, which Top-K is prone to. `diversity_weighted` is the default and generally works well. Increase `beam_diversity_weight` if runs tend to converge prematurely; decrease it on tasks where exploitation matters more.
