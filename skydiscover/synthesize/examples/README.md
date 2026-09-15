# Synthesize Examples

New here? Start with the [tutorial](tutorial/): one prompt and one workload file in, a cache built for
that workload out, with a notebook of a real run.

| Example | What it builds | Checked by | Needs |
|---|---|---|---|
| [`tutorial/`](tutorial/) | a cache that beats FIFO and LRU | tests | any machine |
| [`single-machine-kvstore/`](single-machine-kvstore/) | a larger-than-memory key-value store | tests | Linux, local SSD |
| [`proved-dist-kvstore/`](proved-dist-kvstore/) | a distributed key-value store, proved read-your-writes | Rocq proof | Rocq |
| [`llm-router/`](llm-router/) | a per-tenant LLM router | tests | any machine |
| [`inference-engine/`](inference-engine/) | a prefix-sharing inference engine for Qwen3-4B | tests | NVIDIA L4 GPU |

Each README has the exact `/skysynth` prompt. `/skysynth build me <any system>` also works with
nothing checked in: the agents write the specification and tests during the run.

## Layout

```
<name>/
    README.md      what it builds, the prompt to run, what each file is
    task.md        what to build, how it is scored, the interface
    spec/          optional: what is fixed before the run: cards with hardware and workload facts
                   the run's specification builds on
    evaluator/     optional: what a candidate is checked and scored against: the interface, the
                   benchmark, a correct reference implementation (for a proof, the formal spec),
                   and in tests/ the tests the task ships with the test.sh that runs them
```

`spec/` seeds the run's specification; `evaluator/` seeds the run's `synthesis/evaluator/`, and
its `tests/` the run's `synthesis/tests/`. `task.md` is the only file the agents read; it names
the rest. Whatever else is checked in is fixed for every run; whatever is left out, the agents
write during the run.

## Adding a domain

1. Copy the closest example and write `task.md`: what to build, how it is scored, the interface,
   how to build and run it.
2. Optionally add `evaluator/` (interface, benchmark, reference implementation; for a proof, the
   formal spec) and `spec/` (hardware and workload cards). Everything is in whatever language suits the
   system. Tests the task ships go in `evaluator/tests/` with a `test.sh` that builds and runs
   them against `$SKYDISCOVER_IMPL` (see `single-machine-kvstore/evaluator/tests/`).
3. Write the README from the template below and add a row to the table above.

### A formal domain

Here a proof checker (Rocq, Lean, Dafny, or Verus) replaces the tests. The agent writes the code
and its proof; the spec and theorem statements are read-only, so it cannot weaken them. `task.md`
starts with `checked_by: proof`, the spec goes in `evaluator/` (it is the interface the code and
the proof are checked against), and the proof check is the test the task ships: an
`evaluator/tests/test.sh` and one test beside it (`proved-dist-kvstore/evaluator/tests/proof.sh`)
that, given the agent's files in `$SKYDISCOVER_IMPL` and the spec in `$SKYDISCOVER_INTERFACE`,
exits 0 only when the proof holds. Write it for your prover; the one in `proved-dist-kvstore` checks that

- the spec is byte-for-byte what the task shipped (a `sha256sum --check` of its files);
- the agent's files contain no word that lets the prover accept a claim without a proof
  (`Admitted`, `admit`, `Axiom`, `sorry`, ...);
- everything type-checks when built from an empty directory;
- the target theorems depend on no assumption beyond the ones the spec declares (Coq's
  `Print Assumptions`);
- a concrete example compiled, so the theorems are not true only of an empty case.

Keep any reference solution out of the task directory; the agent must not see it.

### README template

```markdown
# <What It Builds>

One or two sentences: what it builds, what it is scored on, what machine it needs.

## Run it

Wire the skill once with `uv run skydiscover init` and restart your coding agent. <Any one-time
data setup.> Then, from the repo root:

    /skysynth build the <system> specified in skydiscover/synthesize/examples/<name>/task.md

The result lands in `outputs/synthesize/<slug>_<timestamp>/best/` (`<slug>` is a short name made from the prompt).

## What's here

| Path | What it is |
|---|---|
| `task.md` | ... |
| `spec/` | ... |
| `evaluator/` | ... |
```
