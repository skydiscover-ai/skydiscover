# Formally Verified Distributed Key-Value Store

Build a distributed key-value store that guarantees read-your-writes, together with a Rocq proof
that it does. The test is the proof assistant: the result is accepted only if the proof holds.
Needs [Rocq](https://rocq-prover.org) installed.

## Run it

Wire the skill once with `uv run skydiscover init` and restart your coding agent (see the
[quick start](../../README.md#-quick-start)). Then, from the repo root:

```
/skysynth prove the Read-Your-Writes store specified in skydiscover/synthesize/examples/proved-dist-kvstore/task.md
```

The result lands in `outputs/synthesize/<slug>_<timestamp>/best/`. To check a delivered proof
yourself:

```
SKYDISCOVER_IMPL=outputs/synthesize/<slug>_<timestamp>/best/artifact bash skydiscover/synthesize/examples/proved-dist-kvstore/evaluator/tests/test.sh
```

## What's here

| Path | What it is |
|---|---|
| `task.md` | what to build and prove, and the efficiency bar; the input to `/skysynth` |
| `evaluator/` | the spec, read-only: the interface (`KVStore.v`), the abstract semantics to refine (`I_RYW_star.v`), and the theorem to prove (`RYW_Correct.v`) |
| `evaluator/tests/proof.sh` | the one test: builds the spec and the agent's `RYW_Impl.v`, `RYW_Refinement.v`, `RYW_Examples.v` from an empty directory, then checks for escape hatches, smuggled axioms, and that the example is real |
| `evaluator/tests/test.sh` | runs the tests against `$SKYDISCOVER_IMPL` |

## Set up your own

Another property in the same framework: copy this directory, write a new `task.md`, put the new
spec and theorem in `evaluator/` (reusing `KVStore.v`), and update the file names, the theorem, and the
spec hashes at the top of `evaluator/tests/proof.sh`. A different prover or framework: see
[A formal domain](../README.md#a-formal-domain).
