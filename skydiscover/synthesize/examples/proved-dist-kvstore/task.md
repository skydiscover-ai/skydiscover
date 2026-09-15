---
checked_by: proof
---

# Task: implement Read-Your-Writes and prove it refines the abstract spec

Read-Your-Writes (RYW) is the guarantee that a client always sees its own writes.
`evaluator/I_RYW_star.v` is the abstract reference spec (`Module I_RYW_star <: AlgDef`): it tracks
every put a client has done as a set, and its read guard requires only that the reading replica
has not missed any of that client's own past puts (set inclusion).

1. Implement RYW as a concrete, efficient `AlgDef`.
2. Prove, in Coq, that your implementation refines `I_RYW_star`: its observable behaviour is
   admitted by the set-inclusion guard for every reachable state.

## The quality bar

obligation: compression

"Concrete, efficient" is a hard requirement. The verifier checks that the proof is correct; this bar
states what makes it worth proving. A candidate that type-checks but misses it is rejected and
re-searched. The critic judges a candidate against the `obligation:` line above.

1. Bounded by the store's dimensions, never by history: a fixed-size summary sized by keys and
   clients or replicas, not by operation count. Re-indexing the spec's own history is not a summary.
2. End to end: every side is compressed, the client state, each wire payload, and the replica entry.
3. Lossy, many to one: `R` in `RYW_Refinement.v` is many-to-one. Equality or a reversible re-tag
   compresses nothing and is rejected.
4. Complete: every operation is proved on the new representation and exercised by `RYW_Examples.v`.

## What you are given
`evaluator/` holds the spec. It is read-only; never edit it.

- `evaluator/KVStore.v`, the framework:
  - `Module Type AlgDef`: the interface your implementation fills in: client and replica state,
    the get and put operations, and decidable payload equalities.
  - `Module Type Refinement (I S : AlgDef)`: the proof obligation: a relation `R` between the two
    worlds, and proofs of `init_sim` (initial states related) and `step_sim` (every step is matched
    with the same observable effect).
  - `TraceInclusion`: the functor that turns any such `Refinement` into the end guarantee, trace
    inclusion.
- `evaluator/I_RYW_star.v`: `Module I_RYW_star <: AlgDef`, the abstract RYW semantics to refine.
- `evaluator/RYW_Correct.v`: states `ryw_impl_refines_spec`, the theorem you must make hold. It applies
  `TraceInclusion` to your modules against `I_RYW_star`, so a weakened proof fails to compile.

## What to write
Three files, with exactly these names, in the implementation directory of the run.

1. `RYW_Impl.v`: `Module RYW_Impl <: AlgDef`, a concrete, efficient RYW replica.
2. `RYW_Refinement.v`: `Module RYW_Refinement <: Refinement RYW_Impl I_RYW_star`. Define `R`
   and prove `init_sim` and `step_sim`. The `<:` ascription forces the spec's exact statements; you
   cannot prove something weaker.
3. `RYW_Examples.v`: at least one concrete, computable non-vacuity example that exercises
   `RYW_Impl` doing real work: a `put` then a `get` returning it, and a stale replica rejected. This
   blocks the do-nothing reward hack, "guards always false, so it refines by doing nothing".

Import the spec as `KVS.KVStore`, `KVS.I_RYW_star`; your own files as `KVS.RYW_Impl`,
`KVS.RYW_Refinement`.

## Testing
The proof is the test: `evaluator/tests/proof.sh`, run by `evaluator/tests/test.sh` with `SKYDISCOVER_IMPL` set to the
directory holding your three files. It passes (exit 0) when:

- `evaluator/` is byte-for-byte what the task shipped.
- Your files contain none of `Admitted`, `admit`, `Axiom`, `Conjecture`, `sorry`, `bypass_check`,
  `Unset Guard/Positivity/Universe Checking`, anywhere, comments included.
- The spec and your files type-check from an empty directory, including `RYW_Correct.v`, so
  `ryw_impl_refines_spec` genuinely holds.
- `Print Assumptions ryw_impl_refines_spec` rests on nothing beyond the framework's own constants
  (`KVStore.allReplicas`, `KVStore.allClients`, `KVStore.c0` and their axioms): no smuggled axiom, in
  particular no `functional_extensionality`.
- The non-vacuity example built (`RYW_Examples.vo`) and mentions `RYW_Impl`.
