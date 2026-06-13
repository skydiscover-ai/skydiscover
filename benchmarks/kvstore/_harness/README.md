# `_harness/` — the one canonical C++ harness (populate before use)

This dir holds the single shared copy of the runtime's harness:

- `kvstore_interface.h` — the `IKVStore` API the seed and candidates implement.
- `benchmark_harness.cc` — the YCSB-style driver that measures peak Mops/s.
- `consistency_harness.cc` — the 6 correctness tests (the hard gate).
- `CMakeLists.txt` — builds the candidate `kvstore_impl.cc` against the above.

**It is intentionally empty in the repo.** These are private-runtime C++ files; how to bring them in
(git submodule vs a synced/vendored copy) is an open team decision (plan §8.1). Populate it from the
runtime submodule:

```bash
git submodule update --init --recursive
ln -s ../../skydiscover/search/jitskit/runtime/_harness benchmarks/kvstore/_harness
```

The evaluator (`0001_*/evaluator/evaluator.py`) refuses to score (returns `validity: 0` with an error)
until these files are present, so a missing harness can never silently pass.
