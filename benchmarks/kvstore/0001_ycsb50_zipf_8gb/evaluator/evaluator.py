"""KV-store task evaluator.

`-s jitskit` does NOT use this: jitskit drives the runtime's own build -> 6-test gate
-> benchmark and reports its leaderboard peak. This evaluator exists only to score
*non-jitskit* strategies (claude_code, evolutionary) on the same task.

Scoring a KV candidate correctly means running the runtime's REAL harness: build the
`kvstore_bench` and `consistency_test` targets, run the 6-test gate and the benchmark
with the real argv (num_threads, load/run `.dat` traces, mem-budget-in-bytes,
storage_path), and parse the text output. That is exactly the runtime's eval pipeline.
Re-implementing it here drifts from the harness contract, so this evaluator does NOT.

Until the runtime exposes a reusable eval entrypoint (a measurement-box task), this
fails closed with a clear message rather than returning a wrong score. It never
pretends to pass.
"""


def evaluate(program_path: str) -> dict:
    return {
        "combined_score": 0.0,
        "metrics": {"combined_score": 0.0, "validity": 0.0},
        "artifacts": {
            "error": (
                "Non-jitskit KV scoring is not implemented in-repo. It must reuse the "
                "runtime's real eval (build kvstore_bench + consistency_test, the 6-test "
                "gate, and the benchmark with real traces) on the measurement box — not a "
                "hand-rolled reimplementation. Use `-s jitskit` (self-evaluates) for now."
            ),
        },
    }
