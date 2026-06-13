"""KV-store task evaluator: build -> 6 consistency tests (HARD GATE) -> benchmark.

This is the scorer used by NON-jitskit strategies (claude_code, evolutionary) and
for cross-strategy comparability. (`-s jitskit` reports the runtime's own
leaderboard peak and does not call this.)

Contract: ``evaluate(program_path) -> dict`` with at least ``combined_score`` and
``validity``. The correctness tests are a hard gate — any failure scores 0.

Status: builds against the shared C++ harness at ``benchmarks/kvstore/_harness`` — a symlink
to the vendored runtime's ``interface/`` (``kvstore_interface.h``, ``benchmark_harness.cc``,
``consistency_harness.cc``, ``CMakeLists.txt``): the single source, no copy. If the harness is
somehow absent, this evaluator returns ``validity: 0`` with a clear error rather than pretending
to pass. It is intentionally NOT exercised in CI; run it on the build/bare-metal tier.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# The shared harness: in-container it is set via SKYKV_HARNESS_DIR; in-repo it
# lives two levels up at benchmarks/kvstore/_harness/.
_HARNESS_DIR = Path(
    os.environ.get("SKYKV_HARNESS_DIR", Path(__file__).resolve().parents[2] / "_harness")
)
_REQUIRED_HARNESS = (
    "kvstore_interface.h",
    "benchmark_harness.cc",
    "consistency_harness.cc",
    "CMakeLists.txt",
)
_NUM_CONSISTENCY_TESTS = 6


def _fail(error: str, **extra) -> dict:
    """A failed evaluation: zero score, validity 0, error artifact."""
    return {
        "combined_score": 0.0,
        "metrics": {"combined_score": 0.0, "validity": 0.0},
        "artifacts": {"error": error, **extra},
    }


def _harness_ready() -> bool:
    return _HARNESS_DIR.is_dir() and all((_HARNESS_DIR / f).exists() for f in _REQUIRED_HARNESS)


def evaluate(program_path: str) -> dict:
    if not _harness_ready():
        return _fail(
            f"Shared harness missing at {_HARNESS_DIR} (the _harness symlink should resolve "
            f"to the vendored runtime's interface/ — see benchmarks/kvstore/README.md). "
            f"Required files: {list(_REQUIRED_HARNESS)}."
        )

    spec = {key: os.environ[key] for key in os.environ if key.startswith("SKYKV_")}
    work = Path(tempfile.mkdtemp(prefix="kvstore_eval_"))
    try:
        # Stage the candidate as the impl the harness compiles.
        shutil.copy(program_path, work / "kvstore_impl.cc")
        for f in _REQUIRED_HARNESS:
            shutil.copy(_HARNESS_DIR / f, work / f)

        build = subprocess.run(
            ["cmake", "-S", str(work), "-B", str(work / "build")],
            capture_output=True,
            text=True,
        )
        if build.returncode != 0:
            return _fail("cmake configure failed", build_log=build.stderr[-4000:])
        make = subprocess.run(
            ["cmake", "--build", str(work / "build"), "-j"],
            capture_output=True,
            text=True,
        )
        if make.returncode != 0:
            return _fail("build failed", build_log=make.stderr[-4000:])

        # HARD GATE: all 6 consistency tests must pass.
        consistency = work / "build" / "consistency_harness"
        for test_id in range(1, _NUM_CONSISTENCY_TESTS + 1):
            res = subprocess.run([str(consistency), str(test_id)], capture_output=True, text=True)
            if res.returncode != 0:
                return _fail(
                    f"consistency test {test_id}/{_NUM_CONSISTENCY_TESTS} failed",
                    test_log=res.stdout[-2000:] + res.stderr[-2000:],
                )

        # Benchmark: the harness prints the peak Mops/s as JSON on stdout.
        bench = subprocess.run(
            [str(work / "build" / "benchmark_harness"), json.dumps(spec)],
            capture_output=True,
            text=True,
        )
        if bench.returncode != 0:
            return _fail("benchmark failed", bench_log=bench.stderr[-4000:])

        peak_mops = _parse_peak_mops(bench.stdout)
        if peak_mops is None:
            return _fail(
                "could not parse peak Mops/s from benchmark output",
                bench_stdout=bench.stdout[-2000:],
            )

        return {
            "combined_score": peak_mops,
            "metrics": {"combined_score": peak_mops, "validity": 1.0},
            "artifacts": {"spec": spec},
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _parse_peak_mops(stdout: str):
    """The benchmark harness emits a JSON line; pull peak_mops out of it."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ("peak_mops", "combined_score", "mops"):
            if isinstance(obj.get(key), (int, float)):
                return float(obj[key])
    return None


if __name__ == "__main__":
    from wrapper import run

    run(evaluate)
