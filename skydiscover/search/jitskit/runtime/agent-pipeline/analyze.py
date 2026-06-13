#!/usr/bin/env python3
"""
agent-pipeline/analyze.py - Parse generated-store benchmark outputs and compare them to
available reference numbers.

Usage: python3 analyze.py <results_dir>
  results_dir: directory containing wl{id}_t{threads}.txt files
"""
import json
import re
import sys
from pathlib import Path

SKYKV_ROOT = Path(__file__).resolve().parents[1]       # skykv-claude/
PROJECT_ROOT = Path(__file__).resolve().parents[2]      # skydiscover-private/
LOCAL_RESULTS_DIR = PROJECT_ROOT / "faster" / "baselines"

# Paper-backed fallback references. Prefer local one-NUMA-node FASTER results
# from reproduction_results/ when available.
PAPER_REFERENCE_TARGETS = {
    "rmw_100": {
        1: 4.26,
        2: 7.99,
        4: 15.52,
        8: 30.47,
        16: 59.15,
        32: 84.95,
        56: 123.07,
    },
}

WORKLOAD_NAMES = {0: "ycsb_a", 1: "rmw_100", 2: "ycsb_b", 3: "ycsb_c",
                  4: "w_0_100", 5: "ts_head_delete"}
WORKLOAD_DESCS = {
    "ycsb_a": "50% Read / 50% Upsert",
    "rmw_100": "100% Read-Modify-Write",
    "ycsb_b": "95% Read / 5% Upsert",
    "ycsb_c": "100% Read",
    "w_0_100": "0% Read / 100% Upsert",
    "ts_head_delete": "Time-series head-delete (append at tail, delete from head)",
}
LOCAL_RESULT_WORKLOADS = {
    "rmw_100": "rmw_zipf_8B",
    "ycsb_a": "ycsba_zipf_8B",
    "ycsb_b": "ycsbb_zipf_8B",
    "ycsb_c": "ycsbc_zipf_8B",
    "w_0_100": "w0100_zipf_8B",
}
# LTM baselines (100-byte values, 8 GB budget) — checked as fallback
LOCAL_RESULT_WORKLOADS_LTM = {
    "rmw_100": "ltm8g_rmw",
    "ycsb_a": "ltm8g_ycsba",
    "ycsb_b": "ltm8g_ycsbb",
    "ycsb_c": "ltm8g_ycsbc",
    "w_0_100": "ltm8g_w0100",
}

OPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s+ops/second/thread")
VALIDATION_SUMMARY_RE = re.compile(
    r"VALIDATION load_keys checked=(\d+) retained=(\d+) missing=(\d+) "
    r"wrong_size=(\d+) wrong_value=(\d+) retention_pct=([\d.]+) "
    r"elapsed_sec=([\d.]+) stride=(\d+)"
)


def parse_validation(text: str) -> dict:
    validation = {"status": "unknown"}
    match = VALIDATION_SUMMARY_RE.search(text)
    if match:
        validation.update({
            "checked": int(match.group(1)),
            "retained": int(match.group(2)),
            "missing": int(match.group(3)),
            "wrong_size": int(match.group(4)),
            "wrong_value": int(match.group(5)),
            "retention_pct": float(match.group(6)),
            "elapsed_sec": float(match.group(7)),
            "stride": int(match.group(8)),
        })

    if "VALIDATION FAILED" in text:
        validation["status"] = "failed"
    elif "VALIDATION PASSED" in text:
        validation["status"] = "passed"
    elif "VALIDATION SKIPPED" in text:
        validation["status"] = "skipped"

    return validation


def parse_ops_text(text: str):
    match = OPS_RE.search(text)
    return float(match.group(1)) if match else None


def parse_ops_file(path: Path):
    """Extract ops/second/thread from a benchmark output file.

    The metric line is always near the end, so we read the last 4KB
    instead of scanning potentially multi-GB FASTER output files.
    """
    try:
        size = path.stat().st_size
        with path.open("r", errors="replace") as handle:
            if size > 4096:
                handle.seek(size - 4096)
                handle.readline()  # skip partial line
            last = None
            for line in handle:
                match = OPS_RE.search(line)
                if match:
                    last = float(match.group(1))
            return last
    except Exception:
        return None


def parse_output_file(path: Path):
    """Return benchmark and validation data from one output file."""
    try:
        text = path.read_text()
    except Exception:
        return {"ops_per_thread": None, "validation": {"status": "unknown"}}

    result = {
        "ops_per_thread": parse_ops_text(text),
        "validation": parse_validation(text),
    }

    # Include diagnostic snippet for failed/timeout runs so LLM can diagnose
    if result["ops_per_thread"] is None:
        snippet = text[:500].strip()
        if snippet:
            result["error_snippet"] = snippet

    return result


def _scan_reference_dir(wl_dir: Path) -> dict[int, float]:
    """Scan a directory for t*_r*.txt files and return {threads: best_total_mops}."""
    if not wl_dir.exists():
        return {}
    results = {}
    for path in wl_dir.glob("t*_r*.txt"):
        match = re.match(r"t(\d+)_r\d+\.txt$", path.name)
        if not match:
            continue
        threads = int(match.group(1))
        ops_per_thread = parse_ops_file(path)
        if ops_per_thread is None:
            continue
        total_mops = ops_per_thread * threads / 1e6
        results[threads] = max(total_mops, results.get(threads, 0.0))
    return results


def _lookup_adapter_csv(workload_name: str, budget_gb: int) -> float | None:
    """Look up the FASTER-adapter-through-our-harness baseline at a given budget.

    Reads `setups/faster-single-machine/baselines/adapter_baseline_results.csv`,
    which measures FASTER through the SAME harness agents run against (same
    cgroup, same sync IKVStore, same integrity audits). This is the FAIR
    comparison target — unlike the native FASTER numbers which use a different
    harness with async I/O pipelining the sync IKVStore can't express.

    CSV columns: budget_gb, faster_5050_mops, rocksdb_5050_mops,
                 faster_0100_mops, rocksdb_0100_mops.
    Returns mops for the given workload+budget, or None if not found.
    """
    csv_path = (SKYKV_ROOT / "setups" / "faster-single-machine" /
                "baselines" / "adapter_baseline_results.csv")
    if not csv_path.is_file():
        return None
    col_map = {"ycsb_a": 1, "w_0_100": 3}  # FASTER-adapter cols only
    col = col_map.get(workload_name)
    if col is None:
        return None
    try:
        with open(csv_path) as f:
            next(f)  # header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 5:
                    continue
                try:
                    b = int(parts[0])
                except ValueError:
                    continue
                if b != budget_gb:
                    continue
                try:
                    return float(parts[col])
                except (ValueError, IndexError):
                    return None
    except OSError:
        return None
    return None


def best_local_reference_total_mops(workload_name: str, mode: str = "inmem",
                                    budget_gb: int | None = None) -> dict[int, float]:
    # LTM mode with known budget: prefer the FAIR adapter-through-our-harness
    # baseline from adapter_baseline_results.csv. Falls through to native
    # FASTER files if the CSV is missing or has no row for this budget.
    if mode == "ltm" and budget_gb is not None:
        mops = _lookup_adapter_csv(workload_name, budget_gb)
        if mops is not None:
            # Only t=16 is measured in the adapter sweep — same thread
            # count agents use.
            return {16: mops}

    if mode == "ltm":
        # LTM mode: use LTM baselines (100B values, 8 GB budget)
        rel = LOCAL_RESULT_WORKLOADS_LTM.get(workload_name)
        if rel:
            results = _scan_reference_dir(LOCAL_RESULTS_DIR / rel / "numa_0")
            if results:
                return results

    # In-memory baselines (8B values)
    rel = LOCAL_RESULT_WORKLOADS.get(workload_name)
    if rel:
        results = _scan_reference_dir(LOCAL_RESULTS_DIR / rel / "numa_0")
        if results:
            return results

    return {}


def pick_reference(workload_name: str, threads: int, local_refs: dict[int, float]):
    local_target = local_refs.get(threads)
    if local_target is not None:
        return local_target, "local_faster_numa0"

    paper_target = PAPER_REFERENCE_TARGETS.get(workload_name, {}).get(threads)
    if paper_target is not None:
        return paper_target, "paper_reference"

    return None, None


def analyze(results_dir: Path, mode: str = "inmem",
            budget_gb: int | None = None) -> dict:
    """Parse all benchmark outputs in results_dir, return analysis dict.

    Args:
        mode: "inmem" or "ltm" — selects which FASTER baseline to compare against.
        budget_gb: active memory budget; enables adapter-baseline CSV lookup
            for fair through-our-harness comparison. Falls back to native
            FASTER output files when None or when the CSV has no row.
    """
    results_dir = Path(results_dir)
    analysis = {"workloads": {}, "summary": {}}

    # Filename forms in use:
    #   wl0_t16.txt           (original, inmem runs)
    #   wl0_t16_mem8g.txt     (LTM runs, suffix encodes mem budget)
    # Accept both; the optional suffix is ignored for workload/thread parsing.
    pattern = re.compile(r"^wl(\d+)_t(\d+)(?:_[^.]+)?\.txt$")
    by_workload = {}
    for output_path in results_dir.glob("wl*_t*.txt"):
        match = pattern.match(output_path.name)
        if not match:
            continue
        workload_id = int(match.group(1))
        threads = int(match.group(2))
        workload_name = WORKLOAD_NAMES.get(workload_id, f"wl{workload_id}")
        parsed = parse_output_file(output_path)
        by_workload.setdefault(workload_name, {})[threads] = parsed

    if not by_workload:
        analysis["summary"]["error"] = "No result files found"
        return analysis

    all_pcts = []
    num_failed_runs = 0
    num_invalid_runs = 0
    num_runs_missing_validation = 0
    num_runs_with_validation = 0

    local_refs_by_workload = {
        workload_name: best_local_reference_total_mops(
            workload_name, mode=mode, budget_gb=budget_gb)
        for workload_name in by_workload
    }

    for workload_name, thread_map in sorted(by_workload.items()):
        local_refs = local_refs_by_workload.get(workload_name, {})
        wdata = {"threads": {}, "peak_mops": 0.0, "peak_threads": 0}

        for threads, parsed in sorted(thread_map.items()):
            validation = parsed.get("validation", {"status": "unknown"})
            validation_status = validation.get("status", "unknown")

            if validation_status == "failed":
                num_invalid_runs += 1
                wdata["threads"][threads] = {
                    "status": "validation_failed",
                    "validation": validation,
                }
                continue

            if validation_status in ("passed", "skipped"):
                num_runs_with_validation += 1
            else:
                num_runs_missing_validation += 1

            ops_per_thread = parsed.get("ops_per_thread")
            if ops_per_thread is None:
                num_failed_runs += 1
                entry = {
                    "status": "failed",
                    "validation": validation,
                }
                if "error_snippet" in parsed:
                    entry["error_snippet"] = parsed["error_snippet"]
                wdata["threads"][threads] = entry
                continue

            total_mops = ops_per_thread * threads / 1e6
            reference_mops, reference_label = pick_reference(workload_name, threads, local_refs)
            pct = (total_mops / reference_mops * 100) if reference_mops else None

            wdata["threads"][threads] = {
                "ops_per_thread": round(ops_per_thread, 0),
                "total_mops": round(total_mops, 3),
                "reference_mops": round(reference_mops, 3) if reference_mops is not None else None,
                "reference_label": reference_label,
                "paper_target_mops": PAPER_REFERENCE_TARGETS.get(workload_name, {}).get(threads),
                "local_faster_mops": round(local_refs[threads], 3) if threads in local_refs else None,
                "pct_of_reference": round(pct, 1) if pct is not None else None,
                "pct_of_faster": round(pct, 1) if pct is not None else None,
                "validation": validation,
            }

            if total_mops > wdata["peak_mops"]:
                wdata["peak_mops"] = total_mops
                wdata["peak_threads"] = threads

            if pct is not None:
                all_pcts.append(pct)

        analysis["workloads"][workload_name] = wdata

    analysis["summary"]["avg_pct_of_reference"] = round(sum(all_pcts) / len(all_pcts), 1) if all_pcts else 0
    analysis["summary"]["best_pct_of_reference"] = round(max(all_pcts), 1) if all_pcts else 0
    analysis["summary"]["avg_pct_of_faster"] = analysis["summary"]["avg_pct_of_reference"]
    analysis["summary"]["best_pct_of_faster"] = analysis["summary"]["best_pct_of_reference"]
    analysis["summary"]["num_workloads"] = len(by_workload)
    analysis["summary"]["num_failed_runs"] = num_failed_runs
    analysis["summary"]["num_invalid_runs"] = num_invalid_runs
    analysis["summary"]["num_runs_with_validation"] = num_runs_with_validation
    analysis["summary"]["num_runs_missing_validation"] = num_runs_missing_validation
    analysis["summary"]["all_validation_passed"] = (
        num_invalid_runs == 0 and num_runs_missing_validation == 0
    )
    return analysis


def print_table(analysis: dict, label: str = ""):
    if label:
        print(f"\n{'=' * 72}")
        print(f" {label}")
    print(f"{'=' * 72}")

    for workload_name, wdata in sorted(analysis.get("workloads", {}).items()):
        desc = WORKLOAD_DESCS.get(workload_name, workload_name)
        print(f"\n  {workload_name.upper()} — {desc}")
        print(f"  {'Threads':>8}  {'Our Mops/s':>11}  {'Ref Mops/s':>14}  {'% of Ref':>10}")
        print(f"  {'─' * 8}  {'─' * 11}  {'─' * 14}  {'─' * 10}")

        for threads, td in sorted(wdata["threads"].items()):
            status = td.get("status")
            if status == "validation_failed":
                validation = td.get("validation", {})
                retention = validation.get("retention_pct")
                retention_str = f"{retention:.2f}%" if retention is not None else "unknown"
                print(f"  {threads:>8}  {'INVALID':>11}  {'—':>14}  {'ret=' + retention_str:>12}")
                continue
            if status == "failed":
                print(f"  {threads:>8}  {'FAILED':>11}")
                continue

            reference = td.get("reference_mops")
            reference_str = f"{reference:.3f}" if reference is not None else "   —  "
            pct = td.get("pct_of_reference")
            pct_str = f"{pct:.1f}%" if pct is not None else "   —"
            label_suffix = ""
            if td.get("reference_label") == "local_faster_numa0":
                label_suffix = " *"
            print(f"  {threads:>8}  {td['total_mops']:>11.3f}  {reference_str:>14}  {(pct_str + label_suffix):>12}")

        print(f"  Peak: {wdata['peak_mops']:.2f} Mops/s at {wdata['peak_threads']} threads")

    summary = analysis.get("summary", {})
    print(f"\n{'─' * 72}")
    print(f"  Average vs chosen reference:   {summary.get('avg_pct_of_reference', 0):.1f}%")
    print(f"  Best vs chosen reference:      {summary.get('best_pct_of_reference', 0):.1f}%")
    print(f"  Invalid runs rejected:         {summary.get('num_invalid_runs', 0)}")
    print(f"  Failed runs:                   {summary.get('num_failed_runs', 0)}")
    print(f"  Runs missing validation data:  {summary.get('num_runs_missing_validation', 0)}")
    if any(
        td.get("reference_label") == "local_faster_numa0"
        for wdata in analysis.get("workloads", {}).values()
        for td in wdata.get("threads", {}).values()
        if "status" not in td
    ):
        print("  * local one-NUMA-node FASTER reference used when available")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist")
        sys.exit(1)

    result = analyze(results_dir)
    print_table(result, label=f"Results from: {results_dir}")
    print("\nJSON:")
    print(json.dumps(result, indent=2))
