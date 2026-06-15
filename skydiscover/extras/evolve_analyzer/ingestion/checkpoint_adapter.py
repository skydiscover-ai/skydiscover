"""
checkpoint_adapter.py
---------------------
Ingestion module for the evolutionary code optimization diagnostic tool.

Converts framework-specific output into standard JSONL records (list of dicts).
Each adapter normalises the framework's native data format into the standard schema.

Supported sources:
  - "jsonl"       : direct JSONL passthrough  (load_jsonl)
  - "skydiscover" : checkpoint directory tree  (adapt_skydiscover)
  - "shinkaevolve": SQLite database            (adapt_shinkaevolve)
  - "openevolve"  : checkpoint dirs + trace    (adapt_openevolve)

Public entry-point: load_evolve_records(source, path, **kwargs) -> List[dict]
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Iterator, List, Optional


# ---------------------------------------------------------------------------
# Standard field names (documentation only – not enforced at runtime)
# ---------------------------------------------------------------------------
# iteration, child_score, parent_score, score_delta, evaluation_status,
# format_valid, mutation_type, model, parent_code, child_code, diff,
# evolved_block_only, signature_preserved, evaluator_metrics,
# cascade_stage_failed, evaluator_artifacts, score_std, num_runs,
# llm_tokens_used, llm_cost_usd, llm_latency_ms, timestamp, island_id,
# framework_stagnation_event, reasoning_trace, meta_suggestion,
# followed_suggestion, early_stop_suggested


# ---------------------------------------------------------------------------
# Status normalisation helper
# ---------------------------------------------------------------------------

_STATUS_MAP: dict[str, str] = {
    "ok": "success",
    "success": "success",
    "succeeded": "success",
    "timeout": "timeout",
    "error": "crash",
    "exception": "crash",
    "crash": "crash",
    "failed": "crash",
}


def _normalise_status(raw: str) -> str:
    """Map raw status strings to the canonical set: success / timeout / crash / unknown."""
    return _STATUS_MAP.get(str(raw).lower(), "unknown")


# ---------------------------------------------------------------------------
# Derived-field fill
# ---------------------------------------------------------------------------

def _fill_derived_fields(records: List[dict]) -> List[dict]:
    """
    Ensures every record has:
    - score_delta     : child_score - parent_score  (if both present and delta missing)
    - evaluation_status: inferred from 'status' key, or default 'success'
    - format_valid    : default True if missing
    - iteration       : sequential index if missing
    """
    for idx, rec in enumerate(records):
        # iteration
        if rec.get("iteration") is None:
            rec["iteration"] = idx

        # score_delta
        if rec.get("score_delta") is None:
            cs = rec.get("child_score")
            ps = rec.get("parent_score")
            if cs is not None and ps is not None:
                rec["score_delta"] = float(cs) - float(ps)

        # evaluation_status
        if rec.get("evaluation_status") is None:
            raw_status = rec.get("status") or rec.get("outcome")
            if raw_status is not None:
                rec["evaluation_status"] = _normalise_status(raw_status)
            else:
                rec["evaluation_status"] = "success"

        # format_valid
        if rec.get("format_valid") is None:
            rec["format_valid"] = True

    return records


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> Iterator[dict]:
    """
    Passthrough for direct JSONL input.

    Reads the file line by line, parses each as JSON.
    Computes score_delta = child_score - parent_score if both are present and
    score_delta is absent.
    Skips blank lines silently.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec: dict = json.loads(line)

            # compute score_delta inline so callers get it even without
            # going through load_evolve_records
            if rec.get("score_delta") is None:
                cs = rec.get("child_score")
                ps = rec.get("parent_score")
                if cs is not None and ps is not None:
                    rec["score_delta"] = float(cs) - float(ps)

            yield rec


# ---------------------------------------------------------------------------
# adapt_skydiscover
# ---------------------------------------------------------------------------

def _read_first_json(directory: Path, *candidates: str) -> dict:
    """Return parsed JSON from the first matching candidate filename, or {}."""
    for name in candidates:
        p = directory / name
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    return {}


_LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+)")
_LOG_ITER_FAILED = re.compile(r"Iteration (\d+) failed: (.+)")
_LOG_ITER_GEPA_OUTCOME = re.compile(
    r"Iteration (\d+): (REJECTED|ACCEPTED) child \(child_score=([-\d.eE+]+)[^)]*parent_score=([-\d.eE+]+)\)"
)
_LOG_ITER_PROG_COMPLETED = re.compile(
    r"Iteration (\d+): Program (\S+) \(parent: (\S+)\) completed in ([\d.]+)s"
)
_LOG_ITER_SUCCESS = re.compile(
    r"Evaluated program (\S+)(?:\s+\[train\])?\s+in [\d.]+s: .*?combined_score=([\d.e+-]+)"
)
_LOG_METRICS_LINE = re.compile(r"- INFO - Metrics: (.+)")
_LOG_INITIAL_EVAL = re.compile(r"Adding initial program to database")
_LOG_SEED_EVAL = re.compile(
    r"Evaluated program (\S+)(?:\s+\[train\])?\s+in [\d.]+s: (.+)"
)
_LOG_NESTED_DICT = re.compile(r"(\w+)=(\{[^}]+\})")
_LOG_FLAT_KV = re.compile(r"\b(\w+)=([\d.]+)")
_LOG_HTTP_ERROR = re.compile(r"HTTP Error (50[23])")
_LOG_CONN_REFUSED = re.compile(r"ConnectionRefusedError.*Connect call failed \(('[\d.]+', \d+)\)")
_LOG_CLIENT_ERROR = re.compile(r"ERROR - Client error during request")
_LOG_EVAL_EXTRACT_FAIL = re.compile(r"Failed to extract Go code from program")
_LOG_EVAL_BUILD_FAIL = re.compile(r"Build failed: (.+)")


def _extract_infra_log_signals(log_path: Path) -> Optional[dict]:
    """Scan a run log for HTTP 50x bursts and ConnectionRefusedError events.

    Returns a dict with structured infra signals, or None if the log has no
    infra-related errors.
    """
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    # ── Pass 1: collect all HTTP 50x error line indices and timestamps ────────
    http_error_lines: list[tuple[int, str, str]] = []  # (line_idx, code, line)
    for i, line in enumerate(lines):
        m = _LOG_HTTP_ERROR.search(line)
        if m:
            ts_m = _LOG_TS.match(line)
            ts_str = ts_m.group(1) if ts_m else ""
            http_error_lines.append((i, m.group(1), line))

    # Find the largest burst: cluster of HTTP error lines within a 200-line window
    burst_start_line: Optional[str] = None
    burst_start_ts: Optional[str] = None
    burst_count: int = 0
    burst_types: set[str] = set()

    if http_error_lines:
        # Sliding window: group errors within 50 lines of each other.
        # 50 is tight enough to separate isolated early errors from a sustained burst.
        clusters: list[list[tuple[int, str, str]]] = []
        current_cluster: list[tuple[int, str, str]] = [http_error_lines[0]]
        for prev, curr in zip(http_error_lines, http_error_lines[1:]):
            if curr[0] - prev[0] <= 50:
                current_cluster.append(curr)
            else:
                clusters.append(current_cluster)
                current_cluster = [curr]
        clusters.append(current_cluster)

        largest = max(clusters, key=len)
        burst_count = len(largest)
        burst_types = {entry[1] for entry in largest}
        first_entry = largest[0]
        burst_start_line = first_entry[2]
        ts_m = _LOG_TS.match(first_entry[2])
        burst_start_ts = ts_m.group(1) if ts_m else None

    # ── Pass 2: find first ConnectionRefusedError ─────────────────────────────
    crash_timestamp: Optional[str] = None
    crash_host: Optional[str] = None
    client_error_line: Optional[str] = None
    conn_refused_line: Optional[str] = None

    last_client_error: Optional[str] = None
    for line in lines:
        if _LOG_CLIENT_ERROR.search(line):
            last_client_error = line
            continue
        m = _LOG_CONN_REFUSED.search(line)
        if m:
            conn_refused_line = line.strip()
            client_error_line = last_client_error
            # Extract host:port — strip quotes: "'9.47.192.193', 8080" → "9.47.192.193:8080"
            raw = m.group(1)
            host_port = raw.replace("'", "").replace(", ", ":")
            crash_host = host_port
            # Timestamp comes from the preceding ERROR line
            if client_error_line:
                ts_m = _LOG_TS.match(client_error_line)
                crash_timestamp = ts_m.group(1) if ts_m else None
            break

    if not burst_count and not crash_host:
        return None

    # ── Build sample_error_lines (up to 3 key log lines) ─────────────────────
    sample: list[str] = []
    if burst_start_line:
        sample.append(burst_start_line.strip())
    if client_error_line:
        sample.append(client_error_line.strip())
    if conn_refused_line:
        sample.append(conn_refused_line.strip())

    result: dict = {}
    if burst_count:
        result["http_error_burst_start"] = burst_start_ts
        result["http_error_burst_count"] = burst_count
        result["http_error_types"] = sorted(burst_types)
    if crash_host:
        result["crash_timestamp"] = crash_timestamp
        result["crash_host"] = crash_host
    if sample:
        result["sample_error_lines"] = sample

    return result if result else None


def _parse_eval_error_records(lines: list[str]) -> list[dict]:
    """Fallback parser for eval-error-only logs (e.g. runs where every LLM output
    was invalid so no iteration-tracking lines were emitted).

    Groups error events within a 120-second window into per-iteration failed records.
    Returns [] when no recognizable eval-error lines are found.
    """
    from datetime import datetime as _dt

    events: list[dict] = []
    for line in lines:
        ts_m = _LOG_TS.match(line)
        if not ts_m:
            continue
        try:
            ts = _dt.strptime(ts_m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(ts_m.group(2)) / 1000.0
        except ValueError:
            continue
        if _LOG_EVAL_EXTRACT_FAIL.search(line):
            events.append({"ts": ts, "error": "Failed to extract Go code from program"})
        elif _LOG_EVAL_BUILD_FAIL.search(line):
            m = _LOG_EVAL_BUILD_FAIL.search(line)
            events.append({"ts": ts, "error": f"Build failed: {m.group(1).strip()}"})

    if not events:
        return []

    WINDOW = 120.0
    groups: list[list[dict]] = []
    current: list[dict] = [events[0]]
    for ev in events[1:]:
        if ev["ts"] - current[-1]["ts"] <= WINDOW:
            current.append(ev)
        else:
            groups.append(current)
            current = [ev]
    groups.append(current)

    records = []
    for i, group in enumerate(groups, 1):
        error_msgs = list(dict.fromkeys(e["error"] for e in group))
        records.append({
            "iteration": i,
            "outcome": "failed",
            "evaluation_status": "crash",
            "format_valid": False,
            "error": "; ".join(error_msgs),
            "timestamp": group[0]["ts"],
        })
    return records


def _find_run_log(run_dir: Path) -> Optional[Path]:
    """Return the first .log file found in run_dir/logs/, or None."""
    logs_dir = run_dir / "logs"
    if logs_dir.is_dir():
        logs = sorted(logs_dir.glob("*.log"))
        if logs:
            return logs[0]
    return None


def _parse_log_metrics_string(s: str) -> dict:
    """Parse 'key=val, key={...}, ...' from a log eval-result line into a dict.

    Nested dicts (e.g. metrics={...}) are preserved as dict values; all other
    numeric key=value pairs are stored as floats.
    """
    result: dict = {}
    remaining = s
    for m in _LOG_NESTED_DICT.finditer(s):
        try:
            val = ast.literal_eval(m.group(2))
            if isinstance(val, dict):
                result[m.group(1)] = val
        except (ValueError, SyntaxError):
            pass
        remaining = remaining.replace(m.group(0), "")
    for m in _LOG_FLAT_KV.finditer(remaining):
        try:
            result[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return result


def _extract_seed_metrics_from_log(log_path: Path) -> dict:
    """Return the seed program's metrics dict parsed from the run log.

    The seed is the first program evaluated after 'Adding initial program to
    database'. Its full key=value metrics string is parsed into a dict
    (matching the structure of program JSON metrics fields).
    """
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    seen_initial = False
    for line in lines:
        if _LOG_INITIAL_EVAL.search(line):
            seen_initial = True
            continue
        if seen_initial:
            m = _LOG_SEED_EVAL.search(line)
            if m:
                return _parse_log_metrics_string(m.group(2))
    return {}


def _parse_skydiscover_log(log_path: Path, initial_score: Optional[float]) -> list[dict]:
    """Parse a skydiscover log file into per-iteration records.

    Each record has iteration, outcome ('failed'|'succeeded'), and score.
    Failed iterations keep the previous best score (no new program was added).

    Handles evox co-evolution logs where solution-evolution iterations
    ("Iteration N: Program X ... completed") are interleaved with
    search-strategy meta-evolution failures ("Iteration N failed:").
    When both tracks exist, completed (solution) records take precedence.
    """
    records: list[dict] = []
    current_score = initial_score
    seen_initial = False
    pending_success: Optional[dict] = None
    # Completed iterations (from "Iteration N: Program X ... completed" lines)
    completed_records: dict[int, dict] = {}

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return records

    for line in lines:
        ts_match = _LOG_TS.match(line)
        if ts_match:
            from datetime import datetime as _dt
            ts_str = ts_match.group(1)
            ms = int(ts_match.group(2))
            timestamp = _dt.strptime(ts_str, "%Y-%m-%d %H:%M:%S").timestamp() + ms / 1000.0
        else:
            timestamp = None

        if _LOG_INITIAL_EVAL.search(line):
            seen_initial = True
            continue

        success_m = _LOG_ITER_SUCCESS.search(line)
        if success_m and seen_initial:
            pending_success = {
                "child_score": float(success_m.group(2)),
                "child_program_id": success_m.group(1),
                "timestamp": timestamp,
            }
            current_score = float(success_m.group(2))
            continue

        failed_m = _LOG_ITER_FAILED.search(line)
        if failed_m:
            iteration = int(failed_m.group(1))
            reason = failed_m.group(2).strip()
            if pending_success:
                rec = {
                    "iteration": iteration,
                    "outcome": "succeeded",
                    "timestamp": pending_success["timestamp"],
                }
                if pending_success["child_score"] is not None:
                    rec["child_score"] = pending_success["child_score"]
                records.append(rec)
                pending_success = None
            rec: dict = {
                "iteration": iteration,
                "outcome": "failed",
                "error": reason,
                "timestamp": timestamp,
            }
            if current_score is not None:
                rec["child_score"] = current_score
            records.append(rec)
            continue

        gepa_m = _LOG_ITER_GEPA_OUTCOME.search(line)
        if gepa_m:
            iteration = int(gepa_m.group(1))
            outcome = "succeeded" if gepa_m.group(2) == "ACCEPTED" else "failed"
            child_score = float(gepa_m.group(3))
            parent_score_val = float(gepa_m.group(4))
            rec = {
                "iteration": iteration,
                "outcome": outcome,
                "child_score": child_score,
                "parent_score": parent_score_val,
                "score_delta": child_score - parent_score_val,
                "timestamp": timestamp,
            }
            if pending_success and pending_success.get("child_program_id"):
                rec["child_program_id"] = pending_success["child_program_id"]
            pending_success = None
            records.append(rec)
            continue

        prog_m = _LOG_ITER_PROG_COMPLETED.search(line)
        if prog_m and seen_initial:
            prog_id = prog_m.group(2)
            iteration = int(prog_m.group(1))
            total_time = float(prog_m.group(4))
            rec = {
                "iteration": iteration,
                "outcome": "succeeded",
                "child_program_id": prog_id,
                "parent_program_id": prog_m.group(3),
                "timestamp": timestamp,
                "iteration_duration_seconds": total_time,
            }
            if pending_success:
                if pending_success.get("child_program_id", "").startswith(prog_id[:8]):
                    rec["child_score"] = pending_success["child_score"]
                    rec["timestamp"] = pending_success.get("timestamp") or timestamp
                else:
                    rec["child_score"] = pending_success["child_score"]
                pending_success = None
            completed_records[iteration] = rec
            continue

        # "Metrics: key=val, ..." line following a completed iteration
        metrics_m = _LOG_METRICS_LINE.search(line)
        if metrics_m and completed_records:
            last_iter = max(completed_records)
            last_rec = completed_records[last_iter]
            if "child_score" not in last_rec or last_rec.get("child_score") is None:
                parsed = _parse_log_metrics_string(metrics_m.group(1))
                cs = parsed.get("combined_score")
                if cs is not None:
                    last_rec["child_score"] = float(cs)
                    current_score = float(cs)
            continue

    # Flush any trailing successful eval (last iteration succeeded, run ended)
    if pending_success:
        next_iter = (records[-1]["iteration"] + 1) if records else 1
        rec = {
            "iteration": next_iter,
            "outcome": "succeeded",
            "timestamp": pending_success["timestamp"],
        }
        if pending_success["child_score"] is not None:
            rec["child_score"] = pending_success["child_score"]
        records.append(rec)

    # Merge: completed_records (from "Iteration N: ... completed") take
    # precedence over failed records (which may be from meta-evolution).
    if completed_records:
        failed_only = [r for r in records if r["iteration"] not in completed_records]
        all_records = list(completed_records.values()) + failed_only
        all_records.sort(key=lambda r: r["iteration"])
        return all_records

    # Fallback: if the log only contains eval-error lines (e.g. a run where every
    # LLM output was invalid and no iteration-tracking lines were emitted),
    # synthesize failed-iteration records from those error entries.
    if not records:
        records = _parse_eval_error_records(lines)

    return records


def _collect_skydiscover_checkpoints(root: Path) -> list[tuple[int, Path]]:
    """Return sorted (N, checkpoint_dir) pairs for a skydiscover output tree.

    Handles three layouts:
      1. root/checkpoint_N/               (flat — original expected layout)
      2. root/checkpoints/checkpoint_N/   (single run dir passed directly)
      3. root/<run>/checkpoints/checkpoint_N/  (run-collection dir, e.g. outputs/topk)
    """
    pattern = re.compile(r"^checkpoint_(\d+)$")

    def _scan_for_checkpoints(directory: Path) -> list[tuple[int, Path]]:
        found = []
        for entry in directory.iterdir():
            if entry.is_dir():
                m = pattern.match(entry.name)
                if m:
                    found.append((int(m.group(1)), entry))
        return found

    # Layout 1: flat checkpoint_N/ directly under root
    numbered = _scan_for_checkpoints(root)
    if numbered:
        return sorted(numbered, key=lambda t: t[0])

    # Layout 2: root/checkpoints/checkpoint_N/
    checkpoints_subdir = root / "checkpoints"
    if checkpoints_subdir.is_dir():
        numbered = _scan_for_checkpoints(checkpoints_subdir)
        if numbered:
            return sorted(numbered, key=lambda t: t[0])

    # Layout 3: root/<run>/checkpoints/checkpoint_N/ (collection of runs)
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpts_dir = run_dir / "checkpoints"
        if ckpts_dir.is_dir():
            numbered.extend(_scan_for_checkpoints(ckpts_dir))

    return sorted(numbered, key=lambda t: t[0])


def _load_program_json(prog_file: Path) -> Optional[dict]:
    try:
        return json.loads(prog_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_skydiscover_program(ckpt_dir: Path, best_id: Optional[str]) -> dict:
    """Read code and extra fields from programs/<uuid>.json if available."""
    if not best_id:
        return {}
    prog_file = ckpt_dir / "programs" / f"{best_id}.json"
    if prog_file.is_file():
        data = _load_program_json(prog_file)
        if data is not None:
            return data
    return {}


def _collect_skydiscover_run_dirs(root: Path) -> list[Path]:
    """Return skydiscover run directories under root, sorted by name.

    A run dir is any directory that contains a 'checkpoints/' or 'logs/' subdir.
    Handles:
      - root is a run dir itself (root/checkpoints/ or root/logs/ exists)
      - root is a collection of run dirs (root/<run>/checkpoints/ or root/<run>/logs/)
    """
    def _is_run_dir(d: Path) -> bool:
        return (d / "checkpoints").is_dir() or (d / "logs").is_dir()

    if _is_run_dir(root):
        return [root]

    runs = [d for d in sorted(root.iterdir()) if d.is_dir() and _is_run_dir(d)]
    return runs


def _first_not_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


def _extract_score(d: dict) -> Optional[float]:
    if "metrics" in d and isinstance(d["metrics"], dict):
        m = d["metrics"]
        from_metrics = _first_not_none(m.get("combined_score"), m.get("score"))
        if from_metrics is not None:
            return from_metrics
    return d.get("score")


def _best_checkpoint_state(run_dir: Path) -> dict:
    """Return score and code from the highest-numbered checkpoint in run_dir."""
    numbered = _collect_skydiscover_checkpoints(run_dir)
    if not numbered:
        return {}
    _, last_ckpt = numbered[-1]
    metadata = _read_first_json(last_ckpt, "metadata.json", "info.json")
    info = _read_first_json(last_ckpt, "best_program_info.json")
    solution = _read_first_json(last_ckpt, "best_solution.json", "solution.json")
    best_id = metadata.get("best_program_id")
    prog_data = _read_skydiscover_program(last_ckpt, best_id)

    score = _first_not_none(
        _extract_score(prog_data),
        _extract_score(info),
        _extract_score(solution),
        _extract_score(metadata),
    )
    code = (
        prog_data.get("solution") or prog_data.get("code")
        or solution.get("code") or solution.get("program")
    )
    return {"score": score, "code": code}


def _find_adaevolve_jsonl(run_dir: Path) -> Optional[Path]:
    """Return the first adaevolve_iteration_stats_*.jsonl found in run_dir, or None."""
    candidates = sorted(run_dir.glob("adaevolve_iteration_stats_*.jsonl"))
    return candidates[0] if candidates else None


def _build_program_map(run_dir: Path, required_ids: Optional[set] = None) -> dict:
    """Scan checkpoints/*/programs/*.json and return {program_id: program_data}.

    Scans in reverse checkpoint order so the first match per ID is the latest copy,
    and stops loading a file once its ID is already in the map.
    """
    program_map: dict = {}
    ckpts_dir = run_dir / "checkpoints"
    if not ckpts_dir.is_dir():
        return program_map
    for ckpt_dir in sorted(ckpts_dir.iterdir(), reverse=True):
        prog_dir = ckpt_dir / "programs"
        if not prog_dir.is_dir():
            continue
        for prog_file in prog_dir.glob("*.json"):
            pid = prog_file.stem
            if pid in program_map:
                continue
            if required_ids is not None and pid not in required_ids:
                continue
            data = _load_program_json(prog_file)
            if data is not None:
                program_map[data.get("id") or pid] = data
    return program_map


def _compute_unified_diff(parent_code: str, child_code: str) -> str:
    """Return a unified diff string between parent and child code."""
    parent_lines = parent_code.splitlines(keepends=True)
    child_lines = child_code.splitlines(keepends=True)
    return "".join(difflib.unified_diff(parent_lines, child_lines, fromfile="parent", tofile="child"))


def _enrich_jsonl_records(records: list[dict], run_dir: Path) -> None:
    """Enrich JSONL records in-place with system_message, mutation_type, and diffs from program JSONs.

    Diffs are computed from checkpoint program JSONs only when not already present in the records.
    """
    required_ids = {rec["child_program_id"] for rec in records if rec.get("child_program_id")}
    if not required_ids:
        return
    program_map = _build_program_map(run_dir, required_ids)
    if not program_map:
        return
    for rec in records:
        pid = rec.get("child_program_id")
        if not pid or pid not in program_map:
            continue
        prog = program_map[pid]
        for prompt_val in (prog.get("prompts") or {}).values():
            if isinstance(prompt_val, dict):
                system_msg = prompt_val.get("system")
                if system_msg and isinstance(system_msg, str) and system_msg.strip():
                    rec.setdefault("system_message", system_msg)
                break
        changes = (prog.get("metadata") or {}).get("changes")
        if changes:
            rec.setdefault("mutation_type", str(changes))

    # Compute diffs only when no records already have diff/code_diff
    records_with_ids = [r for r in records if r.get("child_program_id")]
    if any(r.get("diff") or r.get("code_diff") for r in records_with_ids):
        return
    records_needing_diffs = records_with_ids
    if not records_needing_diffs:
        return

    # Load child code and collect parent IDs
    child_map = {
        pid: program_map[pid]
        for rec in records_needing_diffs
        if (pid := rec.get("child_program_id")) and pid in program_map
    }
    parent_ids = {d.get("parent_id") for d in child_map.values() if d.get("parent_id")}
    parent_map = _build_program_map(run_dir, parent_ids) if parent_ids else {}

    for rec in records_needing_diffs:
        pid = rec.get("child_program_id")
        if not pid or pid not in child_map:
            continue
        child_prog = child_map[pid]
        child_code = child_prog.get("solution") or child_prog.get("code")
        if child_code:
            rec.setdefault("child_code", child_code)

        parent_id = child_prog.get("parent_id")
        if parent_id and parent_id in parent_map:
            parent_prog = parent_map[parent_id]
            parent_code = parent_prog.get("solution") or parent_prog.get("code")
            if parent_code:
                rec.setdefault("parent_code", parent_code)
                if child_code:
                    rec.setdefault("diff", _compute_unified_diff(parent_code, child_code))


def _parse_adaevolve_jsonl(jsonl_path: Path) -> list[dict]:
    """Convert adaevolve_iteration_stats_*.jsonl into standard per-iteration records."""
    records: list[dict] = []
    try:
        lines = jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return records

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        iteration = entry.get("iteration")
        timestamp = entry.get("timestamp")
        iter_result = entry.get("iteration_result", {})
        success = iter_result.get("success", True)
        error = iter_result.get("error")
        duration = iter_result.get("iteration_time_seconds")
        eval_time = iter_result.get("eval_time_seconds")
        llm_time = iter_result.get("llm_generation_time_seconds")
        child_prog = iter_result.get("child_program") or {}
        child_metrics = child_prog.get("metrics") or {}
        child_score = child_metrics.get("combined_score")
        child_id = child_prog.get("id")
        child_parent_id = child_prog.get("parent_id")

        global_info = entry.get("global", {})
        best_prog = global_info.get("best_program") or {}
        best_metrics = best_prog.get("metrics") or {}

        island_list = entry.get("islands", [])
        island_id = global_info.get("current_island_idx")
        if island_id is None and island_list:
            current = next((isl for isl in island_list if isl.get("is_current")), None)
            island_id = current.get("island_idx") if current else island_list[0].get("island_idx")

        evaluator_metrics = {
            k: v for k, v in child_metrics.items()
            if k != "combined_score" and not isinstance(v, dict)
        }
        nested_metrics = child_metrics.get("metrics") or {}
        if nested_metrics:
            evaluator_metrics.update(nested_metrics)

        rec: dict = {"iteration": iteration}
        rec["evaluation_status"] = "success" if success else "crash"
        if error:
            rec["error"] = str(error)
        if child_score is not None:
            rec["child_score"] = float(child_score)
        if child_id is not None:
            rec["child_program_id"] = child_id
        if child_parent_id is not None:
            rec["_parent_program_id"] = child_parent_id
        if timestamp is not None:
            try:
                from datetime import datetime as _dt
                rec["timestamp"] = _dt.fromisoformat(timestamp).timestamp()
            except (ValueError, TypeError):
                rec["timestamp"] = timestamp
        if duration is not None:
            rec["iteration_duration_seconds"] = float(duration)
        if eval_time is not None:
            rec["eval_time_seconds"] = float(eval_time)
        if llm_time is not None:
            rec["llm_generation_time_seconds"] = float(llm_time)
        if island_id is not None:
            rec["island_id"] = island_id
        if evaluator_metrics:
            rec["evaluator_metrics"] = evaluator_metrics

        # Detect paradigm breakthrough events
        paradigm = entry.get("paradigm", {})
        if paradigm.get("has_active_paradigm"):
            rec["framework_stagnation_event"] = "paradigm_shift"

        # Extract search parameters from adaevolve state for search-space analysis
        sampling = entry.get("sampling", {})
        parameters: dict = {}
        if "mode" in sampling:
            parameters["sampling_mode"] = sampling["mode"]
        if "intensity_used" in sampling:
            parameters["search_intensity"] = float(sampling["intensity_used"])
        island = island_list[0] if island_list else {}
        if "productivity" in island:
            parameters["island_productivity"] = float(island["productivity"])
        if "population_size" in island:
            parameters["population_size"] = int(island["population_size"])
        if "improvement_rate" in paradigm:
            parameters["improvement_rate"] = float(paradigm["improvement_rate"])
        if "is_stagnating" in paradigm:
            parameters["is_stagnating"] = paradigm["is_stagnating"]
        if parameters:
            rec["parameters"] = parameters

        records.append(rec)

    # Fill parent_score from previous child_score
    prev_score: Optional[float] = None
    for rec in records:
        if prev_score is not None:
            rec["parent_score"] = prev_score
            cs = rec.get("child_score")
            if cs is not None:
                rec["score_delta"] = float(cs) - prev_score
        cs = rec.get("child_score")
        if cs is not None:
            prev_score = float(cs)

    return records


def _find_seed_program_in_checkpoints(run_dir: Path, seed_id: Optional[str] = None) -> dict:
    """Return the seed program's data dict from checkpoint program JSONs.

    If seed_id is given, loads that specific program file. Otherwise scans
    checkpoints (earliest first) for a program with generation==0 or
    parent_id==None (which identifies the seed).
    Handles both the flat (checkpoint_N/ under run_dir) and nested
    (run_dir/checkpoints/checkpoint_N/) layouts.
    Returns {} if nothing is found.
    """
    # Use the same layout discovery as adapt_skydiscover for both paths
    numbered = _collect_skydiscover_checkpoints(run_dir)
    if seed_id:
        # Fast path: scan for the specific ID file
        for _, ckpt_dir in numbered:
            prog_file = ckpt_dir / "programs" / f"{seed_id}.json"
            if prog_file.is_file():
                data = _load_program_json(prog_file)
                if data:
                    return data
        return {}

    for _, ckpt_dir in numbered:
        prog_dir = ckpt_dir / "programs"
        if not prog_dir.is_dir():
            continue
        for prog_file in sorted(prog_dir.glob("*.json")):
            data = _load_program_json(prog_file)
            if data and (data.get("generation") == 0 or data.get("parent_id") is None):
                return data
    return {}


def _set_parent_metrics_from_program(first_record: dict, prog_data: dict) -> None:
    """Set parent_metrics on first_record from a program JSON data dict."""
    if first_record.get("parent_metrics"):
        return
    metrics = prog_data.get("metrics")
    if isinstance(metrics, dict) and metrics:
        first_record["parent_metrics"] = metrics


def adapt_skydiscover(checkpoint_dir: str) -> Iterator[dict]:
    """
    Reads SkyDiscover output directories, emitting one record per iteration.

    Primary source: adaevolve_iteration_stats_*.jsonl — rich per-iteration data
    logged directly by adaevolve.

    Secondary source: log file (logs/*.log) — gives one record per iteration,
    including failed attempts where the LLM was called but produced no valid code.

    Fallback: checkpoint dirs (checkpoints/checkpoint_N/) — one record per
    saved checkpoint when no log is available.

    The path may be a single run dir or a run-collection dir (e.g. outputs/topk).
    """
    root = Path(checkpoint_dir)
    run_dirs = _collect_skydiscover_run_dirs(root)

    if not run_dirs:
        # Legacy flat layout: checkpoint_N/ directly under root
        run_dirs = [root]

    for run_dir in run_dirs:
        jsonl_path = _find_adaevolve_jsonl(run_dir)
        if jsonl_path:
            jsonl_records = _parse_adaevolve_jsonl(jsonl_path)
            if jsonl_records:
                _enrich_jsonl_records(jsonl_records, run_dir)
                # Inject seed parent_metrics: look up the seed program in
                # checkpoints using the parent_id stored from the first record,
                # then fall back to log parsing if checkpoints don't have it.
                first = jsonl_records[0]
                seed_id = first.get("_parent_program_id")
                seed_prog = _find_seed_program_in_checkpoints(run_dir, seed_id)
                if seed_prog:
                    _set_parent_metrics_from_program(first, seed_prog)
                log_path = _find_run_log(run_dir)
                if log_path:
                    seed_m = _extract_seed_metrics_from_log(log_path)
                    if seed_m and not first.get("parent_metrics"):
                        first["parent_metrics"] = seed_m
                    infra_signals = _extract_infra_log_signals(log_path)
                    if infra_signals:
                        first["_infra_log_signals"] = infra_signals
                # Remove temp field from all records
                for rec in jsonl_records:
                    rec.pop("_parent_program_id", None)
                for rec in jsonl_records:
                    yield rec
                continue

        log_path = _find_run_log(run_dir)
        best_state = _best_checkpoint_state(run_dir)
        initial_score = best_state.get("score")
        best_code = best_state.get("code")

        if log_path:
            if _is_openevolve_log(log_path):
                log_records = _parse_openevolve_log_full(log_path)
            else:
                log_records = _parse_skydiscover_log(log_path, initial_score)
            if log_records:
                # Inject seed parent_metrics from log (first eval after init)
                seed_m = _extract_seed_metrics_from_log(log_path)
                if seed_m:
                    log_records[0]["parent_metrics"] = seed_m
                if not _is_openevolve_log(log_path):
                    # For skydiscover logs, chain parent_score from previous records
                    # (openevolve log parser already resolves parent_score via score_map)
                    prev_score: Optional[float] = None
                    for rec in log_records:
                        if prev_score is not None:
                            rec.setdefault("parent_score", prev_score)
                            if "child_score" in rec:
                                rec.setdefault(
                                    "score_delta",
                                    rec["child_score"] - rec.get("parent_score", prev_score),
                                )
                        cs = rec.get("child_score")
                        if cs is not None:
                            prev_score = cs if prev_score is None else max(prev_score, cs)
                # Attach best known code to the last record of this run
                if best_code:
                    log_records[-1].setdefault("child_code", best_code)
                for rec in log_records:
                    yield rec
                continue

        # Fallback: checkpoint-based records
        numbered = _collect_skydiscover_checkpoints(run_dir)
        prev_score = None
        prev_code = None
        # Pre-load seed program metrics for parent_metrics injection on first record
        seed_prog_ckpt = _find_seed_program_in_checkpoints(run_dir)
        first_ckpt_record = True
        for n, ckpt_dir in numbered:
            metadata = _read_first_json(ckpt_dir, "metadata.json", "info.json")
            evo_state = _read_first_json(ckpt_dir, "evolution_state.json")
            info = _read_first_json(ckpt_dir, "best_program_info.json")
            solution = _read_first_json(ckpt_dir, "best_solution.json", "solution.json")
            best_id = metadata.get("best_program_id")
            prog_data = _read_skydiscover_program(ckpt_dir, best_id)

            child_score = _first_not_none(
                _extract_score(prog_data),
                _extract_score(info),
                _extract_score(solution),
                _extract_score(metadata),
                _extract_score(evo_state),
            )
            child_code = (
                prog_data.get("solution") or prog_data.get("code")
                or solution.get("code") or solution.get("program")
                or metadata.get("code") or metadata.get("program")
            )
            iteration = (
                metadata.get("last_iteration") or info.get("current_iteration")
                or metadata.get("iteration") or n
            )
            model = metadata.get("model") or evo_state.get("model") or prog_data.get("model")
            mutation_type = (
                metadata.get("mutation_type") or evo_state.get("mutation_type")
                or prog_data.get("mutation_type")
            )
            timestamp = (
                prog_data.get("timestamp") or info.get("timestamp")
                or metadata.get("timestamp") or evo_state.get("timestamp")
            )
            island_id = (
                metadata.get("island_id") or evo_state.get("island_id")
                or prog_data.get("island_id")
            )
            reasoning_trace = (
                metadata.get("reasoning_trace") or evo_state.get("reasoning_trace")
                or prog_data.get("reasoning_trace")
            )

            rec: dict = {"iteration": iteration}
            if child_score is not None:
                rec["child_score"] = float(child_score)
            if prev_score is not None:
                rec["parent_score"] = float(prev_score)
            if prev_score is not None and child_score is not None:
                rec["score_delta"] = float(child_score) - float(prev_score)
            if child_code is not None:
                rec["child_code"] = child_code
            if prev_code is not None:
                rec["parent_code"] = prev_code
            if model is not None:
                rec["model"] = model
            if mutation_type is not None:
                rec["mutation_type"] = mutation_type
            if timestamp is not None:
                rec["timestamp"] = timestamp
            if island_id is not None:
                rec["island_id"] = island_id
            if reasoning_trace is not None:
                rec["reasoning_trace"] = reasoning_trace

            paradigm_shift = (
                evo_state.get("paradigm_shift") or metadata.get("paradigm_shift")
            )
            if paradigm_shift:
                rec["framework_stagnation_event"] = "paradigm_shift"

            evaluator_metrics = (
                prog_data.get("metrics") or info.get("metrics")
                or solution.get("evaluator_metrics") or metadata.get("evaluator_metrics")
                or evo_state.get("evaluator_metrics")
            )
            if evaluator_metrics and isinstance(evaluator_metrics, dict):
                rec["evaluator_metrics"] = evaluator_metrics

            if first_ckpt_record and seed_prog_ckpt:
                _set_parent_metrics_from_program(rec, seed_prog_ckpt)
            first_ckpt_record = False

            if child_score is not None:
                prev_score = float(child_score)
            if child_code is not None:
                prev_code = child_code

            yield rec


# ---------------------------------------------------------------------------
# adapt_shinkaevolve
# ---------------------------------------------------------------------------

def adapt_shinkaevolve(db_path: str) -> Iterator[dict]:
    """
    Reads a ShinkaEvolve SQLite database.

    Tries table 'iterations' first, falls back to 'runs'.

    Column → standard field mapping:
      id             (ignored as record identifier)
      iteration      → iteration
      score          → child_score
      parent_score   → parent_score
      code           → child_code
      parent_code    → parent_code
      diff           → diff
      model          → model
      mutation_type  → mutation_type
      status         → evaluation_status  (normalised)
      tokens_used    → llm_tokens_used
      cost_usd       → llm_cost_usd
      latency_ms     → llm_latency_ms
      timestamp      → timestamp
      island_id      → island_id
      metadata       → merged into record (parsed as JSON)

    Sets framework_stagnation_event = "island_spawn" when island_id changes.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()

        # Determine table name
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('iterations','runs')"
        )
        tables = {row[0] for row in cursor.fetchall()}
        if "iterations" in tables:
            table = "iterations"
        elif "runs" in tables:
            table = "runs"
        else:
            conn.close()
            return

        cursor.execute(f"SELECT * FROM {table} ORDER BY iteration ASC")  # noqa: S608
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        prev_island_id: Optional[str] = None

        for row in rows:
            row_dict = dict(zip(columns, row))
            rec: dict = {}

            # --- iteration ---
            it = row_dict.get("iteration")
            if it is not None:
                rec["iteration"] = int(it)

            # --- scores ---
            score = row_dict.get("score")
            if score is not None:
                rec["child_score"] = float(score)

            ps = row_dict.get("parent_score")
            if ps is not None:
                rec["parent_score"] = float(ps)

            if rec.get("child_score") is not None and rec.get("parent_score") is not None:
                rec["score_delta"] = rec["child_score"] - rec["parent_score"]

            # --- code ---
            code = row_dict.get("code")
            if code is not None:
                rec["child_code"] = code

            pcode = row_dict.get("parent_code")
            if pcode is not None:
                rec["parent_code"] = pcode

            diff = row_dict.get("diff")
            if diff is not None:
                rec["diff"] = diff

            # --- model / mutation ---
            model = row_dict.get("model")
            if model is not None:
                rec["model"] = model

            mt = row_dict.get("mutation_type")
            if mt is not None:
                rec["mutation_type"] = mt

            # --- status ---
            status = row_dict.get("status")
            if status is not None:
                rec["evaluation_status"] = _normalise_status(status)

            # --- LLM fields ---
            tokens = row_dict.get("tokens_used")
            if tokens is not None:
                rec["llm_tokens_used"] = int(tokens)

            cost = row_dict.get("cost_usd")
            if cost is not None:
                rec["llm_cost_usd"] = float(cost)

            latency = row_dict.get("latency_ms")
            if latency is not None:
                rec["llm_latency_ms"] = float(latency)

            # --- timestamp ---
            ts = row_dict.get("timestamp")
            if ts is not None:
                rec["timestamp"] = float(ts)

            # --- island ---
            island_id = row_dict.get("island_id")
            if island_id is not None:
                rec["island_id"] = str(island_id)
                if prev_island_id is not None and str(island_id) != prev_island_id:
                    rec["framework_stagnation_event"] = "island_spawn"
                prev_island_id = str(island_id)
            elif island_id is None and "island_id" in row_dict:
                pass  # explicit NULL island_id – leave field absent

            # --- explicit parent_metrics column (JSON blob) ---
            raw_pm = row_dict.get("parent_metrics")
            if raw_pm and not rec.get("parent_metrics"):
                try:
                    pm = json.loads(raw_pm) if isinstance(raw_pm, str) else raw_pm
                    if isinstance(pm, dict) and pm:
                        rec["parent_metrics"] = pm
                except (json.JSONDecodeError, TypeError):
                    pass

            # --- metadata JSON merge ---
            raw_meta = row_dict.get("metadata")
            if raw_meta:
                try:
                    meta_dict = json.loads(raw_meta)
                    if isinstance(meta_dict, dict):
                        # meta fields fill in gaps; existing keys take priority
                        for k, v in meta_dict.items():
                            if k not in rec and v is not None:
                                rec[k] = v
                except (json.JSONDecodeError, TypeError):
                    pass

            yield rec

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# adapt_openevolve — log helpers
# ---------------------------------------------------------------------------

_OE_LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+)")
_OE_LOG_ITER_ERROR = re.compile(r"WARNING.*Iteration (\d+) error: (.+)")
_OE_LOG_MAP_ELITES_NEW = re.compile(r"New MAP-Elites cell occupied in island (\d+)")
_OE_LOG_MAP_ELITES_IMPROVED = re.compile(r"Island (\d+) MAP-Elites cell improved")
_OE_LOG_ITER_COMPLETED = re.compile(
    r"Iteration (\d+): Program (\S+) \(parent: (\S+)\) completed in [\d.]+s"
)
_OE_LOG_METRICS = re.compile(r"Metrics: (.+)")
_OE_LOG_COMBINED_SCORE = re.compile(r"combined_score=(-?[\d.e]+)")
_OE_LOG_SEED_EVAL = re.compile(
    r"Evaluated program (\S+) in [\d.]+s: combined_score=(-?[\d.e]+)"
)
_OE_LOG_KV_SIGNED = re.compile(r"\b(\w+)=(-?[\d.]+)")


def _is_openevolve_log(log_path: Path) -> bool:
    """Return True when log_path looks like an OpenEvolve run log."""
    return log_path.name.startswith("openevolve_")


def _parse_openevolve_metrics_str(metrics_str: str) -> dict:
    """Parse 'key=value' metrics string including negative values; skips nested dicts."""
    remaining = re.sub(r"\w+=\{[^}]*\}", "", metrics_str)
    result: dict = {}
    for m in _OE_LOG_KV_SIGNED.finditer(remaining):
        try:
            result[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return result


def _parse_openevolve_log_full(log_path: Path) -> list[dict]:
    """Parse all iteration records from an OpenEvolve log file.

    Handles the parallel controller format:
      - "Iteration N: Program <child> (parent: <parent>) completed in <t>s"
        followed by "Metrics: combined_score=<score>, ..."
      - "WARNING ... Iteration N error: <msg>"

    Also captures the seed score so parent_score/score_delta can be filled in.
    Returns records sorted by iteration.
    """
    from datetime import datetime as _dt

    records: list[dict] = []
    score_map: dict[str, float] = {}   # program_id -> combined_score
    island_map: dict[str, int] = {}    # program_id -> island_id

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return records

    # Extract seed score and island (MAP-Elites line follows the seed eval line)
    for idx, line in enumerate(lines):
        seed_m = _OE_LOG_SEED_EVAL.search(line)
        if seed_m:
            seed_id = seed_m.group(1)
            score_map[seed_id] = float(seed_m.group(2))
            for j in range(idx + 1, min(idx + 4, len(lines))):
                isle_m = _OE_LOG_MAP_ELITES_NEW.search(lines[j])
                if isle_m:
                    island_map[seed_id] = int(isle_m.group(1))
                    break
            break

    # pending_island is set by a MAP-Elites line and consumed by the next
    # "Iteration N: Program..." line. The database log writes the MAP-Elites
    # line 1-2 lines before the process_parallel completion line, and not at
    # all when the child doesn't improve or occupy any cell.
    pending_island: Optional[int] = None

    i = 0
    while i < len(lines):
        line = lines[i]

        ts_m = _OE_LOG_TS.match(line)
        timestamp = None
        if ts_m:
            try:
                ts_str = ts_m.group(1)
                ms = int(ts_m.group(2))
                timestamp = _dt.strptime(ts_str, "%Y-%m-%d %H:%M:%S").timestamp() + ms / 1000.0
            except ValueError:
                pass

        island_m = _OE_LOG_MAP_ELITES_NEW.search(line) or _OE_LOG_MAP_ELITES_IMPROVED.search(line)
        if island_m:
            pending_island = int(island_m.group(1))
            i += 1
            continue

        error_m = _OE_LOG_ITER_ERROR.search(line)
        if error_m:
            rec: dict = {
                "iteration": int(error_m.group(1)),
                "evaluation_status": "crash",
                "error": error_m.group(2).strip(),
            }
            if timestamp:
                rec["timestamp"] = timestamp
            pending_island = None
            records.append(rec)
            i += 1
            continue

        completed_m = _OE_LOG_ITER_COMPLETED.search(line)
        if completed_m:
            child_id = completed_m.group(2)
            rec = {
                "iteration": int(completed_m.group(1)),
                "evaluation_status": "success",
                "child_program_id": child_id,
                "parent_program_id": completed_m.group(3),
            }
            if timestamp:
                rec["timestamp"] = timestamp

            if pending_island is not None:
                rec["island_id"] = pending_island
                rec["parameters"] = {"island_id": pending_island}
            pending_island = None

            for j in range(i + 1, min(i + 3, len(lines))):
                metrics_m = _OE_LOG_METRICS.search(lines[j])
                if metrics_m:
                    metrics_str = metrics_m.group(1)
                    score_m = _OE_LOG_COMBINED_SCORE.search(metrics_str)
                    if score_m:
                        child_score = float(score_m.group(1))
                        rec["child_score"] = child_score
                        score_map[child_id] = child_score
                    flat = _parse_openevolve_metrics_str(metrics_str)
                    if flat:
                        rec["evaluator_metrics"] = flat
                    break

            records.append(rec)
            i += 1
            continue

        i += 1

    for rec in records:
        parent_id = rec.get("parent_program_id")
        if parent_id and "parent_score" not in rec and parent_id in score_map:
            rec["parent_score"] = score_map[parent_id]
        if "child_score" in rec and "parent_score" in rec and "score_delta" not in rec:
            rec["score_delta"] = rec["child_score"] - rec["parent_score"]

    # Populate island_map from records that already have island_id, then propagate
    # to iterations where the child didn't improve any MAP-Elites cell (no direct
    # island log line). In OpenEvolve, the child is always generated from a parent
    # in a specific island and placed in the same island.
    for rec in records:
        child_id = rec.get("child_program_id")
        if child_id and rec.get("island_id") is not None:
            island_map[child_id] = rec["island_id"]

    for _ in range(3):  # up to 3 passes for chains of unresolved parents
        for rec in records:
            if rec.get("island_id") is not None:
                continue
            parent_id = rec.get("parent_program_id")
            if parent_id and parent_id in island_map:
                island_id = island_map[parent_id]
                rec["island_id"] = island_id
                rec.setdefault("parameters", {})["island_id"] = island_id
                child_id = rec.get("child_program_id")
                if child_id:
                    island_map[child_id] = island_id

    records.sort(key=lambda r: r.get("iteration") if r.get("iteration") is not None else 0)
    return records


def _parse_openevolve_log(log_path: Path) -> list[dict]:
    """Return one record per failed iteration found in an OpenEvolve log."""
    failed: list[dict] = []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return failed

    for line in lines:
        error_m = _OE_LOG_ITER_ERROR.search(line)
        if not error_m:
            continue
        iteration = int(error_m.group(1))
        error_msg = error_m.group(2).strip()
        rec: dict = {
            "iteration": iteration,
            "evaluation_status": "crash",
            "error": error_msg,
        }
        ts_m = _OE_LOG_TS.match(line)
        if ts_m:
            from datetime import datetime as _dt
            ts_str = ts_m.group(1)
            ms = int(ts_m.group(2))
            rec["timestamp"] = _dt.strptime(ts_str, "%Y-%m-%d %H:%M:%S").timestamp() + ms / 1000.0
        failed.append(rec)
    return failed


# ---------------------------------------------------------------------------
# adapt_openevolve
# ---------------------------------------------------------------------------

def adapt_openevolve(
    checkpoint_dir: str,
    trace_path: Optional[str] = None,
) -> Iterator[dict]:
    """
    Reads OpenEvolve output.

    Auto-detects evolution_trace.jsonl inside checkpoint_dir when trace_path
    is not provided. Also reads logs/*.log to recover failed iterations that
    were not written to the trace (e.g. LLM budget errors).

    Field mapping from trace records:
      'child_metrics.combined_score' → child_score
      'parent_metrics.combined_score' → parent_score
      'improvement_delta.combined_score' → score_delta
      'child_metrics' → evaluator_metrics
      'program' | 'code' → child_code (top-level fallbacks)

    OpenEvolve's parallel controller may write records out of order, so all
    records are buffered and sorted by iteration before yielding.
    """
    records: list[dict] = []

    # Auto-detect trace file inside checkpoint_dir if not explicitly provided
    if not trace_path:
        auto_trace = Path(checkpoint_dir) / "evolution_trace.jsonl"
        if auto_trace.is_file():
            trace_path = str(auto_trace)

    if trace_path and Path(trace_path).is_file():
        # Primary path: read evolution_trace.jsonl
        with open(trace_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw: dict = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rec: dict = {}

                # Map passthrough fields
                for src, dst in (
                    ("iteration", "iteration"),
                    ("evaluation_status", "evaluation_status"),
                    ("format_valid", "format_valid"),
                    ("mutation_type", "mutation_type"),
                    ("model", "model"),
                    ("parent_code", "parent_code"),
                    ("diff", "diff"),
                    ("evolved_block_only", "evolved_block_only"),
                    ("signature_preserved", "signature_preserved"),
                    ("evaluator_metrics", "evaluator_metrics"),
                    ("cascade_stage_failed", "cascade_stage_failed"),
                    ("evaluator_artifacts", "evaluator_artifacts"),
                    ("score_std", "score_std"),
                    ("num_runs", "num_runs"),
                    ("llm_tokens_used", "llm_tokens_used"),
                    ("llm_cost_usd", "llm_cost_usd"),
                    ("llm_latency_ms", "llm_latency_ms"),
                    ("timestamp", "timestamp"),
                    ("island_id", "island_id"),
                    ("framework_stagnation_event", "framework_stagnation_event"),
                    ("reasoning_trace", "reasoning_trace"),
                    ("meta_suggestion", "meta_suggestion"),
                    ("followed_suggestion", "followed_suggestion"),
                    ("early_stop_suggested", "early_stop_suggested"),
                ):
                    val = raw.get(src)
                    if val is not None:
                        rec[dst] = val

                # child_score: nested child_metrics.combined_score, then top-level
                child_metrics = raw.get("child_metrics")
                if isinstance(child_metrics, dict):
                    cs = child_metrics.get("combined_score") or child_metrics.get("score") or child_metrics.get("fitness")
                else:
                    cs = None
                if cs is None:
                    cs = raw.get("score") if raw.get("score") is not None else raw.get("fitness")
                if cs is not None:
                    rec["child_score"] = float(cs)

                # parent_score: nested parent_metrics.combined_score, then top-level
                parent_metrics = raw.get("parent_metrics")
                if isinstance(parent_metrics, dict):
                    ps = parent_metrics.get("combined_score") or parent_metrics.get("score") or parent_metrics.get("fitness")
                else:
                    ps = None
                if ps is None:
                    ps = raw.get("parent_score")
                if ps is not None:
                    rec["parent_score"] = float(ps)

                # score_delta: improvement_delta.combined_score, then top-level
                improvement_delta = raw.get("improvement_delta")
                if isinstance(improvement_delta, dict):
                    sd = improvement_delta.get("combined_score")
                else:
                    sd = None
                if sd is None:
                    sd = raw.get("score_delta")
                if sd is not None:
                    rec["score_delta"] = float(sd)

                # evaluator_metrics: use child_metrics if not already set
                if not rec.get("evaluator_metrics") and isinstance(child_metrics, dict):
                    rec["evaluator_metrics"] = child_metrics

                # parent_metrics: preserve full dict for sub-metric seed extraction
                if isinstance(parent_metrics, dict):
                    rec["parent_metrics"] = parent_metrics

                # child_code: prefer 'program', then 'code'
                cc = raw.get("program") if raw.get("program") is not None else raw.get("child_code") or raw.get("code")
                if cc is not None:
                    rec["child_code"] = cc
                    rec["code"] = cc  # alias for Judge E (exploration_structure)

                # system_message: extract from prompt dict for Judge D (semantic_compliance)
                prompt_field = raw.get("prompt")
                if isinstance(prompt_field, dict) and prompt_field.get("system"):
                    rec["system_message"] = prompt_field["system"]

                # Synthesize parameters dict from OpenEvolve search-space fields so
                # search_space_analyzer can detect dimensionality and diversity.
                params: dict = {}
                for field in ("mutation_type", "island_id", "model", "generation"):
                    val = raw.get(field)
                    if val is not None:
                        params[field] = val
                meta = raw.get("metadata")
                if isinstance(meta, dict) and meta.get("changes"):
                    params["change_type"] = meta["changes"]
                if params:
                    rec["parameters"] = params

                records.append(rec)

        # Supplement with failed iterations from the log file
        logs_dir = Path(checkpoint_dir) / "logs"
        if logs_dir.is_dir():
            log_files = sorted(logs_dir.glob("*.log"))
            if log_files:
                existing_iterations = {r.get("iteration") for r in records}
                for failed_rec in _parse_openevolve_log(log_files[0]):
                    if failed_rec["iteration"] not in existing_iterations:
                        records.append(failed_rec)

    else:
        # No trace file: try log-based parsing first
        logs_dir = Path(checkpoint_dir) / "logs"
        if logs_dir.is_dir():
            log_files = sorted(logs_dir.glob("*.log"))
            if log_files:
                records.extend(_parse_openevolve_log_full(log_files[0]))

        if not records:
            # Fallback: scan checkpoint directories (same logic as SkyDiscover)
            root = Path(checkpoint_dir)
            if root.is_dir():
                pattern = re.compile(r"^checkpoint_(\d+)$")
                numbered: list[tuple[int, Path]] = []

                def _scan_ckpts(directory: Path) -> list[tuple[int, Path]]:
                    found = []
                    for entry in directory.iterdir():
                        if entry.is_dir():
                            m = pattern.match(entry.name)
                            if m:
                                found.append((int(m.group(1)), entry))
                    return found

                numbered = _scan_ckpts(root)
                # Also check root/checkpoints/ subdirectory (common OpenEvolve layout)
                if not numbered:
                    ckpts_subdir = root / "checkpoints"
                    if ckpts_subdir.is_dir():
                        numbered = _scan_ckpts(ckpts_subdir)
                numbered.sort(key=lambda t: t[0])

                prev_score: Optional[float] = None
                prev_code: Optional[str] = None

                for n, ckpt_dir in numbered:
                    solution = _read_first_json(ckpt_dir, "best_solution.json", "solution.json")
                    metadata = _read_first_json(ckpt_dir, "metadata.json", "info.json")
                    evo_state = _read_first_json(ckpt_dir, "evolution_state.json")

                    child_score = solution.get("score") or solution.get("fitness")
                    if child_score is None:
                        child_score = metadata.get("score") or metadata.get("fitness")
                    if child_score is None:
                        child_score = evo_state.get("score") or evo_state.get("fitness")

                    child_code = (
                        solution.get("program")
                        or solution.get("code")
                        or metadata.get("program")
                        or metadata.get("code")
                    )

                    rec = {"iteration": metadata.get("iteration", n)}
                    if child_score is not None:
                        rec["child_score"] = float(child_score)
                    if prev_score is not None:
                        rec["parent_score"] = float(prev_score)
                        if child_score is not None:
                            rec["score_delta"] = float(child_score) - float(prev_score)
                    if child_code is not None:
                        rec["child_code"] = child_code
                    if prev_code is not None:
                        rec["parent_code"] = prev_code

                    for field in ("model", "mutation_type", "timestamp", "island_id",
                                  "reasoning_trace", "evaluator_metrics"):
                        val = metadata.get(field) or evo_state.get(field)
                        if val is not None:
                            rec[field] = val

                    if child_score is not None:
                        prev_score = float(child_score)
                    if child_code is not None:
                        prev_code = child_code

                    records.append(rec)

    # Sort by iteration (parallel writes may be out of order)
    records.sort(key=lambda r: r.get("iteration") if r.get("iteration") is not None else 0)

    yield from records


# ---------------------------------------------------------------------------
# detect_algorithm_class  (public helper)
# ---------------------------------------------------------------------------

_ALWAYS_POPULATION_EVOLUTIONARY = {"skydiscover", "openevolve", "shinkaevolve"}


_LOG_NAME_DATE_SUFFIX = re.compile(r"_\d{8}_\d{6}$")


def detect_algorithm_name(source: str, path: str) -> Optional[str]:
    """Infer the specific algorithm name from output file naming conventions.

    For skydiscover runs the JSONL stats file is named
    ``{algorithm}_iteration_stats_*.jsonl`` (e.g. ``adaevolve_iteration_stats_*.jsonl``).
    The prefix before ``_iteration_stats_`` is the algorithm name.

    Fallback: if no JSONL is found, inspect log file names matching the pattern
    ``{algorithm}_{YYYYMMDD}_{HHMMSS}.log`` (e.g. ``gepa_20260312_143339.log``).

    Returns the algorithm name string (e.g. ``"adaevolve"``, ``"gepa"``) or
    ``None`` when the name cannot be determined from the available files.
    """
    source_key = source.lower().strip()
    if source_key == "skydiscover" and path:
        for match in Path(path).rglob("*_iteration_stats_*.jsonl"):
            stem = match.stem
            idx = stem.find("_iteration_stats_")
            if idx > 0:
                return stem[:idx]
        # Fallback: infer from log file name ({algo}_{YYYYMMDD}_{HHMMSS}.log)
        for log_match in Path(path).rglob("*.log"):
            stem = log_match.stem
            stripped = _LOG_NAME_DATE_SUFFIX.sub("", stem)
            if stripped and stripped != stem:
                return stripped
    return None


def detect_algorithm_class(source: str, records: List[dict]) -> str:
    """Infer the algorithm class from the ingestion source and record contents.

    Returns one of:
      "population_evolutionary" — multi-island / population-based search
                                  (OpenEvolve, AdaEvolve, ShinkaEvolve, EvoX, GEPA, …)
      "bayesian_optimization"   — BO / surrogate-model-guided search
      "serial_refinement"       — sequential single-chain refinement

    Detection logic
    ---------------
    - Structured sources (skydiscover / openevolve / shinkaevolve) are always
      classified as population_evolutionary.
    - For jsonl records the first 10 records are inspected:
        * Any record with a non-None island_id  → population_evolutionary
        * Any record with a 'parameters' dict containing numeric values
          (hinting at a BO search space) → bayesian_optimization
        * Otherwise → serial_refinement
    """
    source_key = source.lower().strip()
    if source_key in _ALWAYS_POPULATION_EVOLUTIONARY:
        return "population_evolutionary"

    sample = records[:10]
    for rec in sample:
        if rec.get("island_id") is not None:
            return "population_evolutionary"

    for rec in sample:
        params = rec.get("parameters") or {}
        if any(isinstance(v, (int, float)) for v in params.values()):
            return "bayesian_optimization"

    return "serial_refinement"


# ---------------------------------------------------------------------------
# load_evolve_records  (public entry-point)
# ---------------------------------------------------------------------------

_ADAPTERS = {
    "jsonl": load_jsonl,
    "skydiscover": adapt_skydiscover,
    "shinkaevolve": adapt_shinkaevolve,
    "openevolve": adapt_openevolve,
}


def load_evolve_records(source: str, path: str, **kwargs) -> List[dict]:
    """
    Auto-detects format and returns normalised records as a list.

    source : "jsonl" | "skydiscover" | "shinkaevolve" | "openevolve"
    path   : path to the file or directory
    **kwargs: forwarded to the specific adapter
              (e.g. trace_path="..." for openevolve)

    After loading:
      - Applies _fill_derived_fields() to ensure score_delta,
        evaluation_status, and format_valid are present.
      - Sorts records by iteration.
    """
    source_key = source.lower().strip()
    adapter = _ADAPTERS.get(source_key)
    if adapter is None:
        raise ValueError(
            f"Unknown source {source!r}. "
            f"Valid options: {sorted(_ADAPTERS)}"
        )

    records: List[dict] = list(adapter(path, **kwargs))  # type: ignore[call-arg]

    records = _fill_derived_fields(records)

    # Final sort by iteration (adapters may already sort, but be defensive)
    records.sort(
        key=lambda r: r.get("iteration") if r.get("iteration") is not None else 0
    )

    return records
