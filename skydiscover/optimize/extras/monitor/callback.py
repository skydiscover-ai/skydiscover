"""
Monitor callback factory for the discovery loop.

Creates a callback function that serializes program data
and pushes it to the MonitorServer for live broadcasting.
"""

import logging
import time
from typing import Any, Callable, Dict, Optional

from skydiscover.optimize.extras.monitor.server import MonitorServer

logger = logging.getLogger(__name__)

SOLUTION_SNIPPET_LENGTH = 500


def create_monitor_callback(
    server: MonitorServer,
    database: Any,
    start_time: float,
) -> Callable:
    """Create an iteration callback that pushes program data to the monitor server."""

    def _callback(program: Any, iteration: int, result: Any = None) -> None:
        """Push a new program event to the monitor. Never raises."""
        try:
            _push_program_event(server, database, program, iteration, result, start_time)
        except Exception:
            # Never crash discovery process due to monitor
            logger.debug("Monitor callback error", exc_info=True)

    return _callback


def _push_program_event(
    server: MonitorServer,
    database: Any,
    program: Any,
    iteration: int,
    result: Any,
    start_time: float,
) -> None:
    """Serialize program data and push to the monitor server."""
    metrics = program.metrics or {}
    score = metrics.get("combined_score", 0.0)
    if not isinstance(score, (int, float)):
        score = 0.0

    parent_id = getattr(program, "parent_id", None)
    parent_score = None
    parent_iter = None
    parent_solution = ""
    if parent_id:
        parent_prog = database.get(parent_id) if hasattr(database, "get") else None
        if parent_prog:
            parent_metrics = parent_prog.metrics or {}
            parent_score = parent_metrics.get("combined_score")
            parent_iter = getattr(parent_prog, "iteration_found", None)
            parent_solution = getattr(parent_prog, "solution", "")

    context_ids = getattr(program, "other_context_ids", None) or []
    context_scores = []
    for cid in context_ids:
        cp = database.get(cid) if hasattr(database, "get") else None
        if cp and cp.metrics:
            context_scores.append(cp.metrics.get("combined_score"))
        else:
            context_scores.append(None)

    # Label type from parent_info or metadata
    label_type = None
    parent_info = getattr(program, "parent_info", None)
    if parent_info and isinstance(parent_info, (list, tuple)) and len(parent_info) >= 1:
        label_str = str(parent_info[0]).lower()
        if "diverge" in label_str:
            label_type = "diverge"
        elif "refine" in label_str:
            label_type = "refine"
        elif "crossover" in label_str:
            label_type = "crossover"
    md = getattr(program, "metadata", {}) or {}
    if label_type is None:
        label_type = md.get("label_type", "unknown")

    island = md.get("island")

    is_best = getattr(database, "best_program_id", None) == program.id

    # Solution snippet (first N chars)
    code = getattr(program, "solution", "") or ""
    solution_snippet = code[:SOLUTION_SNIPPET_LENGTH]

    # Image path from metadata (image evolution mode)
    image_path = (getattr(program, "metadata", {}) or {}).get("image_path")

    total_programs = len(database.programs) if hasattr(database, "programs") else 0
    best_prog = database.get_best_program() if hasattr(database, "get_best_program") else None
    best_score = 0.0
    if best_prog and best_prog.metrics:
        best_score = best_prog.metrics.get("combined_score", 0.0)

    elapsed = time.time() - start_time
    rate = total_programs / elapsed * 60 if elapsed > 0 else 0.0

    iters_since_improvement = 0
    if best_prog:
        best_iter = getattr(best_prog, "iteration_found", 0)
        iters_since_improvement = iteration - best_iter

    prog_data = {
        "id": program.id,
        "iteration": iteration,
        "score": score,
        "metrics": _safe_metrics(metrics),
        "parent_id": parent_id,
        "parent_score": parent_score,
        "parent_iter": parent_iter,
        "context_ids": context_ids,
        "context_scores": context_scores,
        "label_type": label_type,
        "solution_snippet": solution_snippet,
        "island": island,
        "is_best": is_best,
        "generation": getattr(program, "generation", 0),
        "image_path": image_path,
    }

    stats = {
        "total_programs": total_programs,
        "current_iteration": iteration,
        "best_score": best_score,
        "iterations_since_improvement": iters_since_improvement,
        "programs_per_min": round(rate, 1),
        "elapsed_seconds": round(elapsed, 1),
    }

    event = {
        "type": "new_program",
        "program": prog_data,
        "stats": stats,
        "is_best": is_best,
        "full_solution": code[: server.max_solution_length],
        "parent_full_solution": (
            parent_solution[: server.max_solution_length] if parent_solution else ""
        ),
    }

    server.push_event(event)


def create_external_callback(
    server: MonitorServer,
    start_time: float,
) -> Callable:
    """Create a monitor callback for external backends (no ProgramDatabase needed).

    Maintains its own lightweight program store for parent lookups and best tracking.
    Used by OpenEvolve, ShinkaEvolve, and GEPA backends.
    """
    programs: Dict[str, Any] = {}
    best_score = -float("inf")
    best_id: Optional[str] = None

    def _callback(program: Any, iteration: int) -> None:
        nonlocal best_score, best_id
        try:
            programs[program.id] = program

            score = (program.metrics or {}).get("combined_score", 0.0)
            if not isinstance(score, (int, float)):
                score = 0.0
            if score > best_score:
                best_score = score
                best_id = program.id
            is_best = program.id == best_id

            parent_id = getattr(program, "parent_id", None)
            parent_score, parent_solution = None, ""
            if parent_id and parent_id in programs:
                p = programs[parent_id]
                parent_score = (p.metrics or {}).get("combined_score")
                parent_solution = getattr(p, "solution", "")

            code = getattr(program, "solution", "") or ""
            elapsed = time.time() - start_time

            prog_data = {
                "id": program.id,
                "iteration": iteration,
                "score": score,
                "metrics": _safe_metrics(program.metrics or {}),
                "parent_id": parent_id,
                "parent_score": parent_score,
                "parent_iter": None,
                "context_ids": [],
                "context_scores": [],
                "label_type": "unknown",
                "solution_snippet": code[:SOLUTION_SNIPPET_LENGTH],
                "island": None,
                "is_best": is_best,
                "generation": getattr(program, "generation", 0),
                "image_path": (getattr(program, "metadata", {}) or {}).get("image_path"),
            }
            stats = {
                "total_programs": len(programs),
                "current_iteration": iteration,
                "best_score": best_score if best_score > -float("inf") else 0.0,
                "iterations_since_improvement": 0,
                "programs_per_min": round(len(programs) / elapsed * 60, 1) if elapsed > 0 else 0.0,
                "elapsed_seconds": round(elapsed, 1),
            }
            event = {
                "type": "new_program",
                "program": prog_data,
                "stats": stats,
                "is_best": is_best,
                "full_solution": code[: server.max_solution_length],
                "parent_full_solution": (
                    parent_solution[: server.max_solution_length] if parent_solution else ""
                ),
            }
            server.push_event(event)
        except Exception:
            logger.debug("External monitor callback error", exc_info=True)

    return _callback


def _safe_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe copy of metrics."""
    safe = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            safe[k] = v
        else:
            safe[k] = str(v)
    return safe
