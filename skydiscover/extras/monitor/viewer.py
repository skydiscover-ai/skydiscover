"""
Replay viewer for completed SkyDiscover runs.

Loads checkpoint data and serves the live monitor dashboard
for interactive exploration of past runs.

Usage:
    python -m skydiscover.extras.monitor.viewer <path> [--port PORT] [--summary-model MODEL]

<path> can be:
    - A checkpoint directory (contains metadata.json + programs/)
    - An output directory (contains island/checkpoints/ or checkpoints/)
    - A directory containing program JSON files
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _ckpt_num(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def find_checkpoint_dir(path: str) -> Optional[str]:
    """Auto-detect the best checkpoint directory from *path*."""
    p = Path(path)

    # 1. Direct checkpoint dir (metadata.json + programs/)
    if (p / "metadata.json").exists() and (p / "programs").is_dir():
        return str(p)

    # 2. programs/ subdir but no metadata
    if (p / "programs").is_dir() and list((p / "programs").glob("*.json")):
        return str(p)

    # 3. checkpoint_N dirs directly inside path
    ckpts = sorted(p.glob("checkpoint_*"), key=lambda x: _ckpt_num(x.name))
    if ckpts:
        return str(ckpts[-1])

    # 4. checkpoints/ subdir
    if (p / "checkpoints").is_dir():
        ckpts = sorted(
            (p / "checkpoints").glob("checkpoint_*"),
            key=lambda x: _ckpt_num(x.name),
        )
        if ckpts:
            return str(ckpts[-1])

    # 5. <subdir>/checkpoints/ (e.g. island/checkpoints/, sequential/checkpoints/)
    for subdir in sorted(p.iterdir()):
        if subdir.is_dir():
            ckpt_dir = subdir / "checkpoints"
            if ckpt_dir.is_dir():
                ckpts = sorted(
                    ckpt_dir.glob("checkpoint_*"),
                    key=lambda x: _ckpt_num(x.name),
                )
                if ckpts:
                    return str(ckpts[-1])

    # 6. Flat directory with JSON program files
    jsons = [j for j in p.glob("*.json") if j.name != "metadata.json"]
    if jsons:
        return str(p)

    return None


def load_programs(ckpt_dir: str) -> Tuple[List[Dict], Optional[str], int]:
    """Load programs from a checkpoint directory.

    Returns:
        (programs_list_sorted_by_iteration, best_program_id, last_iteration)
    """
    p = Path(ckpt_dir)
    programs: Dict[str, Dict] = {}
    best_program_id: Optional[str] = None
    last_iteration = 0

    # Metadata
    meta_path = p / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        best_program_id = meta.get("best_program_id")
        last_iteration = meta.get("last_iteration", 0)

    # Programs from programs/ subdir
    programs_dir = p / "programs"
    if programs_dir.is_dir():
        for jf in programs_dir.glob("*.json"):
            try:
                with open(jf) as f:
                    data = json.load(f)
                programs[data["id"]] = data
            except Exception as e:
                logger.warning(f"Skipping {jf.name}: {e}")
    else:
        # Flat directory
        for jf in p.glob("*.json"):
            if jf.name == "metadata.json":
                continue
            try:
                with open(jf) as f:
                    data = json.load(f)
                if "id" in data:
                    programs[data["id"]] = data
            except Exception:
                logger.debug("Failed to load program from %s", jf, exc_info=True)

    # Infer best if not in metadata
    if not best_program_id and programs:
        best_score = -float("inf")
        for pid, prog in programs.items():
            s = (prog.get("metrics") or {}).get("combined_score", 0)
            if isinstance(s, (int, float)) and s > best_score:
                best_score = s
                best_program_id = pid

    prog_list = sorted(programs.values(), key=lambda x: x.get("iteration_found", 0))
    return prog_list, best_program_id, last_iteration


def _to_monitor_format(prog: Dict, all_progs: Dict[str, Dict]) -> Dict:
    """Convert checkpoint program dict → monitor event program dict."""
    metrics = prog.get("metrics") or {}
    score = metrics.get("combined_score", 0.0)
    if not isinstance(score, (int, float)):
        score = 0.0

    parent_id = prog.get("parent_id")
    parent_score = None
    parent_iter = None
    if parent_id and parent_id in all_progs:
        pm = all_progs[parent_id].get("metrics") or {}
        parent_score = pm.get("combined_score")
        parent_iter = all_progs[parent_id].get("iteration_found")

    context_ids = prog.get("other_context_ids") or []
    context_scores = []
    for cid in context_ids:
        if cid in all_progs:
            cm = all_progs[cid].get("metrics") or {}
            context_scores.append(cm.get("combined_score"))
        else:
            context_scores.append(None)

    # Label
    label_type = "unknown"
    pi = prog.get("parent_info")
    if pi and isinstance(pi, (list, tuple)) and len(pi) >= 1:
        ls = str(pi[0]).lower()
        if "diverge" in ls:
            label_type = "diverge"
        elif "refine" in ls:
            label_type = "refine"
        elif "crossover" in ls:
            label_type = "crossover"
    if label_type == "unknown":
        label_type = (prog.get("metadata") or {}).get("label_type", "unknown")

    island = (prog.get("metadata") or {}).get("island")
    image_path = (prog.get("metadata") or {}).get("image_path")
    solution = prog.get("solution", "")

    from skydiscover.extras.monitor.callback import _safe_metrics

    return {
        "id": prog["id"],
        "iteration": prog.get("iteration_found", 0),
        "score": score,
        "metrics": _safe_metrics(metrics),
        "parent_id": parent_id,
        "parent_score": parent_score,
        "parent_iter": parent_iter,
        "context_ids": context_ids,
        "context_scores": context_scores,
        "label_type": label_type,
        "solution_snippet": solution[:500],
        "island": island,
        "generation": prog.get("generation", 0),
        "image_path": image_path,
    }


def main() -> None:
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Replay viewer for completed SkyDiscover runs",
    )
    parser.add_argument("path", help="Output directory or checkpoint path")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--summary-model",
        default="",
        help="LLM model for per-program summaries (default: gpt-5-mini). "
        "Requires OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--summary-api-base",
        default="https://api.openai.com/v1",
        help="API base URL for summary generation (default: https://api.openai.com/v1)",
    )
    args = parser.parse_args()

    # Resolve checkpoint
    ckpt_dir = find_checkpoint_dir(args.path)
    if not ckpt_dir:
        print(f"Error: no checkpoint data found in '{args.path}'")
        sys.exit(1)

    logger.info(f"Loading from: {ckpt_dir}")
    prog_list, best_id, last_iter = load_programs(ckpt_dir)
    if not prog_list:
        print(f"Error: no programs found in '{ckpt_dir}'")
        sys.exit(1)

    logger.info(f"Loaded {len(prog_list)} programs (best={best_id}, last_iter={last_iter})")

    all_progs = {p["id"]: p for p in prog_list}
    monitor_programs = [_to_monitor_format(p, all_progs) for p in prog_list]

    # Start server
    from skydiscover.extras.monitor.server import MonitorServer

    server = MonitorServer(host=args.host, port=args.port)

    # Configure per-program & global summary
    summary_model = args.summary_model
    if not summary_model and os.environ.get("OPENAI_API_KEY"):
        summary_model = "gpt-5-mini"
    if summary_model:
        server.configure_summary(model=summary_model, api_base=args.summary_api_base, interval=0)

    server.start()

    # Push all programs
    best_score = -float("inf")
    for mp in monitor_programs:
        pid = mp["id"]
        s = mp["score"]
        is_best = (pid == best_id) or (s > best_score)
        if s > best_score:
            best_score = s

        solution = all_progs[pid].get("solution", "")
        parent_solution = ""
        if mp["parent_id"] and mp["parent_id"] in all_progs:
            parent_solution = all_progs[mp["parent_id"]].get("solution", "")

        server.push_event(
            {
                "type": "new_program",
                "program": mp,
                "stats": {
                    "total_programs": len(monitor_programs),
                    "current_iteration": last_iter,
                    "best_score": best_score,
                    "iterations_since_improvement": 0,
                    "programs_per_min": 0,
                    "elapsed_seconds": 0,
                },
                "is_best": is_best,
                "full_solution": solution[: server.max_solution_length],
                "parent_full_solution": parent_solution[: server.max_solution_length],
            }
        )

    # Wait for queue to flush
    time.sleep(1.5)

    print(f"\n  Dashboard ready at http://localhost:{args.port}/")
    print(f"  {len(prog_list)} programs loaded from {ckpt_dir}")
    if summary_model:
        print(f"  Per-program summaries: {summary_model}")
    else:
        print("  Per-program summaries: disabled (set OPENAI_API_KEY or --summary-model)")
    print("  Press Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    server.stop()
    print("Stopped.")


if __name__ == "__main__":
    main()
