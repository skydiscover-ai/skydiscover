"""
Pure formatting functions for evox prompt generation.
All functions are stateless — no class or LLM dependencies.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from skydiscover.optimize.context_builder.utils import format_artifacts, prog_attr
from skydiscover.optimize.search.base_database import Program

logger = logging.getLogger(__name__)


def filter_db_stats_by_horizon(db_stats: Dict[str, Any], horizon: int) -> Dict[str, Any]:
    """Filter db_stats to only include the last 'horizon' entries for trajectory fields."""
    if not db_stats or horizon <= 0:
        return db_stats

    filtered = dict(db_stats)
    if recent := db_stats.get("recent_solution_stats"):
        filtered_recent = dict(recent)
        for key in ["execution_trace", "score_trajectory", "parent_scores"]:
            if (val := recent.get(key)) and len(val) > horizon:
                filtered_recent[key] = val[-horizon:]
        filtered_recent["num_recent_iterations"] = min(
            horizon, recent.get("num_recent_iterations", 0)
        )
        filtered["recent_solution_stats"] = filtered_recent
    return filtered


def format_execution_trace(execution_trace: list, window_start_score: float = None) -> str:
    """Format execution trace with program/parent/context tuples."""
    if not execution_trace:
        return ""

    def fmt_id(pid):
        return pid[:8] if pid and len(pid) > 8 else (pid or "None")

    def fmt_score(s):
        return f"{s:.4f}" if s is not None else "N/A"

    def unpack_tuple(t):
        if not t:
            return None, None, None
        if len(t) >= 3:
            return t[0], t[1], t[2]
        return None, t[0], t[1]

    def fmt_program_ref(t, prefix=""):
        label, pid, score = unpack_tuple(t)
        if pid is None:
            return f"{prefix}=None (seed program)" if prefix else "None"
        label_str = f'label="{label}", ' if label else ""
        return (
            f"{prefix} ({label_str}id={fmt_id(pid)}, score={fmt_score(score)})"
            if prefix
            else f"({label_str}id={fmt_id(pid)}, score={fmt_score(score)})"
        )

    lines = []
    best = window_start_score

    for entry in execution_trace:
        prog_tuple = entry.get("program")
        if prog_tuple is None:
            continue

        _, _, prog_score = unpack_tuple(prog_tuple)
        _, _, parent_score = unpack_tuple(entry.get("parent"))

        parent_str = fmt_program_ref(entry.get("parent"), "Parent")
        ctx = entry.get("context") or []
        context_str = f"Context=[{', '.join(fmt_program_ref(c) for c in ctx)}]"

        if prog_score is not None:
            prog_score, parent_score = round(prog_score, 4), (
                round(parent_score, 4) if parent_score is not None else None
            )
            if best is None:
                best, outcome = prog_score, "first program"
            elif prog_score > best:
                outcome, best = f"⭐ NEW BEST (was {best:.4f})", prog_score
            elif parent_score is not None and prog_score > parent_score:
                outcome = f"above parent, best still {best:.4f}"
            elif parent_score is not None and prog_score < parent_score:
                outcome = f"regression, best still {best:.4f}"
            else:
                outcome = f"no change, best still {best:.4f}"
        else:
            outcome = "N/A"

        lines.extend(
            [
                f"Iter {entry.get('iteration', '?')}: {parent_str}, {context_str}",
                f"       -> Generated child score={fmt_score(prog_score)} ({outcome})",
                "",
            ]
        )

    return "\n".join(lines[:-1]) if lines else ""


def format_db_stats_diff(
    start_stats: Dict[str, Any], end_stats: Dict[str, Any], horizon: Optional[int] = None
) -> str:
    """Format start -> end db_stats comparison for a search algorithm's window."""
    if not start_stats or not end_stats:
        return ""

    lines = ["Population Statistics Change (Start -> End of Search Window):"]

    start_pop = start_stats.get("population_size", "?")
    end_pop = end_stats.get("population_size", "?")
    lines.append(f"- population_size: {start_pop} -> {end_pop}")

    start_summary = start_stats.get("solution_score_summary", {})
    end_summary = end_stats.get("solution_score_summary", {})
    if start_summary and end_summary:
        parts = []
        key_names = [
            ("best", "current_best"),
            ("q75", "75th_pct"),
            ("q50", "50th_pct (median)"),
            ("q25", "25th_pct"),
            ("worst", "worst"),
        ]
        for key, display_name in key_names:
            s = start_summary.get(key)
            e = end_summary.get(key)
            if s is not None and e is not None:
                diff = e - s
                sign = "+" if diff >= 0 else ""
                parts.append(f"{display_name}: {s:.4f} -> {e:.4f} ({sign}{diff:.4f})")
        if parts:
            lines.append(f"- {', '.join(parts)}")

    start_top = start_stats.get("top_solution_scores", [])
    end_top = end_stats.get("top_solution_scores", [])
    if start_top and end_top:
        k = min(len(start_top), len(end_top))
        start_fmt = [f"{s:.4f}" for s in start_top[:k]]
        end_fmt = [f"{s:.4f}" for s in end_top[:k]]
        lines.append(f"- top_{k}_solution_scores: {start_fmt} -> {end_fmt}")

    start_avg = start_stats.get("avg_solutions_per_parent")
    end_avg = end_stats.get("avg_solutions_per_parent")
    if start_avg is not None and end_avg is not None and start_pop and end_pop:
        start_pct = (start_avg / start_pop * 100) if start_pop != "?" else 0
        end_pct = (end_avg / end_pop * 100) if end_pop != "?" else 0
        lines.append(
            f"- % of solutions share the same parent on average: {start_pct:.1f}% -> {end_pct:.1f}%"
        )

    sota = end_stats.get("SOTA_score")
    if sota is not None and start_summary and end_summary:
        start_best = start_summary.get("best")
        end_best = end_summary.get("best")
        if start_best is not None and end_best is not None:
            start_gap = sota - start_best
            end_gap = sota - end_best
            gap_diff = end_gap - start_gap
            sign = "+" if gap_diff >= 0 else ""
            lines.append(
                f"- gap_to_SOTA (lower is better): {start_gap:.4f} -> {end_gap:.4f} ({sign}{gap_diff:.4f})"
            )

    start_tiers = start_summary.get("score_tiers") if start_summary else None
    end_tiers = end_summary.get("score_tiers") if end_summary else None
    if start_tiers and end_tiers:
        tier_parts = []
        for tier_name in end_tiers.keys():
            start_data = start_tiers.get(tier_name, {})
            end_data = end_tiers.get(tier_name, {})
            start_pct = start_data.get("pct_programs", 0)
            end_pct = end_data.get("pct_programs", 0)
            start_threshold = start_data.get("threshold", "")
            end_threshold = end_data.get("threshold", "")
            diff = end_pct - start_pct
            sign = "+" if diff >= 0 else ""
            tier_parts.append(
                f"\n  {tier_name}: [{start_threshold}] {start_pct:.0f}% -> [{end_threshold}] {end_pct:.0f}% ({sign}{diff:.0f}%)"
            )
        lines.append(f"- programs_by_score_tier:{','.join(tier_parts)}")

    end_recent = end_stats.get("recent_solution_stats", {})
    if end_recent:
        iters_no_improve = end_recent.get("iterations_without_improvement")
        threshold = end_recent.get("improvement_threshold", 0.0)
        if iters_no_improve is not None:
            if threshold > 0:
                lines.append(
                    f"- iterations_without_improvement (improvement <= {threshold:.4f}): {iters_no_improve}"
                )
            else:
                lines.append(f"- iterations_without_improvement: {iters_no_improve}")

        execution_trace = end_recent.get("execution_trace")
        if execution_trace:
            if horizon:
                execution_trace = execution_trace[-horizon:]

            first_iter = execution_trace[0].get("iteration", "?")
            last_iter = execution_trace[-1].get("iteration", "?")
            lines.append(f"\n### Execution Trace (iterations {first_iter}-{last_iter})")
            window_start_score = start_summary.get("best") if start_summary else None
            lines.append(
                format_execution_trace(execution_trace, window_start_score=window_start_score)
            )
        else:

            def fmt_scores(scores):
                return [f"{s:.4f}" if s is not None else "N/A" for s in scores]

            if score_trajectory := end_recent.get("score_trajectory"):
                lines.append(
                    f"- recent_score_trajectory (last {len(score_trajectory)}): {fmt_scores(score_trajectory)}"
                )
                if parent_scores := end_recent.get("parent_scores"):
                    lines.append(f"- recent_parent_scores: {fmt_scores(parent_scores)}")

    return "\n".join(lines)


def format_population_state(db_stats: Dict[str, Any]) -> str:
    """Format the population state from db_stats into clean, actionable lines."""
    if not db_stats:
        return ""

    def fmt_scores(scores):
        return [f"{s:.4f}" if s is not None else "N/A" for s in scores]

    lines = []
    pop_size = db_stats.get("population_size")
    lines.append(f"- population_size: {pop_size}")

    score_summary = db_stats.get("solution_score_summary") or {}
    sota = db_stats.get("SOTA_score")
    best = score_summary.get("best")
    q75, q50, q25 = (
        score_summary.get("q75"),
        score_summary.get("q50") or score_summary.get("median"),
        score_summary.get("q25"),
    )
    worst = score_summary.get("worst")

    if best is not None:
        pct = lambda v: (v / best * 100) if best > 0 and v is not None else 0

        dist_parts = [f"current_best={best:.4f}"]
        for name, val in [("75th_pct", q75), ("50th_pct", q50), ("25th_pct", q25)]:
            if val is not None:
                dist_parts.append(f"{name}={val:.4f} ({pct(val):.0f}%)")
        if worst is not None:
            dist_parts.append(f"worst={worst:.4f}")

        lines.append(f"- score_distribution: {', '.join(dist_parts)}")
        if sota is not None:
            lines.append(f"- gap_to_SOTA: SOTA={sota:.4f}, gap={sota - best:.4f}")

        if tiers := score_summary.get("score_tiers"):
            tier_parts = [
                f"{n} ({d.get('threshold','')}): {d.get('pct_programs',0):.0f}%"
                for n, d in tiers.items()
            ]
            lines.append(f"- programs_by_score_tier: {', '.join(tier_parts)}")

        if (unique := score_summary.get("unique_scores")) is not None:
            lines.append(f"- unique_score_values: {unique}")

    if (avg := db_stats.get("avg_solutions_per_parent")) is not None and pop_size:
        lines.append(f"- {avg / pop_size * 100:.1f}% of solutions share the same parent on average")

    if top_scores := db_stats.get("top_solution_scores"):
        best_score = top_scores[0]
        best_count = (
            sum(
                1
                for s in top_scores
                if isinstance(s, (int, float)) and round(s, 4) == round(best_score, 4)
            )
            if isinstance(best_score, (int, float))
            else 0
        )
        lines.append(f"- top_{len(top_scores)}_scores: {fmt_scores(top_scores)}")
        if best_count > 1:
            lines.append(f"  - Top score ({best_score:.4f}) repeated {best_count}x")
        if best_count == len(top_scores):
            lines.append(f"  (⚠️ ALL {best_count} identical)")

    if recent := db_stats.get("recent_solution_stats"):
        if (iters := recent.get("iterations_without_improvement")) and iters > 0:
            thresh = recent.get("improvement_threshold", 0.0)
            thresh_str = f" by more than {thresh:.4f}" if thresh > 0 else ""
            lines.append(f"- No improvement{thresh_str} for {iters} iterations")

        def score_bucket(score):
            if score is None or best is None:
                return None
            if score >= best:
                return "at best"
            if q75 and score >= q75:
                return "75-100th"
            if q50 and score >= q50:
                return "50-75th"
            if q25 and score >= q25:
                return "25-50th"
            return "0-25th"

        for key, label in [("most_reused_parent", "parent"), ("most_reused_context", "context")]:
            if (ratio := recent.get(f"{key}_ratio")) and ratio > 0:
                bucket = score_bucket(recent.get(f"{key}_score"))
                score_str = f", score {bucket}" if bucket else ""
                lines.append(f"- {label}: {ratio*100:.0f}% reuse rate{score_str}")

        if traj := recent.get("score_trajectory"):
            lines.append(f"- recent_scores (last {len(traj)}): {fmt_scores(traj)}")
            if parent := recent.get("parent_scores"):
                lines.append(f"- recent_parent_scores: {fmt_scores(parent)}")

    return "\n".join(lines)


def format_current_program(
    current_program: Union[Program, Dict[str, Program]],
    language: str,
    improvement_areas: Optional[str] = None,
) -> str:
    """Format current program with metrics and solution."""
    if not current_program:
        return ""

    if isinstance(current_program, dict) and current_program:
        label = list(current_program.keys())[0] or "Current Search Program"
        program = list(current_program.values())[0]
    else:
        label = "Current Search Program"
        program = current_program
    solution = prog_attr(program, "solution")
    metrics = prog_attr(program, "metrics", {})

    window_start = int(metrics.get("window_start_iteration", 0))
    horizon = int(metrics.get("search_window_horizon") or 0)
    window_end = window_start + horizon
    start_score = metrics.get("search_window_start_score", 0.0)
    end_score = metrics.get("search_window_end_score", 0.0)
    combined_score = metrics.get("combined_score", 0.0)
    improvement = end_score - start_score

    lines = [f"## {label}\n", "### Metrics"]
    if improvement_areas:
        lines.append(f"Focus areas:\n{improvement_areas}\n")
    lines.append(f"Search Algorithm Score = {combined_score:.4f}")
    lines.append(
        f"This search algorithm ran from iteration {window_start} to {window_end} ({horizon} iterations)"
    )
    lines.append(
        f"This search algorithm changed the downstream solution combined_score by: {start_score:.4f} -> {end_score:.4f} (+{improvement:.4f})"
    )
    lines.append(f"\n### Solution\n```{language}")
    lines.append(solution)
    lines.append("```\n")

    artifact_section = format_artifacts(program, heading="###")
    if artifact_section:
        lines.append(artifact_section)

    return "\n".join(lines)


def identify_search_improvement_areas(
    current_program: Program,
    metrics: Dict[str, float],
    previous_programs: List[Program],
    simplification_threshold: Optional[int] = None,
) -> str:
    """Identify improvement areas for search algorithms based on combined_score."""

    def safe_float(val):
        if val is None:
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    improvement_areas = []
    current_score = safe_float(metrics.get("combined_score"))

    if previous_programs:
        prev_program = previous_programs[-1]
        prev_metrics = prog_attr(prev_program, "metrics", {}) or {}
        prev_score = safe_float(prev_metrics.get("combined_score"))

        if current_score > prev_score:
            improvement_areas.append(
                f"Search algorithm score improved: {prev_score:.4f} → {current_score:.4f}"
            )
        elif current_score < prev_score:
            improvement_areas.append(
                f"Search algorithm score declined: {prev_score:.4f} → {current_score:.4f}. Consider revising."
            )
        else:
            improvement_areas.append(f"Search algorithm score unchanged at {current_score:.4f}")

    if not improvement_areas:
        improvement_areas.append("Focus on improving the search algorithm score (combined_score)")

    if simplification_threshold:
        code_length = len(prog_attr(current_program, "solution"))
        if code_length > simplification_threshold:
            improvement_areas.append(
                f"Consider simplifying - solution length exceeds {simplification_threshold} characters"
            )

    return "\n".join(f"- {area}" for area in improvement_areas)


def format_search_window_context(context: Dict[str, Any]) -> str:
    """Format the current search window context from context['search_stats']."""
    stats = context.get("search_stats") or {}
    window_start = int(stats.get("window_start_iteration") or 0)
    total = int(stats.get("total_iterations") or 100)
    horizon = int(stats.get("search_window_horizon", 0))
    improvement_threshold = float(stats.get("improvement_threshold") or 0.0)

    lines = []

    window_line = f"- Your newly designed search algorithm will start at iteration {window_start} out of {total}."
    if horizon > 0:
        if improvement_threshold > 0:
            window_line += f" It will run for at least {horizon} iterations (potentially more if improving), but will be cut to just {horizon} iterations if it fails to improve the solution score by more than {improvement_threshold:.4f}."
        else:
            window_line += f" It will run for at least {horizon} iterations (potentially more if improving), but will be cut to just {horizon} iterations if it fails to improve the solution score."
    lines.append(window_line)

    if improvement_threshold > 0:
        lines.append(
            f"- If your algorithm fails to improve the solution score by more than {improvement_threshold:.4f} during this window, it will be replaced."
        )
    else:
        lines.append(
            "- If your algorithm fails to improve the solution score during this window, it will be replaced."
        )

    lines.append(
        "- Goal: Design a better search strategy (e.g. how to select and manage solution programs) to improve the downstream solution score."
    )
    lines.append(
        "- NOTE: Exactly one program is generated per iteration. Keep the population size in mind when designing your search algorithm."
    )

    return "\n".join(lines)


def format_problem_description(problem_config: Any) -> str:
    """Format the problem description from the prompt config."""
    if problem_config is None:
        return "(No problem description provided)"
    if isinstance(problem_config, str):
        return problem_config
    if hasattr(problem_config, "system_message") and problem_config.system_message:
        return str(problem_config.system_message)
    return str(problem_config) if problem_config else "(No problem description provided)"


def format_evaluator_context(evaluator_path: Any) -> str:
    """Format the evaluator context by reading the evaluator file."""
    if evaluator_path is None:
        return "(No evaluator context provided)"

    if isinstance(evaluator_path, str):
        if not evaluator_path.endswith(".py"):
            if evaluator_path.strip().startswith("```"):
                return evaluator_path
            return f"```python\n{evaluator_path}\n```"
        try:
            if os.path.isfile(evaluator_path):
                with open(evaluator_path, "r") as f:
                    return f"```python\n{f.read()}\n```"
        except Exception as e:
            logger.warning(f"Failed to read evaluator file {evaluator_path}: {e}")

    return f"Evaluator file: {evaluator_path}"


def prepare_search_algorithms_data(
    other_context_programs: Union[List[Program], Dict[str, List[Program]]],
    format_stats_diff=format_db_stats_diff,
    filter_by_horizon=filter_db_stats_by_horizon,
) -> List[Dict[str, Any]]:
    """Prepare data for batch summarization of context programs."""
    if not other_context_programs:
        return []

    if isinstance(other_context_programs, dict):
        flat_programs = []
        for programs in other_context_programs.values():
            if programs:
                flat_programs.extend(programs)
        programs_list = flat_programs
    else:
        programs_list = other_context_programs

    all_programs_data = []

    for idx, program in enumerate(programs_list, start=1):
        solution = prog_attr(program, "solution")
        metrics = prog_attr(program, "metrics", {})
        metadata = prog_attr(program, "metadata", {})

        start_db_stats = metadata.get("start_db_stats")
        end_db_stats = metadata.get("end_db_stats")
        horizon = int(metrics.get("search_window_horizon", 0))

        if start_db_stats and end_db_stats:
            start_db_stats = filter_by_horizon(start_db_stats, horizon)
            end_db_stats = filter_by_horizon(end_db_stats, horizon)

        if start_db_stats and end_db_stats:
            db_stats_text = format_stats_diff(start_db_stats, end_db_stats, horizon=horizon)
            all_programs_data.append(
                {
                    "program_num": idx,
                    "solution": solution,
                    "db_stats_text": db_stats_text,
                    "combined_score": metrics.get("combined_score", 0.0),
                    "improvement": metrics.get("search_window_end_score", 0.0)
                    - metrics.get("search_window_start_score", 0.0),
                }
            )

    return all_programs_data


def format_single_program_section(
    program: Program, idx: int, language: str, summaries_by_num: Dict[int, str]
) -> List[str]:
    """Format a single program with metrics and solution/summary."""
    solution = prog_attr(program, "solution")
    metrics = prog_attr(program, "metrics", {})

    window_start = int(metrics.get("window_start_iteration", 0))
    horizon = int(metrics.get("search_window_horizon", 0))
    start_score = metrics.get("search_window_start_score", 0.0)
    end_score = metrics.get("search_window_end_score", 0.0)
    combined_score = metrics.get("combined_score", 0.0)

    lines = [
        f"### Program {idx}\n",
        "#### Metrics",
        f"Search Algorithm Score = {combined_score:.4f}",
        f"Ran iterations {window_start} to {window_start + horizon} ({horizon} iterations)",
        f"Score changed: {start_score:.4f} -> {end_score:.4f} (+{end_score - start_score:.4f})",
    ]

    if idx in summaries_by_num:
        lines.append(f"\n#### Summary\n{summaries_by_num[idx]}\n")
    else:
        lines.extend(["\n#### Solution", f"```{language}", solution, "```\n"])

    artifact_section = format_artifacts(program, heading="####")
    if artifact_section:
        lines.append(artifact_section)

    return lines


def format_search_algorithms(
    other_context_programs: Union[List[Program], Dict[str, List[Program]]],
    language: str,
    summaries_by_num: Optional[Dict[int, str]] = None,
) -> str:
    """Format previous search algorithms with window context."""
    if not other_context_programs:
        return ""

    summaries_by_num = summaries_by_num or {}
    lines = []

    if isinstance(other_context_programs, dict):
        global_idx = 0
        for label, programs in other_context_programs.items():
            display_label = label or "Other Reference Programs"
            lines.extend(
                [f"\n## {display_label}\n", "Diverse search programs that may inspire new ideas:\n"]
            )
            for program in programs or []:
                global_idx += 1
                lines.extend(
                    format_single_program_section(program, global_idx, language, summaries_by_num)
                )
    else:
        lines.append("## Other Reference Programs\n")
        for idx, program in enumerate(other_context_programs, start=1):
            lines.extend(format_single_program_section(program, idx, language, summaries_by_num))

    return "\n".join(lines)


def parse_batch_summaries(response: str, programs_data: List[Dict]) -> Dict[int, str]:
    """Parse batch summary response into individual summaries by program number."""
    summaries = {}
    if not response or not programs_data:
        return summaries

    for prog in programs_data:
        num = prog["program_num"]
        marker = f"[PROGRAM {num}]"
        if marker in response:
            start_idx = response.find(marker) + len(marker)
            next_idx = len(response)
            for other in programs_data:
                if other["program_num"] != num:
                    other_marker = f"[PROGRAM {other['program_num']}]"
                    if other_marker in response:
                        idx = response.find(other_marker)
                        if start_idx < idx < next_idx:
                            next_idx = idx
            summaries[num] = response[start_idx:next_idx].strip()

    if not summaries and response:
        summaries[programs_data[0]["program_num"]] = response
    return summaries
