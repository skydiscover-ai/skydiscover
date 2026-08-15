"""
CLI entry point: run-evolve-analysis
"""
import sys
import json
import logging
import math
import dataclasses
from pathlib import Path
from typing import Any, Optional

import click
import yaml
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

from skydiscover.extras.evolve_analyzer.coordinator import load_config, run_postmortem
from skydiscover.extras.evolve_analyzer.report_synthesizer import (
    DimensionReport,
    EvolveLoopReport,
    StagnationPeriodReport,
    AggregateStats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Report serialisation helpers
# ---------------------------------------------------------------------------

_RATING_EMOJI = {
    1: "🔴",
    2: "🟠",
    3: "🟡",
    4: "🟢",
    5: "✅",
}


def _serialise_value(obj: Any) -> Any:
    """Recursively make *obj* JSON-serialisable."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            k: _serialise_value(v)
            for k, v in dataclasses.asdict(obj).items()
        }
    if isinstance(obj, dict):
        return {k: _serialise_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise_value(i) for i in obj]
    if isinstance(obj, float):
        # JSON does not support NaN / Inf — convert to None
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def report_to_dict(report: EvolveLoopReport) -> dict:
    """
    Serialize an EvolveLoopReport to a plain JSON-compatible dict.

    Uses dataclasses.asdict as the base, with special handling for
    pd.DataFrame fields (converted to records) and non-finite floats.
    """
    raw = dataclasses.asdict(report)
    return _serialise_value(raw)


def report_to_text(report: EvolveLoopReport) -> str:
    """
    Render an EvolveLoopReport as human-readable text.

    Format
    ------
    Header block with experiment ID and aggregate stats, followed by one
    section per dimension, then stagnation periods, cross-dimension
    interactions, and novel observations.
    """
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    sep = "=" * 70
    thin = "-" * 70
    lines.append(sep)
    lines.append("  EVOLVE LOOP POST-MORTEM REPORT")
    if report.experiment_id:
        lines.append(f"  Experiment: {report.experiment_id}")
    lines.append(sep)
    lines.append("")

    # ── Executive summary ─────────────────────────────────────────────────────
    lines.append("EXECUTIVE SUMMARY")
    lines.append(thin)
    lines.append(report.executive_summary)
    lines.append("")

    # ── Aggregate stats ───────────────────────────────────────────────────────
    agg = report.aggregate_stats
    lines.append("AGGREGATE STATS")
    lines.append(thin)
    lines.append(f"  Total iterations        : {agg.total_iterations}")
    lines.append(f"  Evaluations             : {agg.n_evaluations}")
    lines.append(f"  Successful evaluations  : {agg.n_successful}")
    lines.append(f"  Stagnation periods      : {agg.n_stagnation_periods}")
    lines.append(f"  Stagnation iterations   : {agg.total_stagnation_iterations}")
    lines.append(f"  Best score              : {_fmt_float(agg.best_score)}")
    lines.append(f"  Final score             : {_fmt_float(agg.final_score)}")
    lines.append(f"  Worst score             : {_fmt_float(agg.worst_score)}")
    if getattr(agg, "baseline_score", None) is not None:
        lines.append(f"  Baseline score          : {_fmt_float(agg.baseline_score)}")
    if getattr(agg, "score_improvement_vs_baseline", None) is not None:
        sign = "+" if agg.score_improvement_vs_baseline >= 0 else ""
        lines.append(f"  vs baseline             : {sign}{agg.score_improvement_vs_baseline:.2%}")
    if agg.total_llm_cost_usd is not None:
        lines.append(f"  Total LLM cost          : ${agg.total_llm_cost_usd:.4f}")
    if agg.total_duration_hours is not None:
        lines.append(f"  Total duration          : {agg.total_duration_hours:.2f} h")
    lines.append("")

    # ── Experiment analysis parameters ───────────────────────────────────────
    algo_class = getattr(report, "algorithm_class", None)
    js = getattr(report, "llm_judge_status", None)
    has_run_params = any([report.run_source, report.run_path, report.run_config_path, js])
    if has_run_params:
        lines.append("EXPERIMENT ANALYSIS PARAMETERS")
        lines.append(thin)
        if report.run_source:
            lines.append(f"  --source                : {report.run_source}")
        if report.run_config_path:
            lines.append(f"  --config                : {report.run_config_path}")
        if report.run_path:
            lines.append(f"  --path                  : {report.run_path}")
        if js is not None:
            lines.append("")
            lines.append("  LLM Judge")
            endpoint = js.base_url or js.provider
            lines.append(f"  Model                   : {js.model} ({js.provider} via {endpoint})")
            if js.status == "success":
                lines.append("  Status                  : ✅ Connected successfully")
            elif js.status == "skipped":
                lines.append(f"  Status                  : ⏭  Skipped — {js.skip_reason}")
            else:
                lines.append("  Status                  : ❌ Failed")
                if js.error:
                    lines.append(f"  Error                   : {js.error}")
        lines.append("")

    # ── Experiment setup ──────────────────────────────────────────────────────
    num_islands = getattr(report, "num_islands", None)
    has_exp_setup = any([report.run_source, algo_class])
    if has_exp_setup:
        lines.append("EXPERIMENT SETUP")
        lines.append(thin)
        algo_name_display = getattr(report, "algorithm_name", None) or report.run_source
        if algo_name_display:
            lines.append(f"  Algorithm name          : {algo_name_display}")
        if algo_class:
            lines.append(f"  Algorithm class         : {algo_class}")
        if num_islands is not None:
            lines.append(f"  Islands                 : {num_islands}")
        lines.append("")

    # ── Dimensions ────────────────────────────────────────────────────────────
    lines.append("DIMENSION REPORTS")
    lines.append(sep)

    critical_periods = [
        sp for sp in (report.stagnation_periods or [])
        if sp.length > 1 and sp.severity == "critical"
    ]
    crash_periods = [
        sp for sp in (report.stagnation_periods or [])
        if sp.dominant_failure_type == "crash"
        and sp not in critical_periods
    ]

    for dim in report.dimensions:
        lines.append("")
        lines.append(f"Dimension:      {dim.name}")
        if dim.rating is None:
            lines.append("Rating:         N/A")
        else:
            emoji = _RATING_EMOJI.get(dim.rating, "🟡")
            lines.append(f"Rating:         {dim.rating}/5  {emoji}")
        lines.append(f"Summary:        {dim.summary}")

        if dim.evidence:
            lines.append("Evidence:")
            for ev in dim.evidence:
                stripped = ev.lstrip()
                indent = len(ev) - len(stripped)
                if indent >= 4:
                    lines.append(f"      ◦ {stripped}")
                else:
                    lines.append(f"  • {ev}")

        if dim.name == "Stagnation" and crash_periods:
            lines.append("Crash stagnation details:")
            for sp in crash_periods:
                end_str = str(sp.end_iteration) if sp.end_iteration is not None else "ongoing"
                lines.append(
                    f"  Streak {sp.streak_id}: iters {sp.start_iteration}–{end_str}  "
                    f"len={sp.length}  failure={sp.dominant_failure_type}"
                )
                crash_samples = getattr(sp, "crash_samples", None)
                if crash_samples:
                    lines.append("    Crash details:")
                    for sample in crash_samples:
                        lines.append(f"      iter {sample['iteration']}: {sample['error']}")
                if sp.llm_analysis:
                    category = sp.llm_analysis.get("category", "")
                    explanation = sp.llm_analysis.get("explanation", "")
                    if category:
                        lines.append(f"    LLM category:   {category}")
                    if explanation:
                        lines.append(f"    LLM analysis:   {explanation}")
                lines.append(f"    Recommendation: {sp.recommendation}")

        if dim.name == "Stagnation" and critical_periods:
            lines.append("Critical stagnation details:")
            for sp in critical_periods:
                end_str = str(sp.end_iteration) if sp.end_iteration is not None else "ongoing"
                lines.append(
                    f"  Streak {sp.streak_id}: iters {sp.start_iteration}–{end_str}  "
                    f"len={sp.length}  failure={sp.dominant_failure_type}"
                )
                crash_samples = getattr(sp, "crash_samples", None)
                if crash_samples:
                    lines.append("    Crash details:")
                    for sample in crash_samples:
                        lines.append(f"      iter {sample['iteration']}: {sample['error']}")
                if sp.llm_analysis:
                    category = sp.llm_analysis.get("category", "")
                    explanation = sp.llm_analysis.get("explanation", "")
                    if category:
                        lines.append(f"    LLM category:   {category}")
                    if explanation:
                        lines.append(f"    LLM analysis:   {explanation}")
                lines.append(f"    Recommendation: {sp.recommendation}")

        if dim.historical is not None:
            hist = dim.historical
            hist_summary = getattr(hist, "summary", str(hist))
            lines.append(f"Historical:     {hist_summary}")

        lines.append(f"Recommendation: {dim.recommendation}")
        lines.append(thin)

    # ── Cross-dimension interactions ──────────────────────────────────────────
    lines.append("")
    lines.append("CROSS-DIMENSION INTERACTIONS")
    lines.append(thin)
    lines.append(report.cross_dimension_interactions)
    lines.append("")

    # ── Novel observations ────────────────────────────────────────────────────
    if report.novel_observations:
        lines.append("NOVEL OBSERVATIONS")
        lines.append(thin)
        for obs in report.novel_observations:
            lines.append(f"  • {obs}")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


def report_to_markdown(report: EvolveLoopReport) -> str:
    """Render an EvolveLoopReport as a GitHub-flavored Markdown document."""
    lines: list[str] = []

    # ── Title ─────────────────────────────────────────────────────────────────
    lines.append("# Evolve Loop Post-Mortem Report")
    if report.experiment_id:
        lines.append(f"\n**Experiment:** `{report.experiment_id}`")
    lines.append("")

    # ── Executive summary ─────────────────────────────────────────────────────
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(report.executive_summary_md)
    lines.append("")

    # ── Aggregate stats ───────────────────────────────────────────────────────
    agg = report.aggregate_stats
    lines.append("## Aggregate Stats")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total iterations | {agg.total_iterations} |")
    lines.append(f"| Evaluations | {agg.n_evaluations} |")
    lines.append(f"| Successful evaluations | {agg.n_successful} |")
    lines.append(f"| Stagnation periods | {agg.n_stagnation_periods} |")
    lines.append(f"| Stagnation iterations | {agg.total_stagnation_iterations} |")
    lines.append(f"| Best score | {_fmt_float(agg.best_score)} |")
    lines.append(f"| Final score | {_fmt_float(agg.final_score)} |")
    lines.append(f"| Worst score | {_fmt_float(agg.worst_score)} |")
    if getattr(agg, "baseline_score", None) is not None:
        lines.append(f"| Baseline score | {_fmt_float(agg.baseline_score)} |")
    if getattr(agg, "score_improvement_vs_baseline", None) is not None:
        sign = "+" if agg.score_improvement_vs_baseline >= 0 else ""
        lines.append(f"| vs baseline | {sign}{agg.score_improvement_vs_baseline:.2%} |")
    if agg.total_llm_cost_usd is not None:
        lines.append(f"| Total LLM cost | ${agg.total_llm_cost_usd:.4f} |")
    if agg.total_duration_hours is not None:
        lines.append(f"| Total duration | {agg.total_duration_hours:.2f} h |")
    lines.append("")

    # ── Experiment analysis parameters ───────────────────────────────────────
    js = getattr(report, "llm_judge_status", None)
    has_run_params = any([report.run_source, report.run_path, report.run_config_path, js])
    if has_run_params:
        lines.append("## Experiment Analysis Parameters")
        lines.append("")
        if report.run_source:
            lines.append(f"- **--source:** `{report.run_source}`")
        if report.run_config_path:
            lines.append(f"- **--config:** `{report.run_config_path}`")
        if report.run_path:
            lines.append(f"- **--path:** `{report.run_path}`")
        lines.append("")
        if js is not None:
            lines.append("### LLM Judge")
            lines.append("")
            endpoint = js.base_url or js.provider
            lines.append(f"- **Model:** `{js.model}` ({js.provider} via {endpoint})")
            if js.status == "success":
                lines.append("- **Status:** ✅ Connected successfully")
            elif js.status == "skipped":
                lines.append(f"- **Status:** ⏭ Skipped — {js.skip_reason}")
            else:
                lines.append("- **Status:** ❌ Failed")
                if js.error:
                    lines.append(f"- **Error:** {js.error}")
            lines.append("")

    # ── Experiment setup ──────────────────────────────────────────────────────
    algo_class = getattr(report, "algorithm_class", None)
    num_islands = getattr(report, "num_islands", None)
    has_exp_setup = any([report.run_source, algo_class])
    if has_exp_setup:
        lines.append("## Experiment Setup")
        lines.append("")
        algo_name_display = getattr(report, "algorithm_name", None) or report.run_source
        if algo_name_display:
            lines.append(f"- **Algorithm name:** `{algo_name_display}`")
        if algo_class:
            lines.append(f"- **Algorithm class:** `{algo_class}`")
        if num_islands is not None:
            lines.append(f"- **Islands:** {num_islands}")
        lines.append("")

    # ── Dimension overview table ───────────────────────────────────────────────
    lines.append("## Dimension Overview")
    lines.append("")
    lines.append("| Dimension | Rating | Summary |")
    lines.append("|-----------|--------|---------|")
    for dim in report.dimensions:
        if dim.rating is None:
            rating_str = "N/A"
        else:
            rating_str = f"{_RATING_EMOJI.get(dim.rating, '🟡')} {dim.rating}/5"
        lines.append(f"| {dim.name} | {rating_str} | {dim.summary} |")
    lines.append("")

    # ── Individual dimension sections ─────────────────────────────────────────
    md_critical_periods = [
        sp for sp in (report.stagnation_periods or [])
        if sp.length > 1 and sp.severity == "critical"
    ]

    lines.append("## Dimension Reports")
    lines.append("")
    for dim in report.dimensions:
        if dim.rating is None:
            lines.append(f"### {dim.name} — N/A")
        else:
            emoji = _RATING_EMOJI.get(dim.rating, "🟡")
            lines.append(f"### {dim.name} — {emoji} {dim.rating}/5")
        lines.append("")
        lines.append(f"**Summary:** {dim.summary}")
        lines.append("")
        if dim.evidence:
            lines.append("**Evidence:**")
            for ev in dim.evidence:
                stripped = ev.lstrip()
                indent = len(ev) - len(stripped)
                if indent == 0:
                    lines.append(f"- {stripped}")
                elif indent == 2:
                    # sub-item (e.g. violation bullet: "  • [sev] ...")
                    body = stripped.lstrip("• ").strip()
                    lines.append(f"  - {body}")
                elif indent >= 4:
                    if stripped.startswith("Fix: "):
                        lines.append(f"  - **Fix:** {stripped[len('Fix: '):]}")
                    else:
                        lines.append(f"  - {stripped}")
            lines.append("")
        if dim.name == "Stagnation" and md_critical_periods:
            lines.append("**Critical stagnation details:**")
            lines.append("")
            for sp in md_critical_periods:
                end_str = str(sp.end_iteration) if sp.end_iteration is not None else "ongoing"
                lines.append(
                    f"- **Streak {sp.streak_id}** (iters {sp.start_iteration}–{end_str}, "
                    f"len={sp.length}): failure=`{sp.dominant_failure_type}`"
                )
                crash_samples = getattr(sp, "crash_samples", None)
                if crash_samples:
                    for sample in crash_samples:
                        lines.append(f"  - iter {sample['iteration']}: `{sample['error']}`")
                if sp.llm_analysis:
                    category = sp.llm_analysis.get("category", "")
                    explanation = sp.llm_analysis.get("explanation", "")
                    if category:
                        lines.append(f"  - **LLM category:** {category}")
                    if explanation:
                        lines.append(f"  - **LLM analysis:** {explanation}")
                lines.append(f"  - **Recommendation:** {sp.recommendation}")
            lines.append("")
        if dim.historical is not None:
            hist_summary = getattr(dim.historical, "summary", str(dim.historical))
            lines.append(f"**Historical:** {hist_summary}")
            lines.append("")
        lines.append(f"**Recommendation:** {dim.recommendation}")
        lines.append("")

    # ── Cross-dimension interactions ──────────────────────────────────────────
    lines.append("## Cross-Dimension Interactions")
    lines.append("")
    lines.append(report.cross_dimension_interactions)
    lines.append("")

    # ── Novel observations ────────────────────────────────────────────────────
    if report.novel_observations:
        lines.append("## Novel Observations")
        lines.append("")
        for obs in report.novel_observations:
            lines.append(f"- {obs}")
        lines.append("")

    return "\n".join(lines)


def _fmt_float(value: float) -> str:
    """Format a float, returning 'N/A' for NaN/Inf."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    return f"{value:.6g}"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """run-evolve-analysis — evolutionary code optimization diagnostic tool."""


# ---------------------------------------------------------------------------
# postmortem command
# ---------------------------------------------------------------------------

@cli.command("postmortem")
@click.option(
    "--source",
    required=True,
    type=click.Choice(
        ["skydiscover", "shinkaevolve", "openevolve", "jsonl"],
        case_sensitive=False,
    ),
    help="Source format of the experiment output.",
)
@click.option(
    "--path",
    required=True,
    type=click.Path(),
    help="Path to the checkpoint directory, DB file, or JSONL file.",
)
@click.option(
    "--provider",
    default=None,
    help="LLM provider (passed to LiteLLM). Defaults to value in config file.",
)
@click.option(
    "--model",
    default=None,
    help="LLM model name. Defaults to value in config file.",
)
@click.option(
    "--output-dir",
    default="./evolve_analysis_output",
    show_default=True,
    type=click.Path(),
    help="Directory for report output files.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=False),
    help="Path to a YAML config file.",
)
@click.option(
    "--stagnation-threshold",
    default=10,
    show_default=True,
    type=int,
    help="Minimum streak length to trigger a stagnation alert.",
)
@click.option(
    "--min-delta",
    default=0.001,
    show_default=True,
    type=float,
    help="Minimum score improvement counted as progress.",
)
@click.option(
    "--historical-db",
    default=None,
    type=click.Path(),
    help="Path to a historical SQLite DB for comparative analysis.",
)
@click.option(
    "--no-llm",
    is_flag=True,
    default=False,
    help="Skip all qualitative LLM judge steps.",
)
@click.option(
    "--experiment-id",
    default="",
    help="String identifier for this experiment.",
)
@click.option(
    "--trace-path",
    default=None,
    type=click.Path(),
    help="Path to evolution trace JSONL (for openevolve source).",
)
@click.option(
    "--baseline-score",
    default=None,
    type=float,
    help="External baseline score to compare against (e.g. 5-run LRU average).",
)
@click.option(
    "--baseline-metrics",
    default=None,
    type=str,
    help='JSON dict of baseline sub-metrics, e.g. \'{"mean_ttft_ms": 85.34, "p99_ttft_ms": 640.89}\'.',
)
def postmortem(
    source: str,
    path: str,
    provider: str,
    model: str,
    output_dir: str,
    config_path: Optional[str],
    stagnation_threshold: int,
    min_delta: float,
    historical_db: Optional[str],
    no_llm: bool,
    experiment_id: str,
    trace_path: Optional[str],
    baseline_score: Optional[float],
    baseline_metrics: Optional[str],
) -> None:
    """Run a full Phase 1 post-mortem analysis on an evolutionary experiment."""

    # ── Build overrides from CLI args ─────────────────────────────────────────
    llm_overrides: dict = {}
    if provider is not None:
        llm_overrides["provider"] = provider
    if model is not None:
        llm_overrides["model"] = model

    overrides: dict = {
        "ingestion": {
            "source": source,
            "path": path,
        },
        "stagnation": {
            "threshold": stagnation_threshold,
            "min_delta": min_delta,
        },
        "historical": {
            "db_path": historical_db,
        },
        "_no_llm": no_llm,
    }

    if llm_overrides:
        overrides["llm"] = llm_overrides

    # Forward trace_path into ingestion for openevolve
    if trace_path is not None:
        overrides["ingestion"]["trace_path"] = trace_path

    # Forward baseline into config
    baseline_cfg: dict = {}
    if baseline_score is not None:
        baseline_cfg["score"] = baseline_score
    if baseline_metrics is not None:
        try:
            baseline_cfg["metrics"] = json.loads(baseline_metrics)
        except json.JSONDecodeError as exc:
            raise click.BadParameter(
                f"--baseline-metrics is not valid JSON: {exc}", param_hint="--baseline-metrics"
            )
    if baseline_cfg:
        overrides["baseline"] = baseline_cfg

    # ── Load merged config ────────────────────────────────────────────────────
    config = load_config(config_path=config_path, overrides=overrides)
    if config_path:
        config["_config_path"] = config_path

    # ── Run pipeline ──────────────────────────────────────────────────────────
    logger.info(
        "Starting post-mortem analysis: source=%r  path=%r  experiment_id=%r",
        source,
        path,
        experiment_id or "<none>",
    )
    try:
        report, iter_df = run_postmortem(config, experiment_id=experiment_id)
    except Exception as exc:
        logger.error("Post-mortem pipeline failed: %s", exc, exc_info=True)
        raise click.ClickException(str(exc)) from exc

    # ── Write output files ────────────────────────────────────────────────────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report_json_path = out_path / "report.json"
    report_txt_path = out_path / "report.txt"
    report_md_path = out_path / "report.md"

    report_dict = report_to_dict(report)
    with report_json_path.open("w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2, ensure_ascii=False)
    logger.info("JSON report saved to %s", report_json_path)

    report_text = report_to_text(report)
    with report_txt_path.open("w", encoding="utf-8") as fh:
        fh.write(report_text)
    logger.info("Text report saved to %s", report_txt_path)

    report_md = report_to_markdown(report)
    with report_md_path.open("w", encoding="utf-8") as fh:
        fh.write(report_md)
    logger.info("Markdown report saved to %s", report_md_path)

    if iter_df is not None and not iter_df.empty:
        df_path = out_path / "df.parquet"
        iter_df.to_parquet(df_path, index=False)
        logger.info("Iteration DataFrame saved to %s", df_path)

    # ── Print summary to stdout ───────────────────────────────────────────────
    click.echo("")
    click.echo(report_text)
    click.echo(f"\nReports written to: {out_path.resolve()} (report.txt, report.md, report.json)")


# ---------------------------------------------------------------------------
# show-report command
# ---------------------------------------------------------------------------

@cli.command("show-report")
@click.option(
    "--report-dir",
    default="./evolve_analysis_output",
    show_default=True,
    type=click.Path(exists=True),
    help="Directory containing report.json.",
)
def show_report(report_dir: str) -> None:
    """Read report.json from a previous run and print the text report to stdout."""
    report_path = Path(report_dir) / "report.json"

    if not report_path.is_file():
        raise click.ClickException(f"report.json not found in {report_dir}")

    with report_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Re-hydrate a minimal EvolveLoopReport from the stored dict
    report = _dict_to_report(data)
    click.echo(report_to_text(report))


# ---------------------------------------------------------------------------
# Re-hydration helper (show-report)
# ---------------------------------------------------------------------------

def _dict_to_report(data: dict) -> EvolveLoopReport:
    """
    Reconstruct an EvolveLoopReport from the serialised dict produced by
    report_to_dict.  Only fields needed for report_to_text are populated.
    """
    from skydiscover.extras.evolve_analyzer.report_synthesizer import (
        AggregateStats,
        DimensionReport,
        LLMJudgeStatus,
        StagnationPeriodReport,
    )

    def _agg(d: dict) -> AggregateStats:
        return AggregateStats(
            total_iterations=d.get("total_iterations", 0),
            n_evaluations=d.get("n_evaluations", 0),
            n_successful=d.get("n_successful", 0),
            n_stagnation_periods=d.get("n_stagnation_periods", 0),
            total_stagnation_iterations=d.get("total_stagnation_iterations", 0),
            best_score=d.get("best_score") or float("nan"),
            worst_score=d.get("worst_score") or float("nan"),
            final_score=d.get("final_score") or float("nan"),
            total_llm_cost_usd=d.get("total_llm_cost_usd"),
            total_duration_hours=d.get("total_duration_hours"),
            baseline_score=d.get("baseline_score"),
            baseline_metrics=d.get("baseline_metrics"),
            score_improvement_vs_baseline=d.get("score_improvement_vs_baseline"),
        )

    def _dim(d: dict) -> DimensionReport:
        return DimensionReport(
            name=d.get("name", ""),
            rating=d.get("rating"),
            rating_label=d.get("rating_label", ""),
            summary=d.get("summary", ""),
            evidence=d.get("evidence") or [],
            historical=d.get("historical"),
            recommendation=d.get("recommendation", ""),
            data_available=d.get("data_available", True),
        )

    def _stag(d: dict) -> StagnationPeriodReport:
        return StagnationPeriodReport(
            streak_id=d.get("streak_id", ""),
            start_iteration=d.get("start_iteration", 0),
            end_iteration=d.get("end_iteration"),
            length=d.get("length", 0),
            severity=d.get("severity", "warning"),
            dominant_failure_type=d.get("dominant_failure_type", ""),
            llm_analysis=d.get("llm_analysis"),
            recommendation=d.get("recommendation", ""),
            crash_samples=d.get("crash_samples"),
        )

    js_raw = data.get("llm_judge_status")
    llm_judge_status = None
    if isinstance(js_raw, dict):
        llm_judge_status = LLMJudgeStatus(
            provider=js_raw.get("provider", ""),
            model=js_raw.get("model", ""),
            base_url=js_raw.get("base_url"),
            status=js_raw.get("status", "unknown"),
            skip_reason=js_raw.get("skip_reason"),
            error=js_raw.get("error"),
        )

    return EvolveLoopReport(
        experiment_id=data.get("experiment_id", ""),
        executive_summary=data.get("executive_summary", ""),
        executive_summary_md=data.get("executive_summary_md", data.get("executive_summary", "")),
        dimensions=[_dim(d) for d in data.get("dimensions") or []],
        cross_dimension_interactions=data.get("cross_dimension_interactions", ""),
        stagnation_periods=[_stag(s) for s in data.get("stagnation_periods") or []],
        aggregate_stats=_agg(data.get("aggregate_stats") or {}),
        novel_observations=data.get("novel_observations") or [],
        llm_judge_status=llm_judge_status,
        algorithm_class=data.get("algorithm_class", "population_evolutionary"),
        algorithm_name=data.get("algorithm_name"),
        num_islands=data.get("num_islands"),
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main() -> None:
    cli()


def dashboard_main() -> None:
    """Entry point for `dashboard-evolve-analysis`."""
    import subprocess
    from pathlib import Path

    import argparse as _ap
    p = _ap.ArgumentParser(prog="dashboard-evolve-analysis")
    p.add_argument("--report-dir", default="./evolve_analysis_output",
                   help="Directory containing report.json")
    args = p.parse_args()

    script = str(Path(__file__).parent / "dashboard.py")
    try:
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", script,
            "--", "--report-dir", args.report_dir,
        ])
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
