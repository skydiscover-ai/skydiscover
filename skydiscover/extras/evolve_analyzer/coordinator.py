"""
Phase 1 coordinator — orchestrates the full post-mortem analysis pipeline.

Flow:
  load_evolve_records → quantitative pass → qualitative pass → comparative pass → synthesize
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from skydiscover.extras.evolve_analyzer.ingestion.checkpoint_adapter import load_evolve_records, detect_algorithm_class, detect_algorithm_name
from skydiscover.extras.evolve_analyzer.quantitative.bundle import QuantitativeBundle
from skydiscover.extras.evolve_analyzer.quantitative.stagnation_detector import detect_stagnation
from skydiscover.extras.evolve_analyzer.quantitative.evaluator_analyzer import analyze_evaluator
from skydiscover.extras.evolve_analyzer.quantitative.compliance_checker import analyze_compliance
from skydiscover.extras.evolve_analyzer.quantitative.convergence_analyzer import analyze_convergence
from skydiscover.extras.evolve_analyzer.quantitative.exploration_analyzer import analyze_exploration
from skydiscover.extras.evolve_analyzer.quantitative.regression_analyzer import analyze_regressions
from skydiscover.extras.evolve_analyzer.quantitative.efficiency_analyzer import analyze_efficiency
from skydiscover.extras.evolve_analyzer.quantitative.search_space_analyzer import analyze_search_space
from skydiscover.extras.evolve_analyzer.quantitative.meta_analyzer import analyze_meta_quality
from skydiscover.extras.evolve_analyzer.quantitative.ceiling_analyzer import analyze_ceiling
from skydiscover.extras.evolve_analyzer.quantitative.sub_metric_analyzer import analyze_sub_metrics
from skydiscover.extras.evolve_analyzer.quantitative.infrastructure_analyzer import analyze_infrastructure
from skydiscover.extras.evolve_analyzer.qualitative.qualitative_analyzer import QualitativeAnalyzer
from skydiscover.extras.evolve_analyzer.llm.client import get_llm_client
from skydiscover.extras.evolve_analyzer.historical_db import HistoricalDB
from skydiscover.extras.evolve_analyzer.report_synthesizer import synthesize_report, EvolveLoopReport, LLMJudgeStatus

import pandas as pd

logger = logging.getLogger(__name__)


def run_quantitative_pass(records: List[dict], config: dict) -> QuantitativeBundle:
    """
    Runs all 10 deterministic analyzers in sequence.

    Steps with side effects (tagging records) are applied first, then metrics
    are computed from the enriched records.

    Parameters
    ----------
    records:
        List of normalised iteration dicts from load_evolve_records.
    config:
        Full experiment config dict.

    Returns
    -------
    QuantitativeBundle
        Populated bundle containing all sub-metrics and the enriched DataFrame.
    """
    # ── Step 1: tag failure_mode and high_variance ────────────────────────────
    evaluator_cfg = config.get("evaluator", {})
    std_threshold = evaluator_cfg.get("score_variance_threshold", 0.1)
    analyze_evaluator(records, std_threshold=std_threshold)

    # ── Step 2: tag evolved_block_only, format_valid_checked, signature_preserved
    compliance_cfg = config.get("compliance", {})
    analyze_compliance(records, compliance_cfg)

    # ── Step 3: tag streak_id and streak_position; also returns periods ───────
    stagnation_cfg = config.get("stagnation", {})
    threshold = stagnation_cfg.get("threshold", 10)
    min_delta = stagnation_cfg.get("min_delta", 0.001)
    records, stagnation_periods = detect_stagnation(
        records, threshold=threshold, min_delta=min_delta
    )

    # ── Step 4: convergence ───────────────────────────────────────────────────
    convergence_cfg = config.get("convergence", {})
    window = convergence_cfg.get("window", 10)
    convergence = analyze_convergence(records, window=window)

    # ── Step 5: exploration ───────────────────────────────────────────────────
    exploration = analyze_exploration(records)

    # ── Step 6: regressions ───────────────────────────────────────────────────
    regression = analyze_regressions(records)

    # ── Step 7: efficiency ────────────────────────────────────────────────────
    efficiency = analyze_efficiency(records)

    # ── Step 8: search space ──────────────────────────────────────────────────
    search_space_cfg = config.get("search_space", {})
    top_k = search_space_cfg.get("top_k", 10)
    search_space = analyze_search_space(records, top_k=top_k)

    # ── Step 9: meta-analysis quality ─────────────────────────────────────────
    meta_analysis = analyze_meta_quality(records)

    # ── Step 10: ceiling ──────────────────────────────────────────────────────
    ceiling = analyze_ceiling(records)

    # ── Step 11: sub-metric trajectories ─────────────────────────────────────
    sub_metrics = analyze_sub_metrics(records)

    # ── Step 12: infrastructure failure detection ─────────────────────────────
    infrastructure = analyze_infrastructure(records)

    # ── Assemble bundle ───────────────────────────────────────────────────────
    df = pd.DataFrame(records)

    bundle = QuantitativeBundle(
        df=df,
        stagnation_periods=stagnation_periods,
        convergence=convergence,
        exploration=exploration,
        regression=regression,
        efficiency=efficiency,
        search_space=search_space,
        meta_analysis=meta_analysis,
        ceiling=ceiling,
        sub_metrics=sub_metrics,
        infrastructure=infrastructure,
    )

    # ── Data availability map ─────────────────────────────────────────────────
    bundle.data_availability = {
        "convergence": convergence is not None and bool(convergence.best_so_far_curve),
        "stagnation": True,  # always has a result (may be empty list)
        "regression": regression is not None,
        "efficiency": efficiency is not None,
        "exploration": exploration is not None,
        "search_space": (
            search_space is not None and search_space.effective_dimensionality > 0
        ),
        "meta_analysis": (
            meta_analysis is not None
            and (
                meta_analysis.suggestion_follow_rate is not None
                or meta_analysis.conditional_improvement_rate is not None
            )
        ),
        "ceiling": ceiling is not None,
        "sub_metrics": sub_metrics is not None,
        "infrastructure": infrastructure is not None and infrastructure.sentinel_count > 0,
    }

    return bundle


def run_postmortem(config: dict, experiment_id: str = "") -> Tuple[EvolveLoopReport, "pd.DataFrame"]:
    """
    Full Phase 1 pipeline: load → quantitative → qualitative → historical → synthesize.

    Parameters
    ----------
    config:
        Full experiment config dict (use load_config() to build one).
    experiment_id:
        Human-readable identifier for this experiment run.

    Returns
    -------
    EvolveLoopReport
    """
    # ── 1. Load records ───────────────────────────────────────────────────────
    ingestion_cfg = config.get("ingestion", {})
    source = ingestion_cfg.get("source", "jsonl")
    path = ingestion_cfg.get("path", "")

    # Forward any extra ingestion kwargs (e.g. trace_path for openevolve)
    extra_kwargs: dict = {
        k: v
        for k, v in ingestion_cfg.items()
        if k not in ("source", "path")
    }

    records: List[dict] = load_evolve_records(source, path, **extra_kwargs)
    algorithm_class = detect_algorithm_class(source, records)
    algorithm_name = detect_algorithm_name(source, path)

    if not records:
        logger.warning(
            "No records loaded from source=%r path=%r — returning minimal report.",
            source,
            path,
        )
        empty_bundle = QuantitativeBundle(df=pd.DataFrame())
        empty_bundle.data_availability = {}
        return synthesize_report(
            quant=empty_bundle,
            qual=None,
            historical=[],
            config=config,
            experiment_id=experiment_id,
            llm_judge_status=None,
            algorithm_class=algorithm_class,
            algorithm_name=algorithm_name,
        ), pd.DataFrame()

    # ── 2. Quantitative pass ──────────────────────────────────────────────────
    bundle = run_quantitative_pass(records, config)

    # ── 3. Qualitative pass (optional) ───────────────────────────────────────
    qual = None
    llm_judge_status: Optional[LLMJudgeStatus] = None
    llm_cfg = config.get("llm")
    judges_cfg = config.get("judges", {})
    all_judges_disabled = not any(
        v for k, v in judges_cfg.items()
        if k not in ("async_dispatch", "sample_rate", "artifact_batch_size")
        and isinstance(v, bool)
    )
    skip_llm = config.get("_no_llm", False) or all_judges_disabled or not llm_cfg

    if skip_llm:
        _llm = llm_cfg or {}
        if config.get("_no_llm", False):
            skip_reason = "--no-llm flag set"
        elif all_judges_disabled:
            skip_reason = "all judges disabled in config"
        else:
            skip_reason = "no LLM config provided"
        llm_judge_status = LLMJudgeStatus(
            provider=_llm.get("provider", "unknown"),
            model=_llm.get("model", "unknown"),
            base_url=_llm.get("base_url"),
            status="skipped",
            skip_reason=skip_reason,
        )
    else:
        _llm = llm_cfg or {}
        provider = _llm.get("provider", "openai")
        model = _llm.get("model", "gpt-4o-mini")
        base_url = _llm.get("base_url")
        api_key_env = _llm.get("api_key_env", "EVOLVE_ANALYZER_API_KEY")
        try:
            _reserved = {"provider", "model", "base_url", "api_key_env", "overrides", "max_cost_usd"}
            parameters = {k: v for k, v in _llm.items() if k not in _reserved}

            logger.info(
                "Initializing LLM client with: provider=%s, model=%s, base_url=%s, api_key_env=%s",
                provider, model, base_url, api_key_env
            )

            llm_client = get_llm_client(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key_env=api_key_env,
                parameters=parameters,
            )
            analyzer = QualitativeAnalyzer(
                llm_client=llm_client,
                config=config,
            )
            qual = analyzer.run(bundle)
            llm_judge_status = LLMJudgeStatus(
                provider=provider,
                model=model,
                base_url=base_url,
                status="success",
            )
        except Exception as exc:  # noqa: BLE001
            exc_str = str(exc)
            llm_judge_status = LLMJudgeStatus(
                provider=provider,
                model=model,
                base_url=base_url,
                status="failed",
                error=exc_str,
            )
            if "401" in exc_str or "authentication" in exc_str.lower() or "unauthorized" in exc_str.lower():
                logger.warning(
                    "Qualitative pass skipped — LLM authentication failed (401). "
                    "Check that %s is set to a valid virtual key for %s. Error: %s",
                    api_key_env, base_url or provider, exc,
                )
            else:
                logger.warning("Qualitative pass failed (%s); continuing without it.", exc)

    # ── 4. Historical comparison (optional) ───────────────────────────────────
    historical_comparisons: List = []
    historical_cfg = config.get("historical", {})
    db_path = historical_cfg.get("db_path")

    if db_path is not None:
        try:
            min_experiments = historical_cfg.get("min_experiments", 5)
            pattern_threshold = historical_cfg.get("pattern_promotion_threshold", 3)
            db = HistoricalDB(
                db_path=db_path,
                min_experiments=min_experiments,
                pattern_promotion_threshold=pattern_threshold,
            )
            # Compare a standard set of key metrics
            metric_map = {
                "convergence_rate": (
                    getattr(bundle.convergence, "convergence_rate", None)
                ),
                "regression_frequency": (
                    getattr(bundle.regression, "regression_frequency", None)
                ),
                "structural_diversity_index": (
                    getattr(bundle.exploration, "structural_diversity_index", None)
                ),
                "productive_phase_fraction": (
                    getattr(bundle.efficiency, "productive_phase_fraction", None)
                ),
                "estimated_gain_probability": (
                    getattr(bundle.ceiling, "estimated_gain_probability", None)
                ),
                "stagnation_count": float(len(bundle.stagnation_periods)),
                "max_stagnation_length": float(
                    max((p.length for p in bundle.stagnation_periods), default=0)
                ),
                "mean_recovery_time": (
                    getattr(bundle.regression, "mean_recovery_time", None)
                ),
                "time_to_best_fraction": (
                    getattr(bundle.convergence, "time_to_best_fraction", None)
                ),
            }
            for metric_name, value in metric_map.items():
                if value is not None:
                    try:
                        comparison = db.compare(metric_name, float(value))
                        historical_comparisons.append(comparison)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "Historical comparison for %r failed: %s", metric_name, exc
                        )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Historical DB access failed (%s); continuing.", exc)

    # ── 5. Synthesize report ──────────────────────────────────────────────────
    report = synthesize_report(
        quant=bundle,
        qual=qual,
        historical=historical_comparisons,
        config=config,
        experiment_id=experiment_id,
        llm_judge_status=llm_judge_status,
        algorithm_class=algorithm_class,
        algorithm_name=algorithm_name,
    )

    # ── 6. Record experiment in historical DB (optional) ──────────────────────
    if db_path is not None:
        try:
            db.record_experiment(
                experiment_id=experiment_id or report.experiment_id,
                metrics=bundle,
                tool=ingestion_cfg.get("source", ""),
                benchmark=config.get("benchmark", ""),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to record experiment in historical DB: %s", exc)

    return report, bundle.df


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> dict:
    """Load default config and merge with user config and CLI overrides.

    Precedence (highest wins): overrides > user config > default config.

    Parameters
    ----------
    config_path:
        Path to a user-supplied YAML config file.  If None, only the
        package default config and *overrides* are used.
    overrides:
        Dict of key/value pairs that take highest precedence.  Nested
        keys are expressed as nested dicts (not dotted paths).

    Returns
    -------
    dict
        Merged configuration dict.
    """
    # ── Load default config ───────────────────────────────────────────────────
    default_config_path = (
        Path(__file__).resolve().parent / "config" / "default_config.yaml"
    )
    config: dict = {}
    if default_config_path.is_file():
        with default_config_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                config = loaded
    else:
        logger.warning(
            "Default config not found at %s; using empty defaults.",
            default_config_path,
        )

    # ── Merge user config ─────────────────────────────────────────────────────
    if config_path is not None:
        user_path = Path(config_path)
        if user_path.is_file():
            with user_path.open("r", encoding="utf-8") as fh:
                user_cfg = yaml.safe_load(fh)
            if isinstance(user_cfg, dict):
                config = _deep_merge(config, user_cfg)
        else:
            logger.warning("User config path %r not found; ignoring.", config_path)

    # ── Apply CLI overrides ───────────────────────────────────────────────────
    if overrides:
        config = _deep_merge(config, overrides)

    return config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge *override* into a copy of *base*.

    For keys present in both, if both values are dicts they are merged
    recursively; otherwise the override value wins.
    """
    result = dict(base)
    for key, override_val in override.items():
        base_val = result.get(key)
        if isinstance(base_val, dict) and isinstance(override_val, dict):
            result[key] = _deep_merge(base_val, override_val)
        else:
            result[key] = override_val
    return result
