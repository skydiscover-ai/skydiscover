"""
Thin wrapper around ShinkaEvolve (https://github.com/ShinkaiEvolve).

Delegates entirely to ShinkaEvolve's public API so upstream updates are
picked up automatically.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from skydiscover.api import DiscoveryResult
from skydiscover.config import Config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config mapping
# ------------------------------------------------------------------


def _map_config(config: Config, iterations: Optional[int], evaluator_path: str, output_dir: str):
    """Convert SkyDiscover Config to ShinkaEvolve's three config objects."""
    from dataclasses import fields as dc_fields

    from shinka.core.runner import EvolutionConfig
    from shinka.database import DatabaseConfig as ShinkaDBC
    from shinka.launch import LocalJobConfig

    # Power-user escape hatch
    ext = getattr(config, "external_config", None)
    if ext is not None and isinstance(ext, dict):
        evo = ext.get("evo_config", EvolutionConfig())
        job = ext.get("job_config", LocalJobConfig(eval_program_path=evaluator_path))
        dbc = ext.get("db_config", ShinkaDBC())
        if iterations is not None:
            evo.num_generations = iterations
        return evo, job, dbc

    # Load tuned backend defaults
    from skydiscover.extras.external.defaults import load_defaults

    defaults = load_defaults("shinkaevolve_default.yaml")
    evo_defaults = defaults.get("evolution", {})
    db_defaults = defaults.get("database", {})

    # EvolutionConfig
    evo_kwargs: Dict[str, Any] = {
        "num_generations": iterations or config.max_iterations,
        "results_dir": output_dir,
        "job_type": "local",
        "language": getattr(config, "language", None) or "python",
    }

    # Map LLM model names (from --model / -c config)
    if config.llm.models:
        evo_kwargs["llm_models"] = [m.name for m in config.llm.models]

    # Apply tuned defaults for evolution (patch types, LLM kwargs, meta, etc.)
    valid_evo_fields = {f.name for f in dc_fields(EvolutionConfig)}
    for key, value in evo_defaults.items():
        if key in valid_evo_fields and key not in evo_kwargs:
            evo_kwargs[key] = value

    # Cap parallelism to iteration count to avoid over-submission when
    # target generations is small (ShinkaEvolve auto-scales proposal jobs
    # from max_parallel_jobs; too many slots for few generations causes
    # directory collisions and stuck detection).
    iters = evo_kwargs["num_generations"]
    if "max_parallel_jobs" in evo_kwargs and iters < evo_kwargs["max_parallel_jobs"]:
        evo_kwargs["max_parallel_jobs"] = max(1, iters)

    # Meta model follows the main model
    if config.llm.models and evo_defaults.get("meta_rec_interval"):
        evo_kwargs["meta_llm_models"] = [config.llm.models[0].name]

    # System prompt -> task_sys_msg
    sys_prompt = config.system_prompt_override
    if sys_prompt is None and hasattr(config, "context_builder"):
        sp = config.context_builder.system_message
        if sp and sp not in ("system_message", "evaluator_system_message"):
            sys_prompt = sp
    if sys_prompt:
        evo_kwargs["task_sys_msg"] = sys_prompt

    evo = EvolutionConfig(**evo_kwargs)

    # --- JobConfig ---
    job_kwargs: Dict[str, Any] = {"eval_program_path": evaluator_path}
    if hasattr(config, "evaluator") and config.evaluator.timeout:
        secs = int(config.evaluator.timeout)
        h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
        job_kwargs["time"] = f"{h:02d}:{m:02d}:{s:02d}"
    job = LocalJobConfig(**job_kwargs)

    # --- DatabaseConfig ---
    valid_db_fields = {f.name for f in dc_fields(ShinkaDBC)}
    dbc_kwargs = {k: v for k, v in db_defaults.items() if k in valid_db_fields}
    dbc = ShinkaDBC(**dbc_kwargs)

    return evo, job, dbc


# ------------------------------------------------------------------
# Initial score extraction
# ------------------------------------------------------------------


def _get_initial_score(all_programs: list) -> float:
    """Extract initial (generation 0) score from ShinkaEvolve programs list."""
    initial_score = 0.0
    for p in all_programs:
        gen = getattr(p, "generation", 0)
        if gen == 0:
            score = getattr(p, "combined_score", None)
            if score is not None:
                initial_score = max(initial_score, float(score))
    return initial_score


# ------------------------------------------------------------------
# Program conversion
# ------------------------------------------------------------------


def _to_skydiscover_program(sp):
    """Convert a ShinkaEvolve Program to SkyDiscover's Program dataclass."""
    from skydiscover.search.base_database import Program

    metrics = dict(sp.public_metrics) if sp.public_metrics else {}
    metrics["combined_score"] = float(sp.combined_score or 0.0)
    metrics["correct"] = sp.correct

    return Program(
        id=sp.id,
        solution=sp.code,
        language=getattr(sp, "language", "python"),
        metrics=metrics,
        iteration_found=getattr(sp, "generation", 0),
        parent_id=getattr(sp, "parent_id", None),
        generation=getattr(sp, "generation", 0),
        timestamp=getattr(sp, "timestamp", 0.0),
    )


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------


async def run(
    program_path: str,
    evaluator_path: str,
    config_obj: Config,
    iterations: int,
    output_dir: str,
    monitor_callback=None,
    feedback_reader=None,
) -> DiscoveryResult:
    """Run evolution using the ShinkaEvolve package."""
    from shinka.core import AsyncEvolutionRunner

    from skydiscover.api import DiscoveryResult
    from skydiscover.config import bridge_provider_env

    bridge_provider_env(config_obj)

    evo_config, job_config, db_config = _map_config(
        config_obj,
        iterations,
        evaluator_path,
        output_dir,
    )

    # Human feedback: set initial system prompt on feedback reader for dashboard visibility
    if feedback_reader and evo_config.task_sys_msg:
        feedback_reader.set_current_prompt(evo_config.task_sys_msg)

    # ShinkaEvolve supports passing code as strings directly
    with open(program_path, "r") as f:
        init_str = f.read()
    with open(evaluator_path, "r") as f:
        eval_str = f.read()

    # ShinkaEvolve runs the evaluator as a CLI subprocess:
    #   python evaluate.py --program_path X --results_dir Y
    # and expects results written to results_dir/metrics.json.
    # SkyDiscover evaluators are just a function: evaluate(path) -> dict.
    # Bridge the gap by appending a CLI wrapper.
    if "__main__" not in eval_str:
        eval_str += """

if __name__ == "__main__":
    import argparse, json, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    result = evaluate(args.program_path)
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(result, f)
"""

    runner = AsyncEvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        init_program_str=init_str,
        evaluate_str=eval_str,
    )

    # Monitor polling task + Human feedback injection
    seen_ids: set = set()
    poll_task = None

    if monitor_callback or feedback_reader:

        async def _poll_programs():
            _last_feedback = ""
            while True:
                await asyncio.sleep(2)
                # Poll new programs for monitor
                if monitor_callback:
                    try:
                        all_progs = runner.db.get_all_programs()
                        for p in all_progs:
                            if p.id not in seen_ids:
                                seen_ids.add(p.id)
                                sky_prog = _to_skydiscover_program(p)
                                monitor_callback(sky_prog, getattr(p, "generation", 0))
                    except Exception:
                        logger.debug("Monitor poll error", exc_info=True)
                # Human feedback: inject feedback into ShinkaEvolve's prompt sampler
                if feedback_reader:
                    try:
                        feedback = feedback_reader.read()
                        if feedback != _last_feedback:
                            _last_feedback = feedback
                            sampler = getattr(runner, "prompt_sampler", None)
                            original_prompt = evo_config.task_sys_msg or ""
                            if feedback and sampler:
                                if feedback_reader.mode == "replace":
                                    sampler.task_sys_msg = feedback
                                else:
                                    sampler.task_sys_msg = (
                                        original_prompt + "\n\n## Human Guidance\n" + feedback
                                    )
                                feedback_reader.set_current_prompt(sampler.task_sys_msg)
                                logger.debug(
                                    f"Human feedback injected into ShinkaEvolve ({len(feedback)} chars, mode={feedback_reader.mode})"
                                )
                            elif sampler and not feedback:
                                # Feedback cleared — revert to original
                                sampler.task_sys_msg = original_prompt
                                feedback_reader.set_current_prompt(original_prompt)
                    except Exception:
                        logger.debug("Human feedback injection error", exc_info=True)

        poll_task = asyncio.create_task(_poll_programs())

    await runner.run()

    if poll_task:
        poll_task.cancel()
        # Flush remaining programs
        try:
            for p in runner.db.get_all_programs():
                if p.id not in seen_ids:
                    seen_ids.add(p.id)
                    try:
                        monitor_callback(_to_skydiscover_program(p), getattr(p, "generation", 0))
                    except Exception:
                        logger.debug("Monitor flush error", exc_info=True)
        except Exception:
            logger.debug("Final program flush error", exc_info=True)

    # Extract results from the ShinkaEvolve database
    best_sp = runner.db.get_best_program()
    all_programs = runner.db.get_all_programs()

    # get_best_program() only returns "correct" programs. For continuous-score
    # problems (no pass/fail), fall back to the highest-scoring program overall.
    if best_sp is None and all_programs:
        best_sp = max(all_programs, key=lambda p: float(getattr(p, "combined_score", 0) or 0))

    initial_score = _get_initial_score(all_programs)

    best_skydiscover = _to_skydiscover_program(best_sp) if best_sp else None
    best_score = float(best_sp.combined_score or 0.0) if best_sp else 0.0

    return DiscoveryResult(
        best_program=best_skydiscover,
        best_score=best_score,
        best_solution=best_sp.code if best_sp else "",
        metrics=best_skydiscover.metrics if best_skydiscover else {},
        output_dir=output_dir,
        initial_score=initial_score,
    )
