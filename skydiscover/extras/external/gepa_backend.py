"""
Thin wrapper around GEPA's optimize_anything API.

Adapts SkyDiscover's file-based evaluator interface to GEPA's candidate-string
evaluator, and maps domain context (system_prompt) to GEPA's
objective/background parameters.
"""

import importlib.util
import logging
import os
import sys
import tempfile
import uuid
from typing import Any, Dict

from skydiscover.api import DiscoveryResult
from skydiscover.config import Config, _parse_model_spec

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Evaluator adapter: SkyDiscover file-based -> GEPA candidate-string
# ------------------------------------------------------------------


def _make_gepa_evaluator(
    evaluator_path: str, monitor_callback=None, solution_prefix: str = "", solution_suffix: str = ""
):
    """
    Wrap a SkyDiscover-style evaluator (evaluate(program_path) -> dict)
    into a GEPA-style evaluator (evaluate(candidate_str) -> (score, side_info)).
    """
    spec = importlib.util.spec_from_file_location("_skydiscover_eval", evaluator_path)
    eval_module = importlib.util.module_from_spec(spec)

    eval_dir = os.path.dirname(os.path.abspath(evaluator_path))
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    spec.loader.exec_module(eval_module)
    user_evaluate = eval_module.evaluate
    eval_counter = [0]

    def gepa_evaluator(candidate: str, **kwargs) -> tuple[float, dict]:
        # Write candidate code to a temp file so the SkyDiscover evaluator can load it.
        # Reconstruct the full file if prefix/suffix exist (code outside EVOLVE block).
        full_solution = solution_prefix + candidate + solution_suffix
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="gepa_candidate_",
            delete=False,
        )
        try:
            tmp.write(full_solution)
            tmp.flush()
            tmp.close()

            metrics = user_evaluate(tmp.name)

            # Normalise to (score, side_info)
            if isinstance(metrics, (int, float)):
                score = float(metrics)
                metrics = {"combined_score": score}
            else:
                score = metrics.get("combined_score")
                if score is None:
                    nums = [
                        float(v)
                        for v in metrics.values()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    ]
                    score = sum(nums) / len(nums) if nums else 0.0

            eval_counter[0] += 1

            # Push to monitor if callback provided
            if monitor_callback and score > 0:
                try:
                    from skydiscover.search.base_database import Program

                    prog = Program(
                        id=str(uuid.uuid4()),
                        solution=candidate,
                        language="python",
                        metrics={
                            "combined_score": float(score),
                            **(metrics if isinstance(metrics, dict) else {}),
                        },
                        iteration_found=eval_counter[0],
                        generation=eval_counter[0],
                    )
                    monitor_callback(prog, eval_counter[0])
                except Exception:
                    logger.debug("Monitor callback error", exc_info=True)

            return float(score), metrics
        except Exception as e:
            logger.warning("GEPA evaluator error: %s", e)
            return 0.0, {"error": str(e)}
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    return gepa_evaluator


# ------------------------------------------------------------------
# Config mapping
# ------------------------------------------------------------------


def _ensure_litellm_api_key(config: Config):
    """Warn early if the provider-specific env var that litellm expects is missing."""
    if not config.llm.models:
        return
    model_name = config.llm.models[0].name or ""
    provider, _, _, env_vars = _parse_model_spec(model_name)
    if provider == "openai" or not env_vars:
        return  # OPENAI_API_KEY already set via bridge_provider_env
    if not any(os.environ.get(v) for v in env_vars):
        logger.warning(
            "GEPA backend with model '%s' (provider=%s) requires one of %s. "
            "Export it before running: export %s=<your-key>",
            model_name,
            provider,
            env_vars,
            env_vars[0],
        )


def _build_gepa_config(config: Config, iterations: int):
    from gepa.optimize_anything import EngineConfig, GEPAConfig, RefinerConfig, ReflectionConfig

    # Power-user escape hatch
    ext = getattr(config, "external_config", None)
    if isinstance(ext, GEPAConfig):
        if iterations is not None:
            ext.engine.max_candidate_proposals = iterations
        return ext

    engine_kwargs: Dict[str, Any] = {"max_candidate_proposals": iterations}
    reflection_kwargs: Dict[str, Any] = {}
    refiner_kwargs: Dict[str, Any] = {}

    # GEPA supports two model roles:
    #   reflection_lm  — generates candidate mutations  (models[0])
    #   refiner_lm     — optional refinement pass       (models[1], else defaults to reflection_lm)
    if config.llm.models:
        primary = config.llm.models[0]
        if primary.name is not None:
            provider, bare_name, _, _ = _parse_model_spec(primary.name)
            reflection_kwargs["reflection_lm"] = f"{provider}/{bare_name}"

        if len(config.llm.models) >= 2:
            secondary = config.llm.models[1]
            if secondary.name is not None:
                provider, bare_name, _, _ = _parse_model_spec(secondary.name)
                refiner_kwargs["refiner_lm"] = f"{provider}/{bare_name}"

            logger.debug(
                "GEPA model mapping: reflection_lm='%s', refiner_lm='%s'",
                primary.name,
                secondary.name,
            )
            if len(config.llm.models) > 2:
                logger.warning(
                    "GEPA supports at most 2 models (reflection + refiner); "
                    "ignoring %d extra model(s)",
                    len(config.llm.models) - 2,
                )
        else:
            logger.debug(
                "GEPA model mapping: reflection_lm='%s' (also used as refiner_lm)",
                primary.name,
            )

    gepa_kwargs: Dict[str, Any] = {
        "engine": EngineConfig(**engine_kwargs),
        "reflection": ReflectionConfig(**reflection_kwargs),
    }
    if refiner_kwargs:
        gepa_kwargs["refiner"] = RefinerConfig(**refiner_kwargs)

    return GEPAConfig(**gepa_kwargs)


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
    """Run evolution using GEPA's optimize_anything API."""
    from gepa.optimize_anything import optimize_anything

    from skydiscover.api import DiscoveryResult
    from skydiscover.config import bridge_provider_env
    from skydiscover.search.base_database import Program

    bridge_provider_env(config_obj)
    _ensure_litellm_api_key(config_obj)

    with open(program_path, "r") as f:
        seed_solution = f.read()

    # Handle EVOLVE-BLOCK markers: GEPA replaces the entire candidate text,
    # so if the program has code outside the markers we need to split it and
    # only evolve the block, reconstructing the full file for evaluation.
    _START_TAG = "# EVOLVE-BLOCK-START"
    _END_TAG = "# EVOLVE-BLOCK-END"
    prefix = ""
    suffix = ""
    if _START_TAG in seed_solution and _END_TAG in seed_solution:
        start_idx = seed_solution.index(_START_TAG)
        end_idx = seed_solution.index(_END_TAG) + len(_END_TAG)
        prefix = seed_solution[:start_idx]
        suffix = seed_solution[end_idx:]
        seed_solution = seed_solution[start_idx:end_idx]

    # Build evaluator adapter
    evaluator = _make_gepa_evaluator(
        evaluator_path,
        monitor_callback=monitor_callback,
        solution_prefix=prefix,
        solution_suffix=suffix,
    )

    # Build GEPA config
    gepa_config = _build_gepa_config(config_obj, iterations)

    # Extract system prompt for domain context
    system_prompt = config_obj.system_prompt_override
    if system_prompt is None and hasattr(config_obj, "context_builder"):
        sp = config_obj.context_builder.system_message
        # Only use it if it's actual text, not a template name
        if sp and sp not in ("system_message", "evaluator_system_message"):
            system_prompt = sp

    # Human feedback: apply any pending feedback to the system prompt and set for dashboard
    if feedback_reader:
        if system_prompt:
            feedback_reader.set_current_prompt(system_prompt)
        feedback = feedback_reader.read()
        if feedback:
            if feedback_reader.mode == "replace":
                system_prompt = feedback
            else:
                system_prompt = (system_prompt or "") + "\n\n## Human Guidance\n" + feedback
            feedback_reader.set_current_prompt(system_prompt)
            logger.debug(
                f"Human feedback applied to GEPA background ({len(feedback)} chars, mode={feedback_reader.mode})"
            )
            logger.debug("Note: GEPA runs synchronously; feedback is applied once at startup.")

    # Log to file so screen output is captured
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime

    log_file = os.path.join(log_dir, f"gepa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    # Run GEPA — optimize_anything is synchronous
    try:
        result = optimize_anything(
            seed_candidate=seed_solution,
            evaluator=evaluator,
            objective="Evolve the solution to maximise the combined_score metric.",
            background=system_prompt,
            config=gepa_config,
        )
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

    # Extract results — reconstruct full file if we split on EVOLVE markers
    best_block = (
        result.best_candidate
        if isinstance(result.best_candidate, str)
        else str(result.best_candidate)
    )
    best_solution = prefix + best_block + suffix
    best_score = result.val_aggregate_scores[result.best_idx]
    scores = result.val_aggregate_scores
    initial_score = scores[0] if scores else 0.0

    best_program = Program(
        id=str(uuid.uuid4()),
        solution=best_solution,
        language=getattr(config_obj, "language", None) or "python",
        metrics={"combined_score": best_score},
        iteration_found=result.best_idx,
        generation=result.best_idx,
    )

    # Save best program and info to output dir
    import json

    best_dir = os.path.join(output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    with open(os.path.join(best_dir, "best_program.py"), "w") as f:
        f.write(best_solution)
    with open(os.path.join(best_dir, "best_program_info.json"), "w") as f:
        json.dump(
            {
                "id": best_program.id,
                "iteration": result.best_idx,
                "best_score": best_score,
                "initial_score": initial_score,
                "total_candidates": result.num_candidates,
            },
            f,
            indent=2,
        )

    return DiscoveryResult(
        best_program=best_program,
        best_score=best_score,
        best_solution=best_solution,
        metrics={"combined_score": best_score, "total_candidates": result.num_candidates},
        output_dir=output_dir,
        initial_score=initial_score,
    )
