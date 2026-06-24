"""
Thin wrapper around OpenEvolve (https://github.com/codelion/openevolve).

Delegates entirely to OpenEvolve's public API so upstream updates are
picked up automatically.
"""

import asyncio
import logging
import os
from typing import Optional

from skydiscover.api import DiscoveryResult
from skydiscover.config import Config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config mapping
# ------------------------------------------------------------------


def _map_config(config: Config, iterations: Optional[int], output_dir: str):
    """Convert SkyDiscover Config to an OpenEvolve Config, mapping shared fields."""
    from openevolve.config import Config as OEConfig

    # Power-user escape hatch: if they attached an OpenEvolve config, use it.
    ext = getattr(config, "external_config", None)
    if isinstance(ext, OEConfig):
        if iterations is not None:
            ext.max_iterations = iterations
        return ext

    oe = OEConfig()

    # Apply tuned backend defaults (population_size, prompt, evaluator, etc.)
    from skydiscover.extras.external.defaults import apply_defaults, load_defaults

    apply_defaults(oe, load_defaults("openevolve_default.yaml"))

    # CLI overrides
    oe.max_iterations = iterations or config.max_iterations

    # LLM models (from --model / -c config)
    if config.llm.models:
        from openevolve.config import LLMModelConfig as OEModel

        # Resolve the correct api_base for OpenEvolve models.
        # The openai library ignores OPENAI_BASE_URL when base_url is
        # explicitly passed, so we must resolve the env var here and pass it
        # through.  Priority: env var > SkyDiscover config value.
        resolved_api_base = (
            os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or getattr(config.llm, "api_base", None)
            or "https://api.openai.com/v1"
        )
        oe.llm.api_base = resolved_api_base
        oe.llm.models = [
            OEModel(
                name=m.name,
                weight=getattr(m, "weight", 1.0),
                temperature=getattr(m, "temperature", oe.llm.temperature),
                api_key=getattr(m, "api_key", None),
                api_base=resolved_api_base,
            )
            for m in config.llm.models
        ]
        # Sync evaluator models (OE's __post_init__ ran before we set models)
        oe.llm.evaluator_models = oe.llm.models.copy()
        # Propagate OE shared config (max_tokens, timeout, etc.) to new models
        oe.llm.update_model_params(
            {
                "max_tokens": oe.llm.max_tokens,
                "timeout": oe.llm.timeout,
                "retries": oe.llm.retries,
                "retry_delay": oe.llm.retry_delay,
                "top_p": oe.llm.top_p,
                "reasoning_effort": oe.llm.reasoning_effort,
            }
        )

    # LLM timeout
    if config.llm.timeout:
        oe.llm.timeout = config.llm.timeout

    # System prompt — propagate to OpenEvolve's prompt config and models
    sys_prompt = config.system_prompt_override
    if sys_prompt is None and hasattr(config, "context_builder"):
        sp = config.context_builder.system_message
        if sp and sp not in ("system_message", "evaluator_system_message"):
            sys_prompt = sp
    if sys_prompt:
        if hasattr(oe, "prompt"):
            oe.prompt.system_message = sys_prompt
        for m in oe.llm.models:
            m.system_message = sys_prompt

    # Evaluator settings
    if hasattr(config, "evaluator"):
        if hasattr(config.evaluator, "timeout") and config.evaluator.timeout:
            oe.evaluator.timeout = config.evaluator.timeout
        if hasattr(config.evaluator, "max_retries") and config.evaluator.max_retries:
            oe.evaluator.max_retries = config.evaluator.max_retries
        if hasattr(config.evaluator, "cascade_evaluation"):
            oe.evaluator.cascade_evaluation = config.evaluator.cascade_evaluation

    oe.diff_based_generation = config.diff_based_generation

    return oe


# ------------------------------------------------------------------
# Initial score extraction
# ------------------------------------------------------------------


def _get_initial_score(programs) -> float:
    """Extract initial (iteration 0) score from OpenEvolve programs dict."""
    initial_score = 0.0
    for p in programs.values():
        it = getattr(p, "iteration_found", None)
        if it == 0:
            score = _score_of(p.metrics)
            if score is not None:
                initial_score = max(initial_score, score)
    return initial_score


def _score_of(metrics: dict) -> Optional[float]:
    if not metrics:
        return None
    if "combined_score" in metrics:
        return float(metrics["combined_score"])
    nums = [
        float(v)
        for v in metrics.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return sum(nums) / len(nums) if nums else None


# ------------------------------------------------------------------
# Program conversion
# ------------------------------------------------------------------


def _to_skydiscover_program(oe_prog):
    from skydiscover.search.base_database import Program

    return Program(
        id=oe_prog.id,
        solution=oe_prog.code,
        language=getattr(oe_prog, "language", "python"),
        metrics=oe_prog.metrics or {},
        iteration_found=getattr(oe_prog, "iteration_found", 0),
        parent_id=getattr(oe_prog, "parent_id", None),
        generation=getattr(oe_prog, "generation", 0),
        timestamp=getattr(oe_prog, "timestamp", 0.0),
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
    """Run evolution using the OpenEvolve package."""
    from openevolve.controller import OpenEvolve

    from skydiscover.api import DiscoveryResult
    from skydiscover.config import bridge_provider_env

    bridge_provider_env(config_obj)

    oe_config = _map_config(config_obj, iterations, output_dir)

    # Human feedback: set initial system prompt on feedback reader for dashboard visibility
    original_sys_prompt = ""
    if hasattr(oe_config, "prompt") and hasattr(oe_config.prompt, "system_message"):
        original_sys_prompt = oe_config.prompt.system_message or ""
    if feedback_reader and original_sys_prompt:
        feedback_reader.set_current_prompt(original_sys_prompt)

    controller = OpenEvolve(
        initial_program_path=program_path,
        evaluation_file=evaluator_path,
        config=oe_config,
        output_dir=output_dir,
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
                        db = getattr(controller, "database", None)
                        if db is None:
                            continue
                        for pid, p in list(db.programs.items()):
                            if pid not in seen_ids:
                                seen_ids.add(pid)
                                sky_prog = _to_skydiscover_program(p)
                                monitor_callback(sky_prog, getattr(p, "iteration_found", 0))
                    except Exception:
                        logger.debug("Monitor poll error", exc_info=True)
                # Human feedback: inject feedback into OpenEvolve's config
                if feedback_reader:
                    try:
                        feedback = feedback_reader.read()
                        if feedback != _last_feedback:
                            _last_feedback = feedback
                            if feedback:
                                if feedback_reader.mode == "replace":
                                    new_prompt = feedback
                                else:
                                    new_prompt = (
                                        original_sys_prompt + "\n\n## Human Guidance\n" + feedback
                                    )
                            else:
                                new_prompt = original_sys_prompt
                            # Update OpenEvolve's prompt config and model configs
                            if hasattr(oe_config, "prompt"):
                                oe_config.prompt.system_message = new_prompt
                            for m in getattr(oe_config.llm, "models", []):
                                if hasattr(m, "system_message"):
                                    m.system_message = new_prompt
                            feedback_reader.set_current_prompt(new_prompt)
                            if feedback:
                                logger.debug(
                                    f"Human feedback injected into OpenEvolve ({len(feedback)} chars, mode={feedback_reader.mode})"
                                )
                    except Exception:
                        logger.debug("Human feedback injection error", exc_info=True)

        poll_task = asyncio.create_task(_poll_programs())

    best = await controller.run(iterations=iterations)

    if poll_task:
        poll_task.cancel()
        # Flush remaining programs
        db = getattr(controller, "database", None)
        if db:
            for pid, p in db.programs.items():
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    try:
                        monitor_callback(
                            _to_skydiscover_program(p), getattr(p, "iteration_found", 0)
                        )
                    except Exception:
                        logger.debug("Monitor flush error", exc_info=True)

    # Extract results from the OpenEvolve database
    programs = getattr(controller, "database", None)
    programs_dict = programs.programs if programs else {}

    initial_score = _get_initial_score(programs_dict)

    best_skydiscover = _to_skydiscover_program(best) if best else None
    best_score = _score_of(best.metrics) if best else 0.0

    return DiscoveryResult(
        best_program=best_skydiscover,
        best_score=best_score or 0.0,
        best_solution=best.code if best else "",
        metrics=(best.metrics or {}) if best else {},
        output_dir=output_dir,
        initial_score=initial_score,
    )
