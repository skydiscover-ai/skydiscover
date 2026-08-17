"""
AlphaEvolve external backend for SkyDiscover.

Wraps the vendored AlphaEvolve EAP SDK as a skydiscover external backend.
The ``run()`` entry point orchestrates the full experiment lifecycle:
session creation, experiment setup, initial program upload, controller loop
(sampling + evaluation workers), and best-program extraction.

Evaluation bridging converts between skydiscover's file-based evaluator
(``evaluate(path) -> dict``) and AlphaEvolve's program-dict evaluator
(``evaluate(program_dict) -> {scores, insights}``).
"""

import importlib.util
import logging
import os
import re
import sys
import tempfile
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

from skydiscover.api import DiscoveryResult
from skydiscover.config import Config
from skydiscover.search.base_database import Program

logger = logging.getLogger(__name__)

# EVOLVE-BLOCK bare marker constants
_EVOLVE_BLOCK_MARKER_START = "EVOLVE-BLOCK-START"
_EVOLVE_BLOCK_MARKER_END = "EVOLVE-BLOCK-END"

# Backward-compatible derived constants (Python-style, used by existing tests)
_EVOLVE_BLOCK_START = f"# {_EVOLVE_BLOCK_MARKER_START}"
_EVOLVE_BLOCK_END = f"# {_EVOLVE_BLOCK_MARKER_END}"

# File extensions that use C/C++ comment style
_CPP_EXTENSIONS = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"}


def _comment_prefix_for_path(path: str) -> str:
    """Return the comment prefix for the given file path.

    C/C++ extensions get ``//``, everything else gets ``#``.
    """
    ext = os.path.splitext(path)[1].lower()
    return "//" if ext in _CPP_EXTENSIONS else "#"


def _has_evolve_block_markers(content: str) -> bool:
    """Check whether *content* already contains EVOLVE-BLOCK markers.

    The check is comment-style-agnostic: it looks for the bare marker
    string ``EVOLVE-BLOCK-START`` which appears in both ``# EVOLVE-BLOCK-START``
    and ``// EVOLVE-BLOCK-START``.
    """
    return _EVOLVE_BLOCK_MARKER_START in content


# ---------------------------------------------------------------------------
# Environment variable expansion helper
# ---------------------------------------------------------------------------


def _expand_env_vars_in_value(value: str) -> str:
    """Expand ``${VAR}`` patterns in *value* using ``os.environ``.

    Uses the same ``re.sub`` pattern as ``config.py:_expand_env_vars``.
    Unmatched variables are left as-is.
    """

    def _replacer(match: re.Match) -> str:  # type: ignore[type-arg]
        return os.environ.get(match.group(1), match.group(0))

    return re.sub(r"\$\{(\w+)\}", _replacer, value)


# ---------------------------------------------------------------------------
# Credentials-file resolution
# ---------------------------------------------------------------------------


def _resolve_credentials_file(ae_config: dict) -> None:
    """Set ``GOOGLE_APPLICATION_CREDENTIALS`` from config if provided.

    When ``ae_config["credentials_file"]`` is present and non-empty:
    1. Expand ``${VAR}`` patterns in the path.
    2. Validate the expanded path exists on disk.
    3. Set the environment variable so that ``google.auth.default()``
       picks it up.

    When the key is absent or empty, this is a no-op — existing ADC
    behaviour is preserved.
    """
    cred_path = ae_config.get("credentials_file")
    if not cred_path:
        return

    cred_path = _expand_env_vars_in_value(cred_path)

    if not os.path.isfile(cred_path):
        raise ValueError(f"credentials_file not found: {cred_path}")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
    logger.info("Using credentials file: %s", cred_path)


# ---------------------------------------------------------------------------
# GCP credential validation
# ---------------------------------------------------------------------------


def _validate_gcp_credentials() -> None:
    """Fail fast with actionable error if GCP credentials are missing.

    Checks that ``google-auth`` is installed and that Application Default
    Credentials (ADC) are configured.
    """
    try:
        import google.auth  # noqa: F811
        import google.auth.exceptions

        google.auth.default()
    except ImportError:
        raise ImportError(
            "The 'alphaevolve' backend requires the google-auth package.\n"
            "Install with: pip install google-auth"
        )
    except google.auth.exceptions.DefaultCredentialsError:
        raise ValueError(
            "AlphaEvolve backend requires GCP Application Default Credentials.\n"
            "Set up credentials with one of:\n"
            "  1. gcloud auth application-default login\n"
            "  2. Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json\n"
            "  3. Run on a GCP VM/Cloud Shell with default service account"
        )


# ---------------------------------------------------------------------------
# Initial program construction
# ---------------------------------------------------------------------------


def _build_initial_program(program_path: str) -> dict:
    """Build an AlphaEvolve initial-program payload from a local file.

    * Raises ``ValueError`` with a clear message when the file is
      missing or empty.
    * Auto-wraps content with EVOLVE-BLOCK markers when absent.
    * Passes through existing markers as-is.

    Returns the dict structure expected by
    ``AlphaEvolveExperiment.create_initial_program``.
    """
    if not program_path:
        raise ValueError(
            "Initial program file is required for the AlphaEvolve backend but was not provided."
        )
    if not os.path.exists(program_path):
        raise ValueError(f"Initial program file not found at path: {program_path}")

    with open(program_path, "r") as fh:
        content = fh.read()

    if not content.strip():
        raise ValueError(f"Initial program file is empty: {program_path}")

    if not _has_evolve_block_markers(content):
        prefix = _comment_prefix_for_path(program_path)
        start = f"{prefix} {_EVOLVE_BLOCK_MARKER_START}"
        end = f"{prefix} {_EVOLVE_BLOCK_MARKER_END}"
        content = f"{start}\n{content}\n{end}"

    return {
        "content": {
            "files": [
                {
                    "path": os.path.basename(program_path),
                    "content": content,
                }
            ]
        },
        "evaluation": {"scores": {"scores": [{"metric": "combined_score", "score": 0.0}]}},
    }


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _get_alphaevolve_config(config_obj: Config) -> dict:
    """Resolve the AlphaEvolve config section.

    Precedence (lowest to highest): ``alphaevolve_default.yaml`` <
    the ``alphaevolve:`` section of the user config < environment
    variables (``ALPHAEVOLVE_PROJECT_ID``, ``ALPHAEVOLVE_ENGINE_ID``, ...).

    Raises ``ValueError`` when required fields (``project_id``,
    ``engine_id``) are still missing after merging.
    """
    from skydiscover.extras.external.defaults import load_defaults

    raw = load_defaults("alphaevolve_default.yaml")
    ae: dict = raw.get("alphaevolve", {}) if raw else {}

    # User config (-c config.yaml) overrides the shipped defaults.
    obj_ae = getattr(config_obj, "alphaevolve", None)
    if isinstance(obj_ae, dict):
        ae.update(obj_ae)

    _env_map = {
        "ALPHAEVOLVE_PROJECT_ID": "project_id",
        "ALPHAEVOLVE_ENGINE_ID": "engine_id",
        "ALPHAEVOLVE_LOCATION": "location",
        "ALPHAEVOLVE_COLLECTION": "collection",
        "ALPHAEVOLVE_ASSISTANT": "assistant",
        "ALPHAEVOLVE_NUM_EVALUATORS": "num_evaluators",
        "ALPHAEVOLVE_NUM_SAMPLERS": "num_samplers",
        "ALPHAEVOLVE_CREDENTIALS_FILE": "credentials_file",
    }
    _int_keys = {"num_evaluators", "num_samplers"}

    for env_var, config_key in _env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            ae[config_key] = int(val) if config_key in _int_keys else val

    logger.debug("AlphaEvolve config resolved: %s", ae)

    # Validate required fields
    if not ae.get("project_id") or not ae.get("engine_id"):
        raise ValueError(
            "AlphaEvolve backend requires 'project_id' and 'engine_id'.\n"
            "Provide them via environment variables:\n"
            "  export ALPHAEVOLVE_PROJECT_ID=your-gcp-project\n"
            "  export ALPHAEVOLVE_ENGINE_ID=your-engine-id\n"
            "Or via the alphaevolve: section in your config YAML."
        )

    return ae


# ---------------------------------------------------------------------------
# Score bridging
# ---------------------------------------------------------------------------


def _metrics_to_ae_scores(metrics: dict) -> dict:
    """Convert a skydiscover metrics dict to AlphaEvolve score format.

    Returns the **pre-wrapped** format so that
    ``AlphaEvolveExperiment.evaluator()`` detects the ``scores`` key and
    passes through without legacy wrapping.
    """
    scores_list = [
        {"metric": k, "score": float(v)}
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return {"scores": {"scores": scores_list}, "insights": {}}


def _ae_scores_to_metrics(evaluation: dict) -> dict:
    """Convert an AlphaEvolve evaluation dict to a skydiscover metrics dict.

    If no ``combined_score`` key is present but other scores exist, the
    first score value is used as ``combined_score``.
    """
    if not evaluation:
        return {}
    scores = (evaluation.get("scores") or {}).get("scores") or []
    metrics: Dict[str, float] = {}
    for s in scores:
        metric = s.get("metric", "score")
        score = s.get("score", 0.0)
        metrics[metric] = float(score) if score is not None else 0.0

    if "combined_score" not in metrics and metrics:
        metrics["combined_score"] = list(metrics.values())[0]

    return metrics


# ---------------------------------------------------------------------------
# Evaluator adapter
# ---------------------------------------------------------------------------


def _make_alphaevolve_evaluator(
    evaluator_path: str,
    monitor_callback: Optional[Callable] = None,
    file_suffix: str = ".py",
    language: str = "python",
) -> Callable[[dict], dict]:
    """Wrap a skydiscover file-based evaluator for the AlphaEvolve SDK.

    The returned closure accepts an AlphaEvolve program dict
    (``{content: {files: [...]}, ...}``) and returns an evaluation dict
    in the pre-wrapped ``{scores: {scores: [...]}, insights: {}}`` format.

    ``file_suffix`` is the extension used for the temporary candidate file
    handed to the user's ``evaluate(path)``. It matters for non-Python runs,
    where evaluators dispatch on the extension.
    """
    spec = importlib.util.spec_from_file_location("_skydiscover_eval", evaluator_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load evaluator spec from {evaluator_path}")
    eval_module = importlib.util.module_from_spec(spec)

    eval_dir = os.path.dirname(os.path.abspath(evaluator_path))
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    spec.loader.exec_module(eval_module)
    user_evaluate = eval_module.evaluate

    eval_counter = [0]

    def ae_evaluator(program: dict) -> dict:
        files = program.get("content", {}).get("files", [])
        if not files:
            return {
                "scores": {"scores": []},
                "insights": {"error": "No files in candidate"},
            }

        if len(files) > 1:
            logger.warning(
                "AlphaEvolve returned %d files; only single-file is "
                "supported, using files[0] only.",
                len(files),
            )

        code = files[0].get("content", "")

        # Prefer the extension AlphaEvolve used for the candidate file, so a
        # C++/CUDA candidate reaches the evaluator as e.g. ".cu" and not ".py".
        suffix = os.path.splitext(files[0].get("path", ""))[1] or file_suffix

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=suffix,
                prefix="ae_candidate_",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp_path = tmp.name
                tmp.write(code)

            metrics = user_evaluate(tmp_path)

            # Normalize metrics to dict
            if metrics is None:
                metrics = {}
            elif isinstance(metrics, (int, float)):
                metrics = {"combined_score": float(metrics)}
            elif not isinstance(metrics, dict):
                try:
                    metrics = dict(metrics)
                except (TypeError, ValueError):
                    metrics = {}

            eval_counter[0] += 1

            # Push to monitor if callback provided
            if monitor_callback:
                try:
                    prog = Program(
                        id=str(uuid.uuid4()),
                        solution=code,
                        language=language,
                        metrics=dict(metrics),
                        iteration_found=eval_counter[0],
                        generation=eval_counter[0],
                    )
                    monitor_callback(prog, eval_counter[0])
                except Exception:
                    logger.debug("Monitor callback error", exc_info=True)

            # Bridge to AlphaEvolve score format (pre-wrapped)
            return _metrics_to_ae_scores(metrics)

        except Exception as e:
            logger.warning("AlphaEvolve evaluator error: %s", e)
            return {
                "scores": {"scores": []},
                "insights": {"error": str(e)},
            }
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    return ae_evaluator


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------


def _build_experiment_config(config_obj: Config, ae_config: dict, iterations: int) -> dict:
    """Build an AlphaEvolve experiment config from skydiscover Config.

    Auto-derives sensible defaults from skydiscover's config and allows
    overrides from the ``alphaevolve:`` YAML section.
    """
    exp_config: Dict[str, Any] = {
        "title": ae_config.get("title", "SkyDiscover Evolution"),
        "problem_description": ae_config.get(
            "problem_description",
            getattr(config_obj, "system_prompt_override", None)
            or "Evolve the program to maximize the evaluation score.",
        ),
        "run_settings": {
            "max_programs": ae_config.get("max_programs", iterations),
            "concurrency": ae_config.get("concurrency", ae_config.get("num_evaluators", 1)),
        },
    }

    # Tell AlphaEvolve which language it is evolving (defaults to skydiscover's
    # `language:` setting) so generated candidates are in the right language.
    language = getattr(config_obj, "language", None)
    if language:
        exp_config["program_language"] = language

    # Allow override of optional keys from ae_config
    for key in ("program_language", "generation_settings"):
        if key in ae_config:
            exp_config[key] = ae_config[key]

    return exp_config


# ---------------------------------------------------------------------------
# Best-program extraction
# ---------------------------------------------------------------------------


def _extract_best_program(experiment: Any, metric_name: str = "combined_score") -> Tuple[str, dict]:
    """Extract the best program and its metrics from a completed experiment.

    Asks the API for programs sorted by score descending, then picks the
    best one on *metric_name* locally. The AlphaEvolve examples do the same,
    because the server-side ordering is not guaranteed to match the metric
    this run optimises.
    """
    response = experiment.list_programs(params={"order_by": "score desc"})
    if not response or "alphaEvolvePrograms" not in response:
        return ("", {})

    programs = response["alphaEvolvePrograms"]
    if not programs:
        return ("", {})

    scored = [(_ae_scores_to_metrics(p.get("evaluation", {})), p) for p in programs]
    metrics, best = max(scored, key=lambda pair: pair[0].get(metric_name, float("-inf")))

    files = best.get("content", {}).get("files", [])
    code = files[0].get("content", "") if files else ""

    return (code, metrics)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run(
    program_path: str,
    evaluator_path: str,
    config_obj: Config,
    iterations: int,
    output_dir: str,
    monitor_callback: Optional[Callable] = None,
    feedback_reader: Optional[Any] = None,
) -> DiscoveryResult:
    """Run evolution using AlphaEvolve's cloud API.

    This is the external backend entry point called by ``api.py`` when
    ``search_type == "alphaevolve"``.  It orchestrates the full experiment
    lifecycle: GCP auth validation, initial program upload, experiment
    creation, controller loop (sampling + evaluation workers), and
    best-program extraction.

    Calls the public AlphaEvolve SDK controller with
    minimal wrapper code for config mapping and result extraction.
    """
    # Lazy imports of vendored SDK (per external backend pattern)
    from alpha_evolve.client import AlphaEvolveClient
    from alpha_evolve.controller import run_controller_loop
    from alpha_evolve.experiment import AlphaEvolveExperiment

    # 1. Read alphaevolve config section — resolved first so
    #    credentials_file can set GOOGLE_APPLICATION_CREDENTIALS before
    #    the ADC check.
    ae_config = _get_alphaevolve_config(config_obj)
    _resolve_credentials_file(ae_config)

    # 2. Validate prerequisites
    _validate_gcp_credentials()
    file_suffix = getattr(config_obj, "file_suffix", ".py") or ".py"
    language = getattr(config_obj, "language", None) or "python"
    initial_program = _build_initial_program(program_path)

    # 3. Build client
    client = AlphaEvolveClient(
        project_id=ae_config["project_id"],
        engine=ae_config["engine_id"],
        location=ae_config.get("location", "global"),
        collection=ae_config.get("collection", "default_collection"),
        assistant=ae_config.get("assistant", "default_assistant"),
    )

    # 4. Build evaluator adapter
    evaluator = _make_alphaevolve_evaluator(evaluator_path, monitor_callback, file_suffix, language)

    # 5. Create experiment
    num_evaluators = ae_config.get("num_evaluators", 1)
    experiment = AlphaEvolveExperiment(
        client,
        evaluator,
        max_programs_evaluated=iterations,
        parallel_evaluation=(num_evaluators > 1),
    )
    exp_config = _build_experiment_config(config_obj, ae_config, iterations)
    experiment.create_experiment(exp_config)
    experiment.create_initial_program(initial_program)
    experiment.start_experiment()

    # 6. Run controller loop
    # Directly await -- do NOT use asyncio.run()
    await run_controller_loop(
        experiment,
        num_samplers=ae_config.get("num_samplers", 1),
        num_evaluators=num_evaluators,
        idle_timeout_s=ae_config.get("idle_timeout_s", 120),
    )

    # 7. Post-loop submission gap check
    stats = experiment.stats
    generated = stats.get("num_programs_generated", 0)
    evaluated = stats.get("num_programs_evaluated", 0)
    if generated > 0 and evaluated < generated:
        failed = generated - evaluated
        failure_rate = failed / generated
        if failure_rate >= 0.25 and generated >= 4:
            logger.warning(
                "HIGH SUBMISSION FAILURE RATE: %d of %d generated programs "
                "(%.0f%%) were never evaluated successfully. Common causes: "
                "evaluator too slow (lock-token expiry), auth errors, or API "
                "rate limits.",
                failed,
                generated,
                failure_rate * 100,
            )

    # 8. Extract best program
    best_code, metrics = _extract_best_program(experiment)
    best_score = metrics.get("combined_score", 0.0)

    # 9. Build and return DiscoveryResult
    best_program = Program(
        id=str(uuid.uuid4()),
        solution=best_code,
        language=language,
        metrics=metrics,
        iteration_found=0,
    )

    return DiscoveryResult(
        best_program=best_program,
        best_score=best_score,
        best_solution=best_code,
        metrics=metrics,
        output_dir=output_dir,
    )
