"""
AdaEvolve context builder for SkyDiscover.

Extends DefaultContextBuilder with AdaEvolve-specific prompt sections:
- Evaluator feedback from parent artifacts
- Paradigm breakthrough guidance
- Sibling context (previous mutations of the same parent)
- Error retry context

These are assembled into a ``search_guidance`` string and injected into
AdaEvolve-specific templates via the ``{search_guidance}`` placeholder.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from skydiscover.config import Config
from skydiscover.context_builder.default import DefaultContextBuilder
from skydiscover.context_builder.utils import TemplateManager, prog_attr
from skydiscover.search.base_database import Program
from skydiscover.utils.metrics import compute_proxy_score, get_score

logger = logging.getLogger(__name__)


class AdaEvolveContextBuilder(DefaultContextBuilder):
    """
    Context builder for AdaEvolve's adaptive evolutionary search.

    Adds a ``{search_guidance}`` section to the prompt containing:
    - Evaluator diagnostic feedback (from parent's artifacts)
    - Paradigm breakthrough guidance (when search is globally stagnating)
    - Sibling context (previous mutations of the same parent)
    - Error retry context (when retrying after a failed generation)

    The controller passes raw data via the ``context`` dict:
    - ``context["paradigm"]``: paradigm dict or None
    - ``context["siblings"]``: list of Program objects
    - ``context["error_context"]``: error string or None

    Evaluator feedback is extracted from the parent program's artifacts.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        default_templates = str(Path(__file__).parent.parent / "default" / "templates")
        adaevolve_templates = str(Path(__file__).parent / "templates")
        self.template_manager = TemplateManager(
            default_templates, adaevolve_templates, self.context_config.template_dir
        )

    def _db_config(self) -> Any:
        return getattr(self.config.search, "database", None)

    def _is_multiobjective_enabled(self) -> bool:
        return bool(getattr(self._db_config(), "pareto_objectives", None) or [])

    def _objective_descriptions(self) -> List[str]:
        db_config = self._db_config()
        higher_is_better = getattr(db_config, "higher_is_better", None) or {}
        descriptions = []
        for objective in getattr(db_config, "pareto_objectives", None) or []:
            direction = "maximize" if higher_is_better.get(objective, True) else "minimize"
            descriptions.append(f"{objective} ({direction})")
        return descriptions

    def _metric_to_maximization_value(self, metric_name: str, value: Any) -> Optional[float]:
        from skydiscover.utils.metrics import normalize_metric_value

        higher_is_better = getattr(self._db_config(), "higher_is_better", None) or {}
        return normalize_metric_value(metric_name, value, higher_is_better)

    _PROGRESS_SCORE_MISSING = float("-inf")

    def _get_progress_score(self, metrics: Dict[str, Any]) -> float:
        """Scalar proxy used only for prompt-side progress descriptions.

        Returns ``_PROGRESS_SCORE_MISSING`` (``-inf``) for empty/missing metrics
        so that callers can distinguish "no data" from "score is zero".
        """
        db_config = self._db_config()
        pareto_objectives = getattr(db_config, "pareto_objectives", None) or None
        return compute_proxy_score(
            metrics,
            fitness_key=getattr(db_config, "fitness_key", None),
            pareto_objectives=pareto_objectives,
            higher_is_better=getattr(db_config, "higher_is_better", None) or {},
        )

    def _task_objective_text(self) -> str:
        subject = (
            "prompt" if (self.config.language or "").lower() in ("text", "prompt") else "program"
        )
        if not self._is_multiobjective_enabled():
            return f"Suggest improvements to the {subject} that will improve its COMBINED_SCORE."
        return (
            f"Suggest improvements to the {subject} that improve its Pareto trade-offs across: "
            + ", ".join(self._objective_descriptions())
            + "."
        )

    def _diversity_dimensions_text(self) -> str:
        if not self._is_multiobjective_enabled():
            return "The system maintains diversity across these dimensions: score, complexity."
        return "The system maintains diversity across Pareto trade-offs, novelty, and solution structure."

    def _diversity_note_text(self) -> str:
        if not self._is_multiobjective_enabled():
            return "Different solutions with similar combined_score but different features are valuable."
        return "Different solutions with similar overall trade-offs but different objective balances are valuable."

    def build_prompt(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Build prompt with AdaEvolve-specific search guidance.

        Computes the ``search_guidance`` string from AdaEvolve context keys,
        then delegates to the parent's ``build_prompt`` which fills the
        ``{search_guidance}`` placeholder in AdaEvolve templates.
        """
        context = context or {}

        # Build the search guidance from AdaEvolve-specific context
        search_guidance = self._build_search_guidance(current_program, context)

        # Override any caller-supplied search_guidance with our computed one
        kwargs.pop("search_guidance", None)

        # Pass search_guidance through **kwargs to template.format()
        result = super().build_prompt(
            current_program,
            context,
            search_guidance=search_guidance,
            task_objective=self._task_objective_text(),
            diversity_dimensions=self._diversity_dimensions_text(),
            diversity_note=self._diversity_note_text(),
            **kwargs,
        )

        return result

    # =========================================================================
    # Suppress default artifact feedback rendering
    # =========================================================================

    def _format_current_program(
        self,
        current_program: Union[Program, Dict[str, Program]],
        language: str,
    ) -> str:
        """Override to suppress artifacts["feedback"] from {current_program}.

        AdaEvolve renders evaluator feedback explicitly via _build_search_guidance
        into {search_guidance}, so we strip it here to avoid duplication.
        """
        # Remove feedback from artifacts so parent renderer skips it (rendered via search_guidance instead)
        if isinstance(current_program, dict):
            program = list(current_program.values())[0]
        else:
            program = current_program

        artifacts = getattr(program, "artifacts", None)
        saved_feedback = None
        if isinstance(artifacts, dict) and "feedback" in artifacts:
            saved_feedback = artifacts.pop("feedback")

        try:
            return super()._format_current_program(current_program, language)
        finally:
            if saved_feedback is not None and isinstance(artifacts, dict):
                artifacts["feedback"] = saved_feedback

    # =========================================================================
    # Search Guidance Assembly
    # =========================================================================

    def _build_search_guidance(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context: Dict[str, Any],
    ) -> str:
        """
        Assemble all AdaEvolve-specific guidance sections into one string.

        Sections are included in priority order:
        1. Evaluator feedback (highest value — shows why parent fails)
        2. Paradigm breakthrough guidance (when globally stagnating)
        3. Sibling context (previous mutations of this parent)
        4. Error retry context (when retrying after failure)
        """
        # Extract parent program from current_program dict
        if isinstance(current_program, dict):
            parent_program = list(current_program.values())[0]
        else:
            parent_program = current_program

        language = self.config.language or "python"
        paradigm = context.get("paradigm")
        siblings = context.get("siblings", [])
        error_context = context.get("error_context")

        sections: List[str] = []

        # 1. Evaluator feedback from parent artifacts
        feedback_section = self._format_evaluator_feedback(parent_program)
        if feedback_section:
            sections.append(feedback_section)

        # 2. Paradigm breakthrough guidance
        if paradigm:
            sections.append(self._format_paradigm_guidance(paradigm, language))

        # 3. Sibling context
        if siblings:
            sibling_section = self._format_sibling_context(siblings, parent_program)
            if sibling_section:
                sections.append(sibling_section)

        # 4. Error retry context
        if error_context:
            sections.append(self._format_error_context(error_context))

        if not sections:
            return ""

        return "\n\n".join(sections)

    def _identify_improvement_areas(
        self,
        current_program: str,
        metrics: Dict[str, float],
        previous_programs: List[Program],
    ) -> str:
        """Generate improvement bullets for scalar or Pareto mode."""
        if not self._is_multiobjective_enabled():
            return super()._identify_improvement_areas(current_program, metrics, previous_programs)

        improvement_areas = [
            "Focus on Pareto trade-offs across: " + ", ".join(self._objective_descriptions())
        ]

        current_score = self._get_progress_score(metrics)
        if previous_programs:
            prev_metrics = prog_attr(previous_programs[-1], "metrics", {}) or {}
            prev_score = self._get_progress_score(prev_metrics)
            # Only show delta text when both scores are valid (not missing).
            missing = self._PROGRESS_SCORE_MISSING
            if current_score != missing and prev_score != missing:
                if current_score > prev_score + 1e-6:
                    improvement_areas.append(
                        f"Pareto proxy improved: {prev_score:.4f} -> {current_score:.4f}"
                    )
                elif current_score < prev_score - 1e-6:
                    improvement_areas.append(
                        f"Pareto proxy declined: {prev_score:.4f} -> {current_score:.4f}. Revisit recent trade-offs."
                    )
                else:
                    improvement_areas.append(f"Pareto proxy unchanged at {current_score:.4f}")
            elif current_score != missing:
                improvement_areas.append(f"Pareto proxy at {current_score:.4f} (first measurement)")

        threshold = self.context_config.suggest_simplification_after_chars
        if threshold and len(current_program) > threshold:
            improvement_areas.append(
                f"Consider simplifying - solution length exceeds {threshold} characters"
            )

        return "\n".join(f"- {area}" for area in improvement_areas)

    # =========================================================================
    # Section Formatters
    # =========================================================================

    @staticmethod
    def _format_evaluator_feedback(parent_program: Program) -> Optional[str]:
        """
        Format evaluator feedback from parent's artifacts.

        The evaluator may return diagnostic feedback (e.g. analysis of failed
        examples) in artifacts["feedback"]. This is injected into the prompt
        so the LLM can make targeted improvements instead of guessing.
        """
        artifacts = getattr(parent_program, "artifacts", None)
        if not artifacts:
            return None

        feedback = artifacts.get("feedback")
        if not feedback or not isinstance(feedback, str):
            return None

        # Truncate very long feedback to keep prompt focused
        max_len = 2000
        if len(feedback) > max_len:
            feedback = feedback[:max_len] + "\n... (truncated)"

        return (
            "## EVALUATOR FEEDBACK ON CURRENT PROGRAM\n"
            "The evaluator analyzed cases where the current program failed "
            "and produced the following diagnostic feedback. "
            "Use this to make targeted improvements:\n\n"
            f"{feedback}"
        )

    @staticmethod
    def _format_paradigm_guidance(paradigm: Dict[str, Any], language: str) -> str:
        """
        Format paradigm breakthrough guidance for the LLM.

        Uses different framing for prompt optimization vs code optimization.
        """
        is_prompt_opt = (language or "").lower() in ("text", "prompt")

        idea = paradigm.get("idea", "N/A")
        description = paradigm.get("description", "N/A")
        target = paradigm.get("what_to_optimize", "score")
        cautions = paradigm.get("cautions", "N/A")
        approach_type = paradigm.get("approach_type", "N/A")

        if is_prompt_opt:
            header = "## BREAKTHROUGH STRATEGY - APPLY THIS"
            intro = "The search has stagnated globally. You MUST apply this breakthrough prompt strategy:"
            fields = (
                f"**STRATEGY:** {idea}\n\n"
                f"**HOW TO APPLY:**\n{description}\n\n"
                f"**TARGET:** {target}\n\n"
                f"**CAUTIONS:** {cautions}\n\n"
                f"**APPROACH TYPE:** {approach_type}"
            )
            critical_bullets = (
                "- You MUST rewrite the prompt using this strategy\n"
                "- The strategy must be reflected in the actual prompt structure and content\n"
                "- Keep the prompt clear and well-structured\n"
                "- Do not add unnecessary verbosity — every sentence should serve a purpose\n"
                "- Ensure the prompt still addresses the core task"
            )
        else:
            header = "## BREAKTHROUGH IDEA - IMPLEMENT THIS"
            intro = "The search has stagnated globally. You MUST implement this breakthrough idea:"
            fields = (
                f"**IDEA:** {idea}\n\n"
                f"**HOW TO IMPLEMENT:**\n{description}\n\n"
                f"**TARGET METRIC:** {target}\n\n"
                f"**CAUTIONS:** {cautions}\n\n"
                f"**APPROACH TYPE:** {approach_type}"
            )
            critical_bullets = (
                "- You MUST implement the breakthrough idea\n"
                "- Ensure the paradigm is actually used in your implementation (not just mentioned in comments)\n"
                "- Correctness is essential - your implementation must be correct and functional\n"
                "- Verify output format matches evaluator requirements\n"
                "- Make purposeful changes that implement the idea\n"
                "- Test your implementation logic carefully"
            )

        return f"{header}\n\n{intro}\n\n{fields}\n\n**CRITICAL:**\n{critical_bullets}"

    def _format_sibling_context(
        self, siblings: List[Program], parent_program: Program
    ) -> Optional[str]:
        """
        Format sibling context showing previous mutations of the parent.

        Shows what mutations have been tried before, whether they improved
        or regressed, so the LLM can avoid repeating failed approaches.
        """
        if not siblings:
            return None

        parent_fitness = self._get_progress_score(getattr(parent_program, "metrics", {}))
        missing = self._PROGRESS_SCORE_MISSING

        improved, regressed, unchanged = 0, 0, 0
        entries: List[str] = []

        for i, child in enumerate(siblings, 1):
            child_fitness = self._get_progress_score(getattr(child, "metrics", {}))

            if parent_fitness == missing or child_fitness == missing:
                entries.append(f"  {i}. (metrics unavailable) [UNKNOWN]")
                unchanged += 1
                continue

            delta = child_fitness - parent_fitness

            if delta > 0.001:
                status = "IMPROVED"
                improved += 1
            elif delta < -0.001:
                status = "REGRESSED"
                regressed += 1
            else:
                status = "NO CHANGE"
                unchanged += 1

            entries.append(
                f"  {i}. {parent_fitness:.4f} -> {child_fitness:.4f} " f"({delta:+.4f}) [{status}]"
            )

        lines = [
            "## PREVIOUS ATTEMPTS ON THIS PARENT",
            f"Summary: {improved} improved, {unchanged} unchanged, {regressed} regressed",
            *entries,
            "Avoid repeating approaches that didn't work.",
        ]
        return "\n".join(lines)

    def _format_previous_attempts(
        self, previous_programs: List[Program], num_previous_attempts: int = 3
    ) -> str:
        """Format recent attempts using AdaEvolve's scalar proxy in Pareto mode."""
        if not self._is_multiobjective_enabled():
            return super()._format_previous_attempts(previous_programs, num_previous_attempts)

        if not previous_programs:
            return "No previous attempts yet."

        try:
            previous_attempt_template = self.template_manager.get_template("previous_attempt")
        except (ValueError, KeyError):
            previous_attempt_template = "### Attempt {attempt_number}\n- Changes: {changes}\n- Metrics: {performance}\n- Outcome: {outcome}"

        previous_programs = sorted(
            previous_programs,
            key=lambda program: self._get_progress_score(prog_attr(program, "metrics", {}) or {}),
            reverse=True,
        )
        selected = previous_programs[: min(num_previous_attempts, len(previous_programs))]

        lines = []
        for i, program in enumerate(reversed(selected)):
            attempt_number = len(selected) - i
            metadata = prog_attr(program, "metadata", {}) or {}
            metrics = prog_attr(program, "metrics", {}) or {}

            changes = metadata.get("changes", "Unknown changes")
            performance_parts = []
            for name, value in metrics.items():
                if not isinstance(value, bool) and isinstance(value, (int, float)):
                    try:
                        performance_parts.append(f"{name}: {value:.4f}")
                    except (ValueError, TypeError):
                        performance_parts.append(f"{name}: {value}")
                else:
                    performance_parts.append(f"{name}: {value}")
            performance_str = ", ".join(performance_parts) if performance_parts else "No metrics"

            parent_metrics = metadata.get("parent_metrics", {})
            outcome = self._determine_outcome(metrics, parent_metrics)

            lines.append(
                previous_attempt_template.format(
                    attempt_number=attempt_number,
                    changes=changes,
                    performance=performance_str,
                    outcome=outcome,
                )
                + "\n\n"
            )

        return "".join(lines)

    def _determine_outcome(
        self, program_metrics: Dict[str, Any], parent_metrics: Dict[str, Any]
    ) -> str:
        """Describe attempt outcome using the configured scalar proxy in Pareto mode."""
        if not self._is_multiobjective_enabled():
            return super()._determine_outcome(program_metrics, parent_metrics)

        prog_value = self._get_progress_score(program_metrics)
        parent_value = self._get_progress_score(parent_metrics)
        missing = self._PROGRESS_SCORE_MISSING
        if prog_value == missing or parent_value == missing:
            return "Insufficient metrics for comparison"
        if prog_value > parent_value + 1e-6:
            return "Improvement in Pareto proxy"
        if prog_value < parent_value - 1e-6:
            return "Regression in Pareto proxy"
        return "No change in Pareto proxy"

    @staticmethod
    def _format_error_context(error_context: str) -> str:
        """Format retry error context."""
        return (
            "## RETRY CONTEXT\n"
            f"Previous attempt failed with error:\n```\n{error_context}\n```\n"
            "Please fix this issue in your response."
        )
