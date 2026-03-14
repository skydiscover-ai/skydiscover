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
from skydiscover.context_builder.utils import TemplateManager
from skydiscover.search.base_database import Program
from skydiscover.utils.metrics import get_score

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

    @staticmethod
    def _format_sibling_context(siblings: List[Program], parent_program: Program) -> Optional[str]:
        """
        Format sibling context showing previous mutations of the parent.

        Shows what mutations have been tried before, whether they improved
        or regressed, so the LLM can avoid repeating failed approaches.
        """
        if not siblings:
            return None

        parent_fitness = get_score(getattr(parent_program, "metrics", {}))

        improved, regressed, unchanged = 0, 0, 0
        entries: List[str] = []

        for i, child in enumerate(siblings, 1):
            child_fitness = get_score(getattr(child, "metrics", {}))
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

    @staticmethod
    def _format_error_context(error_context: str) -> str:
        """Format retry error context."""
        return (
            "## RETRY CONTEXT\n"
            f"Previous attempt failed with error:\n```\n{error_context}\n```\n"
            "Please fix this issue in your response."
        )
