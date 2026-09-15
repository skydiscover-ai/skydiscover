"""
GEPA Native context builder for SkyDiscover.

Extends DefaultContextBuilder with GEPA-specific reflective prompting:
- Reflective analysis framing (instructs the LLM to reason about failures)
- Rejection history (recently rejected programs with scores, errors, and code)

These are assembled into a ``search_guidance`` string and injected into
templates via the ``{search_guidance}`` placeholder.

Metrics and evaluator diagnostics are already rendered by the default
template in the ``{current_program}`` section, so they are NOT duplicated
in the reflective guidance.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from skydiscover.optimize.config import Config
from skydiscover.optimize.context_builder.default import DefaultContextBuilder
from skydiscover.optimize.context_builder.utils import TemplateManager
from skydiscover.optimize.search.base_database import Program
from skydiscover.optimize.utils.metrics import get_score

logger = logging.getLogger(__name__)


class GEPANativeContextBuilder(DefaultContextBuilder):
    """
    Context builder for GEPA Native's reflective evolutionary search.

    Adds a ``{search_guidance}`` section to the prompt containing:
    - Reflective analysis framing (tells the LLM to reason about failures)
    - Recent rejected programs (code that didn't improve on the parent)

    The controller passes raw data via the ``context`` dict:
    - ``context["rejection_history"]``: list of rejected Program objects
    - ``context["rejection_parent_scores"]``: dict mapping parent_id -> float score

    Metrics and evaluator diagnostics are already in {current_program}
    via the default template, so they are not repeated here.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        default_templates = str(Path(__file__).parent.parent / "default" / "templates")
        gepa_templates = str(Path(__file__).parent / "templates")
        self.template_manager = TemplateManager(
            default_templates, gepa_templates, self.context_config.template_dir
        )

    def build_prompt(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Build prompt with GEPA reflective search guidance.

        Computes the ``search_guidance`` string from GEPA context keys,
        then delegates to the parent's ``build_prompt`` which fills the
        ``{search_guidance}`` placeholder in the template.
        """
        context = context or {}

        search_guidance = self._build_search_guidance(current_program, context)

        kwargs.pop("search_guidance", None)

        result = super().build_prompt(
            current_program,
            context,
            search_guidance=search_guidance,
            **kwargs,
        )

        # Collapse 3+ consecutive newlines to 2 (empty search_guidance leaves extra blank lines)
        if "user" in result:
            result["user"] = re.sub(r"\n{3,}", "\n\n", result["user"])

        return result

    # =========================================================================
    # Search Guidance Assembly
    # =========================================================================

    def _build_search_guidance(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context: Dict[str, Any],
    ) -> str:
        """
        Assemble GEPA-specific reflective guidance into one string.

        Sections:
        1. Reflective analysis framing (always present when there's content)
        2. Rejection history (recently rejected programs)
        """
        rejection_history = context.get("rejection_history", [])
        rejection_parent_scores = context.get("rejection_parent_scores", {})

        sections: List[str] = []

        # Rejection history
        if rejection_history:
            rejection_section = self._format_rejection_history(
                rejection_history, rejection_parent_scores
            )
            if rejection_section:
                sections.append(rejection_section)

        if not sections:
            return ""

        # Prepend reflective framing header
        header = (
            "## Reflective Analysis\n"
            "Review the evaluation results and diagnostics in the program "
            "information above. Identify root causes and domain-specific "
            "insights. Address these failure modes in your solution."
        )

        return header + "\n\n" + "\n\n".join(sections)

    # =========================================================================
    # Section Formatters
    # =========================================================================

    @staticmethod
    def _format_rejection_history(
        rejected: List[Program],
        parent_scores: Dict[str, float],
    ) -> Optional[str]:
        """
        Format recently rejected programs for reflective prompting.

        Shows what mutations were tried and rejected (didn't improve on parent),
        including their scores, error messages, changes descriptions, and code
        snippets so the LLM can avoid repeating the same mistakes.
        """
        if not rejected:
            return None

        entries: List[str] = []

        for i, prog in enumerate(rejected, 1):
            prog_score = get_score(prog.metrics) if prog.metrics else 0.0

            parent_score_str = ""
            if prog.parent_id and prog.parent_id in parent_scores:
                parent_score_str = f", parent_score: {parent_scores[prog.parent_id]:.4f}"

            error_msg = ""
            if prog.metrics:
                error_msg = prog.metrics.get("error", "")
                if not error_msg:
                    error_msg = prog.metrics.get("error_message", "")

            changes = ""
            if prog.metadata:
                changes = prog.metadata.get("changes", "")

            entry_lines = [f"#### Attempt {i} (score: {prog_score:.4f}{parent_score_str})"]
            if changes:
                entry_lines.append(f"Changes: {changes}")
            if error_msg:
                entry_lines.append(f"Error: {error_msg}")

            # Code snippet so the LLM can see what was tried
            if prog.solution:
                code_lines = prog.solution.splitlines()
                if len(code_lines) > 30:
                    snippet = "\n".join(code_lines[:30]) + "\n... (truncated)"
                else:
                    snippet = prog.solution
                entry_lines.append(f"Code tried:\n```\n{snippet}\n```")

            entries.append("\n".join(entry_lines))

        lines = [
            "### Recent Rejected Attempts",
            "The following mutations were rejected because they did not "
            "improve on the parent. Avoid repeating these approaches:",
            *entries,
        ]
        return "\n\n".join(lines)
