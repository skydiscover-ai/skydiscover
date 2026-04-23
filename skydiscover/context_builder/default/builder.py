"""Default context builder for SkyDiscover.

Builds LLM prompts with the following structure:

System message:
    Role/instructions loaded from config or template override.

User message (populated from a template, all templates share this layout):
    1. Metrics & focus areas   — current combined_score + per-metric breakdown,
                                 and suggested improvement directions.
    2. Previous attempts        — top recent programs with their changes, metrics,
                                 and whether they improved/regressed.
    3. context programs     — other programs from the database
                                 shown with their metrics and code.
    4. Current program          — the current program's code, metrics, score
                                 breakdown, and evaluator feedback (if any).
    5. Task instructions        — how to respond (diff format, full rewrite, or
                                 image generation depending on the template).

Each section is formatted by a dedicated helper (_format_metrics,
_format_previous_attempts, _format_other_context_programs, _format_current_program).
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from skydiscover.config import Config
from skydiscover.context_builder.base import ContextBuilder
from skydiscover.context_builder.utils import TemplateManager, format_artifacts, prog_attr
from skydiscover.search.base_database import Program

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = str(Path(__file__).parent / "templates")
_TEXT_LANGUAGES = {"text", "prompt", "text/plain"}

_KNOWN_TEMPLATE_VARS = {
    "current_program",
    "metrics",
    "previous_attempts",
    "other_context_programs",
    "improvement_areas",
    "language",
    "timeout_warning",
    "search_guidance",
}

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


class _EnvFormatDict(dict):
    def __missing__(self, key):
        val = os.environ.get(key.upper())
        if val is not None:
            return val
        return f"{{{key}}}"


def _substitute_placeholders(system_msg: str) -> str:
    """Substitute {placeholder} patterns in system message from env vars."""
    if not system_msg or "{" not in system_msg:
        return system_msg
    try:
        return system_msg.format_map(_EnvFormatDict())
    except (ValueError, IndexError):
        return system_msg


def _warn_unsubstituted_placeholders(system_msg: str) -> None:
    if not system_msg:
        return
    found = _PLACEHOLDER_RE.findall(system_msg)
    suspicious = [name for name in found if name not in _KNOWN_TEMPLATE_VARS]
    if suspicious:
        logger.warning(
            "Unsubstituted placeholder(s) in system message: %s. "
            "Set env vars (e.g. PROBLEM_STATEMENT for {problem_statement}) "
            "or replace with actual content.",
            ", ".join(f"{{{s}}}" for s in suspicious),
        )


def _filter_other_metrics(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k not in {"combined_score", "error"}}


class DefaultContextBuilder(ContextBuilder):
    """
    Builds LLM prompts from current program, metrics, and context programs.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.system_template_override = None
        self.user_template_override = None
        self.template_manager = TemplateManager(_TEMPLATES_DIR, self.context_config.template_dir)
        self._system_message_warned = False

    def set_templates(
        self, system_template: Optional[str] = None, user_template: Optional[str] = None
    ) -> None:
        """Override the default system/user template keys.

        Pass None for either argument to keep the current value.
        """
        self.system_template_override = system_template
        self.user_template_override = user_template
        logger.info(f"Templates set: system={system_template}, user={user_template}")

    # ------------------------------------------------------------------
    # Main Prompt Builder
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Build a prompt dict with "system" and "user" keys.

        Args:
            current_program: Program or {info: Program} to evolve from.
            context: optional dict with program_metrics, other_context_programs,
                previous_programs, etc.

        Returns:
            {"system": str, "user": str} ready for LLM.generate().
        """
        context = context or {}

        # EXPERIENCES: information needed from the database
        program_metrics = context.get("program_metrics", {})
        other_context_programs = context.get("other_context_programs", {})
        previous_programs = context.get("previous_programs", [])

        # Information needed from the config
        language = self.config.language or "python"
        diff_based_generation = self.config.diff_based_generation

        # Format experiences
        metrics_str = self._format_metrics(program_metrics)
        previous_attempts_section = self._format_previous_attempts(previous_programs)
        other_context_section = self._format_other_context_programs(
            other_context_programs, language
        )
        current_program_section = self._format_current_program(current_program, language)
        has_current_program = bool(current_program_section)

        if isinstance(current_program, dict) and current_program:
            actual_program = list(current_program.values())[0]
            current_solution = prog_attr(actual_program, "solution")
        else:
            current_solution = prog_attr(current_program, "solution")

        improvement_areas = self._identify_improvement_areas(
            current_solution, program_metrics, previous_programs
        )

        if context.get("errors"):
            other_context_section += self._format_failed_attempts(context["errors"], language)

        evaluator_timeout = getattr(self.config.evaluator, "timeout", None)
        timeout_warning = (
            f"- Time limit: Programs should complete execution within {evaluator_timeout} seconds; otherwise, they will timeout."
            if evaluator_timeout
            else ""
        )

        user_template_key = self._select_template_key(
            language, diff_based_generation, has_current_program
        )
        user_template = self.template_manager.get_template(user_template_key)

        user_message = user_template.format(
            current_program=current_program_section,
            metrics=metrics_str,
            previous_attempts=previous_attempts_section,
            other_context_programs=other_context_section,
            improvement_areas=improvement_areas,
            language=language,
            timeout_warning=timeout_warning,
            **kwargs,
        )

        return {"system": self._get_system_message(), "user": user_message}

    def _select_template_key(
        self, language: str, diff_based: bool, has_current_program: bool = True
    ) -> str:
        """Pick template: override > auto (from_scratch / image / diff / full rewrite)."""
        if self.user_template_override:
            return self.user_template_override

        if not has_current_program:
            return "from_scratch_user_message"

        if language == "image":
            return "image_user_message"

        if diff_based:
            return "diff_user_message"

        if language.lower() in _TEXT_LANGUAGES:
            return "full_rewrite_prompt_opt_user_message"
        return "full_rewrite_user_message"

    def _get_system_message(self) -> str:
        """Return system message, substituting {placeholder} patterns from env vars."""
        if self.system_template_override:
            return self.template_manager.get_template(self.system_template_override)
        system_msg = self.context_config.system_message
        if system_msg in self.template_manager.templates:
            system_msg = self.template_manager.get_template(system_msg)

        system_msg = _substitute_placeholders(system_msg)

        if not self._system_message_warned:
            self._system_message_warned = True
            _warn_unsubstituted_placeholders(system_msg)

        return system_msg

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_current_program(
        self, current_program: Union[Program, Dict[str, Program]], language: str
    ) -> str:
        """Format parent program with heading, score breakdown, code, and feedback.

        Returns empty string when there is no current program so the heading
        is omitted entirely from the rendered prompt.
        """
        if not current_program:
            return ""

        if isinstance(current_program, dict):
            info = list(current_program.keys())[0]
            program = list(current_program.values())[0]
        else:
            info = ""
            program = current_program

        solution = prog_attr(program, "solution")
        metrics = prog_attr(program, "metrics", {})

        lines = ["# Current Solution\n"]
        if info:
            lines.append(f"\n{info}\n")

        lines.append("\n## Program Information\n")
        if metrics:
            combined = metrics.get("combined_score")
            if combined is not None and isinstance(combined, (int, float)):
                lines.append(f"combined_score: {combined:.4f}\n")

            error = metrics.get("error")
            if error:
                lines.append(f"error: {error}\n")

            other_metrics = _filter_other_metrics(metrics)
            if other_metrics:
                lines.append("Score breakdown:")
                for key, value in other_metrics.items():
                    if isinstance(value, float):
                        lines.append(f"\n  - {key}: {value:.4f}")
                    elif isinstance(value, (int, str, bool)):
                        lines.append(f"\n  - {key}: {value}")
                lines.append("\n")

        if language != "image":
            lines.append(f"\n```{language}\n{solution}\n```\n")

        feedback_section = format_artifacts(program, heading="##")
        if feedback_section:
            lines.append(feedback_section)

        return "".join(lines)

    def _identify_improvement_areas(
        self,
        current_program: str,
        metrics: Dict[str, float],
        previous_programs: List[Program],
    ) -> str:
        """Generate bullet points: score trend vs previous attempt, simplification hint."""
        improvement_areas = []

        current_score = metrics.get("combined_score", 0.0)
        if not isinstance(current_score, (int, float)):
            try:
                current_score = float(current_score)
            except (ValueError, TypeError):
                current_score = 0.0

        if previous_programs:
            prev = previous_programs[-1]
            prev_metrics = prog_attr(prev, "metrics", {})
            prev_score = prev_metrics.get("combined_score", 0.0)
            if not isinstance(prev_score, (int, float)):
                try:
                    prev_score = float(prev_score)
                except (ValueError, TypeError):
                    prev_score = 0.0

            if current_score > prev_score:
                improvement_areas.append(
                    f"Combined score improved: {prev_score:.4f} → {current_score:.4f}"
                )
            elif current_score < prev_score:
                improvement_areas.append(
                    f"Combined score declined: {prev_score:.4f} → {current_score:.4f}. Consider revising recent changes."
                )
            elif abs(current_score - prev_score) < 1e-6:
                improvement_areas.append(f"Combined score unchanged at {current_score:.4f}")

        threshold = self.context_config.suggest_simplification_after_chars
        if threshold and len(current_program) > threshold:
            improvement_areas.append(
                f"Consider simplifying - solution length exceeds {threshold} characters"
            )

        if not improvement_areas:
            improvement_areas.append("Focus on improving the combined_score")

        return "\n".join(f"- {area}" for area in improvement_areas)

    def _format_single_context_program(
        self, program: Program, index: int, language: str, lines: list
    ) -> None:
        """Append one context program's header, metrics, and code to lines."""
        if program is None:
            return

        solution = prog_attr(program, "solution")
        metrics = prog_attr(program, "metrics", {})

        combined = metrics.get("combined_score") if metrics else None
        if combined is not None and isinstance(combined, (int, float)):
            lines.append(f"### Program {index} (combined_score: {combined:.4f})\n")
        else:
            lines.append(f"### Program {index}\n")

        if metrics:
            error = metrics.get("error")
            if error:
                lines.append(f"- error: {error}\n")

            other_metrics = _filter_other_metrics(metrics)
            if other_metrics:
                lines.append("Score breakdown:")
                for key, value in other_metrics.items():
                    if isinstance(value, float):
                        lines.append(f"  - {key}: {value:.4f}")
                    elif isinstance(value, (int, str, bool)):
                        lines.append(f"  - {key}: {value}")
                lines.append("\n")

        if language != "image":
            lines.append(f"\n```{language}\n{solution}\n```\n")
        lines.append("\n")

    def _format_other_context_programs(
        self,
        other_context_programs: Union[List[Program], Dict[str, List[Program]]],
        language: str,
    ) -> str:
        """Format all context programs, grouped by key when dict-wrapped."""
        if not other_context_programs:
            return ""

        lines = []
        if isinstance(other_context_programs, dict):
            for label, programs in other_context_programs.items():
                if not programs:
                    continue
                lines.append(f"\n## {label or 'Other Context Solutions'}\n")
                lines.append(
                    "These programs represent diverse approaches and creative solutions that may be relevant to the current task:\n\n"
                )
                for i, program in enumerate(programs, start=1):
                    self._format_single_context_program(program, i, language, lines)
        else:
            lines.append(
                "These programs represent diverse approaches and creative solutions that may inspire new ideas:\n"
            )
            for i, program in enumerate(other_context_programs, start=1):
                self._format_single_context_program(program, i, language, lines)

        return "".join(lines)

    def _format_failed_attempts(self, errors: list, language: str) -> str:
        """Format failed retry attempts for the prompt."""
        lines = ["\n## ❌ Previous Failed Attempts (this retry):\n"]
        lines.append("The following attempts failed. Avoid these errors:\n\n")
        for attempt in errors:
            err_msg = attempt.get("metadata", {}).get("error", "Unknown error")
            attempt_num = attempt.get("metadata", {}).get("attempt_number", "?")
            lines.append(f"### Attempt {attempt_num}:\n")
            lines.append(f"**Error:** {err_msg}\n")

            failed_solution = attempt.get("solution", "")
            llm_response = attempt.get("llm_response", "")

            if "SEARCH" in err_msg and llm_response:
                if len(llm_response) > 1500:
                    llm_response = llm_response[:1500] + "\n... (truncated)"
                lines.append(f"**Your response that failed:**\n```\n{llm_response}\n```\n\n")
            elif failed_solution:
                if len(failed_solution) > 1500:
                    failed_solution = failed_solution[:1500] + "\n... (truncated)"
                lines.append(
                    f"**Generated solution that failed:**\n```{language}\n{failed_solution}\n```\n"
                )

                traceback_str = attempt.get("metadata", {}).get("traceback", "")
                if traceback_str:
                    if len(traceback_str) > 800:
                        traceback_str = "... (truncated)\n" + traceback_str[-800:]
                    lines.append(f"**Traceback:**\n```\n{traceback_str}\n```\n\n")
                else:
                    lines.append("\n")
        return "".join(lines)

    def _format_previous_attempts(
        self, previous_programs: List[Program], num_previous_attempts: int = 3
    ) -> str:
        """Format top N previous attempts with their changes, metrics, and outcome."""
        if not previous_programs:
            return "No previous attempts yet."

        try:
            previous_attempt_template = self.template_manager.get_template("previous_attempt")
        except (ValueError, KeyError):
            previous_attempt_template = "### Attempt {attempt_number}\n- Changes: {changes}\n- Metrics: {performance}\n- Outcome: {outcome}"

        previous_programs = sorted(
            previous_programs,
            key=lambda p: prog_attr(p, "metrics", {}).get("combined_score", 0.0),
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
                if isinstance(value, (int, float)):
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

    @staticmethod
    def _determine_outcome(program_metrics: Dict[str, Any], parent_metrics: Dict[str, Any]) -> str:
        """Compare combined_score to parent: 'Improvement', 'Regression', or 'No change'."""
        prog_value = program_metrics.get("combined_score")
        parent_value = parent_metrics.get("combined_score", 0)
        if isinstance(prog_value, (int, float)) and isinstance(parent_value, (int, float)):
            if prog_value > parent_value:
                return "Improvement in combined_score"
            elif prog_value < parent_value:
                return "Regression in combined_score"
        return "No change in combined_score"

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics: combined_score first, then error, then per-metric breakdown."""
        if not metrics:
            return "No metrics available"

        lines = []
        combined_score = metrics.get("combined_score")
        if combined_score is not None:
            lines.append(f"- combined_score: {combined_score:.4f}")

        error = metrics.get("error")
        if error:
            lines.append(f"- error: {error}")

        other_metrics = _filter_other_metrics(metrics)
        if other_metrics:
            lines.append("")
            lines.append("Metrics:")
            for key, value in other_metrics.items():
                if isinstance(value, float):
                    lines.append(f"  - {key}: {value:.4f}")
                elif isinstance(value, (int, str, bool)):
                    lines.append(f"  - {key}: {value}")

        return "\n".join(lines) if lines else "No metrics available"
