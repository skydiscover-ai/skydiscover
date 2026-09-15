"""
File-based human feedback reader for human feedback during discovery process.

The human edits a markdown file via the dashboard or any text editor.
The discovery loop reads it each iteration -- if it has content,
that content is appended to (or replaces) the LLM system message.
"""

import logging
import os
import time as _time

logger = logging.getLogger(__name__)

_INITIAL_TEMPLATE = """\
# Human Feedback for SkyDiscover
# Edit this file to guide the discovery process.
# Your text will be APPENDED to the LLM system message at the next iteration.
# Toggle between Append and Replace mode in the dashboard.
# Clear this file (or delete all non-comment lines) to revert to the default.
# Lines starting with # are ignored.
#
# Examples:
#   Focus on hexagonal packing and computational geometry approaches.
#   Use numpy vectorization, avoid loops. Prioritize cache-friendly access patterns.
"""

MAX_FEEDBACK_CHARS = 4000


class HumanFeedbackReader:
    """
    Reads human feedback from a markdown file on disk.

    The dashboard writes via write_from_dashboard(); the discovery loop
    reads via read(). External editors can also modify the file directly.

    Supports two modes:
    - "append" (default): feedback is appended to the system message
    - "replace": feedback replaces the system message entirely
    """

    def __init__(self, feedback_file_path: str, mode: str = "append"):
        self.path = os.path.abspath(feedback_file_path)
        self.mode = mode if mode in ("append", "replace") else "append"
        self._last_content: str = ""
        self._current_system_prompt: str = ""
        self._history: list = []
        self._create_initial_file()

    def _create_initial_file(self) -> None:
        """Create the feedback file with instructions if it doesn't exist."""
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                f.write(_INITIAL_TEMPLATE)
            logger.debug(f"Created human feedback file: {self.path}")

    def read(self) -> str:
        """
        Read current feedback, stripping comment lines.
        Returns empty string if file is empty, missing, or only has comments.
        """
        try:
            with open(self.path, "r") as f:
                raw = f.read()
        except (FileNotFoundError, PermissionError):
            return ""

        lines = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(line)

        content = "\n".join(lines).strip()
        if len(content) > MAX_FEEDBACK_CHARS:
            content = content[:MAX_FEEDBACK_CHARS]

        if content != self._last_content:
            if content:
                logger.debug(f"Human feedback updated ({len(content)} chars)")
            elif self._last_content:
                logger.debug("Human feedback cleared")
            self._last_content = content

        return content

    def write_from_dashboard(self, text: str) -> None:
        """
        Write feedback from the dashboard UI.
        Pass empty string to clear feedback.
        """
        self._write_feedback(text)

    def set_mode(self, mode: str) -> None:
        """Set feedback mode: 'append' or 'replace'."""
        if mode not in ("append", "replace"):
            logger.warning(f"Invalid human feedback mode '{mode}', ignoring")
            return
        self.mode = mode
        logger.debug(f"Human feedback mode set to: {mode}")

    def apply_feedback(self, prompt: dict) -> dict:
        """Apply current feedback to a prompt dict.

        In append mode, feedback is added after the system message.
        In replace mode, feedback replaces the system message entirely.
        Returns the modified prompt.
        """
        feedback = self.read()
        if not feedback:
            return prompt

        if self.mode == "replace":
            prompt["system"] = feedback
        else:
            prompt["system"] = prompt["system"] + "\n\n## Human Guidance\n" + feedback
        return prompt

    def set_current_prompt(self, system_prompt: str) -> None:
        """Store the current system prompt for dashboard visibility."""
        self._current_system_prompt = system_prompt

    def get_current_prompt(self) -> str:
        """Return the current system prompt."""
        return self._current_system_prompt

    def log_usage(self, iteration: int, feedback_text: str, mode: str) -> None:
        """Record that feedback was applied at a given iteration."""
        entry = {
            "iteration": iteration,
            "timestamp": _time.time(),
            "text": feedback_text,
            "mode": mode,
        }
        self._history.append(entry)
        logger.debug(
            f"Human feedback logged: iteration={iteration}, mode={mode}, "
            f"chars={len(feedback_text)}"
        )

    def get_history(self) -> list:
        """Return the full feedback usage history."""
        return list(self._history)

    def to_serializable(self) -> dict:
        """Return current state for pickling to Island workers."""
        return {
            "feedback_text": self._last_content,
            "mode": self.mode,
            "current_prompt": self._current_system_prompt,
        }

    def _write_feedback(self, text: str) -> None:
        """Write feedback text to the file, preserving the comment header."""
        with open(self.path, "w") as f:
            if text:
                f.write(_INITIAL_TEMPLATE + "\n" + text + "\n")
            else:
                f.write(_INITIAL_TEMPLATE)
