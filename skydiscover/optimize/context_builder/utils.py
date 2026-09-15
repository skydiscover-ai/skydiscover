"""Shared utilities for context builders."""

from pathlib import Path
from typing import Any, Optional


class TemplateManager:
    """Loads .txt templates from one or more directories.

    Directories are processed in order; later directories override
    templates with the same name from earlier ones.
    """

    def __init__(self, *directories: Optional[str]):
        """
        Initializes the TemplateManager with the given directories.
        If there are multiple directories, the templates from the later directories will override
        the templates from the earlier directories.
        """
        self.templates: dict[str, str] = {}
        for d in directories:
            if d:
                path = Path(d)
                if path.exists():
                    self._load_from_directory(path)

    def _load_from_directory(self, directory: Path) -> None:
        for txt_file in directory.glob("*.txt"):
            with open(txt_file, "r") as f:
                self.templates[txt_file.stem] = f.read()

    def get_template(self, name: str) -> str:
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]


def prog_attr(program: Any, key: str, default: Any = "") -> Any:
    """Read an attribute from a Program object or a plain dict."""
    if hasattr(program, key):
        return getattr(program, key)
    if isinstance(program, dict):
        return program.get(key, default)
    return default


def format_artifacts(program: Any, heading: str = "##", max_len: int = 2000) -> str:
    """Format evaluator artifacts (e.g. feedback) into markdown sections."""
    artifacts = prog_attr(program, "artifacts", None)
    if not artifacts:
        return ""
    sections = []
    for key, value in artifacts.items():
        if value is None:
            continue
        text = str(value)
        if len(text) > max_len:
            text = text[:max_len] + "\n... (truncated)"
        if key == "feedback":
            sections.append(f"{heading} Evaluator Feedback\n{text}")
        else:
            sections.append(f"{heading} {key}\n{text}")
    if not sections:
        return ""
    return "\n" + "\n\n".join(sections) + "\n"
