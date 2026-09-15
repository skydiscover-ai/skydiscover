"""The config reference must describe the config that actually exists: every key `configs/README.md`
shows must load through `Config.from_yaml`, and the defaults it prints must be the real ones.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from skydiscover.optimize.config import Config

README = Path(__file__).resolve().parents[2] / "skydiscover" / "optimize" / "configs" / "README.md"

# Sections whose yaml blocks are full config fragments; the model table and the
# prose snippets elsewhere are not.
_SECTION_RE = re.compile(r"^### (?P<name>[a-z_]+)\n(?P<body>.*?)(?=^### |^## |\Z)", re.S | re.M)


def _yaml_blocks():
    text = README.read_text()
    for section in _SECTION_RE.finditer(text):
        for block in re.findall(r"```yaml\n(.*?)```", section.group("body"), re.S):
            name = section.group("name")
            try:
                parsed = yaml.safe_load(block)
            except yaml.YAMLError:
                continue  # illustrative fragment, not loadable on its own
            if isinstance(parsed, dict):
                yield name, parsed


def test_readme_has_yaml_to_check():
    assert sum(1 for _ in _yaml_blocks()) >= 5


@pytest.mark.parametrize("section,payload", list(_yaml_blocks()), ids=lambda v: str(v)[:40])
def test_every_documented_key_loads(section, payload, tmp_path):
    """A documented fragment must survive Config.from_yaml, not raise TypeError."""
    # Top-level blocks are already whole configs; a subsection block needs wrapping.
    doc = payload if section == "Top-level" else payload
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(doc))
    Config.from_yaml(str(path))  # raises if a key does not exist


def test_documented_defaults_match_the_dataclasses():
    """Where the reference prints a default, it must be the real one."""
    config = Config()
    owners = {
        "llm": config.llm,
        "evaluator": config.evaluator,
        "monitor": config.monitor,
        "agentic": config.agentic,
    }
    text = README.read_text()
    wrong = []
    for section, obj in owners.items():
        match = re.search(rf"^### {section}\n(.*?)(?=^### |^## |\Z)", text, re.S | re.M)
        if not match:
            continue
        block = re.search(r"```yaml\n(.*?)```", match.group(1), re.S)
        if not block:
            continue
        parsed = yaml.safe_load(block.group(1)) or {}
        for key, documented in (parsed.get(section) or {}).items():
            if not hasattr(obj, key):
                continue  # covered by test_every_documented_key_loads
            actual = getattr(obj, key)
            if isinstance(actual, (int, float, str, bool, type(None))) and actual != documented:
                wrong.append(f"{section}.{key}: doc={documented!r} actual={actual!r}")
    assert not wrong, "config reference is out of date:\n  " + "\n  ".join(wrong)
