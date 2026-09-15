from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "skydiscover" / "synthesize" / "workflow"


def test_workflow_has_three_focused_references_and_no_numbered_rules():
    references = {path.name for path in (WORKFLOW / "references").glob("*.md")}

    assert references == {"artifacts.md", "verification.md", "state.md"}
    assert not list((WORKFLOW / "rules").glob("*.md"))


def test_workflow_markdown_references_resolve():
    repo_path = re.compile(r"`(skydiscover/synthesize/workflow/[^`]+\.md)`")
    relative_path = re.compile(r"`((?:agents|references)/[^`]+\.md)`")
    missing: list[str] = []

    for document in WORKFLOW.rglob("*.md"):
        text = document.read_text(encoding="utf-8")
        for value in repo_path.findall(text):
            if not (ROOT / value).is_file():
                missing.append(f"{document.relative_to(ROOT)} -> {value}")
        if document == WORKFLOW / "SKILL.md":
            for value in relative_path.findall(text):
                if "<" in value:
                    continue
                if not (WORKFLOW / value).is_file():
                    missing.append(f"{document.relative_to(ROOT)} -> {value}")

    assert not missing, "broken workflow documentation references:\n" + "\n".join(missing)


def test_agent_briefs_are_grouped_by_phase_and_named_after_their_role():
    """Briefs live in agents/<phase>/<role>.md; installers glob one level down and key on basename."""
    agents = WORKFLOW / "agents"
    briefs = sorted(agents.glob("*/*.md"))

    assert briefs, "no role briefs under agents/<phase>/"
    assert [p.name for p in agents.glob("*.md")] == ["README.md"]
    assert len({p.name for p in briefs}) == len(briefs), "role basenames must be unique"
    for brief in briefs:
        head = brief.read_text(encoding="utf-8").split("\n", 3)
        assert head[0] == "---", f"{brief.relative_to(ROOT)} lacks front matter"
        assert f"name: {brief.stem}" in head[1:3], f"{brief.relative_to(ROOT)} name != basename"

    skill = (WORKFLOW / "SKILL.md").read_text(encoding="utf-8")
    for brief in briefs:
        assert f"{brief.name}`" in skill, f"SKILL.md never spawns {brief.relative_to(WORKFLOW)}"


def test_every_command_the_docs_name_exists():
    """A brief that tells an agent to run `spec.<module> <verb>` (long or short form) must name a
    module that imports and a verb its --help lists; SKILL.md says the short form expands to
    `python3 -m skydiscover.synthesize.spec.<module>`."""
    long_form = re.compile(r"python3? -m (skydiscover\.synthesize(?:\.[a-zA-Z_]+)+)")
    # `spec.<dotted module>`, optionally followed by placeholders and a verb. File names such as
    # spec.json / spec.md / spec.html are not modules.
    short_form = re.compile(
        r"(?<![\w/])spec\.((?!(?:json|md|html|py)\b)[a-z_]+(?:\.[a-z_]+)*)"
        r"(?:\s+(?:<[^>]+>\s+)*([a-z][a-z-]+))?"
    )
    docs = list(WORKFLOW.rglob("*.md")) + [
        ROOT / "skydiscover" / "synthesize" / "spec" / "README.md"
    ]

    modules: set[str] = set()
    verbs: dict[str, set[str]] = {}
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        modules.update(long_form.findall(text))
        for module, verb in short_form.findall(text):
            modules.add(f"skydiscover.synthesize.spec.{module}")
            if verb:
                verbs.setdefault(module, set()).add(verb)

    unimportable = []
    for module in sorted(modules):
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            unimportable.append(f"{module}: {exc}")
    assert not unimportable, "docs name a module that does not import:\n" + "\n".join(unimportable)

    unknown = []
    for module, wanted in sorted(verbs.items()):
        proc = subprocess.run(
            [sys.executable, "-m", f"skydiscover.synthesize.spec.{module}", "--help"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        help_text = proc.stdout + proc.stderr
        for verb in sorted(wanted):
            if not re.search(r"\b%s\b" % re.escape(verb), help_text):
                unknown.append(f"spec.{module} {verb}")
    assert not unknown, "docs name a subcommand --help does not list:\n" + "\n".join(unknown)
