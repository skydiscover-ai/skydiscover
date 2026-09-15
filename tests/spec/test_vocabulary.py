"""The workflow speaks with one name per thing: the figure's names, pinned in SKILL.md's Vocabulary.
This scans every brief, reference, script, hook, README, and docs page of SkySynth for a retired
synonym. Examples are domain content: only the words that name a piece of the pipeline are checked
there, since "builder" or "the engine" can be the domain's own."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SYNTH = REPO / "skydiscover" / "synthesize"

# Retired word -> the pinned name. Each is a regex over the text, case-insensitive.
RETIRED = {
    r"\bdiscoverer\b": "spec-builder (discovery mode)",
    r"\bbuilders?\b(?<!spec-builder)(?<!kb-builder)(?<!spec builder)(?<!base builder)": "coding agent",
    r"adversarial[ -]auditor": "auditor",
    r"reward-hacking-auditor": "auditor",
    r"\b(gate|test)-author\b": "evaluator (correctness mode)",
    r"\b(correctness|performance)-evaluator\b": "evaluator",
    r"\bworkload-profiler\b|\bspec-auditor\b": "spec-builder",
    r"\bproduction-reviewer\b|\bcode-attacker\b": "auditor",
    r"\b(the|a) profiler\b|\bprofiler's\b": "spec-builder (workload / environment mode)",
    r"\bgates?\b": "test / release checks",
    r"\bnotes?/": "wiki/",
    r"\bledger\b": "decision log",
    r"\brulings?\b": "decision",
    r"\bcertif\w*": "validated",
    r"\bharvest\w*": "save",
    r"\bthe winner\b": "the selected candidate",
    r"\bvalidator\b": "kbtool.py",
    r"(?<!standard )(?<!YAML )(?<!static )(?<!numeric )\blibrary\b": "knowledge base",
    r"\bauthority\b|\boutranks?\b|\bin force\b": "decision / the user's answer",
    r"\(#\d+\)": "no PR or issue numbers in shipped text",
    r"\bKernelWiki\b|\bKDA\b": "no references to other projects",
    r"(?<!final )\bdeliverables?\b": "the result (Final Deliverables is the figure's phase name)",
    r"\bloopholes?\b|\bcheat\w*": "reward hack",
    r"\bthe engine\b|\bskydiscover engine\b": "the skydiscover package",
    r"\bworkspace\b": "run directory / working files",
    r"<ts>|YYYYMMDD": "<timestamp>",
    r"\bfindings\.json\b": "decisions.json",
    r"\bsystem class\b": "domain",
}


def _files():
    yield from (SYNTH / "workflow").rglob("*.md")
    yield from (SYNTH / "workflow").rglob("*.py")
    yield from (SYNTH / "workflow" / "hooks").glob("*")
    yield from (SYNTH / "spec").rglob("*.py")
    yield from (SYNTH / "scripts").glob("*.sh")
    yield SYNTH / "README.md"
    yield SYNTH / "spec" / "README.md"
    yield REPO / "skydiscover" / "main.py"
    yield REPO / "skydiscover" / "README.md"
    yield REPO / "README.md"
    yield REPO / "CONTRIBUTING.md"
    yield from (REPO / "docs" / "content" / "docs" / "synthesize").rglob("*.mdx")


# Domain content may say "builder", "library", or "the engine" about its own subject; these words
# only ever name a piece of the pipeline, so an example must use the pinned name for them.
_PIPELINE_ONLY = (
    r"\bgates?\b",
    r"\bcertif\w*",
    r"\bloopholes?\b|\bcheat\w*",
    r"\bfindings\.json\b",
    r"(?<!final )\bdeliverables?\b",
    r"\brulings?\b",
    r"\bharvest\w*",
    r"\bvalidator\b",
    r"\bKernelWiki\b|\bKDA\b",
    r"\bworkspace\b",
    r"\bsystem class\b",
    r"\bworkload-profiler\b|\bspec-auditor\b|\b(gate|test)-author\b|\bdiscoverer\b",
)


def _example_files():
    for f in (SYNTH / "examples").rglob("*"):
        if f.suffix in {".md", ".py", ".sh", ".yaml"} and f.is_file():
            yield f


def _hits():
    out = []
    for f in _files():
        if not f.is_file() or f.suffix in {".json", ".pyc"} or "__pycache__" in f.parts:
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        # The vocabulary section itself may name a retired word to retire it; and this test does.
        if f.name == "test_vocabulary.py":
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if "workspace-write" in line or "*deliverable*" in line or "first releases" in line:
                continue  # Codex's own flag; phrases the delivery hook matches; a reader of old files
            for pat, pinned in RETIRED.items():
                if re.search(pat, line, re.IGNORECASE):
                    out.append(f"{f.relative_to(REPO)}:{i}: {line.strip()[:110]}  -> say: {pinned}")
    return out


def test_no_retired_word_survives():
    hits = _hits()
    assert not hits, "retired vocabulary:\n  " + "\n  ".join(hits)


@pytest.mark.parametrize(
    "term",
    [
        "Spec Builder",
        "Coding Agent",
        "Evaluator",
        "Auditor",
        "Planner",
        "Critic",
        "Knowledge base",
        "Final Deliverables",
        "Formal spec",
        "Candidate",
        "Checkpoint",
        "Delivery hook",
    ],
)
def test_the_pinned_names_are_defined_in_skill_vocabulary(term):
    skill = (SYNTH / "workflow" / "SKILL.md").read_text(encoding="utf-8")
    vocab = skill.split("## Vocabulary", 1)[1].split("\n## ", 1)[0]
    assert re.search(r"\*\*[^*]*" + re.escape(term) + r"[^*]*\*\*", vocab, re.IGNORECASE), term


def test_examples_use_the_pipeline_names_for_the_pipeline_things():
    hits = []
    for f in _example_files():
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for pat in _PIPELINE_ONLY:
                if re.search(pat, line, re.IGNORECASE):
                    hits.append(
                        f"{f.relative_to(REPO)}:{i}: {line.strip()[:110]}  ({RETIRED[pat]})"
                    )
    assert not hits, "retired vocabulary in an example:\n    " + "\n    ".join(hits)
