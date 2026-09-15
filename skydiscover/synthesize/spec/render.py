"""Render best/spec.md from the run's cards.

spec.md is a fixed template. Each cell is a card field cut to its first
sentence; a section without a card is skipped.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .paths import Run, test_files, test_id

CELL = 140  # max characters per cell

_PATH = re.compile(r"(?:^|\s)(?:~|\.{0,2}/)\S*/\S+")


def first_sentence(text: Any, limit: int = CELL) -> str:
    """Return the first sentence of `text`, at most `limit` characters."""
    flat = re.sub(r"\s+", " ", str(text or "")).strip()
    match = re.match(r"(.+?[.!?])(?:\s|$)", flat)
    flat = match.group(1) if match else flat
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def label(key: str) -> str:
    """Turn a card key into a label: `floor:bounded-memory` -> `Bounded memory`."""
    words = key.split(":", 1)[-1].replace("_", " ").replace("-", " ").strip()
    return words[:1].upper() + words[1:]


def is_published(row: Any) -> bool:
    """Whether a history row's checkpoint was published as best (`best` in the first releases)."""
    return isinstance(row, dict) and bool(row.get("published", row.get("best")))


def passed_final_tests(row: Any) -> bool:
    """Whether a history row's checkpoint was published and fails none of the final tests."""
    return is_published(row) and row.get("fails") == []


def baselines_of(score: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The baselines a score.json compares against, headline first. The first releases wrote one
    `baseline`, either `{"name", "score"}` or the bare metrics; read those too."""
    listed = score.get("baselines")
    if isinstance(listed, list):
        return [b for b in listed if isinstance(b, dict) and isinstance(b.get("score"), dict)]
    single = score.get("baseline")
    if not isinstance(single, dict):
        return []
    if isinstance(single.get("score"), dict):
        return [single]
    return [{"name": "the baseline", "score": single}]


def render_spec(
    run_dir: Path,
    score: Mapping[str, Any],
    history: Optional[list[dict]] = None,
    *,
    measured_config: Optional[Mapping[str, Any]] = None,
    direction: Optional[str] = None,
) -> str:
    """Return the spec.md page for the checkpoint described by `score`."""
    run = Run(run_dir)
    req = _load(run.cards / "requirements.json", {})
    props = _load(run.cards / "properties.json", [])
    workload = _load(run.cards / "workload.json", {})
    env = _load(run.cards / "environment.json", {})
    log = _load(run.decision_log, [])
    tests = _Tests(run.tests)

    out = _header(run, req, score, tests, direction)
    passed = any(
        passed_final_tests(row) and row.get("checkpoint") == score.get("checkpoint")
        for row in history or []
    )
    out += [
        (
            "Verified: every test in `tests/` passes against `artifact/`."
            if passed
            else "Not yet verified: the final check has not run on this checkpoint."
        ),
        "",
    ]
    out += _properties(req, props, tests)
    out += _workload(workload, measured_config)
    out += _environment(env, req)
    out += _hacks(log, tests)
    out += _checkpoints(history or [], score.get("checkpoint"))
    return "\n".join(out).rstrip() + "\n"


def _header(
    run: Run,
    req: dict,
    score: Mapping[str, Any],
    tests: "_Tests",
    direction: Optional[str] = None,
) -> list[str]:
    title = str(req.get("title") or _task_heading(run.task) or run.slug)
    title = re.split(r"\s+(?:--|—)\s+", title, maxsplit=1)[0]
    out = [f"# {title}", "", f"Built by SkySynth on {datetime.now(timezone.utc):%Y-%m-%d}.", ""]

    facts = []
    baselines = baselines_of(score)
    for metric, value in (score.get("score") or {}).items():
        cell = f"**{_number(value)} {metric}**"
        margins = []
        for base in baselines:
            ref = (base.get("score") or {}).get(metric)
            # "How much better": value/ref when higher is better, ref/value for a latency or a
            # cost (direction "min"). Non-positive numbers have no ratio.
            if (
                isinstance(ref, (int, float))
                and isinstance(value, (int, float))
                and ref > 0
                and value > 0
            ):
                ratio = ref / value if direction == "min" else value / ref
                margins.append(f"{ratio:.2f}× {base.get('name', 'the baseline')} ({_number(ref)})")
        if margins:
            cell += " — " + ", ".join(margins)
        facts.append(["Score", cell])
    if tests:
        facts.append(["Tests", f"{len(tests)} in `tests/`"])
    if score.get("checkpoint"):
        facts.append(["Checkpoint", f"{score['checkpoint']}, the best of the run"])
    return out + _table(None, facts)


def _properties(req: dict, props: list, tests: "_Tests") -> list[str]:
    out: list[str] = []
    test_col = ["Test"] if tests else []

    axes = [p for p in req.get("required_properties", []) if isinstance(p, dict) and p.get("axis")]
    if axes:
        out += ["## Properties", ""]
        out += [
            "The specification is a set of questions, one per property; the answer is what the system guarantees.",
            "",
        ]
        rows = []
        for p in axes:
            row = [
                label(p["axis"]),
                first_sentence(p.get("requirement")),
                first_sentence(p.get("value")),
            ]
            rows.append(row + ([tests.for_property(p.get("id") or p["axis"])] if tests else []))
        out += _table(["Property", "Question", "Answer"] + test_col, rows)

    answered = {p.get("id") or p["axis"] for p in axes}
    always = [p for p in props if isinstance(p, dict) and p.get("id") and p["id"] not in answered]
    if always:
        if not axes:
            out += ["## Properties", ""]
        out += [
            "Always required, whatever the answers above:" if axes else "Required guarantees:",
            "",
        ]
        rows = []
        for p in always:
            row = [label(p["id"]), first_sentence(p.get("guarantee") or p.get("property"))]
            rows.append(row + ([tests.for_property(p["id"])] if tests else []))
        out += _table(["Property", "Guarantee"] + test_col, rows)
    return out


def _workload(workload: dict, measured_config: Optional[Mapping[str, Any]]) -> list[str]:
    settings = _scalars(
        measured_config
        if measured_config is not None
        else workload.get("scored_configuration") or {}
    )
    if not settings:
        return []
    description = (
        "The configuration the score is measured at."
        if measured_config is not None
        else "Requested benchmark configuration; no measured configuration is available."
    )
    out = ["## Workload", "", description, ""]
    return out + _table(None, [[label(k), _number(v)] for k, v in settings])


def _environment(env: dict, req: dict) -> list[str]:
    rows = []
    machine = env.get("machine") or {}
    rows += [[label(k), first_sentence(v)] for k, v in machine.items() if isinstance(v, str)]
    resource = (env.get("bound") or {}).get("resource")
    if resource:
        rows.append(["Binding resource", first_sentence(resource)])
    ceiling = env.get("ceiling") or {}
    if isinstance(ceiling.get("value"), (int, float)):
        rows.append(["Ceiling", f"{_number(ceiling['value'])} {ceiling.get('unit', '')}".strip()])

    op = req.get("operating_point") or {}
    parts = []
    for name, bounds in (op.get("params") or {}).items():
        if not isinstance(bounds, dict):
            continue
        if "eq" in bounds:
            parts.append(f"{name} = {_number(bounds['eq'])}")
        if "max" in bounds:
            parts.append(f"{name} ≤ {_number(bounds['max'])}")
        if "min" in bounds:
            parts.append(f"{name} ≥ {_number(bounds['min'])}")
    checkers = op.get("checker")
    if isinstance(checkers, str) and checkers:
        checkers = [checkers]
    checkers = [c for c in (checkers if isinstance(checkers, list) else []) if c != "plain"]
    if checkers:  # "plain" is an ordinary build; a sanitizer is worth naming
        parts.append(f"under {', '.join(map(str, checkers))}")
    if parts:
        rows.append(["Tests run at", "; ".join(parts)])

    return ["## Environment", ""] + _table(None, rows) if rows else []


def _hacks(log: list, tests: "_Tests") -> list[str]:
    hacks = [r for r in log if isinstance(r, dict) and r.get("kind") in ("hack", "overfit")]
    if not hacks:
        return []
    out = ["## Reward hacks found", ""]
    out += [
        "Shortcuts that would raise the score while breaking a property. Each is closed by a test.",
        "",
    ]
    rows = []
    for r in hacks:
        if r.get("test") and tests.has(str(r["test"])):
            caught = f"`{Path(str(r['test'])).name}`"
        elif r.get("status") == "waived":
            caught = "no longer possible (waived)"
        else:
            caught = "**open**"
        rows.append([first_sentence(r.get("title")), caught])
    return out + _table(["Hack", "Caught by"], rows)


def _checkpoints(history: list, best: Any) -> list[str]:
    rows = [r for r in history if isinstance(r, dict) and isinstance(r.get("checkpoint"), int)]
    if not rows:
        return []
    metric = next((k for r in rows for k in (r.get("score") or {})), "score")
    out = ["## Checkpoints", "", "Every scored iteration, from `history.json`.", ""]
    body = []
    for r in rows:
        value = (r.get("score") or {}).get(metric)
        fails = r.get("fails") or []
        if fails:
            more = f" and {len(fails) - 1} more" if len(fails) > 1 else ""
            note = f"fails `{fails[0]}`{more}"
        elif "fails" in r and r["fails"] is None:
            note = "score not re-checked"
        elif r["checkpoint"] == best and is_published(r):
            note = "**best**"
        else:
            note = ""
        score = _number(value) if value is not None else ""
        tests = r.get("tests")
        body.append([r["checkpoint"], score, "" if tests is None else tests, note])
    return out + _table(["#", metric, "Tests when scored", ""], body)


class _Tests:
    """The kept test files, looked up by property id."""

    def __init__(self, suite: Path):
        self.names = [p.name for p in test_files(suite)] if suite.is_dir() else []

    def __len__(self) -> int:
        return len(self.names)

    def for_property(self, pid: str) -> str:
        """`ordering` -> "`check_ordering.<ext>`", or "—" when no test matches."""
        stem = re.escape(pid.split(":", 1)[-1].replace("-", "_"))
        boundary = re.compile(rf"(?:^|_){stem}(?:_|$)")
        hits = [n for n in self.names if boundary.search(test_id(Path(n)))]
        return " · ".join(f"`{h}`" for h in hits) if hits else "—"

    def has(self, name: str) -> bool:
        return Path(name).name in self.names


def _load(path: Path, default: Any) -> Any:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read result input {path}: {exc}") from exc
    if not isinstance(doc, type(default)):
        raise ValueError(f"Invalid result input {path}: expected {type(default).__name__}")
    return doc


def _task_heading(task: Path) -> Optional[str]:
    try:
        text = task.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^# (.+)$", text, re.M)
    if not match:
        return None
    return re.sub(r"^(?:task|build):\s*", "", match.group(1).strip(), flags=re.I)


def _scalars(block: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Short settings only: numbers and strings under 60 characters, no paths or commands."""
    rows = []
    for key, value in block.items():
        if key.startswith("_") or key == "command" or value is None:
            continue
        if isinstance(value, (dict, list)):
            continue
        if isinstance(value, str) and (len(value) > 60 or _PATH.search(value)):
            continue
        rows.append((key, value))
    return rows


def _number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        # Plain digits with thousands separators; never scientific notation in a headline.
        if abs(value) >= 1e6:
            return f"{value:,.0f}"
        text = f"{value:.6g}"
        return f"{value:.10f}".rstrip("0").rstrip(".") if "e" in text else text
    return str(value)


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(header: Optional[Iterable[str]], rows: Iterable[Iterable[Any]]) -> list[str]:
    head = list(header) if header else []
    body = [list(r) for r in rows]
    if not body:
        return []
    width = len(head) if head else len(body[0])
    lines = ["| " + " | ".join(head or [""] * width) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in body]
    return lines + [""]
