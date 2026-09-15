"""Draft and check the questions about what a system must do (questions.json).

A good question asks about behavior ("what may a crash lose?"), not design ("checkpoint or
write-ahead log?"). seed_from_axes makes one slot per discovered axis; validate checks the shape;
lint flags a question that names a design. The words that count as designs come from the run
itself, never from a built-in list: the reference systems' names (references/<name>/) and the
mechanisms discovery listed in references/axes.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def seed_from_axes(axes_doc: Dict[str, Any]) -> Dict[str, Any]:
    """One question slot per discovered axis. Options are left empty; choosing them is the agent's job."""
    questions: List[Dict[str, Any]] = []
    for axis in axes_doc.get("axes", []):
        questions.append(
            {
                "id": axis.get("id"),
                "axis": axis.get("id"),
                "q": axis.get("question", ""),  # rewrite to a need before posing
                "why": axis.get("rationale", ""),
                "options": [],  # fill with property VALUES (no systems, no mechanisms)
                "default": None,
                "multi": False,
            }
        )
    return {
        "title": f"Requirements for: {axes_doc.get('domain', 'a system')}",
        "domain": axes_doc.get("domain"),
        "intro": (
            "Answer each question for the system you need, in terms of what it must do -- not how. "
            "The architecture that meets your answers is designed afterwards."
        ),
        "questions": questions,
    }


def gather_value_space(
    axis_id: str, specs_by_system: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """For one axis, the property values the reference systems actually exhibit, with file:line
    evidence. The agent turns these into system-free options."""
    out: List[Dict[str, Any]] = []
    for sys_name, spec in specs_by_system.items():
        ax = (spec.get("axes") or {}).get(axis_id)
        if isinstance(ax, dict):
            out.append(
                {
                    "from": sys_name,
                    "property": ax.get("property"),
                    "guarantee": ax.get("guarantee"),
                    "evidence": ax.get("evidence"),
                }
            )
    return out


def seed_from_specs(
    axes_doc: Dict[str, Any], specs_by_system: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Like seed_from_axes, but each slot carries the axis's real value_space across the
    pulled systems -- the evidence the skill distills option values from. Grounds authoring so the
    options reflect what real systems do (eventual vs strong vs snapshot), not generic guesses.
    """
    scaffold = seed_from_axes(axes_doc)
    for q in scaffold["questions"]:
        q["value_space"] = gather_value_space(q["axis"], specs_by_system)
    return scaffold


def validate(doc: Dict[str, Any]) -> List[str]:
    """Structural validation. Returns a list of human-readable problems (empty == valid)."""
    problems: List[str] = []
    if not isinstance(doc, dict):
        return ["expected an object containing a 'questions' list, not a bare list"]
    qs = doc.get("questions")
    if not isinstance(qs, list) or not qs:
        return ["document has no 'questions' list"]
    seen = set()
    for i, q in enumerate(qs):
        if not isinstance(q, dict):
            problems.append(f"question[{i}]: expected an object")
            continue
        qid = q.get("id")
        where = f"question[{i}] (id={qid!r})"
        if not isinstance(qid, str) or not qid:
            problems.append(f"{where}: missing id")
        elif qid in seen:
            problems.append(f"{where}: duplicate id")
        else:
            seen.add(qid)
        if not q.get("q"):
            problems.append(f"{where}: missing question text 'q'")
        if not q.get("why"):
            problems.append(f"{where}: missing 'why' (the requirement's rationale)")
        opts = q.get("options")
        if not isinstance(opts, list) or len(opts) < 2:
            problems.append(f"{where}: needs >=2 value options")
            continue
        for j, o in enumerate(opts):
            if not isinstance(o, dict) or not o.get("value"):
                problems.append(f"{where}.options[{j}]: each option needs a 'value'")
            # a property option must not smuggle in the name of the system it came from
            if isinstance(o, dict) and ("ref" in o or "system" in o):
                problems.append(
                    f"{where}.options[{j}]: options must be property values, not designs (drop 'ref'/'system')"
                )
    return problems


def terms_from(specification: Path) -> List[str]:
    """The words a question may not use, read from the run: every reference system's name (the
    folder under references/ and the repository in its spec.json `source`) and every entry of
    `mechanisms` in references/axes.json."""
    refs = Path(specification) / "references"
    terms: List[str] = []
    axes = _read_json(refs / "axes.json")
    if isinstance(axes, dict):
        terms += [str(t) for t in axes.get("mechanisms") or [] if t]
    if refs.is_dir():
        for d in sorted(p for p in refs.iterdir() if p.is_dir() and not p.name.startswith(".")):
            terms.append(d.name)
            spec = _read_json(d / "spec.json")
            source = spec.get("source") if isinstance(spec, dict) else None
            m = re.match(r"\s*(?:[\w.-]+/)?([\w.-]+)", str(source or ""))
            if m:
                terms.append(m.group(1))
    seen: set = set()
    out: List[str] = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def lint(doc: Dict[str, Any], terms: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Flag every question, rationale, or option that contains one of `terms` (a reference
    system's name or a mechanism). Returns [{question_id, where, term, text}]."""
    words = [t.lower() for t in (terms or []) if t]
    findings: List[Dict[str, str]] = []

    def _scan(text: str, qid: str, where: str) -> None:
        low = (text or "").lower()
        for t in words:
            if re.search(
                rf"(?<![a-z0-9]){re.escape(t)}(?:e?s)?(?![a-z0-9])", low
            ):  # whole word, plural too
                findings.append({"question_id": qid, "where": where, "term": t, "text": text})

    for q in doc.get("questions", []):
        qid = q.get("id", "?")
        _scan(q.get("q", ""), qid, "q")
        _scan(q.get("why", ""), qid, "why")
        for j, o in enumerate(q.get("options", [])):
            if isinstance(o, dict):
                _scan(str(o.get("value", "")), qid, f"options[{j}].value")
                _scan(str(o.get("note", "")), qid, f"options[{j}].note")
    return findings


def _read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m skydiscover.synthesize.spec.requirements",
        description="Draft and check the questions about what the system must do (questions.json).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sf = sub.add_parser(
        "seed-from-specs",
        help="one question per axis in axes.json, each carrying the values the reference systems exhibit",
    )
    sf.add_argument("--axes", required=True, help="specification/references/axes.json")
    sf.add_argument("--specs", nargs="+", required=True, help="each reference's spec.json")
    sf.add_argument("-o", "--out", help="where to write the draft (default: stdout)")

    c = sub.add_parser("check", help="a questions.json asks about behavior, not design")
    c.add_argument("requirements", help="specification/questions.json")
    c.add_argument(
        "--terms",
        nargs="*",
        default=None,
        help="the words to forbid; default: the reference systems' names and axes.json's "
        "`mechanisms`, read from the specification/ folder beside questions.json",
    )

    args = ap.parse_args(argv)

    if args.cmd == "seed-from-specs":
        with open(args.axes, encoding="utf-8") as f:
            axes_doc = json.load(f)
        specs: Dict[str, Any] = {}
        for path in args.specs:
            # system name = the spec's parent directory (output/<system>/spec.json)
            name = os.path.basename(os.path.dirname(os.path.abspath(path)))
            with open(path, encoding="utf-8") as f:
                specs[name] = json.load(f)
        doc = seed_from_specs(axes_doc, specs)
        text = json.dumps(doc, indent=2) + "\n"
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
            with_values = sum(1 for q in doc["questions"] if q.get("value_space"))
            print(
                f"drafted {len(doc['questions'])} slot(s); {with_values} carry a value-space "
                f"from {len(specs)} system(s): {', '.join(specs)} -> {args.out}"
            )
        else:
            print(text, end="")
        return 0

    # check
    with open(args.requirements, encoding="utf-8") as f:
        doc = json.load(f)
    terms = args.terms if args.terms is not None else terms_from(Path(args.requirements).parent)
    problems = validate(doc)
    leaks = lint(doc, terms) if not problems else []  # lint needs a well-formed document
    for p in problems:
        print(f"INVALID: {p}")
    for lk in leaks:
        print(
            f"DESIGN-LEAK: question {lk['question_id']} {lk['where']} mentions {lk['term']!r} -> {lk['text']!r}"
        )
    if problems or leaks:
        print(f"\nFAILED: {len(problems)} structural, {len(leaks)} design-leak issue(s).")
        return 1
    print(
        f"OK: {len(doc['questions'])} property questions; none of {len(terms)} design word(s) appears."
    )
    return 0


if __name__ == "__main__":
    from .paths import run_cli

    raise SystemExit(run_cli(main, "requirements"))
