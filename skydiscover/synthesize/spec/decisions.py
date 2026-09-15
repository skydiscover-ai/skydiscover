"""The decision log: every question, its answer, and who decided it.

The lead is a coding agent the user is talking to. It asks the user the few questions the sources
and the trace cannot settle, in the chat, and records what the user said here; everything else it
decides itself and records as its own. Every answer is one row in <run>/decision_log.json (the
Findings store, findings.py); specification/answers.json, the file build.py reads, is
exported from the log after every change. One rule holds it together: the user's answer is
never overwritten by an AI answer, in the run or in the knowledge base.

    decisions <run> answer                   answer every open question with its default, as the AI
    decisions <run> list                     every decision, who made it, and where it stands
    decisions <run> set <row|id> "<value>"   record an answer (yours; `--by ai` for the agent's own)
    decisions <run> drop <row|id>            set a row aside (--by human when it is the user's own call)

Nothing else is needed: `run finish` carries the user's answers and confirmed hacks into the domain's knowledge base.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .findings import Finding, Findings, new_id
from .paths import Domain, Run

SOURCE = "question"  # the `source` of every row that answers a question from questions.json

_STATUS = {
    "open": "AI answer",
    "confirmed": "your answer",
    "waived": "set aside",
    "proposed": "draft",
}


# ---------------------------------------------------------------------------------- questions


def question_id(question: Dict[str, Any]) -> str:
    """The row id of a question: stable across edits to its wording when it carries an `id`."""
    return new_id("question", str(question.get("id") or question.get("q") or ""))


def _question_context(run: Run, question: dict) -> str:
    definition = {k: question.get(k) for k in ("id", "axis", "q", "options", "multi")}
    definition["project"] = str(run.path.resolve().parent)
    return hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()


def _questions(run: Run) -> List[Dict[str, Any]]:
    if not run.questions.is_file():
        return []
    doc = json.loads(run.questions.read_text(encoding="utf-8"))
    questions = doc.get("questions") if isinstance(doc, dict) else None
    if not isinstance(questions, list) or any(
        not isinstance(q, dict) or not isinstance(q.get("q"), str) or not q["q"].strip()
        for q in questions
    ):
        raise ValueError(
            f'{run.questions}: expected an object with a "questions" list; each question needs text in "q". Run `requirements check` before recording answers.'
        )
    return questions


def _default(question: Dict[str, Any]) -> str:
    """The default discovery declared for the question, or "" when it declared none: the
    first option listed is not an answer, and an empty answer leaves the question open."""
    return str(question.get("default") or "")


def _row(question: Dict[str, Any], value: str, by: str, note: str) -> Finding:
    return Finding(
        id=question_id(question),
        title=str(question.get("q", "")),
        kind="spec",
        detail=value,
        status="confirmed" if by == "human" else "open",
        asked=True,
        answered_by=by,
        note=note,
        source=SOURCE,
    )


# ---------------------------------------------------------------------------------- the verbs


def answer(run: Run) -> List[Finding]:
    """Answer every question that has no row yet with its declared default, as the AI; a question
    with no default gets an empty answer, which exports no requirement and so stays open for the
    user. A matching saved human answer takes precedence over an AI default. Returns changed rows.
    """
    store = Findings(run.decision_log, track_severity=True)
    have = {f.id: f for f in store.all()}
    settled = _settled_by_user(run)
    added = []
    for q in _questions(run):
        qid = question_id(q)
        if qid in have and (have[qid].answered_by == "human" or qid not in settled):
            continue
        if qid in settled:
            row = settled[qid]  # the user's answer from an earlier run in this domain
        else:
            row = _row(
                q, _default(q), "ai", "declared default" if _default(q) else "no default declared"
            )
            row.question_context = _question_context(run, q)
        store.upsert(row)
        added.append(row)
    export(run)
    return added


def _settled_by_user(run: Run) -> Dict[str, Finding]:
    """The questions the user answered in earlier runs of this domain, from the knowledge base's
    decisions.json; reuse needs the same project, question, and options."""
    try:
        slug = run.domain()
    except ValueError:  # a reserved name in task.md's domain: line
        return {}
    if not slug:
        return {}
    kb = Domain(slug).decisions
    if not kb.is_file():
        return {}
    contexts = {_question_context(run, q): question_id(q) for q in _questions(run)}
    settled = {}
    for f in Findings(kb).all():
        if f.source == SOURCE and f.answered_by == "human" and f.detail and f.status == "confirmed":
            if f.question_context in contexts:
                qid = contexts[f.question_context]
                settled[qid] = replace(f, id=qid)
            elif f.id in {question_id(q) for q in _questions(run)}:
                print(f"Previous answer needs confirmation: {f.title}", file=sys.stderr)
    return settled


def set_answer(run: Run, target: str, value: str, *, by: str = "human", note: str = "") -> Finding:
    """Record `value` as the answer. `target` is a row number from `list`, a question id, or a
    row id; a question with no row yet gets one. The user's answer is never overwritten by an AI one.
    """
    store = Findings(run.decision_log, track_severity=True)
    current, question = _find(store, run, target)
    _guard(current, by, target)
    definition = question or next(
        (q for q in _questions(run) if current and question_id(q) == current.id), None
    )
    if definition is not None:
        _guard(_settled_by_user(run).get(question_id(definition)), by, target)
    if question is not None:
        row = _row(question, value, by, note or f"{by} answer")
    elif current is not None:
        row = Finding(
            id=current.id,
            title=current.title,
            kind=current.kind,
            detail=value,
            status="confirmed" if by == "human" else "open",
            asked=True,
            answered_by=by,
            note=note or f"{by} answer",
            severity=current.severity,
            test=current.test,
            mutant=current.mutant,
            source=current.source,
        )
    else:
        raise ValueError(f"nothing matches {target!r} (see `list`, or use a question id)")
    if definition is not None:
        row.question_context = _question_context(run, definition)
    store.upsert(row)
    export(run)
    return row


def drop(run: Run, target: str, *, by: str = "ai") -> Finding:
    """Drop an answer from the specification. The row stays in the log, marked set aside."""
    store = Findings(run.decision_log, track_severity=True)
    current, _ = _find(store, run, target)
    if current is None:
        raise ValueError(f"nothing matches {target!r} (see `list`)")
    _guard(current, by, target)
    store.set_status(current.id, "waived", who=by)
    export(run)
    return next(f for f in store.all() if f.id == current.id)


def export(run: Run) -> Path:
    """Write specification/answers.json: the current answer to each question, and who gave it.
    Dropped and empty answers are left out, as is a row whose question is gone."""
    rows = {f.id: f for f in Findings(run.decision_log, track_severity=True).all()}
    decisions = []
    for q in _questions(run):
        f = rows.get(question_id(q))
        if f is None or f.status == "waived" or not f.detail:
            continue
        if f.question_context != _question_context(run, q):
            print(
                f"Question changed; confirm its answer again: {q.get('q', q.get('id'))}",
                file=sys.stderr,
            )
            continue
        decisions.append(
            {
                "id": q.get("id"),
                "axis": q.get("axis") or q.get("id"),
                "q": q.get("q"),
                "chosen": f.detail,
                "answered_by": f.answered_by,
                "rationale": f.note,
            }
        )
    run.answers.parent.mkdir(parents=True, exist_ok=True)
    run.answers.write_text(json.dumps({"decisions": decisions}, indent=2) + "\n", encoding="utf-8")
    return run.answers


# ---------------------------------------------------------------------------------- knowledge base


def kept_decisions(run_dir: Path, domain: Optional[str] = None) -> Findings:
    """The knowledge base this run feeds: ~/.skydiscover/<domain>/decisions.json. The domain is the
    `domain:` line of the run's task.md front matter unless given explicitly."""
    slug = domain or Run(run_dir).domain()
    if not slug:
        raise ValueError(
            f"no domain for {run_dir}: put `domain: <name>` in the front matter of "
            f"{Run(run_dir).task}, or pass the domain explicitly"
        )
    return Findings(Domain(slug).decisions)


def save(log: Findings, knowledge_base: Findings) -> Tuple[int, int]:
    """Copy what outlives the run into the knowledge base: every answer the user gave (reused only in
    matching context) and every active reward hack (a confirmed reward hack is workload-agnostic knowledge).
    Idempotent by id; the store keeps an AI row from overwriting the user's.
    Called by `run finish`. Returns (written, picked)."""
    picked = [
        f
        for f in log.all()
        if f.answered_by == "human" or (f.kind == "hack" and f.status in ("open", "confirmed"))
    ]
    written = sum(
        1
        for f in picked
        if knowledge_base.upsert(
            replace(f, id=new_id("saved-question", f.question_context, f.id))
            if f.source == SOURCE and f.question_context
            else f
        )
    )
    return written, len(picked)


# ---------------------------------------------------------------------------------- helpers


def _find(store: Findings, run: Run, target: str):
    """(row, question) for a target: a row number, a row id, or a question id. `question` is set
    only when the question has no row yet."""
    rows = store.all()
    if target.isdigit():
        n = int(target)
        if not 1 <= n <= len(rows):
            raise ValueError(f"no row {n}; the log has {len(rows)} (see `list`)")
        return rows[n - 1], None
    for f in rows:
        if f.id == target:
            return f, None
    for q in _questions(run):
        if str(q.get("id")) == target or question_id(q) == target:
            row = next((f for f in rows if f.id == question_id(q)), None)
            return row, (None if row else q)
    return None, None


def _guard(current: Optional[Finding], by: str, target: str) -> None:
    if current is not None and current.answered_by == "human" and by != "human":
        raise ValueError(f"row {target} is the user's answer; only the user may change it")


def _show(store: Findings) -> None:
    rows = store.all()
    if not rows:
        print("(the decision log is empty)")
        return
    width = max(len(f.title) for f in rows)
    for i, f in enumerate(rows, 1):
        who = "you" if f.answered_by == "human" else "AI "
        print(
            f"{i:>3}. [{who}] {f.title.ljust(width)}  → {f.detail}  ({_STATUS.get(f.status, f.status)})"
        )


def _label(f: Finding) -> str:
    return f"{f.title} → {f.detail}  ({_STATUS.get(f.status, f.status)})"


# ---------------------------------------------------------------------------------- cli


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m skydiscover.synthesize.spec.decisions",
        description=(__doc__ or "").split("\n\n")[0],
    )
    ap.add_argument("run", help="the run directory, e.g. .skydiscover/<slug>")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("answer", help="answer every open question with its default option, as the AI")
    sub.add_parser("list", help="every decision, who made it, and where it stands")

    p = sub.add_parser("set", help="record an answer: yours by default, the agent's with --by ai")
    p.add_argument("row", help="a row number from `list`, or a question id from questions.json")
    p.add_argument("value", help="the chosen option")
    p.add_argument("--note", default="", help="why")
    _by(p)

    p = sub.add_parser(
        "drop", help="set a row aside: an answer leaves the specification, a defect stops blocking"
    )
    p.add_argument("row", help="a row number from `list`, or a question id")
    # Unlike `set`, whose default caller is transcribing the user, `drop` closes defects, so the
    # default attribution is the agent's own: an AI-dropped defect still needs a proven fix before
    # release, while `--by human` records the user's decision and closes it outright. Recording a
    # user's decision is always the explicit act, never the default.
    p.add_argument(
        "--by",
        choices=("human", "ai"),
        default="ai",
        help="who is setting the row aside: ai (default; a dropped defect still needs a proven "
        "fix) or human (the user's own decision, closes a defect outright — only for words the "
        "user actually said)",
    )

    args = ap.parse_args(argv)
    run = Run(args.run)

    if args.cmd == "answer":
        added = answer(run)
        for f in added:
            print(f"answered: {_label(f)}")
        print(f"{len(added)} answered; answers.json → {run.answers}")
    elif args.cmd == "list":
        _show(Findings(run.decision_log, track_severity=True))
    elif args.cmd == "set":
        print(f"set: {_label(set_answer(run, args.row, args.value, by=args.by, note=args.note))}")
    elif args.cmd == "drop":
        print(f"dropped: {_label(drop(run, args.row, by=args.by))}")
    return 0


def _by(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--by",
        choices=("human", "ai"),
        default="human",
        help="who is deciding: human (default; what the user said) or ai (the agent's own call)",
    )


if __name__ == "__main__":
    from .paths import run_cli

    raise SystemExit(run_cli(main, "decisions"))
