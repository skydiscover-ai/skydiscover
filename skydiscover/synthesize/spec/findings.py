"""The findings file: a JSON list of spec gaps and reward hacks found during a run.

A finding is confirmed only when a human answered it; other claims are downgraded on load. Writes
are atomic, and a file that does not parse raises rather than reading as empty.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

# kind -> where the fix belongs. Only "spec" reaches the spec.
_TARGETS = {
    "spec": "spec",  # a missing requirement
    "hack": "test",  # a reward-hack -> forbid it with a test
    "overfit": "eval",  # overfits the trace -> broaden the workload
    "measure": "report",  # measurement / reporting integrity
}

_KINDS = tuple(_TARGETS)
_SEVERITIES = ("advisory", "defect")
# The full status vocabulary. An AI draft awaiting a human is "proposed"; "waived" is closed.
_STATUSES = ("open", "proposed", "confirmed", "waived")
# A spec finding is enforced when it is machine evidence ("open") or a human
# answer ("confirmed"). An AI draft awaiting a human is "proposed"; "waived" is closed.
_ACTIVE = ("open", "confirmed")


class FindingsUnreadable(RuntimeError):
    """The store file exists but cannot be read as a findings list.

    Distinct from "no findings": a truncated file read as empty would pass a build that has open
    defects, so every read raises instead of returning [].
    """


class FindingsWriteRefused(RuntimeError):
    """A save that would have destroyed rows already on disk, refused before writing."""


def new_id(*parts: str) -> str:
    """A stable id from content. The \\x1f delimiter is stripped from each part first,
    so untrusted text containing it cannot collide two distinct findings onto one id."""
    return hashlib.sha1(
        "\x1f".join(p.replace("\x1f", "") for p in parts).encode("utf-8")
    ).hexdigest()[:12]


@dataclass
class Finding:
    id: str
    title: str  # short label
    kind: str  # one of _TARGETS
    detail: str  # the requirement, or what's wrong
    status: str = "open"  # open | proposed | confirmed | waived
    asked: bool = False  # did a person/AI decide this?
    answered_by: str = ""  # "human" or "ai", when asked
    note: str = ""  # the chosen answer, or the rationale
    severity: str = (
        "advisory"  # advisory | defect -- a defect blocks the release until closed (waived)
    )
    # A defect the AI closes as fixed must name the test that passes the build and the broken
    # implementation that test catches (check_release.py), so an unrelated test cannot be cited.
    test: str = ""  # test file that shows this defect's fix
    mutant: str = ""  # broken implementation that exhibits this defect
    source: str = ""  # "question" for a row answering questions.json; else the role that added it

    question_context: str = ""  # matching question and project; older records need confirmation

    def __post_init__(self) -> None:
        if self.kind not in _TARGETS:
            raise ValueError(f"unknown kind {self.kind!r}; expected one of {tuple(_TARGETS)}")
        # Absent severity is advisory; any present value other than 'advisory' is a defect, so a typo blocks
        # production-ready rather than slipping through.
        if self.severity is None:
            self.severity = "advisory"
        sev = str(self.severity).strip().lower()
        self.severity = "advisory" if sev == "advisory" else "defect"
        # Unknown status normalizes to "open", so a typo cannot drop a finding from the active set.
        st = "open" if self.status is None else str(self.status).strip().lower()
        self.status = st if st in _STATUSES else "open"
        # only a human answer may be "confirmed" -- anything else is a draft.
        if self.status == "confirmed" and self.answered_by != "human":
            self.status = "proposed"

    @property
    def target(self) -> str:
        return _TARGETS[self.kind]

    @property
    def is_open_defect(self) -> bool:
        """A defect-severity finding not yet closed (closed -> 'waived'). Blocks the release."""
        return self.severity == "defect" and self.status != "waived"


SEVERITY_SNAPSHOT = ".severity_snapshot.json"


def snapshot_path(log: Path | str) -> Path:
    """The hidden record, beside a run's decision log, of the severity each finding had when it was
    first written. check_release.py compares against it so a defect cannot be quietly relabelled
    'advisory' before delivery; only a change the user made is accepted."""
    return Path(log).parent / SEVERITY_SNAPSHOT


class Findings:
    """The findings on disk, as a JSON list. Read it, add to it, save it. With `track_severity`
    (the run's decision log; not the knowledge base's decisions.json), every save also records each
    new id's severity in the snapshot beside the log."""

    def __init__(self, path: Path | str, *, track_severity: bool = False):
        self.path = Path(path)
        self.track_severity = track_severity

    @contextmanager
    def _lock(self):
        """Serialize read-modify-write across processes: the lead and the off-thread auditor both
        append to the shared findings log, so each add/upsert/set_status must re-read and write under
        an exclusive advisory lock (a hidden sidecar .lock file), or concurrent writers would lose each
        other's updates."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f".{self.path.name}.lock")  # hidden, like tests/.index.lock
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def all(self) -> list[Finding]:
        """Every finding on disk. An absent file is an empty store ([]). A file that is present but
        unreadable raises FindingsUnreadable: it may hold open defects and the user's answers, so no
        caller may proceed as if the store were empty."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []  # never written yet -- legitimately empty
        except OSError as e:
            raise FindingsUnreadable(f"cannot read findings at {self.path}: {e}") from e
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise FindingsUnreadable(f"findings at {self.path} are not valid JSON: {e}") from e
        if not isinstance(data, list):
            raise FindingsUnreadable(
                f"findings at {self.path} are a {type(data).__name__}, not a JSON list"
            )
        out: list[Finding] = []
        for d in data:
            try:  # Finding.__post_init__ re-checks who may be "confirmed" on every read
                out.append(Finding(**{k: d[k] for k in Finding.__dataclass_fields__ if k in d}))
            except (TypeError, ValueError):
                continue  # skip a malformed entry, never crash
        return out

    def add(self, finding: Finding) -> bool:
        """Add a finding. A duplicate id is a no-op (used for seeds, derivables, hacks)."""
        with self._lock():
            items = self.all()
            if any(f.id == finding.id for f in items):
                return False
            items.append(finding)
            self._save(items)
            return True

    def upsert(self, finding: Finding) -> bool:
        """Add, or replace an existing row by id -- so re-answering a question updates one
        row instead of duplicating. An AI answer never overrides the user's."""
        with self._lock():
            items = self.all()
            for i, existing in enumerate(items):
                if existing.id == finding.id:
                    if existing.answered_by == "human" and finding.answered_by != "human":
                        return False
                    if existing == finding:
                        return False  # already present, unchanged -- idempotent no-op
                    items[i] = finding
                    self._save(items)
                    return True
            items.append(finding)
            self._save(items)
            return True

    def set_status(
        self,
        finding_id: str,
        status: str,
        note: str = "",
        who: str = "",
        test: str = "",
        mutant: str = "",
    ) -> bool:
        """Update a finding's status (and, if given, who decided it, and the test and mutant that
        prove a fix). The same rule is re-applied, so set_status can never confirm a non-human
        answer."""
        with self._lock():
            items = self.all()
            found = False
            for f in items:
                if f.id == finding_id:
                    if f.answered_by == "human" and who != "human":
                        continue  # only a human may change a human-decided finding (empty/ai who cannot)
                    st = str(status).strip().lower()
                    f.status = (
                        st if st in _STATUSES else "open"
                    )  # same vocabulary guard as __post_init__
                    if note:
                        f.note = note
                    if test:
                        f.test = test
                    if mutant:
                        f.mutant = mutant
                    if who:
                        f.answered_by = who
                        f.asked = True
                    if f.status == "confirmed" and f.answered_by != "human":
                        f.status = "proposed"
                    found = True
            if found:
                self._save(items)
            return found

    def for_spec(self) -> list[Finding]:
        """The real requirements: spec-routed and enforced (machine evidence or a human
        answer). An AI-proposed answer is excluded until a human confirms it."""
        return [f for f in self.all() if f.target == "spec" and f.status in _ACTIVE]

    def open_defects(self) -> list[Finding]:
        """Every unresolved defect-severity finding, regardless of kind. The release is blocked
        while this list is non-empty (check_release.py).
        """
        return [f for f in self.all() if f.is_open_defect]

    def _save(self, items: list[Finding], *, allow_shrink: bool = False) -> None:
        """Write atomically. Refuses a save that would lose recoverable rows unless allow_shrink; a file
        that no longer parses is side-renamed to <name>.corrupt.<pid> first.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not allow_shrink:
            self._guard_overwrite(len(items))
        tmp = self.path.with_suffix(  # unique per writer, so concurrent saves never collide
            self.path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )  # write-then-rename = atomic
        tmp.write_text(json.dumps([asdict(f) for f in items], indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)
        if self.track_severity:
            self._record_severities(items)

    def _record_severities(self, items: list[Finding]) -> None:
        """Add each id the snapshot has not seen, with the severity it was written with. An id
        already recorded keeps its first severity: that is the point of the record."""
        snap = snapshot_path(self.path)
        try:
            doc = json.loads(snap.read_text(encoding="utf-8"))
            was = doc["severities"] if isinstance(doc.get("severities"), dict) else {}
        except (OSError, ValueError, AttributeError):
            was = {}
        now = dict(was)
        for f in items:
            now.setdefault(f.id, f.severity)
        if now != was or not snap.is_file():
            snap.write_text(json.dumps({"severities": now}, indent=2) + "\n", encoding="utf-8")

    def _guard_overwrite(self, writing: int) -> None:
        """Refuse a write that would lose rows a read would recover. Counts recoverable rows the way all()
        does, so dropping junk rows is allowed and losing a valid row raises.
        """
        try:
            on_disk = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return  # first write
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            # The file became unreadable under us; keep its bytes beside it before the write proceeds.
            quarantine = self.path.with_name(f"{self.path.name}.corrupt.{os.getpid()}")
            try:
                os.replace(self.path, quarantine)
            except OSError:
                raise FindingsUnreadable(
                    f"findings at {self.path} are unreadable ({e}) and cannot be set aside"
                ) from e
            return
        if not isinstance(on_disk, list):
            return  # a non-list file cannot be shrunk row-wise; all() raises on it at read time
        valid = 0
        for d in on_disk:
            try:  # the exact recovery all() performs, so this count matches what a read returns
                Finding(**{k: d[k] for k in Finding.__dataclass_fields__ if k in d})
                valid += 1
            except (TypeError, ValueError):
                continue
        if valid < len(on_disk):
            # Junk rows on disk are dropped by this write; set their bytes aside once, best-effort.
            sidecar = self.path.with_name(f"{self.path.name}.malformed.{os.getpid()}")
            try:
                if not sidecar.exists():
                    sidecar.write_text(json.dumps(on_disk, indent=2) + "\n", encoding="utf-8")
            except OSError:
                pass
        if writing < valid:
            raise FindingsWriteRefused(
                f"refusing to write {writing} finding(s) over the {valid} recoverable on disk at "
                f"{self.path}: a read-modify-write that shrinks the store would drop the user's answers"
            )


def _cmd_add(args: argparse.Namespace) -> int:
    """Append one finding. The id defaults to a content hash so re-logging is an idempotent no-op."""
    fid = args.id or new_id(args.title, args.kind, args.detail)
    finding = Finding(
        id=fid,
        title=args.title,
        kind=args.kind,
        detail=args.detail,
        status=args.status,
        answered_by=args.answered_by,
        asked=bool(args.answered_by),
        note=args.note,
        severity=args.severity,
        test=args.test,
        mutant=args.mutant,
        source=args.source,
    )
    from .paths import resolve_decision_log

    added = Findings(str(resolve_decision_log(args.run)), track_severity=True).add(finding)
    print(fid if added else f"{fid} (exists)")
    return 0


def _cmd_set_status(args: argparse.Namespace) -> int:
    """Move an existing finding to one of the known statuses -- the only allowed way to waive/confirm."""
    from .paths import resolve_decision_log

    ok = Findings(str(resolve_decision_log(args.run)), track_severity=True).set_status(
        args.id, args.status, note=args.note, who=args.who, test=args.test, mutant=args.mutant
    )
    print("ok" if ok else "not found")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """The write path for the findings file; every value is normalized through Finding."""
    ap = argparse.ArgumentParser(
        prog="python3 -m skydiscover.synthesize.spec.findings",
        description="Write findings to the run's decision log through the store (never hand-edit the JSON).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="append one finding (dedup by id)")
    a.add_argument(
        "run", help="the run dir, e.g. .skydiscover/<slug> (the decision log is derived)"
    )
    a.add_argument("--title", required=True, help="short label")
    a.add_argument("--kind", required=True, choices=_KINDS, help="what the finding is about")
    a.add_argument("--detail", required=True, help="the requirement, or what is wrong")
    a.add_argument("--status", default="open", choices=_STATUSES)
    a.add_argument("--severity", default="advisory", choices=_SEVERITIES)
    a.add_argument(
        "--answered-by",
        dest="answered_by",
        default="ai",
        help="who decided this row: ai (default; the agent is the one running this) | human",
    )
    a.add_argument("--note", default="", help="the chosen answer, or the rationale")
    a.add_argument(
        "--test", default="", help="the kept test that shows the fix (to close a defect)"
    )
    a.add_argument(
        "--mutant", default="", help="the broken copy that test catches (to close a defect)"
    )
    a.add_argument(
        "--source",
        default="",
        help="the role adding the row (spec-builder, evaluator, auditor, ...)",
    )
    a.add_argument("--id", default="", help="override the content-hash id")
    a.set_defaults(fn=_cmd_add)

    s = sub.add_parser("set-status", help="move a finding to one of the known statuses")
    s.add_argument(
        "run", help="the run dir, e.g. .skydiscover/<slug> (the decision log is derived)"
    )
    s.add_argument("--id", required=True, help="the finding id")
    s.add_argument("--status", required=True, choices=_STATUSES)
    s.add_argument("--note", default="", help="the rationale")
    s.add_argument("--who", default="", help="human | ai (only a human may confirm)")
    s.add_argument(
        "--test", default="", help="to close a fixed defect: the kept test that shows the fix"
    )
    s.add_argument(
        "--mutant", default="", help="to close a fixed defect: the broken copy that test catches"
    )
    s.set_defaults(fn=_cmd_set_status)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    from .paths import run_cli

    raise SystemExit(run_cli(main, "findings"))
