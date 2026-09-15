"""The tests a domain's knowledge base keeps across runs: ~/.skydiscover/<domain>/tests/.

lookup suggests tests to validate (exact id = candidate, keyword = related); sync copies a
finished run's suite in, test.sh included; record adds one entry. The index (tests/index.json) is
{"label": <human spelling>, "tests": [{id, property, keywords, source}]}.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import TEST_SCRIPT, Domain, Run, test_files, test_id

# A header line shorter than this, or a single word, is a tag or a file name, not a property statement.
MIN_STATEMENT_LEN = 12


def _store(domain: str) -> Domain:
    return Domain(domain)


def _load(domain: str) -> Dict[str, Any]:
    """Read a domain's index. A missing file means no tests are kept yet. A file that exists but does
    not parse is an error rather than an empty index: reading it as empty would hide a damaged
    index and re-author every test."""
    index = _store(domain).tests_index
    try:
        text = index.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"label": domain, "tests": []}
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"kept tests index is corrupt ({index}): {e}") from e
    if not isinstance(doc, dict) or not isinstance(doc.get("tests"), list):
        raise ValueError(f"kept tests index is corrupt ({index}): expected {{label, tests}}")
    return doc


@contextmanager
def _lock(domain: str):
    """Serialize the index's read-modify-write across processes (two runs may finish
    concurrently). An exclusive advisory lock on a hidden sidecar, same pattern as the findings
    store, so concurrent writers never lose each other's rows."""
    store = _store(domain)
    store.tests.mkdir(parents=True, exist_ok=True)
    with open(store.tests / ".index.lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _tokens(text: str) -> set:
    return set(re.findall(r"[^\W_]+", (text or "").lower()))  # words in any script


def _norm_id(s: str) -> str:
    """Normalize a test/requirement id for matching: lowercase and unify runs of separators to one
    _, so a hyphenated requirement id (no-false-negative) matches an underscored test id
    (no_false_negative). Domain-neutral -- only case/separators are touched, never any content.
    """
    return re.sub(r"[\W_]+", "_", (s or "").lower()).strip("_")


def _same_bytes(a: Path, b: Path) -> bool:
    """True iff two files hold identical bytes; sync skips an unchanged test and refreshes a
    changed one."""
    try:
        return a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def tests_for(domain: str) -> List[Dict[str, Any]]:
    """A domain's kept tests. The domain is slugged, so "My Domain", "my-domain" and "my  domain"
    resolve to one folder."""
    return _load(domain).get("tests", [])


def _row_path(domain: str, row: Dict[str, Any]) -> Optional[Path]:
    """Absolute path to a row's test file, or None if the file is not there (a dangling row is
    never offered for reuse).
    """
    tests_dir = _store(domain).tests
    src = row.get("source")
    if src:
        p = tests_dir / Path(src).name
        return p if p.is_file() else None
    rid = str(row.get("id", ""))
    if not rid:
        return None
    return next((p for p in test_files(tests_dir) if test_id(p) == rid), None)


def lookup(domain: str, required: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Match required [{id, text}] against the domain's kept tests. Every requirement is a gap
    until a test is validated in this run; a kept test with the same id, or a keyword overlap, is
    offered as a candidate to validate first. Returns {domain, candidates, gaps}.
    """
    tests = tests_for(domain)
    candidates, gaps = [], []
    for req in required:
        gaps.append(req)
        rid = _norm_id(req.get("id", ""))
        exact = next((g for g in tests if _norm_id(g.get("id", "")) == rid), None)
        if exact is not None and _row_path(domain, exact) is not None:
            candidates.append({"requirement": req, "test": exact, "match": "id"})
            continue
        rtoks = _tokens(req.get("id", "")) | _tokens(req.get("text", ""))
        best, best_overlap = None, 0
        for g in tests:
            overlap = len(rtoks & (_tokens(g["id"]) | set(g.get("keywords", []))))
            if overlap > best_overlap:
                best, best_overlap = g, overlap
        if best is not None and best_overlap >= 2:
            candidates.append(
                {"requirement": req, "test": best, "match": "keywords", "overlap": best_overlap}
            )
    return {"domain": domain, "candidates": candidates, "gaps": gaps}


def record(domain: str, row: Dict[str, Any], *, overwrite: bool = False) -> bool:
    """Add one test to the index, deduped by id; the file must already be saved. overwrite=True
    replaces an existing row. Returns True iff the index changed.
    """
    f = _row_path(domain, row)
    if f is None:
        return False  # no saved file -> refuse a dangling row
    store = _store(domain)
    row = {**row, "source": f.name}
    with _lock(domain):  # re-read under the lock so a concurrent writer's rows are never lost
        idx = _load(domain)
        idx.setdefault("label", domain)  # keep the human-readable spelling for display
        d = idx.setdefault("tests", [])
        for i, g in enumerate(d):
            if g["id"] == row["id"]:
                if not overwrite or g == row:
                    return False  # already present (unchanged, or overwrite not requested)
                d[i] = row
                break
        else:
            d.append(row)
        tmp = store.tests_index.with_suffix(  # unique per writer, so concurrent saves never collide
            f".json.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        tmp.write_text(json.dumps(idx, indent=2) + "\n", encoding="utf-8")
        tmp.replace(store.tests_index)
    return True


def _property_from_header(cc: Path) -> str:
    """Best-effort property statement: the first meaningful line of the file's top comment block."""
    try:
        for raw in cc.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if re.match(r"#\s*(include|pragma|define|if|endif)\b", line):
                continue  # a C preprocessor line is neither comment nor the end of the header
            marker = next((m for m in ("//", "#", "/*", "*/", "*") if line.startswith(m)), "")
            if not marker:
                if line:  # first non-comment code line ends the header
                    break
                continue
            text = line.removeprefix(marker).removesuffix("*/").strip()
            # skip bare markers like "DELIVERY CHECK:" / "REQUIREMENT" and pick the first line with substance
            body = re.sub(r"^(TEST|REQUIREMENT|WHY[^:]*)\s*[:.]?\s*", "", text, flags=re.I).strip()
            # a header line that is just the file's own name (optionally + a parenthetical tag) is
            # a title, not a property statement -- keep scanning.
            titleless = re.sub(r"\s*\([^)]*\)\s*$", "", body).strip()
            if re.fullmatch(r"[\w.\-]+\.\w+", titleless):
                continue
            # a real statement has words with spaces; skip a bare "Delivery check: <id>" title line
            if len(body) >= MIN_STATEMENT_LEN and " " in body:
                return body[:200]
    except OSError:
        pass
    return ""


def sync(run_dir: Path, domain: str, *, refresh_bodies: bool = False) -> Dict[str, Any]:
    """Copy every test in the run's suite, its test.sh, and its helper subdirectories into the
    domain's tests/ and record each test, deduped by id. An unchanged test is skipped. A test whose
    BODY changed is held back unless refresh_bodies is passed: the kept copy is what earlier runs
    validated, and silently replacing it would hand a weakened test to every later run in the
    domain. test.sh and the helpers are always the latest run's. Returns {recorded, refreshed,
    held, skipped, suite}.
    """
    suite = Run(run_dir).tests
    dest = _store(domain).tests
    ccs = test_files(suite)
    script = suite / TEST_SCRIPT
    if ccs and script.is_file():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script, dest / TEST_SCRIPT)
        for sub in suite.iterdir():
            if sub.is_dir() and not sub.name.startswith(".") and sub.name != "__pycache__":
                shutil.copytree(
                    sub,
                    dest / sub.name,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", ".*"),
                )
    # Two files with one id would lose one of them, so refuse.
    by_id: Dict[str, str] = {}
    for cc in ccs:
        gid = test_id(cc)
        if gid in by_id and by_id[gid] != cc.name:
            raise ValueError(
                f"test id collision in {suite}: {by_id[gid]!r} and {cc.name!r} both map to id "
                f"{gid!r}; rename one so every test has a distinct id (never drop a test)"
            )
        by_id[gid] = cc.name
    current = {g["id"]: g for g in tests_for(domain)}
    recorded, refreshed, held, skipped = [], [], [], []
    for cc in ccs:
        gid = test_id(cc)
        libfile = dest / cc.name
        row = {
            "id": gid,
            "property": _property_from_header(cc),
            "keywords": sorted(_tokens(gid)),
            "source": libfile.name,
        }
        same_file = libfile.is_file() and _same_bytes(cc, libfile)
        if gid in current and same_file and current[gid] == row:
            skipped.append(gid)  # test bytes and index row are already current
            continue
        is_refresh = gid in current
        if is_refresh and not same_file and not refresh_bodies:
            # The kept body is the one earlier runs validated; a changed body is not promoted on
            # the run's say-so. Re-validate it, then finish with --refresh-tests to replace it.
            held.append(gid)
            continue
        if not same_file:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cc, libfile)  # copy a new test, or deliberately replace a body
        record(domain, row, overwrite=True)  # keep the index row in sync with the file
        (refreshed if is_refresh else recorded).append(gid)
        current[gid] = row
    return {
        "recorded": recorded,
        "refreshed": refreshed,
        "held": held,
        "skipped": skipped,
        "suite": [test_id(c) for c in ccs],
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m skydiscover.synthesize.spec.kept_tests",
        description="The tests a domain has kept from earlier runs (~/.skydiscover/<domain>/tests/). "
        "`run finish` adds to it; the evaluator looks it up before writing a test.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser(
        "lookup", help="which required properties already have a test, and which need one"
    )
    lp.add_argument("domain")
    lp.add_argument("--props", nargs="+", required=True, help="required property ids (id[:text])")
    ls = sub.add_parser("list", help="every test the domain has kept")
    ls.add_argument("domain")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for g in tests_for(args.domain):
            print(f"  {g['id']:20} {g.get('source', '')}")
        return 0

    required = []
    for p in args.props:
        rid, _, text = p.partition(":")
        required.append({"id": rid, "text": text or rid})
    r = lookup(args.domain, required)
    by_req = {c["requirement"]["id"]: c for c in r["candidates"]}
    print(
        f"{len(required)} requirement(s): {len(by_req)} with a kept test to validate first, "
        f"{len(required) - len(by_req)} to write"
    )
    for g in r["gaps"]:
        c = by_req.get(g["id"])
        if c is None:
            print(f"  WRITE      {g['id']}")
        elif c["match"] == "id":
            print(f"  CANDIDATE  {g['id']:20} -> {c['test']['id']} (same id; validate first)")
        else:
            print(
                f"  CANDIDATE  {g['id']:20} -> {c['test']['id']} (check meaning and validate first)"
            )
    return 0


if __name__ == "__main__":
    from .paths import run_cli

    raise SystemExit(run_cli(main, "kept_tests"))
