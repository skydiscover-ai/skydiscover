"""Tests for the failure-history miner (spec/failure_patterns.py).

Pure-stdlib; loaded directly from its file so the suite runs under a bare ``python3``. The
commit-lane tests build real throwaway git repos; the GitHub API lane is exercised with
``_http_json`` monkeypatched, so the suite never touches the network.
"""

import importlib.util
import json
import pathlib
import subprocess

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PKG = _ROOT / "skydiscover" / "synthesize" / "spec"


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, _PKG / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mine = _load("failure_patterns.py", "_mine_mine")


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _repo(tmp_path, commits):
    """A throwaway git repo; commits = [(message, filename)]."""
    d = tmp_path / "clone"
    d.mkdir()
    _git(d, "init", "-q")
    for i, (msg, fname) in enumerate(commits):
        f = d / fname
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"rev {i}\n", encoding="utf-8")
        _git(d, "add", "-A")
        _git(d, "commit", "-q", "-m", msg)
    return d


# commit lane, offline


def test_offline_commit_lane_degrades_not_blocks(tmp_path):
    # The always-available floor: no gh, plain local history -> fix commits mined, housekeeping
    # excluded, tools recorded honestly, and the CLI exits 0.
    clone = _repo(
        tmp_path,
        [
            ("Fix race between flush and compaction on shutdown", "db/flush.cc"),
            ("docs: fix typo in readme", "README.md"),
            ("Add shiny new feature", "db/feature.cc"),
        ],
    )
    out = tmp_path / "run" / "specification" / "references" / "sys" / "failure_patterns.json"
    rc = mine.main(["repo", str(clone), "--name", "sys", "--out", str(out), "--offline"])
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    titles = [e["title"] for e in doc["entries"]]
    assert any("race between flush" in t for t in titles)
    assert not any("typo" in t for t in titles)
    assert doc["tools"] == {"github": "off", "auth": "none", "history": "full"}
    assert doc["entries"][0]["kind"] == "commit"
    assert doc["entries"][0]["id"].startswith("commit:")
    assert mine.selfcheck(doc) == []


def test_missing_clone_dir_is_exit_2(tmp_path):
    rc = mine.main(
        [
            "repo",
            str(tmp_path / "nope"),
            "--name",
            "x",
            "--out",
            str(tmp_path / "o.json"),
            "--offline",
        ]
    )
    assert rc == 2


def test_shallow_history_recorded(tmp_path):
    # A --depth 1 clone still mines what it has and records history: shallow, exit 0.
    src = _repo(
        tmp_path,
        [
            ("Fix deadlock in compaction scheduler", "db/sched.cc"),
            ("Fix crash on empty WAL replay", "db/wal.cc"),
        ],
    )
    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{src}", str(shallow)],
        check=True,
        capture_output=True,
    )
    doc = mine.mine_repo(shallow, "sys", use_api=False)
    assert doc["tools"]["history"] == "shallow"
    assert mine.selfcheck(doc) == []


# GitHub API lane (no gh needed)


def _search_payload(items):
    return {"items": items}


def _pr(n, title, body=""):
    return {
        "number": n,
        "title": title,
        "body": body,
        "html_url": f"https://github.com/o/s/pull/{n}",
        "closed_at": "2025-06-01T00:00:00Z",
        "pull_request": {},
    }


def _issue(n, title, body=""):
    return {
        "number": n,
        "title": title,
        "body": body,
        "html_url": f"https://github.com/o/s/issues/{n}",
        "closed_at": "2025-05-01T00:00:00Z",
    }


def test_api_lane_needs_no_gh_and_merges_prs_and_issues(tmp_path, monkeypatch):
    # The PR/issue half of the feature runs over plain HTTPS: no `gh`, no token. Same PR number
    # returned by two searches must dedup to one entry.
    clone = _repo(tmp_path, [("Fix race in flush", "a.cc")])
    monkeypatch.setattr(mine, "_token", lambda: (None, "none"))
    monkeypatch.setattr(mine, "_SEARCH_SLEEP_SECS", 0)
    calls = []

    def fake(url, token, timeout):
        calls.append(url)
        assert token is None  # unauthenticated path
        if "label%3Abug" in url and "is%3Apr" in url:
            return (
                _search_payload([_pr(1201, "Fix use-after-free in async callback", "dangling")]),
                "ok",
            )
        if "is%3Aissue" in url:
            return _search_payload([_issue(1050, "Memory leak in reader on protocol error")]), "ok"
        if "label%3Aregression" in url:
            return _search_payload([_pr(1201, "Fix use-after-free in async callback")]), "ok"
        return _search_payload([]), "ok"

    monkeypatch.setattr(mine, "_http_json", fake)
    monkeypatch.setattr(mine, "_identity", lambda d, n: ("o/s", "abc123def456", True))
    doc = mine.mine_repo(clone, "s")
    assert doc["tools"] == {"github": "ok", "auth": "none", "history": "full"}
    by_id = {e["id"]: e for e in doc["entries"]}
    assert by_id["pr:1201"]["kind"] == "pr" and by_id["pr:1201"]["tier"] == "merged-fix"
    assert by_id["issue:1050"]["kind"] == "issue" and by_id["issue:1050"]["tier"] == "labeled-issue"
    assert len([e for e in doc["entries"] if e["id"] == "pr:1201"]) == 1  # deduped
    assert not any("/pulls/" in u and "/files" in u for u in calls)  # unauth: no file fetches
    assert mine.selfcheck(doc) == []


def test_api_rate_limit_degrades_to_commit_lane(tmp_path, monkeypatch):
    clone = _repo(tmp_path, [("Fix deadlock in scheduler", "a.cc")])
    monkeypatch.setattr(mine, "_token", lambda: (None, "none"))
    monkeypatch.setattr(mine, "_SEARCH_SLEEP_SECS", 0)
    monkeypatch.setattr(mine, "_http_json", lambda url, token, timeout: (None, "rate-limited"))
    monkeypatch.setattr(mine, "_identity", lambda d, n: ("o/s", "abc123def456", True))
    doc = mine.mine_repo(clone, "s")
    assert doc["tools"]["github"] == "rate-limited"
    assert any("deadlock" in e["title"] for e in doc["entries"])  # commit lane still delivered
    assert mine.selfcheck(doc) == []


def test_token_discovery_prefers_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
    assert mine._token() == ("ghp_fake", "token")
    monkeypatch.delenv("GITHUB_TOKEN")
    monkeypatch.setenv("GH_TOKEN", "ghs_fake")
    assert mine._token() == ("ghs_fake", "token")


# relevance filter


def test_relevance_gate_include_exclude_defer():
    assert mine._relevant("Fix race in flush path", ["db/f.cc"]) is True
    assert mine._relevant("Prevent use-after-free in iterator", []) is True
    assert mine._relevant("docs: fix typo", []) is False
    assert mine._relevant("Improve wording", ["docs/guide.md", "README.md"]) is False  # docs-only
    assert mine._relevant("Refactor scheduler internals", ["db/s.cc"]) is None  # defer
    assert mine._relevant("Anything at all", [], labeled=True) is True  # labeled rows pass
    # a fix that changed source counts even when the title names no classic failure noun ...
    assert mine._relevant("Fix quota accounting for burst traffic", ["src/router.py"]) is True
    assert mine._relevant("Fix miscompile of nested loops", []) is True
    # ... but a fix with no failure noun and no file list is deferred, not guessed
    assert mine._relevant("Fix quota accounting for burst traffic", []) is None


def test_api_lane_is_only_asked_about_github_remotes(tmp_path, monkeypatch):
    clone = _repo(tmp_path, [("Fix deadlock in scheduler", "a.cc")])
    subprocess.run(
        ["git", "-C", str(clone), "remote", "add", "origin", "https://gitlab.com/o/s.git"],
        check=True,
        capture_output=True,
    )
    called = []
    monkeypatch.setattr(
        mine, "_http_json", lambda url, token, timeout: called.append(url) or (None, "failed")
    )
    doc = mine.mine_repo(clone, "s")
    assert called == [] and doc["tools"]["github"] == "off"
    assert doc["source"].startswith("o/s @ ")


# write-time invariants


def test_selfcheck_flags_tampered_counts(tmp_path):
    clone = _repo(tmp_path, [("Fix race in flush", "a.cc")])
    doc = mine.mine_repo(clone, "sys", use_api=False)
    assert mine.selfcheck(doc) == []
    doc["counts"]["scanned"] += 1  # break included+excluded+deferred == scanned
    assert any("scanned" in v for v in mine.selfcheck(doc))
    doc["counts"]["scanned"] -= 1
    doc["entries"][0]["pattern_category"] = "not-a-category"
    assert any("off-taxonomy" in v for v in mine.selfcheck(doc))


# untrusted text


def test_untrusted_text_sanitized_truncated_and_noted(tmp_path):
    evil = "Fix race in flush\n\n" + "\x07\x1b[31m" + "A" * 5000
    clone = _repo(tmp_path, [(evil, "a.cc")])
    doc = mine.mine_repo(clone, "sys", use_api=False)
    (entry,) = doc["entries"]
    assert len(entry["excerpt"]) <= 400
    assert "\x07" not in entry["excerpt"] and "\x1b" not in entry["excerpt"]
    assert "third-party text" in doc["note"]


# category heuristic


def test_category_heuristic_seed():
    assert mine._categorize("deadlock in compaction") == "concurrency-fault"
    assert mine._categorize("crash during WAL replay loses acked writes") == "recovery-fault"
    assert mine._categorize("hardcoded result returned for benchmark keys") == "fabricated-result"
    assert mine._categorize("miscellaneous cleanup") == "uncategorized"


# query


def _patterns_doc(system, entries):
    return {
        "system": system,
        "source": f"o/{system} @ c1",
        "mined_at": "2026-08-02T00:00:00+00:00",
        "tools": {"github": "off", "auth": "none", "history": "full"},
        "queries": [],
        "counts": {"scanned": len(entries), "included": len(entries), "excluded": 0, "deferred": 0},
        "note": mine.UNTRUSTED_NOTE,
        "entries": entries,
    }


def _e(eid, title, category, kw, date, excerpt=""):
    return {
        "id": eid,
        "kind": "commit",
        "url": "",
        "title": title,
        "excerpt": excerpt,
        "files": [],
        "date": date,
        "pattern_category": category,
        "category_by": "heuristic",
        "keywords": kw,
        "tier": "commit-message",
    }


def _run_with_patterns(tmp_path):
    run = tmp_path / "run"
    for sys_name, entries in (
        (
            "rocksdb",
            [
                _e(
                    "commit:aaa",
                    "Fix race between flush and compaction",
                    "concurrency-fault",
                    ["race"],
                    "2025-01-02",
                    excerpt="a race under concurrent flush",
                ),
                _e(
                    "commit:bbb",
                    "Fix fsync skipped on shutdown",
                    "skipped-guarantee",
                    ["fsync"],
                    "2025-01-03",
                ),
            ],
        ),
        (
            "redis",
            [
                _e(
                    "commit:ccc",
                    "Fix crash replaying truncated AOF",
                    "recovery-fault",
                    ["crash"],
                    "2025-01-04",
                    excerpt="race mentioned only in the excerpt",
                ),
            ],
        ),
    ):
        d = run / "specification" / "references" / sys_name
        d.mkdir(parents=True)
        (d / "failure_patterns.json").write_text(
            json.dumps(_patterns_doc(sys_name, entries)), encoding="utf-8"
        )
    return run


def test_query_ranking_title_beats_excerpt_and_category_boosts(tmp_path):
    run = _run_with_patterns(tmp_path)
    hits = mine.query_run(run, keywords=["race"])
    assert hits[0]["entry"]["id"] == "commit:aaa"  # title+keyword+excerpt beats excerpt-only
    assert hits[-1]["entry"]["id"] == "commit:ccc"
    hits = mine.query_run(run, category="recovery-fault")
    assert [h["entry"]["id"] for h in hits] == ["commit:ccc"]  # category alone selects
    assert mine.query_run(run, keywords=["nonesuch-term"]) == []


def test_query_cli_prints_advisory_header(tmp_path, capsys):
    run = _run_with_patterns(tmp_path)
    assert mine.main(["query", str(run), "--keywords", "race"]) == 0
    out = capsys.readouterr().out
    assert "advisory priors" in out.splitlines()[0]
    assert "never a finding" in out
    assert "commit:aaa" in out


def test_query_skips_unreadable_file_cache_tier(tmp_path, capsys):
    run = _run_with_patterns(tmp_path)
    (run / "specification" / "references" / "broken").mkdir(parents=True)
    (run / "specification" / "references" / "broken" / "failure_patterns.json").write_text(
        "{not json", encoding="utf-8"
    )
    hits = mine.query_run(run, keywords=["race"])  # warns, never raises
    assert len(hits) == 2
    assert "WARNING" in capsys.readouterr().err
