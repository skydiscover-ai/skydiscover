"""End-to-end: the wiki IS the knowledge base, and it is actually used.

Two halves, one crisp worked example (a `kvstore` domain grounded in Redis):

  1. CREATE — a wiki authored the way `kb-builder` authors it (typed frontmatter pages under
     ~/.skydiscover/<domain>/wiki/: a repo source pinned to a commit sha, a property, a test, a
     hack) passes the shipped `kbtool.py validate` (fail-closed) and `kbtool.py index` builds the
     cross-reference views the consumers read first (by-property / by-test / by-hack / by-source).

  2. USE — the discover phase's deterministic acquire-first reuse (the PreToolUse clone guard, the
     most important consumer: code-to-spec) reads that same wiki. `wiki_resolve` returns an EXACT
     hit for the repo at its pinned sha, so `decide` BLOCKS a re-clone and points the agent at the
     page to ground from instead. A drifted commit or an ungrounded repo does not block.

Stdlib only; the wiki validator runs as a subprocess, exactly as the coding agent invokes it.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = _ROOT / "skydiscover" / "synthesize" / "workflow" / "scripts" / "kb"
_GUARD = _ROOT / "skydiscover" / "synthesize" / "workflow" / "hooks" / "clone_reuse_guard.py"

# A real 40-char commit sha the repo page is pinned to; the clone's remote HEAD must match it exactly.
PINNED = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
DRIFTED = "0000000000000000000000000000000000000000"  # same repo, a different commit

# Every tag below is in the shipped tags.yaml; the wiki never carries its own copy of the rulebook.
_PAGES = {
    "kvstore/wiki/sources/repo-redis.md": f"""\
---
kind: repo
id: repo-redis
domain: kvstore
tags: [consistency, production]
url: https://github.com/redis/redis
sha: {PINNED}
license: BSD-3-Clause
following: high
production: yes
---

Redis is an in-memory key-value store, widely deployed in production.
""",
    "kvstore/wiki/properties/prop-kvstore-consistency.md": """\
---
kind: property
id: prop-kvstore-consistency
domain: kvstore
tags: [consistency]
values: [eventual, strong]
seen_in: [repo-redis]
sources: [repo-redis]
---

What a read must observe relative to the writes that precede it.
""",
    "kvstore/wiki/tests/test-kvstore-atomicity.md": """\
---
kind: test
id: test-kvstore-atomicity
domain: kvstore
tags: [atomicity]
enforces: prop-kvstore-consistency
command: pytest tests/test_atomicity.py
mutant: drop-the-lock
confidence: source-reported
sources: [repo-redis]
---

A read must never observe a partially applied write.
""",
    "kvstore/wiki/hacks/hack-kvstore-skip.md": """\
---
kind: hack
id: hack-kvstore-skip
domain: kvstore
tags: [correctness]
tell: returns a cached constant instead of doing the work
killed_by: test-kvstore-atomicity
sources: [repo-redis]
---

A candidate that returns a constant value to clear the benchmark without computing.
""",
}


def _seed_wiki(root: Path) -> Path:
    """Author a minimal, fully grounded kvstore wiki the way kb-builder would, under a knowledge base root
    (the ~/.skydiscover stand-in)."""
    for rel, body in _PAGES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return root


def _kbtool(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_TOOLS / "kbtool.py"), "--root", str(root), *args],
        capture_output=True,
        text=True,
    )


def _load_guard():
    spec = importlib.util.spec_from_file_location("_clone_reuse_guard_wiki", _GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 1. CREATE


def test_authored_wiki_validates_green(tmp_path):
    root = _seed_wiki(tmp_path / "home")
    r = _kbtool(root, "validate")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WIKI: OK" in r.stdout


def test_a_verified_test_note_is_proven_by_its_own_command(tmp_path):
    """The validator runs the page's `command` from wiki/tests/, in whatever language the domain
    uses; it does not assume a Python test suite."""
    root = _seed_wiki(tmp_path / "home")
    page = root / "kvstore" / "wiki" / "tests" / "test-kvstore-atomicity.md"
    body = page.read_text(encoding="utf-8").replace(
        "confidence: source-reported", "confidence: verified"
    )
    (root / "kvstore" / "wiki" / "tests" / "check.sh").write_text("#!/bin/sh\ntest -f check.sh\n")
    page.write_text(body.replace("command: pytest tests/test_atomicity.py", "command: sh check.sh"))
    r = _kbtool(root, "validate")
    assert r.returncode == 0, r.stdout + r.stderr
    page.write_text(body.replace("command: pytest tests/test_atomicity.py", "command: exit 3"))
    r = _kbtool(root, "validate")
    assert r.returncode != 0 and "`command` did not pass" in r.stdout + r.stderr


def test_index_builds_the_cross_reference_views(tmp_path):
    root = _seed_wiki(tmp_path / "home")
    assert _kbtool(root, "index").returncode == 0
    wiki = root / "kvstore" / "wiki"
    by_prop = (wiki / "index" / "by-property.md").read_text()
    # the property is indexed to the test that enforces it and the hack that attacks it
    assert "prop-kvstore-consistency" in by_prop
    assert "test-kvstore-atomicity" in by_prop
    assert "hack-kvstore-skip" in by_prop
    assert (wiki / "index.md").exists()  # the domain's own front page
    assert not (root / "index.md").exists()  # nothing generated outside the domain's wiki


# 2. READ


def test_find_ranks_pages_by_words_and_filters_by_kind_and_tag(tmp_path):
    """`find` is how a role gets from a question to page ids: words rank, filters narrow."""
    root = _seed_wiki(tmp_path / "home")
    r = _kbtool(root, "find", "consistency")
    ids = [l.split()[0] for l in r.stdout.splitlines()[2:] if not l.startswith(" ")]
    # the property's id and tag say consistency; the test enforces it; the hack does not mention it
    assert ids[0] == "prop-kvstore-consistency" and "test-kvstore-atomicity" in ids
    assert "hack-kvstore-skip" not in ids
    assert str(root / "kvstore" / "wiki" / "properties" / "prop-kvstore-consistency.md") in r.stdout
    r = _kbtool(root, "find", "--kind", "hack")
    assert "hack-kvstore-skip  [hack, kvstore]" in r.stdout
    assert "killed_by: test-kvstore-atomicity" in r.stdout
    r = _kbtool(root, "find", "--tag", "atomicity")
    assert "test-kvstore-atomicity  [test" in r.stdout
    assert "enforces: prop-kvstore-consistency" in r.stdout
    r = _kbtool(root, "find", "consist")
    assert ids[0] in r.stdout  # substring match
    r = _kbtool(root, "find", "zebra")
    assert r.returncode == 0 and "no page matches" in r.stdout


def test_page_prints_one_page_and_follows_its_links(tmp_path):
    """`page <id>` resolves an id to its file; --follow-sources glimpses every page it names."""
    root = _seed_wiki(tmp_path / "home")
    r = _kbtool(root, "page", "hack-kvstore-skip")
    assert r.stdout.startswith(
        "# " + str(root / "kvstore" / "wiki" / "hacks" / "hack-kvstore-skip.md")
    )
    assert "tell: returns a cached constant" in r.stdout and "--- " not in r.stdout
    r = _kbtool(root, "page", "hack-kvstore-skip", "--follow-sources")
    # sources and killed_by, each once, with its first paragraph
    assert r.stdout.count("--- repo-redis  [repo]") == 1
    assert "--- test-kvstore-atomicity  [test]" in r.stdout
    assert "A read must never observe a partially applied write." in r.stdout
    r = _kbtool(root, "page", "prop-kvstore-nope")
    assert r.returncode == 0 and "no page 'prop-kvstore-nope'" in r.stdout


def test_an_id_used_in_two_domains_is_a_validate_error_and_page_asks_for_a_domain(tmp_path):
    root = _seed_wiki(tmp_path / "home")
    twin = root / "queue" / "wiki" / "sources" / "repo-redis.md"
    twin.parent.mkdir(parents=True)
    twin.write_text(
        _PAGES["kvstore/wiki/sources/repo-redis.md"].replace("domain: kvstore", "domain: queue")
    )
    r = _kbtool(root, "validate")
    assert r.returncode == 1 and "repo-redis: the same id is in kvstore, queue" in r.stdout
    r = _kbtool(root, "page", "repo-redis")
    assert "is in 2 domains (kvstore, queue); pass --domain" in r.stdout
    r = _kbtool(root, "page", "repo-redis", "--domain", "queue")
    assert "domain: queue" in r.stdout


def test_a_domain_declares_its_own_tags_without_editing_the_framework(tmp_path):
    """A compiler domain needs `soundness`; a run may not edit the shipped tags.yaml, so the word
    goes in the domain's own wiki/tags.yaml and validate accepts it there and only there."""
    root = _seed_wiki(tmp_path / "home")
    page = root / "kvstore" / "wiki" / "properties" / "prop-kvstore-consistency.md"
    page.write_text(
        page.read_text().replace("tags: [consistency]", "tags: [consistency, soundness]")
    )
    r = _kbtool(root, "validate")
    assert r.returncode == 1 and "kvstore/wiki/tags.yaml" in r.stdout
    (root / "kvstore" / "wiki" / "tags.yaml").write_text("tags:\n  - soundness\n")
    r = _kbtool(root, "validate")
    assert r.returncode == 0, r.stdout + r.stderr
    r = _kbtool(root, "find", "--tag", "soundness")
    assert "prop-kvstore-consistency" in r.stdout


def test_root_is_read_from_the_flag_wherever_it_stands(tmp_path):
    root = _seed_wiki(tmp_path / "home")
    r = subprocess.run(
        [sys.executable, str(_TOOLS / "kbtool.py"), "validate", "--root", str(root)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0 and "WIKI: OK" in r.stdout, r.stdout + r.stderr


# 3. USE (code-to-spec)


def test_wiki_resolve_reads_the_repo_page(tmp_path):
    root = _seed_wiki(tmp_path / "home")
    g = _load_guard()
    # exact commit -> reuse (skip the clone), and it names the page to ground from
    hit = g.wiki_resolve("redis/redis", PINNED, root=root)
    assert hit["tier"] == "exact"
    assert (
        hit["page"] == "kvstore/wiki/sources/repo-redis.md"
    )  # relative to the knowledge base root
    # same repo, a different commit -> the grounding is stale, do not block reuse
    assert g.wiki_resolve("redis/redis", DRIFTED, root=root)["tier"] == "prior"
    # a repo the wiki does not ground at all -> miss
    assert g.wiki_resolve("other/unknown", PINNED, root=root)["tier"] == "miss"


def test_guard_blocks_a_reclone_of_a_wiki_grounded_source(tmp_path, monkeypatch):
    root = _seed_wiki(tmp_path / "home")
    g = _load_guard()
    monkeypatch.setenv("SKYDISCOVER_HOME", str(root))  # decide() uses the real wiki_resolve
    payload = {
        "tool_name": "Bash",
        "tool_input": {
            "command": "git clone https://github.com/redis/redis .sky/specification/sources/redis"
        },
    }
    block, msg = g.decide(payload, head_fn=lambda url: PINNED)
    assert block is True
    assert "already in the wiki" in msg
    assert "repo-redis.md" in msg  # points the agent at the page, not a re-clone
    assert ".sky/specification/references/redis" in msg  # where to write the grounded spec


def test_guard_allows_the_clone_when_the_wiki_grounding_is_stale(tmp_path, monkeypatch):
    root = _seed_wiki(tmp_path / "home")
    g = _load_guard()
    monkeypatch.setenv("SKYDISCOVER_HOME", str(root))
    payload = {
        "tool_name": "Bash",
        "tool_input": {
            "command": "git clone https://github.com/redis/redis .sky/specification/sources/redis"
        },
    }
    # remote HEAD has drifted off the pinned sha -> the grounding is stale -> allow the fresh clone
    block, _ = g.decide(payload, head_fn=lambda url: DRIFTED)
    assert block is False
