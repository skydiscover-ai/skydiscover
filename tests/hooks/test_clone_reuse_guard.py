"""Tests for the PreToolUse reuse guard (workflow/hooks/clone_reuse_guard.py).

The guard is the deterministic half of reuse: it blocks a `git clone` of a source the wiki already
grounds AT the commit the clone would check out, so reuse cannot be skipped no matter what the
spec-builder's brief-reading decided. It must be fail-open everywhere EXCEPT a confirmed exact hit --
a parse miss, an unreadable remote HEAD, a drifted/ungrounded repo, or any error all allow the clone.

The decision is a pure function with injectable resolve/head lookups, so these exercise the real
shipped logic with no network and no wiki state. The wiki-reading half (``wiki_resolve``) is covered
end-to-end against a real validated wiki in tests/spec/test_wiki_usage.py.
"""

import importlib.util
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_GUARD = _ROOT / "skydiscover" / "synthesize" / "workflow" / "hooks" / "clone_reuse_guard.py"


def _load():
    spec = importlib.util.spec_from_file_location("_clone_reuse_guard", _GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g = _load()


# parse_clone / repo_from_url


def test_parse_clone_extracts_url_and_dest():
    url, dest = g.parse_clone(
        "git clone --filter=blob:none https://github.com/efficient/cuckoofilter .sky/specification/sources/cuckoofilter"
    )
    assert url == "https://github.com/efficient/cuckoofilter"
    assert dest == ".sky/specification/sources/cuckoofilter"


def test_parse_clone_dest_optional_and_flags_skipped():
    url, dest = g.parse_clone("git clone https://github.com/o/r.git")
    assert url == "https://github.com/o/r.git" and dest is None


def test_parse_clone_returns_none_for_a_non_clone():
    assert g.parse_clone("ls -la") is None
    assert g.parse_clone("git status") is None
    assert g.parse_clone("") is None
    assert g.parse_clone("git clone --depth 1") is None  # a clone with no URL


def test_repo_from_url_is_host_agnostic():
    assert g.repo_from_url("https://github.com/efficient/cuckoofilter") == "efficient/cuckoofilter"
    assert g.repo_from_url("https://gitlab.com/o/r.git") == "o/r"
    assert g.repo_from_url("git@github.com:torvalds/linux.git") == "torvalds/linux"
    assert g.repo_from_url("not a url") is None


# decide (the blocking rule)


def _clone_payload(cmd):
    return {"tool_name": "Bash", "tool_input": {"command": cmd}}


def test_decide_blocks_an_exact_cached_clone_and_points_at_the_wiki():
    block, msg = g.decide(
        _clone_payload("git clone https://github.com/o/r .sky/specification/sources/r --into x"),
        resolve_fn=lambda repo, sha: {"tier": "exact", "page": "kvstore/wiki/sources/repo-r.md"},
        head_fn=lambda url: "abc123def456",
    )
    assert block is True
    assert "already in the wiki" in msg
    assert "kvstore/wiki/sources/repo-r.md" in msg  # names the page to ground from
    assert ".sky/specification/references/r" in msg  # run/name derived from the dest
    assert "abc123def456" in msg


def test_decide_allows_a_prior_or_miss_clone():
    for tier in ("prior", "miss"):
        block, _ = g.decide(
            _clone_payload("git clone https://github.com/o/r .sky/specification/sources/r"),
            resolve_fn=lambda repo, sha: {"tier": tier},
            head_fn=lambda url: "abc123def456",
        )
        assert block is False, tier


def test_decide_fails_open_when_head_is_unreadable():
    # offline / bad remote -> cannot be sure -> never block
    block, _ = g.decide(
        _clone_payload("git clone https://github.com/o/r .sky/specification/sources/r"),
        resolve_fn=lambda repo, sha: {"tier": "exact"},  # would block IF head were known
        head_fn=lambda url: "",
    )
    assert block is False


def test_decide_fails_open_on_a_resolver_error():
    def boom(repo, sha):
        raise RuntimeError("kb blew up")

    block, _ = g.decide(
        _clone_payload("git clone https://github.com/o/r .sky/specification/sources/r"),
        resolve_fn=boom,
        head_fn=lambda url: "abc123def456",
    )
    assert block is False


def test_decide_ignores_non_bash_and_non_clone():
    # a non-Bash tool is never inspected
    block, _ = g.decide(
        {"tool_name": "Read", "tool_input": {"command": "git clone https://github.com/o/r"}},
        resolve_fn=lambda r, s: {"tier": "exact"},
        head_fn=lambda u: "abc123def456",
    )
    assert block is False
    # a Bash command that is not a clone is never inspected
    block, _ = g.decide(
        _clone_payload("echo git clone"),
        resolve_fn=lambda r, s: {"tier": "exact"},
        head_fn=lambda u: "abc123def456",
    )
    assert block is False


# kb_root env precedence


def test_kb_root_follows_skydiscover_home(monkeypatch, tmp_path):
    # The guard reads the same knowledge base root a run relocated by $SKYDISCOVER_HOME writes to,
    # and the default home otherwise.
    monkeypatch.setenv("SKYDISCOVER_HOME", str(tmp_path / "home"))
    assert g.kb_root() == tmp_path / "home"
    monkeypatch.delenv("SKYDISCOVER_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    assert g.kb_root() == tmp_path / "user" / ".skydiscover"


def test_flags_between_git_and_clone_are_still_a_clone(tmp_path):
    """`git -C /tmp clone` and `git --no-pager clone` are clones; the guard must see them.
    Non-clone commands that merely mention the word stay invisible to it."""
    from skydiscover.synthesize.workflow.hooks import clone_reuse_guard as g

    for cmd in (
        "git -C /tmp clone https://github.com/o/r d",
        "git --no-pager clone https://github.com/o/r d",
        "git -c core.sshCommand=ssh clone https://github.com/o/r",
    ):
        assert g.parse_clone(cmd), cmd
    for cmd in ("git log | grep clone", "git status && echo clone"):
        assert g.parse_clone(cmd) is None, cmd


def test_parse_clone_reads_the_clone_url_not_the_first_url_in_the_command():
    from skydiscover.synthesize.workflow.hooks import clone_reuse_guard as g

    url, dest = g.parse_clone(
        "curl -sO https://a.example/b.tar && git clone https://github.com/o/r d"
    )
    assert (url, dest) == ("https://github.com/o/r", "d")
    url, dest = g.parse_clone("git clone https://github.com/o/r && cd r")
    assert (url, dest) == ("https://github.com/o/r", None)


def test_parse_clone_ignores_a_clone_pinned_to_a_branch_or_tag():
    """The guard compares the remote HEAD with the wiki page's sha; a clone of another ref is not
    the same bytes, so it is never blocked."""
    from skydiscover.synthesize.workflow.hooks import clone_reuse_guard as g

    for cmd in (
        "git clone -b v1.2 https://github.com/o/r",
        "git clone --branch release https://github.com/o/r d",
        "git clone --branch=release https://github.com/o/r d",
    ):
        assert g.parse_clone(cmd) is None, cmd
