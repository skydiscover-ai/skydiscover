#!/usr/bin/env python3
"""Shell hook: block a git clone of a repo a domain's wiki already covers at that exact commit
(exit 2, with a pointer to the page). Anything else is allowed, including any internal
error. Wiki folders live at <home>/<domain>/wiki (spec/paths.py); stdlib only.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

# owner/repo out of an https or ssh git URL, host-agnostic (no hardcoded host), .git optional.
_URL_RE = re.compile(r"(?:https?://[^/\s]+/|git@[^:\s]+:)([\w.-]+/[\w.-]+?)(?:\.git)?(?:\s|/|$)")
# Flags may sit between git and clone (`git -C /tmp clone`, `git --no-pager clone`); each is
# either a lone flag or a flag with a separate value token, so allow any run of such tokens.
_CLONE_RE = re.compile(r"\bgit(?:\s+-[\w-]+(?:=\S+)?(?:\s+(?!clone\b)[^\s-]\S*)?)*\s+clone\b")


# One remote HEAD lookup; a hook must answer quickly, so a slow remote is treated as unknown.
LS_REMOTE_SECS = int(os.environ.get("SKYDISCOVER_LS_REMOTE_SECS", "20") or "20")


def kb_root() -> Path:
    """The knowledge base root that holds every <domain>/wiki: spec.paths.home(), stdlib fallback."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # a source checkout's spec/
    try:
        try:
            from spec.paths import home  # noqa: WPS433
        except ModuleNotFoundError:
            from skydiscover.synthesize.spec.paths import home  # noqa: WPS433
        return home()
    except Exception:
        env = os.environ.get("SKYDISCOVER_HOME")
        return Path(env).expanduser() if env else Path.home() / ".skydiscover"


def wiki_page(root: Path, pattern: str) -> list:
    """Pages matching <domain>/wiki/<pattern> under the knowledge base root, plus shared/wiki. Never
    descends into .cache/, where whole clones live."""
    hits = []
    for wiki in sorted(glob.glob(str(root / "*" / "wiki"))):
        hits += glob.glob(os.path.join(wiki, pattern), recursive=True)
    return hits


def _page_frontmatter(path: str) -> dict:
    """Parse just the scalar frontmatter of a page (enough for repo pages: url + sha). Returns
    {} on any read or parse problem, so a malformed page never blocks a command."""
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return {}
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).splitlines():
        mm = re.match(r"^([A-Za-z_]+):\s*(.*)$", line)
        if mm:
            fm[mm.group(1).strip()] = mm.group(2).split("#")[0].strip().strip("\"'")
    return fm


def _sha_matches(a: str, b: str) -> bool:
    """True if two commit shas name the same commit, tolerant of abbreviation (a page may pin a
    12-char sha while ls-remote returns the full 40). Requires at least 7 hex chars to match."""
    a, b = (a or "").lower(), (b or "").lower()
    n = min(len(a), len(b))
    return n >= 7 and (a.startswith(b) or b.startswith(a))


def wiki_resolve(repo: str, sha: str, root: Optional[Path] = None) -> dict:
    """Look up owner/repo at sha in the wiki's repo pages. Returns {"tier": exact | prior | miss,
    "page": path}.
    """
    root = Path(root) if root is not None else kb_root()
    want = (repo or "").lower()
    prior = ""
    for page in wiki_page(root, os.path.join("**", "sources", "repo-*.md")):
        fm = _page_frontmatter(page)
        url = fm.get("url") or ""
        page_repo = repo_from_url(url)
        if not page_repo or page_repo.lower() != want:
            continue
        rel = os.path.relpath(page, root)
        if _sha_matches(fm.get("sha") or "", sha):
            return {"tier": "exact", "page": rel}
        prior = rel  # same repo, different commit -> page exists but is stale
    return {"tier": "prior" if prior else "miss", "page": prior}


def _ls_remote_head(url: str) -> str:
    """The remote HEAD commit sha for a git URL, via git ls-remote. Empty string on any failure
    (offline, bad url, git missing) so the guard fails open."""
    try:
        r = subprocess.run(
            ["git", "ls-remote", url, "HEAD"],
            capture_output=True,
            text=True,
            timeout=LS_REMOTE_SECS,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if r.returncode != 0 or not r.stdout.strip():
        return ""
    return r.stdout.split()[0].strip()


def parse_clone(command: str) -> Optional[Tuple[str, Optional[str]]]:
    """From a shell command, return (url, dest) for a git clone (dest = the target path
    token if present), or None when the command is not a git clone / has no parseable URL."""
    if not command:
        return None
    clone = _CLONE_RE.search(command)
    if not clone:
        return None
    rest = command[clone.end() :]
    m = re.search(r"(?:https?://\S+|git@\S+)", rest)
    if not m:
        return None
    # A clone pinned to a branch or tag checks out something other than HEAD, which is all the
    # wiki page's sha can be compared with.
    if re.search(r"(?:^|\s)(?:-b|--branch)(?:\s|=)", rest[: m.start()]):
        return None
    url = m.group(0).rstrip("/")
    # the destination is the first non-flag token after the url, if any
    dest = None
    for tok in rest[m.end() :].split():
        if tok in ("&&", "||", ";", "|"):
            break
        if not tok.startswith("-"):
            dest = tok
            break
    return url, dest


def repo_from_url(url: str) -> Optional[str]:
    """owner/repo from a git URL, or None."""
    m = _URL_RE.search(url if url.endswith(("/", " ")) else url + " ")
    return m.group(1) if m else None


def _run_and_name(dest: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Derive (run_dir, system_name) from a clone dest of shape <run>/specification/sources/<name>
    so the guidance can name the exact reference dir; (None, None) if it does not match."""
    if not dest:
        return None, None
    p = Path(dest)
    if p.parent.name == "sources" and p.parent.parent.name == "specification":
        return str(p.parent.parent.parent), p.name
    return None, None


def decide(
    payload: dict,
    resolve_fn: Optional[Callable] = None,
    head_fn: Optional[Callable] = None,
) -> Tuple[bool, str]:
    """Pure decision: (block, message). Injectable resolve_fn/head_fn for testing; both
    default to the wiki resolver and a real git ls-remote. Blocks only on a confirmed exact hit.
    """
    if (payload.get("tool_name") or payload.get("tool")) in ("Bash", "bash"):
        tool_input = payload.get("tool_input") or payload.get("input") or {}
        command = tool_input.get("command") if isinstance(tool_input, dict) else ""
    elif isinstance(payload.get("command"), str):
        # Cursor's beforeShellExecution hook carries the command text at the top level.
        command = payload["command"]
    else:
        return False, ""
    parsed = parse_clone(command or "")
    if not parsed:
        return False, ""
    url, dest = parsed
    repo = repo_from_url(url)
    if not repo:
        return False, ""
    resolve_fn = resolve_fn or wiki_resolve
    head_fn = head_fn or _ls_remote_head
    try:
        sha = head_fn(url)
        if not sha:
            return False, ""  # cannot read HEAD -> cannot be sure -> allow
        res = resolve_fn(repo, sha)
    except Exception:
        return False, ""
    if not isinstance(res, dict) or res.get("tier") != "exact":
        return False, ""  # prior / miss / anything unexpected -> allow the clone
    run, name = _run_and_name(dest)
    into = (
        f"{run}/specification/references/{name}"
        if run and name
        else "<run>/specification/references/<name>"
    )
    page = res.get("page") or "<domain>/wiki/sources/repo-<name>.md"
    wiki = (
        page.split(f"{os.sep}sources{os.sep}", 1)[0]
        if f"{os.sep}sources{os.sep}" in page
        else "<domain>/wiki"
    )
    msg = (
        f"reuse-guard: {repo} is already in the wiki at this exact commit ({sha}); "
        "re-cloning and re-analyzing it wastes the whole extraction. Ground the spec from the wiki "
        "instead of cloning:\n"
        f"  read the repo page {page} (pinned at this sha) and the wiki's index views "
        f"({wiki}/index/by-property.md, by-test.md, by-source.md), then write the spec to "
        f"{into}.\n"
        "Only clone the systems the wiki does not already have at their current HEAD."
    )
    return True, msg


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            return 0
    except Exception:
        return 0  # unreadable payload -> never block
    try:
        block, msg = decide(payload)
    except Exception:
        return 0  # a bug in the guard must never block a command
    if block:
        sys.stderr.write(msg + "\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
