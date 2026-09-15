#!/usr/bin/env python3
"""Read, check, and index the wiki: the markdown files under every <kb root>/<domain>/wiki/.
Each file is one page, about one thing learned: a header of `key: value` lines, then text.

Read it:

    python3 kbtool.py find durability --kind hack      rank pages by words (title, then id/tags, then body);
                                                       filter by --kind, --tag, --domain
    python3 kbtool.py page <id> [--follow-sources]     print one page; with the flag, the first
                                                       paragraph of every page it links to

Keep it honest:

    python3 kbtool.py validate    reject any page whose header breaks schema.yaml or tags.yaml
                                  (a domain adds its own words in <domain>/wiki/tags.yaml), cites
                                  a page that does not exist, or claims a test result the tool
                                  cannot reproduce
    python3 kbtool.py index       rewrite each domain's wiki/index/ folder (which property has a test,
                                  which hack is caught, which source is cited)

All take --root <folder>; the default is the knowledge base root spec/paths.py resolves
(~/.skydiscover unless config.toml or $SKYDISCOVER_HOME moves it).
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))


def default_root():
    """The knowledge base root spec/paths.py resolves (config, $SKYDISCOVER_HOME, ~/.skydiscover)."""
    sys.path.insert(0, str(Path(HERE).resolve().parents[2]))  # a source checkout's spec/
    try:
        from spec.paths import home  # noqa: WPS433
    except ModuleNotFoundError:
        try:
            from skydiscover.synthesize.spec.paths import home  # noqa: WPS433
        except ModuleNotFoundError:
            return os.path.expanduser(os.environ.get("SKYDISCOVER_HOME") or "~/.skydiscover")
    return str(home())


ROOT = ""  # set by main() from --root, else default_root()

# `shared/` holds pages that apply to every domain and is validated like one; the clone cache and
# lock files are hidden (dot-prefixed) and skipped below.
# Header fields whose values are ids of other pages.
LINK_FIELDS = ("sources", "enforces", "killed_by", "seen_in")
# Prefixes that mark a reference to another page, for integrity checks.
REF_PREFIXES = (
    "repo-",
    "pr-",
    "issue-",
    "doc-",
    "prop-",
    "test-",
    "bench-",
    "roof-",
    "hack-",
    "design-",
)
# An inline PR/issue link in a page body is a source cited outside what this tool can
# see. Every such link must have a backing source page, or a fabricated one would pass unchecked.
INLINE_REF = re.compile(r"github\.com/([\w.-]+/[\w.-]+)/(issues|pull)/(\d+)")


def parse(path):
    """Read a page's frontmatter into a dict. Scalars, inline [a, b] lists, and
    dash-item lists are all handled; a trailing # comment is stripped."""
    with open(path, encoding="utf-8") as fh:
        t = fh.read()
    m = re.match(r"^---\n(.*?)\n---\n", t, re.S)
    if not m:
        return None
    fm, key = {}, None
    for line in m.group(1).splitlines():
        if re.match(r"^[A-Za-z_]+:", line):
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if val.startswith("[") and val.endswith("]"):
                fm[key] = [x.strip() for x in val[1:-1].split(",") if x.strip()]
            elif val == "":
                fm[key] = []
            else:
                fm[key] = val.split("#")[0].strip() if not val.startswith(('"', "'")) else val
        elif key and re.match(r"^\s*-\s+", line):
            fm.setdefault(key, [])
            if isinstance(fm[key], list):
                fm[key].append(re.sub(r"^\s*-\s+", "", line).split("#")[0].strip())
    fm["_path"] = path
    fm["_body"] = t[m.end() :]
    return fm


def wiki_dir(domain):
    return os.path.join(ROOT, domain, "wiki")


def load_all(only=None):
    """Every typed page in every domain's wiki/ (never the clone cache beside them), by id. An id
    used in two domains keeps the first (domains sorted) and lists the others under `_also`."""
    pages = {}
    for domain in domains_on_disk():
        if only and domain != only:
            continue
        for p in sorted(glob.glob(wiki_dir(domain) + "/**/*.md", recursive=True)):
            fm = parse(p)
            if fm and "id" in fm and "kind" in fm:
                fm["_folder"] = domain
                if fm["id"] in pages:
                    pages[fm["id"]].setdefault("_also", []).append(domain)
                else:
                    pages[fm["id"]] = fm
    return pages


def _lists(text):
    """`key: [a, b]` lines (at any indent) and dash-item lists under a `key:` line, as {key: [..]}.
    Enough YAML for schema.yaml and tags.yaml, which are written to this shape."""
    out, key = {}, None
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        m = re.match(r"^\s*([A-Za-z_-]+):\s*(.*)$", line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            if val.startswith("[") and val.endswith("]"):
                out[key] = [x.strip() for x in val[1:-1].split(",") if x.strip()]
            elif not val:
                out.setdefault(key, [])
        elif key and re.match(r"^\s*-\s+", line):
            out.setdefault(key, []).append(re.sub(r"^\s*-\s+", "", line).strip())
    return out


def load_rules():
    """schema.yaml and tags.yaml beside this script: (required fields, fields per kind, self-source
    kinds, confidence values, tags)."""
    schema = _lists(open(os.path.join(HERE, "schema.yaml")).read())
    tags = set(_lists(open(os.path.join(HERE, "tags.yaml")).read()).get("tags", []))
    kinds = {
        k: v
        for k, v in schema.items()
        if k not in ("required", "kinds", "self_source", "confidence")
    }
    return (
        schema.get("required", []),
        kinds,
        set(schema.get("self_source", [])),
        set(schema.get("confidence", [])),
        tags,
    )


def domain_tags(domain):
    """Extra tags a domain declares in its own wiki/tags.yaml (and shared/wiki/tags.yaml): the words
    one kind of system needs that the shipped list does not have. The framework is never edited."""
    extra = set()
    for d in (domain, "shared"):
        path = os.path.join(wiki_dir(d), "tags.yaml")
        if os.path.isfile(path):
            extra |= set(_lists(open(path, encoding="utf-8").read()).get("tags", []))
    return extra


def domains_on_disk():
    """Every domain folder under the knowledge-base root that has wiki/ (shared/ counts: its pages hold
    for every domain)."""
    if not os.path.isdir(ROOT):
        return []
    return sorted(
        d for d in os.listdir(ROOT) if os.path.isdir(wiki_dir(d)) and not d.startswith(".")
    )


_tests_cache = {}
TEST_COMMAND_SECS = int(os.environ.get("SKYDISCOVER_SLOW_SECS", "600") or "600")


def test_passes(fm):
    """Re-run one `verified` test page's own `command` from its domain's wiki/tests/ directory, in
    whatever language or runner the command names. Cached per (domain, command). Returns False when
    there is no command or it fails: a page cannot claim `verified` when the check that would prove
    it does not run here."""
    domain, command = fm.get("domain"), (fm.get("command") or "").strip()
    key = (domain, command)
    if key in _tests_cache:
        return _tests_cache[key]
    tdir = os.path.join(wiki_dir(domain), "tests")
    ok = False
    if command and os.path.isdir(tdir):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")  # keep __pycache__ out of the wiki tree
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=tdir,
                capture_output=True,
                text=True,
                env=env,
                timeout=TEST_COMMAND_SECS,
            )
            ok = r.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
    _tests_cache[key] = ok
    return ok


def validate(pages):
    required, kinds, self_source, confidences, tags = load_rules()
    domains = set(domains_on_disk())
    # Every PR/issue that HAS a source page, keyed by (repo, kind, number).
    backed = {
        ((fm.get("repo") or "").lower(), fm["kind"], str(fm.get("number") or ""))
        for fm in pages.values()
        if fm["kind"] in ("pr", "issue")
    }
    errs = []
    for pid, fm in pages.items():
        k = fm["kind"]
        if fm.get("_also"):
            errs.append(
                f"{pid}: the same id is in {', '.join([fm['_folder'], *fm['_also']])} "
                "(an id names one page)"
            )
        if k not in kinds:
            errs.append(f"{pid}: kind '{k}' is not in schema.yaml ({', '.join(sorted(kinds))})")
            continue
        for field in list(required) + kinds[k]:
            if not fm.get(field):
                errs.append(f"{pid}: a {k} page needs '{field}' (schema.yaml)")
        if fm.get("confidence") and fm["confidence"] not in confidences:
            errs.append(
                f"{pid}: confidence '{fm['confidence']}' is not one of {sorted(confidences)}"
            )
        if fm.get("domain") != fm["_folder"]:
            errs.append(
                f"{pid}: domain '{fm.get('domain')}' but the page lives in {fm['_folder']}/wiki "
                "(a page's domain is the folder it is in)"
            )
        allowed = tags | domain_tags(fm["_folder"])
        for t in fm.get("tags", []):
            if t not in allowed:
                errs.append(
                    f"{pid}: tag '{t}' is not in tags.yaml; a word this kind of system needs goes "
                    f"in {fm['_folder']}/wiki/tags.yaml"
                )
        if k not in self_source and not fm.get("sources"):
            errs.append(
                f"{pid}: no sources (only {'/'.join(sorted(self_source))} pages are sources themselves)"
            )
        # referential integrity: every internal reference must resolve to a page
        for field in LINK_FIELDS:
            for r in _links(fm, field):
                if r.startswith(REF_PREFIXES) and r not in pages:
                    errs.append(f"{pid}: {field} -> unknown page '{r}'")
        if (
            k == "test"
            and fm.get("confidence") == "verified"
            and fm.get("domain") in domains
            and not test_passes(fm)
        ):
            errs.append(f"{pid}: confidence=verified but its `command` did not pass here")
        # A PR/issue link in the body must be a source page, not a bare inline claim.
        for repo, gh_kind, num in INLINE_REF.findall(fm.get("_body", "")):
            key = (repo.lower(), "pr" if gh_kind == "pull" else "issue", num)
            if key not in backed:
                errs.append(
                    f"{pid}: inline citation github.com/{repo}/{gh_kind}/{num} has no "
                    f"source page (cite sources by page id, never inline)"
                )
        # Prose style: plain sentences, no em-dash.
        if "\u2014" in fm.get("_body", ""):
            errs.append(
                f"{pid}: em-dash in body (write plain prose, split the sentence or use a colon)"
            )
    return errs


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def index_domain(domain, pages):
    """Regenerate one domain's four cross-reference views."""
    dom = [p for p in pages.values() if p.get("domain") == domain]
    props = [p for p in dom if p["kind"] == "property"]
    tests = [p for p in dom if p["kind"] == "test"]
    hacks = [p for p in dom if p["kind"] == "hack"]
    srcs = [p for p in dom if p["kind"] in load_rules()[2]]
    d = os.path.join(wiki_dir(domain), "index")

    def join(xs, empty):
        return ", ".join(xs) if xs else empty

    lines = ["# by-property (auto-generated by kbtool.py)\n"]
    for pr in sorted(props, key=lambda x: x["id"]):
        g = [x["id"] for x in tests if x.get("enforces") == pr["id"]]
        h = [x["id"] for x in hacks if x.get("killed_by") in g]
        lines.append(
            f"- **{pr['id']}** enforced by {join(g, 'GAP (no test)')}; "
            f"attacked by {join(h, 'none catalogued')}"
        )
    _write(d + "/by-property.md", "\n".join(lines) + "\n")

    lines = ["# by-test (auto-generated)\n"]
    for g in sorted(tests, key=lambda x: x["id"]):
        lines.append(
            f"- **{g['id']}** enforces {g.get('enforces')} (confidence: {g.get('confidence')})"
        )
    _write(d + "/by-test.md", "\n".join(lines) + "\n")

    lines = ["# by-hack (auto-generated)\n"]
    for h in sorted(hacks, key=lambda x: x["id"]):
        kb = h.get("killed_by")
        lines.append(f"- **{h['id']}** killed by {kb if kb in pages else f'GAP ({kb} missing)'}")
    _write(d + "/by-hack.md", "\n".join(lines) + "\n")

    cited = {}
    for p in dom:
        for s in p.get("sources") or []:
            cited.setdefault(s, []).append(p["id"])
    lines = ["# by-source (auto-generated)\n"]
    for s in sorted(srcs, key=lambda x: x["id"]):
        users = [u for u in cited.get(s["id"], []) if u != s["id"]]
        lines.append(
            f"- **{s['id']}** ({s['kind']}) grounds {join(users, 'ORPHAN (nothing cites it)')}"
        )
    _write(d + "/by-source.md", "\n".join(lines) + "\n")

    c = Counter(p["kind"] for p in dom)
    _write(
        os.path.join(wiki_dir(domain), "index.md"),
        f"# {domain} wiki index (auto)\n\n" + "\n".join(f"- {k}: {c[k]}" for k in sorted(c)) + "\n",
    )


def index(pages):
    for domain in domains_on_disk():
        index_domain(domain, pages)


# --- reading -------------------------------------------------------------------------------------


def _links(fm, field):
    """The page ids a header field names, as a list."""
    v = fm.get(field)
    return v if isinstance(v, list) else ([v] if v else [])


def _title(fm):
    """The page's `# heading`, else its id."""
    m = re.search(r"^# (.+)$", fm.get("_body", ""), re.M)
    return m.group(1).strip() if m else fm["id"]


def _first_paragraph(fm, limit=500):
    """The first paragraph after the heading, cut at `limit` characters."""
    body = re.sub(r"^# .+$", "", fm.get("_body", ""), count=1, flags=re.M).strip()
    para = re.split(r"\n\s*\n", body, maxsplit=1)[0]
    para = re.sub(r"\s+", " ", para)
    return para if len(para) <= limit else para[: limit - 1].rstrip() + "…"


# The header fields a `find` hit shows under its title, per kind.
GLANCE = {
    "property": ("values", "seen_in"),
    "test": ("enforces", "confidence"),
    "hack": ("killed_by",),
    "bench": ("metric", "baseline"),
    "roof": ("bound", "ceiling"),
    "design": ("outcome", "score"),
}


def _glance(fm, limit=100):
    """The header facts a reader wants first, by kind; sources show their url."""
    parts = []
    for field in GLANCE.get(fm["kind"], ("url",)):
        v = fm.get(field)
        if isinstance(v, list):
            v = f"{len(v)} values" if field == "values" else ", ".join(v)
        if not v:
            continue
        v = str(v) if len(str(v)) <= limit else str(v)[: limit - 1].rstrip() + "…"
        parts.append(v if field in ("url", "values") else f"{field}: {v}")
    return "; ".join(parts)


def score(fm, words):
    """How well a page answers `words`: a hit in the title counts 10, in the id, tags, or the pages
    it enforces or kills 5, and body hits count up to 3. Matching is by substring,
    case-insensitive, so `durab` finds durability."""
    title = _title(fm).lower()
    keys = " ".join(
        [fm["id"], *fm.get("tags", []), *_links(fm, "enforces"), *_links(fm, "killed_by")]
    ).lower()
    body = fm["_body"].lower()
    total = 0
    for w in (w.lower() for w in words):
        total += 10 * (w in title) + 5 * (w in keys) + min(body.count(w), 3)
    return total


def find(pages, words=(), kind=None, tag=None, domain=None, limit=10):
    """Pages matching the filters, best first. With no words, every match in id order."""
    hits = [
        fm
        for fm in pages.values()
        if (not kind or fm["kind"] == kind)
        and (not tag or tag in fm.get("tags", []))
        and (not domain or fm["_folder"] == domain)
    ]
    if words:
        hits = [fm for fm in hits if score(fm, words) > 0]
        hits.sort(key=lambda fm: (-score(fm, words), fm["id"]))
    else:
        hits.sort(key=lambda fm: fm["id"])
    return hits[:limit]


def show_find(hits):
    """One hit per block: id, kind and domain, title; a glance at its header; its path."""
    if not hits:
        return "no page matches; try fewer words, or `kbtool.py find --kind <kind>` to list a kind"
    lines = [f"# {len(hits)} page(s)", ""]
    for fm in hits:
        lines.append(f"{fm['id']}  [{fm['kind']}, {fm['_folder']}]  {_title(fm)}")
        detail = _glance(fm)
        if detail:
            lines.append(f"    {detail}")
        lines.append(f"    {fm['_path']}")
    return "\n".join(lines)


def show_page(pages, pid, follow_sources=False):
    """One page in full, then, if asked, a glimpse of every page it links to."""
    fm = pages.get(pid)
    if fm is None:
        return f"no page '{pid}'; `kbtool.py find {pid.split('-')[-1]}` may find it"
    if fm.get("_also"):
        where = ", ".join([fm["_folder"], *fm["_also"]])
        return f"id {pid} is in {len(fm['_also']) + 1} domains ({where}); pass --domain to pick one"
    out = [f"# {fm['_path']}", "", open(fm["_path"], encoding="utf-8").read().rstrip()]
    if follow_sources:
        seen = []
        for field in LINK_FIELDS:
            seen += [r for r in _links(fm, field) if r in pages and r not in seen and r != pid]
        for r in seen:
            linked = pages[r]
            out += [
                "",
                f"--- {r}  [{linked['kind']}]  {linked['_path']}",
                _title(linked),
                _first_paragraph(linked),
            ]
    return "\n".join(out)


def main(argv=None):
    global ROOT
    root = argparse.ArgumentParser(add_help=False)
    # SUPPRESS: the flag may stand before or after the subcommand, and an absent one must not
    # overwrite a present one.
    root.add_argument(
        "--root",
        default=argparse.SUPPRESS,
        help="knowledge base root (default: what spec/paths.py resolves)",
    )
    ap = argparse.ArgumentParser(
        prog="kbtool.py", description=__doc__.split("\n\n")[0], parents=[root]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser(
        "find", parents=[root], help="rank pages by words; filter by kind, tag, domain"
    )
    f.add_argument("words", nargs="*", help="words to look for in the title, id, tags, and body")
    f.add_argument("--kind", help="repo, pr, issue, doc, property, bench, roof, test, hack, design")
    f.add_argument("--tag", help="a tag from tags.yaml or a domain's wiki/tags.yaml")
    f.add_argument("--domain", help="one domain folder; default all, including shared/")
    f.add_argument("--limit", type=int, default=10, help="pages to show (default 10)")
    p = sub.add_parser("page", parents=[root], help="print one page by id")
    p.add_argument("id")
    p.add_argument("--domain", help="the domain to read from, when an id is in more than one")
    p.add_argument("--follow-sources", action="store_true", help="also glimpse every linked page")
    sub.add_parser(
        "validate", parents=[root], help="check every page against schema.yaml and tags.yaml"
    )
    sub.add_parser("index", parents=[root], help="rewrite each domain's wiki/index/")
    args = ap.parse_args(argv)
    given = getattr(args, "root", None)
    ROOT = os.path.abspath(os.path.expanduser(given)) if given else default_root()

    pages = load_all(only=getattr(args, "domain", None))
    if args.cmd == "find":
        print(show_find(find(pages, args.words, args.kind, args.tag, args.domain, args.limit)))
    elif args.cmd == "page":
        print(show_page(pages, args.id, args.follow_sources))
    elif args.cmd == "validate":
        errs = validate(pages)
        if errs:
            print("WIKI: PROBLEMS\n" + "\n".join(" - " + e for e in errs))
            return 1
        print(f"WIKI: OK ({len(pages)} pages, {len(domains_on_disk())} domains)")
    else:
        index(pages)
        print("indexes regenerated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
