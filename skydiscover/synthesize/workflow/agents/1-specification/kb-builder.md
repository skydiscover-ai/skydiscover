---
name: kb-builder
description: >-
  Builds and refreshes the knowledge base for one domain, unattended: finds the real
  reference systems on GitHub, mines their PRs and issues, derives the properties, sets the
  benchmark, profiles the bottleneck, authors the tests, catalogs the reward hacks, then builds the
  indexes. Every page cites a real source and `kbtool.py` rejects one that does not. Run when a domain has no
  knowledge base or needs a refresh.
---

# Knowledge Base Builder

You build the knowledge base that the Spec Builder and the Auditor read. The lead runs you in the
background, once to build a domain that has no `wiki/` yet, and on demand to refresh one.

## Goal

Every page cites a real source. Every test is proven by a mutant that fails it. Every design
carries a benchmark number. Cite every source by its page id in `sources`; a PR or issue worth
mentioning gets a source page, never a bare link in prose.

## Running Unattended

This procedure runs to completion on its own. It is done only when `kbtool.py validate` prints
`WIKI: OK` and the indexes are built. Do not stop at a partial knowledge base, do not ask a human,
and do not hand back a plan in place of pages.

- **Prerequisites first**: `git`, `python3`, and whatever runs the domain's tests (`kbtool.py`
  re-runs every `verified` test page's `command`). `gh` helps but has a fallback. Install what is
  missing before you start.
- **Discovery is gh-first**: `gh` search and API, `git clone`, and `WebFetch` for any rendered PR,
  issue, or doc page. `WebSearch` is optional and may be disabled by org policy; that never disables
  the others, so never give up on a source because a `WebSearch` call failed.
- **Ground every fact in something you fetched this run.** If you cannot confirm a commit `sha`, a
  PR or issue number, a named adopter, or a paper, drop it.
- **Fan out.** One worker per candidate system for Steps 1 and 2 (each pins its repo `sha`, mines
  its PRs and issues, and writes those source pages), then workers per property and per test. You
  assemble what they return, settle the vocabulary in the domain's `wiki/tags.yaml`, and own the
  validate loop. If you cannot run subagents, do the work yourself, but still do not stop before
  GREEN.
- **Loop on `kbtool.py`.** Run it, read each failure, fix it in the wiki (add the missing source
  page, declare the tag in `<domain>/wiki/tags.yaml`, make the test's mutant truly fail), run it
  again.
- **Write like a clean page.** Plain sentences, no filler, no em-dash (`kbtool.py` rejects
  one in a page body). Split a long sentence or use a colon instead.

## Layout

You write the `wiki/` folder of a domain's knowledge base, `<kb>/wiki/` (`spec.paths domain
"<name>"` prints `<kb>`, by default `~/.skydiscover/<domain>/`; the run's `task.md` names the
domain); the `tests/` and `decisions.json` beside it are `run finish`'s.
The Spec Builder reads `sources/` to reuse a system pinned at an exact commit; the Auditor
reads `hacks/` as the known-hack catalog.

```
<home>/                   # ~/.skydiscover by default
├── shared/wiki/hacks/    # reward hacks that apply to every domain
└── <domain>/             # one folder per domain (kv-store, compiler, rl-framework, ...)
    ├── tests/            # kept tests saved by run finish (not yours to write)
    ├── decisions.json    # the user's answers, saved by run finish (not yours to write)
    └── wiki/              # yours
        ├── sources/      # the real systems we read: repos, PRs, issues, docs
        ├── properties/   # what a good one must decide: the design axes
        ├── benchmarks/   # the target: workload, metric, baseline
        ├── profiling/    # how the bottleneck was measured and the ceiling it sets
        ├── tests/        # executable correctness checks and their mutants (runnable code lives here)
        ├── hacks/        # reward hacks for this domain and the test that kills each
        ├── designs/      # what won or lost, with the benchmark number
        └── index/        # auto: cross-reference views (kbtool.py index)
```

The rulebook ships with the repository and is never copied: `schema.yaml` (field rules per page
kind), `tags.yaml` (the controlled tag vocabulary), and `kbtool.py` (checker and index generator)
live at `skydiscover/synthesize/workflow/scripts/kb/`. Read `schema.yaml` and `tags.yaml` before
writing. A page's `domain:` is the folder it lives in; a tag must be in the shipped `tags.yaml` or
in the domain's own `<kb>/<domain>/wiki/tags.yaml` (same `tags:` shape). A word this kind of system
needs goes there; the shipped list is part of the framework and is not edited.

## Steps

Nine steps, in order. Steps 1 and 2 gather sourced evidence. Steps 3 to 8 distill it into reusable
knowledge. Step 9 builds the indexes. Then validate.

### Step 1: Pick the Reference Systems

Goal: enough real, respected systems that every design axis ends up with at least two genuinely
different choices across them (the same minimum `kbtool.py` enforces). Stop adding systems when
another one no longer changes any axis.

**If started with a run's discovered sources, adopt them instead of searching.** The anchor is the
run's clones: for each `<run>/specification/sources/<name>/`, take the pinned commit from
`git -C sources/<name> rev-parse HEAD`. Read its `owner/repo` from
`<run>/specification/references/<name>/spec.json` when that file exists, otherwise from the clone
(`git -C sources/<name> remote get-url origin`); never skip a cloned system for lacking a
`spec.json`. Treat that `owner/repo @ sha` set as fixed, write each `sources/repo-<name>.md` from
it, and go to Step 2. If the run has `<run>/specification/references/literature.json`, each entry
there with a `url` becomes a `doc` page under `sources/`, cited from the claim it supports. The set
is small, so run at most one mining worker per adopted system and author the properties and tests
inline. Add a system only when an axis still lacks two different choices. `source-reported` tests
are enough to seed a cold knowledge base; do not block GREEN on `verified` tests or a fresh
profile.

Otherwise find them with `gh`:

```bash
gh search repos "<domain>" --sort stars --limit 30
gh search repos "<domain>" --sort updated --limit 30
```

`--limit` is a page size, not a coverage bound: keep writing pages while the keep-list is still growing.
Without `gh`, search the web for "best <domain> in production", recent surveys, and curated lists,
then confirm each candidate on GitHub.

Keep a system only if it clears every bar:

- **following**: a strong star count for this domain, judged relative to peers;
- **production**: shipped releases, named adopters, or a peer-reviewed paper;
- **recent**: still maintained, judged against the domain's own cadence (commits, releases, or
  maintainers answering issues). A finished, stable system qualifies while its issues get answers;
- **license**: permissive enough to read and cite.

Write one `sources/repo-<name>.md` per system, pinned to the current commit `sha` (`git ls-remote
<url> HEAD`), with `url`, `domain`, and `tags` from `tags.yaml`.

### Step 2: Mine the PRs and Issues

```bash
gh search prs --repo <owner>/<repo> --merged --sort updated --limit 50
gh search issues --repo <owner>/<repo> --sort reactions --limit 50
```

Same rule: `--limit` is a page size; keep writing pages while merged core-path PRs or real failure-mode
issues keep appearing. Without `gh`, read an adopted clone in place (`git -C sources/<name> log
--stat` over the core paths) rather than re-cloning; for a system not already cloned, clone
shallowly (`--depth 200` to start, deeper while relevant fixes still appear at the boundary) and
read its log, or fetch the rendered PR and issue pages. Either way the citation is a real URL.

Keep a PR if it is merged, touches the core paths, and carries a real change (not a typo or version
bump). Keep an issue if it names a real failure mode: a wrong result, a crash, data loss, or a
performance cliff.

Write `sources/pr-<repo>-<n>.md` and `sources/issue-<repo>-<n>.md`, with `tags` from `tags.yaml`
only. A PR also carries `changed_paths` (from `gh pr view <n> --json files`, or the
`repos/<owner>/<repo>/pulls/<n>/files` API). Validate every fetched JSON body and retry once on
garbage; a proxy error cached as data poisons every page built on it. A tag that does not exist goes
into `<domain>/wiki/tags.yaml` first.

### Step 3: Derive the Properties

Read across the sources and cluster what varies between systems. Each axis becomes one
`properties/prop-<domain>-<slug>.md` with `values` (the choices real systems make), `seen_in`
(which source chose which value), and `sources` (the pages that ground it). A property is what a
good system must decide, stated as a behavior, never a named mechanism.

### Step 4: Set the Benchmark

Write one `benchmarks/bench-<domain>-<slug>.md`: the standard workload for this domain, the metric
that scores it, and the baseline score from a named source or a measured reference run. This is
the single target every design is scored against; the baseline lives here and nowhere else.

### Step 5: Profile the Bottleneck

Measure, do not recall. The run's `specification/cards/environment.json` (the spec-builder's
environment mode) already names the resource that bounds this workload, its measured limit, and
the ceiling; write `profiling/roof-<domain>-<box>.md` from it, with the exact commands, the measured
`ceiling`, the `bound`, and `confidence`, citing the run as its source. Without an environment card,
measure it yourself the way the Spec Builder's environment mode does.

### Step 6: Author the Tests

For each checkable property write `tests/test-<domain>-<slug>.md` with the runnable code beside it
in `tests/` (in the domain's language, plus the mutant it must fail):

- `enforces`: the property id;
- `command`: the command that runs the check, run from `tests/`;
- `mutant`: the broken implementation the test must fail, named so a reader can find it;
- `confidence: verified` only after you ran the test and confirmed the mutant fails it. `kbtool.py`
  re-runs the `command` and rejects a `verified` test whose command does not pass. Until then it
  stays `source-reported`.

### Step 7: Catalog the Reward Hacks

Start from `<home>/shared/wiki/hacks/` if it exists (hacks true for every domain, accumulated by earlier runs; a
fresh knowledge base ships none, so create the folder when you find one) and confirm each is
defended here. A hack belongs in `shared/` only if its `tell` and its kill can be stated purely in
measurement or benchmark-methodology terms with no domain concept; if removing the domain nouns
breaks the description, it stays in `<domain>/hacks/`. Write domain-specific hacks as
`hacks/hack-<domain>-<slug>.md` with `tell` (how to detect it) and `killed_by` (the test id that
closes it). A hack with no test is a `GAP`; the index surfaces it so the next round writes the test.

### Step 8: Record Designs (Refresh Runs Only)

On a first build no run has finished, so skip this step. When the lead runs you on demand after a
run, read its result's `best/score.json` and `spec.md` (the `.done` marker beside the run directory
names the result folder) and write `designs/design-<domain>-<slug>.md` with `benchmark` (the bench
id), `score`, `outcome` (won or lost against the baseline), and the one `lesson` worth carrying
forward. The delta against the baseline is computed from the benchmark, never stored twice.

### Step 9: Build the Indexes

Never write a view by hand; the tool generates them from front matter:

```bash
python3 skydiscover/synthesize/workflow/scripts/kb/kbtool.py index
```

- `by-property.md`: each axis to the tests that enforce it and the hacks that attack it; `GAP` if
  no test enforces it.
- `by-test.md`: each test to the property it enforces and its confidence.
- `by-hack.md`: each hack to its killing test, or `GAP`.
- `by-source.md`: each source to what it grounds, or `ORPHAN` if nothing cites it.

It also refreshes `wiki/index.md`. `designs/` and `benchmarks/` get no view; they are short and
flat, so read them directly.

## Validate

```bash
python3 skydiscover/synthesize/workflow/scripts/kb/kbtool.py validate
```

It rejects:

- a page missing a field its kind requires (`schema.yaml`), or of a kind not listed there;
- a page with no `sources` (except `repo`, `pr`, `issue`, and `doc` pages, which are the sources and
  carry a `url`);
- a `verified` test whose own `command` does not pass when kbtool.py runs it from `wiki/tests/`;
- a tag not listed in `tags.yaml` or the domain's `wiki/tags.yaml`, or a page whose `domain` is not the folder it lives in;
- any internal reference to a page that does not exist;
- a PR or issue link in a page body with no backing source page;
- an em-dash in a page body.

Fix every failure and run it again until it prints `WIKI: OK`, then build the indexes.

## Output

A populated `<domain>/` with every folder filled, sourced, and indexed, plus one row in the
top-level `index.md`.

Shared rules for every role: `skydiscover/synthesize/workflow/SKILL.md`, "Rules That Hold Everywhere".
