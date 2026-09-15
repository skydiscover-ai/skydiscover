# State and Decisions

The workflow keeps enough state to resume a run, explain every enforced choice, and reuse verified
knowledge across runs without treating a cache as truth.

## The Decision Log

`<run>/decision_log.json` records requirements, findings, who decided each, and the test that
enforces a fix. `skydiscover/synthesize/spec/findings.py` is the schema; never edit the
JSON by hand.

```bash
python3 -m skydiscover.synthesize.spec.findings add <run> \
  --title "<label>" --kind spec|hack|overfit|measure --detail "<what>"

```

| Kind | Use for |
|---|---|
| `spec` | a missing requirement or an operational blocker |
| `hack` | a reward hack |
| `overfit` | behavior that only holds on the benchmark's exact workload |
| `measure` | an invalid or misleading measurement |

A **defect** violates an acknowledged correctness, safety, or robustness requirement. An
**advisory** is a deliberate trade-off or out-of-envelope behavior. The release check blocks while
a defect is open. A defect closes in one of two ways:

```bash
# the coding agent fixed it: the kept test that shows the fix, and the broken copy that test catches
# (paths relative to the run; both may already be on the row from `findings add`)
python3 -m skydiscover.synthesize.spec.findings set-status <run> \
  --id <id> --status waived --who ai --note "<the fix>" \
  --test <id>.<ext> --mutant synthesis/evaluator/mutants/<id>
# the user set it aside, in the chat; <row> is its number in `decisions <run> list`
python3 -m skydiscover.synthesize.spec.decisions <run> drop <row> --by human
```

The release check re-runs the recorded test against the delivered build and against the mutant, so
a fix is proven, not declared. A row's severity never changes: `findings add` records each row's
first severity in a hidden snapshot beside the log, and a defect relabelled by hand is refused at
release.

## Asking the User

The lead is a coding agent the user is talking to. It asks the user only what the sources, the
prompt, and the trace cannot settle, and only when the choice is both ambiguous and consequential;
it asks in domain language, as a short multiple-choice question in the chat, and records the reply
word for word. Everything else it decides itself, with a one-line reason, and records as its own. The
user can change any answer later by saying so in the chat; the lead records that too.

The record is the decision log, written through one command; `specification/answers.json` is
exported from it after every change, so there is nothing else to keep in sync.

```bash
python3 -m skydiscover.synthesize.spec.decisions <run> set <question-id> "<value>"            # what the user said
python3 -m skydiscover.synthesize.spec.decisions <run> set <question-id> "<value>" --by ai   # the lead's own call, with --note "<why>"
python3 -m skydiscover.synthesize.spec.decisions <run> answer                                # defaults for whatever is still open
python3 -m skydiscover.synthesize.spec.decisions <run> list                                  # every decision, who made it, where it stands
python3 -m skydiscover.synthesize.spec.decisions <run> drop <row> --by human                 # the user sets a row aside: an answer leaves the specification, a defect stops blocking
```

Every row says who decided it: the AI or the user. `set` without `--by ai` records the user's
answer, so the lead uses it only for what the user actually said. The user's answer is never
overwritten by an AI answer, in the run or in the knowledge base (`run finish` carries the user's answers into
`<kb>/decisions.json`). Automatic reuse requires the same project, question, and options; changed
or reworded questions need confirmation. The old answer remains in the log, not in the new requirements.

## The Knowledge Base: State Kept Across Runs

What outlives a run is the domain's knowledge base, one folder per domain under `~/.skydiscover/`
(`SKYDISCOVER_HOME`).
The domain is the `domain:` line of the run's `task.md`, spelled the one way `spec.paths domain`
spells it, so two runs on "KV Store" and "kv-store" share a folder.

| Path | Purpose | Written by |
|---|---|---|
| `<domain>/tests/` | kept tests from finished runs, the last run's `test.sh`, and `index.json` (property id, keywords) | `run finish`, automatically |
| `<domain>/decisions.json` | every answer the user gave and every confirmed reward hack, saved from decision logs | `run finish`, automatically |
| `<domain>/wiki/` | wiki pages: reference systems pinned to a commit, properties, benchmarks, profiling, tests, known hacks, designs | `kb-builder`, on a Standard or Thorough budget |
| `shared/wiki/` | pages that hold for every domain | `kb-builder` |
| `.cache/sources/` | cloned reference systems shared across runs; finishing one run keeps them | spec-builder (discovery) |

Tests and decisions are saved automatically: `run finish` writes them, and the next run's
evaluator and spec-builder read them. A matching test is a suggestion, not coverage, so check the
requirement's meaning and revalidate the test against the current reference and targeted mutants
before reusing it.

The wiki is written by an agent. A `sources/` page pinned to a repository's exact current commit
lets the spec-builder reuse that clone (`hooks/clone_reuse_guard.py` blocks the redundant one),
while a page pinned to an older commit is only a hint. The Auditor reads `hacks/` as the known
reward hacks. Roles read pages with `kbtool.py find <words> [--kind] [--tag]` and `kbtool.py page
<id> --follow-sources`, described in `workflow/scripts/kb/README.md`.

The knowledge base speeds a run up but is never a substitute for discovery and never required; a
thin or missing folder means fresh discovery. Run-local evidence stays in the run directory until
`run finish` deletes it, and what the user keeps is the result under `outputs/`.
