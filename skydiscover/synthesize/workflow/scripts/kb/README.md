# The Knowledge Base Wiki

What earlier runs in a domain learned, as markdown pages under `~/.skydiscover/<domain>/wiki/`.
The `kb-builder` agent writes them once for a new domain; later runs read them. A run works
without them.

```text
wiki/
├── sources/      repo-*, pr-*, issue-*, doc-*   what was read, pinned to a commit or URL
├── properties/   prop-*                         a question about the system and how real systems answer it
├── tests/        test-*                         a test for one property, with the command that runs it
├── hacks/        hack-*                         a reward hack seen before, and the test that catches it
├── benchmarks/   bench-*                        how the domain is scored
├── profiling/    roof-*                         how the bottleneck was measured
├── designs/      design-*                       what a finished run built and learned
└── index/        by-property, by-test, by-hack, by-source   generated cross-references
```

Each page starts with `key: value` lines: `id` (its file name), `kind` (its folder), `tags`, and
links to other pages (`sources`, `enforces`, `killed_by`, `seen_in`).

## Reading

```bash
K=skydiscover/synthesize/workflow/scripts/kb/kbtool.py
python3 $K find durability                        # rank pages by words
python3 $K find --kind hack --tag memory          # list a kind, narrow by tag
python3 $K page prop-kv-store-durability-window   # one page; --follow-sources adds what it links to
```

## Checking

```bash
python3 $K validate   # every page against schema.yaml and tags.yaml; prints WIKI: OK or each problem
python3 $K index      # rebuild index/
```

`schema.yaml` lists the header fields each kind of page needs; `tags.yaml` the allowed tags (a domain
adds its own in `<domain>/wiki/tags.yaml`). `kbtool.py` is the only thing that writes `index/`.
`--root <folder>` or `SKYDISCOVER_HOME` points at a knowledge base other than `~/.skydiscover`.
