# skydiscover

Source for the `skydiscover` command. The package is split into two tools that share the same CLI
entry point.

## Optimize

Use `optimize/` when you already have a scoring function and want search to improve a program.

```bash
uv run skydiscover optimize <program> <evaluator> --search evox
uv run skydiscover viewer <checkpoint>
```

See [`optimize/README.md`](optimize/README.md) for flags, evaluator formats, and the Python API.

## Synthesize

Use `synthesize/` when you want a full system built from a description or a formal spec, with
correctness enforced by deterministic tests.

```bash
uv run skydiscover init                 # wires /skysynth into every coding agent found on PATH
uv run skydiscover init --agent codex   # or just one
```

See [`synthesize/README.md`](synthesize/README.md) for the pipeline, supported agents, and how to
add a domain.

## Layout

```
skydiscover/
├── optimize/     improve a program with evolutionary search
├── synthesize/   build a whole system with coding agents
├── main.py       the `skydiscover` command
└── __init__.py   public Python API: run_discovery, discover_solution, Runner, DiscoveryResult
```
