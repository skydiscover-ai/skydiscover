# agent-pipeline/ -- Core Agent Evolution Loop

Drives Claude/Codex agents to iteratively generate, build, test, and benchmark
KV store implementations.

## Files

| File | Purpose |
|------|---------|
| `run.sh` | CLI entry point (parses args → orchestrator.py) |
| `orchestrator.py` | Main loop: prompt → agent → build → test → bench → feedback |
| `format_feedback.py` | Parse benchmark output into agent-readable feedback |
| `critique.py` | Review critique agent (code quality feedback) |
| `critique_audit.py` | Adversarial audit agent (test-writing + gate) |
| `memory.py` | Codex session rotation (prevents 429 TPM cascades) |
| `analyze.py` | Post-iteration analysis |
| `checkpoint.py` | Track best implementation (leaderboard.json) |
| `prompts/` | Prompt templates for generator, reviewer, auditor |

## Usage

```bash
bash agent-pipeline/run.sh --backend claude --mode ltm --setup 50:50 --distribution zipf

bash agent-pipeline/run.sh --backend claude --mode ltm --setup 50:50 \
    --trace-load /path/to/load.dat --trace-run /path/to/run.dat
```

## Critique modes (`--critique`)

| Mode | Reviewer | Auditor |
|------|----------|---------|
| `off` | -- | -- |
| `review` | yes | -- |
| `audit` | -- | yes |
| `full` | yes | yes |
