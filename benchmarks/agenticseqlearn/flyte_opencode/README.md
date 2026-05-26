# AgenticSeqLearn Flyte OpenCode

This benchmark evolves the `program.md` instructions used by the
`ps-agenticseqlearn` OpenCode sandbox. Each SkyDiscover candidate is submitted to
the real mapped Flyte workflow, equivalent to `make run-opencode-sandbox-benchmarks`,
and the returned benchmark artifacts are aggregated into `combined_score`.

## Setup

Install `ps-agenticseqlearn` into the same Python environment used to run
SkyDiscover. The evaluator imports the installed package modules instead of
referencing a local checkout.

```bash
uv pip install "ps-agenticseqlearn @ git+https://github.com/fl97inc/ps-agenticseqlearn.git@7e301b0b9eb16ad90a29e62fdc8b771b1e15b396"
```

Configure Flyte/chariot access as you would for `ps-agenticseqlearn`.

## Environment

The evaluator is configured with environment variables:

- `AGENTICSEQLEARN_BENCHMARK_NAMES`: comma- or space-separated benchmark names.
  If unset, the evaluator uses `opencode_sandbox.benchmark_project.default_benchmark_names()`.
- `AGENTICSEQLEARN_PROJECT`: Flyte project. Default: `buzz`.
- `AGENTICSEQLEARN_DOMAIN`: Flyte domain. Default: `production`.
- `AGENTICSEQLEARN_QUEUE`: Flyte queue. Default: `phys-sci`.
- `AGENTICSEQLEARN_KUBE_CONTEXT`: Kubernetes context passed to `chariot`.
  Default: `buzz`.
- `AGENTICSEQLEARN_FLYTE_CONFIG`: Flyte config used for artifact download.
  Default: `~/.config/chariot/flyte/houston.yaml`.
- `AGENTICSEQLEARN_OUTPUT_S3_PREFIX`: S3 prefix for workflow artifacts.
  Default: `s3://lila-ps-kepler/opencode-sandbox/skydiscover`.
- `AGENTICSEQLEARN_MODEL`: OpenCode model used inside the sandbox.
  Default: `openai/gpt-5.5`.
- `AGENTICSEQLEARN_N_BO_STEPS`: BO steps per benchmark. Default: `200`.
- `AGENTICSEQLEARN_WANDB_STATUS`: W&B mode. Default: `disabled`.
- `AGENTICSEQLEARN_WANDB_PROJECT`: W&B project. Default: `kepler-agenticseqlearn`.
- `AGENTICSEQLEARN_DOWNLOAD_TIMEOUT_SECONDS`: remote wait timeout. Default: no timeout.
- `AGENTICSEQLEARN_POLL_INTERVAL_SECONDS`: artifact polling interval. Default: `30`.
- `AGENTICSEQLEARN_SCORE_AGGREGATION`: `mean` or `median`. Default: `mean`.
- `AGENTICSEQLEARN_SCORE_KEY`: optional explicit key to use from each
  `evaluation_result.json`.
- `AGENTICSEQLEARN_FAILURE_SCORE`: score assigned to failed benchmark tasks.
  Default: `-1000000000.0`.
- `AGENTICSEQLEARN_CHARIOT_BIN`: chariot executable. Default: `chariot`.
- `AGENTICSEQLEARN_AWS_PROFILE`: AWS profile used when downloading Flyte/S3
  output artifacts. Default: `AWS_PROFILE` if set, otherwise `ml-ops-dev`.
- `AGENTICSEQLEARN_SOURCE_ROOT`: optional local `ps-agenticseqlearn` checkout.
  If unset, the evaluator builds a temporary launch tree from the installed package.
- `AGENTICSEQLEARN_PYPROJECT_PATH` or `AGENTICSEQLEARN_PYPROJECT_URL`: optional
  override for the `pyproject.toml` used by the temporary launch tree. When the
  package was installed from GitHub, the evaluator fetches the matching
  `pyproject.toml` automatically from the package's `direct_url.json` metadata.

For early smoke tests, set `AGENTICSEQLEARN_BENCHMARK_NAMES=negative_ackley` and
use `-i 1`.

The temporary launch tree uses `pyproject.toml` both for the local `uv run
pyflyte` environment and for the staged Flyte image dependency list. The remote
image build still needs valid credentials for the Lila package index, typically
through the `.netrc` mounted by the `ps-agenticseqlearn` `ImageSpec`.

## Run

```bash
uv run skydiscover-run \
  benchmarks/agenticseqlearn/flyte_opencode/initial_program.md \
  benchmarks/agenticseqlearn/flyte_opencode/evaluator.py \
  -c benchmarks/agenticseqlearn/flyte_opencode/config_adaevolve.yaml \
  --search adaevolve \
  -i 10
```
