from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from skydiscover.evaluation.evaluation_result import EvaluationResult

PROMPT_SUFFIX = (
    "\n\nUse the available file-writing tools to create the required files under "
    "/workspace/output. Do not finish with only an explanation; before exiting, "
    "verify that /workspace/output/bo_algorithm.py and /workspace/output/summary.md exist."
)
DEFAULT_S3_PREFIX = "s3://lila-ps-kepler/opencode-sandbox/skydiscover"
SCORE_KEYS = (
    "combined_score",
    "normalized_score",
    "score",
    "improvement",
    "best_improvement",
    "best_value",
    "final_value",
    "value",
)


@dataclass(frozen=True)
class Settings:
    benchmarks: list[str]
    project: str
    domain: str
    queue: str
    kube_context: str
    flyte_config: str
    output_s3_prefix: str
    model: str
    n_bo_steps: int
    wandb_status: str
    wandb_project: str
    download_timeout_seconds: int | None
    poll_interval_seconds: int
    aggregation: str
    score_key: str | None
    failure_score: float
    chariot_bin: str
    source_root: str | None
    pyproject_path: str | None
    pyproject_url: str | None
    aws_profile: str | None


@dataclass(frozen=True)
class BenchmarkOutcome:
    benchmark: str
    score: float
    success: bool
    source_key: str | None = None
    metric: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    summary: str | None = None


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _split_names(value: str) -> list[str]:
    return [part for part in re.split(r"[\s,]+", value.strip()) if part]


def _benchmark_names_from_package() -> list[str]:
    module = importlib.import_module("opencode_sandbox.benchmark_project")
    return list(module.default_benchmark_names())


def _settings() -> Settings:
    benchmark_env = os.environ.get("AGENTICSEQLEARN_BENCHMARK_NAMES")
    benchmarks = _split_names(benchmark_env) if benchmark_env else _benchmark_names_from_package()
    aggregation = _env("AGENTICSEQLEARN_SCORE_AGGREGATION", "mean").lower()
    if aggregation not in {"mean", "median"}:
        raise ValueError("AGENTICSEQLEARN_SCORE_AGGREGATION must be 'mean' or 'median'")

    return Settings(
        benchmarks=benchmarks,
        project=_env("AGENTICSEQLEARN_PROJECT", "buzz"),
        domain=_env("AGENTICSEQLEARN_DOMAIN", "production"),
        queue=_env("AGENTICSEQLEARN_QUEUE", "phys-sci"),
        kube_context=_env("AGENTICSEQLEARN_KUBE_CONTEXT", "buzz"),
        flyte_config=os.path.expanduser(
            _env("AGENTICSEQLEARN_FLYTE_CONFIG", "~/.config/chariot/flyte/houston.yaml")
        ),
        output_s3_prefix=_env("AGENTICSEQLEARN_OUTPUT_S3_PREFIX", DEFAULT_S3_PREFIX).rstrip("/"),
        model=_env("AGENTICSEQLEARN_MODEL", "openai/gpt-5.5"),
        n_bo_steps=int(_env("AGENTICSEQLEARN_N_BO_STEPS", "200")),
        wandb_status=_env("AGENTICSEQLEARN_WANDB_STATUS", "disabled"),
        wandb_project=_env("AGENTICSEQLEARN_WANDB_PROJECT", "kepler-agenticseqlearn"),
        download_timeout_seconds=_optional_int_env("AGENTICSEQLEARN_DOWNLOAD_TIMEOUT_SECONDS"),
        poll_interval_seconds=int(_env("AGENTICSEQLEARN_POLL_INTERVAL_SECONDS", "30")),
        aggregation=aggregation,
        score_key=os.environ.get("AGENTICSEQLEARN_SCORE_KEY"),
        failure_score=float(_env("AGENTICSEQLEARN_FAILURE_SCORE", "-1000000000.0")),
        chariot_bin=_env("AGENTICSEQLEARN_CHARIOT_BIN", "chariot"),
        source_root=os.environ.get("AGENTICSEQLEARN_SOURCE_ROOT"),
        pyproject_path=os.environ.get("AGENTICSEQLEARN_PYPROJECT_PATH"),
        pyproject_url=os.environ.get("AGENTICSEQLEARN_PYPROJECT_URL"),
        aws_profile=os.environ.get("AGENTICSEQLEARN_AWS_PROFILE")
        or os.environ.get("AWS_PROFILE")
        or "ml-ops-dev",
    )


def _workflow_path() -> Path:
    module = importlib.import_module("workflows.opencode_sandbox")
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError("Could not resolve workflows.opencode_sandbox.__file__")
    return Path(module_file).resolve()


def _copytree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _direct_url_metadata() -> tuple[str, str] | None:
    try:
        direct_url = importlib.metadata.distribution("ps-agenticseqlearn").read_text(
            "direct_url.json"
        )
    except importlib.metadata.PackageNotFoundError:
        return None
    if not direct_url:
        return None
    payload = json.loads(direct_url)
    repo_url = str(payload.get("url", "")).removesuffix(".git")
    commit = str(payload.get("vcs_info", {}).get("commit_id", ""))
    if not repo_url or not commit:
        return None
    return repo_url, commit


def _direct_url_file_url(filename: str) -> str | None:
    metadata = _direct_url_metadata()
    if not metadata:
        return None
    repo_url, commit = metadata
    match = re.fullmatch(r"https://github.com/([^/]+)/([^/]+)", repo_url)
    if not match:
        return None
    owner, repo = match.groups()
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{filename}"


def _uv_cache_pyproject() -> Path | None:
    metadata = _direct_url_metadata()
    if not metadata:
        return None
    _, commit = metadata
    short_commit = commit[:7]
    checkouts_root = Path.home() / ".cache" / "uv" / "git-v0" / "checkouts"
    if not checkouts_root.exists():
        return None
    for candidate in checkouts_root.glob(f"*/*/pyproject.toml"):
        if candidate.parent.name.startswith(short_commit):
            return candidate
    return None


def _copy_project_file(
    *,
    launch_root: Path,
    filename: str,
    contents: bytes,
) -> None:
    for destination in (
        launch_root / filename,
        launch_root / "src" / filename,
        launch_root / "src" / "workflows" / filename,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)


def _read_pyproject_bytes(settings: Settings, package_root: Path) -> bytes:
    if settings.pyproject_path:
        return Path(settings.pyproject_path).expanduser().read_bytes()
    if (package_root / "pyproject.toml").exists():
        return (package_root / "pyproject.toml").read_bytes()
    if (package_root.parent / "pyproject.toml").exists():
        return (package_root.parent / "pyproject.toml").read_bytes()
    cached_pyproject = _uv_cache_pyproject()
    if cached_pyproject:
        return cached_pyproject.read_bytes()

    pyproject_url = settings.pyproject_url or _direct_url_file_url("pyproject.toml")
    if not pyproject_url:
        raise RuntimeError(
            "Could not locate pyproject.toml for ps-agenticseqlearn. Set "
            "AGENTICSEQLEARN_PYPROJECT_PATH or AGENTICSEQLEARN_PYPROJECT_URL."
        )
    with urlopen(pyproject_url, timeout=60) as response:
        return response.read()


def _write_pyproject(settings: Settings, package_root: Path, launch_root: Path) -> None:
    pyproject_bytes = _read_pyproject_bytes(settings, package_root)
    (launch_root / "pyproject.toml").write_bytes(pyproject_bytes)
    readme_path = launch_root / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "# ps-agenticseqlearn\n\nTemporary Flyte launch tree generated by SkyDiscover.\n",
            encoding="utf-8",
        )


def _patch_images_for_pyproject_packages(launch_root: Path) -> None:
    images_path = launch_root / "src" / "images.py"
    images_text = images_path.read_text(encoding="utf-8")
    helper = """

def _image_packages() -> list[str]:
    import re
    import tomllib
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    packages = []
    for dependency in pyproject.get("project", {}).get("dependencies", []):
        match = re.match(r"([A-Za-z0-9_.-]+)", dependency)
        name = match.group(1) if match else dependency
        source = sources.get(name)
        if isinstance(source, dict) and source.get("git"):
            rev = source.get("rev")
            git_url = source["git"]
            suffix = f"@{rev}" if rev else ""
            packages.append(f"{name} @ git+{git_url}{suffix}")
        else:
            packages.append(dependency)
    return packages
"""
    if "def _image_packages()" not in images_text:
        images_text = images_text.replace(
            "from flytekit import ImageSpec\n", f"from flytekit import ImageSpec\n{helper}\n"
        )
    images_text = images_text.replace('requirements="uv.lock"', "packages=_image_packages()")
    images_text = images_text.replace(
        'requirements: str = "uv.lock",',
        "requirements: str | None = None,\n    packages: Sequence[str] | None = None,",
    )
    images_text = images_text.replace(
        "        requirements=requirements,\n",
        "        requirements=requirements,\n"
        "        packages=list(_image_packages() if packages is None else packages),\n",
    )
    images_path.write_text(images_text, encoding="utf-8")


def _patch_task_body_to_return_error_artifacts(launch_root: Path) -> None:
    task_body_path = launch_root / "src" / "workflows" / "opencode_sandbox_task_body.py"
    task_body = task_body_path.read_text(encoding="utf-8")
    if "returning error artifacts for downstream scoring" in task_body:
        return

    patched, replacements = re.subn(
        r'(?P<indent>\s*)log_task_event\(f"uploaded \{uploaded_count\} error artifact file\(s\)"\)\n'
        r"(?P=indent)raise\n",
        r'\g<indent>log_task_event(f"uploaded {uploaded_count} error artifact file(s)")\n'
        r'\g<indent>log_task_event("returning error artifacts for downstream scoring")\n'
        r"\g<indent>return FlyteDirectory(path=str(output_dir))\n",
        task_body,
    )
    if replacements != 1:
        return
    task_body_path.write_text(patched, encoding="utf-8")


def _copy_lockfile_if_present(package_root: Path, launch_root: Path) -> bool:
    for candidate in (package_root / "uv.lock", package_root.parent / "uv.lock"):
        if candidate.exists():
            shutil.copy2(candidate, launch_root / "uv.lock")
            return True
    return False


def _prepare_launch_tree(
    settings: Settings,
    root: Path,
    package_root: Path,
    *,
    patch_image_packages: bool,
) -> Path:
    launch_root = root / "launch_source"
    launch_src = launch_root / "src"
    launch_src.mkdir(parents=True, exist_ok=True)

    for directory_name in ("workflows", "opencode_sandbox", "marcopolo_integration", "configs"):
        _copytree(package_root / directory_name, launch_src / directory_name)
    shutil.copy2(package_root / "images.py", launch_src / "images.py")
    _write_pyproject(settings, package_root, launch_root)
    if patch_image_packages:
        _patch_images_for_pyproject_packages(launch_root)
    else:
        _copy_lockfile_if_present(package_root, launch_root)
    _patch_task_body_to_return_error_artifacts(launch_root)
    return launch_src / "workflows" / "opencode_sandbox.py"


def _prepare_package_launch_tree(settings: Settings, root: Path) -> Path:
    package_workflow_path = _workflow_path()
    return _prepare_launch_tree(
        settings,
        root,
        package_workflow_path.parents[1],
        patch_image_packages=True,
    )


def _prepare_workflow_path(settings: Settings, root: Path) -> Path:
    if settings.source_root:
        source_root = Path(settings.source_root).expanduser().resolve()
        return _prepare_launch_tree(
            settings,
            root,
            source_root / "src",
            patch_image_packages=False,
        )
    return _prepare_package_launch_tree(settings, root)


def _candidate_hash(program_path: str) -> str:
    return hashlib.sha256(Path(program_path).read_bytes()).hexdigest()[:12]


def _write_project_input(program_path: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "program.md").write_text(Path(program_path).read_text(encoding="utf-8"))


def _run_chariot(
    *,
    settings: Settings,
    workflow_path: Path,
    project_input_dir: Path,
    output_s3_uris: list[str],
    candidate_prompt: str,
    candidate_program_md: str,
) -> tuple[str, str]:
    prompts = [candidate_prompt] * len(settings.benchmarks)
    program_mds = [candidate_program_md] * len(settings.benchmarks)
    command = [
        settings.chariot_bin,
        "flyte",
        "run",
        "--project",
        settings.project,
        "--domain",
        settings.domain,
        "--kube-context",
        settings.kube_context,
        "--queue",
        settings.queue,
        "--copy",
        "auto",
        "--destination-dir",
        "/root/src",
        "--entity",
        "opencode_sandbox_benchmarks_wf",
        str(workflow_path),
        "--prompt",
        json.dumps(prompts),
        "--output_s3_uri",
        json.dumps(output_s3_uris),
        "--model",
        settings.model,
        "--benchmark_name",
        json.dumps(settings.benchmarks),
        "--program_md",
        json.dumps(program_mds),
        "--n_bo_steps",
        str(settings.n_bo_steps),
        "--wandb_status",
        settings.wandb_status,
        "--wandb_project",
        settings.wandb_project,
        "--project_input_dir",
        str(project_input_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=workflow_path.parent.parent,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout
    if completed.returncode != 0:
        raise RuntimeError(f"chariot failed with exit code {completed.returncode}\n{output}")
    execution_id = _parse_execution_id(output)
    return execution_id, output


@contextmanager
def _aws_profile_env(settings: Settings):
    if not settings.aws_profile:
        yield
        return

    old_profile = os.environ.get("AWS_PROFILE")
    old_load_config = os.environ.get("AWS_SDK_LOAD_CONFIG")
    os.environ["AWS_PROFILE"] = settings.aws_profile
    os.environ.setdefault("AWS_SDK_LOAD_CONFIG", "1")
    try:
        yield
    finally:
        if old_profile is None:
            os.environ.pop("AWS_PROFILE", None)
        else:
            os.environ["AWS_PROFILE"] = old_profile
        if old_load_config is None:
            os.environ.pop("AWS_SDK_LOAD_CONFIG", None)
        else:
            os.environ["AWS_SDK_LOAD_CONFIG"] = old_load_config


def _parse_execution_id(output: str) -> str:
    patterns = (
        r"Submitted:\s*([^\s]+)",
        r"Execution(?:\s+ID)?:\s*([a-zA-Z0-9_.:-]+)",
        r"executions/([a-zA-Z0-9_.:-]+)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, output)
        if matches:
            return matches[-1]
    raise RuntimeError(f"Could not parse Flyte execution id from chariot output:\n{output}")


def _download_outputs(settings: Settings, execution_id: str, destination: Path) -> None:
    module = importlib.import_module("opencode_sandbox.flyte_artifacts")
    with _aws_profile_env(settings):
        module.download_execution_output(
            execution_id=execution_id,
            destination=destination,
            config_file=settings.flyte_config,
            project=settings.project,
            domain=settings.domain,
            output_name="o0",
            timeout_seconds=settings.download_timeout_seconds,
            poll_interval_seconds=settings.poll_interval_seconds,
            subdirs=settings.benchmarks,
        )


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"Expected S3 bucket and key in {uri!r}")
    return bucket, key.rstrip("/")


def _download_explicit_s3_outputs(
    settings: Settings,
    output_s3_uris: list[str],
    destination: Path,
) -> None:
    import boto3
    from botocore.exceptions import ClientError

    session_kwargs = {"profile_name": settings.aws_profile} if settings.aws_profile else {}
    client = boto3.Session(**session_kwargs).client("s3")
    artifact_names = [
        "evaluation_result.json",
        "evaluation_error.json",
        "evaluation_error.txt",
        "evaluation_summary.md",
        "summary.md",
        "bo_algorithm.py",
        "opencode_stdout.txt",
        "opencode_stderr.txt",
        "opencode_run.json",
        "output_manifest.json",
        "project_manifest.json",
        "sandbox_error.json",
        "sandbox_error.txt",
        "evaluation_observations.json",
        "evaluation_observations.csv",
    ]

    downloaded_count = 0
    for benchmark, uri in zip(settings.benchmarks, output_s3_uris):
        bucket, prefix = _parse_s3_uri(uri)
        benchmark_dir = destination / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        for artifact_name in artifact_names:
            key = f"{prefix}/{artifact_name}"
            local_path = benchmark_dir / artifact_name
            try:
                client.download_file(bucket, key, str(local_path))
                downloaded_count += 1
            except ClientError as error:
                code = str(error.response.get("Error", {}).get("Code", ""))
                if code in {"403", "404", "NoSuchKey", "NotFound"}:
                    continue
                raise

    if downloaded_count == 0:
        raise RuntimeError("Could not download any artifacts from explicit output_s3_uri prefixes")


def _recursive_numeric(data: Any, key: str) -> float | None:
    if isinstance(data, dict):
        if key in data and isinstance(data[key], int | float):
            return float(data[key])
        for value in data.values():
            found = _recursive_numeric(value, key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _recursive_numeric(item, key)
            if found is not None:
                return found
    return None


def _metric_direction(benchmark: str, metric: str | None) -> str:
    if not metric:
        return "maximize"
    try:
        registry = importlib.import_module("lila.benchmarking_gyms.registry")
        gym = registry.make_default_gym(benchmark)
        for candidate in getattr(gym.output_space, "metrics", []):
            if str(getattr(candidate, "name", "")) == metric:
                return str(getattr(candidate, "direction", "maximize"))
    except Exception:
        return "maximize"
    return "maximize"


def _score_needs_negation(source_key: str, metric_direction: str, metric_key: str | None) -> bool:
    key = source_key.lower()
    if any(token in key for token in ("regret", "loss", "error", "rmse", "distance")):
        return True
    if metric_key and source_key == metric_key and metric_direction == "minimize":
        return True
    if key in {"best_value", "final_value", "value"} and metric_direction == "minimize":
        return True
    return False


def _extract_score(
    result: dict[str, Any],
    *,
    benchmark: str,
    preferred_key: str | None,
) -> tuple[float, str, str | None]:
    metric = str(result.get("metric")) if result.get("metric") is not None else None
    keys = [preferred_key] if preferred_key else []
    keys.extend(key for key in SCORE_KEYS if key not in keys)
    if metric and metric not in keys:
        keys.append(metric)

    for key in keys:
        if not key:
            continue
        value = _recursive_numeric(result, key)
        if value is None:
            continue
        direction = _metric_direction(benchmark, metric)
        if _score_needs_negation(key, direction, metric):
            value = -value
        return value, key, metric

    raise ValueError("No numeric score field found in evaluation_result.json")


def _read_text(path: Path, limit: int = 2000) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) > limit:
        return text[:limit] + "\n... (truncated)"
    return text


def _parse_outcome(benchmark: str, directory: Path, settings: Settings) -> BenchmarkOutcome:
    result_path = directory / "evaluation_result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        score, source_key, metric = _extract_score(
            result,
            benchmark=benchmark,
            preferred_key=settings.score_key,
        )
        summary = _read_text(directory / "evaluation_summary.md") or _read_text(
            directory / "summary.md"
        )
        return BenchmarkOutcome(
            benchmark=benchmark,
            score=score,
            success=True,
            source_key=source_key,
            metric=metric,
            result=result,
            summary=summary,
        )

    error = _read_text(directory / "evaluation_error.txt")
    if error is None and (directory / "evaluation_error.json").exists():
        error = _read_text(directory / "evaluation_error.json")
    if error is None:
        error = _read_text(directory / "sandbox_error.txt")
    if error is None and (directory / "sandbox_error.json").exists():
        error = _read_text(directory / "sandbox_error.json")
    if error is None:
        error = _read_text(directory / "opencode_stderr.txt")
    if error is None:
        error = f"Missing evaluation_result.json in {directory}"
    return BenchmarkOutcome(
        benchmark=benchmark,
        score=settings.failure_score,
        success=False,
        error=error,
        summary=_read_text(directory / "summary.md"),
    )


def _aggregate_scores(outcomes: list[BenchmarkOutcome], settings: Settings) -> float:
    scores = [outcome.score for outcome in outcomes]
    if not scores:
        return settings.failure_score
    if settings.aggregation == "median":
        return float(statistics.median(scores))
    return float(statistics.fmean(scores))


def _feedback(
    *,
    execution_id: str | None,
    output_dir: Path | None,
    outcomes: list[BenchmarkOutcome],
    command_output: str | None = None,
    error: str | None = None,
) -> str:
    lines = ["# Flyte Evaluation Feedback", ""]
    if execution_id:
        lines.append(f"- Flyte execution: `{execution_id}`")
    if output_dir:
        lines.append(f"- Downloaded artifacts: `{output_dir}`")
    if command_output:
        lines.extend(["", "## Chariot Output", command_output.strip()[-2000:]])
    if error:
        lines.extend(["", "## Evaluation Error", error.strip()[-12000:]])
    if outcomes:
        success_count = sum(1 for outcome in outcomes if outcome.success)
        lines.extend(
            [
                "",
                "## Benchmark Results",
                f"- Successful benchmarks: {success_count}/{len(outcomes)}",
            ]
        )
        for outcome in outcomes:
            status = "ok" if outcome.success else "failed"
            source = f" from `{outcome.source_key}`" if outcome.source_key else ""
            lines.append(f"- {outcome.benchmark}: {status}, score={outcome.score:.6g}{source}")
            if outcome.error:
                lines.append(f"  Error: {outcome.error[:800]}")
            elif outcome.summary:
                lines.append(f"  Summary: {outcome.summary[:800]}")
            output_dir_for_benchmark = output_dir / outcome.benchmark if output_dir else None
            if output_dir_for_benchmark:
                opencode_run = _read_text(
                    output_dir_for_benchmark / "opencode_run.json", limit=1200
                )
                opencode_stderr = _read_text(
                    output_dir_for_benchmark / "opencode_stderr.txt",
                    limit=1200,
                )
                if opencode_run:
                    lines.append(f"  OpenCode run: {opencode_run[:1200]}")
                if opencode_stderr:
                    lines.append(f"  OpenCode stderr: {opencode_stderr[:1200]}")
    feedback = "\n".join(lines)
    if len(feedback) <= 6000:
        return feedback
    return feedback[:1500] + "\n... (truncated; showing tail) ...\n" + feedback[-4500:]


def _failure_result(message: str, settings: Settings | None = None) -> EvaluationResult:
    failure_score = settings.failure_score if settings else 0.0
    return EvaluationResult(
        metrics={
            "combined_score": failure_score,
            "success_rate": 0.0,
            "failure_count": 1.0,
            "benchmark_count": 0.0,
        },
        artifacts={
            "feedback": _feedback(execution_id=None, output_dir=None, outcomes=[], error=message)
        },
    )


def evaluate(program_path: str) -> EvaluationResult:
    try:
        settings = _settings()
    except Exception as error:
        return _failure_result(f"Could not initialize evaluator settings: {error}")

    if not settings.benchmarks:
        return _failure_result("No benchmarks configured for evaluation", settings)

    try:
        candidate_program_md = Path(program_path).read_text(encoding="utf-8")
        candidate_prompt = candidate_program_md + PROMPT_SUFFIX
        candidate_hash = _candidate_hash(program_path)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{candidate_hash}-{timestamp}"
        output_s3_uris = [
            f"{settings.output_s3_prefix}/{run_name}/{benchmark}/"
            for benchmark in settings.benchmarks
        ]

        with tempfile.TemporaryDirectory(prefix="skydiscover-agenticseqlearn-") as tmp:
            root = Path(tmp)
            project_input_dir = root / "project"
            local_output_dir = root / "outputs"
            _write_project_input(program_path, project_input_dir)
            workflow_path = _prepare_workflow_path(settings, root)

            execution_id, command_output = _run_chariot(
                settings=settings,
                workflow_path=workflow_path,
                project_input_dir=project_input_dir,
                output_s3_uris=output_s3_uris,
                candidate_prompt=candidate_prompt,
                candidate_program_md=candidate_program_md,
            )
            try:
                _download_outputs(settings, execution_id, local_output_dir)
            except Exception:
                _download_explicit_s3_outputs(settings, output_s3_uris, local_output_dir)

            outcomes = [
                _parse_outcome(benchmark, local_output_dir / benchmark, settings)
                for benchmark in settings.benchmarks
            ]
            combined_score = _aggregate_scores(outcomes, settings)
            success_count = sum(1 for outcome in outcomes if outcome.success)
            metrics = {
                "combined_score": combined_score,
                "success_rate": success_count / len(outcomes),
                "failure_count": float(len(outcomes) - success_count),
                "benchmark_count": float(len(outcomes)),
            }
            for outcome in outcomes:
                safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", outcome.benchmark).strip("_")
                metrics[f"score_{safe_name}"] = outcome.score

            return EvaluationResult(
                metrics=metrics,
                artifacts={
                    "feedback": _feedback(
                        execution_id=execution_id,
                        output_dir=local_output_dir,
                        outcomes=outcomes,
                        command_output=command_output,
                    ),
                    "execution_id": execution_id,
                },
            )
    except Exception as error:
        return _failure_result(str(error), settings)


if __name__ == "__main__":
    default_program = Path(__file__).with_name("initial_program.md")
    print(json.dumps(evaluate(str(default_program)).to_dict(), indent=2))
