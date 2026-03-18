"""Claude Code baseline controller.

Runs Claude Code CLI inside a Docker container as a single-agent baseline,
letting it iterate on the solution using the evaluator directly.  The
framework's standard evaluator scores the final result.

Docker is always required: --dangerously-skip-permissions needs isolation.

For Docker evaluators, the container runs in --privileged DinD mode so
Claude Code has its own isolated Docker daemon (no host socket mount).
For Python evaluators, the container runs in simple --user mode.
"""

import asyncio
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from skydiscover.evaluation import create_evaluator
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)

logger = logging.getLogger(__name__)

_RUNNER_IMAGE_DIR = Path(__file__).parent / "runner_image"


class ClaudeCodeController(DiscoveryController):
    """Discovery controller that delegates iteration to Claude Code CLI."""

    def __init__(self, controller_input: DiscoveryControllerInput):
        # Skip the parent __init__ which creates LLMPools we don't need.
        self.config = controller_input.config
        self.evaluation_file = controller_input.evaluation_file
        self.database = controller_input.database
        self.file_suffix = controller_input.file_suffix
        self.output_dir = controller_input.output_dir

        self.config.evaluator.evaluation_file = self.evaluation_file
        self.config.evaluator.file_suffix = self.file_suffix
        self.config.evaluator.is_image_mode = self.config.language == "image"

        self.evaluator = create_evaluator(self.config.evaluator)

        self.monitor_callback = None
        self.feedback_reader = None
        self.early_stopping_triggered = False
        self.shutdown_event = None

    def _ensure_image_built(self, image_name: str) -> None:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
        )
        if result.returncode != 0:
            logger.info(f"Building Claude Code runner image '{image_name}'...")
            subprocess.run(
                ["docker", "build", "-t", image_name, str(_RUNNER_IMAGE_DIR)],
                check=True,
            )

    def _save_evaluator_image(self, workspace: Path, image_tag: str) -> None:
        """Export the evaluator Docker image so the DinD daemon can load it."""
        tar_path = workspace / ".evaluator-image.tar"
        logger.info(f"Saving evaluator image '{image_tag}' for DinD...")
        subprocess.run(
            ["docker", "save", "-o", str(tar_path), image_tag],
            check=True,
        )

    def _write_eval_script(
        self, workspace: Path, eval_type: str, evaluator_image: str = "", timeout: int = 360
    ) -> None:
        """Write run_eval.sh that Claude Code can call to score a solution."""
        if eval_type == "python":
            # evaluator.py may not have a __main__ block — call evaluate() directly
            # and print the result dict as JSON so Claude Code can parse it.
            script = (
                "#!/bin/bash\nset -euo pipefail\n"
                f"timeout {timeout} python3 - \"$1\" <<'PYEOF'\n"
                "import sys, json\n"
                "sys.path.insert(0, '/workspace')\n"
                "import evaluator\n"
                "result = evaluator.evaluate(sys.argv[1])\n"
                "print(json.dumps(result))\n"
                "PYEOF\n"
            )
        else:
            # Docker evaluator: the entrypoint starts a persistent evaluator
            # container and writes its ID to .evaluator-container-id. We use
            # docker exec (like ContainerizedEvaluator) to inject the program
            # via stdin and run evaluate.sh — no new container per eval call.
            script = (
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "PROGRAM_PATH=\"$1\"\n"
                "MODE=\"${2:-train}\"\n"
                "EXT=\"${PROGRAM_PATH##*.}\"\n"
                "CID=$(cat /workspace/.evaluator-container-id)\n"
                "CANDIDATE=\"/tmp/candidate_$$.${EXT}\"\n"
                "docker exec -i \"$CID\" tee \"$CANDIDATE\" < \"$PROGRAM_PATH\" > /dev/null\n"
                f"timeout {timeout} docker exec \"$CID\" /benchmark/evaluate.sh \"$CANDIDATE\" \"$MODE\"\n"
                "docker exec \"$CID\" rm -f \"$CANDIDATE\"\n"
            )
        script_path = workspace / "run_eval.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)

    def _write_task_md(self, workspace: Path, suffix: str, max_turns: int = 0) -> None:
        system_msg = getattr(self.config.context_builder, "system_message", "") or ""
        eval_timeout = self.config.evaluator.timeout
        content = (
            "# SkyDiscover: Optimization Task\n\n"
            f"You are an AI assistant iteratively improving a program to maximize its evaluation score. "
            f"You have **{max_turns} turns** total — be mindful of this budget and make sure to fully "
            f"implement and test your ideas before you run out of turns.\n\n"
            "## Current solution\n\n"
            f"`/workspace/solution{suffix}` — read it, understand it, modify it freely.\n\n"
            "## How to evaluate\n\n"
            "```bash\n"
            f"bash /workspace/run_eval.sh /workspace/solution{suffix}\n"
            "```\n\n"
            "Output is JSON. The `combined_score` field is what you want to maximize "
            f"(higher is better). The evaluator has a **{eval_timeout}s timeout** — "
            f"if your solution takes longer than that, it will be killed and score zero.\n\n"
            "## Task description\n\n"
            f"{system_msg}\n\n"
            "## Instructions\n\n"
            "- Run the evaluator once to confirm the baseline score, then **start making changes immediately**.\n"
            "- Don't spend more than 1-2 turns analyzing — get to trying improvements fast.\n"
            "- After each change, evaluate and decide whether to keep or revert it.\n"
            f"- Always keep `/workspace/solution{suffix}` set to your best solution so far.\n"
            "- Aim to try several distinct approaches within your turn budget.\n"
        )
        (workspace / "TASK.md").write_text(content)

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Optional[Program]:
        db_config = self.database.config
        image_name = getattr(db_config, "docker_image", "skydiscover-claude-code:latest")
        max_turns = max_iterations

        model = self.config.llm.models[0].name if self.config.llm.models else None
        _CLAUDE_MODEL_PREFIXES = ("claude-", "sonnet", "opus", "haiku")
        if model and not any(model.startswith(p) for p in _CLAUDE_MODEL_PREFIXES):
            raise ValueError(
                f"claude_code only supports Claude models, got: {model!r}. "
                f"Set llm.models[0].name to a claude-* model or alias (sonnet, opus, haiku)."
            )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._ensure_image_built, image_name)

        initial = self.database.get_best_program()
        initial_code = initial.solution if initial else ""

        tmp_base = os.path.expanduser("~/.tmp")
        os.makedirs(tmp_base, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(dir=tmp_base))
        container_name = f"skydiscover-cc-{uuid.uuid4().hex[:12]}"

        try:
            suffix = self.file_suffix
            solution_path = workspace / f"solution{suffix}"
            solution_path.write_text(initial_code)

            eval_path = Path(self.evaluation_file)
            is_docker_eval = eval_path.is_dir()
            eval_timeout = self.config.evaluator.timeout

            if is_docker_eval:
                evaluator_image = self.evaluator.image_tag
                self._write_eval_script(
                    workspace, "docker", evaluator_image=evaluator_image, timeout=eval_timeout
                )
                # Export the evaluator image for the DinD daemon to load.
                await loop.run_in_executor(
                    None, self._save_evaluator_image, workspace, evaluator_image
                )
            else:
                shutil.copy(eval_path, workspace / "evaluator.py")
                self._write_eval_script(workspace, "python", timeout=eval_timeout)
                req = eval_path.parent / "requirements.txt"
                if req.exists():
                    shutil.copy(req, workspace / "requirements.txt")

            self._write_task_md(workspace, suffix, max_turns=max_turns)
            task_content = (workspace / "TASK.md").read_text()

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Export it before running: export ANTHROPIC_API_KEY=sk-ant-..."
                )
            log_path = workspace / "claude.log"

            # Write prompt to a file — inlining it in bash -c breaks on
            # backticks and quotes in the TASK.md content.
            (workspace / ".prompt.txt").write_text(task_content)

            model_flag = f"--model {shlex.quote(model)} " if model else ""

            # Write a launcher script.  The prompt is fed via stdin
            # (< .prompt.txt) to avoid shell quoting issues.
            script_lines = ["#!/bin/bash"]
            if (workspace / "requirements.txt").exists():
                # Best-effort pip install — don't abort if it fails.
                script_lines.append(
                    "pip install -q --no-warn-script-location"
                    " -r /workspace/requirements.txt >/dev/null 2>&1 || true"
                )
            script_lines.append(
                f"exec claude -p - "
                f"--max-turns {max_turns} "
                f"--dangerously-skip-permissions "
                f"--output-format stream-json "
                f"--verbose "
                f"{model_flag}"
                f"< /workspace/.prompt.txt"
            )
            run_script = workspace / ".run.sh"
            run_script.write_text("\n".join(script_lines) + "\n")
            run_script.chmod(0o755)

            if is_docker_eval:
                cmd = [
                    "docker", "run", "--rm",
                    "--name", container_name,
                    "--privileged",
                    "-e", "DIND=1",
                    "-e", f"ANTHROPIC_API_KEY={api_key}",
                    "-v", f"{workspace}:/workspace",
                    "-w", "/workspace",
                    image_name,
                    "/workspace/.run.sh",
                ]
            else:
                cmd = [
                    "docker", "run", "--rm",
                    "--name", container_name,
                    "--user", f"{os.getuid()}:{os.getgid()}",
                    "-e", "HOME=/workspace",
                    "-e", f"ANTHROPIC_API_KEY={api_key}",
                    "-v", f"{workspace}:/workspace",
                    "-w", "/workspace",
                    "--entrypoint", "bash",
                    image_name,
                    "/workspace/.run.sh",
                ]

            # Wall-clock timeout: allow full eval timeout + 2 min thinking per turn.
            # This ensures the turn budget is never cut short by the wall clock.
            wall_timeout = max(max_turns * (120 + eval_timeout), 600)

            out = Path(self.output_dir) if self.output_dir else None
            progress_log = (out / "progress.log") if out else None
            if out:
                out.mkdir(parents=True, exist_ok=True)

            # Lock ensures progress.log writes are safe whether called from the
            # executor thread (_run_with_turn_limit) or the async checkpoint loop.
            _progress_lock = threading.Lock()

            def _write_progress(line: str) -> None:
                """Append a timestamped line to progress.log and emit as INFO."""
                ts = time.strftime("%H:%M:%S")
                entry = f"[{ts}] {line}"
                logger.info(entry)
                if progress_log:
                    with _progress_lock:
                        with open(progress_log, "a") as f:
                            f.write(entry + "\n")

            _write_progress(
                f"Run started — model={model or 'default'}, "
                f"max_turns={max_turns}, wall_timeout={wall_timeout}s"
            )
            logger.info(
                f"Starting Claude Code container '{container_name}' "
                f"(--max-turns {max_turns}, wall timeout {wall_timeout}s)\n"
                f"  Monitor progress: tail -f {progress_log or log_path}"
            )

            # Shared state modified only from the executor thread.
            cumulative_turns = 0
            total_cost_usd = 0.0
            stream_turns = 0  # live tool-use turn count; fallback when result never fires
            run_start = time.monotonic()

            def _run_with_turn_limit() -> None:
                nonlocal cumulative_turns, total_cost_usd, stream_turns
                start = time.monotonic()
                hard_stop_at = 0.0  # monotonic time when hard stop was triggered
                with open(log_path, "w") as log_file:
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=log_file
                    )
                    try:
                        for raw_line in proc.stdout:
                            log_file.write(raw_line.decode("utf-8", errors="replace"))
                            log_file.flush()

                            try:
                                evt = json.loads(raw_line)
                            except (json.JSONDecodeError, ValueError):
                                pass
                            else:
                                evt_type = evt.get("type")

                                if evt_type == "assistant":
                                    elapsed = time.monotonic() - start
                                    content = evt.get("message", {}).get("content", [])
                                    tool_names = [
                                        c.get("name", "")
                                        for c in content
                                        if c.get("type") == "tool_use"
                                    ]
                                    if tool_names:
                                        # Each tool-use assistant message = one turn.
                                        stream_turns += 1
                                        _write_progress(
                                            f"Active → {', '.join(tool_names)}"
                                            f" (elapsed {elapsed:.0f}s,"
                                            f" turn {stream_turns}/{max_turns})"
                                        )
                                        if stream_turns > max_turns and not hard_stop_at:
                                            # Budget exceeded: don't SIGKILL immediately.
                                            # Keep reading so the result event (with
                                            # authoritative turn count and cost) can
                                            # arrive before we kill.  Force-kill after
                                            # 30 s if result never comes.
                                            hard_stop_at = time.monotonic()
                                            _write_progress(
                                                f"Hard stop: stream turn {stream_turns}"
                                                f" exceeded {max_turns} — waiting for result"
                                            )

                                elif evt_type == "result":
                                    seg_turns = evt.get("num_turns", 0)
                                    cumulative_turns += seg_turns
                                    seg_cost = evt.get("total_cost_usd", 0) or 0
                                    if seg_cost > total_cost_usd:
                                        total_cost_usd = seg_cost
                                    subtype = evt.get("subtype", "")
                                    _write_progress(
                                        f"Segment done ({subtype}): "
                                        f"+{seg_turns} turns, "
                                        f"{cumulative_turns}/{max_turns} cumulative, "
                                        f"cost=${total_cost_usd:.4f}"
                                    )
                                    if cumulative_turns >= max_turns or hard_stop_at:
                                        _write_progress(
                                            f"Turn budget reached — stopping"
                                        )
                                        proc.kill()
                                        break

                            # Hard stop grace-period expired: force kill.
                            if hard_stop_at and time.monotonic() - hard_stop_at > 30:
                                _write_progress(
                                    f"Hard stop grace period elapsed — force killing"
                                )
                                proc.kill()
                                break

                            # Wall-clock safety net.
                            elapsed = time.monotonic() - start
                            if elapsed > wall_timeout:
                                _write_progress(
                                    f"Wall timeout ({wall_timeout}s) exceeded — stopping"
                                )
                                proc.kill()
                                break
                    finally:
                        proc.wait()
                        # Drain any bytes left in the stdout pipe after we broke
                        # from the reading loop (e.g. the result event emitted by
                        # Claude Code just as the hard stop fired).  Capturing it
                        # here gives us authoritative turn count and cost.
                        try:
                            for remaining_line in proc.stdout:
                                log_file.write(remaining_line.decode("utf-8", errors="replace"))
                                log_file.flush()
                                try:
                                    evt = json.loads(remaining_line)
                                    if evt.get("type") == "result":
                                        seg_turns = evt.get("num_turns", 0)
                                        cumulative_turns += seg_turns
                                        seg_cost = evt.get("total_cost_usd", 0) or 0
                                        if seg_cost > total_cost_usd:
                                            total_cost_usd = seg_cost
                                except (json.JSONDecodeError, ValueError):
                                    pass
                        except OSError:
                            pass
                        _write_progress(
                            f"Process exited (code {proc.returncode}),"
                            f" cumulative turns: {cumulative_turns}"
                        )

            # Run process in a thread; poll solution file for checkpoints.
            run_future = loop.run_in_executor(None, _run_with_turn_limit)
            last_ckpt_content = initial_code
            ckpt_count = 0
            ckpt_interval = self.config.checkpoint_interval

            while not run_future.done():
                await asyncio.sleep(60)
                # Check if solution file changed → evaluate + checkpoint.
                try:
                    cur = solution_path.read_text()
                except OSError:
                    continue
                if cur == last_ckpt_content or not cur.strip():
                    continue
                last_ckpt_content = cur
                ckpt_count += 1
                # Use ckpt_count as the iteration proxy so checkpoint_callback
                # fires reliably even during a single long segment where
                # cumulative_turns stays 0 until the segment ends.
                iteration = max(cumulative_turns, ckpt_count)
                try:
                    pid = str(uuid.uuid4())
                    er = await self.evaluator.evaluate_program(cur, pid)
                    prog = Program(
                        id=pid,
                        solution=cur,
                        language=self.config.language or "python",
                        metrics=er.metrics,
                        iteration_found=iteration,
                        parent_id=initial.id if initial else None,
                        other_context_ids=[],
                        metadata={"claude_code_checkpoint_turn": cumulative_turns},
                        artifacts=er.artifacts,
                    )
                    self.database.add(prog, iteration=iteration)
                    score = er.metrics.get("combined_score", "?")
                    _write_progress(f"[CHECKPOINT] turn ~{cumulative_turns}, score={score}")
                    if checkpoint_callback and ckpt_count % ckpt_interval == 0:
                        checkpoint_callback(iteration)
                except Exception:
                    logger.debug("Checkpoint eval failed", exc_info=True)

            await run_future  # propagate exceptions

            # actual_turns: prefer authoritative count from result events; fall back
            # to the live stream_turns count (used when process was hard-killed before
            # result fired).
            actual_turns = cumulative_turns if cumulative_turns > 0 else stream_turns

            # cost: comes from result event; if process was hard-killed scan log for
            # any result event that may have been written before death.
            if total_cost_usd == 0.0:
                try:
                    for line in log_path.read_text(errors="replace").splitlines():
                        try:
                            evt = json.loads(line)
                            if evt.get("type") == "result":
                                c = evt.get("total_cost_usd", 0) or 0
                                if c > total_cost_usd:
                                    total_cost_usd = c
                                if cumulative_turns == 0:
                                    actual_turns = max(actual_turns, evt.get("num_turns", 0))
                        except (json.JSONDecodeError, ValueError):
                            pass
                except OSError:
                    pass

            # Final evaluation: re-evaluate the solution on disk.
            # If it times out or errors (e.g. solution too slow), fall back to the
            # best program already in the database from mid-run checkpoints —
            # that score is already verified and reliable.
            try:
                final_code = solution_path.read_text()
            except OSError:
                final_code = initial_code
            if not final_code.strip():
                final_code = initial_code

            final_iter = max(actual_turns, 1)
            final_score_source = "final_eval"
            program_id = str(uuid.uuid4())
            try:
                eval_result = await self.evaluator.evaluate_program(final_code, program_id)
                if eval_result.metrics.get("timeout") or eval_result.metrics.get("combined_score") is None:
                    raise ValueError("Final eval timed out or returned no score")
            except Exception as e:
                # Fall back to best checkpoint score already in the database.
                logger.warning(f"Final eval failed ({e}), falling back to best checkpoint score")
                best = self.database.get_best_program()
                if best and best.metrics and best.metrics.get("combined_score") is not None:
                    eval_result = type("R", (), {"metrics": best.metrics, "artifacts": best.artifacts or {}})()
                    final_code = best.solution
                    program_id = best.id
                    final_score_source = "best_checkpoint"
                else:
                    eval_result = type("R", (), {"metrics": {}, "artifacts": {}})()
                    final_score_source = "none"

            if final_score_source == "final_eval":
                program = Program(
                    id=program_id,
                    solution=final_code,
                    language=self.config.language or "python",
                    metrics=eval_result.metrics,
                    iteration_found=final_iter,
                    parent_id=initial.id if initial else None,
                    other_context_ids=[],
                    metadata={
                        "claude_code_max_turns": max_turns,
                        "actual_turns": actual_turns,
                    },
                    artifacts=eval_result.artifacts,
                )
                self.database.add(program, iteration=final_iter)

            if checkpoint_callback:
                checkpoint_callback(final_iter)

            # Preserve claude.log and write run summary.
            run_elapsed = time.monotonic() - run_start
            if out:
                try:
                    shutil.copy(log_path, out / "claude.log")
                except OSError:
                    pass
                summary = {
                    "model": model,
                    "max_turns": max_turns,
                    "actual_turns": actual_turns,
                    "cost_usd": round(total_cost_usd, 4),
                    "wall_seconds": round(run_elapsed, 1),
                    "baseline_score": initial.metrics.get("combined_score") if initial and initial.metrics else None,
                    "final_score": eval_result.metrics.get("combined_score"),
                    "final_score_source": final_score_source,
                    "final_metrics": eval_result.metrics,
                }
                (out / "run_summary.json").write_text(
                    json.dumps(summary, indent=2, default=str) + "\n"
                )
                _write_progress(
                    f"Run complete: turns={actual_turns}/{max_turns}, "
                    f"cost=${total_cost_usd:.4f}, "
                    f"time={run_elapsed:.0f}s, "
                    f"score={eval_result.metrics.get('combined_score', '?')}"
                    f" (source={final_score_source})"
                )

        finally:
            # Kill container if still running (timeout, exception, etc.).
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
            )
            shutil.rmtree(workspace, ignore_errors=True)

        return self.database.get_best_program()
