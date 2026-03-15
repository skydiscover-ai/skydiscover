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
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
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
            script = (
                "#!/bin/bash\nset -euo pipefail\n"
                f"timeout {timeout} python3 /workspace/evaluator.py \"$1\"\n"
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
                f"timeout {timeout} docker exec \"$CID\" /benchmark/evaluate.sh \"$MODE\" \"$CANDIDATE\"\n"
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

        try:
            suffix = self.file_suffix
            solution_path = workspace / f"solution{suffix}"
            solution_path.write_text(initial_code)

            eval_path = Path(self.evaluation_file)
            is_docker_eval = eval_path.is_dir()

            if is_docker_eval:
                evaluator_image = self.evaluator.image_tag
                eval_timeout = self.config.evaluator.timeout
                self._write_eval_script(
                    workspace, "docker", evaluator_image=evaluator_image, timeout=eval_timeout
                )
                # Export the evaluator image for the DinD daemon to load.
                await loop.run_in_executor(
                    None, self._save_evaluator_image, workspace, evaluator_image
                )
            else:
                shutil.copy(eval_path, workspace / "evaluator.py")
                eval_timeout = self.config.evaluator.timeout
                self._write_eval_script(workspace, "python", timeout=eval_timeout)
                req = eval_path.parent / "requirements.txt"
                if req.exists():
                    shutil.copy(req, workspace / "requirements.txt")

            self._write_task_md(workspace, suffix, max_turns=max_turns)
            task_content = (workspace / "TASK.md").read_text()

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            log_path = workspace / "claude.log"

            pip_step = (
                "pip install -q --no-warn-script-location -r /workspace/requirements.txt"
                " >/dev/null 2>&1 && "
                if (workspace / "requirements.txt").exists() else ""
            )
            model_flag = f" --model {shlex.quote(model)}" if model else ""
            bash_cmd = (
                f"{pip_step}"
                f"claude -p {shlex.quote(task_content)}"
                f" --max-turns {max_turns}"
                f" --dangerously-skip-permissions"
                f" --output-format stream-json"
                f" --verbose"
                f"{model_flag}"
            )

            if is_docker_eval:
                # DinD mode: --privileged gives the container its own Docker
                # daemon. The entrypoint starts dockerd, loads the evaluator
                # image, then drops to a non-root user before running claude.
                # Write bash_cmd to a script file so it doesn't get mangled
                # by su -c argument expansion in the entrypoint.
                run_script = workspace / ".run.sh"
                run_script.write_text(f"#!/bin/bash\n{bash_cmd}\n")
                run_script.chmod(0o755)
                cmd = [
                    "docker", "run", "--rm",
                    "--privileged",
                    "-e", "DIND=1",
                    "-e", f"ANTHROPIC_API_KEY={api_key}",
                    "-v", f"{workspace}:/workspace",
                    "-w", "/workspace",
                    image_name,
                    "/workspace/.run.sh",
                ]
            else:
                # Simple mode: no Docker needed inside the container.
                cmd = [
                    "docker", "run", "--rm",
                    "--user", f"{os.getuid()}:{os.getgid()}",
                    "-e", "HOME=/workspace",
                    "-e", f"ANTHROPIC_API_KEY={api_key}",
                    "-v", f"{workspace}:/workspace",
                    "-w", "/workspace",
                    "--entrypoint", "bash",
                    image_name,
                    "-c", bash_cmd,
                ]

            logger.info(
                f"Starting Claude Code container (--max-turns {max_turns})\n"
                f"  Monitor progress: tail -f {log_path} | cclean -t"
                f"  (https://github.com/ariel-frischer/claude-clean)"
            )

            def _run_with_turn_limit() -> None:
                with open(log_path, "w") as log_file:
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=log_file
                    )
                    turn_count = 0
                    for raw_line in proc.stdout:
                        log_file.write(raw_line.decode("utf-8", errors="replace"))
                        log_file.flush()
                        if b'"type":"user"' in raw_line or b'"type": "user"' in raw_line:
                            turn_count += 1
                            logger.info(f"Claude Code turn {turn_count}/{max_turns}")
                            if turn_count >= max_turns:
                                logger.warning(
                                    f"Reached max turns ({max_turns}), killing Claude Code process"
                                )
                                proc.kill()
                                break
                    proc.wait()

            await loop.run_in_executor(None, _run_with_turn_limit)

            final_code = solution_path.read_text()
            program_id = str(uuid.uuid4())
            eval_result = await self.evaluator.evaluate_program(final_code, program_id)

            program = Program(
                id=program_id,
                solution=final_code,
                language=self.config.language or "python",
                metrics=eval_result.metrics,
                iteration_found=0,
                parent_id=initial.id if initial else None,
                other_context_ids=[],
                metadata={"claude_code_max_turns": max_turns},
                artifacts=eval_result.artifacts,
            )
            self.database.add(program, iteration=0)

        finally:
            shutil.rmtree(workspace, ignore_errors=True)

        return self.database.get_best_program()
