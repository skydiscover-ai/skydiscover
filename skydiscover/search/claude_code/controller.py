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
                # Write the evaluator container ID so run_eval.sh can
                # docker exec into the already-running evaluator container.
                eval_cid = self.evaluator.container_id
                (workspace / ".evaluator-container-id").write_text(eval_cid)
                eval_timeout = self.config.evaluator.timeout
                self._write_eval_script(
                    workspace, "docker", timeout=eval_timeout
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
            # Put the log in output_dir so it survives workspace cleanup.
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                log_path = Path(self.output_dir) / "claude.log"
            else:
                log_path = workspace / "claude.log"

            pip_step = (
                "pip install -q --no-warn-script-location"
                " -r /workspace/requirements.txt && "
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

            # Run as the built-in 'claude' user (Claude Code refuses
            # --dangerously-skip-permissions as root). pip install runs
            # first as root, then we drop to the claude user.
            #
            # We write two scripts to avoid nested quoting issues:
            # .run.sh     — runs as root: installs deps, fixes perms, drops to claude
            # .claude.sh  — runs as claude: the actual claude CLI invocation
            claude_script = workspace / ".claude.sh"
            claude_script.write_text(
                f"#!/bin/bash\nset -euo pipefail\n"
                f"export HOME=/workspace\n"
                f"claude -p {shlex.quote(task_content)}"
                f" --max-turns {max_turns}"
                f" --dangerously-skip-permissions"
                f" --output-format stream-json"
                f" --verbose"
                f"{model_flag}\n"
            )
            claude_script.chmod(0o755)

            docker_sock_step = ""
            if is_docker_eval:
                # Make the host Docker socket accessible to the non-root claude user.
                docker_sock_step = "chmod 666 /var/run/docker.sock 2>/dev/null || true\n"

            run_script = workspace / ".run.sh"
            run_script.write_text(
                f"#!/bin/bash\nset -euo pipefail\n"
                f"{pip_step}"
                f"{docker_sock_step}"
                f"chown -R claude:claude /workspace 2>/dev/null || chmod -R 777 /workspace 2>/dev/null || true\n"
                f"exec su --preserve-environment -s /bin/bash claude -c /workspace/.claude.sh\n"
            )
            run_script.chmod(0o755)
            cmd = [
                "docker", "run", "--rm",
                "-e", f"ANTHROPIC_API_KEY={api_key}",
                "-v", f"{workspace}:/workspace",
                "-w", "/workspace",
                "--entrypoint", "bash",
                image_name,
                "/workspace/.run.sh",
            ]
            if is_docker_eval:
                # Mount host Docker socket so Claude can docker exec
                # into the evaluator container. Add the socket's group
                # so the non-root claude user has permission.
                import stat as stat_mod
                sock_gid = os.stat("/var/run/docker.sock").st_gid
                cmd[3:3] = [
                    "-v", "/var/run/docker.sock:/var/run/docker.sock",
                    "--group-add", str(sock_gid),
                ]

            logger.info(
                f"Starting Claude Code container (--max-turns {max_turns})\n"
                f"  Monitor progress: tail -f {log_path} | cclean -t\n"
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
