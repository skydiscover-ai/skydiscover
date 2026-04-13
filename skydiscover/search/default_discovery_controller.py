"""
Discovery controller for running discovery processes.

Provides the default execution loop for discovery processes (sample → prompt → LLM → evaluate).
Subclasses only need to override ``run_discovery`` to change orchestration
(e.g. co-evolution interleaves solution and search-algorithm evolution).
"""

import asyncio
import logging
import multiprocessing as mp
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from skydiscover.config import Config
from skydiscover.context_builder.default import DefaultContextBuilder
from skydiscover.context_builder.evox import EvoxContextBuilder
from skydiscover.evaluation import create_evaluator
from skydiscover.evaluation.llm_judge import LLMJudge
from skydiscover.llm.base import LLMResponse
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.search.utils.discovery_utils import SerializableResult, build_image_content
from skydiscover.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryControllerInput:
    """Input to the discovery controller"""

    config: Config
    evaluation_file: str
    database: ProgramDatabase
    file_suffix: str = ".py"
    output_dir: Optional[str] = None
    evaluator_env_vars: Optional[Dict[str, str]] = None


class DiscoveryController:
    """
    Discovery controller with a default sequential execution strategy.

    Handles the full generate-evaluate cycle: prompt building, LLM calls,
    response parsing, evaluation, and result processing.

    The default ``run_discovery`` runs iterations sequentially.  Subclasses
    (e.g. CoEvolutionController) can override it for different orchestration
    while reusing the shared iteration primitives.
    """

    def __init__(self, controller_input: DiscoveryControllerInput):
        self.config = controller_input.config
        self.evaluation_file = controller_input.evaluation_file
        self.database = controller_input.database
        self.file_suffix = controller_input.file_suffix
        self.output_dir = controller_input.output_dir
        self.evaluator_env_vars = controller_input.evaluator_env_vars

        self.shutdown_event = mp.Event()
        self.early_stopping_triggered = False

        self.llms = LLMPool(self.config.llm.models)
        self.evaluator_llms = LLMPool(self.config.llm.evaluator_models)
        self.guide_llms = LLMPool(self.config.llm.guide_models)

        self._init_context_builder()

        self.config.evaluator.evaluation_file = self.evaluation_file
        self.config.evaluator.file_suffix = self.file_suffix
        self.config.evaluator.is_image_mode = self.config.language == "image"

        llm_judge = None
        if self.config.evaluator.llm_as_judge:
            ctx = DefaultContextBuilder(self.config)
            ctx.set_templates("evaluator_system_message")
            llm_judge = LLMJudge(self.evaluator_llms, ctx, self.database)

        self.evaluator = create_evaluator(
            self.config.evaluator,
            llm_judge=llm_judge,
            max_concurrent=max(self.config.max_parallel_iterations, 4),
            env_vars=controller_input.evaluator_env_vars,
        )

        self.agentic_generator = None
        if self.config.agentic.enabled:
            from skydiscover.llm.agentic_generator import AgenticGenerator

            self.agentic_generator = AgenticGenerator(self.llms, self.config.agentic)
            logger.info(f"Agentic mode enabled (codebase: {self.config.agentic.codebase_root})")

        self.num_context_programs = controller_input.config.search.num_context_programs

        self.monitor_callback: Optional[Callable] = None
        self.feedback_reader: Optional[Any] = None
        self._prompt_context: Dict[str, Any] = {}

        # Load evaluator/task description and inject into system message so
        # the LLM knows what problem to solve (especially for from-scratch).
        self._inject_evaluator_context()

        logger.info(
            f"DiscoveryController initialized: num_context_programs={self.num_context_programs}"
        )

    def close(self):
        """Release resources held by the evaluator (e.g. Docker containers)."""
        if hasattr(self.evaluator, "close"):
            self.evaluator.close()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _inject_evaluator_context(self):
        """Load evaluator/task description and prepend to the system message.

        For Harbor tasks this loads instruction.md; for containerized benchmarks
        it loads the evaluator source files. The content gives the LLM essential
        context about the problem it needs to solve.

        Controlled by ``evaluator.inject_evaluator_context`` (default False).
        """
        if not self.config.evaluator.inject_evaluator_context:
            return

        from skydiscover.search.utils.discovery_utils import load_evaluator_code

        task_description = load_evaluator_code(self.evaluation_file)
        if not task_description:
            return

        ctx = self.config.context_builder
        existing = ctx.system_message or ""
        # Prepend the task description so the LLM always sees it.
        ctx.system_message = (
            f"# Task Description\n\n{task_description}\n\n{existing}"
            if existing
            else f"# Task Description\n\n{task_description}"
        )

    def _init_context_builder(self):
        """Initialize the appropriate context builder based on config."""
        if getattr(self.config.context_builder, "template", "default") == "evox":
            self.context_builder = EvoxContextBuilder(self.config)
            template_name = "search_evolution_user_message"
            self.context_builder.set_templates(user_template=template_name)
        else:
            self.context_builder = DefaultContextBuilder(self.config)

    async def _call_llm(self, system_message: str, user_message: str, **kwargs) -> LLMResponse:
        """Call the LLM, using agentic mode if enabled (text-only)."""
        if self.agentic_generator and not kwargs.get("image_output"):
            text = await self.agentic_generator.generate(system_message, user_message)
            if text:
                return LLMResponse(text=text)
        return await self.llms.generate(
            system_message, [{"role": "user", "content": user_message}], **kwargs
        )

    # ------------------------------------------------------------------
    # Main discovery loop
    # ------------------------------------------------------------------

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
        post_process_result: Optional[bool] = True,
        retry_times: Optional[int] = 3,
    ) -> Optional[Union[Program, SerializableResult]]:
        """
        Run the discovery process.

        When ``config.max_parallel_iterations == 1`` (default), iterations
        run sequentially — same behaviour as before.

        When ``> 1``, up to *N* iterations run concurrently as asyncio
        tasks, bounded by a semaphore.  Generation and evaluation naturally
        overlap across iterations: while iteration *i* evaluates, iteration
        *i+1* can generate, and iteration *i+2* can sample.

        Args:
            start_iteration: The iteration to start from.
            max_iterations: The number of iterations to run.
            checkpoint_callback: Optional callback for checkpointing.
            post_process_result: If True, add results to the database and
                return the best Program.  If False, return the raw
                ``SerializableResult`` from the last iteration.
            retry_times: Number of retry attempts per iteration.

        Returns:
            Best ``Program`` found (post_process_result=True) or raw
            ``SerializableResult`` (post_process_result=False).
        """
        max_parallel = self.config.max_parallel_iterations

        if max_parallel > 1:
            return await self._run_discovery_parallel(
                start_iteration,
                max_iterations,
                checkpoint_callback,
                post_process_result,
                retry_times,
                max_parallel,
            )

        return await self._run_discovery_sequential(
            start_iteration,
            max_iterations,
            checkpoint_callback,
            post_process_result,
            retry_times,
        )

    # ------------------------------------------------------------------
    # Sequential loop (original behaviour, max_parallel_iterations=1)
    # ------------------------------------------------------------------

    async def _run_discovery_sequential(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
        post_process_result: Optional[bool] = True,
        retry_times: Optional[int] = 3,
    ) -> Optional[Union[Program, SerializableResult]]:
        total_iterations = start_iteration + max_iterations

        result = None
        for iteration in range(start_iteration, total_iterations):
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested, stopping discovery loop early")
                break

            try:
                result = await self._run_iteration(iteration, retry_times=retry_times)
                if result.error:
                    logger.warning(f"Iteration {iteration} failed: {result.error}")
                    continue

                if post_process_result:
                    self._process_iteration_result(result, iteration, checkpoint_callback)

            except Exception as e:
                logger.exception(f"Error in iteration {iteration}: {e}")

        if not post_process_result:
            return result

        return self._finalize_discovery()

    # ------------------------------------------------------------------
    # Parallel loop (max_parallel_iterations > 1)
    # ------------------------------------------------------------------

    async def _run_discovery_parallel(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
        post_process_result: Optional[bool] = True,
        retry_times: Optional[int] = 3,
        max_parallel: int = 4,
    ) -> Optional[Union[Program, SerializableResult]]:
        total_iterations = start_iteration + max_iterations
        sem = asyncio.Semaphore(max_parallel)
        pending: set = set()
        last_result: Optional[SerializableResult] = None

        logger.info(
            f"Parallel discovery: up to {max_parallel} iterations in flight "
            f"({start_iteration}..{total_iterations - 1})"
        )

        async def _bounded_iteration(iteration: int) -> Tuple[int, Optional[SerializableResult]]:
            """Run one iteration under the semaphore, then process its result.

            Result processing (database.add) happens here rather than being
            collected later so that subsequent iterations see the latest DB
            state as soon as the ``await`` inside ``_run_iteration`` yields.
            """
            async with sem:
                if self.shutdown_event.is_set():
                    return iteration, None
                try:
                    result = await self._run_iteration(iteration, retry_times=retry_times)
                except Exception as e:
                    logger.exception(f"Error in parallel iteration {iteration}: {e}")
                    return iteration, None

            # Process outside the semaphore — database.add() is sync and
            # completes atomically between await-points, so no lock needed.
            if result and not result.error and post_process_result:
                self._process_iteration_result(result, iteration, checkpoint_callback)
            elif result and result.error:
                logger.warning(f"Iteration {iteration} failed: {result.error}")

            return iteration, result

        for iteration in range(start_iteration, total_iterations):
            if self.shutdown_event.is_set():
                break

            task = asyncio.create_task(_bounded_iteration(iteration), name=f"iter_{iteration}")
            pending.add(task)
            task.add_done_callback(pending.discard)

            # When the pipeline is full, wait for at least one to finish
            # before scheduling more — this provides backpressure.
            if len(pending) >= max_parallel:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in done:
                    try:
                        _, res = t.result()
                        if res is not None:
                            last_result = res
                    except Exception as e:
                        logger.warning(
                            f"A task in parallel discovery failed with an exception: {e}"
                        )

        # Drain remaining tasks
        if pending:
            done, _ = await asyncio.wait(pending)
            for t in done:
                try:
                    _, res = t.result()
                    if res is not None:
                        last_result = res
                except Exception as e:
                    logger.warning(
                        f"A task in parallel discovery (drain) failed with an exception: {e}"
                    )

        if not post_process_result:
            return last_result

        return self._finalize_discovery()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _finalize_discovery(self) -> Optional[Program]:
        if self.shutdown_event.is_set():
            logger.info(
                f"✅ Discovery process completed "
                f"(search strategy = {self.database.name}) - Shutdown requested"
            )
        else:
            logger.info(
                f"✅ Discovery process completed "
                f"(search strategy = {self.database.name}) - Maximum iterations reached"
            )
        return self.database.get_best_program()

    # ------------------------------------------------------------------
    # Single-iteration primitives (shared by all controllers)
    # ------------------------------------------------------------------

    async def _run_from_scratch_iteration(self, iteration: int) -> SerializableResult:
        """Generate a first solution from scratch when the database is empty."""
        try:
            iteration_start = time.time()

            prompt = self.context_builder.build_prompt(current_program=None, context={})

            if self.feedback_reader:
                self.feedback_reader.set_current_prompt(prompt["system"])
                feedback = self.feedback_reader.read()
                if feedback:
                    prompt = self.feedback_reader.apply_feedback(prompt)

            llm_generation_time = 0.0
            llm_start = time.time()
            result = await self._call_llm(prompt["system"], prompt["user"])
            llm_generation_time = time.time() - llm_start
            llm_response = result.text
            if not llm_response:
                return SerializableResult(error="Empty LLM response", iteration=iteration)

            child_solution = parse_full_rewrite(llm_response, self.config.language)
            if not child_solution:
                return SerializableResult(
                    error="No valid solution in response",
                    iteration=iteration,
                    prompt=prompt,
                    llm_response=llm_response,
                )

            child_id = str(uuid.uuid4())
            eval_start = time.time()
            eval_result = await self.evaluator.evaluate_program(child_solution, child_id)
            eval_time = time.time() - eval_start

            child = Program(
                id=child_id,
                solution=child_solution,
                language=self.config.language,
                parent_id=None,
                metrics=eval_result.metrics,
                iteration_found=iteration,
                metadata={"changes": "Generated from scratch"},
                artifacts=eval_result.artifacts or {},
            )

            return SerializableResult(
                child_program_dict=child.to_dict(),
                parent_id=None,
                other_context_ids=[],
                iteration_time=time.time() - iteration_start,
                llm_generation_time=llm_generation_time,
                eval_time=eval_time,
                prompt=prompt,
                llm_response=llm_response,
                iteration=iteration,
            )
        except Exception as e:
            logger.exception(f"From-scratch generation failed: {e}")
            return SerializableResult(error=str(e), iteration=iteration)

    async def _run_iteration(
        self,
        iteration: int,
        retry_times: int = 1,
    ) -> SerializableResult:
        """Run a single generate-evaluate iteration."""
        try:
            if not self.database.programs:
                return await self._run_from_scratch_iteration(iteration)

            raw_parent, raw_context_programs = self.database.sample(
                num_context_programs=self.num_context_programs
            )

            # Normalize sample() result — databases may return plain or dict-wrapped
            if isinstance(raw_parent, dict):
                if len(raw_parent) != 1:
                    raise ValueError(
                        f"sample() must return exactly one parent, got {len(raw_parent)}"
                    )
                parent_info_key = list(raw_parent.keys())[0]
                parent = list(raw_parent.values())[0]
            else:
                parent_info_key = ""
                parent = raw_parent

            # Other context programs that are relevant
            if isinstance(raw_context_programs, dict):
                context_programs_dict = raw_context_programs
            else:
                context_programs_dict = {"": raw_context_programs}

            parent_info = (parent_info_key, parent.id)
            context_info = [
                (key, p.id) for key, programs in context_programs_dict.items() for p in programs
            ]
            context_program_ids = [
                p.id for programs in context_programs_dict.values() for p in programs
            ]

            logger.debug(
                f"Iteration {iteration}: parent {parent.id} ({parent_info_key}), "
                f"other_context_programs keys: {list(context_programs_dict.keys())}"
            )

            iteration_start = time.time()

            failed_attempts = []
            child_solution, child_id, child_metrics, llm_response, changes_summary = (
                None,
                None,
                None,
                None,
                None,
            )

            image_path = None  # set by image mode or evaluator
            eval_time = 0.0

            # Build prompt with parent and context programs
            for retry in range(retry_times):
                prompt = self._build_prompt(
                    current_program=raw_parent,
                    context_programs=context_programs_dict,
                    failed_attempts=failed_attempts,
                )

                if failed_attempts:
                    logger.info(
                        f"Retry {retry + 1}/{retry_times}: rebuilding prompt with {len(failed_attempts)} failed attempt(s)"
                    )

                # Apply human feedback (append or replace mode)
                if self.feedback_reader:
                    self.feedback_reader.set_current_prompt(prompt["system"])
                    feedback = self.feedback_reader.read()
                    if feedback:
                        prompt = self.feedback_reader.apply_feedback(prompt)
                        self.feedback_reader.log_usage(
                            iteration, feedback, self.feedback_reader.mode
                        )

                try:
                    llm_generation_time = 0.0
                    llm_start = time.time()
                    if self.config.language == "image":
                        child_id = str(uuid.uuid4())
                        user_content = build_image_content(
                            prompt["user"], parent, context_programs_dict
                        )
                        result = await self._call_llm(
                            prompt["system"],
                            user_content,
                            image_output=True,
                            output_dir=self._get_image_output_dir(),
                            program_id=child_id,
                        )
                        llm_response = result.text or ""
                        image_path = result.image_path
                        if image_path:
                            child_solution = result.text or "(image generated)"
                            changes_summary = "Image generation"
                            parse_error = None
                        else:
                            child_solution = None
                            changes_summary = None
                            parse_error = "VLM did not generate an image"
                    else:
                        result = await self._call_llm(prompt["system"], prompt["user"])
                        llm_response = result.text
                    llm_generation_time = time.time() - llm_start
                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
                    return SerializableResult(
                        error=f"LLM generation failed: {str(e)}",
                        iteration=iteration,
                        attempts_used=retry + 1,
                    )

                if self.config.language != "image":
                    # Text/code mode: parse LLM response
                    if llm_response is None:
                        return SerializableResult(
                            error="LLM returned None response",
                            iteration=iteration,
                            attempts_used=retry + 1,
                        )

                    child_solution, changes_summary, parse_error = self._parse_llm_response(
                        llm_response, parent.solution, iteration, retry + 1, retry_times
                    )

                    if child_solution and len(child_solution) > self.config.max_solution_length:
                        logger.warning(
                            "Generated solution exceeds maximum length (iteration=%s, attempt %s/%s): %s > %s",
                            iteration,
                            retry + 1,
                            retry_times,
                            len(child_solution),
                            self.config.max_solution_length,
                        )
                        parse_error = f"Generated solution exceeds maximum length ({len(child_solution)} > {self.config.max_solution_length})"
                        child_solution = None

                if parse_error:
                    failed_attempts.append(
                        {
                            "solution": child_solution or "",
                            "llm_response": llm_response,
                            "metrics": {},
                            "metadata": {
                                "error": parse_error,
                                "attempt_number": retry + 1,
                            },
                        }
                    )
                    if retry < retry_times - 1:
                        continue
                    logger.error(
                        "All %s retry attempts failed due to parse/validation error: %s",
                        retry_times,
                        parse_error,
                    )
                    return SerializableResult(
                        error=f"{parse_error} (after {retry_times} attempts)",
                        iteration=iteration,
                        prompt=prompt,
                        llm_response=llm_response,
                        attempts_used=retry_times,
                    )

                if self.config.language != "image":
                    child_id = str(uuid.uuid4())

                eval_input = image_path if self.config.language == "image" else child_solution
                eval_start = time.time()
                child_eval_result = await self.evaluator.evaluate_program(eval_input, child_id)
                eval_time = time.time() - eval_start
                child_metrics = child_eval_result.metrics
                # Extract image_path from evaluator metrics (non-image mode fallback)
                if not image_path:
                    image_path = (
                        child_metrics.pop("image_path", None)
                        if isinstance(child_metrics.get("image_path"), str)
                        else None
                    )

                if (
                    child_metrics.get("validity") in (0, -1)
                    or (
                        child_metrics.get("timeout") is True
                        and child_metrics.get("validity") is None
                    )
                    or (
                        child_metrics.get("combined_score") == 0
                        and child_metrics.get("error") is not None
                    )
                ):
                    error_msg = (
                        (
                            child_metrics.get("error")
                            if isinstance(child_metrics.get("error"), str)
                            else None
                        )
                        or child_metrics.get("error_message")
                        or "Evaluation failed (validity=0)"
                    )

                    logger.warning(
                        "Evaluation failed (attempt %s/%s): validity=%s, error=%s",
                        retry + 1,
                        retry_times,
                        child_metrics.get("validity"),
                        error_msg,
                    )
                    logger.debug(
                        "Failed solution (attempt %s/%s):\n%s",
                        retry + 1,
                        retry_times,
                        child_solution,
                    )

                    failed_attempts.append(
                        {
                            "solution": child_solution,
                            "metrics": child_metrics,
                            "metadata": {
                                "changes": changes_summary,
                                "parent_metrics": parent.metrics,
                                "error": error_msg,
                                "attempt_number": retry + 1,
                            },
                        }
                    )

                    if retry < retry_times - 1:
                        continue
                    logger.error(
                        "All %s retry attempts failed. Final error: %s", retry_times, error_msg
                    )
                    iteration_time = time.time() - iteration_start
                    failed_extra = {"failed_attempts": failed_attempts}
                    if image_path:
                        failed_extra["image_path"] = image_path
                    failed_child_program = self._create_child_program(
                        child_id=child_id,
                        child_solution=child_solution,
                        parent=parent,
                        context_program_ids=context_program_ids,
                        parent_info=parent_info,
                        context_info=context_info,
                        child_metrics=child_metrics or {},
                        iteration=iteration,
                        changes_summary=changes_summary,
                        extra_metadata=failed_extra,
                        artifacts=child_eval_result.artifacts,
                    )
                    return SerializableResult(
                        error=f"Evaluator failed after {retry_times} attempts: {error_msg}",
                        iteration=iteration,
                        child_program_dict=failed_child_program.to_dict(),
                        parent_id=parent.id,
                        other_context_ids=context_program_ids,
                        iteration_time=iteration_time,
                        llm_generation_time=llm_generation_time,
                        eval_time=eval_time,
                        prompt=prompt,
                        llm_response=llm_response,
                        attempts_used=retry_times,
                    )
                break

            extra_meta = {}
            if image_path:
                extra_meta["image_path"] = image_path
            child_program = self._create_child_program(
                child_id=child_id,
                child_solution=child_solution,
                parent=parent,
                context_program_ids=context_program_ids,
                parent_info=parent_info,
                context_info=context_info,
                child_metrics=child_metrics,
                iteration=iteration,
                changes_summary=changes_summary,
                extra_metadata=extra_meta if extra_meta else None,
                artifacts=child_eval_result.artifacts,
            )
            iteration_time = time.time() - iteration_start

            return SerializableResult(
                child_program_dict=child_program.to_dict(),
                parent_id=parent.id,
                other_context_ids=context_program_ids,
                iteration_time=iteration_time,
                llm_generation_time=llm_generation_time,
                eval_time=eval_time,
                prompt=prompt,
                llm_response=llm_response,
                iteration=iteration,
                attempts_used=retry + 1,
            )
        except Exception as e:
            logger.exception(f"Error in iteration {iteration}")
            return SerializableResult(error=str(e), iteration=iteration, attempts_used=1)

    # ------------------------------------------------------------------
    # Prompt / parsing / program creation helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context_programs: Union[List[Program], Dict[str, List[Program]]],
        failed_attempts: list,
    ) -> Dict[str, str]:
        """Build the prompt for LLM generation."""
        parent = (
            list(current_program.values())[0]
            if isinstance(current_program, dict)
            else current_program
        )
        db_stats = self._prompt_context.get("db_stats") or self.database.get_statistics()

        # Build context with parent program and any other relevant information
        context = {
            "program_metrics": parent.metrics,
            "other_context_programs": context_programs,
            "previous_programs": db_stats.get("previous_programs", []),
            "db_stats": db_stats,
        }
        for k, v in self._prompt_context.items():
            if k not in context:
                context[k] = v

        if failed_attempts:
            context["errors"] = failed_attempts

        return self.context_builder.build_prompt(current_program=current_program, context=context)

    def _parse_llm_response(
        self,
        llm_response: str,
        parent_solution: str,
        iteration: int,
        attempt: int,
        retry_times: int,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse LLM response to extract child solution.

        Returns:
            Tuple of (child_solution, changes_summary, parse_error)
        """
        if self.config.diff_based_generation:
            diff_blocks = extract_diffs(llm_response)
            if not diff_blocks:
                preview = llm_response[:2000] + (
                    "\n... (truncated) ..." if len(llm_response) > 2000 else ""
                )
                logger.warning(
                    "No valid diffs found in LLM response (iteration=%s, attempt %s/%s). "
                    "Expected SEARCH/REPLACE blocks. Preview:\n%s",
                    iteration,
                    attempt,
                    retry_times,
                    preview,
                )
                return None, None, "No valid diffs found in response"

            child_solution = apply_diff(parent_solution, llm_response)
            changes_summary = format_diff_summary(diff_blocks)

            if child_solution == parent_solution:
                logger.warning(
                    "Diff blocks found but none matched parent solution (iteration=%s, attempt %s/%s).",
                    iteration,
                    attempt,
                    retry_times,
                )
                return (
                    None,
                    None,
                    "Diff SEARCH blocks did not match parent solution - no changes applied",
                )

            return child_solution, changes_summary, None
        else:
            new_solution = parse_full_rewrite(llm_response, self.config.language)
            if not new_solution:
                logger.warning(
                    "No valid solution found in LLM response (iteration=%s, attempt %s/%s).",
                    iteration,
                    attempt,
                    retry_times,
                )
                return None, None, "No valid solution found in response"
            return new_solution, "Full rewrite", None

    def _create_child_program(
        self,
        child_id: str,
        child_solution: str,
        parent: Program,
        context_program_ids: list,
        parent_info: tuple,
        context_info: list,
        child_metrics: Dict[str, Any],
        iteration: int,
        changes_summary: Optional[str],
        extra_metadata: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> Program:
        """Create a child program with the given attributes."""
        metadata = {
            "changes": changes_summary,
            "parent_metrics": parent.metrics,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return Program(
            id=child_id,
            solution=child_solution,
            language=self.config.language,
            parent_id=parent.id,
            other_context_ids=context_program_ids,
            parent_info=parent_info,
            context_info=context_info,
            metrics=child_metrics,
            iteration_found=iteration,
            metadata=metadata,
            artifacts=artifacts or {},
        )

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    async def postprocess_result(
        self, result: SerializableResult, iteration_number: int, verbose: bool = True
    ):
        """
        Process the iteration result and return the best program from the database.

        Used by co-evolution where evaluation can be delayed.
        """
        self._process_iteration_result(
            result, iteration_number, checkpoint_callback=None, verbose=verbose
        )
        return self.database.get_best_program()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_image_output_dir(self) -> str:
        """Return the directory for saving VLM-generated images."""
        base = self.output_dir or "."
        d = os.path.join(base, "generated_images")
        os.makedirs(d, exist_ok=True)
        return d

    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        logger.info("Graceful shutdown requested...")
        self.shutdown_event.set()

    def _process_iteration_result(
        self,
        result: Any,
        iteration: int,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Process the result from a single iteration.

        Args:
            result: The iteration result to process.
            iteration: Current iteration number.
            checkpoint_callback: Optional callback for checkpoint intervals.
            verbose: If True, log progress and metrics; if False, suppress logging.
        """
        if result.error:
            if verbose:
                logger.warning(f"Iteration {iteration} failed: {result.error}")
            return

        program_class = getattr(self.database, "_program_class", Program)
        child_program = program_class(**result.child_program_dict)

        self.database.add(child_program, iteration=iteration)

        # Fire monitor callback (live dashboard)
        if self.monitor_callback:
            try:
                self.monitor_callback(child_program, iteration)
            except Exception:
                logger.debug("Monitor callback error", exc_info=True)

        if result.prompt:
            self.database.log_prompt(
                template_key=(
                    "full_rewrite_user_message"
                    if not self.config.diff_based_generation
                    else "diff_user_message"
                ),
                program_id=child_program.id,
                prompt=result.prompt,
                responses=[result.llm_response] if result.llm_response else [],
            )

        if verbose:
            logger.info(
                f"Iteration {iteration}: "
                f"Program {child_program.id} "
                f"(parent: {result.parent_id}) "
                f"completed in {result.iteration_time:.2f}s"
                f" (llm: {result.llm_generation_time:.2f}s,"
                f" eval: {result.eval_time:.2f}s)"
            )

        if iteration > 0 and iteration % self.config.checkpoint_interval == 0:
            if verbose:
                logger.info(f"[CHECKPOINT] Checkpoint interval reached at iteration {iteration}")

            self.database.log_status()
            if checkpoint_callback:
                checkpoint_callback(iteration)

        if child_program.metrics:
            if verbose:
                metrics_str = ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                    for k, v in child_program.metrics.items()
                )
                logger.info(f"Metrics: {metrics_str}")

            if not hasattr(self, "_warned_about_combined_score"):
                self._warned_about_combined_score = False

            if (
                "combined_score" not in child_program.metrics
                and not self._warned_about_combined_score
            ):
                if verbose:
                    logger.warning(
                        "⚠️  No 'combined_score' metric found in evaluation results. "
                        "Using 0.0 for discovery process guidance. "
                        "For better solution discovery results, please modify your evaluator to return a 'combined_score' "
                        "metric that properly weights different aspects of program performance."
                    )
                self._warned_about_combined_score = True

        if self.database.best_program_id == child_program.id and verbose:
            logger.info(f"🌟 New best solution found at iteration {iteration}")
