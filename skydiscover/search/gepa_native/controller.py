"""
GEPA Native Controller - Discovery loop with reflective prompting,
acceptance gating, and LLM-mediated merge.

Implements GEPA's three core ideas as a SkyDiscover search strategy:
1. Reflective prompting — surfaces evaluator diagnostics and rejected-program
   code as actionable feedback in the LLM prompt.
2. Acceptance gating — rejects mutations that don't strictly improve on the
   parent, preventing population pollution.
3. LLM-mediated merge — combines two complementary programs both proactively
   (after each acceptance) and reactively (on stagnation).

Configuration options (via GEPANativeDatabaseConfig):
    acceptance_gating: Enable strict parent-improvement gating (default: True)
    use_merge: Enable LLM-mediated merge (default: True)
    merge_after_stagnation: Iterations without improvement before merge (default: 15)
    max_merge_attempts: Total merge budget per run (default: 10)
    max_recent_failures: Rejected programs shown in reflective section (default: 5)
"""

import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from skydiscover.context_builder.gepa_native import GEPANativeContextBuilder
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.utils.discovery_utils import SerializableResult
from skydiscover.utils.code_utils import parse_full_rewrite
from skydiscover.utils.metrics import get_score

logger = logging.getLogger(__name__)


class GEPANativeController(DiscoveryController):
    """
    Discovery controller implementing GEPA's guided evolution.

    Key Features:
    1. Reflective prompting: Structures evaluation failures and evaluator
       diagnostics as actionable feedback in the LLM prompt.
    2. Acceptance gating: Rejects mutations that don't improve on the parent,
       preventing population pollution.
    3. LLM-mediated merge: Combines two complementary programs both
       proactively (after each acceptance) and reactively (on stagnation).
    4. Merge deduplication: Tracks tried pairs to avoid redundant merges.
    5. Merge budget: Caps total merge attempts to bound LLM cost.
    """

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

        # Override context builder with GEPA-specific one
        self.context_builder = GEPANativeContextBuilder(self.config)

        db_config = self.config.search.database
        self.acceptance_gating: bool = getattr(db_config, "acceptance_gating", True)
        self.use_merge: bool = getattr(db_config, "use_merge", True)
        self.merge_after_stagnation: int = getattr(db_config, "merge_after_stagnation", 15)
        self.max_recent_failures: int = getattr(db_config, "max_recent_failures", 5)
        self.max_merge_attempts: int = getattr(db_config, "max_merge_attempts", 10)

        # Stagnation tracking
        self._best_score_seen: float = -float("inf")
        self._iterations_without_improvement: int = 0

        # Merge state
        self._merge_due: bool = False
        self._merge_attempts_used: int = 0
        self._merge_pairs_tried: Set[Tuple[str, str]] = set()

        logger.info(
            f"GEPANativeController initialized: "
            f"acceptance_gating={self.acceptance_gating}, "
            f"use_merge={self.use_merge}, "
            f"merge_after_stagnation={self.merge_after_stagnation}, "
            f"max_merge_attempts={self.max_merge_attempts}"
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
        """Run GEPA discovery with acceptance gating and merge scheduling.

        The loop follows this order each iteration:
        1. Proactive merge (if scheduled from a prior acceptance).
        2. Normal mutation via ``_run_iteration``.
        3. Acceptance gating — reject if child <= parent.
        4. If accepted: process result, schedule next proactive merge.
        5. Track improvement; trigger stagnation merge if stuck.
        """
        total_iterations = start_iteration + max_iterations

        # Initialise best-score from database if resuming
        best = self.database.get_best_program()
        if best and best.metrics:
            self._best_score_seen = get_score(best.metrics)

        result = None
        for iteration in range(start_iteration, total_iterations):
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested, stopping discovery loop early")
                break

            try:
                # Proactive merge: attempt before normal iteration if scheduled
                if self._merge_due:
                    self._merge_due = False
                    await self._attempt_merge(iteration)

                result = await self._run_iteration(iteration, retry_times=retry_times)

                if result.error:
                    logger.warning(f"Iteration {iteration} failed: {result.error}")
                    self._iterations_without_improvement += 1

                    if self._should_merge():
                        await self._attempt_merge(iteration)
                    continue

                # --- Acceptance gating (skip for from-scratch programs) ---
                if self.acceptance_gating and result.child_program_dict and result.parent_id:
                    accepted = self._acceptance_gate(result, iteration)
                    if not accepted:
                        if self._should_merge():
                            await self._attempt_merge(iteration)
                        continue

                # --- Accept: process normally ---
                if post_process_result:
                    self._process_iteration_result(result, iteration, checkpoint_callback)

                # Schedule a proactive merge after successful acceptance
                if self.use_merge and self._merge_attempts_used < self.max_merge_attempts:
                    self._merge_due = True

                # Track improvement
                if result.child_program_dict:
                    child_score = get_score(result.child_program_dict.get("metrics", {}))
                    if child_score > self._best_score_seen:
                        self._best_score_seen = child_score
                        self._iterations_without_improvement = 0
                    else:
                        self._iterations_without_improvement += 1
                else:
                    self._iterations_without_improvement += 1

                # Stagnation-triggered merge (fallback)
                if self._should_merge():
                    await self._attempt_merge(iteration)

            except Exception as e:
                logger.exception(f"Error in iteration {iteration}: {e}")
                self._iterations_without_improvement += 1

        if not post_process_result:
            return result

        if self.shutdown_event.is_set():
            logger.info(
                f"Discovery completed (search strategy = {self.database.name}) "
                "- Shutdown requested"
            )
        else:
            logger.info(
                f"Discovery completed (search strategy = {self.database.name}) "
                "- Maximum iterations reached"
            )

        return self.database.get_best_program()

    # ------------------------------------------------------------------
    # Reflective prompting (via GEPANativeContextBuilder)
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        current_program: Dict[str, Program],
        context_programs: Union[List, Dict[str, list]],
        failed_attempts: list,
    ) -> Dict[str, str]:
        """Build prompt with GEPA reflective context.

        Gathers rejection history from the database and passes it through
        the context dict so GEPANativeContextBuilder can format it into
        the {search_guidance} template placeholder.
        """
        parent = (
            list(current_program.values())[0]
            if isinstance(current_program, dict)
            else current_program
        )
        db_stats = self._prompt_context.get("db_stats") or self.database.get_statistics()

        # Gather rejection history for reflective prompting
        rejected = self.database.get_rejection_history(limit=self.max_recent_failures)

        # Pre-compute parent scores for rejected programs
        rejection_parent_scores: Dict[str, float] = {}
        for prog in rejected:
            if prog.parent_id and prog.parent_id in self.database.programs:
                p = self.database.programs[prog.parent_id]
                rejection_parent_scores[prog.parent_id] = get_score(p.metrics)

        context: Dict[str, Any] = {
            "program_metrics": parent.metrics,
            "other_context_programs": context_programs,
            "previous_programs": db_stats.get("previous_programs", []),
            "db_stats": db_stats,
            # GEPA-specific keys (consumed by GEPANativeContextBuilder)
            "rejection_history": rejected,
            "rejection_parent_scores": rejection_parent_scores,
        }
        for k, v in self._prompt_context.items():
            if k not in context:
                context[k] = v

        if failed_attempts:
            context["errors"] = failed_attempts

        return self.context_builder.build_prompt(current_program=current_program, context=context)

    # ------------------------------------------------------------------
    # Acceptance gating
    # ------------------------------------------------------------------

    def _acceptance_gate(self, result: SerializableResult, iteration: int) -> bool:
        """Apply GEPA acceptance gating.

        A child is accepted only if its fitness strictly exceeds the parent's.
        Rejected children are stored in the database's rejection history for
        use in reflective prompting.

        Returns:
            True if the child is accepted, False otherwise.
        """
        child_score = get_score(result.child_program_dict.get("metrics", {}))

        parent_score = 0.0
        if result.parent_id and result.parent_id in self.database.programs:
            parent = self.database.programs[result.parent_id]
            parent_score = get_score(parent.metrics)

        if child_score <= parent_score:
            child = Program.from_dict(result.child_program_dict)
            self.database.add_rejected(child)

            logger.info(
                f"Iteration {iteration}: REJECTED child "
                f"(child_score={child_score:.4f} <= parent_score={parent_score:.4f})"
            )
            self._iterations_without_improvement += 1
            return False

        return True

    # ------------------------------------------------------------------
    # LLM-mediated merge
    # ------------------------------------------------------------------

    def _should_merge(self) -> bool:
        """Check if a stagnation-triggered merge should be attempted."""
        return (
            self.use_merge
            and self._iterations_without_improvement >= self.merge_after_stagnation
            and self._merge_attempts_used < self.max_merge_attempts
        )

    async def _attempt_merge(self, iteration: int) -> None:
        """Attempt an LLM-mediated merge of two complementary programs.

        Guards against budget exhaustion, self-merges, and duplicate pairs.
        On success the merged program is added to the database and stagnation
        is reset.  On failure (LLM error, parse error, eval error, or
        rejected merge) the stagnation counter is left unchanged.
        """
        if self._merge_attempts_used >= self.max_merge_attempts:
            return

        if len(self.database.programs) < 2:
            logger.debug("Not enough programs for merge, skipping")
            return

        prog_a, prog_b = self.database.get_merge_candidates()

        # Skip self-merge (happens when pool has < 2 distinct programs)
        if prog_a.id == prog_b.id:
            logger.debug(f"Iteration {iteration}: Only one program available, skipping merge")
            return

        # Deduplication: skip if this pair was already tried
        pair_key = tuple(sorted([prog_a.id, prog_b.id]))
        if pair_key in self._merge_pairs_tried:
            logger.debug(f"Iteration {iteration}: Merge pair already tried, skipping")
            return
        self._merge_pairs_tried.add(pair_key)
        self._merge_attempts_used += 1

        score_a = get_score(prog_a.metrics)
        score_b = get_score(prog_b.metrics)

        logger.info(
            f"Iteration {iteration}: Attempting merge "
            f"(stagnation={self._iterations_without_improvement}, "
            f"attempt={self._merge_attempts_used}/{self.max_merge_attempts}, "
            f"scores: {score_a:.4f}, {score_b:.4f})"
        )

        merge_prompt = self._build_merge_prompt(prog_a, prog_b)

        try:
            llm_start = time.time()
            llm_result = await self.llms.generate(
                system_message=merge_prompt["system"],
                messages=[{"role": "user", "content": merge_prompt["user"]}],
            )
            llm_generation_time = time.time() - llm_start
        except Exception as e:
            logger.warning(f"Merge LLM call failed: {e}")
            return

        llm_response = llm_result.text if llm_result else ""
        if not llm_response:
            logger.warning("Merge LLM returned empty response")
            return

        # Always parse as full rewrite (merge prompt asks for complete program)
        child_solution = parse_full_rewrite(llm_response, self.config.language)
        if not child_solution:
            logger.warning("Merge parse failed: no valid solution in response")
            return

        # Evaluate the merged solution
        child_id = str(uuid.uuid4())
        try:
            eval_start = time.time()
            eval_result = await self.evaluator.evaluate_program(child_solution, child_id)
            eval_time = time.time() - eval_start
        except Exception as e:
            logger.warning(f"Merge evaluation failed: {e}")
            return

        merged_score = get_score(eval_result.metrics)
        logger.info(
            f"Iteration {iteration}: Merge completed"
            f" (llm: {llm_generation_time:.2f}s,"
            f" eval: {eval_time:.2f}s)"
        )

        # GEPA acceptance criterion for merges: must meet or exceed both parents
        if merged_score >= max(score_a, score_b):
            merged_program = Program(
                id=child_id,
                solution=child_solution,
                language=self.config.language,
                metrics=eval_result.metrics,
                iteration_found=iteration,
                parent_id=prog_a.id,
                other_context_ids=[prog_b.id],
                parent_info=("Merge Parent A", prog_a.id),
                context_info=[("Merge Parent B", prog_b.id)],
                metadata={
                    "changes": "LLM-mediated merge",
                    "merge_score_a": score_a,
                    "merge_score_b": score_b,
                    "parent_metrics": prog_a.metrics,
                },
                artifacts=eval_result.artifacts or {},
            )
            self.database.add(merged_program, iteration=iteration)

            self.database.log_prompt(
                template_key="merge",
                program_id=child_id,
                prompt=merge_prompt,
                responses=[llm_response],  # already str via .text extraction above
            )

            logger.info(
                f"Merge ACCEPTED: score={merged_score:.4f} "
                f"(>= max({score_a:.4f}, {score_b:.4f}))"
            )

            if merged_score > self._best_score_seen:
                self._best_score_seen = merged_score

            # Reset stagnation only on successful merge
            self._iterations_without_improvement = 0

            # Fire monitor callback
            if self.monitor_callback:
                try:
                    self.monitor_callback(merged_program, iteration)
                except Exception as e:
                    logger.warning(
                        f"Monitor callback failed: {e}"
                    )  # Never crash discovery due to monitor
        else:
            logger.info(
                f"Merge REJECTED: score={merged_score:.4f} " f"< max({score_a:.4f}, {score_b:.4f})"
            )
            # Stagnation counter NOT reset on rejected merge

    def _build_merge_prompt(self, prog_a: Program, prog_b: Program) -> Dict[str, str]:
        """Build a prompt asking the LLM to merge two programs.

        Includes both programs' code, per-metric strengths comparison,
        and evaluator diagnostics from each program's artifacts.
        """
        score_a = get_score(prog_a.metrics)
        score_b = get_score(prog_b.metrics)

        # Summarise per-metric strengths
        strengths_a: List[str] = []
        strengths_b: List[str] = []
        if prog_a.metrics and prog_b.metrics:
            for key in set(prog_a.metrics.keys()) | set(prog_b.metrics.keys()):
                va = prog_a.metrics.get(key)
                vb = prog_b.metrics.get(key)
                if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
                    continue
                if va > vb:
                    strengths_a.append(f"{key}: {va}")
                elif vb > va:
                    strengths_b.append(f"{key}: {vb}")

        strengths_section = ""
        if strengths_a or strengths_b:
            strengths_section = "\n## Per-Metric Strengths\n"
            if strengths_a:
                strengths_section += f"Program A leads on: {', '.join(strengths_a)}\n"
            if strengths_b:
                strengths_section += f"Program B leads on: {', '.join(strengths_b)}\n"

        # Include evaluator diagnostics from parent artifacts
        diagnostics_section = ""
        for label, prog in [("A", prog_a), ("B", prog_b)]:
            if not prog.artifacts:
                continue
            diag_parts = []
            for key, value in prog.artifacts.items():
                if not isinstance(value, str) or not value.strip():
                    continue
                display = value if len(value) <= 500 else value[:500] + "... (truncated)"
                diag_parts.append(f"- {key}: {display}")
            if diag_parts:
                diagnostics_section += (
                    f"\n## Program {label} Diagnostics\n" + "\n".join(diag_parts) + "\n"
                )

        system = (
            "You are an expert programmer. Your task is to merge two programs into "
            "a single improved program that combines the strengths of both. "
            "Output only the complete merged program inside a code block."
        )

        user = (
            f"## Program A (score: {score_a:.4f})\n"
            f"```\n{prog_a.solution}\n```\n\n"
            f"## Program B (score: {score_b:.4f})\n"
            f"```\n{prog_b.solution}\n```\n"
            f"{strengths_section}"
            f"{diagnostics_section}\n"
            "## Instructions\n"
            "Combine the best ideas from both programs into a single solution. "
            "Preserve any approach that contributes to a higher score. "
            "Resolve conflicts by choosing the strategy that is more likely to "
            "generalise across all test cases. Output the complete merged program."
        )

        return {"system": system, "user": user}
