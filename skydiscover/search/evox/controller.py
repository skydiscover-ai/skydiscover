"""
Co-evolution controller.

Runs evolution for the main *solution* database while also evolving a
separate *search* program/database in the same process.
"""

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.evox.utils.coevolve_logging import (
    handle_generation_failure,
    log_search_algorithm_generated,
    make_json_serializable,
    update_saved_search_algorithm_score,
)
from skydiscover.search.evox.utils.search_scorer import LogWindowScorer
from skydiscover.search.evox.utils.variation_operator_generator import generate_variation_operators
from skydiscover.search.registry import get_program, setup_search
from skydiscover.search.utils.discovery_utils import SerializableResult, load_database_from_file

logger = logging.getLogger(__name__)


class CoEvolutionController(DiscoveryController):
    """
    Co-evolves solution programs alongside search algorithms.

    The solution database uses an evolving search algorithm for sampling,
    while the search algorithm itself is scored based on solution improvements.
    """

    # Adaptive mode defaults
    DEFAULT_SWITCH_RATIO = 0.10  # Evolve search after 10% of total iterations stagnate
    DEFAULT_IMPROVEMENT_THRESHOLD = 0.01

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

        self._init_search_evolution_controller()
        self._init_output_dir(controller_input)

    def _init_search_evolution_controller(self) -> None:
        """Initialize search controller, scorer, and load initial algorithm."""
        db_cfg = self.config.search.database
        if not db_cfg.database_file_path:
            raise ValueError(
                "config.search.database.database_file_path is required for co-evolution"
            )

        controller_input, self._search_initial_code = setup_search(
            initial_program_path=db_cfg.database_file_path,
            evaluation_file=db_cfg.evaluation_file,
            config_path=db_cfg.config_path,
            output_dir=self.config.search.output_dir,
        )
        self.search_controller = DiscoveryController(controller_input)
        self.search_scorer = LogWindowScorer()
        self._active_search_algorithm_code = self._search_initial_code

        self._log_coevolution_setup(db_cfg)
        self._init_search_tracking()

    def _init_search_tracking(self) -> None:
        """Initialize search evolution tracking state."""
        self._pending_search_result: Optional[SerializableResult] = None
        self._best_search_score: Optional[float] = None
        self._num_search_evolutions = 0

        self._switch_interval = None
        self._stagnant_count = 0
        self._last_tracked_best_score: Optional[float] = None

        self._diverge_label = ""
        self._refine_label = ""

        self._fallback_database = None
        self._fallback_search_code = None

    def _init_output_dir(self, controller_input: DiscoveryControllerInput) -> None:
        base_dir = controller_input.output_dir or os.path.join(
            os.path.dirname(controller_input.evaluation_file),
            "outputs",
            self.config.search.type,
        )
        self.search_outputs_dir = os.path.join(base_dir, "search")
        os.makedirs(self.search_outputs_dir, exist_ok=True)

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
        post_process_result: Optional[bool] = True,
    ):
        """Run co-evolution of solution programs and search algorithms."""
        self.total_solution_iterations = start_iteration + max_iterations
        self._max_solution_iterations = max_iterations

        if self._switch_interval is None:
            self._switch_interval = max(1, int(max_iterations * self.DEFAULT_SWITCH_RATIO))
            logger.info(f"Switch if {self._switch_interval} iterations of stagnation detected")

        self.start_db_stats = self.database.get_statistics(
            improvement_threshold=self.DEFAULT_IMPROVEMENT_THRESHOLD
        )

        # Set up search window and labels
        self._reset_search_window()

        # Generate variation labels for the search algorithm
        await self._generate_variation_operators()

        # Run co-evolution
        iteration = start_iteration
        while iteration < self.total_solution_iterations:
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested")
                break

            try:
                # Run solution iteration
                result = await self._run_iteration(iteration, retry_times=3)
                attempts_used = getattr(result, "attempts_used", 1)

                if result.error:
                    logger.warning(
                        f"Iteration {iteration} failed (used {attempts_used} attempts): {result.error}"
                    )
                    # Database error after a switch: fall back and retry
                    if self._fallback_database is not None and result.prompt is None:
                        self._restore_fallback_database()
                        continue  # Retry same iteration with restored database
                else:
                    self._process_iteration_result(result, iteration, checkpoint_callback)

                for _ in range(attempts_used):
                    self._record_search_window_step()

                completed_solution_iter = iteration
                iteration += attempts_used

                # Co-evolve search strategy if needed (skip on final iteration)
                if iteration < self.total_solution_iterations and self._should_evolve_search():
                    logger.info(
                        f"Stagnation detected -> evolving search strategy (solution_iter={completed_solution_iter})"
                    )
                    await self._evolve_search(completed_solution_iter)

            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}", exc_info=True)
                # Exception from database.add() after a switch: fall back and retry
                if self._fallback_database is not None:
                    self._restore_fallback_database()
                    continue  # Retry same iteration
                iteration += 1  # Normal error — advance

        if self._pending_search_result:
            await self._finalize_pending_search()

        logger.info(f"[SOLUTION EVOLUTION] Evolution completed: {self.database.name}")
        return self.database.get_best_program()

    def _should_evolve_search(self) -> bool:
        """Check if it's time to evolve the search algorithm (stagnation-based)."""
        current = self._get_best_score()

        if self._last_tracked_best_score is None:
            self._stagnant_count = 0
        elif (current - self._last_tracked_best_score) > self.DEFAULT_IMPROVEMENT_THRESHOLD:
            self._stagnant_count = 0
        else:
            self._stagnant_count += 1

        self._last_tracked_best_score = current

        if self._stagnant_count >= self._switch_interval:
            self._stagnant_count = 0
            return True

        return False

    async def _evolve_search(self, solution_iter: int) -> None:
        """Handle search evolution: score previous algorithm, generate and switch to new one."""

        if not self.search_controller.database.programs:
            await self._initialize_first_search_program(solution_iter)
            return

        # If there is a pending search result, finalize it (as search window is reset)
        if self._pending_search_result:
            await self._finalize_pending_search()

        self._reset_search_window()
        await self._generate_and_validate_search_algorithm(solution_iter)

    async def _finalize_pending_search(self) -> None:
        """Score the pending search algorithm and add it to the search strategy database."""
        pending_iteration = self._num_search_evolutions
        is_new_best = self._assign_search_score()

        await update_saved_search_algorithm_score(
            self.search_outputs_dir,
            pending_iteration,
            self._pending_search_result,
            is_new_best=is_new_best,
            db_stats=self.database.get_statistics(),
        )
        await self.search_controller.postprocess_result(
            self._pending_search_result, self._num_search_evolutions, verbose=False
        )

        self._pending_search_result = None
        self._num_search_evolutions += 1

    async def _initialize_first_search_program(self, solution_iter: int) -> None:
        """Initialize and score the first (file-based) search program."""
        start_score = (
            self.search_scorer.get_start_score()
            or getattr(self.database, "initial_program_score", None)
            or 0.0
        )
        metrics = self._compute_search_metrics(
            start_score=start_score,
            best_scores=None,
            horizon=self._switch_interval,
            start_iteration=0,
        )
        search_score = float(metrics.get("combined_score", 0.0) or 0.0)

        initial_program = get_program(
            self.search_controller.config,
            self._search_initial_code,
            str(uuid.uuid4()),
            metrics,
            self._num_search_evolutions,
        )
        initial_program.metadata = initial_program.metadata or {}
        initial_program.metadata["start_db_stats"] = make_json_serializable(self.start_db_stats)
        initial_program.metadata["end_db_stats"] = make_json_serializable(
            self.database.get_statistics(improvement_threshold=self.DEFAULT_IMPROVEMENT_THRESHOLD)
        )

        initial_result = SerializableResult(
            child_program_dict=initial_program.to_dict(), iteration=self._num_search_evolutions
        )
        self._best_search_score = search_score
        await self.search_controller.postprocess_result(
            initial_result, self._num_search_evolutions, verbose=False
        )

        self.search_controller.database.initial_program_id = initial_program.id
        self.search_controller.database.initial_program_score = search_score
        self._num_search_evolutions += 1

        self._reset_search_window()
        await self._generate_and_validate_search_algorithm(solution_iter)

    async def _generate_variation_operators(self) -> None:
        """Generate diverge/refine labels once and assign to the current database."""
        if self._diverge_label and self._refine_label:
            self._assign_labels_to_db(self.database)
            return

        db_cfg = self.config.search.database
        if not getattr(db_cfg, "auto_generate_variation_operators", True):
            from skydiscover.search.evox.utils.template import (
                DEFAULT_DIVERGE_TEMPLATE,
                DEFAULT_REFINE_TEMPLATE,
            )

            self._diverge_label = DEFAULT_DIVERGE_TEMPLATE
            self._refine_label = DEFAULT_REFINE_TEMPLATE
            logger.info(
                "Using default variation operators (auto_generate_variation_operators=false)"
            )
            self._assign_labels_to_db(self.database)
            return

        system_message = self.config.context_builder.system_message or ""
        from skydiscover.search.utils.discovery_utils import load_evaluator_code
        evaluator_code = load_evaluator_code(self.evaluation_file)

        try:
            problem_dir = Path(self.evaluation_file).parent if self.evaluation_file else None
            label_llms = self.search_controller.guide_llms
            model_names = ", ".join(m.name for m in label_llms.models_cfg)
            logger.info(f"Label generation: using guide_model = [{model_names}]")
            self._diverge_label, self._refine_label = await generate_variation_operators(
                system_message,
                evaluator_code,
                problem_dir=problem_dir,
                llm_pool=label_llms,
            )
            logger.info(
                f"Generated variation operator labels ({len(self._diverge_label)}/{len(self._refine_label)} chars)"
            )
        except Exception as e:
            self._diverge_label = ""
            self._refine_label = ""
            logger.error(f"Label generation failed: {e}, setting labels to empty strings")

        self._assign_labels_to_db(self.database)

    def _assign_labels_to_db(self, db) -> None:
        """Assign the variation operators to a database instance."""
        db.DIVERGE_LABEL = self._diverge_label
        db.REFINE_LABEL = self._refine_label

    async def _generate_and_validate_search_algorithm(self, solution_iter: int) -> None:
        """Generate a new search algorithm and switch to it if valid."""
        iteration = self._num_search_evolutions
        search_stats = self._build_search_stats(solution_iter)

        self.search_controller._prompt_context = {
            "search_stats": search_stats["search_algorithm_stats"],
            "db_stats": search_stats["db_stats"],
        }
        result = await self.search_controller.run_discovery(
            start_iteration=iteration,
            max_iterations=1,
            post_process_result=False,
        )

        if not result or result.error:
            await handle_generation_failure(
                self.search_outputs_dir,
                self._active_search_algorithm_code,
                iteration,
                result,
                solution_iter,
            )
            self._num_search_evolutions += 1
            return

        result.child_program_dict.setdefault("metadata", {})["start_db_stats"] = (
            make_json_serializable(search_stats["db_stats"])
        )
        await log_search_algorithm_generated(
            self.search_outputs_dir,
            result,
            iteration,
            diverge_label=self._diverge_label,
            refine_label=self._refine_label,
        )

        if not self._switch_to_new_search_algorithm(result):
            await handle_generation_failure(
                self.search_outputs_dir,
                self._active_search_algorithm_code,
                iteration,
                result,
                solution_iter,
                "validation",
            )
            self._num_search_evolutions += 1
            return

        self._pending_search_result = result
        self._reset_search_window(start_iteration=solution_iter)

    def _build_search_stats(self, solution_iter: int) -> Dict[str, Any]:
        """Build statistics dict for search algorithm generation."""
        return {
            "search_algorithm_stats": {
                "window_start_iteration": solution_iter,
                "total_iterations": self._max_solution_iterations,
                "search_window_horizon": self._switch_interval,
                "problem_description": self.config.context_builder.system_message,
                "evaluator_context": self.evaluation_file,
                "improvement_threshold": self.DEFAULT_IMPROVEMENT_THRESHOLD,
            },
            "db_stats": self.database.get_statistics(
                improvement_threshold=self.DEFAULT_IMPROVEMENT_THRESHOLD
            ),
        }

    def _switch_to_new_search_algorithm(self, result: SerializableResult) -> bool:
        """Switch solution database to use the new search algorithm."""
        child_dict = result.child_program_dict or {}
        search_code = child_dict.get("solution")
        if not search_code:
            logger.warning("No solution in search result; skipping transition")
            return False

        search_program_id = child_dict.get("id", "unknown")
        fd, file_path = tempfile.mkstemp(suffix=".py", prefix="evox_search_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(search_code)

            # Load the new search algorithm database
            new_db_class, prog_class = load_database_from_file(file_path)
            # Ensure labels exist for databases that use them in __init__ (before _assign_labels_to_db)
            if not hasattr(new_db_class, "DIVERGE_LABEL"):
                new_db_class.DIVERGE_LABEL = ""
            if not hasattr(new_db_class, "REFINE_LABEL"):
                new_db_class.REFINE_LABEL = ""
            new_db = new_db_class(self.config.search.type, self.config.search.database)
            new_db._program_class = prog_class

            # Assign labels to the new search algorithm database
            self._assign_labels_to_db(new_db)

            # Migrate programs and prompts from the current database to the new database
            migrated_count = self._migrate_to_db(new_db)

            self._wrap_add_method(new_db)
            new_db.get_best_program()  # Sets best_program_id (None -> actual best)

            self._fallback_database = self.database
            self._fallback_search_code = self._active_search_algorithm_code

            self.database = new_db
            if self.evaluator.llm_judge:
                self.evaluator.llm_judge.database = new_db
            logger.info(
                f"Switched to search algorithm {search_program_id} ({migrated_count} programs migrated)"
            )

            self._active_search_algorithm_code = search_code
            return True

        except Exception as e:
            logger.error(f"Failed to load search algorithm {search_program_id}: {e}")
            return False
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def _restore_fallback_database(self) -> None:
        """Restore the previous search strategy after a failed switch."""
        broken_db = self.database
        old_db = self._fallback_database

        # Migrate new programs found during the broken strategy's successful runs
        old_ids = set(old_db.programs)
        migrated = 0
        for pid, program in broken_db.programs.items():
            if pid not in old_ids:
                try:
                    old_db.add(program, iteration=program.iteration_found)
                    migrated += 1
                except Exception:
                    logger.debug("Migration failed for program %s", program.id, exc_info=True)

        logger.warning(
            "New search strategy caused database error — "
            f"restoring previous search strategy ({migrated} new programs preserved)"
        )
        self.database = old_db
        if self.evaluator.llm_judge:
            self.evaluator.llm_judge.database = old_db
        self._active_search_algorithm_code = self._fallback_search_code
        self._pending_search_result = None
        self._num_search_evolutions += 1  # Count the failed attempt
        self._fallback_database = None
        self._fallback_search_code = None

    def _migrate_to_db(self, new_db) -> int:
        """Migrate all programs and prompts from current database to new database."""
        prog_class = getattr(new_db, "_program_class", None)
        for program in sorted(self.database.programs.values(), key=lambda x: x.iteration_found):
            converted = prog_class(**program.to_dict()) if prog_class else program
            new_db.add(converted, iteration=program.iteration_found)
        migrated = len(self.database.programs)

        # Migrate prompts
        if self.database.config.log_prompts:
            if new_db.prompts_by_program is None:
                new_db.prompts_by_program = {}

            old_prompts = self.database.prompts_by_program or {}
            new_db.prompts_by_program.update(
                {k: v for k, v in old_prompts.items() if k not in new_db.prompts_by_program}
            )

            for p in new_db.programs.values():
                if p.prompts and p.id not in new_db.prompts_by_program:
                    new_db.prompts_by_program[p.id] = p.prompts

        return migrated

    def _get_best_score(self) -> float:
        """Get the current best solution score (combined_score metric)."""

        best = self.database.get_best_program()

        if best and best.metrics:
            score = best.metrics.get("combined_score")
            return float(score) if isinstance(score, (int, float)) else 0.0
        return getattr(self.database, "initial_program_score", None) or 0.0

    def _reset_search_window(self, start_iteration: Optional[int] = None) -> None:
        """Start a fresh scoring window for the active search algorithm."""
        self.search_scorer.reset_window(self._get_best_score(), start_iteration=start_iteration)

    def _record_search_window_step(self) -> None:
        """Record current best score for search algorithm scoring."""

        if self.search_scorer.get_start_score() is None:
            self._reset_search_window()

        self.search_scorer.record_step(self._get_best_score())

    def _compute_search_metrics(
        self,
        start_score: Optional[float] = None,
        best_scores: Optional[List[float]] = None,
        horizon: Optional[int] = None,
        start_iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute scoring metrics using the configured scorer."""

        return self.search_scorer.compute_metrics(
            start_score=start_score,
            best_scores=best_scores,
            horizon=horizon,
            total_iterations=self._max_solution_iterations,
            start_iteration=start_iteration,
        )

    def _wrap_add_method(self, db) -> None:
        """Wrap database.add() to ensure _update_best_program is always called."""
        original_add = db.add

        def wrapped_add(program, iteration=None, **kwargs):
            result = original_add(program, iteration=iteration, **kwargs)
            db._update_best_program(program)  # Idempotent safety for LLM-generated databases
            return result

        db.add = wrapped_add

    def _assign_search_score(self) -> bool:
        """Assign score to pending search algorithm. Returns True if new best."""
        if not self._pending_search_result:
            return False

        if self.search_scorer.get_window_size() > 0:
            metrics = self._compute_search_metrics(horizon=self._switch_interval)
        else:
            start = self.search_scorer.get_start_score() or 0.0
            metrics = self._compute_search_metrics(
                start_score=start,
                best_scores=[self._get_best_score()],
                horizon=self._switch_interval,
            )

        score = float(metrics.get("combined_score", 0.0) or 0.0)

        child_dict = self._pending_search_result.child_program_dict or {}
        child_dict.setdefault("metrics", {}).update(metrics)
        child_dict.setdefault("metadata", {})["end_db_stats"] = make_json_serializable(
            self.database.get_statistics(improvement_threshold=self.DEFAULT_IMPROVEMENT_THRESHOLD)
        )
        self._pending_search_result.child_program_dict = child_dict

        is_new_best = self._best_search_score is not None and score > self._best_search_score
        if is_new_best:
            logger.info(
                f"New best search score: {score:.6f} (+{score - self._best_search_score:.6f})"
            )
        if is_new_best or self._best_search_score is None:
            self._best_search_score = score
        return is_new_best

    def _log_coevolution_setup(self, db_cfg) -> None:
        logger.info("=" * 70)
        logger.info("[EVOX CO-EVOLUTION SETUP]")
        logger.info("-" * 70)
        logger.info(f"  [SOLUTION EVOLUTION]")
        logger.info(f"    Initial search strategy file : {db_cfg.database_file_path}")
        logger.info(f"    Solution database class      : {self.database.__class__.__name__}")
        logger.info(f"  [META EVOLUTION OF SEARCH STRATEGY]")
        logger.info(
            f"    Search strategy database class: {self.search_controller.database.__class__.__name__}"
        )
        logger.info(f"    Search strategy evaluator     : {db_cfg.evaluation_file}")
        logger.info(f"    Search strategy config        : {db_cfg.config_path}")
        logger.info("=" * 70)
