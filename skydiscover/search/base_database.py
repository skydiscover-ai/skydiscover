"""
Clean interfaces for ProgramDatabase classes.

This module defines abstract base classes that capture the essential operations
needed for program database operations.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union

from skydiscover.config import DatabaseConfig
from skydiscover.utils.metrics import get_score

logger = logging.getLogger(__name__)


@dataclass
class Program:
    """Represents a program in the database"""

    # Program identification
    id: str
    solution: str
    language: str = "python"

    # Performance
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Tracking information
    iteration_found: int = 0
    parent_id: Optional[str] = None  # Parent program ID it mutates from
    other_context_ids: Optional[List[str]] = (
        None  # other program IDs to provide as context to generate this program
    )
    parent_info: Optional[Tuple[str, str]] = None  # information about the parent program
    context_info: Optional[List[Tuple[str, str]]] = None  # information about the context programs

    timestamp: float = field(default_factory=time.time)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # Prompts
    prompts: Optional[Dict[str, Any]] = None
    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Program:
        """Create from dictionary representation"""
        # Get the valid field names for the Program dataclass
        valid_fields = {f.name for f in fields(cls)}

        # Filter the data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Log if we're filtering out any fields
        if len(filtered_data) != len(data):
            filtered_out = set(data.keys()) - set(filtered_data.keys())
            logger.debug(f"Filtered out unsupported fields when loading Program: {filtered_out}")

        return cls(**filtered_data)


class ProgramDatabase(ABC):
    """
    Abstract base class for program storage and sampling.

    This interface captures the essential operations needed for any discovery process:
    - Add a program to the database
    - Sample a program and context programs to learn from past experiences for the next discovery step
    """

    def __init__(self, name: str, config: DatabaseConfig, **kwargs: Any):
        self.name = name
        self.config = config

        # In-memory program storage
        self.programs: Dict[str, Program] = {}

        # Track the last iteration number (for resuming)
        self.last_iteration: int = 0

        # Optionally track initial program info (set by controller on first add)
        self.initial_program_id: Optional[str] = None
        self.initial_program_score: Optional[float] = None

        # Best program tracking
        self.best_program_id: Optional[str] = None

        # Prompt log
        self.prompts_by_program: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None

        # Initialize checkpoint manager (imported here to avoid circular imports)
        from skydiscover.search.utils.checkpoint_manager import CheckpointManager

        self.checkpoint_manager = CheckpointManager(self.config)

        # Load database from disk if path is provided
        if config.db_path and os.path.exists(config.db_path):
            self.load(config.db_path)

    # ------------------------------------------------------------------
    # Abstract methods — implement these in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def add(self, program: Program, iteration: Optional[int] = None, **kwargs: Any) -> str:
        """Add a program to the database.

        Args:
            program: Program to add.
            iteration: Current iteration (for tracking).

        Returns:
            Program ID.
        """
        ...

    @abstractmethod
    def sample(
        self,
        num_context_programs: Optional[int] = 4,
        **kwargs: Any,
    ) -> Tuple[
        Union[Program, Dict[str, Program]],
        Union[List[Program], Dict[str, List[Program]]],
    ]:
        """Sample a parent program and context programs for discovery.

        Args:
            num_context_programs: Number of context programs to sample.
            **kwargs: Search-specific parameters.

        Returns:
            (parent, context_programs) — each can be plain or dict-wrapped.
            Plain: (Program, [Program, ...])
            Dict-wrapped: ({info: Program}, {info: [Program, ...]})
                where the key is additional information about the program.
        """
        ...

    # ------------------------------------------------------------------
    # Save and load
    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:
        """
        Save the database to disk

        Args:
            path: Path to save to (uses config.db_path if None)
            iteration: Current iteration number
        """
        self.checkpoint_manager.save(
            programs=self.programs,
            prompts_by_program=self.prompts_by_program,
            best_program_id=self.best_program_id,
            last_iteration=iteration if iteration is not None else self.last_iteration,
            path=path,
        )

    def load(self, path: str) -> None:
        """
        Load the database from disk

        Args:
            path: Path to load from
        """
        programs, best_id, last_iter = self.checkpoint_manager.load(path)
        self.programs = programs
        self.best_program_id = best_id
        self.last_iteration = last_iter

        self.log_status()

    def _save_program(
        self,
        program: Program,
        base_path: Optional[str] = None,
        prompts: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        """
        Save a single program to disk.

        This is a convenience method that delegates to CheckpointManager.
        Subclasses should use this method when they need to save individual programs
        (e.g., during add() operations).

        Args:
            program: Program to save
            base_path: Base path to save to (uses config.db_path if None)
            prompts: Optional prompts to save with the program
        """
        self.checkpoint_manager._save_program(program, base_path, prompts)

    # ------------------------------------------------------------------
    # Best program tracking
    # ------------------------------------------------------------------

    def _is_better(self, program1: Program, program2: Program) -> bool:
        """Determine if program1 has better fitness than program2."""
        if not program1.metrics and not program2.metrics:
            # No evidence either way — keep the current best.
            return False
        if program1.metrics and not program2.metrics:
            return True
        if not program1.metrics and program2.metrics:
            return False
        return get_score(program1.metrics) > get_score(program2.metrics)

    def _update_best_program(self, program: Program) -> None:
        """Update the best program tracking after a new program is added."""
        if self.best_program_id is None:
            self.best_program_id = program.id
            logger.debug(f"Set initial best program to {program.id}")
            return
        # If the best program is not in the database, set it to the new program
        if self.best_program_id not in self.programs:
            self.best_program_id = program.id
            return

        current_best = self.programs[self.best_program_id]

        # If the new program is better than the current best, set it to the new program
        if self._is_better(program, current_best):
            self.best_program_id = program.id

    def get_best_program(self, metric: Optional[str] = None) -> Optional[Program]:
        """Get the best program, optionally by a specific metric."""
        if not self.programs:
            return None

        if metric is None and self.best_program_id:
            if self.best_program_id in self.programs:
                return self.programs[self.best_program_id]
            else:
                logger.warning(
                    f"Tracked best program {self.best_program_id} no longer exists, will recalculate"
                )
                self.best_program_id = None

        if metric:
            sorted_programs = sorted(
                [p for p in self.programs.values() if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=True,
            )
        else:
            sorted_programs = sorted(
                self.programs.values(),
                key=lambda p: get_score(p.metrics),
                reverse=True,
            )

        if sorted_programs and (
            self.best_program_id is None or sorted_programs[0].id != self.best_program_id
        ):
            self.best_program_id = sorted_programs[0].id

        return sorted_programs[0] if sorted_programs else None

    def get_top_programs(self, n: int = 10, metric: Optional[str] = None) -> List[Program]:
        """Get the top N programs, optionally by a specific metric."""
        if not self.programs:
            return []

        if metric:
            sorted_programs = sorted(
                [p for p in self.programs.values() if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=True,
            )
        else:
            sorted_programs = sorted(
                self.programs.values(),
                key=lambda p: get_score(p.metrics),
                reverse=True,
            )

        return sorted_programs[:n]

    def get(self, program_id: str) -> Optional[Program]:
        """Get a program by ID"""
        return self.programs.get(program_id)

    # ------------------------------------------------------------------
    # Prompt logging
    # ------------------------------------------------------------------
    def log_prompt(
        self,
        program_id: str,
        template_key: str,
        prompt: Dict[str, str],
        responses: Optional[List[str]] = None,
    ) -> None:
        """
        Log a prompt for a program.
        Only logs if self.config.log_prompts is True.

        Args:
        program_id: ID of the program to log the prompt for
        template_key: Key for the prompt template
        prompt: Prompts in the format {template_key: { 'system': str, 'user': str }}.
        responses: Optional list of responses to the prompt, if available.
        """

        if not self.config.log_prompts:
            return

        if responses is None:
            responses = []
        prompt["responses"] = responses

        if self.prompts_by_program is None:
            self.prompts_by_program = {}

        if program_id not in self.prompts_by_program:
            self.prompts_by_program[program_id] = {}
        self.prompts_by_program[program_id][template_key] = prompt

    def log_status(self) -> None:
        """Log the status of the database"""
        best_program = self.get_best_program()
        best_score = None
        if best_program and best_program.metrics:
            best_score = best_program.metrics.get("combined_score")

        score_str = f"{best_score:.6f}" if best_score is not None else "N/A"
        logger.info(
            f"Database has {len(self.programs)} programs, best program score is {score_str}"
        )

    def get_statistics(
        self, num_recent_iterations: int = 100, k: int = 20, improvement_threshold: float = 0.10
    ) -> Dict[str, Any]:
        """
        Get statistics about the database population.

        Args:
            num_recent_iterations: Number of recent iterations to include in trajectory stats
            k: Number of top scores to return
            improvement_threshold: Minimum score improvement to count as meaningful improvement.
        """
        import statistics

        population_size = len(self.programs)

        if self.programs:
            actual_last_iter = max(
                (
                    p.iteration_found
                    for p in self.programs.values()
                    if isinstance(p.iteration_found, int)
                ),
                default=0,
            )
            self.last_iteration = max(self.last_iteration, actual_last_iter)
        else:
            actual_last_iter = 0

        scores = [
            p.metrics.get("combined_score")
            for p in self.programs.values()
            if p.metrics.get("combined_score") is not None
        ]

        if scores:
            scores_sorted = sorted(scores, reverse=True)
            n = len(scores_sorted)
            if n >= 4:
                quartiles = statistics.quantiles(scores, n=4)
                q25, q50, q75 = quartiles[0], quartiles[1], quartiles[2]
            else:
                q25 = q50 = q75 = statistics.median(scores)
            score_stats = {
                "best": scores_sorted[0],
                "q75": q75,
                "q50": q50,
                "q25": q25,
                "worst": scores_sorted[-1],
            }

            top_scores = scores_sorted[:k]
            n = len(scores)

            pct_top = sum(1 for s in scores if s >= q75) / n * 100
            pct_upper_mid = sum(1 for s in scores if q50 <= s < q75) / n * 100
            pct_lower_mid = sum(1 for s in scores if q25 <= s < q50) / n * 100
            pct_bottom = sum(1 for s in scores if s < q25) / n * 100

            unique_scores = len(set(round(s, 4) for s in scores))

            score_stats["score_tiers"] = {
                "top": {"threshold": f"score >= {q75:.4f}", "pct_programs": pct_top},
                "upper_mid": {
                    "threshold": f"{q50:.4f} <= score < {q75:.4f}",
                    "pct_programs": pct_upper_mid,
                },
                "lower_mid": {
                    "threshold": f"{q25:.4f} <= score < {q50:.4f}",
                    "pct_programs": pct_lower_mid,
                },
                "bottom": {"threshold": f"score < {q25:.4f}", "pct_programs": pct_bottom},
            }
            score_stats["unique_scores"] = unique_scores
        else:
            score_stats = {"best": None, "q75": None, "q50": None, "q25": None, "worst": None}
            top_scores = []

        programs_with_parents = [p for p in self.programs.values() if p.parent_id is not None]
        unique_parents = len({p.parent_id for p in programs_with_parents})
        avg_solutions_per_parent = (
            len(programs_with_parents) / unique_parents if unique_parents > 0 else 0.0
        )

        iterations_without_improvement = 0
        if scores:
            best_score = max(scores)
            near_best_programs = [
                p
                for p in self.programs.values()
                if p.metrics.get("combined_score") is not None
                and p.metrics.get("combined_score") >= best_score - improvement_threshold
            ]
            if near_best_programs:
                iteration_near_best_achieved = min(
                    p.iteration_found
                    for p in near_best_programs
                    if isinstance(p.iteration_found, int)
                )
                iterations_without_improvement = actual_last_iter - iteration_near_best_achieved

        if not self.programs:
            recent_stats = {}
            recent_programs = []
        else:
            recent_programs = [
                p
                for p in self.programs.values()
                if p.metrics.get("combined_score") is not None
                and isinstance(p.iteration_found, int)
                and p.iteration_found > actual_last_iter - num_recent_iterations
            ]
            recent_programs.sort(key=lambda p: p.iteration_found)

            execution_trace = []
            recent_scores = []
            parent_scores = []

            for p in recent_programs:
                prog_id = p.id
                prog_score = p.metrics.get("combined_score")
                recent_scores.append(prog_score)

                if p.parent_id is not None:
                    parent_id = p.parent_id
                    parent_label = None
                    if p.parent_info is not None:
                        parent_label = p.parent_info[0]
                    if parent_id in self.programs:
                        parent_program = self.programs[parent_id]
                        parent_score = parent_program.metrics.get("combined_score")
                        parent_scores.append(parent_score)
                    else:
                        parent_score = None
                        parent_scores.append(None)
                    parent_tuple = (parent_label, parent_id, parent_score)
                else:
                    parent_tuple = None
                    parent_scores.append(None)

                context_tuples = []
                if p.other_context_ids:
                    context_label_map = {}
                    if p.context_info is not None:
                        for label, ctx_id in p.context_info:
                            context_label_map[ctx_id] = label

                    for other_context_id in p.other_context_ids:
                        ctx_label = context_label_map.get(other_context_id)
                        if other_context_id in self.programs:
                            ctx_score = self.programs[other_context_id].metrics.get(
                                "combined_score"
                            )
                            context_tuples.append((ctx_label, other_context_id, ctx_score))
                        else:
                            context_tuples.append((ctx_label, other_context_id, None))

                trace_entry = {
                    "iteration": p.iteration_found,
                    "program": (prog_id, prog_score),
                    "parent": parent_tuple,
                    "context": context_tuples if context_tuples else None,
                }
                execution_trace.append(trace_entry)

            from collections import Counter

            recent_programs_with_parents = [p for p in recent_programs if p.parent_id is not None]

            most_reused_parent_ratio = 0.0
            most_reused_parent_score = None
            most_reused_context_ratio = 0.0
            most_reused_context_score = None

            if recent_programs_with_parents:
                parent_counts = Counter(
                    p.parent_id for p in recent_programs_with_parents if p.parent_id
                )
                if parent_counts:
                    top_parent_id, parent_count = parent_counts.most_common(1)[0]
                    most_reused_parent_ratio = parent_count / len(recent_programs_with_parents)
                    if top_parent_id in self.programs:
                        most_reused_parent_score = self.programs[top_parent_id].metrics.get(
                            "combined_score"
                        )

            if recent_programs:
                context_counts = Counter()
                programs_with_context = 0
                for p in recent_programs:
                    if p.other_context_ids:
                        programs_with_context += 1
                        for ctx_id in p.other_context_ids:
                            context_counts[ctx_id] += 1
                if context_counts and programs_with_context > 0:
                    top_ctx_id, context_count = context_counts.most_common(1)[0]
                    most_reused_context_ratio = context_count / programs_with_context
                    if top_ctx_id in self.programs:
                        most_reused_context_score = self.programs[top_ctx_id].metrics.get(
                            "combined_score"
                        )

            recent_stats = {
                "num_recent_iterations": min(num_recent_iterations, len(recent_scores)),
                "execution_trace": execution_trace,
                "score_trajectory": recent_scores,
                "parent_scores": parent_scores,
                "iterations_without_improvement": iterations_without_improvement,
                "improvement_threshold": improvement_threshold,
                "most_reused_parent_ratio": most_reused_parent_ratio,
                "most_reused_parent_score": most_reused_parent_score,
                "most_reused_context_ratio": most_reused_context_ratio,
                "most_reused_context_score": most_reused_context_score,
            }

        return {
            "previous_programs": recent_programs,
            "population_size": population_size,
            "solution_score_summary": score_stats,
            "avg_solutions_per_parent": avg_solutions_per_parent,
            "top_solution_scores": top_scores,
            "recent_solution_stats": recent_stats,
        }
