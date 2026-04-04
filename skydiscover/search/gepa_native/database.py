"""
GEPA Native Database - Elite pool with epsilon-greedy selection.

Implements the population management layer for GEPA's guided evolution:
1. Fixed-size elite pool sorted by fitness score (descending)
2. Epsilon-greedy parent selection for exploration/exploitation balance
3. Per-metric best tracking for merge candidate selection
4. Rejection history deque for reflective prompting context

Configuration options (via GEPANativeDatabaseConfig):
    population_size: Maximum elite pool size (default: 40)
    candidate_selection_strategy: "epsilon_greedy" or "best" (default: "epsilon_greedy")
    epsilon: Exploration probability for epsilon-greedy (default: 0.1)
    max_rejection_history: Bounded deque size for rejected programs (default: 20)
    random_seed: RNG seed for reproducible selection (default: 42)
"""

import collections
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.search.utils.checkpoint_manager import SafeJSONEncoder
from skydiscover.utils.metrics import get_score

from .pareto_utils import select_program_candidate_from_pareto_front

logger = logging.getLogger(__name__)


class GEPANativeDatabase(ProgramDatabase):
    """
    Program database for GEPA Native search.

    Maintains a fixed-size elite pool sorted by combined_score.
    Supports epsilon-greedy parent selection, per-metric best tracking,
    and a rejection history deque for reflective prompting.

    Configuration options (via GEPANativeDatabaseConfig):
        population_size: Maximum elite pool size (default: 40)
        candidate_selection_strategy: Parent selection strategy (default: "epsilon_greedy")
            - "epsilon_greedy": Pick best with probability (1-epsilon), random otherwise
            - "best": Always pick the highest-scoring program
            - "pareto": Frequency-weighted sampling from the Pareto front across metrics
        epsilon: Exploration probability for epsilon-greedy (default: 0.1)
        max_rejection_history: Max rejected programs to keep (default: 20)
    """

    def __init__(self, name: str, config: DatabaseConfig, **kwargs: Any):
        # Read GEPA-specific config before super().__init__ (which may call load)
        self.population_size: int = getattr(config, "population_size", 40)
        self.candidate_selection_strategy: str = getattr(
            config, "candidate_selection_strategy", "epsilon_greedy"
        )
        self.epsilon: float = getattr(config, "epsilon", 0.1)
        max_rejection_history: int = getattr(config, "max_rejection_history", 20)
        seed: int = getattr(config, "random_seed", 42) or 42

        self.elite_pool: List[str] = []  # program IDs sorted by score desc
        self.rejection_history: collections.deque = collections.deque(maxlen=max_rejection_history)
        self.metric_best: Dict[str, Tuple[str, float]] = {}  # metric -> (prog_id, value)
        self.program_at_metric_front: Dict[str, Set[str]] = {}  # metric -> set of prog_ids at best
        self.rng = random.Random(seed)

        super().__init__(name, config, **kwargs)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def add(self, program: Program, iteration: Optional[int] = None, **kwargs: Any) -> str:
        """Add a program to the database and elite pool.

        Inserts into the elite pool (sorted descending by fitness), evicts
        the weakest members if the pool exceeds ``population_size``, and
        updates per-metric best tracking.

        Args:
            program: Program to add.
            iteration: Current iteration number for tracking.

        Returns:
            The program's ID.
        """
        if not self.programs:
            self.initial_program_id = program.id

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        # Insert into elite pool, keep sorted by score descending
        if program.id not in self.elite_pool:
            self.elite_pool.append(program.id)
        self.elite_pool.sort(
            key=lambda pid: get_score(self.programs[pid].metrics if pid in self.programs else {}),
            reverse=True,
        )

        # Cap at population_size, but pin best and initial programs.
        # Only remove from elite_pool (sampling); keep self.programs as
        # a full archive so parent lookups for reflective prompting work.
        if len(self.elite_pool) > self.population_size:
            pinned = {self.best_program_id, self.initial_program_id, program.id} - {None}
            keep = []
            for pid in self.elite_pool:
                if pid in pinned or len(keep) < self.population_size:
                    keep.append(pid)
            self.elite_pool = keep

        # Update per-metric best tracking
        if program.metrics:
            for metric_name, value in program.metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                current = self.metric_best.get(metric_name)
                if current is None or value > current[1]:
                    self.metric_best[metric_name] = (program.id, value)
                    self.program_at_metric_front[metric_name] = {program.id}
                elif value == current[1]:
                    self.program_at_metric_front[metric_name].add(program.id)

        # Update global best
        self._update_best_program(program)

        # Persist to disk
        if self.config.db_path:
            self._save_program(program)

        logger.debug(
            f"Added program {program.id} to GEPA elite pool " f"(pool size: {len(self.elite_pool)})"
        )
        return program.id

    def sample(
        self,
        num_context_programs: Optional[int] = 4,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Program], Dict[str, List[Program]]]:
        """Sample a parent and context programs.

        Uses epsilon-greedy selection for the parent and top-of-pool +
        metric leaders for other context programs.

        Returns:
            Tuple of ({"": parent}, {"": [other context, ...]}).
        """
        if not self.programs:
            raise ValueError("Cannot sample: no programs in database")

        parent = self._select_parent()
        other_context_programs = self._select_other_context_programs(
            parent.id, num_context_programs or 4
        )

        return {"": parent}, {"": other_context_programs}

    # ------------------------------------------------------------------
    # GEPA-specific methods
    # ------------------------------------------------------------------

    def add_rejected(self, program: Program) -> None:
        """Store a rejected program for reflective prompting.

        The program is NOT added to the elite pool or ``self.programs``.
        """
        self.rejection_history.append(program)

    def get_rejection_history(self, limit: Optional[int] = None) -> List[Program]:
        """Return recent rejected programs, most-recent last.

        Args:
            limit: If given, return only the *limit* most recent entries.
        """
        items = list(self.rejection_history)
        if limit is not None:
            items = items[-limit:]
        return items

    def get_merge_candidates(self) -> Tuple[Program, Program]:
        """Select two complementary programs for LLM-mediated merge.

        Selection strategy:
        1. Prefer two programs that each lead on a different metric.
        2. Fallback: best program + random from top 5.
        3. Last resort: best program returned twice (caller should guard).
        """
        if len(self.elite_pool) < 2:
            best = self.get_best_program()
            return best, best

        # Try to find two programs that each lead on a different metric
        leaders: Dict[str, str] = {}
        for metric_name, (pid, _score) in self.metric_best.items():
            if pid in self.programs and pid in self.elite_pool:
                leaders[metric_name] = pid

        unique_leaders = sorted(set(leaders.values()))
        if len(unique_leaders) >= 2:
            pids = self.rng.sample(unique_leaders, 2)
            return self.programs[pids[0]], self.programs[pids[1]]

        # Fallback: best + random from top 5
        best = self.get_best_program()
        top5_ids = [pid for pid in self.elite_pool[:5] if pid != best.id]
        if top5_ids:
            other_id = self.rng.choice(top5_ids)
            return best, self.programs[other_id]

        return best, best

    # ------------------------------------------------------------------
    # Save / Load — persist elite pool + rejection history
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:
        """Save base state plus GEPA-specific metadata."""
        super().save(path=path, iteration=iteration)

        save_path = path or self.config.db_path
        if not save_path:
            return

        metadata = {
            "elite_pool": self.elite_pool,
            "initial_program_id": self.initial_program_id,
            "metric_best": {k: list(v) for k, v in self.metric_best.items()},
            "program_at_metric_front": {
                k: list(v) for k, v in self.program_at_metric_front.items()
            },
            "rejection_history": [prog.to_dict() for prog in self.rejection_history],
        }
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "gepa_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, cls=SafeJSONEncoder)

    def load(self, path: str) -> None:
        """Load base state plus GEPA-specific metadata."""
        super().load(path)

        metadata_path = os.path.join(path, "gepa_metadata.json")
        if not os.path.exists(metadata_path):
            # Legacy checkpoint — rebuild from programs
            self._rebuild_elite_pool()
            return

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Restore elite pool (filter out IDs no longer in programs)
        self.elite_pool = [pid for pid in metadata.get("elite_pool", []) if pid in self.programs]

        # Restore initial program
        self.initial_program_id = metadata.get("initial_program_id")

        # Restore per-metric best
        for k, v in metadata.get("metric_best", {}).items():
            pid, score = v[0], v[1]
            if pid in self.programs:
                self.metric_best[k] = (pid, score)

        # Restore metric Pareto front
        self.program_at_metric_front = {
            k: {pid for pid in v if pid in self.programs}
            for k, v in metadata.get("program_at_metric_front", {}).items()
        }

        # Restore rejection history
        self.rejection_history.clear()
        for prog_dict in metadata.get("rejection_history", []):
            try:
                self.rejection_history.append(Program.from_dict(prog_dict))
            except Exception as e:
                logger.warning(f"Failed to load rejected program from history: {e}")

    def _rebuild_elite_pool(self) -> None:
        """Rebuild elite pool and metric_best from loaded programs."""
        self.elite_pool = sorted(
            self.programs.keys(),
            key=lambda pid: get_score(self.programs[pid].metrics or {}),
            reverse=True,
        )[: self.population_size]

        # Infer initial program as the earliest-seen one
        if self.programs:
            self.initial_program_id = min(
                self.programs,
                key=lambda pid: self.programs[pid].iteration_found,
            )

        for pid, prog in self.programs.items():
            if not prog.metrics:
                continue
            for metric_name, value in prog.metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                current = self.metric_best.get(metric_name)
                if current is None or value > current[1]:
                    self.metric_best[metric_name] = (pid, value)
                    self.program_at_metric_front[metric_name] = {pid}
                elif value == current[1]:
                    self.program_at_metric_front[metric_name].add(pid)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_parent(self) -> Program:
        """Epsilon-greedy or Pareto-based parent selection."""
        if self.candidate_selection_strategy == "best" or not self.elite_pool:
            return self.get_best_program()

        if self.candidate_selection_strategy == "pareto":
            return self._select_parent_pareto()

        if self.rng.random() < self.epsilon and len(self.elite_pool) > 1:
            pid = self.rng.choice(self.elite_pool)
            return self.programs[pid]
        return self.get_best_program()

    def _select_parent_pareto(self) -> Program:
        """Frequency-weighted selection from the Pareto front across metrics."""
        if not self.program_at_metric_front or len(self.programs) < 2:
            return self.get_best_program()
        scores = {pid: get_score(prog.metrics) for pid, prog in self.programs.items()}
        try:
            pid = select_program_candidate_from_pareto_front(
                self.program_at_metric_front, scores, self.rng
            )
        except AssertionError:
            return self.get_best_program()
        return self.programs[pid]

    def _select_other_context_programs(
        self, parent_id: str, num_context_programs: int
    ) -> List[Program]:
        """Select context programs from elite pool + metric leaders.

        Picks top programs from the elite pool (excluding the parent),
        then appends any metric-best programs not already included.
        """
        seen = {parent_id}
        other_context_programs: List[Program] = []

        # Top programs from elite pool (excluding parent)
        for pid in self.elite_pool:
            if pid in seen or pid not in self.programs:
                continue
            other_context_programs.append(self.programs[pid])
            seen.add(pid)
            if len(other_context_programs) >= num_context_programs:
                break

        # Add metric-best programs not already included
        for _metric, (pid, _score) in self.metric_best.items():
            if pid in seen or pid not in self.programs:
                continue
            other_context_programs.append(self.programs[pid])
            seen.add(pid)

        return other_context_programs[:num_context_programs]
