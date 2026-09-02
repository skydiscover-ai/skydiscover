"""
Beam Search Database for selecting programs from the solution database.

Beam search maintains a fixed-width "beam" of the most promising candidates,
exploring multiple paths in parallel while pruning less promising directions.

Key features:
- Maintains top-K candidates (beam) at each depth level
- Supports multiple parent selection strategies
- Tracks search tree depth for analysis
- Optional diversity bonus to prevent beam collapse
"""

import json
import logging
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


class BeamSearchDatabase(ProgramDatabase):
    """
    Database implementing beam search for parent selection.

    Beam search maintains a fixed number of "active" candidates (the beam),
    expanding from the most promising programs while pruning others.

    Configuration options (via DatabaseConfig attributes):
        beam_width: Number of candidates to keep in beam (default: 5)
        beam_selection_strategy: How to pick parent from beam
            - "best": Always pick highest scoring
            - "stochastic": Weighted random by score
            - "round_robin": Cycle through beam members
            - "diversity_weighted": Balance score and diversity (default)
        beam_diversity_weight: Weight for diversity in selection (default: 0.3)
        beam_temperature: Temperature for stochastic selection (default: 1.0)
        beam_depth_penalty: Penalty factor for deep programs (default: 0.0)
    """

    def __init__(self, name: str, config: DatabaseConfig):
        # Initialize beam-specific attributes BEFORE super().__init__()
        # because super().__init__() may call load() which needs these
        self.beam_width = getattr(config, "beam_width", 5)
        self.selection_strategy = getattr(config, "beam_selection_strategy", "diversity_weighted")
        self.diversity_weight = getattr(config, "beam_diversity_weight", 0.3)
        self.temperature = getattr(config, "beam_temperature", 1.0)
        self.depth_penalty = getattr(config, "beam_depth_penalty", 0.0)

        # Track program depths in search tree
        self.depth: Dict[str, int] = {}

        # Current beam (set of program IDs)
        self.beam: Set[str] = set()

        # Track which programs have been expanded (had children generated)
        self.expanded: Set[str] = set()

        # Round-robin state
        self._rr_index = 0

        # Statistics for analysis
        self.stats: Dict[str, Any] = {
            "total_expansions": 0,
            "max_depth_reached": 0,
            "beam_updates": 0,
            "diversity_scores": [],
        }

        # Now call super().__init__() which may trigger load()
        super().__init__(name, config)

        logger.debug(
            f"BeamSearchDatabase initialized: width={self.beam_width}, "
            f"strategy={self.selection_strategy}, diversity_weight={self.diversity_weight}"
        )

    def add(self, program: Program, iteration: Optional[int] = None, **kwargs) -> str:
        """
        Add a program to the database and update the beam.

        The beam is updated to always contain the top beam_width programs,
        considering both fitness scores and optionally diversity.

        Args:
            program: Program to add
            iteration: Current iteration (for tracking)

        Returns:
            Program ID
        """
        # Store the program
        self.programs[program.id] = program

        # Track iteration
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        # Calculate depth from parent
        if program.parent_id and program.parent_id in self.depth:
            self.depth[program.id] = self.depth[program.parent_id] + 1
        else:
            self.depth[program.id] = 0

        # Update max depth statistic
        self.stats["max_depth_reached"] = max(
            self.stats["max_depth_reached"], self.depth[program.id]
        )

        # Update the beam
        self._update_beam(program)

        # Update best program tracking (from base class)
        self._update_best_program(program)

        # Save to disk if configured
        if self.config.db_path:
            self._save_program(program)

        logger.debug(
            f"Added program {program.id} at depth {self.depth[program.id]}, "
            f"beam size: {len(self.beam)}"
        )

        return program.id

    def _update_beam(self, new_program: Program) -> None:
        """
        Update the beam to include the new program if it's good enough.

        The beam always contains the best beam_width programs based on
        fitness score, with optional diversity consideration.
        """
        # Add to beam candidates
        self.beam.add(new_program.id)

        # If beam exceeds width, prune to best beam_width
        if len(self.beam) > self.beam_width:
            self._prune_beam()
            self.stats["beam_updates"] += 1

    def _prune_beam(self) -> None:
        """
        Prune beam to beam_width, keeping the best candidates.

        Uses a combination of fitness score and diversity to select
        which programs to keep.
        """
        if len(self.beam) <= self.beam_width:
            return

        # Get all beam programs with their scores
        beam_programs = []
        for pid in self.beam:
            prog = self.programs.get(pid)
            if prog:
                score = self._get_program_score(prog)
                beam_programs.append((pid, prog, score))

        if self.diversity_weight > 0:
            # Use diversity-aware selection
            selected = self._diverse_selection(beam_programs, self.beam_width)
        else:
            # Pure fitness-based selection
            beam_programs.sort(key=lambda x: x[2], reverse=True)
            selected = [bp[0] for bp in beam_programs[: self.beam_width]]

        self.beam = set(selected)

    def _diverse_selection(self, candidates: List[Tuple[str, Program, float]], k: int) -> List[str]:
        """
        Select k programs balancing fitness and diversity.

        Uses a greedy algorithm that iteratively selects the program
        that maximizes a combination of fitness and minimum distance
        to already selected programs.

        Args:
            candidates: List of (id, program, score) tuples
            k: Number of programs to select

        Returns:
            List of selected program IDs
        """
        if len(candidates) <= k:
            return [c[0] for c in candidates]

        selected = []
        remaining = list(candidates)

        # Always include the best program
        remaining.sort(key=lambda x: x[2], reverse=True)
        selected.append(remaining.pop(0))

        # Greedily add remaining programs
        while len(selected) < k and remaining:
            best_idx = -1
            best_combined_score = -float("inf")

            for i, (pid, prog, score) in enumerate(remaining):
                # Calculate diversity as min distance to selected
                min_diversity = min(
                    self._solution_distance(prog.solution, self.programs[s[0]].solution)
                    for s in selected
                )

                # Normalize score (assume scores are in [0, 1] or similar)
                normalized_score = score

                # Combined score
                combined = (
                    1 - self.diversity_weight
                ) * normalized_score + self.diversity_weight * min_diversity

                if combined > best_combined_score:
                    best_combined_score = combined
                    best_idx = i

            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))

        return [s[0] for s in selected]

    def _solution_distance(self, solution1: str, solution2: str) -> float:
        """
        Calculate normalized distance between two code strings.

        Uses a simple character-level comparison. For production use,
        consider AST-based or embedding-based similarity.

        Returns:
            Distance in [0, 1] where 1 means completely different
        """
        if not solution1 or not solution2:
            return 1.0

        # Simple Jaccard distance on character n-grams
        n = 3

        def get_ngrams(s: str, n: int) -> Set[str]:
            return set(s[i : i + n] for i in range(len(s) - n + 1))

        ngrams1 = get_ngrams(solution1, n)
        ngrams2 = get_ngrams(solution2, n)

        if not ngrams1 and not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        similarity = intersection / union if union > 0 else 0
        return 1.0 - similarity

    def _get_program_score(self, program: Program) -> float:
        """
        Get the fitness score for a program.

        Uses combined_score if available, otherwise averages all metrics.
        Applies depth penalty if configured.

        Args:
            program: Program to score

        Returns:
            Fitness score (higher is better)
        """
        if not program.metrics:
            return 0.0

        # Get base score
        if "combined_score" in program.metrics:
            score = program.metrics["combined_score"]
        elif "score" in program.metrics:
            score = program.metrics["score"]
        else:
            # Average of all metrics
            values = [v for v in program.metrics.values() if isinstance(v, (int, float))]
            score = sum(values) / len(values) if values else 0.0

        # Apply depth penalty if configured
        if self.depth_penalty > 0 and program.id in self.depth:
            depth = self.depth[program.id]
            score = score * math.exp(-self.depth_penalty * depth)

        return score

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Program, List[Program]]:
        """
        Sample a parent program and context programs using beam search strategy.

        The parent is selected from the current beam using the configured
        selection strategy. context programs are drawn from top programs.

        Args:
            num_context_programs: Number of context programs to return

        Returns:
            Tuple of (parent, other_context_programs).
        """
        if not self.beam:
            # Fallback: use best program if beam is empty
            best = self.get_best_program()
            if best:
                self.beam.add(best.id)
            else:
                raise ValueError("Cannot sample: no programs in database")

        # Select parent based on strategy
        parent = self._select_parent()

        # Mark as expanded
        self.expanded.add(parent.id)
        self.stats["total_expansions"] += 1

        # Get context programs from top programs, excluding the parent
        n = num_context_programs or 4
        top_programs = self.get_top_programs(n + 1)
        other_context_programs = [p for p in top_programs if p.id != parent.id][:n]

        logger.debug(
            f"Beam search: selected parent {parent.id} (depth={self.depth.get(parent.id, 0)}, "
            f"score={self._get_program_score(parent):.4f}), "
            f"beam_size={len(self.beam)}, other_context_programs={len(other_context_programs)}"
        )

        return parent, other_context_programs

    def _select_parent(self) -> Program:
        """
        Select a parent program from the beam using the configured strategy.

        Returns:
            Selected parent program
        """
        beam_list = [self.programs[pid] for pid in self.beam if pid in self.programs]

        if not beam_list:
            raise ValueError("Beam is empty, cannot select parent")

        if self.selection_strategy == "best":
            return self._select_best(beam_list)
        elif self.selection_strategy == "stochastic":
            return self._select_stochastic(beam_list)
        elif self.selection_strategy == "round_robin":
            return self._select_round_robin(beam_list)
        elif self.selection_strategy == "diversity_weighted":
            return self._select_diversity_weighted(beam_list)
        else:
            logger.warning(f"Unknown strategy {self.selection_strategy}, using best")
            return self._select_best(beam_list)

    def _select_best(self, candidates: List[Program]) -> Program:
        """Select the highest scoring program."""
        return max(candidates, key=self._get_program_score)

    def _select_stochastic(self, candidates: List[Program]) -> Program:
        """
        Select using softmax-weighted random sampling.

        Higher temperature = more uniform distribution.
        Lower temperature = more greedy selection.
        """
        scores = [self._get_program_score(p) for p in candidates]

        # Apply temperature and softmax
        if self.temperature > 0:
            # Shift scores for numerical stability
            max_score = max(scores)
            exp_scores = [math.exp((s - max_score) / self.temperature) for s in scores]
            total = sum(exp_scores)
            probs = [e / total for e in exp_scores]
        else:
            # Temperature = 0 means greedy
            return self._select_best(candidates)

        # Weighted random selection
        r = random.random()
        cumsum = 0
        for i, prob in enumerate(probs):
            cumsum += prob
            if r <= cumsum:
                return candidates[i]

        return candidates[-1]

    def _select_round_robin(self, candidates: List[Program]) -> Program:
        """
        Select in round-robin order through the beam.

        Ensures all beam members get expanded equally.
        """
        # Sort by score for consistent ordering
        sorted_candidates = sorted(candidates, key=self._get_program_score, reverse=True)

        selected = sorted_candidates[self._rr_index % len(sorted_candidates)]
        self._rr_index += 1

        return selected

    def _select_diversity_weighted(self, candidates: List[Program]) -> Program:
        """
        Select balancing exploitation (high score) and exploration (diversity).

        Programs that are more different from recently expanded programs
        get a diversity bonus.
        """
        if not self.expanded:
            # No expansion history, use stochastic
            return self._select_stochastic(candidates)

        # Calculate combined scores
        combined_scores = []
        for prog in candidates:
            fitness = self._get_program_score(prog)

            # Calculate diversity from expanded programs
            recent_expanded = list(self.expanded)[-10:]  # Last 10 expanded
            if recent_expanded:
                diversity = sum(
                    self._solution_distance(prog.solution, self.programs[eid].solution)
                    for eid in recent_expanded
                    if eid in self.programs
                ) / len(recent_expanded)
            else:
                diversity = 1.0

            combined = (1 - self.diversity_weight) * fitness + self.diversity_weight * diversity
            combined_scores.append(combined)

        # Select using softmax on combined scores
        if self.temperature > 0:
            max_score = max(combined_scores)
            exp_scores = [math.exp((s - max_score) / self.temperature) for s in combined_scores]
            total = sum(exp_scores)
            probs = [e / total for e in exp_scores]

            r = random.random()
            cumsum = 0
            for i, prob in enumerate(probs):
                cumsum += prob
                if r <= cumsum:
                    return candidates[i]

        # Fallback to best combined score
        best_idx = combined_scores.index(max(combined_scores))
        return candidates[best_idx]

    def get_beam_programs(self) -> List[Program]:
        """
        Get all programs currently in the beam.

        Returns:
            List of programs in the beam, sorted by score (descending)
        """
        beam_programs = [self.programs[pid] for pid in self.beam if pid in self.programs]
        return sorted(beam_programs, key=self._get_program_score, reverse=True)

    def get_unexpanded_beam(self) -> List[Program]:
        """
        Get beam programs that haven't been expanded yet.

        Useful for analysis or alternative expansion strategies.

        Returns:
            List of unexpanded beam programs
        """
        unexpanded = [
            self.programs[pid]
            for pid in self.beam
            if pid in self.programs and pid not in self.expanded
        ]
        return sorted(unexpanded, key=self._get_program_score, reverse=True)

    def get_search_stats(self) -> Dict:
        """
        Get statistics about the beam search progress.

        Returns:
            Dictionary with search statistics
        """
        return {
            "beam_size": len(self.beam),
            "total_programs": len(self.programs),
            "total_expansions": self.stats["total_expansions"],
            "max_depth_reached": self.stats["max_depth_reached"],
            "beam_updates": self.stats["beam_updates"],
            "unexpanded_in_beam": len(self.beam - self.expanded),
            "avg_beam_depth": (
                sum(self.depth.get(pid, 0) for pid in self.beam) / len(self.beam)
                if self.beam
                else 0
            ),
        }

    def log_status(self) -> None:
        """Log the status of the beam search database."""
        stats = self.get_search_stats()
        logger.debug(
            f"BeamSearchDatabase status: {stats['total_programs']} programs, "
            f"beam_size={stats['beam_size']}, max_depth={stats['max_depth_reached']}, "
            f"expansions={stats['total_expansions']}"
        )

        # Log beam contents
        if self.beam:
            beam_progs = self.get_beam_programs()
            logger.debug("Current beam:")
            for i, prog in enumerate(beam_progs[:5]):  # Show top 5
                logger.debug(
                    f"  {i+1}. {prog.id}: score={self._get_program_score(prog):.4f}, "
                    f"depth={self.depth.get(prog.id, 0)}"
                )

    # ------------------------------------------------------------------
    # Save and Load
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:
        """
        Save the database to disk, including beam search state.

        Args:
            path: Path to save to (uses config.db_path if None)
            iteration: Current iteration number
        """
        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save each program
        for program in self.programs.values():
            prompts = None
            if (
                self.config.log_prompts
                and self.prompts_by_program
                and program.id in self.prompts_by_program
            ):
                prompts = self.prompts_by_program[program.id]
            self._save_program(program, save_path, prompts=prompts)

        # Save metadata including beam search state
        metadata = {
            "best_program_id": self.best_program_id,
            "last_iteration": iteration if iteration is not None else self.last_iteration,
            # Beam search specific state
            "beam": list(self.beam),
            "depth": self.depth,
            "expanded": list(self.expanded),
            "rr_index": self._rr_index,
            "stats": self.stats,
        }

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(
            f"Saved BeamSearchDatabase with {len(self.programs)} programs, "
            f"beam_size={len(self.beam)} to {save_path}"
        )

    def load(self, path: str) -> None:
        """
        Load the database from disk, restoring beam search state.

        Args:
            path: Path to load from
        """
        if not os.path.exists(path):
            logger.warning(f"Database path {path} does not exist, skipping load")
            return

        # Load metadata first
        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.best_program_id = metadata.get("best_program_id")
            self.last_iteration = metadata.get("last_iteration", 0)

            # Restore beam search state
            self.beam = set(metadata.get("beam", []))
            self.depth = metadata.get("depth", {})
            self.expanded = set(metadata.get("expanded", []))
            self._rr_index = metadata.get("rr_index", 0)
            saved_stats = metadata.get("stats", {})
            self.stats.update(saved_stats)

            logger.debug(
                f"Loaded metadata: last_iteration={self.last_iteration}, "
                f"beam_size={len(self.beam)}"
            )

        # Load programs
        programs_dir = os.path.join(path, "programs")
        if os.path.exists(programs_dir):
            for program_file in os.listdir(programs_dir):
                if program_file.endswith(".json"):
                    program_path = os.path.join(programs_dir, program_file)
                    try:
                        with open(program_path, "r") as f:
                            program_data = json.load(f)

                        program = Program.from_dict(program_data)
                        self.programs[program.id] = program
                    except Exception as e:
                        logger.warning(f"Error loading program {program_file}: {str(e)}")

        # Validate and reconstruct beam if needed
        self._validate_and_reconstruct_beam()

        logger.debug(f"Loaded BeamSearchDatabase with {len(self.programs)} programs from {path}")
        self.log_status()

    def _validate_and_reconstruct_beam(self) -> None:
        """
        Validate beam state after loading and reconstruct if necessary.

        This handles cases where:
        - Beam contains IDs that no longer exist in programs
        - Beam is empty but programs exist
        - Depth information is missing for some programs
        """
        # Remove invalid beam entries (programs that don't exist)
        valid_beam = {pid for pid in self.beam if pid in self.programs}
        if len(valid_beam) != len(self.beam):
            removed = len(self.beam) - len(valid_beam)
            logger.warning(f"Removed {removed} invalid entries from beam")
            self.beam = valid_beam

        # Remove invalid expanded entries
        valid_expanded = {pid for pid in self.expanded if pid in self.programs}
        self.expanded = valid_expanded

        # Reconstruct depth for programs missing it
        missing_depth = [pid for pid in self.programs if pid not in self.depth]
        if missing_depth:
            logger.debug(f"Reconstructing depth for {len(missing_depth)} programs")
            self._reconstruct_depths()

        # If beam is empty but we have programs, reconstruct beam from top programs
        if not self.beam and self.programs:
            logger.debug("Beam is empty, reconstructing from top programs")
            top_programs = self.get_top_programs(self.beam_width)
            self.beam = {p.id for p in top_programs}
            logger.debug(f"Reconstructed beam with {len(self.beam)} programs")

    def _reconstruct_depths(self) -> None:
        """
        Reconstruct depth information for all programs based on parent relationships.

        Uses BFS from root programs (those without parents) to assign depths.
        """
        # Find all root programs (no parent or parent not in database)
        roots = []
        for pid, prog in self.programs.items():
            if not prog.parent_id or prog.parent_id not in self.programs:
                self.depth[pid] = 0
                roots.append(pid)

        # BFS to assign depths
        queue = list(roots)
        visited = set(roots)

        # Build child lookup
        children: Dict[str, List[str]] = defaultdict(list)
        for pid, prog in self.programs.items():
            if prog.parent_id and prog.parent_id in self.programs:
                children[prog.parent_id].append(pid)

        while queue:
            current = queue.pop(0)
            current_depth = self.depth.get(current, 0)

            for child_id in children[current]:
                if child_id not in visited:
                    self.depth[child_id] = current_depth + 1
                    visited.add(child_id)
                    queue.append(child_id)

        # Handle orphaned programs (shouldn't happen but just in case)
        for pid in self.programs:
            if pid not in self.depth:
                self.depth[pid] = 0
                logger.warning(f"Orphaned program {pid}, assigned depth 0")
