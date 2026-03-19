"""
Unified Archive for Quality-Diversity Search.

A flat-list archive that balances quality and diversity through:
- Unified elite score (fitness + novelty)
- k-NN based novelty computation
- Deterministic crowding for replacement
- Pluggable diversity strategies

Key invariant: Programs with high elite score are protected from eviction.
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from skydiscover.search.adaevolve.archive.diversity import (
    CodeDiversity,
    DiversityStrategy,
)
from skydiscover.search.base_database import Program

logger = logging.getLogger(__name__)


@dataclass
class ArchiveConfig:
    """Configuration for UnifiedArchive."""

    # Maximum number of programs in archive
    max_size: int = 100

    # Number of neighbors for k-NN novelty computation
    k_neighbors: int = 5

    # Fraction of archive protected as elite (top by elite_score)
    elite_ratio: float = 0.2

    # Weights for elite score components (should sum to ~1.0)
    # NOTE: pareto_weight is deprecated and redistributed to fitness_weight
    pareto_weight: float = 0.0  # Deprecated - added to fitness_weight
    fitness_weight: float = 0.7  # Weight for fitness rank
    novelty_weight: float = 0.3  # Weight for novelty rank

    # Primary metric key for fitness (None = auto-detect from common names)
    fitness_key: Optional[str] = None

    # Higher is better for each metric (default True for all)
    higher_is_better: Dict[str, bool] = field(default_factory=dict)

    # Pareto multi-objective selection (opt-in: empty list = disabled)
    pareto_objectives: List[str] = field(default_factory=list)
    pareto_objectives_weight: float = 0.0


class UnifiedArchive:
    """
    Flat-list archive with unified elite scoring.

    Elite Score = fitness_weight * fitness_percentile + novelty_weight * novelty_percentile

    Where:
    - fitness_percentile: position when sorted by primary metric / n
    - novelty_percentile: position when sorted by k-NN distance / n

    Programs with high elite_score are protected from eviction.
    Replacement uses deterministic crowding (compete with most similar).
    """

    def __init__(
        self,
        config: Optional[ArchiveConfig] = None,
        diversity_strategy: Optional[DiversityStrategy] = None,
    ):
        """
        Args:
            config: Archive configuration
            diversity_strategy: Strategy for computing program distance.
                               Defaults to CodeDiversity.
        """
        self.config = config or ArchiveConfig()
        self.diversity = diversity_strategy or CodeDiversity()

        # Core storage
        self._programs: Dict[str, Program] = {}

        # Genealogy tracking
        self._parents: Dict[str, List[str]] = {}
        self._children: Dict[str, List[str]] = defaultdict(list)

        # Caches (invalidated on changes)
        self._elite_scores: Dict[str, float] = {}
        self._novelty_scores: Dict[str, float] = {}
        self._fitness_ranks: Dict[str, int] = {}
        self._dominated_flags: Dict[str, bool] = {}
        self._pareto_ranks: Dict[str, int] = {}
        self._crowding_distances: Dict[str, float] = {}
        self._pareto_percentiles: Dict[str, float] = {}
        self._cache_valid: bool = False

        logger.debug(
            f"UnifiedArchive initialized: max_size={self.config.max_size}, "
            f"k={self.config.k_neighbors}, elite_ratio={self.config.elite_ratio}"
        )

    # =========================================================================
    # Core Operations
    # =========================================================================

    def add(self, program: Program) -> bool:
        """
        Add a program to the archive.

        Strategy:
        1. If under capacity, add directly
        2. Otherwise, find eviction candidate (most similar non-protected)
        3. Replace if new program has higher elite score

        Note: Genealogy is tracked ONLY after successful addition to prevent
        orphaned entries when programs are rejected.

        Args:
            program: Program to add

        Returns:
            True if program was added, False if rejected
        """
        if program.id in self._programs:
            logger.debug(f"Program {program.id[:8]} already in archive")
            return False

        # Case 1: Under capacity - add directly
        if len(self._programs) < self.config.max_size:
            self._insert(program)
            self._track_genealogy(program)
            logger.debug(f"Added {program.id[:8]} (under capacity)")
            return True

        # Case 2: At capacity - find eviction candidate
        self._ensure_cache_valid()
        candidate_id = self._find_eviction_candidate(program)

        if candidate_id is None:
            logger.debug(f"Rejected {program.id[:8]} (all protected)")
            return False

        # Compare elite scores
        new_score = self._compute_elite_score_for_new(program)
        old_score = self._elite_scores.get(candidate_id, 0.0)

        if new_score > old_score:
            self._evict(candidate_id)
            self._insert(program)
            self._track_genealogy(program)
            logger.debug(
                f"Replaced {candidate_id[:8]} with {program.id[:8]} "
                f"(score {old_score:.3f} → {new_score:.3f})"
            )
            return True

        logger.debug(f"Rejected {program.id[:8]} " f"(score {new_score:.3f} <= {old_score:.3f})")
        return False

    def _track_genealogy(self, program: Program) -> None:
        """
        Track parent-child relationship.

        ONLY call after program is successfully added to archive.
        Safe to call when parent doesn't exist (was evicted).
        """
        parent_id = getattr(program, "parent_id", None)

        if parent_id:
            self._parents[program.id] = [parent_id]
            # Note: parent_id might not be in archive (evicted) - that's OK
            self._children[parent_id].append(program.id)
        else:
            self._parents[program.id] = []

    def _insert(self, program: Program) -> None:
        """Insert program and invalidate caches."""
        self._programs[program.id] = program
        self._invalidate_cache()

    def _evict(self, program_id: str) -> None:
        """
        Remove program from archive with complete genealogy cleanup.

        Cleans up:
        - Program's entry in _parents
        - Program's entry in _children (its children list)
        - References to program in other entries' children lists
        """
        if program_id not in self._programs:
            return

        self._cleanup_genealogy(program_id)
        del self._programs[program_id]
        self._invalidate_cache()

    def _cleanup_genealogy(self, program_id: str) -> None:
        """
        Remove all genealogy references for a program being evicted.

        This is O(n) in the number of parent entries, but eviction is already
        O(n) for finding candidates, so this doesn't change complexity.
        """
        # 1. Remove program's parent tracking
        if program_id in self._parents:
            del self._parents[program_id]

        # 2. Remove program from its parent's children list
        parents_to_clean = []
        for parent_id, children in self._children.items():
            if program_id in children:
                children.remove(program_id)
                if not children:
                    parents_to_clean.append(parent_id)

        # 3. Remove empty children lists to prevent memory bloat
        for parent_id in parents_to_clean:
            del self._children[parent_id]

        # 4. Remove program's children list
        if program_id in self._children:
            del self._children[program_id]

    def _invalidate_cache(self) -> None:
        """Mark caches as invalid."""
        self._cache_valid = False

    def _ensure_cache_valid(self) -> None:
        """Recompute caches if invalid."""
        if self._cache_valid:
            return

        programs = list(self._programs.values())
        n = len(programs)

        if n == 0:
            self._cache_valid = True
            return

        # Update diversity strategy bounds
        self.diversity.update(programs)

        # Compute fitness ranks (higher fitness = higher rank)
        fitness_sorted = sorted(programs, key=lambda p: self._get_fitness(p), reverse=True)
        self._fitness_ranks = {p.id: i for i, p in enumerate(fitness_sorted)}

        # Compute Pareto ranking on explicit objectives (no-op when unconfigured)
        self._compute_pareto_ranking(programs)
        if self._pareto_ranks:
            entries = [
                (pid, self._pareto_ranks[pid], self._crowding_distances.get(pid, 0.0))
                for pid in self._pareto_ranks
            ]
            entries.sort(key=lambda x: (x[1], -x[2]))
            n_entries = max(len(entries) - 1, 1)
            self._pareto_percentiles = {
                pid: 1.0 - (i / n_entries) for i, (pid, _, _) in enumerate(entries)
            }
        else:
            self._pareto_percentiles = {}
        self._dominated_flags = {}

        # Compute novelty scores (O(n²) but used for diversity-based sampling)
        self._novelty_scores = {p.id: self._compute_novelty(p, programs) for p in programs}

        # Compute elite scores
        self._elite_scores = {}
        for p in programs:
            self._elite_scores[p.id] = self._compute_elite_score(p, n)

        self._cache_valid = True

    # =========================================================================
    # Elite Score Computation
    # =========================================================================

    def _compute_elite_score(self, program: Program, n: int) -> float:
        """
        Compute unified elite score.

        When pareto_objectives configured:
            elite_score = fitness_weight * fitness_pct
                        + novelty_weight * novelty_pct
                        + pareto_objectives_weight * pareto_pct

        Otherwise (backward compatible):
            elite_score = (fitness_weight + pareto_weight) * fitness_pct
                        + novelty_weight * novelty_pct
        """
        fitness_rank = self._fitness_ranks.get(program.id, n - 1)
        fitness_percentile = 1.0 - (fitness_rank / max(n - 1, 1))

        novelty = self._novelty_scores.get(program.id, 0.0)
        novelty_percentile = self._novelty_to_percentile(novelty)

        if self._pareto_percentiles and self.config.pareto_objectives:
            pareto_percentile = self._pareto_percentiles.get(program.id, 0.0)
            return (
                self.config.fitness_weight * fitness_percentile
                + self.config.novelty_weight * novelty_percentile
                + self.config.pareto_objectives_weight * pareto_percentile
            )

        # No Pareto objectives: redistribute pareto_weight to fitness
        effective_fitness_weight = self.config.fitness_weight + self.config.pareto_weight
        return (
            effective_fitness_weight * fitness_percentile
            + self.config.novelty_weight * novelty_percentile
        )

    def _compute_elite_score_for_new(self, program: Program) -> float:
        """
        Compute elite score for a new program (admission decision).

        Uses same formulas as cached programs for consistent comparison.
        """
        programs = list(self._programs.values())
        n = len(programs)
        use_pareto = bool(self.config.pareto_objectives and self._pareto_ranks)

        if n == 0:
            total_weight = self.config.fitness_weight + self.config.novelty_weight
            if use_pareto:
                total_weight += self.config.pareto_objectives_weight
            else:
                total_weight += self.config.pareto_weight
            return total_weight  # max score (all components = 1.0)

        # === Fitness percentile ===
        fitness = self._get_fitness(program)
        better_count = sum(1 for p in programs if self._get_fitness(p) > fitness)

        if n == 1:
            fitness_percentile = 1.0 if better_count == 0 else 0.0
        else:
            fitness_percentile = 1.0 - (better_count / (n - 1))
        fitness_percentile = max(0.0, min(1.0, fitness_percentile))

        # === Novelty percentile ===
        novelty = self._compute_novelty(program, programs)

        if self._cache_valid and self._novelty_scores:
            existing_novelties = [self._novelty_scores.get(p.id, 0.0) for p in programs]
        else:
            existing_novelties = [self._compute_novelty(p, programs) for p in programs]

        lower_count = sum(1 for n_val in existing_novelties if n_val < novelty)
        novelty_percentile = lower_count / n
        novelty_percentile = max(0.0, min(1.0, novelty_percentile))

        # === Pareto percentile ===
        if use_pareto:
            new_vec = self._get_objective_vector(program)
            dominated_by = sum(
                1 for p in programs if self._dominates(self._get_objective_vector(p), new_vec)
            )
            pareto_percentile = 1.0 - (dominated_by / max(n, 1))
            pareto_percentile = max(0.0, min(1.0, pareto_percentile))

            return (
                self.config.fitness_weight * fitness_percentile
                + self.config.novelty_weight * novelty_percentile
                + self.config.pareto_objectives_weight * pareto_percentile
            )

        # No Pareto objectives: redistribute pareto_weight to fitness
        effective_fitness_weight = self.config.fitness_weight + self.config.pareto_weight
        return (
            effective_fitness_weight * fitness_percentile
            + self.config.novelty_weight * novelty_percentile
        )

    def _novelty_to_percentile(self, novelty: float) -> float:
        """Convert novelty score to percentile based on archive."""
        if not self._novelty_scores:
            return 0.5

        all_novelties = list(self._novelty_scores.values())
        lower_count = sum(1 for n in all_novelties if n < novelty)
        return lower_count / max(len(all_novelties), 1)

    # =========================================================================
    # Novelty (k-NN)
    # =========================================================================

    def _compute_novelty(self, program: Program, all_programs: List[Program]) -> float:
        """
        Compute novelty as average distance to k nearest neighbors.

        Higher novelty = more different from neighbors.
        """
        others = [p for p in all_programs if p.id != program.id]

        if not others:
            return 1.0  # Max novelty if alone

        # Compute distances to all other programs
        distances = [self.diversity.distance(program, other) for other in others]

        # Sort and take k nearest
        distances.sort()
        k = min(self.config.k_neighbors, len(distances))
        k_nearest = distances[:k]

        if not k_nearest:
            return 1.0

        return sum(k_nearest) / len(k_nearest)

    # =========================================================================
    # Pareto Ranking (NSGA-II)
    # =========================================================================

    def _compute_pareto_ranking(self, programs: List[Program]) -> None:
        """
        NSGA-II non-dominated sorting + crowding distance on explicit objectives.

        Only runs when self.config.pareto_objectives is non-empty.
        Populates self._pareto_ranks (layer number) and self._crowding_distances.
        """
        objectives = self.config.pareto_objectives
        if not objectives:
            self._pareto_ranks = {}
            self._crowding_distances = {}
            return

        higher_is_better = self.config.higher_is_better

        # Build objective vectors (internally: higher is always better)
        obj_vectors: Dict[str, List[float]] = {}
        for p in programs:
            vec = []
            for obj_key in objectives:
                raw_val = p.metrics.get(obj_key, 0.0)
                if not isinstance(raw_val, (int, float)):
                    raw_val = 0.0
                if not higher_is_better.get(obj_key, True):
                    raw_val = -raw_val
                vec.append(float(raw_val))
            obj_vectors[p.id] = vec

        # Non-dominated sorting into layers
        remaining = set(p.id for p in programs)
        rank = 0
        pareto_ranks: Dict[str, int] = {}
        layers: List[List[str]] = []

        while remaining:
            front = []
            for pid_a in remaining:
                dominated = False
                for pid_b in remaining:
                    if pid_a == pid_b:
                        continue
                    if self._dominates(obj_vectors[pid_b], obj_vectors[pid_a]):
                        dominated = True
                        break
                if not dominated:
                    front.append(pid_a)

            for pid in front:
                pareto_ranks[pid] = rank
                remaining.discard(pid)
            layers.append(front)
            rank += 1

        self._pareto_ranks = pareto_ranks

        # Crowding distance within each layer
        num_objectives = len(objectives)
        crowding: Dict[str, float] = {pid: 0.0 for pid in pareto_ranks}

        for layer in layers:
            if len(layer) <= 2:
                for pid in layer:
                    crowding[pid] = float("inf")
                continue

            for m in range(num_objectives):
                sorted_layer = sorted(layer, key=lambda pid: obj_vectors[pid][m])
                crowding[sorted_layer[0]] = float("inf")
                crowding[sorted_layer[-1]] = float("inf")

                obj_range = obj_vectors[sorted_layer[-1]][m] - obj_vectors[sorted_layer[0]][m]
                if obj_range < 1e-10:
                    continue

                for i in range(1, len(sorted_layer) - 1):
                    crowding[sorted_layer[i]] += (
                        obj_vectors[sorted_layer[i + 1]][m] - obj_vectors[sorted_layer[i - 1]][m]
                    ) / obj_range

        self._crowding_distances = crowding

    @staticmethod
    def _dominates(vec_a: List[float], vec_b: List[float]) -> bool:
        """True if vec_a dominates vec_b (all >= and at least one >)."""
        at_least_one_better = False
        for a, b in zip(vec_a, vec_b):
            if a < b:
                return False
            if a > b:
                at_least_one_better = True
        return at_least_one_better

    def _get_objective_vector(self, program: Program) -> List[float]:
        """Extract objective vector for a program (higher is always better internally)."""
        vec = []
        for obj_key in self.config.pareto_objectives:
            raw_val = program.metrics.get(obj_key, 0.0)
            if not isinstance(raw_val, (int, float)):
                raw_val = 0.0
            if not self.config.higher_is_better.get(obj_key, True):
                raw_val = -raw_val
            vec.append(float(raw_val))
        return vec

    # =========================================================================
    # Fitness
    # =========================================================================

    def _normalize_metric_value(self, key: str, value: Any) -> Optional[float]:
        """Convert a metric to an internal score where larger is always better."""
        from skydiscover.utils.metrics import normalize_metric_value

        return normalize_metric_value(key, value, self.config.higher_is_better)

    def _get_fitness(self, program: Program) -> float:
        """Get primary fitness value from metrics."""
        metrics = program.metrics

        # Use configured fitness key if specified
        if self.config.fitness_key is not None:
            key = self.config.fitness_key
            normalized = self._normalize_metric_value(key, metrics.get(key))
            if normalized is not None:
                return normalized
            # Configured key not found - log warning and fallback
            logger.debug(
                f"Configured fitness_key '{key}' not found in metrics, "
                f"falling back to auto-detection"
            )

        # Prefer combined_score as the canonical scalar fallback.
        normalized = self._normalize_metric_value("combined_score", metrics.get("combined_score"))
        if normalized is not None:
            return normalized

        # Try common metric names as fallback
        for key in ["score", "fitness", "accuracy", "reward"]:
            normalized = self._normalize_metric_value(key, metrics.get(key))
            if normalized is not None:
                return normalized

        # Use first numeric metric
        for key, val in metrics.items():
            normalized = self._normalize_metric_value(key, val)
            if normalized is not None:
                return normalized

        return 0.0

    # =========================================================================
    # Eviction
    # =========================================================================

    def _find_eviction_candidate(self, new_program: Program) -> Optional[str]:
        """
        Find program to potentially evict.

        Uses deterministic crowding: find most similar NON-PROTECTED program.

        Protected programs:
        - Top elite_ratio by elite_score
        """
        protected = self._get_protected_ids()

        # Find most similar non-protected program
        best_id = None
        best_dist = float("inf")

        for pid, p in self._programs.items():
            if pid in protected:
                continue

            dist = self.diversity.distance(new_program, p)
            if dist < best_dist:
                best_dist = dist
                best_id = pid

        return best_id

    def _get_protected_ids(self) -> Set[str]:
        """Get IDs of protected programs (top by elite_score + best by fitness + Pareto front)."""
        protected = set()

        # Protect top programs by elite score
        if self._elite_scores:
            elite_count = max(1, int(len(self._programs) * self.config.elite_ratio))
            sorted_ids = sorted(
                self._elite_scores.keys(), key=lambda pid: self._elite_scores[pid], reverse=True
            )
            protected.update(sorted_ids[:elite_count])

        # CRITICAL: Always protect the best program by fitness
        if self._programs:
            best_fitness_id = max(
                self._programs.keys(), key=lambda pid: self._get_fitness(self._programs[pid])
            )
            protected.add(best_fitness_id)

        # Protect Pareto front members (rank 0) when objectives are configured
        if self._pareto_ranks and self.config.pareto_objectives:
            for pid, rank in self._pareto_ranks.items():
                if rank == 0:
                    protected.add(pid)

        return protected

    # =========================================================================
    # Sampling
    # =========================================================================

    def sample_parent(self, mode: str = "balanced") -> Optional[Program]:
        """
        Sample a parent program for mutation.

        Args:
            mode: Sampling mode
                - "exploitation": Sample from top programs by fitness
                - "exploration": Sample proportional to novelty
                - "balanced": Mix of both

        Returns:
            Selected parent program, or None if archive empty
        """
        if not self._programs:
            return None

        self._ensure_cache_valid()
        programs = list(self._programs.values())

        if mode == "exploitation":
            # Sample from top programs by fitness
            top_progs = self.get_top_programs()
            if top_progs:
                return random.choice(top_progs)
            return random.choice(programs)

        elif mode == "exploration":
            # Sample proportional to novelty
            novelties = [max(self._novelty_scores.get(p.id, 0.0), 0.001) for p in programs]
            total = sum(novelties)
            if total <= 0:
                return random.choice(programs)

            r = random.random() * total
            cumsum = 0.0
            for p, n in zip(programs, novelties):
                cumsum += n
                if cumsum >= r:
                    return p
            return programs[-1]

        else:  # balanced
            if random.random() < 0.5:
                return self.sample_parent("exploitation")
            else:
                return self.sample_parent("exploration")

    def sample_other_context_programs(
        self,
        parent: Program,
        n: int = 5,
        top_k_ratio: float = 0.5,
    ) -> List[Program]:
        """
        Sample context programs for LLM context.

        Strategy: Pick programs MOST DIFFERENT from parent, but only from
        top performers. This ensures other context programs are both diverse AND
        high-quality.

        Args:
            parent: The parent program (to be diverse from)
            n: Number of other context programs to sample
            top_k_ratio: Fraction of archive to consider as "top" (default 50%)

        Returns:
            List of context programs
        """
        if not self._programs or n <= 0:
            return []

        # First, get top performers by fitness
        all_programs = list(self._programs.values())
        all_programs.sort(key=lambda p: self._get_fitness(p), reverse=True)

        # Consider top 50% (or at least 2*n programs) as candidate pool
        top_k = max(2 * n, int(len(all_programs) * top_k_ratio))
        top_programs = all_programs[:top_k]

        # Now pick most diverse FROM the top performers
        candidates = []
        for p in top_programs:
            if p.id != parent.id:
                dist = self.diversity.distance(parent, p)
                candidates.append((p, dist))

        candidates.sort(key=lambda x: -x[1])

        return [p for p, _ in candidates[:n]]

    # =========================================================================
    # Genealogy
    # =========================================================================

    def get_children(self, program_id: str) -> List[Program]:
        """Get all children of a program (for sibling context)."""
        child_ids = self._children.get(program_id, [])
        return [self._programs[cid] for cid in child_ids if cid in self._programs]

    def get_parents(self, program_id: str) -> List[Program]:
        """Get parents of a program."""
        parent_ids = self._parents.get(program_id, [])
        return [self._programs[pid] for pid in parent_ids if pid in self._programs]

    def find_merge_candidates(self) -> Optional[Tuple[Program, Program, Program]]:
        """
        Find two programs suitable for merging.

        Looks for two top programs that share a common ancestor.

        Returns:
            Tuple of (program_a, program_b, common_ancestor) or None
        """
        # Get top programs by fitness
        top_progs = self.get_top_programs()

        if len(top_progs) < 2:
            return None

        # Try to find a pair with common ancestor
        for i, pa in enumerate(top_progs[:-1]):
            for pb in top_progs[i + 1 :]:
                ancestor_id = self._find_common_ancestor(pa.id, pb.id)
                if ancestor_id and ancestor_id in self._programs:
                    return (pa, pb, self._programs[ancestor_id])

        return None

    def _find_common_ancestor(self, id_a: str, id_b: str) -> Optional[str]:
        """Find most recent common ancestor of two programs."""
        # Get all ancestors of a
        ancestors_a: Set[str] = set()
        queue = [id_a]
        while queue:
            current = queue.pop()
            for parent in self._parents.get(current, []):
                if parent not in ancestors_a:
                    ancestors_a.add(parent)
                    queue.append(parent)

        # Walk up from b and find first intersection
        queue = [id_b]
        visited: Set[str] = set()
        while queue:
            current = queue.pop()
            if current in ancestors_a:
                return current
            visited.add(current)
            for parent in self._parents.get(current, []):
                if parent not in visited:
                    queue.append(parent)

        return None

    def add_merged_program(self, program: Program, parent_ids: List[str]) -> bool:
        """
        Add a program created by merging multiple parents.

        Note: Genealogy is tracked ONLY after successful addition to prevent
        orphaned entries when programs are rejected.

        Args:
            program: The merged program
            parent_ids: List of parent program IDs

        Returns:
            True if added, False if rejected
        """
        if program.id in self._programs:
            return False

        # Case 1: Under capacity - add directly
        if len(self._programs) < self.config.max_size:
            self._insert(program)
            self._track_merged_genealogy(program, parent_ids)
            return True

        # Case 2: At capacity - find eviction candidate
        self._ensure_cache_valid()
        candidate_id = self._find_eviction_candidate(program)

        if candidate_id is None:
            return False

        new_score = self._compute_elite_score_for_new(program)
        old_score = self._elite_scores.get(candidate_id, 0.0)

        if new_score > old_score:
            self._evict(candidate_id)
            self._insert(program)
            self._track_merged_genealogy(program, parent_ids)
            return True

        return False

    def _track_merged_genealogy(self, program: Program, parent_ids: List[str]) -> None:
        """Track genealogy for merged program with multiple parents."""
        self._parents[program.id] = list(parent_ids)
        for parent_id in parent_ids:
            self._children[parent_id].append(program.id)

    # =========================================================================
    # Accessors
    # =========================================================================

    def get_best(self) -> Optional[Program]:
        """Get program with highest fitness."""
        if not self._programs:
            return None
        return max(self._programs.values(), key=lambda p: self._get_fitness(p))

    def get_top_programs(self, n: Optional[int] = None) -> List[Program]:
        """
        Get top programs by fitness.

        Args:
            n: Number of programs to return. If None, returns top ~20%
               (at least 1, at most 10).

        Returns:
            List of top programs sorted by fitness (best first)
        """
        programs = list(self._programs.values())
        if not programs:
            return []

        programs.sort(key=lambda p: self._get_fitness(p), reverse=True)

        if n is not None:
            return programs[:n]

        # Default: top ~20% (at least 1, at most 10)
        top_count = max(1, min(10, len(programs) // 5))
        return programs[:top_count]

    def get_pareto_front(self) -> List[Program]:
        """Get non-dominated Pareto front when objectives configured, else top programs."""
        if self.config.pareto_objectives and self._pareto_ranks:
            self._ensure_cache_valid()
            return [
                self._programs[pid]
                for pid, rank in self._pareto_ranks.items()
                if rank == 0 and pid in self._programs
            ]
        return self.get_top_programs()

    def get_all(self) -> List[Program]:
        """Get all programs in archive."""
        return list(self._programs.values())

    def get(self, program_id: str) -> Optional[Program]:
        """Get program by ID."""
        return self._programs.get(program_id)

    def size(self) -> int:
        """Number of programs in archive."""
        return len(self._programs)

    def contains(self, program_id: str) -> bool:
        """Check if program is in archive."""
        return program_id in self._programs

    def stats(self) -> Dict[str, Any]:
        """Get archive statistics."""
        self._ensure_cache_valid()

        top_count = len(self.get_top_programs())
        pareto_front = self.get_pareto_front() if self.config.pareto_objectives else []
        return {
            "size": len(self._programs),
            "max_size": self.config.max_size,
            "top_count": top_count,
            "pareto_count": len(pareto_front) if pareto_front else top_count,
            "pareto_front_size": len(pareto_front),
            "protected_count": len(self._get_protected_ids()),
            "k_neighbors": self.config.k_neighbors,
        }

    def __len__(self) -> int:
        return len(self._programs)

    def __contains__(self, program_id: str) -> bool:
        return program_id in self._programs

    # =========================================================================
    # Serialization
    # =========================================================================

    def get_genealogy_state(self) -> Dict[str, Any]:
        """
        Get genealogy state for checkpointing.

        Returns:
            Dict with parents and children mappings.
        """
        return {
            "parents": dict(self._parents),
            "children": {k: list(v) for k, v in self._children.items()},
        }

    def set_genealogy_state(self, state: Dict[str, Any]) -> None:
        """
        Restore genealogy state from checkpoint.

        Only restores relationships for programs currently in the archive.

        Args:
            state: Dict with parents and children mappings from get_genealogy_state()
        """
        if not state:
            return

        # Restore parents (only for programs in archive)
        parents_data = state.get("parents", {})
        for prog_id, parent_list in parents_data.items():
            if prog_id in self._programs:
                self._parents[prog_id] = list(parent_list)

        # Restore children (only relationships where both exist in archive)
        children_data = state.get("children", {})
        for parent_id, child_list in children_data.items():
            valid_children = [cid for cid in child_list if cid in self._programs]
            if valid_children:
                self._children[parent_id] = valid_children
