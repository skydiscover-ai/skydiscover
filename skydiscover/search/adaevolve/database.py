"""
AdaEvolve Database - Population management with adaptive search intensity.

A clean implementation that embodies adaptive optimization principles:
1. Accumulated improvement signal per island determines search intensity
2. UCB with decayed magnitude rewards for island selection
3. High productivity → exploit, Low productivity → explore
4. UnifiedArchive per island maintains diversity even during exploitation
5. Dynamic island spawning when global stagnation is detected
6. Paradigm breakthrough for high-level strategy shifts
"""

import json
import logging
import os
import random
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.adaevolve.adaptation import AdaptiveState, MultiDimensionalAdapter
from skydiscover.search.adaevolve.archive import (
    ArchiveConfig,
    UnifiedArchive,
    create_diversity_strategy,
)
from skydiscover.search.adaevolve.paradigm import ParadigmTracker
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.utils.metrics import compute_proxy_score, get_score

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Sampling Mode Labels (injected into prompt by the framework's
# _format_current_program via the parent dict key)
# ------------------------------------------------------------------

# --- Code / Algorithm Optimization Labels ---

EXPLORE_LABEL = """\
## PARENT SELECTION CONTEXT
This parent was selected through diversity-driven sampling to explore different regions.

### EXPLORATION GUIDANCE
- Consider alternative algorithmic approaches
- Don't be constrained by the parent's approach
- Look for fundamentally different algorithms or novel techniques
- Balance creativity with correctness

Your goal: Discover new approaches that might outperform current solutions."""

EXPLOIT_LABEL = """\
## PARENT SELECTION CONTEXT
This parent was selected from the archive of top-performing programs.

### OPTIMIZATION GUIDANCE
- This solution works well, but meaningful improvements are still possible
- You may refine the existing approach OR introduce better algorithms
- Consider: algorithmic improvements, better data structures, efficient libraries
- Ensure correctness is maintained

Your goal: Improve upon this solution."""

# --- Prompt Optimization Labels ---

EXPLORE_LABEL_PROMPT_OPT = """\
## PARENT SELECTION CONTEXT
This prompt was selected through diversity-driven sampling to explore different instruction strategies.

### EXPLORATION GUIDANCE
- Try a fundamentally different prompt structure or instruction strategy
- Don't be constrained by the parent prompt's phrasing or approach
- Consider: different reasoning guidance, output format changes, adding/removing examples, role changes
- A completely different style of instruction may unlock better LLM performance

Your goal: Discover new prompt strategies that might outperform current approaches."""

EXPLOIT_LABEL_PROMPT_OPT = """\
## PARENT SELECTION CONTEXT
This prompt was selected from the archive of top-performing prompts.

### REFINEMENT GUIDANCE
- This prompt works well, but meaningful improvements are still possible
- Refine the wording, tighten constraints, clarify ambiguous instructions
- Consider: more precise language, better reasoning guidance, stronger output format enforcement
- Small targeted edits to a good prompt can yield significant score gains

Your goal: Refine and improve this prompt."""


# ------------------------------------------------------------------
# Heterogeneous Island Configuration Presets
# ------------------------------------------------------------------
# Each preset defines different weights for the elite score computation,
# creating islands that specialize in different aspects of the search.

ISLAND_CONFIG_PRESETS = [
    {
        "name": "balanced",
        "description": "Balanced quality-diversity tradeoff (default)",
        "pareto_weight": 0.4,
        "fitness_weight": 0.3,
        "novelty_weight": 0.3,
        "elite_ratio": 0.2,
    },
    {
        "name": "quality",
        "description": "Focuses on fitness/quality over diversity",
        "pareto_weight": 0.2,
        "fitness_weight": 0.6,
        "novelty_weight": 0.2,
        "elite_ratio": 0.3,
    },
    {
        "name": "diversity",
        "description": "Focuses on novelty/diversity over quality",
        "pareto_weight": 0.3,
        "fitness_weight": 0.2,
        "novelty_weight": 0.5,
        "elite_ratio": 0.1,
    },
    {
        "name": "pareto",
        "description": "Strongly favors Pareto-optimal solutions",
        "pareto_weight": 0.6,
        "fitness_weight": 0.2,
        "novelty_weight": 0.2,
        "elite_ratio": 0.2,
    },
    {
        "name": "exploration",
        "description": "Aggressive exploration with minimal elite protection",
        "pareto_weight": 0.2,
        "fitness_weight": 0.3,
        "novelty_weight": 0.5,
        "elite_ratio": 0.05,
    },
]


def get_island_config_preset(name: str) -> Dict[str, Any]:
    """Get an island configuration preset by name."""
    for preset in ISLAND_CONFIG_PRESETS:
        if preset["name"] == name:
            return preset.copy()
    raise ValueError(f"Unknown island config preset: {name}")


class AdaEvolveDatabase(ProgramDatabase):
    """
    AdaEvolve population database with adaptive multi-island search.

    Key Design Principles:
    1. MultiDimensionalAdapter handles ALL per-island adaptive state
    2. No separate island arrays - adapter.states[i] is the adaptive state for island i
    3. UnifiedArchive per island for quality-diversity (can be disabled for ablation)
    4. No explicit stagnation tracking - search intensity handles exploration automatically
    5. UCB with decayed magnitude rewards prevents breakthrough memory problem
    6. Dynamic island spawning when global productivity drops
    7. Paradigm breakthrough for high-level strategy shifts
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)

        # Language-aware label selection (set by Runner after creation)
        # Default to "python"; overridden to "text" for prompt optimization
        self.language: str = "python"

        # Configuration
        self.num_islands = getattr(config, "num_islands", 4)
        self.current_island = 0
        self.migration_interval = getattr(config, "migration_interval", 50)
        self.migration_count = getattr(config, "migration_count", 3)
        self._iteration_count = 0
        self.population_size = config.population_size
        self.higher_is_better = getattr(config, "higher_is_better", {}) or {}
        self.fitness_key = getattr(config, "fitness_key", None)
        self.pareto_objectives = list(getattr(config, "pareto_objectives", []) or [])

        # Unified archive flag (can be disabled for ablation studies)
        self.use_unified_archive = getattr(config, "use_unified_archive", True)

        # Adaptive configuration
        self.decay = getattr(config, "decay", 0.9)
        self.intensity_min = getattr(config, "intensity_min", 0.1)
        self.intensity_max = getattr(config, "intensity_max", 0.7)

        # Ablation flags for adaptive mechanisms
        # use_adaptive_search: When False, use fixed exploration ratio instead of G-based intensity
        # use_ucb_selection: When False, use round-robin island selection instead of UCB
        # use_migration: When False, disable inter-island migration
        self.use_adaptive_search = getattr(config, "use_adaptive_search", True)
        self.use_ucb_selection = getattr(config, "use_ucb_selection", True)
        self.use_migration = getattr(config, "use_migration", True)
        self.fixed_intensity = getattr(config, "fixed_intensity", 0.4)

        # Validate intensity bounds
        if self.intensity_min > self.intensity_max:
            logger.warning(
                f"intensity_min ({self.intensity_min}) > intensity_max ({self.intensity_max}). "
                f"This inverts the exploration/exploitation logic! Swapping values."
            )
            self.intensity_min, self.intensity_max = self.intensity_max, self.intensity_min

        if not (0.0 <= self.decay <= 1.0):
            logger.warning(f"decay ({self.decay}) should be in [0, 1]. Clamping.")
            self.decay = max(0.0, min(1.0, self.decay))

        # other context program mix (local vs global)
        self.local_context_program_ratio = getattr(config, "local_context_program_ratio", 0.6)

        # Dynamic island spawning configuration
        self.use_dynamic_islands = getattr(config, "use_dynamic_islands", False)
        self.max_islands = getattr(config, "max_islands", 8)
        self.spawn_productivity_threshold = getattr(config, "spawn_productivity_threshold", 0.02)
        self.spawn_cooldown = getattr(config, "spawn_cooldown_iterations", 50)
        self.last_spawn_iteration = -self.spawn_cooldown
        self.island_config_names: List[str] = ["balanced"] * self.num_islands

        if self.use_dynamic_islands and not self.use_unified_archive:
            logger.warning(
                "use_dynamic_islands=true requires use_unified_archive=true. "
                "Dynamic island spawning will be disabled."
            )

        # Paradigm breakthrough configuration
        self.use_paradigm_breakthrough = getattr(config, "use_paradigm_breakthrough", False)
        if self.use_paradigm_breakthrough:
            self.paradigm_tracker = ParadigmTracker(
                window_size=getattr(config, "paradigm_window_size", 30),
                improvement_threshold=getattr(config, "paradigm_improvement_threshold", 0.05),
                max_paradigm_uses=getattr(config, "paradigm_max_uses", 5),
                max_tried_paradigms=getattr(config, "paradigm_max_tried", 10),
                num_paradigms_to_generate=getattr(config, "paradigm_num_to_generate", 3),
            )
        else:
            self.paradigm_tracker = None

        # Multi-dimensional adapter handles ALL per-island adaptive state
        self.adapter = MultiDimensionalAdapter(decay=self.decay)
        for i in range(self.num_islands):
            state = AdaptiveState(
                decay=self.decay,
                intensity_min=self.intensity_min,
                intensity_max=self.intensity_max,
            )
            self.adapter.add_dimension(state)

        # Per-island storage: UnifiedArchive (default) or legacy list
        if self.use_unified_archive:
            self.archives: List[UnifiedArchive] = []
            self._init_archives(config)
            self.islands = None  # Not used in archive mode
            self.children_map = None  # Archive handles genealogy
        else:
            self.archives = None  # Not used in legacy mode
            self.islands: List[List[Program]] = [[] for _ in range(self.num_islands)]
            self.children_map: List[Dict[str, List[str]]] = [{} for _ in range(self.num_islands)]
            self._diversity_strategy_type = getattr(config, "diversity_strategy", "code")

        # Global best tracking
        self._global_best_score = float("-inf")

        # Cached global Pareto front (lazy, invalidated on population changes)
        self._global_pareto_cache: Optional[List[Program]] = None
        self._global_pareto_cache_valid: bool = False

        # Last sampling mode (stashed by sample() for the controller to read)
        self._last_sampling_mode: Optional[str] = None

        logger.info(
            f"AdaEvolveDatabase initialized: "
            f"num_islands={self.num_islands}, "
            f"decay={self.decay}, "
            f"intensity=[{self.intensity_min}, {self.intensity_max}], "
            f"migration={self.use_migration} (interval={self.migration_interval}), "
            f"unified_archive={self.use_unified_archive}, "
            f"adaptive_search={self.use_adaptive_search}, "
            f"ucb_selection={self.use_ucb_selection}, "
            f"dynamic_islands={self.use_dynamic_islands}, "
            f"paradigm_breakthrough={self.use_paradigm_breakthrough}, "
            f"multiobjective={self.is_multiobjective_enabled()}"
        )

    def _init_archives(self, config: DatabaseConfig) -> None:
        """Initialize per-island UnifiedArchives."""
        higher_is_better = getattr(config, "higher_is_better", {})
        pareto_objectives = getattr(config, "pareto_objectives", [])
        pareto_objectives_weight = getattr(config, "pareto_objectives_weight", 0.0)
        self._diversity_strategy_type = getattr(config, "diversity_strategy", "code")

        for i in range(self.num_islands):
            archive_config = ArchiveConfig(
                max_size=config.population_size,
                k_neighbors=getattr(config, "k_neighbors", 5),
                elite_ratio=getattr(config, "archive_elite_ratio", 0.2),
                pareto_weight=getattr(config, "pareto_weight", 0.4),
                fitness_weight=getattr(config, "fitness_weight", 0.3),
                novelty_weight=getattr(config, "novelty_weight", 0.3),
                higher_is_better=higher_is_better,
                pareto_objectives=pareto_objectives,
                pareto_objectives_weight=pareto_objectives_weight,
                fitness_key=getattr(config, "fitness_key", None),
            )

            # Create FRESH diversity strategy per island
            # This is critical for stateful strategies like MetricDiversity
            # which maintain internal state (KNN archive) that would be
            # contaminated if shared across islands
            diversity_strategy = create_diversity_strategy(
                self._diversity_strategy_type,
                higher_is_better=higher_is_better,
            )

            archive = UnifiedArchive(
                config=archive_config,
                diversity_strategy=diversity_strategy,
            )
            self.archives.append(archive)

        logger.debug(
            f"Initialized {self.num_islands} archives: "
            f"max_size={config.population_size}, diversity={self._diversity_strategy_type}"
        )

    # =========================================================================
    # Population Storage Access
    # =========================================================================

    @property
    def active_programs(self) -> Dict[str, Program]:
        """Programs currently in all island populations."""
        result = {}
        if self.use_unified_archive and self.archives:
            for archive in self.archives:
                for p in archive.get_all():
                    result[p.id] = p
        else:
            for island in self.islands:
                for p in island:
                    result[p.id] = p
        return result

    def get_island_population(self, island_idx: int) -> List[Program]:
        """Get all programs in a specific island."""
        if 0 <= island_idx < self.num_islands:
            if self.use_unified_archive and self.archives:
                return self.archives[island_idx].get_all()
            else:
                return list(self.islands[island_idx])
        return []

    def get_island_size(self, island_idx: int) -> int:
        """Get number of programs in a specific island."""
        if 0 <= island_idx < self.num_islands:
            if self.use_unified_archive and self.archives:
                return self.archives[island_idx].size()
            else:
                return len(self.islands[island_idx])
        return 0

    # =========================================================================
    # Core Interface
    # =========================================================================

    def _get_mode_labels(self) -> Tuple[str, str]:
        """Return (explore_label, exploit_label) appropriate for the language."""
        if self.language.lower() in ("text", "prompt"):
            return EXPLORE_LABEL_PROMPT_OPT, EXPLOIT_LABEL_PROMPT_OPT
        return EXPLORE_LABEL, EXPLOIT_LABEL

    def seed_all_islands(self, program: Program, iteration: Optional[int] = None) -> None:
        """
        Seed all islands with copies of the initial program.

        Args:
            program: The initial/seed program to copy to all islands
            iteration: Current iteration (for tracking)
        """
        logger.info(f"Seeding all {self.num_islands} islands with initial program")

        for island_idx in range(self.num_islands):
            if island_idx == 0:
                # Add original program to island 0
                self.add(program, iteration=iteration, target_island=0)
            else:
                # Create a copy with new ID for other islands
                copy = Program(
                    id=str(uuid.uuid4()),
                    solution=program.solution,
                    language=program.language,
                    metrics=program.metrics.copy() if program.metrics else {},
                    iteration_found=iteration or 0,
                    parent_id=None,
                    generation=0,
                    metadata={"seeded_to_island": island_idx},
                )
                self.add(copy, iteration=iteration, target_island=island_idx)

        logger.info(
            f"All islands seeded. Island sizes: "
            f"{[self.get_island_size(i) for i in range(self.num_islands)]}"
        )

    def add(
        self,
        program: Program,
        iteration: Optional[int] = None,
        parent_id: Optional[str] = None,
        target_island: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Add a program to the population and update adaptive state.

        Args:
            program: Program to add
            iteration: Current iteration (for tracking)
            parent_id: Parent's ID (for genealogy)
            target_island: Specific island (for migrations). None = current_island.

        Returns:
            Program ID
        """
        island_idx = target_island if target_island is not None else self.current_island
        is_migration = target_island is not None and target_island != self.current_island

        if island_idx < 0 or island_idx >= self.num_islands:
            raise ValueError(f"Invalid island index {island_idx}")

        # Update iteration tracking
        if iteration is not None:
            program.iteration_found = iteration
            self.last_iteration = max(self.last_iteration, iteration)

        # Add to archive or legacy list
        was_added = False
        if self.use_unified_archive and self.archives:
            was_added = self.archives[island_idx].add(program)
            if was_added:
                self.programs[program.id] = program
            else:
                logger.debug(
                    f"Archive rejected program {program.id[:8]} on island {island_idx} "
                    f"(fitness={self._get_fitness(program):.4f})"
                )
        else:
            # Legacy mode: list-based storage
            self.programs[program.id] = program
            self.islands[island_idx].append(program)
            was_added = True

            # Track sibling relationship (only for mutations, not migrations)
            if parent_id is not None and not is_migration:
                self.children_map[island_idx].setdefault(parent_id, []).append(program.id)

            # Enforce population limit in legacy mode
            self._enforce_island_population_limit(island_idx)

        if was_added:
            # Update adaptive state
            fitness = self._get_fitness(program)
            if not is_migration:
                # Regular evaluation: full update (UCB rewards, visits, G, best_score)
                self.adapter.record_evaluation(island_idx, fitness)
            else:
                # Migration: update best_score and G only (for correct search intensity)
                # UCB stats remain unchanged (island didn't earn the improvement)
                # This fixes: 1) future delta calculations, 2) exploitation mode trigger
                self.adapter.receive_external_improvement(island_idx, fitness)

            # Invalidate BEFORE _update_best_program so it can read the stale
            # cache as the "previous" front and detect front membership changes.
            self._invalidate_global_pareto_cache()

            # Update global best and track for paradigm
            global_improved = self._update_best_program(program)

            # Record improvement for paradigm tracking
            if self.paradigm_tracker is not None and not is_migration:
                self.paradigm_tracker.record_improvement(global_improved, self._global_best_score)

            # Save if configured
            if self.config.db_path:
                self._save_program(program)

            logger.debug(
                f"Added program {program.id[:8]} to island {island_idx} "
                f"(migration={is_migration})"
            )

        return program.id

    def sample(
        self,
        num_context_programs: Optional[int] = 4,
        force_exploration: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Program], Dict[str, List[Program]]]:
        """
        Sample parent and other context programs using adaptive search intensity.

        The search intensity determines sampling mode:
        - High intensity → exploration mode (sample by novelty)
        - Low intensity → exploitation mode (sample by fitness)

        UnifiedArchive maintains diversity even during exploitation via
        elite_score which combines fitness, novelty, and Pareto status.

        Returns the standard framework format:
        - parent_dict: Dict mapping a label string to one parent Program.
          The label is EXPLORE_LABEL, EXPLOIT_LABEL, or "" (balanced).
        - context_programs_dict: Dict mapping "" to a list of context programs.

        The sampling mode is also stored on self._last_sampling_mode for
        the controller to read (for logging, paradigm, sibling context).

        Args:
            num_context_programs: Number of context programs
            force_exploration: Force exploration mode

        Returns:
            Tuple of (parent_dict, context_programs_dict)
        """
        island_idx = self.current_island

        if self.use_unified_archive and self.archives:
            return self._sample_from_archive(island_idx, num_context_programs, force_exploration)
        else:
            return self._sample_legacy(island_idx, num_context_programs, force_exploration)

    def _sample_from_archive(
        self,
        island_idx: int,
        num_context_programs: Optional[int] = 4,
        force_exploration: bool = False,
    ) -> Tuple[Dict[str, Program], Dict[str, List[Program]]]:
        """Sample using the per-island unified archive."""
        archive = self.archives[island_idx]

        if archive.size() == 0:
            raise ValueError(f"Cannot sample: island {island_idx} is empty")

        # Get search intensity: adaptive (G-based) or fixed
        if self.use_adaptive_search:
            intensity = self.adapter.get_search_intensity(island_idx)
        else:
            intensity = self.fixed_intensity

        if force_exploration:
            intensity = self.intensity_max

        # Determine sampling mode based on intensity
        # Formula: exploration=intensity%, exploitation=(1-intensity)*70%, balanced=(1-intensity)*30%
        # Example with intensity=0.4: exploration=40%, exploitation=42%, balanced=18%
        rand = random.random()
        if rand < intensity:
            mode = "exploration"
        elif rand < intensity + (1 - intensity) * 0.7:
            mode = "exploitation"
        else:
            mode = "balanced"

        # Sample parent based on mode
        population = archive.get_all()
        if mode == "exploitation":
            if archive.config.pareto_objectives and archive._pareto_ranks:
                parent = self._sample_pareto_front(archive, population)
            else:
                parent = self._sample_top(population)
        else:
            # exploration and balanced use archive's novelty-aware sampling
            parent = archive.sample_parent(mode)

        # Hybrid context programs: local diversity + global top
        num = num_context_programs or 4
        local_count = max(1, int(num * self.local_context_program_ratio))
        global_count = num - local_count

        # Local: most different from parent (but from top performers - see sample_other_context_programs)
        local_context_programs = archive.sample_other_context_programs(parent, local_count)

        # Global: top performers across all islands (cross-pollination)
        global_context_programs = self._sample_global_top(parent.id, global_count)

        other_context_programs = local_context_programs + global_context_programs

        # Map mode to label for the framework's prompt injection
        explore_label, exploit_label = self._get_mode_labels()
        if mode == "exploration":
            label = explore_label
        elif mode == "exploitation":
            label = exploit_label
        else:
            label = ""

        # Stash mode for controller to read (logging, paradigm, sibling context)
        self._last_sampling_mode = mode

        logger.debug(
            f"Sampled parent {parent.id[:8]} from island {island_idx} "
            f"in {mode} mode (intensity={intensity:.2f})"
        )

        return {label: parent}, {"": other_context_programs}

    def _sample_legacy(
        self,
        island_idx: int,
        num_context_programs: Optional[int] = 4,
        force_exploration: bool = False,
    ) -> Tuple[Dict[str, Program], Dict[str, List[Program]]]:
        """Sample using legacy list-based logic."""
        population = self.islands[island_idx]

        if not population:
            raise ValueError(f"Cannot sample: island {island_idx} is empty")

        # Get search intensity: adaptive (G-based) or fixed
        if self.use_adaptive_search:
            intensity = self.adapter.get_search_intensity(island_idx)
        else:
            intensity = self.fixed_intensity

        if force_exploration:
            intensity = self.intensity_max

        # Determine sampling mode based on intensity
        # Formula: exploration=intensity%, exploitation=(1-intensity)*70%, balanced=(1-intensity)*30%
        # Example with intensity=0.4: exploration=40%, exploitation=42%, balanced=18%
        rand = random.random()
        if rand < intensity:
            parent = self._sample_random(population)
            mode = "exploration"
        elif rand < intensity + (1 - intensity) * 0.7:
            parent = self._sample_top(population)
            mode = "exploitation"
        else:
            parent = self._sample_weighted(population)
            mode = "balanced"

        # Sample context programs from ALL islands (global cross-pollination)
        num = num_context_programs or 4
        other_context_programs = self._sample_global_top(parent.id, num)

        # Map mode to label for the framework's prompt injection
        explore_label, exploit_label = self._get_mode_labels()
        if mode == "exploration":
            label = explore_label
        elif mode == "exploitation":
            label = exploit_label
        else:
            label = ""

        # Stash mode for controller to read (logging, paradigm, sibling context)
        self._last_sampling_mode = mode

        logger.debug(
            f"Sampled parent {parent.id[:8]} from island {island_idx} "
            f"in {mode} mode (intensity={intensity:.2f})"
        )

        return {label: parent}, {"": other_context_programs}

    def _sample_random(self, population: List[Program]) -> Program:
        """Sample uniformly at random (exploration)."""
        return random.choice(population)

    def _sample_top(self, population: List[Program]) -> Program:
        """Sample from top performers (exploitation)."""
        sorted_pop = sorted(population, key=self._get_fitness, reverse=True)
        top_k = max(1, len(sorted_pop) // 4)
        return random.choice(sorted_pop[:top_k])

    def _sample_pareto_front(self, archive, population: List[Program]) -> Program:
        """Sample from Pareto front weighted by crowding distance.

        Falls back to _sample_top if front is too small.
        """
        archive._ensure_cache_valid()
        front_programs = [
            archive.get(pid)
            for pid, rank in archive._pareto_ranks.items()
            if rank == 0 and archive.get(pid) is not None
        ]

        if len(front_programs) < 2:
            return self._sample_top(population)

        weights = []
        for p in front_programs:
            cd = archive._crowding_distances.get(p.id, 0.0)
            if cd == float("inf"):
                cd = 1e6
            weights.append(max(cd, 0.001))

        return random.choices(front_programs, weights=weights, k=1)[0]

    def _sample_weighted(self, population: List[Program]) -> Program:
        """Sample weighted by fitness (balanced)."""
        weights = []
        for prog in population:
            fitness = self._get_fitness(prog)
            weights.append(max(fitness, 0.001))  # Avoid zero weights

        total = sum(weights)
        weights = [w / total for w in weights]

        return random.choices(population, weights=weights, k=1)[0]

    def _sample_global_top(self, exclude_id: str, n: int) -> List[Program]:
        """Sample top programs from ALL islands for cross-pollination."""
        all_programs = self._all_population_programs()
        candidates = [p for p in all_programs if p.id != exclude_id]

        if len(candidates) <= n:
            return candidates

        if self.is_multiobjective_enabled():
            pareto_front = [p for p in self.get_global_pareto_front() if p.id != exclude_id]
            if len(pareto_front) >= n:
                return pareto_front[:n]

            front_ids = {program.id for program in pareto_front}
            remaining = sorted(
                [program for program in candidates if program.id not in front_ids],
                key=self._get_fitness,
                reverse=True,
            )
            return pareto_front + remaining[: max(0, n - len(pareto_front))]

        sorted_candidates = sorted(candidates, key=self._get_fitness, reverse=True)
        return sorted_candidates[:n]

    def _enforce_island_population_limit(self, island_idx: int) -> None:
        """Remove worst programs if island exceeds population limit (legacy mode only)."""
        if self.use_unified_archive:
            return  # Archives handle their own limits

        population = self.islands[island_idx]

        if len(population) <= self.population_size:
            return

        # Sort by fitness (best first)
        population.sort(key=self._get_fitness, reverse=True)

        # Keep top population_size, remove rest
        removed = population[self.population_size :]
        self.islands[island_idx] = population[: self.population_size]

        # Also remove from global registry (but preserve best program)
        for prog in removed:
            if prog.id in self.programs and prog.id != self.best_program_id:
                del self.programs[prog.id]

        logger.debug(
            f"Removed {len(removed)} programs from island {island_idx} "
            f"to enforce population limit"
        )

    # =========================================================================
    # Island Lifecycle
    # =========================================================================

    def end_iteration(self, iteration: int) -> None:
        """
        End-of-iteration housekeeping.

        Handles:
        - Dynamic island spawning (if enabled and stagnating)
        - Island selection (UCB with decayed magnitude rewards OR round-robin)
        - Migration (at interval)
        """
        self._iteration_count = iteration

        # Check if we should spawn a new island
        if self._should_spawn_island():
            self._spawn_island()

        # Select next island: UCB (adaptive) or round-robin (ablation)
        if self.use_ucb_selection:
            self.current_island = self.adapter.select_dimension_ucb(iteration)
        else:
            # Round-robin selection for ablation
            # Use (iteration + 1) because this is called at END of current iteration
            # and sets the island for the NEXT iteration
            self.current_island = (iteration + 1) % self.num_islands

        # Periodic migration (can be disabled for ablation)
        if self.use_migration and iteration > 0 and iteration % self.migration_interval == 0:
            self._migrate()
            logger.info(f"Migration completed at iteration {iteration}")

    def _migrate(self) -> None:
        """
        Ring migration: copy top programs to next island.

        Ring topology: island i → island (i+1) % num_islands
        """
        if self.use_unified_archive and self.archives:
            self._migrate_archives()
        else:
            self._migrate_legacy()

    def _migrate_archives(self) -> None:
        """Migrate top programs between archives."""
        for src_island in range(self.num_islands):
            dest_island = (src_island + 1) % self.num_islands

            # Get top programs from source
            top_programs = self.archives[src_island].get_top_programs(self.migration_count)

            if not top_programs:
                continue

            for program in top_programs:
                # Skip if already in destination
                if self._has_duplicate_solution(dest_island, program.solution):
                    continue

                # Create migrant copy
                migrant = Program(
                    id=str(uuid.uuid4()),
                    solution=program.solution,
                    language=program.language,
                    metrics=program.metrics.copy() if program.metrics else {},
                    iteration_found=program.iteration_found,
                    parent_id=program.id,
                    generation=program.generation,
                    metadata={"migrated_from": src_island, "migrated_to": dest_island},
                )

                self.add(migrant, parent_id=None, target_island=dest_island)

            if top_programs:
                logger.debug(
                    f"Migrated {len(top_programs)} programs from island {src_island} "
                    f"to island {dest_island}"
                )

    def _migrate_legacy(self) -> None:
        """Legacy migration: copy single best program to next island."""
        migrants: List[Tuple[int, Program]] = []

        for i in range(self.num_islands):
            if self.islands[i]:
                best = max(self.islands[i], key=self._get_fitness)
                migrants.append((i, best))

        for src_island, program in migrants:
            dest_island = (src_island + 1) % self.num_islands

            if self._has_duplicate_solution(dest_island, program.solution):
                continue

            migrant = Program(
                id=str(uuid.uuid4()),
                solution=program.solution,
                language=program.language,
                metrics=program.metrics.copy() if program.metrics else {},
                iteration_found=program.iteration_found,
                parent_id=program.id,
                generation=program.generation,
                metadata={"migrated_from": src_island, "migrated_to": dest_island},
            )

            self.add(migrant, parent_id=None, target_island=dest_island)

    def _has_duplicate_solution(self, island_idx: int, solution: str) -> bool:
        """Check if island already has a program with identical solution."""
        if self.use_unified_archive and self.archives:
            return any(p.solution == solution for p in self.archives[island_idx].get_all())
        else:
            return any(p.solution == solution for p in self.islands[island_idx])

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for logging/debugging."""
        adapter_stats = self.adapter.get_stats()

        island_stats = []
        for i in range(self.num_islands):
            dim_stats = (
                adapter_stats["dimensions"][i] if i < len(adapter_stats["dimensions"]) else {}
            )

            if self.use_unified_archive and self.archives:
                archive = self.archives[i]
                island_stats.append(
                    {
                        "island": i,
                        "population_size": archive.size(),
                        "top_count": len(archive.get_top_programs()),
                        "is_current": i == self.current_island,
                        **dim_stats,
                    }
                )
            else:
                island_stats.append(
                    {
                        "island": i,
                        "population_size": len(self.islands[i]),
                        "top_count": 0,
                        "is_current": i == self.current_island,
                        **dim_stats,
                    }
                )

        return {
            "num_islands": self.num_islands,
            "current_island": self.current_island,
            "global_best_score": self._global_best_score,
            "global_productivity": adapter_stats["global_productivity"],
            "iteration": self._iteration_count,
            "use_unified_archive": self.use_unified_archive,
            "use_adaptive_search": self.use_adaptive_search,
            "use_ucb_selection": self.use_ucb_selection,
            "islands": island_stats,
        }

    def get_comprehensive_iteration_stats(
        self,
        iteration: int,
        sampling_mode: Optional[str] = None,
        sampling_intensity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics for JSON logging at each iteration.

        This method collects ALL AdaEvolve signals for detailed analysis including:
        - Island-level adaptive state (G, intensity, UCB stats)
        - Global evolution state
        - Paradigm breakthrough state
        - Dynamic island spawning state

        Args:
            iteration: Current iteration number
            sampling_mode: The sampling mode used this iteration (exploration/exploitation/balanced)
            sampling_intensity: The search intensity value used this iteration

        Returns:
            Comprehensive dictionary with all AdaEvolve signals
        """
        import math

        # =========================================================================
        # Island-level statistics
        # =========================================================================
        island_stats = []
        for i in range(self.num_islands):
            state = self.adapter.states[i] if i < len(self.adapter.states) else None

            island_data = {
                "island_idx": i,
                "is_current": i == self.current_island,
                "config_name": (
                    self.island_config_names[i] if i < len(self.island_config_names) else "unknown"
                ),
            }

            # Population stats
            if self.use_unified_archive and self.archives and i < len(self.archives):
                archive = self.archives[i]
                island_data["population_size"] = archive.size()
                island_data["top_count"] = len(archive.get_top_programs())
                if hasattr(archive, "stats"):
                    archive_stats = archive.stats()
                    island_data["archive_stats"] = archive_stats
            elif self.islands and i < len(self.islands):
                island_data["population_size"] = len(self.islands[i])
                island_data["top_count"] = 0

            # Adaptive state (G, intensity, etc.)
            if state:
                island_data["accumulated_signal_G"] = state.accumulated_signal
                island_data["best_score"] = (
                    state.best_score if not math.isinf(state.best_score) else None
                )
                island_data["search_intensity"] = state.get_search_intensity()
                island_data["improvement_count"] = state.improvement_count
                island_data["total_evaluations"] = state.total_evaluations
                island_data["productivity"] = state.get_productivity()

                # Hyperparameters
                island_data["decay"] = state.decay
                island_data["intensity_min"] = state.intensity_min
                island_data["intensity_max"] = state.intensity_max

            # UCB stats
            if i < len(self.adapter.dimension_visits):
                island_data["ucb_raw_visits"] = self.adapter.dimension_visits[i]
            if i < len(self.adapter.decayed_visits):
                island_data["ucb_decayed_visits"] = self.adapter.decayed_visits[i]
            if i < len(self.adapter.dimension_rewards):
                island_data["ucb_decayed_rewards"] = self.adapter.dimension_rewards[i]
                dec_visits = (
                    self.adapter.decayed_visits[i] if i < len(self.adapter.decayed_visits) else 0.0
                )
                island_data["ucb_reward_avg"] = (
                    self.adapter.dimension_rewards[i] / dec_visits if dec_visits > 0 else 0.0
                )

            island_stats.append(island_data)

        # =========================================================================
        # Global statistics
        # =========================================================================
        best_program = self.get_best_program()
        pareto_front = self.get_global_pareto_front() if self.is_multiobjective_enabled() else []
        global_stats = {
            "iteration": iteration,
            "num_islands": self.num_islands,
            "current_island_idx": self.current_island,
            "global_best_score": (
                self._global_best_score if not math.isinf(self._global_best_score) else None
            ),
            "global_best_program_id": self.best_program_id,
            "optimization_mode": "pareto" if self.is_multiobjective_enabled() else "scalar",
            "pareto_objectives": list(self.pareto_objectives),
            "higher_is_better": dict(self.higher_is_better),
            "fitness_proxy_key": self.fitness_key,
            "global_pareto_front_size": len(pareto_front),
            "global_pareto_front_ids": [program.id for program in pareto_front],
            "global_productivity": self.adapter.get_global_productivity(),
            "total_programs": len(self.programs),
            # UCB global state
            "ucb_global_best_score": (
                self.adapter.global_best_score
                if not math.isinf(self.adapter.global_best_score)
                else None
            ),
            "ucb_exploration_constant": self.adapter.ucb_exploration,
            "ucb_min_visits": self.adapter.min_visits,
        }

        # Best program details (truncated code for logging)
        if best_program:
            code_preview = (
                best_program.solution[:500] + "..."
                if len(best_program.solution) > 500
                else best_program.solution
            )
            global_stats["best_program"] = {
                "id": best_program.id,
                "metrics": best_program.metrics,
                "generation": best_program.generation,
                "iteration_found": best_program.iteration_found,
                "is_pareto_representative": self.is_multiobjective_enabled(),
                "code_length": len(best_program.solution),
                "code_preview": code_preview,
            }

        # =========================================================================
        # Sampling state (for this iteration)
        # =========================================================================
        sampling_stats = {
            "mode": sampling_mode,
            "intensity_used": sampling_intensity,
            "use_adaptive_search": self.use_adaptive_search,
            "use_ucb_selection": self.use_ucb_selection,
            "fixed_intensity": self.fixed_intensity if not self.use_adaptive_search else None,
        }

        # =========================================================================
        # Paradigm breakthrough state
        # =========================================================================
        paradigm_stats = {
            "enabled": self.use_paradigm_breakthrough,
        }

        if self.use_paradigm_breakthrough and self.paradigm_tracker is not None:
            tracker = self.paradigm_tracker

            paradigm_stats.update(
                {
                    "is_stagnating": tracker.is_paradigm_stagnating(),
                    "has_active_paradigm": tracker.has_active_paradigm(),
                    "improvement_rate": tracker.get_improvement_rate(),
                    "improvement_threshold": tracker.improvement_threshold,
                    "window_size": tracker.window_size,
                    "improvement_history_length": len(tracker.improvement_history),
                    # Active paradigms
                    "num_active_paradigms": len(tracker.active_paradigms),
                    "current_paradigm_index": tracker.current_paradigm_index,
                    "max_paradigm_uses": tracker.max_paradigm_uses,
                    # Count non-exhausted paradigms
                    "num_non_exhausted_paradigms": sum(
                        1
                        for i in range(len(tracker.active_paradigms))
                        if tracker.paradigm_usage_counts.get(i, 0) < tracker.max_paradigm_uses
                    ),
                    # Paradigm usage counts
                    "paradigm_usage_counts": dict(tracker.paradigm_usage_counts),
                    # Current paradigm details
                    "current_paradigm": None,
                    # Previously tried paradigms
                    "num_tried_paradigms": len(tracker.tried_paradigms),
                    "tried_paradigms_summary": [
                        {
                            "idea": p.get("idea", "N/A"),
                            "outcome": p.get("outcome", "UNCLEAR"),
                            "score_improvement": p.get("score_improvement", 0.0),
                            "uses": p.get("uses", 0),
                        }
                        for p in tracker.tried_paradigms[-5:]  # Last 5 tried
                    ],
                    # Score tracking
                    "best_score_at_paradigm_gen": tracker.best_score_at_paradigm_gen,
                    "best_score_during_paradigm": tracker.best_score_during_paradigm,
                }
            )

            # Current paradigm details (if available)
            current = tracker.get_current_paradigm()
            if current:
                paradigm_stats["current_paradigm"] = {
                    "idea": current.get("idea", "N/A"),
                    "description": current.get("description", "N/A"),
                    "approach_type": current.get("approach_type", "N/A"),
                    "what_to_optimize": current.get("what_to_optimize", "N/A"),
                    "cautions": current.get("cautions", "N/A"),
                    "uses_remaining": (
                        tracker.max_paradigm_uses
                        - tracker.paradigm_usage_counts.get(tracker.current_paradigm_index, 0)
                    ),
                }

            # All active paradigms summary
            paradigm_stats["active_paradigms"] = [
                {
                    "index": i,
                    "idea": p.get("idea", "N/A"),
                    "approach_type": p.get("approach_type", "N/A"),
                    "uses": tracker.paradigm_usage_counts.get(i, 0),
                    "exhausted": tracker.paradigm_usage_counts.get(i, 0)
                    >= tracker.max_paradigm_uses,
                }
                for i, p in enumerate(tracker.active_paradigms)
            ]

        # =========================================================================
        # Dynamic island spawning state
        # =========================================================================
        dynamic_island_stats = {
            "enabled": self.use_dynamic_islands,
        }

        if self.use_dynamic_islands:
            dynamic_island_stats.update(
                {
                    "max_islands": self.max_islands,
                    "current_num_islands": self.num_islands,
                    "islands_remaining": self.max_islands - self.num_islands,
                    "last_spawn_iteration": self.last_spawn_iteration,
                    "spawn_cooldown": self.spawn_cooldown,
                    "iterations_since_spawn": iteration - self.last_spawn_iteration,
                    "spawn_productivity_threshold": self.spawn_productivity_threshold,
                    "would_spawn": self._should_spawn_island(),
                }
            )

        # =========================================================================
        # Configuration summary
        # =========================================================================
        config_stats = {
            "decay": self.decay,
            "intensity_min": self.intensity_min,
            "intensity_max": self.intensity_max,
            "population_size": self.population_size,
            "migration_interval": self.migration_interval,
            "migration_count": self.migration_count,
            "use_migration": self.use_migration,
            "use_unified_archive": self.use_unified_archive,
            "local_context_program_ratio": self.local_context_program_ratio,
        }

        # =========================================================================
        # Assemble complete stats
        # =========================================================================
        return {
            "iteration": iteration,
            "timestamp": None,  # Will be filled by controller
            "global": global_stats,
            "islands": island_stats,
            "sampling": sampling_stats,
            "paradigm": paradigm_stats,
            "dynamic_islands": dynamic_island_stats,
            "config": config_stats,
        }

    # =========================================================================
    # Save and Load (Override base class)
    # =========================================================================

    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:
        """
        Save database with AdaEvolve-specific state.

        This properly saves:
        1. All programs (via base class)
        2. Island membership (which programs in which island)
        3. Archive genealogy state (parent-child tracking)
        4. Adaptive state (UCB rewards, accumulated signals)
        5. Paradigm tracker state
        """
        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return

        # Sync programs dict from archives/islands
        # CRITICAL: Preserve best program before rebuilding programs dict
        best_id = self.best_program_id
        best_program = self.programs.get(best_id) if best_id else None

        self.programs = {}
        if self.use_unified_archive and self.archives:
            for archive in self.archives:
                for p in archive.get_all():
                    self.programs[p.id] = p
        else:
            for island in self.islands:
                for p in island:
                    self.programs[p.id] = p

        # Restore best program if it was evicted (safety net)
        if best_program and best_id not in self.programs:
            self.programs[best_id] = best_program
            # Re-add to first archive to ensure it survives future save cycles
            if self.use_unified_archive and self.archives:
                self.archives[0].add(best_program)
            logger.warning(f"Restored evicted best program {best_id[:8]} during save")

        # Save base state (programs, prompts, artifacts)
        super().save(save_path, iteration)

        # Build AdaEvolve metadata
        metadata = {
            "num_islands": self.num_islands,
            "current_island": self.current_island,
            "iteration_count": self._iteration_count,
            "global_best_score": self._global_best_score,
            "decay": self.decay,
            "intensity_min": self.intensity_min,
            "intensity_max": self.intensity_max,
            "migration_interval": self.migration_interval,
            "diversity_strategy_type": self._diversity_strategy_type,
            "use_unified_archive": self.use_unified_archive,
            # Ablation flags
            "use_adaptive_search": self.use_adaptive_search,
            "use_ucb_selection": self.use_ucb_selection,
            "fixed_intensity": self.fixed_intensity,
            # Adapter state (UCB rewards, accumulated signals, etc.)
            "adapter": self.adapter.to_dict(),
            # Island config names for dynamic spawning
            "island_config_names": self.island_config_names,
        }

        # Island membership and genealogy depend on mode
        if self.use_unified_archive and self.archives:
            metadata["islands"] = [[p.id for p in archive.get_all()] for archive in self.archives]
            metadata["archive_genealogies"] = [
                archive.get_genealogy_state() for archive in self.archives
            ]
        else:
            metadata["islands"] = [[p.id for p in island] for island in self.islands]
            metadata["children_map"] = self.children_map

        # Save dynamic island state if enabled
        if self.use_dynamic_islands:
            metadata["use_dynamic_islands"] = True
            metadata["max_islands"] = self.max_islands
            metadata["last_spawn_iteration"] = self.last_spawn_iteration

        # Save paradigm tracker state if enabled
        if self.use_paradigm_breakthrough and self.paradigm_tracker is not None:
            metadata["use_paradigm_breakthrough"] = True
            metadata["paradigm_tracker"] = self.paradigm_tracker.to_dict()

        os.makedirs(save_path, exist_ok=True)
        metadata_path = os.path.join(save_path, "adaevolve_metadata.json")
        with open(metadata_path, "w") as f:
            from skydiscover.search.utils.checkpoint_manager import SafeJSONEncoder

            json.dump(metadata, f, indent=2, cls=SafeJSONEncoder)

        logger.info(f"Saved AdaEvolve state to {save_path}")

    def load(self, path: str) -> None:
        """
        Load database with AdaEvolve-specific state.

        Restores:
        1. All programs (via base class)
        2. Island membership (programs to correct archives/islands)
        3. Archive genealogy state (or children_map for legacy)
        4. Adaptive state (UCB rewards, accumulated signals)
        5. Paradigm tracker state
        """
        # Load base state (programs dict, best_program_id, last_iteration)
        super().load(path)

        # Load AdaEvolve metadata
        metadata_path = os.path.join(path, "adaevolve_metadata.json")
        if not os.path.exists(metadata_path):
            logger.warning(
                f"No AdaEvolve metadata found at {path}, distributing programs to islands"
            )
            self._distribute_programs_to_islands()
            return

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Restore scalar state
        saved_num_islands = metadata.get("num_islands", self.num_islands)
        self.current_island = metadata.get("current_island", 0)
        self._iteration_count = metadata.get("iteration_count", 0)
        self._global_best_score = metadata.get("global_best_score", float("-inf"))
        self._diversity_strategy_type = metadata.get("diversity_strategy_type", "code")

        # NOTE: Ablation flags are NOT restored from checkpoint.
        # The current config's ablation settings take precedence.
        # This allows running ablation experiments from existing checkpoints.
        # (e.g., load a baseline checkpoint and run no_adaptive_search ablation)
        #
        # The adaptive STATE (G, UCB rewards, visits) IS restored from checkpoint,
        # only the FLAGS are kept from current config.

        # Handle dynamic island count - may need to expand
        if saved_num_islands > self.num_islands:
            logger.info(
                f"Checkpoint has {saved_num_islands} islands, " f"expanding from {self.num_islands}"
            )
            self._expand_to_island_count(saved_num_islands, metadata)

        self.num_islands = saved_num_islands

        # Load adapter state
        if "adapter" in metadata:
            self.adapter = MultiDimensionalAdapter.from_dict(metadata["adapter"])

        # Restore island config names
        self.island_config_names = metadata.get(
            "island_config_names", ["balanced"] * self.num_islands
        )

        # Restore dynamic island state
        if metadata.get("use_dynamic_islands", False):
            self.use_dynamic_islands = True
            self.max_islands = metadata.get("max_islands", self.max_islands)
            self.last_spawn_iteration = metadata.get("last_spawn_iteration", 0)

        # Restore paradigm tracker state IF current config has it enabled
        # We respect the current config's flag, not the checkpoint's flag
        # This allows ablation: load checkpoint with paradigm, run without it
        if self.use_paradigm_breakthrough and "paradigm_tracker" in metadata:
            # Current config wants paradigm - restore state from checkpoint
            self.paradigm_tracker = ParadigmTracker.from_dict(metadata["paradigm_tracker"])

        # Restore island membership based on mode
        island_ids = metadata.get("islands", [])

        if self.use_unified_archive:
            # Reinitialize archives to ensure clean state before restoring
            self.archives = []
            self._init_archives(self.config)
            genealogies = metadata.get("archive_genealogies", [])

            for island_idx, program_ids in enumerate(island_ids):
                if island_idx >= len(self.archives):
                    break

                archive = self.archives[island_idx]

                # Restore genealogy state first (for parent-child tracking)
                if island_idx < len(genealogies):
                    archive.set_genealogy_state(genealogies[island_idx])

                # Add programs to archive
                for pid in program_ids:
                    if pid in self.programs:
                        archive.add(self.programs[pid])
        else:
            # Legacy mode: restore to island lists
            self.islands = [[] for _ in range(self.num_islands)]
            self.children_map = metadata.get("children_map", [{} for _ in range(self.num_islands)])

            for island_idx, program_ids in enumerate(island_ids):
                if island_idx >= self.num_islands:
                    break

                for pid in program_ids:
                    if pid in self.programs:
                        self.islands[island_idx].append(self.programs[pid])

        self._invalidate_global_pareto_cache()
        logger.info(
            f"Loaded AdaEvolve state from {path}: "
            f"{self.num_islands} islands, {len(self.programs)} programs, "
            f"unified_archive={self.use_unified_archive}"
        )

    def _distribute_programs_to_islands(self) -> None:
        """
        Distribute programs to islands when no island membership info is available.

        Used as fallback when loading from a checkpoint without AdaEvolve metadata.
        """
        programs_list = list(self.programs.values())
        if not programs_list:
            return

        # Sort by fitness (best first)
        programs_list.sort(key=lambda p: self._get_fitness(p), reverse=True)

        # Distribute round-robin to islands
        for i, program in enumerate(programs_list):
            island_idx = i % self.num_islands
            if self.use_unified_archive and self.archives:
                if island_idx < len(self.archives):
                    self.archives[island_idx].add(program)
            else:
                if island_idx < len(self.islands):
                    self.islands[island_idx].append(program)

        self._invalidate_global_pareto_cache()
        logger.info(f"Distributed {len(programs_list)} programs across {self.num_islands} islands")

    def _expand_to_island_count(self, target_count: int, metadata: Dict[str, Any]) -> None:
        """
        Expand archives/islands to accommodate more islands from checkpoint.

        Args:
            target_count: Target number of islands
            metadata: Checkpoint metadata for config restoration
        """
        # Legacy mode: just expand island lists
        if not self.use_unified_archive:
            while len(self.islands) < target_count:
                self.islands.append([])
                self.children_map.append({})
                self.island_config_names.append("balanced")
                # Add adaptive state dimension
                state = AdaptiveState(
                    decay=self.decay,
                    intensity_min=self.intensity_min,
                    intensity_max=self.intensity_max,
                )
                self.adapter.add_dimension(state)
            return

        higher_is_better = getattr(self.config, "higher_is_better", {})
        saved_config_names = metadata.get("island_config_names", [])

        while len(self.archives) < target_count:
            new_idx = len(self.archives)

            # Get config name from saved state or default to "balanced"
            config_name = (
                saved_config_names[new_idx] if new_idx < len(saved_config_names) else "balanced"
            )
            preset = get_island_config_preset(config_name)

            archive_config = ArchiveConfig(
                max_size=self.population_size,
                k_neighbors=getattr(self.config, "k_neighbors", 5),
                elite_ratio=preset["elite_ratio"],
                pareto_weight=preset["pareto_weight"],
                fitness_weight=preset["fitness_weight"],
                novelty_weight=preset["novelty_weight"],
                higher_is_better=higher_is_better,
            )

            # Create fresh diversity strategy
            diversity_strategy = create_diversity_strategy(
                self._diversity_strategy_type,
                higher_is_better=higher_is_better,
            )

            new_archive = UnifiedArchive(
                config=archive_config,
                diversity_strategy=diversity_strategy,
            )
            self.archives.append(new_archive)
            self.island_config_names.append(config_name)

            # Add adaptive state dimension
            state = AdaptiveState(
                decay=self.decay,
                intensity_min=self.intensity_min,
                intensity_max=self.intensity_max,
            )
            self.adapter.add_dimension(state)

    # =========================================================================
    # Helpers
    # =========================================================================

    def is_multiobjective_enabled(self) -> bool:
        """Return True when explicit Pareto objectives are configured."""
        return bool(self.pareto_objectives)

    def _metric_to_maximization_value(self, metric_name: str, value: Any) -> Optional[float]:
        """Convert a metric to an internal score where larger is always better."""
        from skydiscover.utils.metrics import normalize_metric_value

        return normalize_metric_value(metric_name, value, self.higher_is_better)

    def _get_multiobjective_proxy_score(self, program: Program) -> float:
        """Return a scalar proxy for adaptive state and deterministic tie-breaking."""
        metrics = getattr(program, "metrics", None) or {}
        return compute_proxy_score(
            metrics,
            fitness_key=self.fitness_key,
            pareto_objectives=self.pareto_objectives if self.is_multiobjective_enabled() else None,
            higher_is_better=self.higher_is_better,
        )

    def get_program_proxy_score(self, program: Optional[Program]) -> float:
        """Public wrapper for the scalar proxy used by AdaEvolve internals."""
        if program is None:
            return float("-inf")
        return self._get_multiobjective_proxy_score(program)

    def _all_population_programs(self) -> List[Program]:
        """Return all currently active programs across islands."""
        if self.use_unified_archive and self.archives:
            programs = []
            for archive in self.archives:
                programs.extend(archive.get_all())
            return programs
        if self.islands:
            programs = []
            for island in self.islands:
                programs.extend(island)
            return programs
        return list(self.programs.values())

    def _get_objective_vector(self, program: Program) -> Optional[List[float]]:
        """Return the configured objective vector for a program.

        Missing or non-numeric objectives are filled with ``-inf`` so that
        programs with incomplete metrics cannot accidentally dominate
        fully-evaluated programs (all objectives are in "higher is better"
        space after normalisation).
        """
        if not self.is_multiobjective_enabled():
            return None

        metrics = getattr(program, "metrics", None) or {}
        vector: List[float] = []
        for objective in self.pareto_objectives:
            normalized = self._metric_to_maximization_value(objective, metrics.get(objective))
            vector.append(normalized if normalized is not None else float("-inf"))
        return vector

    @staticmethod
    def _dominates(vec_a: List[float], vec_b: List[float]) -> bool:
        """True if vec_a Pareto-dominates vec_b (same-length vectors required)."""
        if len(vec_a) != len(vec_b):
            raise ValueError(
                f"Objective vectors must have equal length, got {len(vec_a)} vs {len(vec_b)}"
            )
        at_least_one_better = False
        for a, b in zip(vec_a, vec_b):
            if a < b:
                return False
            if a > b:
                at_least_one_better = True
        return at_least_one_better

    def _get_archive_crowding_distance(self, program: Program) -> float:
        """Return archive crowding distance when available."""
        if not (self.use_unified_archive and self.archives):
            return 0.0

        for archive in self.archives:
            if archive.contains(program.id):
                archive._ensure_cache_valid()
                return archive._crowding_distances.get(program.id, 0.0)
        return 0.0

    def _get_archive_elite_score(self, program: Program) -> float:
        """Return cached archive elite score when available."""
        if not (self.use_unified_archive and self.archives):
            return 0.0

        for archive in self.archives:
            if archive.contains(program.id):
                archive._ensure_cache_valid()
                return archive._elite_scores.get(program.id, 0.0)
        return 0.0

    def _get_pareto_representative_sort_key(
        self, program: Program
    ) -> Tuple[float, float, float, int, str]:
        """Sort key for choosing one stable representative from a Pareto front.

        Higher values win (used with ``max``).  Ties are broken by:
        proxy score → crowding distance → elite score → newer iteration → ID.
        """
        return (
            self._get_multiobjective_proxy_score(program),
            self._get_archive_crowding_distance(program),
            self._get_archive_elite_score(program),
            getattr(program, "iteration_found", 0),  # newer wins ties
            program.id,
        )

    def _choose_pareto_representative(self, front: List[Program]) -> Optional[Program]:
        """Choose a deterministic representative program from a Pareto front."""
        if not front:
            return None
        return max(front, key=self._get_pareto_representative_sort_key)

    def _invalidate_global_pareto_cache(self) -> None:
        """Mark the cached global Pareto front as stale.

        The *stale* cache is intentionally preserved (not cleared) so that
        ``_update_best_program`` can read the pre-mutation front and detect
        whether a newly added program entered the front.
        """
        self._global_pareto_cache_valid = False

    def _compute_global_pareto_front(self) -> List[Program]:
        """O(n²) computation of the non-dominated front across all islands."""
        programs = self._all_population_programs()
        if not programs:
            return []

        objective_vectors = {
            program.id: self._get_objective_vector(program) or [] for program in programs
        }
        front = []
        for candidate in programs:
            vec_candidate = objective_vectors[candidate.id]
            dominated = False
            for challenger in programs:
                if challenger.id == candidate.id:
                    continue
                if self._dominates(objective_vectors[challenger.id], vec_candidate):
                    dominated = True
                    break
            if not dominated:
                front.append(candidate)

        return sorted(front, key=self._get_pareto_representative_sort_key, reverse=True)

    def get_global_pareto_front(self) -> List[Program]:
        """Return the non-dominated Pareto front across all islands (cached)."""
        if not self.is_multiobjective_enabled():
            return []

        if not self._global_pareto_cache_valid:
            self._global_pareto_cache = self._compute_global_pareto_front()
            self._global_pareto_cache_valid = True

        return list(self._global_pareto_cache or [])

    def _get_fitness(self, program: Program) -> float:
        """Get scalar fitness score used by adaptive state and fallbacks."""
        return self._get_multiobjective_proxy_score(program)

    def _update_best_program(self, program: Program) -> bool:
        """
        Update global best program tracking.

        Returns:
            True if this program is a new global best, False otherwise
        """
        if self.is_multiobjective_enabled():
            previous_best_id = self.best_program_id
            previous_best_score = self._global_best_score

            # Read the STALE cache (snapshot of the front before this program
            # was added).  The cache was invalidated by add() but the old list
            # is intentionally preserved for exactly this comparison.
            previous_front_ids: Set[str] = (
                {p.id for p in (self._global_pareto_cache or [])}
                if not self._global_pareto_cache_valid
                else set()
            )

            # Now recompute (cache is invalid, so this triggers O(n²) rebuild).
            front = self.get_global_pareto_front()
            representative = self._choose_pareto_representative(front)
            if representative is None:
                return False

            self.best_program_id = representative.id
            self._global_best_score = self._get_fitness(representative)

            front_ids = {p.id for p in front}
            entered_front = program.id in front_ids and program.id not in previous_front_ids
            representative_changed = representative.id != previous_best_id
            score_improved = self._global_best_score > previous_best_score
            return entered_front or representative_changed or score_improved

        fitness = self._get_fitness(program)
        if fitness > self._global_best_score:
            self._global_best_score = fitness
            self.best_program_id = program.id
            logger.debug(f"New global best: {program.id[:8]} with fitness {fitness:.6f}")
            return True
        return False

    def get_children(self, parent_id: str, limit: int = 5) -> List[Program]:
        """
        Get recent children of a parent on the current island.

        Used by controller for sibling context - shows what mutations
        have been tried on this parent before.

        Args:
            parent_id: ID of the parent program
            limit: Maximum number of children to return

        Returns:
            List of child programs (most recent last)
        """
        if self.use_unified_archive and self.archives:
            archive = self.archives[self.current_island]

            # Use archive's genealogy tracking if available
            if hasattr(archive, "get_children"):
                children = archive.get_children(parent_id)
                return children[-limit:]

            # Fallback: scan all programs (less efficient)
            children = [p for p in archive.get_all() if getattr(p, "parent_id", None) == parent_id]
        else:
            # Legacy mode: use children_map
            child_ids = self.children_map[self.current_island].get(parent_id, [])
            children = [self.programs[cid] for cid in child_ids if cid in self.programs]

        # Sort by iteration_found to get most recent
        children.sort(key=lambda p: getattr(p, "iteration_found", 0))
        return children[-limit:]

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_best_program(self, metric: Optional[str] = None) -> Optional[Program]:
        """
        Get the best program across all islands.

        Uses tracked best_program_id as authoritative source, falling back to
        archive/island search. This prevents silent data loss when the best program
        has been evicted from archives but is still tracked.
        """
        if metric is None and self.is_multiobjective_enabled():
            front = self.get_global_pareto_front()
            representative = self._choose_pareto_representative(front)
            if representative is not None:
                self.best_program_id = representative.id
                self._global_best_score = self._get_fitness(representative)
            return representative

        # First, check if we have a tracked best program (authoritative)
        # This handles the case where best program was evicted from archives
        if self.best_program_id and self.best_program_id in self.programs:
            tracked_best = self.programs[self.best_program_id]
            tracked_fitness = self._get_fitness(tracked_best)

            # Verify it's still actually the best by checking archives/islands
            population_best = None
            population_best_fitness = float("-inf")

            if self.use_unified_archive and self.archives:
                for archive in self.archives:
                    if hasattr(archive, "get_best"):
                        candidate = archive.get_best()
                    else:
                        all_progs = archive.get_all()
                        candidate = max(all_progs, key=self._get_fitness) if all_progs else None

                    if candidate:
                        fitness = self._get_fitness(candidate)
                        if fitness > population_best_fitness:
                            population_best_fitness = fitness
                            population_best = candidate
            else:
                for island in self.islands:
                    if island:
                        candidate = max(island, key=self._get_fitness)
                        fitness = self._get_fitness(candidate)
                        if fitness > population_best_fitness:
                            population_best_fitness = fitness
                            population_best = candidate

            # Return the better of tracked vs population best
            if tracked_fitness >= population_best_fitness:
                return tracked_best
            else:
                # Population has a better program - update tracking
                self.best_program_id = population_best.id
                self._global_best_score = population_best_fitness
                return population_best

        # Fallback: search archives/islands (for cases where tracking is not set)
        best = None
        best_fitness = float("-inf")

        if self.use_unified_archive and self.archives:
            for archive in self.archives:
                if hasattr(archive, "get_best"):
                    candidate = archive.get_best()
                else:
                    all_progs = archive.get_all()
                    candidate = max(all_progs, key=self._get_fitness) if all_progs else None

                if candidate:
                    fitness = self._get_fitness(candidate)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best = candidate
        else:
            for island in self.islands:
                if island:
                    candidate = max(island, key=self._get_fitness)
                    fitness = self._get_fitness(candidate)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best = candidate

        return best

    def get_top_programs(self, n: int = 10, metric: Optional[str] = None) -> List[Program]:
        """Get top n programs across all islands.

        When *metric* is provided, programs are sorted by that specific metric
        (respecting ``higher_is_better`` if configured).  Otherwise, multiobjective
        mode returns the non-dominated front padded with proxy-score-ranked
        programs, and scalar mode sorts by the default proxy fitness.
        """
        all_programs = self._all_population_programs()

        if metric:
            # Sort by the requested metric, applying direction normalisation.
            def _metric_key(p: Program) -> float:
                val = (getattr(p, "metrics", None) or {}).get(metric)
                normalized = self._metric_to_maximization_value(metric, val)
                return normalized if normalized is not None else float("-inf")

            sorted_programs = sorted(all_programs, key=_metric_key, reverse=True)
            return sorted_programs[:n]

        if not self.is_multiobjective_enabled():
            sorted_programs = sorted(all_programs, key=self._get_fitness, reverse=True)
            return sorted_programs[:n]

        pareto_front = self.get_global_pareto_front()
        if len(pareto_front) >= n:
            return pareto_front[:n]

        front_ids = {program.id for program in pareto_front}
        remaining = sorted(
            [program for program in all_programs if program.id not in front_ids],
            key=self._get_fitness,
            reverse=True,
        )
        return pareto_front + remaining[: max(0, n - len(pareto_front))]

    def get_top_programs_for_island(self, island_idx: Optional[int] = None) -> List[Program]:
        """Get top programs for an island (current island if not specified)."""
        idx = island_idx if island_idx is not None else self.current_island
        if 0 <= idx < self.num_islands:
            if self.use_unified_archive and self.archives:
                return self.archives[idx].get_top_programs()
            else:
                # Legacy mode: return top 25% programs
                population = self.islands[idx]
                if not population:
                    return []
                sorted_pop = sorted(population, key=self._get_fitness, reverse=True)
                return sorted_pop[: max(1, len(sorted_pop) // 4)]
        return []

    def get_pareto_front(self, island_idx: Optional[int] = None) -> List[Program]:
        """Get the Pareto front for a specific island or globally across all islands."""
        if not self.is_multiobjective_enabled():
            return self.get_top_programs_for_island(island_idx)

        if island_idx is None:
            return self.get_global_pareto_front()

        if 0 <= island_idx < self.num_islands:
            if self.use_unified_archive and self.archives:
                return self.archives[island_idx].get_pareto_front()

            population = self.get_island_population(island_idx)
            if not population:
                return []

            front = []
            objective_vectors = {
                program.id: self._get_objective_vector(program) or [] for program in population
            }
            for candidate in population:
                dominated = False
                for challenger in population:
                    if challenger.id == candidate.id:
                        continue
                    if self._dominates(
                        objective_vectors[challenger.id], objective_vectors[candidate.id]
                    ):
                        dominated = True
                        break
                if not dominated:
                    front.append(candidate)
            return sorted(front, key=self._get_pareto_representative_sort_key, reverse=True)

        return []

    def get_archive_stats(self, island_idx: Optional[int] = None) -> Dict[str, Any]:
        """Get archive statistics for an island."""
        idx = island_idx if island_idx is not None else self.current_island
        if 0 <= idx < self.num_islands:
            if self.use_unified_archive and self.archives and hasattr(self.archives[idx], "stats"):
                return self.archives[idx].stats()
        top_count = len(self.get_top_programs_for_island(idx))
        return {
            "size": self.get_island_size(idx),
            "max_size": self.population_size,
            "top_count": top_count,
            "pareto_count": top_count,  # Backwards compatibility
        }

    # =========================================================================
    # Program Merging
    # =========================================================================

    def find_merge_candidates(
        self, island_idx: Optional[int] = None
    ) -> Optional[Tuple[Program, Program, Program]]:
        """Find merge candidates on an island."""
        idx = island_idx if island_idx is not None else self.current_island
        if 0 <= idx < self.num_islands:
            if (
                self.use_unified_archive
                and self.archives
                and hasattr(self.archives[idx], "find_merge_candidates")
            ):
                return self.archives[idx].find_merge_candidates()
        # Legacy mode doesn't support merging
        return None

    def add_merged_program(
        self,
        program: Program,
        parent_ids: List[str],
        iteration: Optional[int] = None,
        island_idx: Optional[int] = None,
    ) -> str:
        """Add a merged program to an island."""
        idx = island_idx if island_idx is not None else self.current_island

        if idx < 0 or idx >= self.num_islands:
            raise ValueError(f"Invalid island index {idx}")

        if iteration is not None:
            program.iteration_found = iteration
            self.last_iteration = max(self.last_iteration, iteration)

        was_added = False
        if self.use_unified_archive and self.archives:
            if hasattr(self.archives[idx], "add_merged_program"):
                was_added = self.archives[idx].add_merged_program(program, parent_ids)
            else:
                was_added = self.archives[idx].add(program)
        else:
            # Legacy mode: just add to island list
            self.islands[idx].append(program)
            was_added = True
            self._enforce_island_population_limit(idx)

        if was_added:
            self.programs[program.id] = program
            fitness = self._get_fitness(program)
            self.adapter.record_evaluation(idx, fitness)
            self._invalidate_global_pareto_cache()
            self._update_best_program(program)

            if self.config.db_path:
                self._save_program(program)

            logger.debug(f"Added merged program {program.id[:8]} to island {idx}")

        return program.id

    # =========================================================================
    # Dynamic Island Spawning
    # =========================================================================

    def _should_spawn_island(self) -> bool:
        """
        Check if we should spawn a new island.

        Triggers spawning when:
        1. Dynamic islands is enabled
        2. Using unified archives (legacy mode doesn't support spawning)
        3. Haven't reached max_islands limit
        4. Cooldown period has passed since last spawn
        5. Global productivity is below threshold (all islands struggling)
        """
        if not self.use_dynamic_islands:
            return False

        # Dynamic spawning only works with unified archives
        if not self.use_unified_archive:
            return False

        if not self.programs:
            return False

        if self.num_islands >= self.max_islands:
            return False

        iterations_since_spawn = self._iteration_count - self.last_spawn_iteration
        if iterations_since_spawn < self.spawn_cooldown:
            return False

        # Check global productivity from adapter
        global_productivity = self.adapter.get_global_productivity()
        if global_productivity >= self.spawn_productivity_threshold:
            return False

        logger.info(
            f"Spawn conditions met: global_productivity={global_productivity:.3f} "
            f"< threshold={self.spawn_productivity_threshold}, "
            f"islands={self.num_islands}/{self.max_islands}"
        )
        return True

    def _spawn_island(self) -> int:
        """
        Spawn a new island and initialize it with top programs.

        Returns:
            Index of the newly created island
        """
        new_island_idx = self.num_islands

        # Select config for new island
        config_name, preset = self._select_spawn_config()

        # Create new archive with the selected preset
        higher_is_better = getattr(self.config, "higher_is_better", {})
        archive_config = ArchiveConfig(
            max_size=self.population_size,
            k_neighbors=getattr(self.config, "k_neighbors", 5),
            elite_ratio=preset["elite_ratio"],
            pareto_weight=preset["pareto_weight"],
            fitness_weight=preset["fitness_weight"],
            novelty_weight=preset["novelty_weight"],
            higher_is_better=higher_is_better,
            pareto_objectives=getattr(self.config, "pareto_objectives", []),
            pareto_objectives_weight=getattr(self.config, "pareto_objectives_weight", 0.0),
            fitness_key=getattr(self.config, "fitness_key", None),
        )

        # Create FRESH diversity strategy for new island
        # This is critical for stateful strategies like MetricDiversity
        # which maintain internal state that would be contaminated if shared
        diversity_strategy = create_diversity_strategy(
            self._diversity_strategy_type,
            higher_is_better=higher_is_better,
        )

        new_archive = UnifiedArchive(
            config=archive_config,
            diversity_strategy=diversity_strategy,
        )
        self.archives.append(new_archive)
        self.island_config_names.append(config_name)

        # Add new dimension to adapter
        state = AdaptiveState(
            decay=self.decay,
            intensity_min=self.intensity_min,
            intensity_max=self.intensity_max,
        )
        self.adapter.add_dimension(state)

        # Seed new island with top programs
        self._seed_new_island(new_island_idx)

        # Update count and record spawn
        self.num_islands += 1
        self.last_spawn_iteration = self._iteration_count

        logger.info(
            f"Spawned new island {new_island_idx} with config '{config_name}' "
            f"(total islands: {self.num_islands}/{self.max_islands})"
        )

        return new_island_idx

    def _select_spawn_config(self) -> Tuple[str, Dict[str, Any]]:
        """
        Select a configuration preset for a new island.

        Prefers presets that are not yet used or underused.
        """
        usage_counts = {preset["name"]: 0 for preset in ISLAND_CONFIG_PRESETS}
        for name in self.island_config_names:
            if name in usage_counts:
                usage_counts[name] += 1

        min_usage = min(usage_counts.values())
        underused = [
            preset for preset in ISLAND_CONFIG_PRESETS if usage_counts[preset["name"]] == min_usage
        ]

        selected = random.choice(underused)
        return selected["name"], selected

    def _seed_new_island(self, island_idx: int) -> None:
        """Seed a new island with top programs from existing islands."""
        # Gather top programs from all existing islands
        all_programs = []
        for i in range(island_idx):  # Don't include the new island
            all_programs.extend(self.archives[i].get_all())

        if not all_programs:
            return

        # Get top programs to seed
        sorted_programs = sorted(all_programs, key=self._get_fitness, reverse=True)
        seed_count = min(5, len(sorted_programs))

        for program in sorted_programs[:seed_count]:
            # Create copy for new island
            copy = Program(
                id=str(uuid.uuid4()),
                solution=program.solution,
                language=program.language,
                metrics=program.metrics.copy() if program.metrics else {},
                iteration_found=self._iteration_count,
                parent_id=program.id,
                generation=program.generation,
                metadata={"seeded_to_spawned_island": island_idx},
            )
            self.archives[island_idx].add(copy)
            self.programs[copy.id] = copy

        self._invalidate_global_pareto_cache()

    # =========================================================================
    # Paradigm Breakthrough
    # =========================================================================

    def is_paradigm_stagnating(self) -> bool:
        """Check if global improvement rate is below threshold for paradigm generation."""
        if self.paradigm_tracker is None:
            return False
        return self.paradigm_tracker.is_paradigm_stagnating()

    def has_active_paradigm(self) -> bool:
        """Check if there's an active paradigm available."""
        if self.paradigm_tracker is None:
            return False
        return self.paradigm_tracker.has_active_paradigm()

    def get_current_paradigm(self) -> Optional[Dict[str, Any]]:
        """Get the current active paradigm if available."""
        if self.paradigm_tracker is None:
            return None
        return self.paradigm_tracker.get_current_paradigm()

    def use_paradigm(self) -> None:
        """Record one use of the current paradigm."""
        if self.paradigm_tracker is not None:
            self.paradigm_tracker.use_paradigm()

    def set_paradigms(self, paradigms: List[Dict[str, Any]]) -> None:
        """Set new paradigms from generator."""
        if self.paradigm_tracker is not None:
            self.paradigm_tracker.set_paradigms(paradigms, self._global_best_score)

    def get_previously_tried_ideas(self) -> List[str]:
        """Get formatted list of previously tried paradigm ideas."""
        if self.paradigm_tracker is None:
            return []
        return self.paradigm_tracker.get_previously_tried_ideas()

    def get_paradigm_num_to_generate(self) -> int:
        """Get the configured number of paradigms to generate."""
        if self.paradigm_tracker is None:
            return 3
        return self.paradigm_tracker.num_paradigms_to_generate
