"""
OpenEvolve Native Database — island-based MAP-Elites search for SkyDiscover.

Faithful port of the OpenEvolve search algorithm (MAP-Elites with island-based
population model).  All core search logic — sampling, MAP-Elites grid,
archive management, migration — mirrors the reference implementation at
``openevolve/database.py``.

SkyDiscover adaptations (minimum necessary):
  - ``sample()`` returns ``(Program, List[Program])`` — the framework's
    ``DiscoveryController`` normalises both plain and dict-wrapped returns.
  - Uses ``Program.solution`` (SkyDiscover's field name).
  - Island rotation lives in ``sample()``, generation increment + migration
    check live in ``add()``, because SkyDiscover's ``DiscoveryController`` does
    not call these as separate steps.

Named ``openevolve_native`` (not ``openevolve``) to avoid any confusion with
the external ``openevolve`` package.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fitness helpers — port of openevolve/utils/metrics_utils.py
# ---------------------------------------------------------------------------


def _safe_numeric_average(metrics: Dict[str, Any]) -> float:
    """Average of numeric metric values, ignoring non-numeric entries."""
    if not metrics:
        return 0.0
    numeric_values = []
    for value in metrics.values():
        if isinstance(value, (int, float)):
            try:
                fv = float(value)
                if fv == fv:  # NaN guard
                    numeric_values.append(fv)
            except (ValueError, TypeError, OverflowError):
                continue
    return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0


def _get_fitness(
    metrics: Dict[str, Any],
    feature_dimensions: List[str] = (),
) -> float:
    """Fitness score, preferring ``combined_score`` and excluding feature dims."""
    if not metrics:
        return 0.0
    if "combined_score" in metrics:
        try:
            return float(metrics["combined_score"])
        except (ValueError, TypeError):
            pass

    feature_dims = set(feature_dimensions) if feature_dimensions else set()
    fitness_metrics: Dict[str, float] = {}
    for key, value in metrics.items():
        if key not in feature_dims and isinstance(value, (int, float)):
            try:
                fv = float(value)
                if fv == fv:
                    fitness_metrics[key] = fv
            except (ValueError, TypeError, OverflowError):
                continue

    if not fitness_metrics:
        return _safe_numeric_average(metrics)
    return _safe_numeric_average(fitness_metrics)


# ===========================================================================
# OpenEvolveNativeDatabase
# ===========================================================================


class OpenEvolveNativeDatabase(ProgramDatabase):
    """Island-based MAP-Elites database — native port of OpenEvolve."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, name: str, config: DatabaseConfig, **kwargs: Any):
        # --- Read config (getattr for fields not on base DatabaseConfig) ---
        self.num_islands: int = getattr(config, "num_islands", 5)
        self.population_size: int = getattr(config, "population_size", 40)
        self.archive_size: int = getattr(config, "archive_size", 100)
        self.exploration_ratio: float = getattr(config, "exploration_ratio", 0.2)
        self.exploitation_ratio: float = getattr(config, "exploitation_ratio", 0.7)
        self.elite_selection_ratio: float = getattr(config, "elite_selection_ratio", 0.1)
        self.feature_dimensions: List[str] = list(
            getattr(config, "feature_dimensions", ["complexity", "diversity"])
        )

        raw_bins = getattr(config, "feature_bins", 10)
        if isinstance(raw_bins, int):
            self.feature_bins: int = max(
                raw_bins,
                int(pow(self.archive_size, 1 / max(len(self.feature_dimensions), 1)) + 0.99),
            )
        else:
            self.feature_bins = 10

        if isinstance(raw_bins, dict):
            self.feature_bins_per_dim: Dict[str, int] = raw_bins
        else:
            self.feature_bins_per_dim = {d: self.feature_bins for d in self.feature_dimensions}

        self.diversity_reference_size: int = getattr(config, "diversity_reference_size", 20)
        self.migration_interval: int = getattr(config, "migration_interval", 10)
        self.migration_rate: float = getattr(config, "migration_rate", 0.1)

        # --- Island state (MUST be set before super().__init__ which may
        #     call self.load() if db_path exists) ---
        self.islands: List[Set[str]] = [set() for _ in range(self.num_islands)]
        self.island_feature_maps: List[Dict[str, str]] = [{} for _ in range(self.num_islands)]
        self.island_best_programs: List[Optional[str]] = [None] * self.num_islands
        self.island_generations: List[int] = [0] * self.num_islands
        self.current_island: int = 0
        self.last_migration_generation: int = 0

        # --- Global archive ---
        self.archive: Set[str] = set()

        # --- Feature scaling ---
        self.feature_stats: Dict[str, Dict[str, Any]] = {}

        # --- Diversity cache ---
        self.diversity_cache: Dict[int, Dict[str, float]] = {}
        self.diversity_cache_size: int = 1000
        self.diversity_reference_set: List[str] = []

        # --- Now safe to call super (which may trigger self.load) ---
        super().__init__(name, config, **kwargs)

        # --- Seed RNG ---
        random_seed = getattr(config, "random_seed", None)
        if random_seed is not None:
            random.seed(random_seed)

        logger.debug(
            "OpenEvolveNativeDatabase: %d islands, pop=%d, archive=%d, "
            "features=%s, bins=%d, migration_interval=%d, migration_rate=%.2f",
            self.num_islands,
            self.population_size,
            self.archive_size,
            self.feature_dimensions,
            self.feature_bins,
            self.migration_interval,
            self.migration_rate,
        )

    # ==================================================================
    # sample()
    # ==================================================================

    def sample(
        self,
        num_context_programs: Optional[int] = 4,
        **kwargs: Any,
    ) -> Tuple[Program, List[Program]]:
        if not self.programs:
            raise ValueError("Cannot sample: no programs in database")

        parent = self._sample_parent()
        if num_context_programs is None:
            num_context_programs = 4
        other_context_programs = self._sample_other_context_programs(parent, n=num_context_programs)

        logger.debug(
            "Sampled parent %s from island %d, %d other context programs",
            parent.id,
            self.current_island,
            len(other_context_programs),
        )

        # Round-robin island rotation AFTER sampling so the first call
        # uses island 0 (OpenEvolve's controller calls next_island()
        # separately; we do it here because DiscoveryController doesn't).
        self.current_island = (self.current_island + 1) % self.num_islands

        return parent, other_context_programs

    # ==================================================================
    # add()
    # ==================================================================

    def add(
        self,
        program: Program,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        target_island: Optional[int] = kwargs.get("target_island")
        is_migration: bool = kwargs.get("_is_migration", False)

        if iteration is not None:
            program.iteration_found = iteration
            self.last_iteration = max(self.last_iteration, iteration)

        self.programs[program.id] = program

        # Feature coordinates for MAP-Elites
        feature_coords = self._calculate_feature_coords(program)

        # --- Determine target island ---
        if target_island is not None:
            island_idx = target_island
        elif program.parent_id:
            parent = self.programs.get(program.parent_id)
            if parent and "island" in parent.metadata:
                island_idx = parent.metadata["island"]
            else:
                island_idx = self.current_island
        else:
            island_idx = self.current_island
        island_idx = island_idx % self.num_islands

        # --- MAP-Elites: insert into island feature map if better ---
        feature_key = self._feature_coords_to_key(feature_coords)
        island_feature_map = self.island_feature_maps[island_idx]

        should_replace = feature_key not in island_feature_map
        if not should_replace:
            existing_id = island_feature_map[feature_key]
            if existing_id not in self.programs:
                should_replace = True  # stale reference
            else:
                should_replace = self._is_better(program, self.programs[existing_id])

        if should_replace:
            if feature_key in island_feature_map:
                existing_id = island_feature_map[feature_key]
                if existing_id in self.programs:
                    # Swap archive membership
                    if existing_id in self.archive:
                        self.archive.discard(existing_id)
                        self.archive.add(program.id)
                # Remove replaced program from island set
                self.islands[island_idx].discard(existing_id)
            island_feature_map[feature_key] = program.id

        # Add to island
        self.islands[island_idx].add(program.id)
        program.metadata["island"] = island_idx

        # Archive, population limit, best-program tracking
        self._update_archive(program)
        self._enforce_population_limit(exclude_program_id=program.id)
        self._update_best_program(program)
        self._update_island_best_program(program, island_idx)

        # Save to disk
        if self.config.db_path:
            self._save_program(program)

        # --- Post-add: generation increment + migration ---
        # (OpenEvolve's controller calls these separately; we fold them
        # into add() because DiscoveryController doesn't.)
        if not is_migration:
            self.island_generations[island_idx] += 1
            if self._should_migrate():
                self._migrate_programs()

        logger.debug("Added program %s to island %d", program.id, island_idx)
        return program.id

    # ==================================================================
    # Parent sampling (exploration / exploitation / random)
    # ==================================================================

    def _sample_parent(self) -> Program:
        rand_val = random.random()
        if rand_val < self.exploration_ratio:
            return self._sample_exploration_parent()
        elif rand_val < self.exploration_ratio + self.exploitation_ratio:
            return self._sample_exploitation_parent()
        else:
            return self._sample_random_parent()

    def _sample_exploration_parent(self) -> Program:
        """Random program from current island (diverse sampling)."""
        island_programs = self.islands[self.current_island]

        if not island_programs:
            return self._seed_empty_island(self.current_island)

        valid = [pid for pid in island_programs if pid in self.programs]
        # Remove stale refs
        if len(valid) < len(island_programs):
            for stale in island_programs - set(valid):
                self.islands[self.current_island].discard(stale)

        if not valid:
            return self._seed_empty_island(self.current_island)

        return self.programs[random.choice(valid)]

    def _sample_exploitation_parent(self) -> Program:
        """Elite program from archive, preferring current island."""
        if not self.archive:
            return self._sample_exploration_parent()

        valid_archive = [pid for pid in self.archive if pid in self.programs]
        # Remove stale refs
        if len(valid_archive) < len(self.archive):
            for stale in self.archive - set(valid_archive):
                self.archive.discard(stale)

        if not valid_archive:
            return self._sample_exploration_parent()

        # Prefer archive programs from current island
        in_island = [
            pid
            for pid in valid_archive
            if self.programs[pid].metadata.get("island") == self.current_island
        ]

        if in_island:
            return self.programs[random.choice(in_island)]
        return self.programs[random.choice(valid_archive)]

    def _sample_random_parent(self) -> Program:
        """Uniformly random program from entire population."""
        return self.programs[random.choice(list(self.programs.keys()))]

    def _seed_empty_island(self, island_idx: int) -> Program:
        """Seed an empty island with a copy of the best program."""
        if self.best_program_id and self.best_program_id in self.programs:
            best = self.programs[self.best_program_id]
            copy = Program(
                id=str(uuid.uuid4()),
                solution=best.solution,
                language=best.language,
                parent_id=best.id,
                generation=best.generation,
                metrics=best.metrics.copy(),
                metadata={"island": island_idx},
                iteration_found=self.last_iteration,
            )
            self.programs[copy.id] = copy
            self.islands[island_idx].add(copy.id)
            return copy
        return next(iter(self.programs.values()))

    # ==================================================================
    # Context program(s) sampling (island-scoped)
    # ==================================================================

    def _sample_other_context_programs(self, parent: Program, n: int = 4) -> List[Program]:
        """Sample other context programs from parent's island.

        Strategy (matching OpenEvolve):
          1. Island best (if different from parent)
          2. Top elite programs from island
          3. Programs from nearby MAP-Elites cells (±2 perturbation)
          4. Random fill from island
        """
        parent_island = parent.metadata.get("island", self.current_island)
        island_program_ids = list(self.islands[parent_island])
        island_programs = [self.programs[pid] for pid in island_program_ids if pid in self.programs]

        if not island_programs:
            return []

        other_context_programs: List[Program] = []
        used_ids: set = {parent.id}

        # 1. Island best
        island_best_id = self.island_best_programs[parent_island]
        if (
            island_best_id is not None
            and island_best_id != parent.id
            and island_best_id in self.programs
        ):
            other_context_programs.append(self.programs[island_best_id])
            used_ids.add(island_best_id)
        elif island_best_id is not None and island_best_id not in self.programs:
            self.island_best_programs[parent_island] = None

        # 2. Top elite programs from island
        top_n = max(1, int(n * self.elite_selection_ratio))
        top_island = sorted(
            island_programs,
            key=lambda p: _get_fitness(p.metrics, self.feature_dimensions),
            reverse=True,
        )[:top_n]
        for prog in top_island:
            if prog.id not in used_ids:
                other_context_programs.append(prog)
                used_ids.add(prog.id)

        # 3. Nearby MAP-Elites cells (±2 perturbation)
        if len(island_programs) > n and len(other_context_programs) < n:
            remaining_slots = n - len(other_context_programs)
            feature_coords = self._calculate_feature_coords(parent)

            # Build local feature-cell → program mapping for this island
            cell_map: Dict[str, str] = {}
            for pid in island_program_ids:
                if pid in self.programs:
                    coords = self._calculate_feature_coords(self.programs[pid])
                    cell_map[self._feature_coords_to_key(coords)] = pid

            nearby: List[Program] = []
            for _ in range(remaining_slots * 3):
                perturbed = [
                    max(0, min(self.feature_bins - 1, c + random.randint(-2, 2)))
                    for c in feature_coords
                ]
                key = self._feature_coords_to_key(perturbed)
                if key in cell_map:
                    pid = cell_map[key]
                    if (
                        pid not in used_ids
                        and pid not in {p.id for p in nearby}
                        and pid in self.programs
                    ):
                        nearby.append(self.programs[pid])
                        if len(nearby) >= remaining_slots:
                            break

            # 4. Random fill from island
            if len(other_context_programs) + len(nearby) < n:
                remaining = n - len(other_context_programs) - len(nearby)
                all_used = used_ids | {p.id for p in nearby}
                available = [
                    pid
                    for pid in island_program_ids
                    if pid not in all_used and pid in self.programs
                ]
                if available:
                    sampled = random.sample(available, min(remaining, len(available)))
                    nearby.extend(self.programs[pid] for pid in sampled)

            other_context_programs.extend(nearby)

        return other_context_programs[:n]

    # ==================================================================
    # MAP-Elites feature coordinates
    # ==================================================================

    def _calculate_feature_coords(self, program: Program) -> List[int]:
        coords: List[int] = []
        for dim in self.feature_dimensions:
            # Priority 1: custom metric from evaluator
            if dim in program.metrics:
                coords.append(self._to_bin(dim, program.metrics[dim]))
            # Priority 2: built-in features
            elif dim == "complexity":
                coords.append(self._to_bin("complexity", float(len(program.solution))))
            elif dim == "diversity":
                if len(self.programs) < 2:
                    coords.append(0)
                else:
                    coords.append(self._to_bin("diversity", self._get_cached_diversity(program)))
            elif dim == "score":
                if not program.metrics:
                    coords.append(0)
                else:
                    coords.append(
                        self._to_bin(
                            "score",
                            _get_fitness(program.metrics, self.feature_dimensions),
                        )
                    )
            else:
                raise ValueError(
                    f"Feature dimension '{dim}' not found in program metrics. "
                    f"Available metrics: {list(program.metrics.keys())}. "
                    f"Built-in features: 'complexity', 'diversity', 'score'."
                )
        return coords

    def _to_bin(self, dim: str, value: float) -> int:
        """Update running stats, min-max scale, and return bin index."""
        self._update_feature_stats(dim, value)
        scaled = self._scale_feature_value(dim, value)
        num_bins = self.feature_bins_per_dim.get(dim, self.feature_bins)
        return max(0, min(num_bins - 1, int(scaled * num_bins)))

    @staticmethod
    def _feature_coords_to_key(coords: List[int]) -> str:
        return "-".join(str(c) for c in coords)

    # ==================================================================
    # Feature scaling (min-max)
    # ==================================================================

    def _update_feature_stats(self, feature_name: str, value: float) -> None:
        if feature_name not in self.feature_stats:
            self.feature_stats[feature_name] = {
                "min": value,
                "max": value,
                "values": [],
            }
        stats = self.feature_stats[feature_name]
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)
        stats["values"].append(value)
        if len(stats["values"]) > 1000:
            stats["values"] = stats["values"][-1000:]

    def _scale_feature_value(self, feature_name: str, value: float) -> float:
        if feature_name not in self.feature_stats:
            return min(1.0, max(0.0, value))
        stats = self.feature_stats[feature_name]
        min_val, max_val = stats["min"], stats["max"]
        if max_val == min_val:
            return 0.5
        return min(1.0, max(0.0, (value - min_val) / (max_val - min_val)))

    # ==================================================================
    # Fast code diversity
    # ==================================================================

    @staticmethod
    def _fast_code_diversity(code1: str, code2: str) -> float:
        if code1 == code2:
            return 0.0
        length_diff = abs(len(code1) - len(code2))
        line_diff = abs(code1.count("\n") - code2.count("\n"))
        char_diff = len(set(code1).symmetric_difference(set(code2)))
        return length_diff * 0.1 + line_diff * 10 + char_diff * 0.5

    def _get_cached_diversity(self, program: Program) -> float:
        code_hash = hash(program.solution)

        if code_hash in self.diversity_cache:
            return self.diversity_cache[code_hash]["value"]

        if (
            not self.diversity_reference_set
            or len(self.diversity_reference_set) < self.diversity_reference_size
        ):
            self._update_diversity_reference_set()

        scores = [
            self._fast_code_diversity(program.solution, ref)
            for ref in self.diversity_reference_set
            if ref != program.solution
        ]
        diversity = sum(scores) / max(1, len(scores)) if scores else 0.0

        # LRU eviction
        if len(self.diversity_cache) >= self.diversity_cache_size:
            oldest = min(self.diversity_cache, key=lambda h: self.diversity_cache[h]["timestamp"])
            del self.diversity_cache[oldest]

        self.diversity_cache[code_hash] = {
            "value": diversity,
            "timestamp": time.time(),
        }
        return diversity

    def _update_diversity_reference_set(self) -> None:
        if not self.programs:
            return
        all_progs = list(self.programs.values())

        if len(all_progs) <= self.diversity_reference_size:
            self.diversity_reference_set = [p.solution for p in all_progs]
            return

        # Greedy-diverse selection
        selected: List[Program] = []
        remaining = all_progs.copy()

        selected.append(remaining.pop(random.randint(0, len(remaining) - 1)))

        while len(selected) < self.diversity_reference_size and remaining:
            best_idx, best_div = -1, -1.0
            for i, cand in enumerate(remaining):
                min_d = min(self._fast_code_diversity(cand.solution, s.solution) for s in selected)
                if min_d > best_div:
                    best_div = min_d
                    best_idx = i
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))

        self.diversity_reference_set = [p.solution for p in selected]

    # ==================================================================
    # Fitness comparison (excluding feature dimensions)
    # ==================================================================

    def _is_better(self, program1: Program, program2: Program) -> bool:
        if not program1.metrics and not program2.metrics:
            return program1.timestamp > program2.timestamp
        if program1.metrics and not program2.metrics:
            return True
        if not program1.metrics and program2.metrics:
            return False
        return _get_fitness(program1.metrics, self.feature_dimensions) > _get_fitness(
            program2.metrics, self.feature_dimensions
        )

    # ==================================================================
    # Archive management
    # ==================================================================

    def _update_archive(self, program: Program) -> None:
        if len(self.archive) < self.archive_size:
            self.archive.add(program.id)
            return

        # Clean stale refs
        valid = []
        for pid in list(self.archive):
            if pid in self.programs:
                valid.append(self.programs[pid])
            else:
                self.archive.discard(pid)

        if len(self.archive) < self.archive_size:
            self.archive.add(program.id)
            return

        # Replace worst if new program is better
        if valid:
            worst = min(
                valid,
                key=lambda p: _get_fitness(p.metrics, self.feature_dimensions),
            )
            if self._is_better(program, worst):
                self.archive.discard(worst.id)
                self.archive.add(program.id)
        else:
            self.archive.add(program.id)

    # ==================================================================
    # Best program tracking
    # ==================================================================

    def _update_best_program(self, program: Program) -> None:
        if self.best_program_id is None:
            self.best_program_id = program.id
            return
        if self.best_program_id not in self.programs:
            self.best_program_id = program.id
            return
        current_best = self.programs[self.best_program_id]
        if self._is_better(program, current_best):
            old_id = self.best_program_id
            self.best_program_id = program.id
            if "combined_score" in program.metrics and "combined_score" in current_best.metrics:
                logger.debug(
                    "New best program %s replaces %s (%.4f -> %.4f)",
                    program.id,
                    old_id,
                    current_best.metrics["combined_score"],
                    program.metrics["combined_score"],
                )

    def _update_island_best_program(self, program: Program, island_idx: int) -> None:
        if island_idx >= len(self.island_best_programs):
            return
        cur_id = self.island_best_programs[island_idx]
        if cur_id is None or cur_id not in self.programs:
            self.island_best_programs[island_idx] = program.id
            return
        if self._is_better(program, self.programs[cur_id]):
            self.island_best_programs[island_idx] = program.id

    # ==================================================================
    # Population limit
    # ==================================================================

    def _enforce_population_limit(self, exclude_program_id: Optional[str] = None) -> None:
        if len(self.programs) <= self.population_size:
            return

        num_to_remove = len(self.programs) - self.population_size
        sorted_progs = sorted(
            self.programs.values(),
            key=lambda p: _get_fitness(p.metrics, self.feature_dimensions),
        )

        protected = {self.best_program_id, exclude_program_id} - {None}
        to_remove: List[Program] = []
        for prog in sorted_progs:
            if len(to_remove) >= num_to_remove:
                break
            if prog.id not in protected:
                to_remove.append(prog)

        for prog in to_remove:
            pid = prog.id
            self.programs.pop(pid, None)

            for imap in self.island_feature_maps:
                for k in [k for k, v in imap.items() if v == pid]:
                    del imap[k]

            for island in self.islands:
                island.discard(pid)

            self.archive.discard(pid)

        self._cleanup_stale_island_bests()
        logger.debug(
            "Population limit: removed %d, now %d",
            len(to_remove),
            len(self.programs),
        )

    def _cleanup_stale_island_bests(self) -> None:
        for i, best_id in enumerate(self.island_best_programs):
            if best_id is None:
                continue
            if best_id not in self.programs or best_id not in self.islands[i]:
                self.island_best_programs[i] = None
                # Recalculate
                progs = [self.programs[pid] for pid in self.islands[i] if pid in self.programs]
                if progs:
                    self.island_best_programs[i] = max(
                        progs,
                        key=lambda p: _get_fitness(p.metrics, self.feature_dimensions),
                    ).id

    # ==================================================================
    # Migration (ring topology)
    # ==================================================================

    def _should_migrate(self) -> bool:
        if self.num_islands < 2:
            return False
        return (
            max(self.island_generations) - self.last_migration_generation
        ) >= self.migration_interval

    def _migrate_programs(self) -> None:
        if self.num_islands < 2:
            return
        logger.debug("Performing migration between islands")

        for i, island in enumerate(self.islands):
            if not island:
                continue
            island_progs = [self.programs[pid] for pid in island if pid in self.programs]
            if not island_progs:
                continue

            island_progs.sort(
                key=lambda p: _get_fitness(p.metrics, self.feature_dimensions),
                reverse=True,
            )
            num_migrants = max(1, int(len(island_progs) * self.migration_rate))
            migrants = island_progs[:num_migrants]

            targets = [
                (i + 1) % self.num_islands,
                (i - 1) % self.num_islands,
            ]

            for migrant in migrants:
                # Skip already-migrated programs to prevent exponential
                # duplication (all copies map to same MAP-Elites cell).
                if migrant.metadata.get("migrant", False):
                    continue

                for target in targets:
                    # Skip if target island already has identical solution
                    target_progs = [
                        self.programs[pid] for pid in self.islands[target] if pid in self.programs
                    ]
                    if any(p.solution == migrant.solution for p in target_progs):
                        continue

                    copy = Program(
                        id=str(uuid.uuid4()),
                        solution=migrant.solution,
                        language=migrant.language,
                        parent_id=migrant.id,
                        generation=migrant.generation,
                        metrics=migrant.metrics.copy(),
                        metadata={
                            **migrant.metadata,
                            "island": target,
                            "migrant": True,
                        },
                    )
                    self.add(
                        copy,
                        target_island=target,
                        _is_migration=True,
                    )

        self.last_migration_generation = max(self.island_generations)
        logger.debug(
            "Migration completed at generation %d",
            self.last_migration_generation,
        )

    # ==================================================================
    # Save / Load (extends base to persist island metadata)
    # ==================================================================

    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:
        super().save(path=path, iteration=iteration)

        save_path = path or getattr(self.config, "db_path", None)
        if not save_path:
            return

        meta = {
            "island_feature_maps": self.island_feature_maps,
            "islands": [list(s) for s in self.islands],
            "archive": list(self.archive),
            "island_best_programs": self.island_best_programs,
            "current_island": self.current_island,
            "island_generations": self.island_generations,
            "last_migration_generation": self.last_migration_generation,
            "feature_stats": self._serialize_feature_stats(),
        }
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "openevolve_native_metadata.json"), "w") as f:
            json.dump(meta, f)

    def load(self, path: str) -> None:
        super().load(path)

        meta_path = os.path.join(path, "openevolve_native_metadata.json")
        if not os.path.exists(meta_path):
            logger.warning(
                "No openevolve_native_metadata.json found; distributing " "programs round-robin"
            )
            self._distribute_programs_to_islands()
            return

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.island_feature_maps = meta.get(
            "island_feature_maps", [{} for _ in range(self.num_islands)]
        )
        saved_islands = meta.get("islands", [])
        self.archive = set(meta.get("archive", []))
        self.island_best_programs = meta.get("island_best_programs", [None] * self.num_islands)
        self.current_island = meta.get("current_island", 0)
        self.island_generations = meta.get("island_generations", [0] * self.num_islands)
        self.last_migration_generation = meta.get("last_migration_generation", 0)
        self.feature_stats = self._deserialize_feature_stats(meta.get("feature_stats", {}))

        self._reconstruct_islands(saved_islands)
        self._log_island_status()

    # ------------------------------------------------------------------
    # Island reconstruction helpers
    # ------------------------------------------------------------------

    def _reconstruct_islands(self, saved_islands: List[List[str]]) -> None:
        num_islands = max(len(saved_islands), self.num_islands)
        self.islands = [set() for _ in range(num_islands)]

        for island_idx, program_ids in enumerate(saved_islands):
            if island_idx >= len(self.islands):
                continue
            for pid in program_ids:
                if pid in self.programs:
                    self.islands[island_idx].add(pid)
                    self.programs[pid].metadata["island"] = island_idx

        # Clean stale refs from archive and feature maps
        self.archive = {pid for pid in self.archive if pid in self.programs}
        for imap in self.island_feature_maps:
            for k in [k for k, pid in imap.items() if pid not in self.programs]:
                del imap[k]

        self._cleanup_stale_island_bests()

        if self.best_program_id and self.best_program_id not in self.programs:
            self.best_program_id = None

        # If no island assignments recovered, distribute round-robin
        if self.programs and sum(len(s) for s in self.islands) == 0:
            self._distribute_programs_to_islands()

        # Ensure list lengths match
        while len(self.island_generations) < len(self.islands):
            self.island_generations.append(0)
        while len(self.island_best_programs) < len(self.islands):
            self.island_best_programs.append(None)

    def _distribute_programs_to_islands(self) -> None:
        for i, pid in enumerate(self.programs):
            idx = i % self.num_islands
            self.islands[idx].add(pid)
            self.programs[pid].metadata["island"] = idx

    # ------------------------------------------------------------------
    # Feature-stats serialisation
    # ------------------------------------------------------------------

    def _serialize_feature_stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, stats in self.feature_stats.items():
            s: Dict[str, Any] = {}
            for k, v in stats.items():
                if k == "values":
                    s[k] = v[-100:] if isinstance(v, list) and len(v) > 100 else v
                elif hasattr(v, "item"):  # numpy scalar
                    s[k] = v.item()
                else:
                    s[k] = v
            out[name] = s
        return out

    def _deserialize_feature_stats(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if not raw:
            return {}
        out: Dict[str, Any] = {}
        for name, stats in raw.items():
            if isinstance(stats, dict):
                out[name] = {
                    "min": float(stats.get("min", 0.0)),
                    "max": float(stats.get("max", 1.0)),
                    "values": list(stats.get("values", [])),
                }
        return out

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_island_status(self) -> None:
        for i, island in enumerate(self.islands):
            progs = [self.programs[pid] for pid in island if pid in self.programs]
            if progs:
                scores = [_get_fitness(p.metrics, self.feature_dimensions) for p in progs]
                best, avg = max(scores), sum(scores) / len(scores)
            else:
                best = avg = 0.0
            cells = len(self.island_feature_maps[i]) if i < len(self.island_feature_maps) else 0
            gen = self.island_generations[i] if i < len(self.island_generations) else 0
            logger.debug(
                "Island %d: %d programs, %d cells, gen=%d, best=%.4f, avg=%.4f%s",
                i,
                len(progs),
                cells,
                gen,
                best,
                avg,
                " [current]" if i == self.current_island else "",
            )
