# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Initial search strategy database.

    When ``config.pareto_objectives`` is set, sampling prefers members of the
    global Pareto front (issue #42) while still mixing in non-front programs
    for exploration. Scalar mode keeps the original uniform random sample.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program to the database."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and set of context programs.

        Multiobjective: parent is drawn from the Pareto front when available;
        context mixes front members with the broader population.
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        front: List[EvolvedProgram] = []
        if self.is_multiobjective_enabled():
            front = [p for p in self.get_pareto_front() if p.id in self.programs]

        if front:
            parent = random.choice(front)
            # Prefer other front members for context, then fill from population.
            pool = [p for p in front if p.id != parent.id]
            if len(pool) < (num_context_programs or 0):
                extras = [p for p in candidates if p.id != parent.id and p not in pool]
                random.shuffle(extras)
                pool.extend(extras)
            examples = pool[:num_context_programs]
            if len(examples) < (num_context_programs or 0):
                # Extremely small front — allow duplicates avoidance only.
                examples = [p for p in candidates if p.id != parent.id][:num_context_programs]
        else:
            parent = random.choice(candidates)
            sample_size = min((num_context_programs or 0) + 1, len(candidates))
            examples = random.sample(candidates, sample_size)
            examples = [p for p in examples if p.id != parent.id][:num_context_programs]

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
