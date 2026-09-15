import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class SearchStrategy(Program):
    """Program entry for the search strategy database."""


class SearchStrategyDatabase(ProgramDatabase):
    """Database for storing and sampling evolved search strategy programs."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)

    def add(self, program: SearchStrategy, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program to the database."""
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, SearchStrategy], Dict[str, List[SearchStrategy]]]:
        """
        Sample a search strategy to refine and other context strategies for evolution.
        """

        def safe_score(p):
            score = p.metrics.get("combined_score") if p.metrics else None
            if not isinstance(score, (int, float)):
                logger.warning(
                    f"Program {p.id} has invalid combined_score: {score}, metrics: {p.metrics}"
                )
            return float(score) if isinstance(score, (int, float)) else float("-inf")

        parent = max(self.programs.values(), key=safe_score)
        available_programs = list(self.programs.values())
        num_to_sample = max(0, min(num_context_programs, len(available_programs)))

        other_context_programs = (
            self.rng.sample(available_programs, num_to_sample) if num_to_sample > 0 else []
        )
        other_context_programs = [p for p in other_context_programs if p.id != parent.id]
        return {"": parent}, {"": other_context_programs}
