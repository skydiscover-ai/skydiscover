"""Minimal database for the Jitskit agentic strategy.

Jitskit (the multi-agent KV-store synthesis loop) runs its own internal
iteration loop in a host subprocess, so this database just stores the
programs the controller streams in as it reads each new best from the
runtime's ``leaderboard.json``.  Mirrors ``claude_code/database.py``.
"""

from skydiscover.search.base_database import Program, ProgramDatabase


class JitsKitDatabase(ProgramDatabase):
    def add(self, program: Program, iteration=None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(self, num_context_programs=4, **kwargs):
        best = self.get_best_program()
        return best, []
