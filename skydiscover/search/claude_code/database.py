"""Minimal database for the Claude Code baseline.

Claude Code handles its own internal iteration loop, so this database
just stores whatever the controller adds (typically one final result).
"""

from skydiscover.search.base_database import Program, ProgramDatabase


class ClaudeCodeDatabase(ProgramDatabase):
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
