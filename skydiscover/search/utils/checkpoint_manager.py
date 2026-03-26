"""
Checkpoint management for program databases.

Handles saving and loading database state to/from disk.
"""

import json
import logging
import os
from typing import Dict, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program

logger = logging.getLogger(__name__)


class SafeJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that handles non-serializable types gracefully.

    This is important for evolved databases where LLM-generated code may
    store non-serializable types (like sets) in program metadata.
    """

    def default(self, obj):
        # Convert numpy arrays/scalars to Python types
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:
            pass
        # Convert sets to sorted lists for consistency
        if isinstance(obj, set):
            return sorted(list(obj))
        # Convert frozensets to sorted lists
        if isinstance(obj, frozenset):
            return sorted(list(obj))
        # Let the base class raise TypeError for other non-serializable types
        return super().default(obj)


class CheckpointManager:
    """
    Manages database checkpointing (save/load operations).
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config

    def save(
        self,
        programs: Dict[str, Program],
        prompts_by_program: Optional[Dict[str, Dict[str, Dict[str, str]]]],
        best_program_id: Optional[str],
        last_iteration: int,
        path: Optional[str] = None,
    ) -> None:
        """
        Save the database to disk

        Args:
            programs: Dictionary of program ID to Program
            prompts_by_program: Optional prompts by program ID
            best_program_id: ID of the best program
            last_iteration: Last iteration number
            path: Path to save to (uses config.db_path if None)
        """
        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return

        # create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save each program
        for program in programs.values():
            prompts = None
            if self.config.log_prompts and prompts_by_program and program.id in prompts_by_program:
                prompts = prompts_by_program[program.id]
            self._save_program(program, save_path, prompts=prompts)

        # Save metadata
        metadata = {
            "best_program_id": best_program_id,
            "last_iteration": last_iteration,
        }

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        logger.info(f"[CHECKPOINT] Saved database with {len(programs)} programs to {save_path}")

    def load(self, path: str) -> Tuple[Dict[str, Program], Optional[str], int]:
        """
        Load the database from disk

        Args:
            path: Path to load from

        Returns:
            Tuple of (programs_dict, best_program_id, last_iteration)
        """
        # Import here to avoid circular import
        from skydiscover.search.base_database import Program

        programs: Dict[str, Program] = {}
        best_program_id: Optional[str] = None
        last_iteration: int = 0

        if not os.path.exists(path):
            logger.warning(f"Database path {path} does not exist, skipping load")
            return programs, best_program_id, last_iteration

        # Load metadata first
        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            best_program_id = metadata.get("best_program_id")
            last_iteration = metadata.get("last_iteration", 0)

            logger.info(f"Loaded database metadata with last_iteration={last_iteration}")

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
                        programs[program.id] = program
                    except Exception as e:
                        logger.warning(f"Error loading program {program_file}: {str(e)}")

        logger.info(f"Loaded database with {len(programs)} programs from {path}")

        return programs, best_program_id, last_iteration

    def _save_program(
        self,
        program: Program,
        base_path: Optional[str] = None,
        prompts: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        """
        Save a program to disk

        Args:
            program: Program to save
            base_path: Base path to save to (uses config.db_path if None)
            prompts: Optional prompts to save with the program, in the format {template_key: { 'system': str, 'user': str }}
        """
        save_path = base_path or self.config.db_path
        if not save_path:
            return

        # Create programs directory if it doesn't exist
        programs_dir = os.path.join(save_path, "programs")
        os.makedirs(programs_dir, exist_ok=True)

        # Save program
        program_dict = program.to_dict()
        if prompts:
            program_dict["prompts"] = prompts
        program_path = os.path.join(programs_dir, f"{program.id}.json")

        with open(program_path, "w") as f:
            json.dump(program_dict, f, cls=SafeJSONEncoder)
