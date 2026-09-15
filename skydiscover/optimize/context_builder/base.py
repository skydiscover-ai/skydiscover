"""Base context builder interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from skydiscover.optimize.config import Config
from skydiscover.optimize.search.base_database import Program


class ContextBuilder(ABC):
    """Abstract base for building LLM prompts.

    Subclass this and implement build_prompt(). Each subclass sets up its
    own template_manager and any other resources it needs.
    """

    def __init__(self, config: Config):
        self.config = config
        self.context_config = config.context_builder

    @abstractmethod
    def build_prompt(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Build a prompt for the LLM.

        Args:
            current_program: Program or {info: Program} to evolve from.
                When a dict, the key is additional context about the program.
            context: optional dict with keys such as program_metrics,
                other_context_programs, etc.

        Returns:
            Dict with "system" and "user" keys containing prompt strings.
        """
        pass
