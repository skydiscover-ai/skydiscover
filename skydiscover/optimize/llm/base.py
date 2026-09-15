"""Base LLM interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Response from an LLM generation call.

    text: generated text content.
    image_path: path to generated image file, or None for text-only.
    """

    text: str = ""
    image_path: Optional[str] = None


class LLMInterface(ABC):
    """Abstract base for LLM backends.

    Subclass this and implement generate() to add a new LLM provider.
    """

    @abstractmethod
    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            system_message: system prompt string.
            messages: conversation history as list of {role, content} dicts.
            **kwargs: backend-specific options (e.g. image_output=True for
                image generation, output_dir, program_id, temperature).

        Returns:
            LLMResponse with text and optional image_path.
        """
        pass
