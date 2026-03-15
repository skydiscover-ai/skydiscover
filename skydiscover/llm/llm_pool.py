"""LLM pool -- weighted sampling over one or more LLM backends."""

import asyncio
import logging
import random
from typing import Any, Dict, List

from skydiscover.config import LLMModelConfig
from skydiscover.llm.base import LLMResponse
from skydiscover.llm.openai import OpenAILLM

logger = logging.getLogger("skydiscover.llm")


class LLMPool:
    """Weighted pool of LLM backends. Samples one per generate() call."""

    def __init__(self, models_cfg: List[LLMModelConfig]):
        if not models_cfg:
            raise ValueError("LLMPool requires at least one model config")

        self.models_cfg = models_cfg

        # Validate weights before creating clients to fail fast on bad config.
        self.weights = [m.weight for m in models_cfg]
        if any(w < 0 for w in self.weights):
            raise ValueError("LLMPool model weights must be non-negative")
        total = sum(self.weights)
        if total <= 0:
            raise ValueError("LLMPool model weights must sum to a positive value")
        self.weights = [w / total for w in self.weights]

        self.models = [
            model_cfg.init_client(model_cfg) if model_cfg.init_client else OpenAILLM(model_cfg)
            for model_cfg in models_cfg
        ]
        self.random_state = random.Random()

        # Logging
        if len(models_cfg) > 1:
            pool_key = tuple((c.name, w) for c, w in zip(models_cfg, self.weights))
            if not hasattr(logger, "_logged_pools"):
                logger._logged_pools = set()
            if pool_key not in logger._logged_pools:
                parts = ", ".join(f"{c.name}={w:.2f}" for c, w in zip(models_cfg, self.weights))
                logger.info(f"Pool weights: {parts}")
                logger._logged_pools.add(pool_key)

    def _sample_model(self):
        """
        Simple weighted sampling mechanism. Override this to implement a more complex sampling mechanism.
        """
        idx = self.random_state.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        return self.models[idx]

    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        """Sample a model and generate a response."""
        model = self._sample_model()
        return await model.generate(system_message, messages, **kwargs)

    async def generate_all(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> List[LLMResponse]:
        """Generate using all models concurrently."""
        return await asyncio.gather(
            *(model.generate(system_message, messages, **kwargs) for model in self.models)
        )
