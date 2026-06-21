"""Tests for meta-search LLM availability checks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skydiscover.config import LLMModelConfig
from skydiscover.llm.llm_pool import LLMPool


class TestLLMPoolCheckAvailability:
    """Tests for LLMPool.check_availability()."""

    def _make_pool(self, api_base="http://localhost:1234/v1"):
        cfg = LLMModelConfig(
            name="test-model",
            api_base=api_base,
            api_key="fake",
            timeout=10,
            retries=0,
        )
        with patch("skydiscover.llm.openai.openai.OpenAI"):
            pool = LLMPool([cfg])
        return pool

    @pytest.mark.asyncio
    async def test_returns_true_when_reachable(self):
        pool = self._make_pool()
        pool.models[0].generate = AsyncMock(return_value=MagicMock(text="ok"))
        result = await pool.check_availability()
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_unreachable(self):
        pool = self._make_pool()
        pool.models[0].generate = AsyncMock(side_effect=ConnectionError("refused"))
        result = await pool.check_availability()
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self):
        pool = self._make_pool()

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(10)

        pool.models[0].generate = slow_generate
        result = await pool.check_availability(timeout=0.1)
        assert result is False



class TestCoEvolutionControllerAvailabilityCheck:
    """Tests for CoEvolutionController._check_meta_llm_availability()."""

    def _make_controller(self):
        """Create a minimal CoEvolutionController without full init."""
        from skydiscover.search.evox.controller import CoEvolutionController

        controller = object.__new__(CoEvolutionController)
        controller.search_controller = MagicMock()

        guide_pool = MagicMock()
        guide_pool.models_cfg = [MagicMock(api_base="http://guide-llm:8000/v1")]
        meta_pool = MagicMock()
        meta_pool.models_cfg = [MagicMock(api_base="http://meta-llm:8000/v1")]

        controller.search_controller.guide_llms = guide_pool
        controller.search_controller.llms = meta_pool

        return controller

    @pytest.mark.asyncio
    async def test_both_reachable_logs_info(self, caplog):
        import logging

        controller = self._make_controller()
        controller.search_controller.guide_llms.check_availability = AsyncMock(return_value=True)
        controller.search_controller.llms.check_availability = AsyncMock(return_value=True)

        with caplog.at_level(logging.INFO, logger="skydiscover.search.evox.controller"):
            await controller._check_meta_llm_availability()

        assert "connectivity verified" in caplog.text
        assert "WARNING" not in caplog.text
        assert controller._guide_llm_available is True
        assert controller._meta_llm_available is True

    @pytest.mark.asyncio
    async def test_guide_unreachable_warns(self, caplog):
        import logging

        controller = self._make_controller()
        controller.search_controller.guide_llms.check_availability = AsyncMock(return_value=False)
        controller.search_controller.llms.check_availability = AsyncMock(return_value=True)

        with caplog.at_level(logging.WARNING, logger="skydiscover.search.evox.controller"):
            await controller._check_meta_llm_availability()

        assert "Guide LLM (label generation)" in caplog.text
        assert "http://guide-llm:8000/v1" in caplog.text
        assert "share_llm: true" in caplog.text
        assert controller._guide_llm_available is False
        assert controller._meta_llm_available is True

    @pytest.mark.asyncio
    async def test_meta_unreachable_warns(self, caplog):
        import logging

        controller = self._make_controller()
        controller.search_controller.guide_llms.check_availability = AsyncMock(return_value=True)
        controller.search_controller.llms.check_availability = AsyncMock(return_value=False)

        with caplog.at_level(logging.WARNING, logger="skydiscover.search.evox.controller"):
            await controller._check_meta_llm_availability()

        assert "Meta-search LLM (search strategy evolution)" in caplog.text
        assert "http://meta-llm:8000/v1" in caplog.text
        assert "share_llm: true" in caplog.text
        assert controller._guide_llm_available is True
        assert controller._meta_llm_available is False

