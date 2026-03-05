"""Tests for Optional temperature/top_p in LLM config and API params."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skydiscover.config import LLMConfig, LLMModelConfig


class TestLLMConfigOptionalParams:
    def test_default_values(self):
        cfg = LLMConfig(name="test-model")
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.95

    def test_top_p_none(self):
        cfg = LLMConfig(name="test-model", top_p=None)
        assert cfg.top_p is None
        assert cfg.temperature == 0.7

    def test_temperature_none(self):
        cfg = LLMConfig(name="test-model", temperature=None)
        assert cfg.temperature is None
        assert cfg.top_p == 0.95

    def test_both_none(self):
        cfg = LLMConfig(name="test-model", temperature=None, top_p=None)
        assert cfg.temperature is None
        assert cfg.top_p is None


class TestOpenAILLMParams:
    def _make_llm(self, temperature=0.7, top_p=0.95):
        from skydiscover.llm.openai import OpenAILLM
        cfg = LLMModelConfig(
            name="test-model",
            temperature=temperature,
            top_p=top_p,
            api_base="http://localhost:1234/v1",
            api_key="fake",
            timeout=10,
            retries=0,
            retry_delay=0,
        )
        with patch("skydiscover.llm.openai.openai.OpenAI"):
            llm = OpenAILLM(cfg)
        return llm

    @pytest.mark.asyncio
    async def test_params_include_temperature_and_top_p(self):
        llm = self._make_llm(temperature=0.5, top_p=0.9)
        llm._call_api = AsyncMock(return_value="response")

        await llm.generate(
            system_message="sys",
            messages=[{"role": "user", "content": "user"}],
            temperature=0.5, top_p=0.9,
        )

        params = llm._call_api.call_args[0][0]
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_params_exclude_none_top_p(self):
        llm = self._make_llm(top_p=None)
        llm._call_api = AsyncMock(return_value="response")

        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])

        params = llm._call_api.call_args[0][0]
        assert "top_p" not in params
        assert "temperature" in params

    @pytest.mark.asyncio
    async def test_params_exclude_none_temperature(self):
        llm = self._make_llm(temperature=None)
        llm._call_api = AsyncMock(return_value="response")

        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])

        params = llm._call_api.call_args[0][0]
        assert "temperature" not in params
        assert "top_p" in params

    @pytest.mark.asyncio
    async def test_params_exclude_both_none(self):
        llm = self._make_llm(temperature=None, top_p=None)
        llm._call_api = AsyncMock(return_value="response")

        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])

        params = llm._call_api.call_args[0][0]
        assert "temperature" not in params
        assert "top_p" not in params
