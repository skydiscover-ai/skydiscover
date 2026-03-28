"""Tests for LLM config, cost tracking, and agentic usage propagation."""

from dataclasses import fields
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import pytest

from skydiscover.config import AgenticConfig, LLMConfig, LLMModelConfig
from skydiscover.llm.base import LLMResponse
from skydiscover.llm.cost import CostTracker, extract_usage_counts, lookup_default_pricing
from skydiscover.llm.agentic_generator import AgenticGenerator
from skydiscover.llm.llm_pool import LLMPool

_OPENAI_DEFAULT_API_BASE: str = next(f.default for f in fields(LLMConfig) if f.name == "api_base")


class TestLLMConfigDefaults:
    def test_default_temperature(self):
        cfg = LLMConfig(name="test-model")
        assert cfg.temperature == 0.7

    def test_default_top_p_is_none(self):
        cfg = LLMConfig(name="test-model")
        assert cfg.top_p is None

    def test_explicit_none_temperature(self):
        cfg = LLMConfig(name="test-model", temperature=None)
        assert cfg.temperature is None

    def test_explicit_none_top_p(self):
        cfg = LLMConfig(name="test-model", top_p=None)
        assert cfg.top_p is None

    def test_both_none(self):
        cfg = LLMConfig(name="test-model", temperature=None, top_p=None)
        assert cfg.temperature is None
        assert cfg.top_p is None


class TestApiBaseRouting:
    def test_unknown_model_preserves_local_api_base(self):
        local = "http://localhost:11434/v1"
        cfg = LLMConfig(
            name="my-custom-local-model",
            api_base=local,
            models=[LLMModelConfig(name="my-custom-local-model")],
        )
        assert cfg.models[0].api_base == local

    def test_unknown_model_gets_openai_default(self):
        cfg = LLMConfig(
            name="my-custom-local-model",
            models=[LLMModelConfig(name="my-custom-local-model")],
        )
        assert cfg.models[0].api_base == _OPENAI_DEFAULT_API_BASE

    def test_mixed_providers_with_local_api_base(self):
        cfg = LLMConfig(
            api_base="http://localhost:11434/v1",
            models=[
                LLMModelConfig(name="anthropic/claude-3-sonnet"),
                LLMModelConfig(name="my-local-model"),
            ],
        )
        assert cfg.models[0].api_base == "https://api.anthropic.com/v1/"
        assert cfg.models[1].api_base == "http://localhost:11434/v1"


class TestOpenAILLMParams:
    def _make_llm(
        self,
        temperature=0.7,
        top_p=0.95,
        input_price_per_million_tokens=None,
        output_price_per_million_tokens=None,
    ):
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
            input_price_per_million_tokens=input_price_per_million_tokens,
            output_price_per_million_tokens=output_price_per_million_tokens,
        )
        with patch("skydiscover.llm.openai.openai.OpenAI"):
            llm = OpenAILLM(cfg)
        return llm

    @pytest.mark.anyio
    async def test_params_include_temperature_and_top_p(self):
        llm = self._make_llm(temperature=0.5, top_p=0.9)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(
            system_message="sys",
            messages=[{"role": "user", "content": "user"}],
            temperature=0.5,
            top_p=0.9,
        )
        params = llm._call_api.call_args[0][0]
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9

    @pytest.mark.anyio
    async def test_params_exclude_none_top_p(self):
        llm = self._make_llm(top_p=None)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])
        params = llm._call_api.call_args[0][0]
        assert "top_p" not in params
        assert "temperature" in params

    @pytest.mark.anyio
    async def test_params_exclude_none_temperature(self):
        llm = self._make_llm(temperature=None)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])
        params = llm._call_api.call_args[0][0]
        assert "temperature" not in params
        assert "top_p" in params

    @pytest.mark.anyio
    async def test_params_exclude_both_none(self):
        llm = self._make_llm(temperature=None, top_p=None)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])
        params = llm._call_api.call_args[0][0]
        assert "temperature" not in params
        assert "top_p" not in params

    @pytest.mark.anyio
    async def test_call_api_includes_usage_and_costs(self):
        llm = self._make_llm(
            input_price_per_million_tokens=2.5,
            output_price_per_million_tokens=10.0,
        )
        response = SimpleNamespace(
            model="provider-model",
            choices=[SimpleNamespace(message=SimpleNamespace(content="response"))],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=250, total_tokens=1250),
        )
        llm.client.chat.completions.create.return_value = response

        result = await llm._call_api({"model": "test-model", "messages": []})

        assert result.text == "response"
        assert result.model_name == "provider-model"
        assert result.input_tokens == 1000
        assert result.output_tokens == 250
        assert result.total_tokens == 1250
        assert result.input_cost_usd == pytest.approx(0.0025)
        assert result.output_cost_usd == pytest.approx(0.0025)
        assert result.total_cost_usd == pytest.approx(0.0050)


class TestLLMPoolCostTracking:
    @pytest.mark.anyio
    async def test_generate_records_usage_in_shared_tracker(self):
        tracker = CostTracker()
        cfg = LLMModelConfig(name="test-model", api_base="http://localhost:1234/v1", api_key="fake")
        with patch("skydiscover.llm.openai.openai.OpenAI"):
            pool = LLMPool([cfg], cost_tracker=tracker, usage_category="generation")

        pool.models[0].generate = AsyncMock(
            return_value=LLMResponse(
                text="ok",
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                input_cost_usd=0.001,
                output_cost_usd=0.002,
                total_cost_usd=0.003,
            )
        )

        await pool.generate("sys", [{"role": "user", "content": "user"}])
        summary = tracker.snapshot()

        assert summary["total"]["call_count"] == 1
        assert summary["by_category"]["generation"]["input_tokens"] == 100
        assert summary["by_category"]["generation"]["output_tokens"] == 20
        assert summary["total"]["total_cost_usd"] == pytest.approx(0.003)


class TestAgenticGeneratorUsage:
    @pytest.mark.anyio
    async def test_generate_returns_aggregated_llm_response(self):
        with TemporaryDirectory() as tmpdir:
            generator = AgenticGenerator(
                llm_pool=SimpleNamespace(),
                config=AgenticConfig(codebase_root=tmpdir, max_steps=1),
            )
            generator._call_llm = AsyncMock(
                return_value=(
                    {"role": "assistant", "content": "final answer"},
                    LLMResponse(
                        text="final answer",
                        model_name="test-model",
                        usage_source="api",
                        input_tokens=100,
                        output_tokens=25,
                        total_tokens=125,
                        input_cost_usd=0.001,
                        output_cost_usd=0.002,
                        total_cost_usd=0.003,
                    ),
                )
            )

            result = await generator.generate("sys", "user")

        assert result is not None
        assert result.text == "final answer"
        assert result.model_name == "test-model"
        assert result.input_tokens == 100
        assert result.output_tokens == 25
        assert result.total_cost_usd == pytest.approx(0.003)


class TestDefaultPricingLookup:
    def test_exact_match(self):
        inp, out = lookup_default_pricing("gpt-5")
        assert inp == 1.25
        assert out == 10.0

    def test_prefix_match_dated_snapshot(self):
        inp, out = lookup_default_pricing("gpt-5-20260301")
        assert inp == 1.25
        assert out == 10.0

    def test_prefix_match_longest_wins(self):
        inp, out = lookup_default_pricing("gpt-5-mini")
        assert inp == 0.25
        assert out == 2.0

    def test_gemini_via_litellm_prefix(self):
        inp, out = lookup_default_pricing("gemini/gemini-3.1-pro")
        assert inp == 2.0
        assert out == 12.0

    def test_claude_model(self):
        inp, out = lookup_default_pricing("claude-sonnet-4-6")
        assert inp == 3.0
        assert out == 15.0

    def test_unknown_model_returns_none(self):
        inp, out = lookup_default_pricing("unknown-model-xyz")
        assert inp is None
        assert out is None

    def test_case_insensitive(self):
        inp, out = lookup_default_pricing("GPT-5")
        assert inp == 1.25

    def test_auto_resolve_in_openai_llm(self):
        """OpenAILLM should auto-resolve pricing from the default table."""
        cfg = LLMModelConfig(
            name="gpt-5",
            api_base="https://api.openai.com/v1",
            api_key="fake",
        )
        with patch("skydiscover.llm.openai.openai.OpenAI"):
            from skydiscover.llm.openai import OpenAILLM

            llm = OpenAILLM(cfg)
        assert llm.input_price_per_million_tokens == 1.25
        assert llm.output_price_per_million_tokens == 10.0

    def test_gemini_2_5_flash_in_table(self):
        inp, out = lookup_default_pricing("gemini-2.5-flash")
        assert inp == 0.15
        assert out == 0.6

    def test_gemini_2_5_pro_in_table(self):
        inp, out = lookup_default_pricing("gemini-2.5-pro")
        assert inp == 1.25
        assert out == 10.0

    def test_thinking_tokens_added_to_output(self):
        """Gemini thinking tokens (total > input + output) should be added to output."""
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=1, total_tokens=104)
        inp, out, total = extract_usage_counts(usage)
        assert inp == 11
        assert out == 93  # 1 + 92 thinking tokens
        assert total == 104

    def test_no_thinking_gap_when_totals_match(self):
        """Normal usage: no gap, no adjustment."""
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        inp, out, total = extract_usage_counts(usage)
        assert inp == 100
        assert out == 50
        assert total == 150

    def test_explicit_config_overrides_default(self):
        """Explicit config prices should override the default table."""
        cfg = LLMModelConfig(
            name="gpt-5",
            api_base="https://api.openai.com/v1",
            api_key="fake",
            input_price_per_million_tokens=99.0,
            output_price_per_million_tokens=199.0,
        )
        with patch("skydiscover.llm.openai.openai.OpenAI"):
            from skydiscover.llm.openai import OpenAILLM

            llm = OpenAILLM(cfg)
        assert llm.input_price_per_million_tokens == 99.0
        assert llm.output_price_per_million_tokens == 199.0
