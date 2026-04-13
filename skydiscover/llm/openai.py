"""OpenAI-compatible LLM backend (Chat Completions + Responses API)."""

import asyncio
import base64
import logging
import os
import tempfile
import uuid as _uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import openai

from skydiscover.config import LLMModelConfig
from skydiscover.llm.base import LLMInterface, LLMResponse
from skydiscover.llm.responses_utils import (
    convert_messages_to_responses_input,
    extract_responses_output,
)

logger = logging.getLogger("skydiscover.llm")

REASONING_MODEL_PREFIXES = (
    "o1-",
    "o1",
    "o3-",
    "o3",
    "o4-",
    "gpt-5-",
    "gpt-5",
    "gpt-oss-120b",
    "gpt-oss-20b",
)

GOOGLE_AI_STUDIO_DOMAIN = "generativelanguage.googleapis.com"

_OPENAI_API_PREFIXES = (
    "https://api.openai.com",
    "https://eu.api.openai.com",
    "https://apac.api.openai.com",
)


def is_openai_reasoning_model(model_name: str, api_base: str) -> bool:
    """Check if a model is an OpenAI reasoning model requiring special parameters."""
    api_base_lower = (api_base or "").lower()
    is_openai_api = (
        any(api_base_lower.startswith(p) for p in _OPENAI_API_PREFIXES)
        or ".openai.azure.com" in api_base_lower
    )
    return is_openai_api and model_name.lower().startswith(REASONING_MODEL_PREFIXES)


class OpenAILLM(LLMInterface):
    """LLM backend using OpenAI-compatible APIs (Chat Completions + Responses)."""

    def __init__(self, model_cfg: Optional[LLMModelConfig] = None):
        self.model = model_cfg.name
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)

        max_retries = self.retries if self.retries is not None else 0
        is_azure = self.api_base and ".openai.azure.com" in self.api_base.lower()

        if is_azure:
            parsed_url = urlparse(self.api_base)
            azure_endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"
            query_params = parse_qs(parsed_url.query)
            api_version = query_params.get("api-version", ["2024-12-01-preview"])[0]

            self.client = openai.AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=self.api_key,
                api_version=api_version,
                timeout=self.timeout,
                max_retries=max_retries,
            )
        else:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=max_retries,
            )

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()
        if self.model not in logger._initialized_models:
            api_base_str = (self.api_base or "").lower()
            if is_azure:
                provider = "AzureOpenAI"
            elif GOOGLE_AI_STUDIO_DOMAIN in api_base_str:
                provider = "Gemini"
            elif "api.anthropic.com" in api_base_str:
                provider = "Anthropic"
            elif "api.deepseek.com" in api_base_str:
                provider = "DeepSeek"
            elif "api.mistral.ai" in api_base_str:
                provider = "Mistral"
            else:
                provider = "OpenAI"
            logger.info(f"{provider} LLM: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        """Generate a response. Pass image_output=True for image generation."""
        if kwargs.get("image_output"):
            return await self._generate_with_image(system_message, messages, **kwargs)
        text = await self._generate_text(system_message, messages, **kwargs)
        return LLMResponse(text=text)

    # ------------------------------------------------------------------
    # Text generation (Chat Completions API)
    # ------------------------------------------------------------------

    async def _generate_text(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> str:
        system_content = system_message if system_message is not None else ""
        formatted_messages = [{"role": "system", "content": system_content}]
        formatted_messages.extend(messages)

        is_reasoning = is_openai_reasoning_model(self.model, self.api_base)

        if is_reasoning:
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort
            if "verbosity" in kwargs:
                params["verbosity"] = kwargs["verbosity"]
        else:
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            temperature = kwargs.get("temperature", self.temperature)
            if temperature is not None:
                params["temperature"] = temperature
            top_p = kwargs.get("top_p", self.top_p)
            if top_p is not None:
                params["top_p"] = top_p
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort

        retries, retry_delay, timeout = self._resolve_retry_options(**kwargs)

        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(self._call_api(params), timeout=timeout)
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout attempt {attempt + 1}/{retries + 1}, retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(f"Error attempt {attempt + 1}/{retries + 1}: {e}, retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None, lambda: self.client.chat.completions.create(**params)
            )
            return response.choices[0].message.content
        except (openai.BadRequestError, openai.APIStatusError) as exc:
            # Some Azure deployments only expose the Responses API.
            # Fall back transparently when Chat Completions is unsupported.
            if "unsupported" not in str(exc).lower() and "not found" not in str(exc).lower():
                raise
            logger.info("Chat Completions unsupported; falling back to Responses API")
            return await self._call_api_via_responses(params)

    async def _call_api_via_responses(self, params: Dict[str, Any]) -> str:
        """Translate a Chat-Completions-style *params* dict into a Responses API
        call and return the assistant text."""
        messages = params.get("messages", [])
        input_items = self._convert_to_responses_input(
            [m for m in messages if m.get("role") != "system"]
        )
        system_msg = next((m["content"] for m in messages if m.get("role") == "system"), None)
        resp_params: Dict[str, Any] = {
            "model": params.get("model", self.model),
            "input": input_items,
        }
        if system_msg:
            resp_params["instructions"] = system_msg
        if params.get("max_tokens"):
            resp_params["max_output_tokens"] = params["max_tokens"]
        if params.get("max_completion_tokens"):
            resp_params["max_output_tokens"] = params["max_completion_tokens"]
        if params.get("temperature") is not None:
            resp_params["temperature"] = params["temperature"]
        if params.get("reasoning_effort") is not None:
            resp_params["reasoning"] = {"effort": params["reasoning_effort"]}

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.responses.create(**resp_params)
        )
        text, _ = self._extract_responses_output(response)
        return text or ""

    def _resolve_retry_options(self, **kwargs) -> Tuple[int, int, int]:
        """Resolve retry/timeout options from kwargs, falling back to instance defaults."""
        retries = kwargs.get("retries", self.retries)
        if retries is None:
            retries = 0
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        if retry_delay is None:
            retry_delay = 2
        timeout = kwargs.get("timeout", self.timeout)
        if timeout is None:
            timeout = 300
        return retries, retry_delay, timeout

    # ------------------------------------------------------------------
    # Image generation (OpenAI Responses API)
    # ------------------------------------------------------------------

    async def _generate_with_image(
        self,
        system_message: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> LLMResponse:
        output_dir = kwargs.get("output_dir", tempfile.gettempdir())
        program_id = kwargs.get("program_id", "")

        input_items = convert_messages_to_responses_input(messages)

        params: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "tools": [
                {
                    "type": "image_generation",
                    "quality": kwargs.get("image_quality", "medium"),
                    "size": kwargs.get("image_size", "1024x1024"),
                    "output_format": "png",
                }
            ],
        }
        if system_message:
            params["instructions"] = system_message
        is_reasoning = self.model.lower().startswith(REASONING_MODEL_PREFIXES)
        if not is_reasoning and self.temperature is not None:
            params["temperature"] = kwargs.get("temperature", self.temperature)
        if self.max_tokens is not None:
            params["max_output_tokens"] = kwargs.get("max_tokens", self.max_tokens)

        retries, retry_delay, timeout = self._resolve_retry_options(**kwargs)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_responses_api(params), timeout=timeout)
                text, image_b64, _ = extract_responses_output(response)

                image_path = None
                if image_b64:
                    os.makedirs(output_dir, exist_ok=True)
                    fname = f"{program_id or _uuid.uuid4().hex[:12]}.png"
                    image_path = os.path.join(output_dir, fname)
                    with open(image_path, "wb") as f:
                        f.write(base64.b64decode(image_b64))
                    logger.info(f"Image saved: {image_path}")

                return LLMResponse(text=text, image_path=image_path)

            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(
                        f"Image timeout attempt {attempt + 1}/{retries + 1}, retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Image error attempt {attempt + 1}/{retries + 1}: {e}, retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    raise

    async def _call_responses_api(self, params: Dict[str, Any]):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.client.responses.create(**params))
