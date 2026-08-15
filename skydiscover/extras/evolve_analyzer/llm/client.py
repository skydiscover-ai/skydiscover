# Vendored from clear_eval/pipeline/inference_utils/llm_client.py
# (CLEAR commit 740bb0c49d782d2e49e9aa3fddabf8378ba88554, 2026-04-16).
# Maintained independently within evolve-analyzer.
# Adapted: LangChain backend removed (not in evolve-analyzer dependencies).
# LiteLLM is the primary backend; direct OpenAI-compatible endpoint is secondary.
# To incorporate upstream improvements, diff manually against the CLEAR source.

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Module-level event loop — must be created before importing litellm
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        try:
            _event_loop = asyncio.get_running_loop()
        except RuntimeError:
            _event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_event_loop)
    return _event_loop


_get_or_create_event_loop()

import litellm  # noqa: E402 — must follow event loop init

litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm._logging._disable_debugging()
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)


@dataclass
class ParallelResult:
    is_success: bool
    result: Optional[Any] = None
    error: Optional[str] = None


def normalize_messages(
    messages: Union[str, List[Dict[str, str]]]
) -> List[Dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return messages


class LLMClient(ABC):
    @abstractmethod
    def invoke(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> str: ...

    async def ainvoke(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        return await asyncio.to_thread(self.invoke, messages, **kwargs)

    def disable_temperature(self) -> None:
        pass


class LiteLLMClient(LLMClient):
    """Primary backend — supports OpenAI, Anthropic, and 100+ providers via LiteLLM."""

    def __init__(
        self,
        provider: str,
        model: str,
        eval_mode: bool = True,
        max_retries: int = 3,
        **params,
    ):
        self.provider = provider
        self.model = model
        self.eval_mode = eval_mode
        self.max_retries = max_retries
        self.params = params
        self._litellm_model = f"{provider}/{model}"
        self._configure_provider()

    def _configure_provider(self) -> None:
        if self.provider == "openai":
            if "api_base" not in self.params and not os.getenv("OPENAI_API_KEY"):
                raise KeyError(
                    "OPENAI_API_KEY env var required (or pass api_base for local endpoints)."
                )
        elif self.provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise KeyError("ANTHROPIC_API_KEY env var required.")
        # Other providers: trust user credentials per litellm docs.

    def disable_temperature(self) -> None:
        self.eval_mode = False

    def _call_params(self, **kwargs) -> dict:
        params = {**self.params, **kwargs}
        if self.eval_mode and "temperature" not in params:
            params["temperature"] = 0
        params["num_retries"] = self.max_retries
        return params

    def invoke(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        from litellm import completion

        response = completion(
            model=self._litellm_model,
            messages=normalize_messages(messages),
            **self._call_params(**kwargs),
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    async def ainvoke(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        from litellm import acompletion

        response = await acompletion(
            model=self._litellm_model,
            messages=normalize_messages(messages),
            **self._call_params(**kwargs),
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


class EndpointClient(LLMClient):
    """Direct OpenAI-compatible HTTP endpoint (Ollama, vLLM, Azure, etc.)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-required",
        eval_mode: bool = True,
        **params,
    ):
        from openai import OpenAI

        self.model = model
        self.eval_mode = eval_mode
        self.params = params
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def disable_temperature(self) -> None:
        self.eval_mode = False

    def _call_params(self, **kwargs) -> dict:
        params = {**self.params, **kwargs}
        if self.eval_mode and "temperature" not in params:
            params["temperature"] = 0
        return params

    def invoke(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=normalize_messages(messages),
            **self._call_params(**kwargs),
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


_TEMPERATURE_ERROR_KEYWORDS = ("temperature", "unsupported parameter", "not supported")


def _validate_temperature_support(client: LLMClient) -> None:
    probe = [{"role": "user", "content": "Say OK"}]
    try:
        client.invoke(probe)
    except Exception as e:
        if any(kw in str(e).lower() for kw in _TEMPERATURE_ERROR_KEYWORDS):
            logger.warning("Model does not support temperature=0 — disabling.")
            client.disable_temperature()
            client.invoke(probe)
        else:
            raise


def get_llm_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    api_key_env: str = "EVOLVE_ANALYZER_API_KEY",
    eval_mode: bool = True,
    parameters: Optional[Dict] = None,
) -> LLMClient:
    """
    Factory for LLM clients.

    - If base_url is provided → EndpointClient (OpenAI-compatible HTTP endpoint).
    - Otherwise → LiteLLMClient (openai, anthropic, and 100+ providers).

    The API key is read from the environment variable named by api_key_env.
    For OpenAI-compatible endpoints that don't require auth, api_key_env may
    point to an unset var — a placeholder is used automatically.
    """
    parameters = parameters or {}

    try:
        if base_url:
            api_key = os.getenv(api_key_env, "not-required")
            client: LLMClient = EndpointClient(
                base_url=base_url,
                model=model,
                api_key=api_key,
                eval_mode=eval_mode,
                **parameters,
            )
        else:
            # Inject the API key into the environment so litellm picks it up.
            api_key = os.getenv(api_key_env)
            if api_key:
                provider_env = {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                }.get(provider)
                if provider_env and not os.getenv(provider_env):
                    os.environ[provider_env] = api_key

            client = LiteLLMClient(
                provider=provider,
                model=model,
                eval_mode=eval_mode,
                **parameters,
            )

        logger.info(f"Initialized LLM client: provider={provider}, model={model}")
    except Exception as e:
        raise RuntimeError(f"Error initializing LLM ({provider}, {model}): {e}") from e

    if eval_mode:
        _validate_temperature_support(client)

    return client
