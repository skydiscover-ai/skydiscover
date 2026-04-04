"""Agentic code generator -- multi-turn tool-calling loop with read_file and search."""

import asyncio
import concurrent.futures
import fnmatch
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.llm.openai import is_openai_reasoning_model
from skydiscover.llm.responses_utils import (
    convert_messages_to_responses_input,
    extract_responses_output,
)
from skydiscover.utils.code_utils import build_repo_map

logger = logging.getLogger(__name__)

_TOOL_SCHEMAS_PATH = Path(__file__).parent / "tool_schemas" / "agentic_tools.json"
with open(_TOOL_SCHEMAS_PATH, "r") as _f:
    TOOL_SCHEMAS = json.load(_f)

# Responses API uses a flattened tool format (name/description/parameters at top level)
TOOL_SCHEMAS_RESPONSES = [
    {
        "type": "function",
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "parameters": t["function"]["parameters"],
    }
    for t in TOOL_SCHEMAS
]

_AGENTIC_PROMPT_PATH = (
    Path(__file__).parent.parent
    / "context_builder"
    / "default"
    / "templates"
    / "agentic_system_message.txt"
)
with open(_AGENTIC_PROMPT_PATH, "r") as _f:
    _AGENTIC_SYSTEM_PROMPT = _f.read()


class AgenticGenerator:
    """
    V0 [simple version]: Multi-turn tool-calling agent that explores a codebase before generating code.

    Tools: read_file, search. When it stops calling tools, its text output
    is the final answer. Returns None if no output is produced (caller falls
    back to direct generation).
    """

    def __init__(self, llm_pool, config):
        self.llm_pool = llm_pool
        self.config = config

    async def generate(self, system_message: str, user_message: str) -> Optional[str]:
        """Run the agent loop. Returns generated text, or None on failure."""
        cfg = self.config
        files_read: set = set()
        conversation: List[Dict[str, Any]] = []
        t0 = time.time()

        sys_prompt = f"{system_message}\n\n{_AGENTIC_SYSTEM_PROMPT}"
        repo_map = build_repo_map(
            cfg.codebase_root,
            max_depth=cfg.repo_map_max_depth,
            allowed_extensions=cfg.allowed_extensions,
            excluded_dirs=cfg.excluded_dirs,
        )

        user_parts = [user_message]
        if repo_map:
            user_parts.append(f"\n## Project structure\n```\n{repo_map}\n```")
        conversation.append({"role": "user", "content": "\n".join(user_parts)})

        for step in range(cfg.max_steps):
            if time.time() - t0 > cfg.overall_timeout:
                logger.warning("Agent timed out at step %d", step)
                break

            if _context_chars(sys_prompt, conversation) > cfg.max_context_chars:
                conversation.append(
                    {
                        "role": "user",
                        "content": "Context limit reached. Output your improved program now.",
                    }
                )

            try:
                assistant_msg = await asyncio.wait_for(
                    self._call_llm(sys_prompt, conversation),
                    timeout=cfg.per_step_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("Step %d: LLM timed out", step)
                conversation.append(
                    {
                        "role": "user",
                        "content": "Timed out. Output your solution or try a simpler action.",
                    }
                )
                continue
            except Exception as e:
                logger.error("Step %d: LLM error: %s", step, e)
                break

            tool_calls = assistant_msg.get("tool_calls", [])
            text_content = assistant_msg.get("content", "").strip()
            conversation.append(assistant_msg)

            if not tool_calls:
                if text_content:
                    logger.info(
                        "Agent produced text at step %d (%d files read)", step, len(files_read)
                    )
                    return text_content
                conversation.append(
                    {
                        "role": "user",
                        "content": "Use a tool to explore, or output your improved program.",
                    }
                )
                continue

            for tc in tool_calls:
                fn = tc.get("function", {})
                name, raw, tc_id = fn.get("name", ""), fn.get("arguments", "{}"), tc.get("id", "")

                try:
                    args = json.loads(raw)
                except (json.JSONDecodeError, TypeError) as e:
                    conversation.append(
                        {"role": "tool", "tool_call_id": tc_id, "content": f"Bad JSON: {e}"}
                    )
                    continue

                logger.info(
                    "Step %d: tool=%s args=%s",
                    step,
                    name,
                    {
                        k: (v[:80] + "...") if isinstance(v, str) and len(v) > 80 else v
                        for k, v in args.items()
                    },
                )

                result = self._run_tool(name, args, files_read)
                conversation.append(
                    {"role": "tool", "tool_call_id": tc_id, "content": result["content"]}
                )

        logger.warning("Agent loop ended without producing code")
        return None

    async def _call_llm(
        self, system_message: str, conversation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Call a sampled LLM with tool schemas.

        Tries Chat Completions first; falls back to Responses API if the
        deployment does not support Chat Completions (common on Azure).
        """
        model = self.llm_pool.models[
            self.llm_pool.random_state.choices(
                range(len(self.llm_pool.models)), weights=self.llm_pool.weights, k=1
            )[0]
        ]

        if not hasattr(model, "client"):
            raise RuntimeError(
                f"Agentic mode requires an OpenAI-compatible LLM ({type(model).__name__} has no .client)"
            )

        # If we already know this model needs the Responses API, skip Chat Completions
        if getattr(model, "_use_responses_api", False):
            return await self._call_llm_responses(model, system_message, conversation)

        messages = [{"role": "system", "content": system_message}] + conversation
        is_reasoning = is_openai_reasoning_model(model.model, getattr(model, "api_base", "") or "")

        params: Dict[str, Any] = {
            "model": model.model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
        }
        if is_reasoning:
            if model.max_tokens:
                params["max_completion_tokens"] = model.max_tokens
            if getattr(model, "reasoning_effort", None):
                params["reasoning_effort"] = model.reasoning_effort
        else:
            if model.temperature is not None:
                params["temperature"] = model.temperature
            if model.top_p is not None:
                params["top_p"] = model.top_p
            if model.max_tokens is not None:
                params["max_tokens"] = model.max_tokens

        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None, lambda: model.client.chat.completions.create(**params)
            )
        except Exception as exc:
            if "unsupported" not in str(exc).lower() and "not found" not in str(exc).lower():
                raise
            logger.info("Chat Completions unsupported for agentic; falling back to Responses API")
            model._use_responses_api = True
            return await self._call_llm_responses(model, system_message, conversation)

        msg = resp.choices[0].message
        out: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        return out

    async def _call_llm_responses(
        self,
        model,
        system_message: str,
        conversation: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Call the LLM via the Responses API (Azure-compatible) with tool support."""
        is_reasoning = is_openai_reasoning_model(model.model, getattr(model, "api_base", "") or "")

        input_items = convert_messages_to_responses_input(conversation)

        resp_params: Dict[str, Any] = {
            "model": model.model,
            "input": input_items,
            "instructions": system_message,
            "tools": TOOL_SCHEMAS_RESPONSES,
            "tool_choice": "auto",
        }
        if is_reasoning:
            if model.max_tokens:
                resp_params["max_output_tokens"] = model.max_tokens
            if getattr(model, "reasoning_effort", None):
                resp_params["reasoning"] = {"effort": model.reasoning_effort}
        else:
            if model.temperature is not None:
                resp_params["temperature"] = model.temperature
            if model.max_tokens is not None:
                resp_params["max_output_tokens"] = model.max_tokens

        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, lambda: model.client.responses.create(**resp_params)
        )

        text, _, tool_calls = extract_responses_output(resp)
        out: Dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            out["tool_calls"] = tool_calls
        return out

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _run_tool(self, name: str, args: Dict[str, Any], files_read: set) -> Dict[str, Any]:
        try:
            if name == "read_file":
                return self._tool_read_file(args, files_read)
            elif name == "search":
                return self._tool_search(args)
            return _err(f"Unknown tool '{name}'. Available: read_file, search.")
        except Exception as e:
            return _err(f"Tool '{name}' error: {e}")

    def _tool_read_file(self, args: Dict[str, Any], files_read: set) -> Dict[str, Any]:
        path = args.get("path", "")
        if not path:
            return _err("'path' is required.")

        root = self.config.codebase_root
        if not root:
            return _err("codebase_root not configured.")
        full = os.path.join(root, path) if not os.path.isabs(path) else path

        ok, resolved, err = _validate_path(
            full, root, self.config.allowed_extensions, self.config.excluded_dirs
        )
        if not ok:
            return _err(err)

        if resolved not in files_read and len(files_read) >= self.config.max_files_read:
            return _err(f"Read limit ({self.config.max_files_read}). Output your solution.")

        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return _err(f"Cannot read: {e}")

        total = len(lines)
        start = max(1, int(args.get("line_start") or 1)) - 1
        end = min(total, int(args.get("line_end") or total))
        content = "".join(lines[start:end])

        if len(content) > self.config.max_file_chars:
            half = self.config.max_file_chars // 2
            content = (
                content[:half]
                + f"\n\n... ({len(content) - self.config.max_file_chars} chars truncated) ...\n\n"
                + content[-half:]
            )

        files_read.add(resolved)
        rel = os.path.relpath(resolved, root)
        numbered = [
            f"{i:4d} | {ln.rstrip(chr(10))}"
            for i, ln in enumerate(content.splitlines(True), start=start + 1)
        ]
        return {"content": f"{rel} (lines {start + 1}-{end} of {total})\n" + "\n".join(numbered)}

    def _tool_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        pattern = args.get("pattern", "")
        glob_pat = args.get("file_glob", "*.py")

        if not pattern:
            return _err("'pattern' is required.")
        if len(pattern) > self.config.max_regex_length:
            return _err(f"Pattern too long ({len(pattern)} > {self.config.max_regex_length}).")

        safety_err = _check_regex_safety(pattern)
        if safety_err:
            return _err(safety_err)

        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return _err(f"Invalid regex: {e}")

        root = self.config.codebase_root
        if not root:
            return _err("codebase_root not configured.")
        excluded = set(self.config.excluded_dirs)
        allowed = set(self.config.allowed_extensions)
        matches: List[str] = []
        n_files = 0
        max_results = self.config.max_search_results

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in excluded]
            for fname in filenames:
                if not fnmatch.fnmatch(fname, glob_pat):
                    continue
                if os.path.splitext(fname)[1].lower() not in allowed:
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    if os.path.getsize(fpath) > self.config.max_file_chars:
                        continue
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                except Exception:
                    continue

                n_files += 1
                ok, hits, err = _safe_regex_search(compiled, text, self.config.regex_timeout)
                if not ok:
                    return _err(err)

                rel = os.path.relpath(fpath, root)
                for hit in hits:
                    matches.append(f"{rel}:{hit}")
                    if len(matches) >= max_results:
                        break
                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break

        if not matches:
            return {"content": f"No matches for '{pattern}' in {n_files} files."}

        suffix = f"\n(capped at {max_results} results)" if len(matches) >= max_results else ""
        return {"content": "\n".join(matches) + suffix}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _err(msg: str) -> Dict[str, Any]:
    return {"content": msg, "_error": True}


def _context_chars(system: str, conversation: List[Dict[str, Any]]) -> int:
    n = len(system)
    for msg in conversation:
        n += len(msg.get("content", ""))
        for tc in msg.get("tool_calls", []):
            n += len(tc.get("function", {}).get("arguments", ""))
    return n


_SENSITIVE_FILENAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        ".env.staging",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
        "credentials.json",
        "credentials.yaml",
        "service-account.json",
        "service_account.json",
        ".netrc",
        ".pgpass",
        ".my.cnf",
    }
)


def _validate_path(
    requested: str, root: str, allowed_ext: tuple, excluded_dirs: tuple
) -> Tuple[bool, str, str]:
    """Validate a file path. Returns (ok, resolved_path, error_message)."""
    try:
        resolved = os.path.realpath(requested)
    except (OSError, ValueError) as e:
        return False, "", f"Invalid path: {e}"

    root_abs = os.path.realpath(root)
    if not resolved.startswith(root_abs + os.sep) and resolved != root_abs:
        return False, "", "Path outside codebase root."

    try:
        rel = os.path.relpath(resolved, root_abs)
        for part in Path(rel).parts:
            if part in excluded_dirs:
                return False, "", f"Path in excluded directory '{part}'."
    except ValueError:
        pass

    basename = os.path.basename(resolved).lower()
    if basename in _SENSITIVE_FILENAMES:
        return False, "", f"Access denied: '{basename}' may contain secrets."

    if not os.path.isfile(resolved):
        parent_dir = os.path.dirname(resolved)
        if os.path.isdir(parent_dir):
            try:
                siblings = sorted(os.listdir(parent_dir))[:15]
                rel_dir = os.path.relpath(parent_dir, root_abs)
                return (
                    False,
                    "",
                    f"Not found: '{os.path.basename(resolved)}'. '{rel_dir}/' contains: {siblings}",
                )
            except OSError:
                pass
        return False, "", f"File not found: '{requested}'."

    ext = os.path.splitext(resolved)[1].lower()
    if ext not in allowed_ext:
        return False, "", f"Extension '{ext}' not allowed."

    return True, resolved, ""


_NESTED_QUANTIFIER_RE = re.compile(r"\([^)]*[+*][^)]*\)\s*[+*?]|\([^)]*[+*][^)]*\)\s*\{")

_MAX_SEARCH_LINE_LEN = 2000


def _check_regex_safety(pattern: str) -> Optional[str]:
    """Reject patterns with nested quantifiers that cause catastrophic backtracking."""
    if _NESTED_QUANTIFIER_RE.search(pattern):
        return "Nested quantifiers detected (e.g. '(a+)+'). Use a simpler pattern."
    return None


_REGEX_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="regex")


def _safe_regex_search(
    compiled: "re.Pattern", text: str, timeout: float = 2.0
) -> Tuple[bool, List[str], str]:
    """Regex search with thread-based timeout."""

    def do_search():
        return [
            f"{i}: {line}"
            for i, line in enumerate(text.splitlines(), 1)
            if len(line) <= _MAX_SEARCH_LINE_LEN and compiled.search(line)
        ]

    fut = _REGEX_EXECUTOR.submit(do_search)
    try:
        result = fut.result(timeout=timeout)
        return True, result, ""
    except concurrent.futures.TimeoutError:
        return False, [], f"Regex timed out ({timeout}s). Simplify the pattern."
