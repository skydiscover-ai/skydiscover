"""Shared utilities for the OpenAI Responses API.

Provides message conversion and output extraction helpers used by both
the non-agentic path (openai.py) and the agentic path (agentic_generator.py).
"""

from typing import Any, Dict, List, Optional, Tuple


def convert_messages_to_responses_input(messages: List[Dict[str, Any]]) -> list:
    """Convert Chat Completions-style messages to Responses API input format.

    Handles:
    - user / assistant text messages (plain string or multipart content)
    - assistant messages with tool_calls -> function_call items
    - tool role messages -> function_call_output items
    """
    items: list = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": content if isinstance(content, str) else "",
                }
            )
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}"),
                        }
                    )
                # If assistant had both text and tool_calls, skip the text
                # (Responses API treats function_call items as the assistant turn)
                if not content:
                    continue

        # Text-only message (user, assistant without tool_calls, or system)
        if isinstance(content, str):
            items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                }
            )
        elif isinstance(content, list):
            parts = []
            for part in content:
                ptype = part.get("type", "")
                if ptype == "text":
                    parts.append({"type": "input_text", "text": part["text"]})
                elif ptype == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    parts.append({"type": "input_image", "image_url": url, "detail": "auto"})
            items.append({"type": "message", "role": role, "content": parts})

    return items


def extract_responses_output(
    response,
) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
    """Extract text, image, and tool calls from a Responses API response.

    Returns:
        (text, image_b64, tool_calls) where tool_calls is a list of
        Chat-Completions-compatible tool call dicts (may be empty).
    """
    text_parts: List[str] = []
    image_b64: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []

    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if hasattr(part, "text"):
                    text_parts.append(part.text)
        elif item.type == "image_generation_call":
            if item.result:
                image_b64 = item.result
        elif item.type == "function_call":
            tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {"name": item.name, "arguments": item.arguments},
                }
            )

    return "\n".join(text_parts), image_b64, tool_calls
