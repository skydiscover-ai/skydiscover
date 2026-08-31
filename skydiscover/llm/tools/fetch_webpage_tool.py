"""fetch_webpage tool – download a URL and save it as a local text file.

The agent calls this tool with a URL (typically from web_search results).
The tool:
  1. Fetches the page (with a sensible timeout + User-Agent).
  2. Extracts readable text from HTML (strips scripts/styles/nav).
  3. Saves the result as  <codebase_root>/reference/web_<slug>.txt
  4. Returns the saved path so the agent can subsequently call read_file.

The saved file is plain text, line-wrapped, and stays within a character
budget so the agent's context does not blow up.
"""

from __future__ import annotations

import asyncio
import hashlib
import html as html_module
import json
import logging
import os
import re
import time
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# Directory (relative to codebase root) where pages are saved.
REFERENCE_DIR = "reference"


# ---------------------------------------------------------------------------
# HTML → plain-text extraction
# ---------------------------------------------------------------------------

# Tags whose entire subtree we skip (content not useful for LLMs).
_SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "head",
        "header",
        "footer",
        "nav",
        "aside",
        "form",
        "button",
        "svg",
        "img",
        "figure",
        "figcaption",
        "iframe",
        "meta",
        "link",
        "input",
        "select",
        "textarea",
        "label",
    }
)

# Block-level tags that get a blank line before/after.
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "dt",
        "dd",
        "blockquote",
        "pre",
        "code",
        "tr",
        "td",
        "th",
        "br",
        "hr",
    }
)


class _TextExtractor(HTMLParser):
    """Extract plain text from HTML, skipping non-content subtrees."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth: int = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if self._skip_depth > 0 or tag_lower in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag_lower in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag_lower in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of blank lines to at most 2.
        cleaned = re.sub(r"\n{3,}", "\n\n", raw)
        # Strip trailing whitespace on each line.
        lines = [line.rstrip() for line in cleaned.splitlines()]
        return "\n".join(lines).strip()


def html_to_text(html_content: str) -> str:
    """Convert HTML to clean plain text."""
    extractor = _TextExtractor()
    try:
        extractor.feed(html_content)
    except Exception:
        # If the parser chokes, fall back to stripping all tags.
        return re.sub(r"<[^>]+>", " ", html_content)
    return extractor.get_text()


# ---------------------------------------------------------------------------
# URL → slug (for filename)
# ---------------------------------------------------------------------------

def _url_to_slug(url: str, max_len: int = 60) -> str:
    """Turn a URL into a short, filesystem-safe slug."""
    parsed = urlparse(url)
    # Use host + path, strip scheme and query.
    raw = (parsed.netloc + parsed.path).lower()
    raw = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    if len(raw) > max_len:
        # Keep the end (usually most specific) and prefix with a short hash.
        h = hashlib.md5(url.encode()).hexdigest()[:6]
        raw = h + "_" + raw[-(max_len - 7):]
    return raw or hashlib.md5(url.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Core fetch + save logic
# ---------------------------------------------------------------------------

def fetch_and_save_webpage(url: str, codebase_root: str) -> dict[str, Any]:
    """
    Fetch *url*, convert to plain text, save under <codebase_root>/reference/,
    and return a result dict with keys:
      saved_path  – path relative to codebase root (for read_file)
      abs_path    – absolute path on disk
      url         – the fetched URL
      char_count  – characters saved
      truncated   – whether the content was truncated
      error       – error message if fetch/parse failed (absent on success)
    """
    started = time.monotonic()

    # ---- Fetch ----
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return {"error": f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s: {url}"}
    except requests.exceptions.RequestException as exc:
        return {"error": f"HTTP error fetching {url}: {exc}"}

    content_type = resp.headers.get("Content-Type", "")
    is_html = "html" in content_type or not content_type

    # ---- Extract text ----
    if is_html:
        text = html_to_text(resp.text)
    else:
        # For PDFs, plain text, etc., just use the raw text as-is.
        text = resp.text

    # ---- Build header ----
    header_lines = [
        f"# Fetched page: {url}",
        f"# Content-Type: {content_type}",
        f"# Fetched at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"# Characters: {len(text)}",
        "",
    ]
    full_content = "\n".join(header_lines) + text

    # ---- Save to disk ----
    ref_dir = os.path.join(codebase_root, REFERENCE_DIR)
    os.makedirs(ref_dir, exist_ok=True)

    slug = _url_to_slug(url)
    filename = f"web_{slug}.txt"
    abs_path = os.path.join(ref_dir, filename)
    rel_path = os.path.join(REFERENCE_DIR, filename)

    try:
        with open(abs_path, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(full_content)
    except OSError as exc:
        return {"error": f"Failed to write {abs_path}: {exc}"}

    elapsed = time.monotonic() - started
    logger.info(
        "fetch_webpage: saved %s → %s (%d chars, %.1fs)",
        url,
        rel_path,
        len(full_content),
        elapsed,
    )

    return {
        "saved_path": rel_path,
        "abs_path": abs_path,
        "url": url,
        "char_count": len(full_content),
        "duration_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Tool handler (async, called from agentic_generator._run_tool)
# ---------------------------------------------------------------------------

async def fetch_webpage_handler(
    arguments: dict[str, Any],
    codebase_root: str,
    **_kw: Any,
) -> tuple[str, bool]:
    """Async handler called by the agentic loop."""
    url = arguments.get("url", "").strip()
    if not url:
        return "Error: fetch_webpage requires a 'url' argument.", False
    if not (url.startswith("http://") or url.startswith("https://")):
        return "Error: URL must start with http:// or https://", False
    if not codebase_root:
        return "Error: codebase_root not configured – cannot save file.", False

    result = await asyncio.to_thread(fetch_and_save_webpage, url, codebase_root)

    if "error" in result:
        return f"Error fetching webpage: {result['error']}", False

    msg = (
        f"Webpage saved successfully.\n"
        f"  URL:         {result['url']}\n"
        f"  Saved to:    {result['saved_path']}\n"
        f"  Characters:  {result['char_count']}\n"
        f"  Duration:    {result['duration_seconds']:.1f}s\n\n"
        f"Use read_file with path='{result['saved_path']}' to read the content."
    )
    return msg, True
