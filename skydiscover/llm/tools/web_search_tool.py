"""DuckDuckGo HTML web search tool.

This mirrors Claw Code's Rust WebSearch behavior: fetch DuckDuckGo's HTML
endpoint, extract result links, optionally filter domains, and return a
JSON payload the model can cite.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, parse_qs, urlencode, urlparse, urlunparse

import requests

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_URL = "https://html.duckduckgo.com/html/"
WEB_SEARCH_BASE_URL_ENV = "CLAWD_WEB_SEARCH_BASE_URL"
AUTO_FETCH_COUNT = 3
AUTO_FETCH_PREVIEW_CHARS = 3000
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT_SECONDS = 20
MAX_RESULTS = 8


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str

    def as_json(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url}


class _AnchorParser(HTMLParser):
    def __init__(self, *, require_result_class: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.require_result_class = require_result_class
        self.hits: list[tuple[str, str]] = []
        self._active_href: str | None = None
        self._active_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {key.lower(): value or "" for key, value in attrs}
        href = attr_map.get("href")
        if not href:
            return
        if self.require_result_class and "result__a" not in attr_map.get("class", ""):
            return
        self._active_href = href
        self._active_text = []

    def handle_data(self, data: str) -> None:
        if self._active_href is not None:
            self._active_text.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._active_href is not None:
            self._active_text.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._active_href is not None:
            self._active_text.append(f"&#{name};")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._active_href is None:
            return
        title = collapse_whitespace(html.unescape("".join(self._active_text))).strip()
        self.hits.append((self._active_href, title))
        self._active_href = None
        self._active_text = []


def build_search_url(query: str) -> str:
    base = os.environ.get(WEB_SEARCH_BASE_URL_ENV, DEFAULT_SEARCH_URL)
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid search base URL: {base}")

    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query_pairs.append(("q", query))
    return urlunparse(parsed._replace(query=urlencode(query_pairs)))


def collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def decode_duckduckgo_redirect(url: str) -> str | None:
    if url.startswith("http://") or url.startswith("https://"):
        return html.unescape(url)
    if url.startswith("//"):
        joined = f"https:{url}"
    elif url.startswith("/"):
        joined = f"https://duckduckgo.com{url}"
    else:
        return None

    parsed = urlparse(joined)
    if parsed.path in {"/l", "/l/"}:
        uddg = parse_qs(parsed.query).get("uddg", [])
        if uddg:
            return html.unescape(uddg[0])
    return joined


def _extract_links(search_html: str, *, require_result_class: bool) -> list[SearchHit]:
    parser = _AnchorParser(require_result_class=require_result_class)
    parser.feed(search_html)

    hits: list[SearchHit] = []
    for raw_url, title in parser.hits:
        if not title:
            continue
        decoded_url = decode_duckduckgo_redirect(raw_url)
        if decoded_url and (
            decoded_url.startswith("http://") or decoded_url.startswith("https://")
        ):
            hits.append(SearchHit(title=title, url=decoded_url))
    return hits


def extract_search_hits(search_html: str) -> list[SearchHit]:
    return _extract_links(search_html, require_result_class=True)


def extract_search_hits_from_generic_links(search_html: str) -> list[SearchHit]:
    return _extract_links(search_html, require_result_class=False)


def normalize_domain_filter(domain: str) -> str:
    trimmed = domain.strip()
    parsed = urlparse(trimmed)
    candidate = parsed.hostname if parsed.scheme and parsed.hostname else trimmed
    return candidate.strip().lstrip(".").rstrip("/").lower()


def host_matches_list(url: str, domains: list[str]) -> bool:
    host = urlparse(url).hostname
    if not host:
        return False
    normalized_host = host.lower()
    for domain in domains:
        normalized = normalize_domain_filter(domain)
        if normalized and (
            normalized_host == normalized or normalized_host.endswith(f".{normalized}")
        ):
            return True
    return False


def dedupe_hits(hits: list[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    deduped: list[SearchHit] = []
    for hit in hits:
        if hit.url in seen:
            continue
        seen.add(hit.url)
        deduped.append(hit)
    return deduped


_SKIP_URL_PATTERNS = ("duckduckgo.com/y.js", "duckduckgo.com/l/")


def _is_fetchable_hit(hit: SearchHit) -> bool:
    """Skip ad redirects and non-content URLs."""
    return not any(pat in hit.url for pat in _SKIP_URL_PATTERNS)


def _robust_html_to_text(raw_html: str) -> str:
    """Extract text from HTML, with a regex fallback for stubborn pages."""
    from skydiscover.llm.tools.fetch_webpage_tool import html_to_text

    text = html_to_text(raw_html)
    if len(text.strip()) >= 100:
        return text
    # Fallback: strip tags with regex, collapse whitespace
    stripped = re.sub(r"<script[^>]*>.*?</script>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    stripped = re.sub(r"<style[^>]*>.*?</style>", " ", stripped, flags=re.DOTALL | re.IGNORECASE)
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    stripped = html.unescape(stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped


def _auto_fetch_top_hits(
    hits: list[SearchHit],
    max_fetch: int = AUTO_FETCH_COUNT,
    max_chars: int = AUTO_FETCH_PREVIEW_CHARS,
) -> str:
    """Fetch the top search result pages and return truncated text previews."""
    fetchable = [h for h in hits if _is_fetchable_hit(h)]
    previews: list[str] = []
    for hit in fetchable:
        if len(previews) >= max_fetch:
            break
        try:
            resp = requests.get(
                hit.url,
                headers={"User-Agent": USER_AGENT},
                timeout=15,
                allow_redirects=True,
            )
            if resp.status_code != 200:
                continue
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" in content_type or "octet-stream" in content_type:
                continue
            if "html" in content_type or not content_type:
                text = _robust_html_to_text(resp.text)
            else:
                text = resp.text
            text = text.strip()
            if len(text) < 200 or "captcha" in text.lower() or "checking your browser" in text.lower():
                continue
            if len(text) > max_chars:
                text = text[:max_chars] + "\n...(truncated)"
            previews.append(f"### {hit.title}\nURL: {hit.url}\n\n{text}")
        except Exception as exc:
            logger.debug("auto-fetch failed for %s: %s", hit.url, exc)
    return "\n\n---\n\n".join(previews)


def _fetch_search_page(search_url: str) -> requests.Response:
    return requests.get(
        search_url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
        allow_redirects=True,
    )


def _s2_fallback_search(query: str, limit: int = 10) -> list[SearchHit]:
    """Fallback: search Semantic Scholar when DuckDuckGo is unavailable."""
    try:
        import httpx

        resp = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/search/bulk",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,externalIds,year,citationCount,venue,publicationDate",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        papers = resp.json().get("data", [])
        hits: list[SearchHit] = []
        for p in papers:
            title = p.get("title", "")
            eids = p.get("externalIds") or {}
            doi = eids.get("DOI")
            arxiv = eids.get("ArXiv")
            if doi:
                url = f"https://doi.org/{doi}"
            elif arxiv:
                url = f"https://arxiv.org/abs/{arxiv}"
            else:
                s2_id = p.get("paperId", "")
                url = f"https://www.semanticscholar.org/paper/{s2_id}" if s2_id else ""
            year = p.get("year", "")
            venue = p.get("venue", "")
            suffix = f" ({venue}, {year})" if venue and year else f" ({year})" if year else ""
            if url:
                hits.append(SearchHit(title=f"{title}{suffix}", url=url))
        return hits
    except Exception as exc:
        logger.debug("S2 fallback search failed: %s", exc)
        return []


def execute_web_search(
    query: str,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    tool_use_id: str = "web_search_1",
) -> dict[str, Any]:
    started = time.monotonic()
    search_url = build_search_url(query)

    hits: list[SearchHit] = []
    for attempt in range(2):
        response = _fetch_search_page(search_url)
        hits = extract_search_hits(response.text)
        if hits:
            break
        if attempt == 0:
            time.sleep(1)

    if not hits and urlparse(response.url or search_url).hostname:
        hits = extract_search_hits_from_generic_links(response.text)

    # Filter out DDG's own pages (homepage, redirects) that aren't real results.
    hits = [h for h in hits if not h.url.rstrip("/").endswith("duckduckgo.com/html")]

    # Fallback to Semantic Scholar academic search when DDG returns nothing.
    used_s2_fallback = False
    if not hits:
        logger.info("web_search: DDG returned no results, falling back to Semantic Scholar")
        hits = _s2_fallback_search(query, limit=MAX_RESULTS)
        used_s2_fallback = bool(hits)

    if allowed_domains is not None:
        hits = [hit for hit in hits if host_matches_list(hit.url, allowed_domains)]
    if blocked_domains is not None:
        hits = [hit for hit in hits if not host_matches_list(hit.url, blocked_domains)]

    hits = dedupe_hits(hits)[:MAX_RESULTS]
    logger.info(
        "web_search: query=%r results=%d%s (%.1fs)",
        query,
        len(hits),
        " [S2 fallback]" if used_s2_fallback else "",
        time.monotonic() - started,
    )

    # Auto-fetch top results to give the agent actual content
    page_previews = _auto_fetch_top_hits(hits, max_fetch=AUTO_FETCH_COUNT)

    rendered_hits = "\n".join(f"- [{hit.title}]({hit.url})" for hit in hits)
    if hits:
        source_note = " (via Semantic Scholar academic search)" if used_s2_fallback else ""
        summary = (
            f"Search results for {query!r}{source_note}. Include a Sources section in the final answer.\n"
            f"{rendered_hits}"
        )
        if page_previews:
            summary += "\n\n--- Fetched page previews ---\n" + page_previews
    else:
        summary = f"No web search results matched the query {query!r}."

    return {
        "query": query,
        "results": [
            summary,
            {
                "tool_use_id": tool_use_id,
                "content": [hit.as_json() for hit in hits],
            },
        ],
        "durationSeconds": time.monotonic() - started,
    }


WEB_SEARCH_TOOL_SPEC = {
    "name": "web_search",
    "description": "Search the web for current information and return cited results.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 2},
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional allowlist of domains or URLs. Subdomains match.",
            },
            "blocked_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional blocklist of domains or URLs. Subdomains match.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def _optional_string_list(arguments: dict[str, Any], key: str) -> list[str] | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be an array of strings")
    return value


async def web_search_handler(
    arguments: dict[str, Any],
    session: Any = None,
    tool_call_id: str | None = None,
    **_kw: Any,
) -> tuple[str, bool]:
    query_value = arguments.get("query", "")
    if not isinstance(query_value, str):
        return "Error: web_search requires a query string with at least 2 characters.", False

    query = query_value.strip()
    if len(query) < 2:
        return "Error: web_search requires a query with at least 2 characters.", False

    try:
        output = await asyncio.to_thread(
            execute_web_search,
            query=query,
            allowed_domains=_optional_string_list(arguments, "allowed_domains"),
            blocked_domains=_optional_string_list(arguments, "blocked_domains"),
            tool_use_id=tool_call_id or "web_search_1",
        )
    except Exception as exc:
        return f"Error executing web search: {exc}", False

    return json.dumps(output, indent=2), True
