# Disk cache for LLM responses — prevents redundant judge calls on re-runs.
# New implementation using diskcache (not vendored from CLEAR).
# CLEAR's caching_utils.py caches DataFrames/JSON for CLEAR's pipeline;
# this module caches LLM string responses keyed on prompt + model identity.

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import diskcache

logger = logging.getLogger(__name__)

_caches: dict[str, diskcache.Cache] = {}


def _get_cache(cache_dir: str) -> diskcache.Cache:
    if cache_dir not in _caches:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _caches[cache_dir] = diskcache.Cache(cache_dir)
    return _caches[cache_dir]


def _make_key(*args, **kwargs) -> str:
    payload = json.dumps({"args": [str(a) for a in args], "kwargs": {k: str(v) for k, v in sorted(kwargs.items())}})
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_call(
    func: Callable[..., str],
    cache_dir: str,
    *args: Any,
    **kwargs: Any,
) -> str:
    """
    Call func(*args, **kwargs) and cache the string result to disk.

    On subsequent calls with identical arguments the cached value is returned
    without invoking func. This makes LLM judge re-runs cheap.

    Args:
        func:      Function to call (typically an LLM invoke call).
        cache_dir: Directory for the diskcache store.
        *args:     Positional arguments forwarded to func.
        **kwargs:  Keyword arguments forwarded to func.
                   The special key ``model`` is included in the cache key
                   but NOT forwarded to func.

    Returns:
        The string result from func (or cache).
    """
    cache = _get_cache(cache_dir)
    model = kwargs.pop("model", "")
    key = _make_key(*args, model=model, **kwargs)

    if key in cache:
        logger.debug(f"Cache hit: {key[:12]}…")
        return cache[key]  # type: ignore[return-value]

    result: str = func(*args, **kwargs)
    cache[key] = result
    logger.debug(f"Cache miss — stored: {key[:12]}…")
    return result


def clear_cache(cache_dir: str) -> None:
    """Delete all cached entries in cache_dir."""
    cache = _get_cache(cache_dir)
    cache.clear()
    logger.info(f"Cache cleared: {cache_dir}")


def cache_size(cache_dir: str) -> Optional[int]:
    """Return number of cached entries, or None if cache doesn't exist."""
    if not Path(cache_dir).exists():
        return None
    return len(_get_cache(cache_dir))
