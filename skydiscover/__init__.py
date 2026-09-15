"""
SkyDiscover: a flexible framework for AI-driven scientific, algorithmic, and end-to-end systems discovery.
"""

# Aliases the module paths used before the optimize/ restructuring onto their current
# locations. Stdlib-only, so `import skydiscover` stays cheap.
from typing import Any, List

from skydiscover import _compat as _compat
from skydiscover._version import __version__

_compat.install()

# Lazily expose the heavy entry points (api/runner pull optional deps like yaml). This keeps
# `import skydiscover` cheap, so the stdlib-only spec helpers
# (`python -m skydiscover.synthesize.spec.requirements|decisions|build`) run with no extra dependencies.
_LAZY = {
    "Runner": ("skydiscover.optimize.runner", "Runner"),
    "run_discovery": ("skydiscover.optimize.api", "run_discovery"),
    "discover_solution": ("skydiscover.optimize.api", "discover_solution"),
    "DiscoveryResult": ("skydiscover.optimize.api", "DiscoveryResult"),
}

__all__ = ["Runner", "__version__", "run_discovery", "discover_solution", "DiscoveryResult"]


def __getattr__(name: str) -> Any:  # PEP 562 — import the heavy modules only on first access
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target[0]), target[1])


def __dir__() -> List[str]:
    return sorted(list(globals()) + __all__)
