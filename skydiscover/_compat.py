"""Import aliases for the module paths used before the ``optimize/`` restructuring.

Modules under ``skydiscover.optimize`` are also importable by their former top-level names, which
emit a ``DeprecationWarning`` naming the current location::

    import skydiscover.utils          # -> skydiscover.optimize.utils

An alias is the same module object as its target at any depth, so module state and ``isinstance``
checks behave identically either way.

For migration only; import from ``skydiscover.optimize`` in new code.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Optional, Sequence

#: Top-level names that moved verbatim into ``skydiscover.optimize``.
MOVED = (
    "api",
    "benchmarks",
    "cli",
    "config",
    "context_builder",
    "evaluation",
    "extras",
    "llm",
    "prompt",
    "runner",
    "search",
    "utils",
)

_PREFIX = "skydiscover."
_TARGET = "skydiscover.optimize."

#: Old names whose target is not simply ``skydiscover.optimize.<same name>``.
_RENAMED = {"prompt": "context_builder"}
_warned: set = set()


def _warn_once(old: str, new: str) -> None:
    if old in _warned:
        return
    _warned.add(old)
    warnings.warn(
        f"{old!r} is a compatibility alias for {new!r} and may be removed in a future "
        f"release; import {new!r} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


class _AliasLoader(Loader):
    """Resolves the alias to the target module object rather than loading a second copy."""

    def __init__(self, target: str) -> None:
        self.target = target

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return importlib.import_module(self.target)

    def exec_module(self, module: ModuleType) -> None:
        # Already executed on import of the target.
        pass


class _MovedModuleFinder(MetaPathFinder):
    """Resolves ``skydiscover.<moved>`` (and anything under it) to ``skydiscover.optimize.<moved>``."""

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[ModuleType] = None,
    ) -> Optional[ModuleSpec]:
        if not fullname.startswith(_PREFIX):
            return None
        rest = fullname[len(_PREFIX) :]
        head, _, tail = rest.partition(".")
        if head not in MOVED:
            return None
        new = _TARGET + _RENAMED.get(head, head) + (f".{tail}" if tail else "")
        try:
            importlib.import_module(new)
        except ImportError:
            return None  # not a module under either name; let the usual ImportError surface
        _warn_once(fullname, new)
        return ModuleSpec(fullname, _AliasLoader(new))


def install() -> None:
    """Register the finder (idempotent).

    It must precede the standard path finder: an aliased package keeps its target's ``__path__``,
    so a later finder would locate submodules by path and import them a second time under the old
    name.
    """
    if not any(isinstance(f, _MovedModuleFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MovedModuleFinder())
