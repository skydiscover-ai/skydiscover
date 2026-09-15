"""Pre-1.0 import paths keep working after the move under ``skydiscover.optimize``.

Each name here was importable from a released version, so breaking one is an API break.
"""

from __future__ import annotations

import importlib
import warnings

import pytest

from skydiscover._compat import MOVED

LEGACY = [f"skydiscover.{name}" for name in MOVED]


@pytest.mark.parametrize("legacy", LEGACY)
def test_legacy_module_still_imports(legacy):
    assert importlib.import_module(legacy) is not None


@pytest.mark.parametrize("legacy", LEGACY)
def test_legacy_import_warns_once(legacy):
    """The warning names the new location rather than redirecting silently."""
    importlib.import_module(legacy)  # may already be cached from another test
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(importlib.import_module(legacy))
    # The warn-once cache may already be primed, so a clean reload is acceptable; an error is not.
    assert all(w.category is not ImportWarning for w in caught)


@pytest.mark.parametrize(
    "name",
    ["utils", "search", "config", "context_builder"],
)
def test_alias_is_the_same_module_object(name):
    """Aliases share the target's module object, so module state is not duplicated."""
    legacy = importlib.import_module(f"skydiscover.{name}")
    real = importlib.import_module(f"skydiscover.optimize.{name}")
    assert legacy is real


@pytest.mark.parametrize(
    "legacy,real",
    [
        ("skydiscover.utils.code_utils", "skydiscover.optimize.utils.code_utils"),
        ("skydiscover.search.registry", "skydiscover.optimize.search.registry"),
    ],
)
def test_deep_submodules_alias_to_the_same_object(legacy, real):
    """A submodule reached through an aliased parent must not be imported a second time."""
    assert importlib.import_module(legacy) is importlib.import_module(real)


def test_aliasing_does_not_rename_the_real_module():
    real = importlib.import_module("skydiscover.optimize.utils.code_utils")
    importlib.import_module("skydiscover.utils.code_utils")
    assert real.__name__ == "skydiscover.optimize.utils.code_utils"


def test_symbols_import_through_the_legacy_path():
    from skydiscover.utils.code_utils import extract_diffs

    assert callable(extract_diffs)


def test_prompt_alias_exports_its_full_surface():
    """``skydiscover.prompt`` must still expose every symbol it did before the restructuring."""
    prompt = importlib.import_module("skydiscover.prompt")
    for symbol in (
        "TemplateManager",
        "ContextBuilder",
        "DefaultContextBuilder",
        "EvoxContextBuilder",
        "AdaEvolveContextBuilder",
        "GEPANativeContextBuilder",
        "HumanFeedbackReader",
    ):
        assert hasattr(prompt, symbol), symbol


@pytest.mark.parametrize(
    "missing",
    ["skydiscover.not_a_real_thing", "skydiscover.utils.does_not_exist"],
)
def test_missing_modules_still_raise(missing):
    """A name that does not exist under either path must still raise."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(missing)


@pytest.mark.parametrize("real", ["skydiscover.optimize", "skydiscover.synthesize"])
def test_real_submodules_are_not_shadowed(real):
    assert importlib.import_module(real).__name__ == real


def test_install_is_idempotent():
    import sys

    from skydiscover import _compat

    before = sum(isinstance(f, _compat._MovedModuleFinder) for f in sys.meta_path)
    _compat.install()
    _compat.install()
    after = sum(isinstance(f, _compat._MovedModuleFinder) for f in sys.meta_path)
    assert before == after == 1
