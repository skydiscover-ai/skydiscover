"""Helpers for loading backend-specific default configs."""

import os

import yaml

_DIR = os.path.dirname(__file__)


def load_defaults(filename: str) -> dict:
    """Load a YAML defaults file from the defaults directory."""
    path = os.path.join(_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_defaults(obj, defaults: dict):
    """Recursively apply dict values to a dataclass-like object.

    Only sets attributes that already exist on the object.
    For nested dicts whose corresponding attribute is also an object,
    recurses into the sub-object.
    """
    for key, value in defaults.items():
        if not hasattr(obj, key):
            continue
        if isinstance(value, dict):
            sub = getattr(obj, key)
            if hasattr(sub, "__dict__"):
                apply_defaults(sub, value)
            else:
                setattr(obj, key, value)
        else:
            setattr(obj, key, value)
