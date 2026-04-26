# -*- coding: utf-8 -*-
"""SCML 2022 agents with lazy imports."""
from __future__ import annotations

import importlib

__all__ = [
    "standard",
    "oneshot",
    "collusion",
    # Individual agents exported at top level for backwards compatibility
    "AdaptiveQlAgent",
]

_SUBMODULES = {"standard", "oneshot", "collusion"}

# Agents that need to be accessible at the top level
_LAZY_AGENTS = {
    "AdaptiveQlAgent": ".standard.team_100",
}

_cache: dict[str, object] = {}


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _LAZY_AGENTS:
        if name not in _cache:
            module = importlib.import_module(_LAZY_AGENTS[name], __name__)
            _cache[name] = getattr(module, name)
        return _cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
