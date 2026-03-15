# -*- coding: utf-8 -*-
"""SCML 2022 Collusion track agents with lazy imports."""
from __future__ import annotations

import importlib

__all__ = [
    "bossagent",
    "team_may",
]

_SUBMODULES = {
    "bossagent",
    "team_may",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
