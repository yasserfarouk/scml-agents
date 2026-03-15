# -*- coding: utf-8 -*-
"""SCML 2025 Oneshot track agents with lazy imports."""
from __future__ import annotations

import importlib

__all__ = [
    "mat",
    "takafam",
    "team_276",
    "team_283",
    "team_284",
    "team_293",
    "team_star_up",
    "team_ukku",
    "teamyuzuru",
]

_SUBMODULES = {
    "mat",
    "takafam",
    "team_276",
    "team_283",
    "team_284",
    "team_293",
    "team_star_up",
    "team_ukku",
    "teamyuzuru",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
