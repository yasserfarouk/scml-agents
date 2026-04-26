# -*- coding: utf-8 -*-
"""SCML 2024 Standard track agents with lazy imports."""
from __future__ import annotations

import importlib

__all__ = [
    "coyoteteam",
    "team_178",
    "team_181",
    "team_193",
    "team_atsunaga",
    "team_miyajima_std",
    "team_penguin",
    "teamyuzuru",
]

_SUBMODULES = {
    "coyoteteam",
    "team_178",
    "team_181",
    "team_193",
    "team_atsunaga",
    "team_miyajima_std",
    "team_penguin",
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
