# -*- coding: utf-8 -*-
"""SCML 2024 Oneshot track agents with lazy imports."""
from __future__ import annotations

import importlib

__all__ = [
    "coyoteteam",
    "ozug4",
    "team_144",
    "team_164",
    "team_171",
    "team_172",
    "team_193",
    "team_abc",
    "team_miyajima_oneshot",
    "teamyuzuru",
]

_SUBMODULES = {
    "coyoteteam",
    "ozug4",
    "team_144",
    "team_164",
    "team_171",
    "team_172",
    "team_193",
    "team_abc",
    "team_miyajima_oneshot",
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
