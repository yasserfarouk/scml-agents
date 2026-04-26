# -*- coding: utf-8 -*-
"""SCML 2024 agents with lazy imports."""
from __future__ import annotations

import importlib

__all__ = [
    "standard",
    "oneshot",
    # Legacy top-level access for some agents
    "team_181",
    "team_178",
    "teamyuzuru",
]

_SUBMODULES = {"standard", "oneshot", "team_181", "team_178", "teamyuzuru"}


def __getattr__(name: str):
    if name in _SUBMODULES:
        # team_181, team_178, teamyuzuru are actually in standard
        if name in ("team_181", "team_178", "teamyuzuru"):
            module = importlib.import_module(f".standard.{name}", __name__)
        else:
            module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
