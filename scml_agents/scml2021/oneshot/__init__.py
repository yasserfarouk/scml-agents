# -*- coding: utf-8 -*-
"""SCML 2021 Oneshot track agents with lazy imports."""
from __future__ import annotations

import importlib

__all__ = [
    "staghunter",
    "team_50",
    "team_51",
    "team_54",
    "team_55",
    "team_61",
    "team_62",
    "team_72",
    "team_73",
    "team_86",
    "team_90",
    "team_corleone",
]

_SUBMODULES = {
    "staghunter",
    "team_50",
    "team_51",
    "team_54",
    "team_55",
    "team_61",
    "team_62",
    "team_72",
    "team_73",
    "team_86",
    "team_90",
    "team_corleone",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
