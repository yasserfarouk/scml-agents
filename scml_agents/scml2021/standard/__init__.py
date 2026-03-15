# -*- coding: utf-8 -*-
"""SCML 2021 Standard track agents with lazy imports."""
from __future__ import annotations

import importlib

__all__ = [
    "bossagent",
    "iyibiteam",
    "team_41",
    "team_44",
    "team_45",
    "team_46",
    "team_49",
    "team_53",
    "team_67",
    "team_78",
    "team_82",
    "team_91",
    "team_may",
    "team_mediocre",
    "wabisabikoalas",
]

_SUBMODULES = {
    "bossagent",
    "iyibiteam",
    "team_41",
    "team_44",
    "team_45",
    "team_46",
    "team_49",
    "team_53",
    "team_67",
    "team_78",
    "team_82",
    "team_91",
    "team_may",
    "team_mediocre",
    "wabisabikoalas",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
