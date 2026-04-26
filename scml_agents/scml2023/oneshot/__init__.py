# -*- coding: utf-8 -*-
"""SCML 2023 Oneshot track agents with lazy imports."""
from __future__ import annotations

import importlib

__all__ = [
    "team_102",
    "team_123",
    "team_126",
    "team_127",
    "team_134",
    "team_139",
    "team_143",
    "team_144",
    "team_145",
    "team_148",
    "team_149",
    "team_151",
    "team_poli_usp",
]

_SUBMODULES = {
    "team_102",
    "team_123",
    "team_126",
    "team_127",
    "team_134",
    "team_139",
    "team_143",
    "team_144",
    "team_145",
    "team_148",
    "team_149",
    "team_151",
    "team_poli_usp",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
