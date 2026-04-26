# -*- coding: utf-8 -*-
"""SCML 2022 Oneshot track agents with lazy imports."""
from __future__ import annotations

import importlib

__all__ = [
    "team_62",
    "team_94",
    "team_96",
    "team_102",
    "team_103",
    "team_105",
    "team_106",
    "team_107",
    "team_123",
    "team_124",
    "team_126",
    "team_131",
    "team_134",
]

_SUBMODULES = {
    "team_62",
    "team_94",
    "team_96",
    "team_102",
    "team_103",
    "team_105",
    "team_106",
    "team_107",
    "team_123",
    "team_124",
    "team_126",
    "team_131",
    "team_134",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
