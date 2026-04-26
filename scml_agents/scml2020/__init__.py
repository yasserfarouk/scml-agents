# -*- coding: utf-8 -*-
"""SCML 2020 agents with lazy imports."""
from __future__ import annotations

import importlib

__all__ = [
    # Submodules (for access like scml2020.team_may)
    "a_sengupta",
    "agent0x111",
    "bargent",
    "biu_th",
    "monty_hall",
    "past_frauds",
    "team_10",
    "team_15",
    "team_17",
    "team_18",
    "team_19",
    "team_20",
    "team_22",
    "team_25",
    "team_27",
    "team_29",
    "team_32",
    "team_may",
    "threadfield",
]

_SUBMODULES = {
    "a_sengupta",
    "agent0x111",
    "bargent",
    "biu_th",
    "monty_hall",
    "past_frauds",
    "team_10",
    "team_15",
    "team_17",
    "team_18",
    "team_19",
    "team_20",
    "team_22",
    "team_25",
    "team_27",
    "team_29",
    "team_32",
    "team_may",
    "threadfield",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
