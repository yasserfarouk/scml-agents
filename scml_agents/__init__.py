# -*- coding: utf-8 -*-
"""
SCML Agents package with lazy imports for fast loading.

Use get_agents() to retrieve agent classes for specific competition years.
"""
from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

# Only import get_agents and FAILING_AGENTS - these are lightweight
from .agents import FAILING_AGENTS, get_agents

__version__ = "0.5.1"

__all__ = [
    "__version__",
    "get_agents",
    "FAILING_AGENTS",
    "scml2019",
    "scml2020",
    "scml2021",
    "scml2022",
    "scml2023",
    "scml2024",
    "scml2025",
    "contrib",
]

# Lazy module loading - submodules are only imported when accessed
_SUBMODULES = {
    "scml2019",
    "scml2020",
    "scml2021",
    "scml2022",
    "scml2023",
    "scml2024",
    "scml2025",
    "contrib",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
