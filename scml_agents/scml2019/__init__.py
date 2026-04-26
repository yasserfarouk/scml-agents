# -*- coding: utf-8 -*-
"""SCML 2019 agents with lazy imports."""
from __future__ import annotations

import importlib

__all__ = [
    "FJ2FactoryManager",
    "RaptFactoryManager",
    "InsuranceFraudFactoryManager",
    "SAHAFactoryManager",
    "CheapBuyerFactoryManager",
    "NVMFactoryManager",
    "Monopoly",
    "PenaltySabotageFactoryManager",
]

# Lazy imports - modules are only loaded when their classes are accessed
_LAZY_IMPORTS = {
    "FJ2FactoryManager": (".fj2", "FJ2FactoryManager"),
    "RaptFactoryManager": (".rapt_fm", "RaptFactoryManager"),
    "InsuranceFraudFactoryManager": (".iffm", "InsuranceFraudFactoryManager"),
    "SAHAFactoryManager": (".saha", "SAHAFactoryManager"),
    "CheapBuyerFactoryManager": (".cheap_buyer.cheapbuyer", "CheapBuyerFactoryManager"),
    "NVMFactoryManager": (".nvm.nmv_agent", "NVMFactoryManager"),
    "Monopoly": (".monopoly", "Monopoly"),
    "PenaltySabotageFactoryManager": (".psfm", "PenaltySabotageFactoryManager"),
}

_cache: dict[str, type] = {}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        if name not in _cache:
            module_path, class_name = _LAZY_IMPORTS[name]
            module = importlib.import_module(module_path, __name__)
            _cache[name] = getattr(module, class_name)
        return _cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
