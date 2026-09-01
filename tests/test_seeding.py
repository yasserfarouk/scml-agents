"""Tests that this package's torch generators follow NegMAS's global seed."""

import os
import subprocess
import sys
import types

import pytest

from scml_agents._rand import seed_torch

seed_all = pytest.importorskip(
    "negmas.helpers.rand", reason="this negmas has no global seed to follow"
).seed_all


def _run_unseeded(code: str) -> str:
    """Run `code` in a subprocess with NEGMAS_RAND_SEED explicitly unset."""
    env = {k: v for k, v in os.environ.items() if k != "NEGMAS_RAND_SEED"}
    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def test_import_is_unaffected_without_a_seed():
    """An unseeded run must not even import torch, let alone seed anything."""
    out = _run_unseeded(
        "import sys, scml_agents;"
        "from negmas.helpers.rand import get_seed;"
        "print(get_seed(), 'torch' in sys.modules)"
    )
    assert out == "None False"


def test_seed_torch_seeds_torch(monkeypatch):
    """`seed_all` reaches torch through the seeder registered on import."""
    seeded = []
    fake = types.ModuleType("torch")
    fake.manual_seed = seeded.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)

    seed_all(42)

    assert 42 in seeded


def test_seed_from_the_environment_reaches_torch():
    """The whole point: NEGMAS_RAND_SEED=42 seeds torch on import."""
    env = {**os.environ, "NEGMAS_RAND_SEED": "42"}
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, types;"
            "calls = [];"
            "fake = types.ModuleType('torch');"
            "fake.manual_seed = calls.append;"
            "sys.modules['torch'] = fake;"
            "__import__('scml_agents');"
            "print(calls)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()
    assert out == "[42]"


def test_seeder_never_raises():
    """A missing or broken torch must not take down a seeded run."""
    seed_torch(-(2**80))
