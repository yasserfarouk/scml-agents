"""Attaches the generators of this package to NegMAS's global seed.

Most agents here draw their randomness from :mod:`random` and
:mod:`numpy.random`, which NegMAS already seeds. A few (``scml2020/team_10``
and ``scml2023/oneshot/team_102``) run torch models, so this module registers a
seeder for torch with `negmas.helpers.rand.register_seeder`.

Nothing here runs unless ``NEGMAS_RAND_SEED`` asks for a seed, so an unseeded
run behaves exactly as it did before and never pays for importing torch.

The `gymnasium` spaces built by some 2025 agents are not covered: they seed
themselves per instance, from inside agent code we do not touch.
"""

from __future__ import annotations

try:
    from negmas.helpers.rand import register_seeder
except ImportError:  # negmas too old; nothing to hook onto
    register_seeder = None

__all__ = ["seed_torch", "register"]


def seed_torch(seed: int) -> None:
    """Seed torch's global generators (CPU and every CUDA device).

    Best-effort: torch may be missing or broken, and a seeder that raises would
    take down the run that asked for a seed, so every failure is ignored.
    """
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def register() -> None:
    """Register `seed_torch` with NegMAS, if this negmas has the hook."""
    if register_seeder is not None:
        register_seeder(seed_torch)
