import pytest
from pytest import mark
from scml.oneshot import SCML2024OneShotWorld
from scml.std import SCML2024StdWorld as SCML2026StdWorld

from scml_agents import get_agents

from .switches import (
    SCMLAGENTS_RUN2026,
    SCMLAGENTS_RUN2026_ONESHOT,
    SCMLAGENTS_RUN2026_STD,
)


def test_get_agents_2026_counts():
    # Full set per track (qualified == not disqualified; finalists/winners
    # are announced later).
    assert len(get_agents(2026, track="oneshot", as_class=False)) == 17
    assert (
        len(get_agents(2026, track="oneshot", qualified_only=True, as_class=False))
        == 16
    )
    assert len(get_agents(2026, track="std", as_class=False)) == 20
    assert len(get_agents(2026, track="std", qualified_only=True, as_class=False)) == 20
    assert len(get_agents(2026, finalists_only=True)) == 0
    assert len(get_agents(2026, winners_only=True)) == 0


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2026 or not SCMLAGENTS_RUN2026_STD, reason="Skipping 2026"
)
@mark.parametrize(
    "fm",
    get_agents(2026, as_class=True, track="std", ignore_failing=True),
    ids=lambda c: c.__name__,
)
def test_can_run_std(fm):
    n_steps = 10
    world = SCML2026StdWorld(
        **SCML2026StdWorld.generate(
            agent_types=["scml.std.agents.rand.RandomStdAgent", fm],
            n_steps=n_steps,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2026 or not SCMLAGENTS_RUN2026_ONESHOT, reason="Skipping 2026"
)
@mark.parametrize(
    "fm",
    get_agents(2026, as_class=True, track="oneshot", ignore_failing=True),
    ids=lambda c: c.__name__,
)
def test_can_run_oneshot(fm):
    n_steps = 10
    world = SCML2024OneShotWorld(
        **SCML2024OneShotWorld.generate(
            agent_types=["scml.oneshot.agents.GreedyOneShotAgent", fm],
            n_steps=n_steps,
            agent_name_reveals_type=False,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


if __name__ == "__main__":
    pytest.main(args=[__file__])
