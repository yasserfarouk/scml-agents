import pytest
from pytest import mark
from scml.oneshot import SCML2020OneShotWorld
from scml.std import SCML2024StdWorld as SCML2025StdWorld

from scml_agents import get_agents
from scml_agents.scml2025.standard.team_atsunaga.as0 import AS0

from .switches import (
    SCMLAGENTS_RUN2025,
    SCMLAGENTS_RUN2025_ONESHOT,
    SCMLAGENTS_RUN2025_STD,
)


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2025 or not SCMLAGENTS_RUN2025_STD, reason="Skipping 2025"
)
def test_can_run_std_example():
    fm = AS0
    n_steps = 10
    world = SCML2025StdWorld(
        **SCML2025StdWorld.generate(
            agent_types=["scml.std.agents.rand.RandomStdAgent", fm],
            n_steps=n_steps,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2025 or not SCMLAGENTS_RUN2025_STD, reason="Skipping 2025"
)
@mark.parametrize("fm", get_agents(2025, as_class=True, track="std"))
def test_can_run_std(fm):
    n_steps = 10
    world = SCML2025StdWorld(
        **SCML2025StdWorld.generate(
            agent_types=["scml.std.agents.rand.RandomStdAgent", fm],
            n_steps=n_steps,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2025_ONESHOT or not SCMLAGENTS_RUN2025, reason="Skipping 2025"
)
@mark.parametrize(
    "fm", get_agents(2025, as_class=True, track="oneshot", finalists_only=True)
)
def test_can_run_oneshot_finalists(fm):
    n_steps = 10
    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(
            agent_types=["scml.oneshot.agents.GreedyOneShotAgent", fm],
            n_steps=n_steps,
            agent_name_reveals_type=False,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2025_ONESHOT or not SCMLAGENTS_RUN2025, reason="Skipping 2025"
)
@mark.parametrize("fm", get_agents(2025, as_class=True, track="oneshot"))
def test_can_run_oneshot(fm):
    n_steps = 10
    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(
            agent_types=["scml.oneshot.agents.GreedyOneShotAgent", fm],
            n_steps=n_steps,
            agent_name_reveals_type=False,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


if __name__ == "__main__":
    pytest.main(args=[__file__])
