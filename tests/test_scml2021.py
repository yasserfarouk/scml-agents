import pytest
from pytest import mark
from scml.oneshot import SCML2020OneShotWorld
from scml.scml2020 import SCML2021World

from scml_agents import get_agents
from scml_agents.scml2020 import *
from scml_agents.scml2021.standard.team_82.perry import PerryTheAgent

from .switches import (
    SCMLAGENTS_RUN2021,
)


@pytest.mark.skipif(not SCMLAGENTS_RUN2021, reason="Skipping 2021")
@mark.parametrize(
    "fm", get_agents(2021, as_class=True, track="collusion", ignore_failing=True)
)
def test_can_run_collusion(fm):
    n_steps = 10
    world = SCML2021World(
        **SCML2021World.generate(
            agent_types=["scml.scml2020.agents.MarketAwareDecentralizingAgent", fm],
            n_steps=n_steps,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(not SCMLAGENTS_RUN2021, reason="Skipping 2021")
def test_can_run_std_example():
    fm = PerryTheAgent
    n_steps = 10
    world = SCML2021World(
        **SCML2021World.generate(
            agent_types=["scml.scml2020.agents.MarketAwareDecentralizingAgent", fm],
            n_steps=n_steps,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(not SCMLAGENTS_RUN2021, reason="Skipping 2021")
@mark.parametrize(
    "fm", get_agents(2021, as_class=True, track="std", ignore_failing=True)
)
def test_can_run_std(fm):
    n_steps = 10
    world = SCML2021World(
        **SCML2021World.generate(
            agent_types=["scml.scml2020.agents.MarketAwareDecentralizingAgent", fm],
            n_steps=n_steps,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(not SCMLAGENTS_RUN2021, reason="Skipping 2021")
@mark.parametrize(
    "fm",
    get_agents(
        2021, as_class=True, track="oneshot", finalists_only=True, ignore_failing=True
    ),
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


@pytest.mark.skipif(not SCMLAGENTS_RUN2021, reason="Skipping 2021")
@mark.parametrize(
    "fm", get_agents(2021, as_class=True, track="oneshot", ignore_failing=True)
)
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
