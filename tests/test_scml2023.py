import pytest
from pytest import mark
from scml.oneshot import SCML2020OneShotWorld
from scml.scml2020 import SCML2023World

from scml_agents import get_agents
from scml_agents.scml2020 import *
from scml_agents.scml2023.oneshot.team_poli_usp import QuantityOrientedAgent

from .switches import (
    SCMLAGENTS_RUN2023,
    SCMLAGENTS_RUN2023_ONESHOT,
    SCMLAGENTS_RUN2023_STD,
)

# from scml_agents.scml2023.oneshot.team102 import GentleS as Gentle
# from scml_agents.scml2023.standard.team_67.polymorphic_agent import PolymorphicAgent
# from scml_agents.scml2023.standard.team_82.perry import PerryTheAgent


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2023 or not SCMLAGENTS_RUN2023_STD, reason="Skipping 2023"
)
@mark.parametrize("fm", get_agents(2023, as_class=True, track="collusion"))
def test_can_run_collusion(fm):
    n_steps = 10
    world = SCML2023World(
        **SCML2023World.generate(
            agent_types=["scml.scml2020.agents.MarketAwareDecentralizingAgent", fm],
            n_steps=n_steps,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2023 or not SCMLAGENTS_RUN2023_STD, reason="Skipping 2023"
)
def test_can_run_std_example():
    fm = QuantityOrientedAgent
    n_steps = 10
    world = SCML2023World(
        **SCML2023World.generate(
            agent_types=["scml.scml2020.agents.MarketAwareDecentralizingAgent", fm],
            n_steps=n_steps,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2023 or not SCMLAGENTS_RUN2023_STD, reason="Skipping 2023"
)
@mark.parametrize("fm", get_agents(2023, as_class=True, track="std"))
def test_can_run_std(fm):
    n_steps = 10
    world = SCML2023World(
        **SCML2023World.generate(
            agent_types=["scml.scml2020.agents.MarketAwareDecentralizingAgent", fm],
            n_steps=n_steps,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2023_ONESHOT or not SCMLAGENTS_RUN2023, reason="Skipping 2023"
)
@mark.parametrize(
    "fm", get_agents(2023, as_class=True, track="oneshot", finalists_only=True)
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
    not SCMLAGENTS_RUN2023_ONESHOT or not SCMLAGENTS_RUN2023, reason="Skipping 2023"
)
@mark.parametrize("fm", get_agents(2023, as_class=True, track="oneshot"))
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
