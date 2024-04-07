import pytest
from pytest import mark
from scml.oneshot import SCML2020OneShotWorld
from scml.scml2020 import SCML2022World

from scml_agents import get_agents
from scml_agents.scml2020 import *
from scml_agents.scml2022.oneshot.team_134.agent119 import PatientAgent

# from scml_agents.scml2022.oneshot.team102 import GentleS as Gentle
# from scml_agents.scml2022.standard.team_67.polymorphic_agent import PolymorphicAgent
# from scml_agents.scml2022.standard.team_82.perry import PerryTheAgent

from .switches import (
    SCMLAGENTS_RUN2022,
)


@pytest.mark.skipif(not SCMLAGENTS_RUN2022, reason="Skipping 2022")
@mark.parametrize(
    "fm", get_agents(2022, as_class=True, track="collusion", ignore_failing=True)
)
def test_can_run_collusion(fm):
    n_steps = 10
    world = SCML2022World(
        **SCML2022World.generate(
            agent_types=["scml.scml2020.agents.MarketAwareDecentralizingAgent", fm],
            n_steps=n_steps,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(not SCMLAGENTS_RUN2022, reason="Skipping 2022")
def test_can_run_std_example():
    fm = PatientAgent
    n_steps = 10
    world = SCML2022World(
        **SCML2022World.generate(
            agent_types=["scml.scml2020.agents.MarketAwareDecentralizingAgent", fm],
            n_steps=n_steps,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(not SCMLAGENTS_RUN2022, reason="Skipping 2022")
@mark.parametrize(
    "fm", get_agents(2022, as_class=True, track="std", ignore_failing=True)
)
def test_can_run_std(fm):
    n_steps = 10
    world = SCML2022World(
        **SCML2022World.generate(
            agent_types=["scml.scml2020.agents.MarketAwareDecentralizingAgent", fm],
            n_steps=n_steps,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(not SCMLAGENTS_RUN2022, reason="Skipping 2022")
@mark.parametrize(
    "fm",
    get_agents(
        2022, as_class=True, track="oneshot", finalists_only=True, ignore_failing=True
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


@pytest.mark.skipif(not SCMLAGENTS_RUN2022, reason="Skipping 2022")
@mark.parametrize(
    "fm", get_agents(2022, as_class=True, track="oneshot", ignore_failing=True)
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
