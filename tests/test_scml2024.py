import pytest
from pytest import mark
from scml.oneshot import SCML2020OneShotWorld
from scml.std import SCML2024StdWorld

from scml_agents import get_agents
from scml_agents.scml2024.standard.team_atsunaga.S22s import S5s

from .switches import (
    SCMLAGENTS_RUN2024,
    SCMLAGENTS_RUN2024_ONESHOT,
    SCMLAGENTS_RUN2024_STD,
)

# from scml_agents.scml2024.oneshot.team102 import GentleS as Gentle
# from scml_agents.scml2024.standard.team_67.polymorphic_agent import PolymorphicAgent
# from scml_agents.scml2024.standard.team_82.perry import PerryTheAgent


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2024 or not SCMLAGENTS_RUN2024_STD, reason="Skipping 2024"
)
def test_can_run_std_example():
    fm = S5s
    n_steps = 10
    world = SCML2024StdWorld(
        **SCML2024StdWorld.generate(
            agent_types=["scml.std.agents.rand.RandomStdAgent", fm],
            n_steps=n_steps,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2024 or not SCMLAGENTS_RUN2024_STD, reason="Skipping 2024"
)
@mark.parametrize("fm", get_agents(2024, as_class=True, track="std"))
def test_can_run_std(fm):
    n_steps = 10
    world = SCML2024StdWorld(
        **SCML2024StdWorld.generate(
            agent_types=["scml.std.agents.rand.RandomStdAgent", fm],
            n_steps=n_steps,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2024_ONESHOT or not SCMLAGENTS_RUN2024, reason="Skipping 2024"
)
@mark.parametrize(
    "fm", get_agents(2024, as_class=True, track="oneshot", finalists_only=True)
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
    not SCMLAGENTS_RUN2024_ONESHOT or not SCMLAGENTS_RUN2024, reason="Skipping 2024"
)
@mark.parametrize("fm", get_agents(2024, as_class=True, track="oneshot"))
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
