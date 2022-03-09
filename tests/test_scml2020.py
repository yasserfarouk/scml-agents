import pytest
from scml.scml2020 import SCML2020World

from scml_agents import get_agents
from scml_agents.scml2020 import *
from scml_agents.scml2020.monty_hall import MontyHall

from .switches import (
    SCMLAGENTS_RUN2020,
    SCMLAGENTS_RUN_COLLUSION_TOURNAMENTS,
    SCMLAGENTS_RUN_SABOTAGE_TOURNAMENTS,
    SCMLAGENTS_RUN_STD_TOURNAMENTS,
)


def do_run(fm):
    n_steps = 5 if issubclass(fm, MontyHall) else 50
    world = SCML2020World(
        **SCML2020World.generate(
            agent_types=["scml.scml2020.DecentralizingAgent", fm],
            n_steps=n_steps,
            n_processes=(3, 4) if issubclass(fm, MontyHall) else 2,
        )
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) >= 0


@pytest.mark.skipif(not SCMLAGENTS_RUN2020, reason="Skipping 2020")
@pytest.mark.parametrize("fm", get_agents(2020, as_class=True))
def test_can_run(fm):
    do_run(fm)


# def test_can_run_agent30():
#     from negmas.helpers.types import get_class
#     do_run(get_class("scml_agents.scml2020.team_25.Agent30"))

if __name__ == "__main__":
    pytest.main(args=[__file__])
