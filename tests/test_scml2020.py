import copy
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import hypothesis.strategies as st
import numpy as np
import pkg_resources
import pytest
from hypothesis import given, settings
from negmas.helpers import unique_name
from negmas.situated import Contract
from pytest import mark
from scml.scml2020 import SCML2020World

from scml_agents import get_agents
from scml_agents.scml2020 import *
from scml_agents.scml2020.monty_hall import MontyHall


@mark.parametrize("fm", get_agents(2020, as_class=True))
def test_can_run(fm):
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


if __name__ == "__main__":
    pytest.main(args=[__file__])
