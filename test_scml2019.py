import copy
from pathlib import Path
from pprint import pprint

from pytest import mark
from typing import List, Dict

import numpy as np
import pkg_resources
import pytest
from hypothesis import given, settings

from scml.scml2019 import *
from scml.scml2019 import (
    InputOutput,
    Job,
    FactoryStatusUpdate,
    GreedyScheduler,
    ProductionFailure,
)
from negmas.helpers import unique_name
from negmas.situated import Contract
import hypothesis.strategies as st
from scml_agents.scml2019 import *
from scml_agents import get_agents


@mark.parametrize("fm", get_agents(2019))
def test_can_run(fm):
    horizon = None
    signing_delay = 0
    n_factory_levels = 1
    n_factories_per_level = 2
    n_steps = 10
    world = SCML2019World.chain_world(
        n_intermediate_levels=n_factory_levels - 1,
        log_file_name="",
        n_steps=n_steps,
        manager_types=(GreedyFactoryManager, fm),
        n_factories_per_level=n_factories_per_level,
        default_signing_delay=signing_delay,
        consumer_kwargs={
            "consumption_horizon": horizon,
            "negotiator_type": "negmas.sao.NiceNegotiator",
        },
        miner_kwargs={"negotiator_type": "negmas.sao.NiceNegotiator"},
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) > 0


if __name__ == "__main__":
    pytest.main(args=[__file__])
