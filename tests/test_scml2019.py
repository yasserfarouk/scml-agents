import pytest
from scml.scml2019.factory_managers.builtins import GreedyFactoryManager
from scml.scml2019.utils19 import anac2019_collusion, anac2019_sabotage, anac2019_std
from scml.scml2019.world import SCML2019World

from scml_agents import get_agents
from scml_agents.scml2019 import *

from .switches import (
    SCMLAGENTS_RUN2019,
    SCMLAGENTS_RUN_COLLUSION_TOURNAMENTS,
    SCMLAGENTS_RUN_SABOTAGE_TOURNAMENTS,
    SCMLAGENTS_RUN_STD_TOURNAMENTS,
)


@pytest.mark.skipif(not SCMLAGENTS_RUN2019, reason="Skipping 2019")
def test_can_run_std_example():
    fm = NVMFactoryManager
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


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2019 or not SCMLAGENTS_RUN_STD_TOURNAMENTS, reason="Skipping 2019"
)
@pytest.mark.parametrize("fm", get_agents(2019, track="all"))
def test_can_run_std(fm):
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


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2019 or not SCMLAGENTS_RUN_STD_TOURNAMENTS, reason="Skipping 2019"
)
def test_can_run_std_tournament():
    competitors = get_agents(2019, track="std", qualified_only=True)
    anac2019_std(
        competitors,
        n_configs=1,
        n_steps=6,
        n_runs_per_world=1,
        max_worlds_per_config=10,
        min_factories_per_level=2,
    )


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2019 or not SCMLAGENTS_RUN_COLLUSION_TOURNAMENTS,
    reason="Skipping 2019",
)
def test_can_run_collusion_tournament():
    competitors = get_agents(2019, track="collusion", qualified_only=True)
    anac2019_collusion(
        competitors,
        n_configs=1,
        n_steps=6,
        n_runs_per_world=1,
        max_worlds_per_config=10,
        min_factories_per_level=2,
    )


@pytest.mark.skipif(
    not SCMLAGENTS_RUN2019 or not SCMLAGENTS_RUN_SABOTAGE_TOURNAMENTS,
    reason="Skipping 2019",
)
def test_can_run_sabotage_tournament():
    competitors = get_agents(2019, track="sabotage", qualified_only=True)
    anac2019_sabotage(
        competitors,
        n_configs=1,
        n_steps=6,
        n_runs_per_world=1,
        max_worlds_per_config=10,
        min_factories_per_level=2,
    )


if __name__ == "__main__":
    pytest.main(args=[__file__])
