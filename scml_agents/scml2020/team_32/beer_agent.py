"""
**Submitted to ANAC 2020 SCML**
*Authors* Benjamin Wexler <benwex93@gmail.com> Elad <eladna812@gmail.com>
"""
from scml.scml2020.agents import DoNothingAgent

# from scml.scml2020 import SCML2020World
# import matplotlib.pyplot as plt
# required for running the test tournament
import time
from tabulate import tabulate
from scml.scml2020.utils import anac2020_std, anac2020_collusion
from scml.scml2020.agents import (
    DecentralizingAgent,
    BuyCheapSellExpensiveAgent,
    IndDecentralizingAgent,
    RandomAgent,
)
from negmas.helpers import humanize_time

# required for typing
from typing import List, Optional, Dict, Any
import numpy as np
from negmas import (
    Issue,
    AgentMechanismInterface,
    Contract,
    Negotiator,
    MechanismState,
    Breach,
    LinearUtilityFunction,
)
from scml.scml2020.components.production import DemandDrivenProductionStrategy
from scml.scml2020.components.production import SupplyDrivenProductionStrategy
from scml.scml2020.components.trading import PredictionBasedTradingStrategy
from scml.scml2020.components import TradePredictionStrategy
from scml.scml2020 import IndependentNegotiationsManager
from typing import Tuple

__all__ = ["BeerAgent"]


class OurNegotiationsManager(IndependentNegotiationsManager):
    def acceptable_unit_price(self, step: int, sell: bool) -> int:

        # Original Decentralizing Agent

        # production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        # if sell:
        #    return production_cost + self.input_cost[step]
        # return self.output_price[step] - production_cost

        # ours
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        return (
            self.awi.catalog_prices[self.awi.my_output_product] * 10
            if sell
            else production_cost
        )

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
            return needed[step] - secured[step]
        else:
            # original

            # needed, secured = self.inputs_needed, self.inputs_secured
            # return needed[step] - secured[step]

            # ours
            return 1 + step

    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        # original
        # if is_seller:
        #    return LinearUtilityFunction((1, 1, 10))
        # return LinearUtilityFunction((1, -1, -10))

        # ours
        if is_seller:
            return LinearUtilityFunction((0, 1, 25))
        return LinearUtilityFunction((0, -1, -25))


# use DemandDrivenProductionStrategy instead of SupplyDrivenProductionStrategy
class BeerAgent(
    OurNegotiationsManager, DemandDrivenProductionStrategy, DecentralizingAgent
):
    pass


def run(
    competition="std",
    reveal_names=True,
    n_steps=20,
    n_configs=2,
    max_n_worlds_per_config=None,
    n_runs_per_world=1,
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competition: The competition type to run (possibilities are std, 
                     collusion).        
        n_steps:     The number of simulation steps.
        n_configs:   Number of different world configurations to try. 
                     Different world configurations will correspond to
                     different number of factories, profiles
                     , production graphs etc
        n_runs_per_world: How many times will each world simulation be run.

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value        

    """
    # competitors = [BeerAgent, DecentralizingAgent]
    competitors = [BeerAgent, BuyCheapSellExpensiveAgent]
    start = time.perf_counter()
    if competition == "std":
        results = anac2020_std(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
            n_runs_per_world=n_runs_per_world,
        )
    elif competition == "collusion":
        results = anac2020_collusion(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
            n_runs_per_world=n_runs_per_world,
        )
    else:
        raise ValueError(f"Unknown competition type {competition}")
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    pass
    # run()
    # world = SCML2020World(
    #     **SCML2020World.generate([DecentralizingAgent, BeerAgent], n_steps=10),
    #     construct_graphs=True,
    # )

    # world.run()
    # world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
    # plt.show()
