# =========比較Agent==========
from scml.scml2020.agents import (
    DecentralizingAgent,
    IndependentNegotiationsAgent,
    MarketAwareMovingRangeAgent,
    MarketAwareReactiveAgent,
)

# ============================

# required for development
from scml.scml2020 import SCML2020Agent

# from sky_trading import SkyTradingStrategy
# from sky_production import SkyProductionStrategy
# from sky_negotiation import SkyNegotiationsManager
from .sky_trading import SkyTradingStrategy
from .sky_production import SkyProductionStrategy
from .sky_negotiation import SkyNegotiationsManager

import warnings

# required for running the test tournament
import time
from tabulate import tabulate
from scml.utils import anac2021_std, anac2021_collusion, anac2021_oneshot

from negmas import LinearUtilityFunction
from negmas.helpers import humanize_time
from typing import Tuple
import numpy as np

warnings.simplefilter("ignore")

__all__ = ["SkyAgent"]


class SkyAgent(
    SkyProductionStrategy,
    SkyTradingStrategy,
    SkyNegotiationsManager,
    SCML2020Agent,
):
    """
    This is the only class you *need* to implement. You can create the agent
    by combining the following strategies:

    1. A trading strategy that decides the quantities to sell and buy
    2. A negotiation manager that decides which negotiations to engage in and
       uses some controller(s) to control the behavior of all negotiators
    3. A production strategy that decides what to produce
    """

    def init(self):
        self.today_input_price = (
            self.awi.catalog_prices[self.awi.my_input_product] * 0.9
        )
        self.today_output_price = (
            self.awi.catalog_prices[self.awi.my_output_product] * 1.1
        )
        self.today_sold = 0
        self.today_purchased = 0
        self.average_input = 0
        self.total_input_n = 0
        self.total_input_cost = 0
        self.production_cost = np.average(
            self.awi.profile.costs[:, self.awi.my_input_product]
        )
        super().init()

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            needed, secured = (
                self.outputs_needed[step:].sum(),
                self.outputs_secured[step:].sum(),
            )
        else:
            needed, secured = (
                self.inputs_needed[:step].sum(),
                self.inputs_secured[:step].sum(),
            )

        return min(self.awi.n_lines, needed - secured)

    def target_quantities(self, steps: Tuple[int, int], sell: bool) -> np.ndarray:
        """Implemented for speed but not really required"""

        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[steps[0] : steps[1]] - secured[steps[0] : steps[1]]

    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        """A utility function that penalizes high cost and late delivery for buying and and awards them for selling"""
        # 高いコストと遅延配達に関してペナルティを課す
        if is_seller:
            return LinearUtilityFunction((0, 1, 10), issues=issues, outcomes=outcomes)
        return LinearUtilityFunction((0, -1, -10), issues=issues, outcomes=outcomes)


def run(
    competition="std",
    reveal_names=True,
    n_steps=10,
    n_configs=2,
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

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value

    """
    competitors = [
        SkyAgent,
        MarketAwareMovingRangeAgent,
        MarketAwareReactiveAgent,
        IndependentNegotiationsAgent,
        DecentralizingAgent,
    ]
    start = time.perf_counter()
    if competition == "std":
        results = anac2021_std(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
        )
    elif competition == "collusion":
        results = anac2021_collusion(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
        )
    elif competition == "oneshot":
        # Standard agents can run in the OneShot environment but cannot win
        # the OneShot track!!
        from scml.oneshot.agents import GreedyOneShotAgent, RandomOneShotAgent

        competitors = [
            RandomOneShotAgent,
            GreedyOneShotAgent,
        ]
        results = anac2021_oneshot(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
        )
    else:
        raise ValueError(f"Unknown competition type {competition}")
    # just make agent types shorter in the results
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(
        "."
    ).str[-1]
    # show results
    pass  # print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    pass  # print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    # will run a short tournament against two built-in agents. Default is "std"
    # You can change this from the command line by running something like:
    # >> python3 myagent.py collusion
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "std")
