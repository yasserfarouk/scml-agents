import time
from pprint import pformat
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from negmas.helpers import humanize_time
from scml.scml2020 import (
    FixedTradePredictionStrategy,
    MeanERPStrategy,
    PredictionBasedTradingStrategy,
    SCML2020Agent,
    StepNegotiationManager,
    SupplyDrivenProductionStrategy,
    TradingStrategy,
)
from scml.scml2020.agents import BuyCheapSellExpensiveAgent, DecentralizingAgent
from scml.scml2020.agents.bcse import BuyCheapSellExpensiveAgent
from scml.scml2020.common import ANY_LINE, is_system_agent
from scml.utils import anac2021_collusion, anac2021_oneshot, anac2021_std
from tabulate import tabulate

__all__ = ["SorceryAgent"]

from typing import Dict, List, Tuple

import numpy as np
from negmas import Contract
from scml.scml2020 import SCML2020Agent, SCML2021World
from scml.scml2020.agents import (
    DecentralizingAgent,
    MarketAwareDecentralizingAgent,
    RandomAgent,
)
from scml.scml2020.common import NO_COMMAND

#####仮エージェント#####


ComparisonAgent = MarketAwareDecentralizingAgent


class PredictionBasedTradingStrategy(
    FixedTradePredictionStrategy, MeanERPStrategy, TradingStrategy
):
    """A trading strategy that uses prediction strategies to manage inputs/outputs needed

    Hooks Into:
        - `init`
        - `on_contracts_finalized`
        - `sign_all_contracts`
        - `on_agent_bankrupt`

    Remarks:
        - `Attributes` section describes the attributes that can be used to construct the component (passed to its
          `__init__` method).
        - `Provides` section describes the attributes (methods, properties, data-members) made available by this
          component directly. Note that everything provided by the bases of this components are also available to the
          agent (Check the `Bases` section above for all the bases of this component).
        - `Requires` section describes any requirements from the agent using this component. It defines a set of methods
          or properties/data-members that must exist in the agent that uses this component. These requirement are
          usually implemented as abstract methods in the component
        - `Abstract` section describes abstract methods that MUST be implemented by any descendant of this component.
        - `Hooks Into` section describes the methods this component overrides calling `super` () which allows other
          components to hook into the same method (by overriding it). Usually callbacks starting with `on_` are
          hooked into this way.
        - `Overrides` section describes the methods this component overrides without calling `super` effectively
          disallowing any other components after it in the MRO to call this method. Usually methods that do some
          action (i.e. not starting with `on_`) are overridden this way.


    """

    def init(self):
        super().init()
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])

        # Calculate the initial price (catalog price ± production cost)
        self.input_cost = (
            self.awi.catalog_prices[self.awi.my_output_product] - production_cost
        ) * np.ones(self.awi.n_steps, dtype=int)
        self.output_price = (
            self.awi.catalog_prices[self.awi.my_input_product] + production_cost
        ) * np.ones(self.awi.n_steps, dtype=int)

        # Maximum number of products that can be produced
        self.inputs_needed = (self.awi.n_steps * self.awi.n_lines) * np.ones(
            self.awi.n_steps, dtype=int
        )
        # If the last production is assigned, make it three quarters (because the number of products that can be sold is limited)
        if self.awi.my_output_product == (self.awi.n_products - 1):
            self.inputs_needed = self.inputs_needed // 3 // 4

        # Zero at first (avoid breach of contract)
        self.outputs_needed = np.zeros(self.awi.n_steps, dtype=int)

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        self.awi.logdebug_agent(
            # f"Enter Contracts Finalized:\n"
            f"Signed {pformat([self._format(_) for _ in signed])}\n"
            f"Cancelled {pformat([self._format(_) for _ in cancelled])}\n"
            f"{pformat(self.internal_state)}"
        )
        super().on_contracts_finalized(signed, cancelled, rejectors)
        consumed = 0
        for contract in signed:
            if contract.annotation["caller"] == self.id:
                continue
            is_seller = contract.annotation["seller"] == self.id
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            if is_seller:
                # if I am a seller, I will buy my needs to produce
                output_product = contract.annotation["product"]
                input_product = output_product - 1
                self.outputs_secured[t] += q
                if input_product >= 0 and t > 0:
                    # find the maximum possible production I can do and saturate to it
                    steps, lines = self.awi.available_for_production(
                        repeats=q, step=(self.awi.current_step, t - 1)
                    )
                    q = min(len(steps) - consumed, q)
                    consumed += q
                    if contract.annotation["caller"] != self.id:
                        # this is a sell contract that I did not expect yet. Update needs accordingly
                        self.inputs_needed[t - 1] += max(1, q)
                continue

            # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
            input_product = contract.annotation["product"]
            output_product = input_product + 1
            self.inputs_secured[t] += q
            if output_product < self.awi.n_products and t < self.awi.n_steps - 1:
                # this is a buy contract that I did not expect yet. Update needs accordingly
                self.outputs_needed[t + 1] += max(1, q)

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        # sort contracts by time and then put system contracts first within each time-step
        signatures = [None] * len(contracts)
        contracts = sorted(
            zip(contracts, range(len(contracts))),
            key=lambda x: (
                x[0].agreement["unit_price"],
                x[0].agreement["time"],
                0
                if is_system_agent(x[0].annotation["seller"])
                or is_system_agent(x[0].annotation["buyer"])
                else 1,
                x[0].agreement["unit_price"],
            ),
        )
        sold, bought = 0, 0
        s = self.awi.current_step
        catalog_buy = self.awi.catalog_prices[self.awi.my_input_product]
        catalog_sell = self.awi.catalog_prices[self.awi.my_output_product]
        for contract, indx in contracts:
            is_seller = contract.annotation["seller"] == self.id
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            # check that the contract is executable in principle
            if t < s and len(contract.issues) == 3:
                continue
            if (is_seller and u < 0.75 * catalog_sell) or (
                not is_seller and u > 1.25 * catalog_buy
            ):
                continue
            if is_seller:
                trange = (s, t)
                secured, needed = (self.outputs_secured, self.outputs_needed)
                taken = sold
            else:
                trange = (t + 1, self.awi.n_steps - 1)
                secured, needed = (self.inputs_secured, self.inputs_needed)
                taken = bought

            # check that I can produce the required quantities even in principle
            steps, lines = self.awi.available_for_production(
                q, trange, ANY_LINE, override=False, method="all"
            )
            if len(steps) - taken < q:
                continue

            if (
                secured[trange[0] : trange[1] + 1].sum() + q + taken
                <= needed[trange[0] : trange[1] + 1].sum()
            ):
                signatures[indx] = self.id
                if is_seller:
                    sold += q
                else:
                    bought += q
        return signatures

    def _format(self, c: Contract):
        return (
            f"{f'>' if c.annotation['seller'] == self.id else '<'}"
            f"{c.annotation['buyer'] if c.annotation['seller'] == self.id else c.annotation['seller']}: "
            f"{c.agreement['quantity']} of {c.annotation['product']} @ {c.agreement['unit_price']} on {c.agreement['time']}"
        )

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: List[int],
        compensation_money: int,
    ) -> None:
        super().on_agent_bankrupt(agent, contracts, quantities, compensation_money)
        for contract, new_quantity in zip(contracts, quantities):
            q = contract.agreement["quantity"]
            if new_quantity == q:
                continue
            t = contract.agreement["time"]
            missing = q - new_quantity
            if t < self.awi.current_step:
                continue
            if contract.annotation["seller"] == self.id:
                self.outputs_secured[t] -= missing
                # if t > 0:
                #     self.inputs_needed[t - 1] -= missing
            else:
                self.inputs_secured[t] += missing
                # if t < self.awi.n_steps - 1:
                #     self.outputs_needed[t + 1] -= missing


class SorceryAgent(
    SupplyDrivenProductionStrategy,
    StepNegotiationManager,
    PredictionBasedTradingStrategy,
    SCML2020Agent,
):
    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        if sell:
            return self.output_price[step]
        else:
            return self.input_cost[step]

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            # In the final step, return the maximum number
            if step >= self.awi.n_steps:
                return self.outputs_needed[step - 1]
            # Up to twice the number of production lines
            return min(self.outputs_needed[step], self.awi.n_lines * 2)
        else:
            # I'm not buying in the final step
            if step >= self.awi.n_steps:
                return 0
            # Up to twice the number of production lines
            return min(self.inputs_needed[step], self.awi.n_lines * 2)


"""
world = SCML2021World(
    **SCML2021World.generate([ComparisonAgent, RandomAgent, MyAgent], n_steps=10),
    construct_graphs=True,
)



world.run()
world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
plt.show() ##########1###########

from collections import defaultdict
def show_agent_scores(world):
    scores = defaultdict(list)
    for aid, score in world.scores().items():
        scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
    scores = {k: sum(v)/len(v) for k, v in scores.items()}
    plt.bar(list(scores.keys()), list(scores.values()), width=0.2)
    plt.show()

show_agent_scores(world)
"""


def run(
    competition="std",
    reveal_names=True,
    n_steps=50,
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
        MyAgent,
        DecentralizingAgent,
        BuyCheapSellExpensiveAgent,
        RandomAgent,
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
            MyAgent,
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
    # print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    # print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    # will run a short tournament against two built-in agents. Default is "std"
    # You can change this from the command line by running something like:
    # >> python3 myagent.py collusion
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "std")
