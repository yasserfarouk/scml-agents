#!/usr/bin/env python
"""
**Submitted to ANAC 2022 SCML**
*Authors* type-your-team-member-names-with-their-emails here


This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2022 SCML.

This module implements a factory manager for the SCM 2022 league of ANAC 2022
competition. This version will use subcomponents. Please refer to the
[game description](http://www.yasserm.com/scml/scml2021.pdf) for all the
callbacks and subcomponents available.

Your agent can sense and act in the world by calling methods in the AWI it has.
For all properties/methods available only to SCM agents check:
  http://www.yasserm.com/scml/scml2020docs/api/scml.scml2020.AWI.html

Documentation, tutorials and other goodies are available at:
  http://www.yasserm.com/scml/scml2020docs/

Competition website is: https://scml.cs.brown.edu


To test this template do the following:

0. Let the path to this file be /{path-to-this-file}/myagent.py

1. Install a venv (recommended)
>> python3 -m venv .venv

2. Activate the venv (required if you installed a venv)
On Linux/Mac:
    >> source .venv/bin/activate
On Windows:
    >> \.venv\Scripts\activate.bat

3. Update pip just in case (recommended)

>> pip install -U pip wheel

4. Install SCML

>> pip install scml

5. [Optional] Install last year's agents for STD/COLLUSION tracks only

>> pip install scml-agents

6. Run the script with no parameters (assuming you are )

>> python /{path-to-this-file}/myagent.py

You should see a short tournament running and results reported.
"""

from __future__ import annotations

# required for typing
from typing import Any, Dict, List, Optional

# required for development
# required for running the test tournament
import time
from negmas.helpers import humanize_time
from negmas import Contract
import numpy as np
from abc import abstractmethod
from scml import ANY_LINE, is_system_agent
from scml.scml2020 import QUANTITY, TIME, UNIT_PRICE, Failure, SCML2020Agent
from scml.scml2020.agents import (
    DecentralizingAgent,
    MarketAwareIndependentNegotiationsAgent,
)
from scml.scml2020.components.negotiation import (
    StepNegotiationManager,
    NegotiationManager,
)
from scml.scml2020.components.production import (
    SupplyDrivenProductionStrategy,
    ProductionStrategy,
)
from scml.scml2020.components.trading import PredictionBasedTradingStrategy
from scml.utils import anac2022_collusion, anac2022_std, anac2022_oneshot
from tabulate import tabulate

__all__ = [
    "SalesAgent",
]


class MyNegotiationManager(StepNegotiationManager):
    def _urange(self, step, is_seller, time_range):
        price = self.acceptable_unit_price(step, is_seller)
        if is_seller:
            return price, price * 2
        return 1, price


class MyTradingStrategy(PredictionBasedTradingStrategy):
    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        signatures = [None] * len(contracts)
        # sort contracts by goodness of price, time and then put system contracts first within each time-step
        contracts = sorted(
            zip(contracts, range(len(contracts))),
            key=lambda x: (
                x[0].agreement["time"],
                (
                    x[0].agreement["unit_price"]
                    - self.output_price[x[0].agreement["time"]]
                )
                if x[0].annotation["seller"] == self.id
                else (
                    self.input_cost[x[0].agreement["time"]]
                    - x[0].agreement["unit_price"]
                ),
                0
                if is_system_agent(x[0].annotation["seller"])
                or is_system_agent(x[0].annotation["buyer"])
                else 1,
            ),
        )
        sold, bought = 0, 0
        s = self.awi.current_step
        b = self.awi.current_balance

        for contract, indx in contracts:
            is_seller = contract.annotation["seller"] == self.id
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            # check that the contract is executable in principle. The second
            # condition checkes that the contract is negotiated and not exogenous
            if t < s and len(contract.issues) == 3:
                continue
            catalog_buy = self.input_cost[t]
            catalog_sell = self.output_price[t]
            # check that the contract has a good price
            if (
                (is_seller and u < catalog_sell and t <= self.awi.n_steps * 0.5)
                or (  # change 0.5->1
                    # The second half lowers the allowable price.
                    is_seller
                    and u < 0.5 * catalog_sell
                    and t > self.awi.n_steps * 0.5
                )
                or (not is_seller and u > catalog_buy)  # change 1.5->1
            ):
                continue

            if is_seller:
                trange = (s, t - 1)
                secured, needed = (self.outputs_secured, self.outputs_needed)
                taken = sold
            else:
                # I don't make contracts the second half of the step.
                # I don't make contracts that will drastically reduce my balance.
                if (b - u * q) < b * 0.5 or t > self.awi.n_steps * 0.5:
                    continue
                trange = (t + 1, self.awi.n_steps - 1)
                secured, needed = (self.inputs_secured, self.inputs_needed)
                taken = bought

            # check that I can produce the required quantities even in principle
            steps, _ = self.awi.available_for_production(
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


class SalesAgent(
    MyNegotiationManager,
    MyTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        if sell:
            return self.output_price[step]
        else:
            return self.input_cost[step]

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            needed, secured = (
                self.outputs_needed[step:].sum(),
                self.outputs_secured[step:].sum(),
            )
            return min(self.awi.n_lines * 2, needed - secured)
        else:
            # I don't buy at the final step
            if step >= self.awi.n_steps:
                return 0
            needed, secured = (
                self.inputs_needed[:step].sum(),
                self.inputs_secured[:step].sum(),
            )
            return min(self.awi.n_lines * 2, needed - secured)


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
        SalesAgent,
        DecentralizingAgent,
        MarketAwareIndependentNegotiationsAgent,
    ]
    start = time.perf_counter()
    if competition == "std":
        results = anac2022_std(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
        )
    elif competition == "collusion":
        results = anac2022_collusion(
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
            SalesAgent,
            RandomOneShotAgent,
            GreedyOneShotAgent,
        ]
        results = anac2022_oneshot(
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
