#!/usr/bin/env python
"""
**Submitted to ANAC 2021 SCML (OneShot track)**
*Authors*
Tal Tikochinski     taltiko26@gmail.com
Moshe Binieli       mmoshikoo@gmail.com
Ran Ezra            ran421@gmail.com


This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2021 SCML.

This module implements a factory manager for the SCM 2021 league of ANAC 2021
competition (one-shot track).
Game Description is available at:
http://www.yasserm.com/scml/scml2021oneshot.pdf

Your agent can sense and act in the world by calling methods in the AWI it has.
For all properties/methods available only to SCM agents check:
  http://www.yasserm.com/scml/scml2020docs/api/scml.oneshot.OneShotAWI.html

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
    >> \\.venv\\Scripts\activate.bat

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

# required for running tournaments and printing
import time

# required for typing
from typing import Any, Dict, List, Optional

import numpy as np
from negmas import (
    AgentMechanismInterface,
    Breach,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    Outcome,
    ResponseType,
)
from negmas.helpers import humanize_time
from negmas.sao import SAOState
from scml.oneshot import OneShotAgent
from scml.oneshot.agents import (
    GreedyOneShotAgent,
    RandomOneShotAgent,
    SingleAgreementAspirationAgent,
    SyncRandomOneShotAgent,
)
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE
from scml.utils import anac2021_collusion, anac2021_oneshot, anac2021_std

# required for development
from tabulate import tabulate

# consts
PRODUCT_STR = "product"
QUANTITY_STR = "quantity"

__all__ = [
    "TheSopranos78",
]


class TheSopranos78(OneShotAgent):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details

    """

    def __init__(self, *args, concession_exponent=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._e = concession_exponent

    def propose(self, negotiator_id: str, state) -> "Outcome":
        offers = self._get_best_offer_for_negotiation(negotiator_id)

        if not offers:
            return None

        offers = list(offers)
        offers[UNIT_PRICE] = self._find_good_price(self.get_nmi(negotiator_id), state)

        return tuple(offers)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        response = self._get_negotiation_status(negotiator_id, state, offer)

        if response != ResponseType.ACCEPT_OFFER:
            return response

        agent_machine_interface = self.get_nmi(negotiator_id)

        return (
            response
            if self._is_good_price(agent_machine_interface, state, offer[UNIT_PRICE])
            else ResponseType.REJECT_OFFER
        )

    def before_step(self):
        self.secured = 0

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement[QUANTITY_STR]

    def _get_negotiation_status(self, negotiator_id, state, offers):
        requirements = self._agent_requirements(negotiator_id)

        if requirements <= 0:
            return ResponseType.END_NEGOTIATION

        return (
            ResponseType.ACCEPT_OFFER
            if offers[QUANTITY] <= requirements
            else ResponseType.REJECT_OFFER
        )

    def _is_good_price(self, agent_machine_interface, state, price):
        min_value = agent_machine_interface.issues[UNIT_PRICE].min_value
        max_value = agent_machine_interface.issues[UNIT_PRICE].max_value
        threshold = self._threshold(state.step, agent_machine_interface.n_steps)

        if self._annotation_awi_product_equals(agent_machine_interface):
            return (price - min_value) >= threshold * (max_value - min_value)
        else:
            return (max_value - price) >= threshold * (max_value - min_value)

    def _find_good_price(self, agent_machine_interface, state):
        min_value = agent_machine_interface.issues[UNIT_PRICE].min_value
        max_value = agent_machine_interface.issues[UNIT_PRICE].max_value
        threshold = self._threshold(state.step, agent_machine_interface.n_steps)

        if self._annotation_awi_product_equals(agent_machine_interface):
            return min_value + threshold * (max_value - min_value)
        else:
            return max_value - threshold * (max_value - min_value)

    def _threshold(self, step, n_steps):
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e

    def _get_best_offer_for_negotiation(self, negotiator_id):
        requirements = self._agent_requirements(negotiator_id)

        if requirements <= 0:
            return None

        agent_machine_interface = self.get_nmi(negotiator_id)
        if not agent_machine_interface:
            return None

        quantity_issue = agent_machine_interface.issues[QUANTITY]
        unit_price_issue = agent_machine_interface.issues[UNIT_PRICE]

        offers = [-1] * 3
        offers[QUANTITY] = max(
            min(requirements, quantity_issue.max_value), quantity_issue.min_value
        )
        offers[TIME] = self.awi.current_step

        if self._annotation_awi_product_equals(agent_machine_interface):
            offers[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offers[UNIT_PRICE] = unit_price_issue.min_value

        return tuple(offers)

    def _agent_requirements(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def _annotation_awi_product_equals(self, agent_machine_interface):
        return (
            agent_machine_interface.annotation[PRODUCT_STR]
            == self.awi.my_output_product
        )


def run(competition="oneshot", reveal_names=True, n_steps=10, n_configs=2):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competition: The competition type to run (possibilities are oneshot, std,
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
    if competition == "oneshot":
        competitors = [MyAgent, RandomOneShotAgent, SyncRandomOneShotAgent]
        # competitors = [GreedyOneShotAgent, MyAgent, SingleAgreementAspirationAgent]
    else:
        from scml.scml2020.agents import BuyCheapSellExpensiveAgent, DecentralizingAgent

        competitors = [MyAgent, DecentralizingAgent, BuyCheapSellExpensiveAgent]

    start = time.perf_counter()
    if competition == "std":
        runner = anac2021_std
    elif competition == "collusion":
        runner = anac2021_collusion
    else:
        runner = anac2021_oneshot

    results = runner(
        competitors=competitors, verbose=True, n_steps=n_steps, n_configs=n_configs
    )
    # just make names shorter
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(
        "."
    ).str[-1]
    # display results
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "oneshot")
