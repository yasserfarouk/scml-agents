#!/usr/bin/env python
# required for running tournaments and printing
import time
import statistics

# required for typing
from typing import Any, Dict, List, Optional

import numpy as np
from negmas.helpers import humanize_time
from negmas.sao import SAOState

# required for development
from scml.oneshot import OneShotAgent
from scml.oneshot.agents import RandomOneShotAgent, SyncRandomOneShotAgent
from scml.utils import anac2022_collusion, anac2022_oneshot, anac2022_std
from tabulate import tabulate

from scml.scml2020 import *
from scml.oneshot import *
import matplotlib.pyplot as plt
import pandas as pd

from negmas import save_stats

from scml.scml2020.world import is_system_agent
from collections import defaultdict


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

QUANTITY = 0
TIME = 1
UNIT_PRICE = 2

__all__ = ["AdaptivePercentile"]


class SimpleAgent(OneShotAgent):
    """A greedy agent based on OneShotAgent"""

    def before_step(self):
        self.secured = 0

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

    def propose(self, negotiator_id: str, state) -> "Outcome":
        return self.best_offer(negotiator_id)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        return (
            ResponseType.ACCEPT_OFFER
            if offer[QUANTITY] <= my_needs
            else ResponseType.REJECT_OFFER
        )

    def best_offer(self, negotiator_id):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        nmi = self.get_nmi(negotiator_id)
        if not nmi:
            return None
        quantity_issue = nmi.issues[QUANTITY]
        unit_price_issue = nmi.issues[UNIT_PRICE]
        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value), quantity_issue.min_value
        )
        offer[TIME] = self.awi.current_step
        if self._is_selling(nmi):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def _is_selling(self, nmi):
        return nmi.annotation["product"] == self.awi.my_output_product


class BetterAgent(SimpleAgent):
    def __init__(self, *args, concession_exponent=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._e = concession_exponent

    def propose(self, negotiator_id: str, state) -> "Outcome":
        offer = super().propose(negotiator_id, state)
        if not offer:
            return None
        offer = list(offer)
        offer[UNIT_PRICE] = self._find_good_price(self.get_nmi(negotiator_id), state)
        return tuple(offer)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state)
        if response != ResponseType.ACCEPT_OFFER:
            return response
        nmi = self.get_nmi(negotiator_id)
        return (
            response
            if self._is_good_price(nmi, state, offer[UNIT_PRICE])
            else ResponseType.REJECT_OFFER
        )

    def _is_good_price(self, nmi, state, price):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps)
        # a good price is one better than the threshold
        if self._is_selling(nmi):
            return (price - mn) >= th * (mx - mn)
        else:
            return (mx - price) >= th * (mx - mn)

    def _find_good_price(self, nmi, state):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps)
        # offer a price that is around th of your best possible price
        if self._is_selling(nmi):
            return mn + th * (mx - mn)
        else:
            return mx - th * (mx - mn)

    def _price_range(self, nmi):
        """Finds the minimum and maximum prices"""
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        return mn, mx

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e


class AdaptiveAgent(BetterAgent):
    """Considers best price offers received when making its decisions"""

    def before_step(self):
        super().before_step()
        self._best_selling, self._best_buying = 0.0, float("inf")

    def respond(self, negotiator_id, state):
        """Save the best price received"""
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state)
        ami = self.get_nmi(negotiator_id)
        if self._is_selling(ami):
            self._best_selling = max(offer[UNIT_PRICE], self._best_selling)
        else:
            self._best_buying = min(offer[UNIT_PRICE], self._best_buying)
        return response

    def _price_range(self, ami):
        """Limits the price by the best price received"""
        mn, mx = super()._price_range(ami)
        if self._is_selling(ami):
            mn = max(mn, self._best_selling)
        else:
            mx = min(mx, self._best_buying)
        return mn, mx


class AdaptivePercentile(BetterAgent):
    def before_step(self):
        super().before_step()
        # Save the last buying/selling prices
        self._best_selling = []
        self._best_buying = []

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state)
        nmi = self.get_nmi(negotiator_id)
        if self._is_selling(nmi):
            if len(self._best_selling) > 0:
                val = max(offer[UNIT_PRICE], max(self._best_selling))
                if val not in self._best_selling:
                    self._best_selling.append(val)
            else:
                self._best_selling.append(offer[UNIT_PRICE])
        else:
            if len(self._best_buying) > 0:
                val = min(offer[UNIT_PRICE], min(self._best_buying))
                if val not in self._best_buying:
                    self._best_buying.append(val)
            else:
                self._best_buying.append(offer[UNIT_PRICE])
        return response

    def _price_range(self, nmi):
        p = 1
        mn, mx = super()._price_range(nmi)
        if self._is_selling(nmi):
            mn = average(self._best_selling, mn) + p
        else:
            mx = average(self._best_buying, mx) - p
        return mn, mx


def average(list_items, n):
    if len(list_items) == 0:
        return n
    else:
        return statistics.mean(list_items)


def percentile(list_items, default, p):
    if len(list_items) == 0:
        return default
    else:
        return np.percentile(list_items, p)


class MMMAgentPercentile(AdaptivePercentile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._percentile = 20

    def _price_range(self, nmi):
        p = 1
        mn, mx = super()._price_range(nmi)
        if self._is_selling(nmi):
            mn = percentile(self._best_selling, mn, 20) + p
        else:
            mx = percentile(self._best_buying, mx, 80) - p
        return mn, mx


def run(
    competition="oneshot",
    reveal_names=True,
    n_steps=5,
    n_configs=15,
):
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
        competitors = [MMMAgentPercentile, AdaptiveAgent, BetterAgent]  #
    else:
        from scml.scml2020.agents import BuyCheapSellExpensiveAgent, DecentralizingAgent

        competitors = [
            SimpleAgent,
            MyAgent,
            DecentralizingAgent,
            BuyCheapSellExpensiveAgent,
        ]

    start = time.perf_counter()
    if competition == "std":
        runner = anac2022_std
    elif competition == "collusion":
        runner = anac2022_collusion
    else:
        runner = anac2022_oneshot
    results = runner(
        competitors=competitors,
        verbose=True,
        n_steps=n_steps,
        n_configs=n_configs,
    )
    # just make names shorter
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(
        "."
    ).str[-1]
    # display results
    pass  # print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    pass  # print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "oneshot")
