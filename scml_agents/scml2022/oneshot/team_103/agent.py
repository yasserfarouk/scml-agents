#!/sr/bin/env python
# required for running tournaments and printing
import time
import statistics

# required for typing
from typing import Any, Dict, List, Optional

import numpy as np
from negmas.helpers import humanize_time
from negmas.sao import SAOState
from collections import defaultdict

# required for development
from scml.oneshot import OneShotAgent
from scml.oneshot.agents import RandomOneShotAgent, SyncRandomOneShotAgent
from scml.utils import anac2022_collusion, anac2022_oneshot, anac2022_std
from tabulate import tabulate

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
__all__ = ["MMMPersonalized"]


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


class Winner(BetterAgent):
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
        mn, mx = super()._price_range(nmi)
        if self._is_selling(nmi):
            mn = max(mn, average(self._best_selling, mn))
        else:
            mx = min(mx, average(self._best_buying, mx))
        return mn, mx


#
class AgentX(BetterAgent):
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

    def propose(self, negotiator_id: str, state) -> "Outcome":
        offer = super().propose(negotiator_id, state)
        if not offer:
            return None
        offer = list(offer)
        nmi = self.get_nmi(negotiator_id)
        offer[UNIT_PRICE] = self._find_good_price(nmi, state)

        if self._is_selling(nmi):
            if len(self._best_selling) > 0:
                offer[UNIT_PRICE] = offer[UNIT_PRICE] + max(self._best_selling)
        else:
            if len(self._best_buying) > 0:
                offer[UNIT_PRICE] = offer[UNIT_PRICE] + min(self._best_buying)

        return tuple(offer)

    def _price_range(self, nmi):
        mn, mx = super()._price_range(nmi)
        if self._is_selling(nmi):
            mn = max(mn, average(self._best_selling, mn))
        else:
            mx = min(mx, average(self._best_buying, mx))
        return mn, mx


def average(list_items, n):
    if len(list_items) == 0:
        return n
    else:
        return statistics.mean(list_items)


class MMMPersonalized(BetterAgent):
    def __init__(self, *args, concession_exponent=0.2, **kwargs):
        super().__init__(*args, concession_exponent=concession_exponent, **kwargs)
        self.nrejections = 2
        self.id2seller_deals = defaultdict(list)
        self.id2buyer_deals = defaultdict(list)
        self.new_discount = 2
        self.increase_rate = 1

    def before_step(self):
        super().before_step()
        self.buy_offers = []
        self.sell_offers = []
        self.id2reject_counter = defaultdict(int)

    def step(self):
        super().step()
        # print('Sellers:')
        # print(self.sell_offers)
        # if len(self.buy_offers) > 0:
        #     print('Buyers:')
        #     print(self.buy_offers)
        # print('Sellers:')
        # print(self.id2seller_deals)
        # print('Buyers:')
        # print(self.id2buyer_deals)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state)

        if response == ResponseType.ACCEPT_OFFER:
            if self.id2reject_counter[negotiator_id] > self.nrejections:
                return ResponseType.ACCEPT_OFFER
            else:
                self.id2reject_counter[negotiator_id] += 1
                return ResponseType.REJECT_OFFER

        nmi = self.get_nmi(negotiator_id)

        if self._is_selling(nmi):
            self.buy_offers.append(offer)
        else:
            self.sell_offers.append(offer)

        if response == ResponseType.ACCEPT_OFFER:
            if self._is_selling(nmi):
                self.id2buyer_deals[negotiator_id].append(offer)
            else:
                self.id2seller_deals[negotiator_id].append(offer)
        return response

    def propose(self, negotiator_id: str, state) -> "Outcome":
        offer = super().propose(negotiator_id, state)
        nmi = self.get_nmi(negotiator_id)
        if not offer:
            return None
        offer = list(offer)
        offer[UNIT_PRICE] = self._find_good_price(self.get_nmi(negotiator_id), state)
        if self._is_selling(nmi):
            # offer[UNIT_PRICE] = 0.5*(np.mean([z[2] for z in self.buy_offers] + [offer[UNIT_PRICE]]) + offer[UNIT_PRICE])
            if negotiator_id not in self.id2buyer_deals.keys():
                offer[UNIT_PRICE] -= self.new_discount
            else:
                offer[UNIT_PRICE] = (
                    self.id2buyer_deals[negotiator_id][-1][UNIT_PRICE]
                    + self.increase_rate
                )
        else:
            # offer[UNIT_PRICE] = 0.5 * (np.mean([z[2] for z in self.sell_offers] + [offer[UNIT_PRICE]]) + offer[UNIT_PRICE])
            if negotiator_id not in self.id2seller_deals.keys():
                offer[UNIT_PRICE] += self.new_discount
            else:
                offer[UNIT_PRICE] = (
                    self.id2seller_deals[negotiator_id][-1][UNIT_PRICE]
                    - self.increase_rate
                )
        return tuple(offer)


#
# class MyAgent(OneShotAgent):
#     """
#     This is the only class you *need* to implement. The current skeleton has a
#     basic do-nothing implementation.
#     You can modify any parts of it as you need. You can act in the world by
#     calling methods in the agent-world-interface instantiated as `self.awi`
#     in your agent. See the documentation for more details
#
#     """
#
#     # =====================
#     # Negotiation Callbacks
#     # =====================
#
#     def propose(self, negotiator_id: str, state: SAOState) -> Optional[Outcome]:
#         """Called when the agent is asking to propose in one negotiation"""
#         pass
#
#     def respond(
#         self, negotiator_id: str, state: SAOState, offer: Outcome
#     ) -> ResponseType:
#         """Called when the agent is asked to respond to an offer"""
#         return ResponseType.END_NEGOTIATION
#
#     # =====================
#     # Time-Driven Callbacks
#     # =====================
#
#     def init(self):
#         """Called once after the agent-world interface is initialized"""
#         pass
#
#     def before_step(self):
#         """Called at at the BEGINNING of every production step (day)"""
#
#     def step(self):
#         """Called at at the END of every production step (day)"""
#
#     # ================================
#     # Negotiation Control and Feedback
#     # ================================
#
#     def on_negotiation_failure(
#         self,
#         partners: List[str],
#         annotation: Dict[str, Any],
#         mechanism: AgentMechanismInterface,
#         state: MechanismState,
#     ) -> None:
#         """Called when a negotiation the agent is a party of ends without
#         agreement"""
#
#     def on_negotiation_success(
#         self, contract: Contract, mechanism: AgentMechanismInterface
#     ) -> None:
#         """Called when a negotiation the agent is a party of ends with
#         agreement"""
#


def run(
    competition="oneshot",
    reveal_names=True,
    n_steps=5,
    n_configs=2,
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
        competitors = [
            MMMPersonalized,
            AgentX,
            Winner,
            SimpleAgent,
            BetterAgent,
        ]  # , SyncRandomOneShotAgent, RandomOneShotAgent] #
    else:
        from scml.scml2020.agents import BuyCheapSellExpensiveAgent, DecentralizingAgent

        competitors = [
            SimpleAgent,
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
