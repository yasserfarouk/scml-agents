#!/usr/bin/env python
import statistics

from scml.oneshot import OneShotAgent

QUANTITY = 0
TIME = 1
UNIT_PRICE = 2
COMPROMISE_CONSTANT = 0.6

from negmas import Outcome, ResponseType

__all__ = [
    "SimpleAgent",
    "BetterAgent",
    "AgentOneOneTwo",
]


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
    def __init__(self, *args, concession_exponent=COMPROMISE_CONSTANT, **kwargs):
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
        if step == n_steps - 1:  # Last turn, take the offer
            return 0
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e


class AgentOneOneTwo(BetterAgent):
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


def average(list_items, n):
    if len(list_items) == 0:
        return n
    else:
        return statistics.mean(list_items)
