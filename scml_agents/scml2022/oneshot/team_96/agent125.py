import numpy as np
from negmas import ResponseType
from scml.oneshot import *

OFFER_NOT_EXIST = None
EXPONENT = 0.2
__all__ = ["Agent125"]


class Agent125(OneShotAgent):
    """A greedy agent based on OneShotAgent"""

    def __init__(self):
        super().__init__()

    def before_step(self):
        self.secured = 0
        self.best_selling = []
        self.best_buying = []

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

    def propose(self, negotiator_id, state):
        offer = self.best_offer(negotiator_id)
        if not offer:
            return OFFER_NOT_EXIST
        price = self.find_good_price(self.get_nmi(negotiator_id), state)
        return (*offer, price)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        quantity, time, price = offer
        agent_needs = self.needs()
        if agent_needs <= 0:
            return ResponseType.END_NEGOTIATION
        if quantity <= agent_needs:
            return ResponseType.REJECT_OFFER
        nmi = self.get_nmi(negotiator_id)
        if not self.is_good_price(nmi, state, price):
            return ResponseType.REJECT_OFFER

        if self.is_selling(nmi):
            if len(self.best_selling) > 0:
                val = max(price, max(self.best_selling))
                if val not in self.best_selling:
                    self.best_selling.append(val)
            else:
                self.best_selling.append(price)
        else:
            if len(self.best_buying) > 0:
                val = min(price, min(self.best_buying))
                if val not in self.best_buying:
                    self.best_buying.append(val)
            else:
                self.best_buying.append(price)
        return ResponseType.ACCEPT_OFFER

    def best_offer(self, negotiator_id):
        agent_needs = self.needs()
        nmi = self.get_nmi(negotiator_id)
        if is_not_connected_to_simulator(agent_needs, nmi):
            return OFFER_NOT_EXIST
        quantity_issue, _, _ = nmi.issues
        offer_quantity = max(
            min(agent_needs, quantity_issue.max_value), quantity_issue.min_value
        )
        offer_time = self.awi.current_step
        return offer_quantity, offer_time

    def needs(self):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def is_selling(self, nmi):
        return nmi.annotation["product"] == self.awi.my_output_product

    def is_good_price(self, nmi, state, price):
        """Checks if a given price is good enough at this stage"""
        min_range, max_range = self.price_range(nmi)
        threshold = self.calculate_threshold(state.step, nmi.n_steps)
        # a good price is one better than the threshold
        if self.is_selling(nmi):
            return (price - min_range) >= threshold * (max_range - min_range)
        else:
            return (max_range - price) >= threshold * (max_range - min_range)

    def find_good_price(self, nmi, state):
        """Finds a good-enough price conceding linearly over time"""
        min_range, max_range = self.price_range(nmi)
        threshold = self.calculate_threshold(state.step, nmi.n_steps)
        # offer a price that is around th of your best possible price
        if self.is_selling(nmi):
            return min_range + threshold * (max_range - min_range)
        else:
            return max_range - threshold * (max_range - min_range)

    def price_range(self, nmi):
        """Finds the minimum and maximum prices"""
        _, _, simulator_price = nmi.issues
        min_range = simulator_price.min_value
        max_range = simulator_price.max_value
        if self.is_selling(nmi):
            min_range = max(min_range, np.mean(self.best_selling))
        else:
            max_range = (
                min(max_range, np.mean(self.best_buying))
                if self.best_buying
                else max_range
            )
        return min_range, max_range

    def calculate_threshold(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** EXPONENT


def is_not_connected_to_simulator(needs, nmi):
    return needs <= 0 or not nmi
