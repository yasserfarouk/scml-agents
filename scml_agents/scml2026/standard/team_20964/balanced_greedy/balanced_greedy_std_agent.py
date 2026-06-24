#!/usr/bin/env python
from __future__ import annotations

import sys

from negmas import ResponseType
from scml.std import StdAgent, QUANTITY, TIME, UNIT_PRICE


class BalancedGreedyStdAgent(StdAgent):
    """
    First simple Standard-track agent.

    This is a minimal working agent:
    - proposes simple offers
    - accepts offers with reasonable prices
    - avoids past delivery dates
    """

    def init(self):
        self._secured_buy = 0
        self._secured_sell = 0

    def before_step(self):
        pass

    def on_negotiation_success(self, contract, mechanism):
        quantity = contract.agreement["quantity"]
        if contract.annotation["seller"] == self.id:
            self._secured_sell += quantity
        else:
            self._secured_buy += quantity

    def step(self):
        pass

    def propose(self, negotiator_id, state, source=None):
        nmi = self.get_nmi(negotiator_id)
        if nmi is None:
            return None

        issues = nmi.issues
        q_issue = issues[QUANTITY]
        t_issue = issues[TIME]
        p_issue = issues[UNIT_PRICE]

        needed = self._needed(negotiator_id)

        if needed <= 0:
            return None

        quantity = min(needed, q_issue.max_value)
        quantity = max(quantity, q_issue.min_value)
        if self._is_selling_to(negotiator_id):
            target_time = self.awi.current_step + 2
        else:
            target_time = self.awi.current_step

        delivery_time = max(target_time, t_issue.min_value)
        delivery_time = min(delivery_time, t_issue.max_value)
        min_price = p_issue.min_value
        max_price = p_issue.max_value
        mid_price = (min_price + max_price) / 2

        relative_time = getattr(state, "relative_time", 0.0)
        relative_time = max(0.0, min(1.0, relative_time))

        if self._is_selling_to(negotiator_id):
            unit_price = max_price - relative_time * (max_price - mid_price)
        else:
            unit_price = min_price + relative_time * (mid_price - min_price)

        return quantity, delivery_time, unit_price

    def respond(self, negotiator_id, state, source=None):
        offer = state.current_offer

        if offer is None:
            return ResponseType.REJECT_OFFER

        quantity = offer[QUANTITY]
        delivery_time = offer[TIME]
        unit_price = offer[UNIT_PRICE]

        if quantity <= 0:
            return ResponseType.REJECT_OFFER

        if delivery_time < self.awi.current_step:
            return ResponseType.REJECT_OFFER

        needed = self._needed(negotiator_id)

        if needed <= 0:
            return ResponseType.REJECT_OFFER

        if quantity > needed:
            return ResponseType.REJECT_OFFER
        
        nmi = self.get_nmi(negotiator_id)
        if nmi is None:
            return ResponseType.REJECT_OFFER

        p_issue = nmi.issues[UNIT_PRICE]
        min_price = p_issue.min_value
        max_price = p_issue.max_value
        mid_price = (min_price + max_price) / 2

        relative_time = getattr(state, "relative_time", 0.0)
        relative_time = max(0.0, min(1.0, relative_time))

        if self._is_selling_to(negotiator_id):
            acceptable_price = max_price - relative_time * (max_price - mid_price)
            if unit_price >= acceptable_price:
                return ResponseType.ACCEPT_OFFER
        else:
            acceptable_price = min_price + relative_time * (mid_price - min_price)
            if unit_price <= acceptable_price:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
    def _is_selling_to(self, partner):
        return partner in self.awi.my_consumers
    def _needed(self, negotiator_id):
        """この相手に対して、あと何個やり取りすべきかを返す。"""
        if self._is_selling_to(negotiator_id):
            return max(0, self.awi.needed_sales)
        else:
            return max(0, self.awi.needed_supplies)


if __name__ == "__main__":
    from .helpers.runner import run

    run([BalancedGreedyStdAgent], sys.argv[1] if len(sys.argv) > 1 else "std")
