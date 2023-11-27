# -*- coding: utf-8 -*-
from collections import defaultdict

from negmas import ResponseType
from negmas.outcomes import Outcome
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE, OneShotAgent

__all__ = ["CCAgent"]


class SimpleAgent(OneShotAgent):
    """A greedy agent based on OneShotAgent"""

    def before_step(self):
        self.secured = 0

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

    def propose(self, negotiator_id: str, state) -> Outcome | None:
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
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None

        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value), quantity_issue.min_value
        )
        offer[TIME] = self.awi.current_step
        if self._is_selling(ami):
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

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product


class CCAgent(SimpleAgent):
    def before_step(self):
        super().before_step()

        if self.awi.level == 0:
            self.q_need = self.awi.state.exogenous_input_quantity
        elif self.awi.level == 1:
            self.q_need = self.awi.state.exogenous_output_quantity

        self.q_opp_offer = defaultdict(lambda: float("inf"))

    def propose(self, negotiator_id: str, state) -> Outcome | None:
        ami = self.get_nmi(negotiator_id)
        offer = super().propose(negotiator_id, state)
        if offer is None:
            return None
        offer = list(offer)

        # compare q_opp_offer and q_need (cooperate)
        if self._is_selling(ami):
            opponent = ami.annotation["buyer"]
            offer[QUANTITY] = min(self.q_opp_offer[opponent], self.q_need)
        else:
            opponent = ami.annotation["seller"]
            offer[QUANTITY] = min(self.q_opp_offer[opponent], self.q_need)

        return tuple(offer)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        # update q_opp_offer
        ami = self.get_nmi(negotiator_id)
        q = offer[QUANTITY]

        if self._is_selling(ami):
            opponent = ami.annotation["buyer"]
            self.q_opp_offer[opponent] = q
        else:
            opponent = ami.annotation["seller"]
            self.q_opp_offer[opponent] = q

        # response (compromise)
        if self.q_need <= 0:
            return ResponseType.END_NEGOTIATION

        else:
            if q <= self.q_need + 1:
                return ResponseType.ACCEPT_OFFER
            else:
                return ResponseType.REJECT_OFFER

    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)

        # update q_need
        self.q_need -= contract.agreement["quantity"]