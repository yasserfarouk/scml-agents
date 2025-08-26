# -*- coding: utf-8 -*-
import math

from negmas import ResponseType
from scml.oneshot import *


# required for development
from scml.std import *

# required for typing
from negmas import Contract, Outcome

# tournament


__all__ = ["SimpleAgent", "AnalysisAgent"]


# BaseAgent
class SimpleAgent(OneShotAgent):
    """A greedy agent based on OneShotAgent"""

    def propose(self, negotiator_id: str, state) -> "Outcome":
        return self.best_offer(negotiator_id)

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
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

        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value), quantity_issue.min_value
        )
        offer[TIME] = self.awi.current_step
        offer[UNIT_PRICE] = self._find_good_price(ami)
        return tuple(offer)

    def _find_good_price(self, ami):
        """Finds a good-enough price."""
        unit_price_issue = ami.issues[UNIT_PRICE]
        if self._is_selling(ami):
            return unit_price_issue.max_value
        return unit_price_issue.min_value

    def is_seller(self, negotiator_id):
        return negotiator_id in self.awi.current_negotiation_details["sell"].keys()

    def _needed(self, negotiator_id=None):
        return (
            self.awi.needed_sales
            if self.is_seller(negotiator_id)
            else self.awi.needed_supplies
        )

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product


class AnalysisAgent(SimpleAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learn_range = 0.2

    def init(self):
        self.total_success_quantity = {
            k: 0
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }
        self.propose_times = {
            k: [0] * 10
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }
        self.success_times = {
            k: [0] * 10
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }

    def before_step(self):
        super().before_step()
        self.pre_propose = {
            k: 0
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        super().on_negotiation_success(contract, mechanism)
        partner_id = [p for p in contract.partners if p != self.id][0]
        self.total_success_quantity[partner_id] += (
            contract.agreement["quantity"] * self.awi.current_step / self.awi.n_steps
        )

        if self.name != mechanism.state.current_proposer:
            self.success_times[partner_id][contract.agreement["quantity"] - 1] += 1
            self.pre_propose[partner_id] = 0

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        if self.name == partners[1]:
            fail_negotiator = partners[0]
        else:
            fail_negotiator = partners[1]
        self.pre_propose[fail_negotiator] = 0

    def propose(self, negotiator_id: str, state) -> "Outcome":
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        ami = self.get_nmi(negotiator_id)
        offer = [-1] * 3
        offer[TIME] = self.awi.current_step
        offer[UNIT_PRICE] = self.find_price(best_price=True)

        if (
            self.awi.current_step <= self.awi.n_steps * self.learn_range
            or sum(self.total_success_quantity.values()) == 0
        ):
            offer[QUANTITY] = my_needs / len(self.total_success_quantity)
            offer[QUANTITY] = math.floor(offer[QUANTITY])
            offer[QUANTITY] = max(1, min(my_needs, offer[QUANTITY]))

        else:
            offer[QUANTITY] = (
                my_needs
                * self.total_success_quantity[negotiator_id]
                / sum(self.total_success_quantity.values())
            )
            offer[QUANTITY] = math.floor(offer[QUANTITY])
            offer[QUANTITY] = max(1, min(my_needs, offer[QUANTITY]))

            success_rate_1 = 0
            success_rate_2 = 0
            success_rate_3 = 0

            if offer[QUANTITY] != 1:
                if self.propose_times[negotiator_id][offer[QUANTITY] - 2] != 0:
                    success_rate_1 = (
                        self.success_times[negotiator_id][offer[QUANTITY] - 2]
                        / self.propose_times[negotiator_id][offer[QUANTITY] - 2]
                    )
                else:
                    success_rate_1 = 0

            if self.propose_times[negotiator_id][offer[QUANTITY] - 1] != 0:
                success_rate_2 = (
                    self.success_times[negotiator_id][offer[QUANTITY] - 1]
                    / self.propose_times[negotiator_id][offer[QUANTITY] - 1]
                )
            else:
                success_rate_2 = 0

            if offer[QUANTITY] != 10:
                if self.propose_times[negotiator_id][offer[QUANTITY]] != 0:
                    success_rate_3 = (
                        self.success_times[negotiator_id][offer[QUANTITY]]
                        / self.propose_times[negotiator_id][offer[QUANTITY]]
                    )
                else:
                    success_rate_3 = 0

                success_rate = max(success_rate_1, success_rate_2, success_rate_3)
                if success_rate == success_rate_2:
                    offer[QUANTITY] = offer[QUANTITY]
                elif success_rate == success_rate_1:
                    offer[QUANTITY] = offer[QUANTITY] - 1
                elif success_rate == success_rate_3:
                    offer[QUANTITY] = offer[QUANTITY] + 1

        if self.propose_times[negotiator_id][offer[QUANTITY] - 1] != 0:
            success_rate = (
                self.success_times[negotiator_id][offer[QUANTITY] - 1]
                / self.propose_times[negotiator_id][offer[QUANTITY] - 1]
            )
            if success_rate >= 0.3:
                self.pre_propose[negotiator_id] = offer[QUANTITY]

        self.propose_times[negotiator_id][math.floor(offer[QUANTITY]) - 1] += 1

        return tuple(offer)

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        return (
            ResponseType.ACCEPT_OFFER
            if offer[QUANTITY] <= my_needs - sum(self.pre_propose.values())
            else ResponseType.REJECT_OFFER
        )

    def find_price(self, best_price=False):
        seller = self.awi.is_first_level
        issues = (
            self.awi.current_output_issues if seller else self.awi.current_input_issues
        )
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        if best_price:
            return pmax if seller else pmin
        return pmin if seller else pmax
