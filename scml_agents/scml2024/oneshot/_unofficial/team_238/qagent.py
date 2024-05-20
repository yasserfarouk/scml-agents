#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Authors*  Burcu-Kaya-burcu.kaya-15288@ozu.edu.tr-Kağan-Güngör-kagan.gungor@ozu.edu.tr
This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""

from __future__ import annotations

import numpy as np

from negmas import ResponseType, Outcome
from scml.oneshot.world import SCML2024OneShotWorld as W
from scml.oneshot import *

import random


class QAgent(OneShotAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_table = {}
        self.alpha = 0.01
        self.gamma = 0.5
        self.epsilon = 0.6

    def propose(self, negotiator_id: str, state) -> "Outcome":
        if (
            random.random() < self.epsilon
            and random.random() < self.gamma
            and random.random() < self.alpha
        ):
            return self.random_offer(negotiator_id)
        else:
            return self.best_q_offer(negotiator_id)

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

    def best_q_offer(self, negotiator_id, state=None):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]

        possible_offers = [
            (
                min(my_needs, quantity_issue.max_value),
                self.awi.current_step,
                self._find_good_price(ami),
            )
            for _ in range(10)
        ]
        best_offer = max(possible_offers, key=lambda x: self.q_table.get((state, x), 0))
        return tuple(best_offer)

    def random_offer(self, negotiator_id):
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        offer = []
        for issue in ami.issues:
            if issue.name == QUANTITY:
                offer.append(random.randint(issue.min_value, issue.max_value))
            elif issue.name == UNIT_PRICE:
                offer.append(random.randint(issue.min_value, issue.max_value))
            elif issue.name == TIME:
                offer.append(self.awi.current_step)
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
