#!/usr/bin/env python
"""
**Submitted to ANAC 2023 SCML (OneShot track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2023 SCML.
"""
from __future__ import annotations

# used to repeat the response to every negotiator.
import itertools
import random
import re

# required for running tournaments and printing
import time
from collections import defaultdict
from pprint import pprint
from time import sleep

# required for typing
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from negmas import Contract, Outcome, ResponseType, SAOState
from negmas.helpers import humanize_time
from negmas.sao import SAOResponse, SAOState

# required for development
from scml.oneshot import *
from scml.oneshot import OneShotAWI, OneShotSyncAgent
from scml.scml2020 import *
from scml.utils import (
    anac2022_oneshot,
    anac2023_collusion,
    anac2023_oneshot,
    anac2023_std,
)
from tabulate import tabulate

import scml_agents

__all__ = ["Shochan"]


class SimpleAgent(OneShotAgent):
    """A greedy agent based on OneShotAgent"""

    def before_step(self):
        self.secured = 0

    def on_negotiation_success(self, contract, mechanism):
        # print("------SUCCESS-----")
        # print(contract)
        # print("------------------")
        self.secured += contract.agreement["quantity"]

    def propose(self, negotiator_id: str, state) -> "Outcome":
        # print("----BESTOFFER-----")
        # print(self.best_offer(negotiator_id))
        # print("------------------")
        return self.best_offer(negotiator_id)

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        my_needs = self._needed(negotiator_id)
        # print(f'----{negotiator_id}`s-OFFER-----')
        # print(offer)
        # print("---------------------------------")

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


class BetterAgent(SimpleAgent):
    """A greedy agent based on OneShotAgent with more sane strategy"""

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

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state, source)
        if response != ResponseType.ACCEPT_OFFER:
            return response
        ami = self.get_nmi(negotiator_id)
        return (
            response
            if self._is_good_price(ami, state, offer[UNIT_PRICE])
            else ResponseType.REJECT_OFFER
        )

    def _is_good_price(self, ami, state, price):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(ami)
        th = self._th(state.step, ami.n_steps)
        # a good price is one better than the threshold
        if self._is_selling(ami):
            return (price - mn) >= th * (mx - mn)
        else:
            return (mx - price) >= th * (mx - mn)

    def _find_good_price(self, ami, state):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(ami)
        th = self._th(state.step, ami.n_steps)
        # offer a price that is around th of your best possible price
        if self._is_selling(ami):
            return mn + th * (mx - mn)
        else:
            return mx - th * (mx - mn)

    def _price_range(self, ami):
        """Finds the minimum and maximum prices"""
        mn = ami.issues[UNIT_PRICE].min_value
        mx = ami.issues[UNIT_PRICE].max_value
        return mn, mx

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e


class AdaptiveAgent(BetterAgent):
    """Considers best price offers received when making its decisions"""

    def before_step(self):
        super().before_step()
        self._best_selling, self._best_buying = 0.0, float("inf")

    def respond(self, negotiator_id, state, source=""):
        """Save the best price received"""
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state, source)
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


class LearningAgent(AdaptiveAgent):
    def __init__(
        self,
        *args,
        acc_price_slack=float("inf"),
        step_price_slack=0.0,
        opp_price_slack=0.0,
        opp_acc_price_slack=0.2,
        range_slack=0.03,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._acc_price_slack = acc_price_slack
        self._step_price_slack = step_price_slack
        self._opp_price_slack = opp_price_slack
        self._opp_acc_price_slack = opp_acc_price_slack
        self._range_slack = range_slack

    def init(self):
        """Initialize the quantities and best prices received so far"""
        super().init()
        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._best_opp_acc_selling = defaultdict(float)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))

    def step(self):
        """Initialize the quantities and best prices received for next step"""
        super().step()
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))

    def on_negotiation_success(self, contract, mechanism):
        """Record sales/supplies secured"""
        super().on_negotiation_success(contract, mechanism)
        # print(self.awi.current_exogenous_input_quantity)
        # print(self.awi.current_exogenous_output_quantity)

        # update my current best price to use for limiting concession in other
        # negotiations
        up = contract.agreement["unit_price"]
        if self._is_selling(mechanism):
            partner = contract.annotation["buyer"]
            self._best_acc_selling = max(up, self._best_acc_selling)
            self._best_opp_acc_selling[partner] = max(
                up, self._best_opp_acc_selling[partner]
            )
        else:
            partner = contract.annotation["seller"]
            self._best_acc_buying = min(up, self._best_acc_buying)
            self._best_opp_acc_buying[partner] = min(
                up, self._best_opp_acc_buying[partner]
            )

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        # find the quantity I still need and end negotiation if I need nothing more
        response = super().respond(negotiator_id, state, source)
        # update my current best price to use for limiting concession in other
        # negotiations
        ami = self.get_nmi(negotiator_id)
        # print(ami.issues)
        up = offer[UNIT_PRICE]
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            self._best_opp_selling[partner] = max(up, self._best_selling)
        else:
            partner = ami.annotation["seller"]
            self._best_opp_buying[partner] = min(up, self._best_buying)
        return response

    def _price_range(self, ami):
        """Limits the price by the best price received"""
        mn = ami.issues[UNIT_PRICE].min_value
        mx = ami.issues[UNIT_PRICE].max_value
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            mn = min(
                mx * (1 - self._range_slack),
                max(
                    [mn]
                    + [
                        p * (1 - slack)
                        for p, slack in (
                            (self._best_selling, self._step_price_slack),
                            (self._best_acc_selling, self._acc_price_slack),
                            (self._best_opp_selling[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_selling[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        else:
            partner = ami.annotation["seller"]
            mx = max(
                mn * (1 + self._range_slack),
                min(
                    [mx]
                    + [
                        p * (1 + slack)
                        for p, slack in (
                            (self._best_buying, self._step_price_slack),
                            (self._best_acc_buying, self._acc_price_slack),
                            (self._best_opp_buying[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_buying[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        return mn, mx


class Shochan(AdaptiveAgent):
    def __init__(self, *args, concession_exponent=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._e = concession_exponent

    def before_step(self, concession_exponent=0.2):
        super().before_step()
        self._best_selling, self._best_buying = 0.0, float("inf")
        self.now = False

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

        if self._needed(self.name) <= 0:
            self.now = True
        else:
            self.now = False

        if self.awi.my_output_product == 1:
            self._best_selling = max(
                contract.agreement["unit_price"], self._best_selling
            )
        else:
            self._best_buying = min(contract.agreement["unit_price"], self._best_buying)

    def propose(self, negotiator_id: str, state) -> "Outcome":
        offer = self.best_offer(negotiator_id)

        if self.now == True:
            return None
        return offer

    def best_offer(self, negotiator_id):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        offer = [-1] * 3
        offer[TIME] = self.awi.current_step

        umin = ami.issues[UNIT_PRICE].value_at(0)
        umax = ami.issues[UNIT_PRICE].value_at(1)
        t = (ami.n_steps - 1 - ami.state.step) / (ami.n_steps - 1)
        t = t**self._e

        num = ami.state.step

        if self.awi.my_output_product == 2:
            offer[UNIT_PRICE] = umin
            if num == 1:
                offer[QUANTITY] = my_needs
            elif num == 2:
                offer[QUANTITY] = max(my_needs - 1, 0)
            elif num == 6:
                offer[UNIT_PRICE] = umax
                offer[QUANTITY] = my_needs
            elif num % 2 == 1:
                if my_needs > ((self.awi.current_exogenous_output_quantity + 1) // 2):
                    offer[QUANTITY] = (my_needs + 1) // 2
            else:
                offer[QUANTITY] = my_needs

        else:
            offer[UNIT_PRICE] = umax
            if num == 1:
                offer[QUANTITY] = my_needs
            elif num == 2:
                offer[QUANTITY] = max(my_needs - 1, 0)
            elif num == 6:
                offer[UNIT_PRICE] = umin
                offer[QUANTITY] = my_needs
            elif num % 2 == 1:
                if my_needs > ((self.awi.current_exogenous_output_quantity + 1) // 2):
                    offer[QUANTITY] = (my_needs + 1) // 2
            else:
                offer[QUANTITY] = my_needs

        return tuple(offer)

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None

        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            response = ResponseType.END_NEGOTIATION
        elif offer[QUANTITY] <= my_needs:
            response = ResponseType.ACCEPT_OFFER
        else:
            response = ResponseType.REJECT_OFFER

        if response == ResponseType.ACCEPT_OFFER:
            if not self._is_good_price(ami, state, offer[UNIT_PRICE]):
                response = ResponseType.REJECT_OFFER

        if self.now == True:
            return ResponseType.END_NEGOTIATION
        else:
            dis = self.awi.current_disposal_cost
            sho = self.awi.current_shortfall_penalty
            tp0 = self.awi.trading_prices[0]
            tp1 = self.awi.trading_prices[1]
            tp2 = self.awi.trading_prices[2]
            dis_q = (
                offer[QUANTITY]
                + self.secured
                - self.awi.current_exogenous_input_quantity
            )
            sho_q = self.awi.current_exogenous_input_quantity - self.secured
            if response == ResponseType.ACCEPT_OFFER:
                if self._is_selling(ami):
                    if sho * tp0 * (min(offer[QUANTITY], sho_q)) < dis * tp1 * max(
                        0, dis_q
                    ):
                        response = ResponseType.REJECT_OFFER
                else:
                    if sho * tp1 * (min(offer[QUANTITY], sho_q)) < dis * tp2 * max(
                        0, dis_q
                    ):
                        response = ResponseType.REJECT_OFFER

                    if state.step < 2:
                        if offer[QUANTITY] > (
                            (self.awi.current_exogenous_output_quantity + 1) // 2
                        ):
                            return ResponseType.REJECT_OFFER

                if response == ResponseType.ACCEPT_OFFER:
                    self.now = True

            if state.step > 6:
                if self._is_selling(ami):
                    if sho * tp0 * (min(offer[QUANTITY], sho_q)) > dis * tp1 * max(
                        0, dis_q
                    ):
                        response = ResponseType.ACCEPT_OFFER
                else:
                    if sho * tp1 * (min(offer[QUANTITY], sho_q)) > dis * tp2 * max(
                        0, dis_q
                    ):
                        response = ResponseType.ACCEPT_OFFER

                if response == ResponseType.ACCEPT_OFFER:
                    self.now = True

        return response

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def _is_good_price(self, ami, state, price):
        """Checks if a given price is good enough at this stage"""
        tp0, tp1, tp2 = self._price_range(ami)
        th = self._th(state.step, ami.n_steps)

        if self._is_selling(ami):
            return (price - tp0) >= th * (tp1 - tp0)
        else:
            return (tp2 - price) >= th * (tp2 - tp1)

    def _price_range(self, ami):
        """Finds the minimum and maximum prices"""

        tp0 = self.awi.trading_prices[0]
        tp1 = self.awi.trading_prices[1]
        tp2 = self.awi.trading_prices[2]
        return tp0, tp1, tp2

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e
