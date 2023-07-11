#!/usr/bin/env python
"""
**Submitted to ANAC 2021 SCML (OneShot track)**
*Authors* type-your-team-member-names-with-their-emails here


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

# required for typing
import logging
import math
import pickle
import random

# required for running tournaments and printing
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, Union

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
    SAOResponse,
)
from negmas.helpers import humanize_time
from negmas.outcomes import Issue
from negmas.preferences import UtilityFunction
from negmas.sao import (
    AspirationNegotiator,
    NaiveTitForTatNegotiator,
    NiceNegotiator,
    SimpleTitForTatNegotiator,
    TopFractionNegotiator,
    ToughNegotiator,
)

# required for development
from scml.oneshot import OneShotAgent
from scml.oneshot.agent import *
from scml.oneshot.agents import (
    GreedyOneShotAgent,
    GreedySingleAgreementAgent,
    GreedySyncAgent,
    RandomOneShotAgent,
    SyncRandomOneShotAgent,
)
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE
from scml.utils import anac2021_collusion, anac2021_oneshot, anac2021_std
from tabulate import tabulate

# warnings.filterwarnings('error')

__all__ = [
    # "StagHunter",
    # "StagHunterV5",
    # "StagHunterV6",
    "StagHunterV7",
]


class StagHunter(OneShotAgent):
    def __init__(
        self,
        *args,
        acc_price_slack=0.1,  # accumlated agreed best price among all oppo
        acc_price_slack2=0.2,
        step_price_slack=0.15,  # current encounted best prices among all oppo
        step_price_slack2=0.15,
        opp_acc_price_slack=0.15,  # per-oppo accumlated agreed best price
        opp_acc_price_slack2=0.15,
        opp_price_slack=0.2,  # per-oppo best prices encounted
        opp_price_slack2=0.1,
        step_agg_price_slack=0.15,  # current encounted agreed best prices
        step_agg_price_slack2=0.15,
        range_slack=0.1,
        trading_price_slack=0.15,
        trading_price_slack2=0.05,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # self._e = concession_exponent
        self._acc_price_slack = acc_price_slack
        self._acc_price_slack2 = acc_price_slack2
        self._step_price_slack = step_price_slack
        self._step_price_slack2 = step_price_slack2
        self._opp_acc_price_slack = opp_acc_price_slack
        self._opp_acc_price_slack2 = opp_acc_price_slack2
        self._opp_price_slack = opp_price_slack
        self._opp_price_slack2 = opp_price_slack2
        self._step_agg_price_slack = step_agg_price_slack
        self._step_agg_price_slack2 = step_agg_price_slack2
        self._range_slack = range_slack
        self._trading_price_slack = trading_price_slack
        self._trading_price_slack2 = trading_price_slack2
        self._num_of_consider_day_price = 3
        self._num_of_consider_day_quantity = 3

    def init(self):
        """Initialize the quantities and best prices received so far"""
        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_acc_selling = defaultdict(lambda: 0.0)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))

        self._best_opp_acc_prices = (
            defaultdict(lambda: [0.0])
            if self.awi.is_first_level
            else defaultdict(lambda: [float("inf")])
        )
        self._best_acc_prices = [0.0] if self.awi.is_first_level else [float("inf")]

        self._prev_opp_quantity = defaultdict(lambda: [])
        self._prev_opp_price = defaultdict(lambda: [])
        self._prev_self_price = []
        self._secured = 0
        self.oppo_list = (
            self.awi.my_consumers if self.awi.is_first_level else self.awi.my_suppliers
        )
        self._e = defaultdict(lambda: 1)

        # self._best_selling, self._best_buying = 0.0, float("inf")
        # self._best_opp_selling = defaultdict(float)
        # self._best_opp_buying = defaultdict(lambda: float("inf"))
        # self._prev_opp_quantity = defaultdict(lambda: [])
        # seslf._prev_opp_price = defaultdict(lambda: [])
        # self._prev_self_price = []
        #
        #
        # self._secured = 0
        # self.oppo_list = self.awi.my_consumers if self.awi.is_first_level else self.awi.my_suppliers
        # self._e = defaultdict(lambda:1)

        # self._e = defaultdict(lambda: 0.2 + random.random() * 0.8)

        # self._acc_rev = {k: [] for k in self.oppo_list}
        # self._range_slack = random.random() * 0.2 + 0.05
        # self._step_price_slack = random.random() * 0.2 + 0.05
        # self._opp_price_slack = random.random() * 0.2 + 0.1

        self.cur_offer_list = {}
        # self._oppo_neg_history = {k: [] for k in self.oppo_list}
        # self._max_util_dist = {k: np.zeros(100) for k in self.oppo_list}

        # self._lr = defaultdict(lambda:0.5)
        # self._is_prev_agg = defaultdict(lambda:False)
        # self._prev_e = defaultdict(lambda:[1])
        self._prev_mag_profit = defaultdict(lambda: [0])

        self._rank = defaultdict(lambda: 1)

        # if(self.awi.is_first_level):
        #     self._quantity_slack = [3/4 for _ in range(len(self.oppo_list))]
        # else:
        self._quantity_slack = [2 / 3 for _ in range(len(self.oppo_list))]
        # self._quantity_slack = [2/3 for _ in range(len(self.oppo_list))]

        for oppo_id in self.oppo_list:
            # toss = np.random.uniform()

            self._e[oppo_id] = np.random.uniform(0.01, 0.5)

    def before_step(self):
        self._best_selling, self._best_buying = 0.0, float("inf")

        self._best_opp_selling = defaultdict(lambda: 0.0)
        self._best_opp_buying = defaultdict(lambda: float("inf"))

        self._cur_best_price = 0 if self.awi.is_first_level else float("inf")

        self._cur_rank = deepcopy(self._rank)

    def step(self, lr=0.1):
        """Initialize the quantities and best prices received for next step"""
        self._best_selling, self._best_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))

        self._prev_opp_quantity = defaultdict(lambda: [])
        self._prev_opp_price = defaultdict(lambda: [])
        self._prev_self_price = []

        self._secured = 0

        u_total = self.ufun.from_offers(
            tuple(tuple(_) for _ in self.cur_offer_list.values()),
            tuple([self.awi.is_first_level] * (len(self.cur_offer_list))),
        )

        for oppo_id in self.oppo_list:
            if oppo_id in self.cur_offer_list:
                cur_offer = []
                for id_p in self.cur_offer_list:
                    if id_p != oppo_id:
                        cur_offer.append(self.cur_offer_list[id_p])

                u_p = self.ufun.from_offers(
                    tuple(tuple(_) for _ in cur_offer),
                    tuple([self.awi.is_first_level] * (len(cur_offer))),
                )

                # if(u_total-u_p >= self._prev_best_mag_profit[oppo_id]):
                self._prev_mag_profit[oppo_id].append(u_total - u_p)
                # self._prev_e[oppo_id].append(self._e[oppo_id])

                # old_inc = 0 if not len(self._acc_rev[oppo_id]) else np. median(self._acc_rev[oppo_id][-3:])
                #
                # inc = u_total-u_p
                # if not self._is_prev_agg[oppo_id]:
                #     self._lr[oppo_id] /= 2
                #     self._lr[oppo_id] = max(0.05, self._lr[oppo_id])
                #
                # self._e[oppo_id] = max(0.1, self._e[oppo_id] * (1-self._lr[oppo_id]))
                # #
                # self._is_prev_agg[oppo_id] = True

            else:
                # self._acc_rev[oppo_id].append(0)
                # old_e = self._e[oppo_id]
                # if self._is_prev_agg[oppo_id]:
                #     idx = np.argmax(self._prev_mag_profit[oppo_id][-3:]) - len(self._prev_mag_profit[oppo_id][-3:])
                #     self._e[oppo_id] = self._prev_e[oppo_id][idx]
                # self._prev_e[oppo_id].append(old_e)
                self._prev_mag_profit[oppo_id].append(0)

                # self._e[oppo_id] = min(self._e[oppo_id] * (1+self._lr[oppo_id]), 10)
                # self._is_prev_agg[oppo_id] = False

        if self.awi.current_step > 0:

            revs = [
                (oppo_id, np.mean(self._prev_mag_profit[oppo_id][1:]))
                for oppo_id in self.oppo_list
            ]
            revs = sorted(revs, key=lambda x: x[1], reverse=True)

            self._rank = {revs[i][0]: i for i in range(len(revs))}

        self.cur_offer_list = {}
        self._oppo_neg_history = {k: [] for k in self.oppo_list}
        self._best_acc_prices.append(self._cur_best_price)

        for oppo_id in self.oppo_list:

            if self.awi.is_first_level:
                self._best_opp_acc_selling[oppo_id] = max(
                    self._best_opp_acc_prices[oppo_id][
                        -self._num_of_consider_day_price :
                    ]
                )
            else:
                self._best_opp_acc_buying[oppo_id] = min(
                    self._best_opp_acc_prices[oppo_id][
                        -self._num_of_consider_day_price :
                    ]
                )

        self._best_acc_selling = max(
            self._best_acc_prices[-self._num_of_consider_day_price :]
        )
        self._best_acc_buying = min(
            self._best_acc_prices[-self._num_of_consider_day_price :]
        )

    def on_negotiation_success(self, contract, mechanism):
        """Record sales/supplies secured"""
        # super().on_negotiation_success(contract, mechanism)

        # update my current best price to use for limiting concession in other
        # negotiations
        self._secured += contract.agreement["quantity"]

        offer = [None for _ in range(3)]
        offer[QUANTITY] = contract.agreement["quantity"]
        offer[UNIT_PRICE] = contract.agreement["unit_price"]
        offer[TIME] = contract.agreement["time"]

        up = contract.agreement["unit_price"]
        if contract.annotation["product"] == self.awi.my_output_product:
            partner = contract.annotation["buyer"]
            self._cur_best_price = max(self._cur_best_price, up)
            # self._best_acc_selling = max(self._best_acc_selling, self._cur_best_price)

        else:
            partner = contract.annotation["seller"]
            self._cur_best_price = min(self._cur_best_price, up)
            # self._best_acc_buying = min(self._best_acc_buying, self._cur_best_price)

        self._best_opp_acc_prices[partner].append(up)

        self.cur_offer_list[partner] = offer

        for oppo_id in self.oppo_list:
            if self._cur_rank[oppo_id] > self._cur_rank[partner]:
                self._cur_rank[oppo_id] = max(0, self._cur_rank[oppo_id] - 1)
            # if(oppo_id != partner):
            #     self._e[oppo_id] *= 1.5
            #     if(self.awi.is_first_level):
            #         self._e[oppo_id] = min(self._e[oppo_id], 1.5)
            #     else:
            #         self._e[oppo_id] = min(self._e[oppo_id], 1)

        if self.awi.is_first_level:
            self._quantity_slack = [
                (quantity_slack * 3 + 1) / 4 for quantity_slack in self._quantity_slack
            ]
        #     self._e[partner] = np.random.uniform(1, 1.5)
        # else:
        self._e[partner] = np.random.uniform(0.01, 0.5)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        # self._quantity_slack = (1 + 2 * self._quantity_slack)/3
        if self.awi.is_first_level:
            partner = annotation["buyer"]
            self._best_opp_acc_prices[partner].append(0)
        else:
            partner = annotation["seller"]
            self._best_opp_acc_prices[partner].append(float("inf"))

        for oppo_id in self.oppo_list:
            if self._cur_rank[oppo_id] > self._cur_rank[partner]:
                self._cur_rank[oppo_id] = max(0, self._cur_rank[oppo_id] - 1)
            # if(oppo_id != partner):
            #     self._e[oppo_id] *= 1.5
            #     if(self.awi.is_first_level):
            #         self._e[oppo_id] = min(self._e[oppo_id], 1.5)
            #     else:
            #         self._e[oppo_id] = min(self._e[oppo_id], 1)

        if self.awi.is_first_level:
            self._quantity_slack = [
                (quantity_slack * 3 + 1) / 4 for quantity_slack in self._quantity_slack
            ]
        self._e[partner] = np.random.uniform(0.01, 0.5)

    def propose(self, negotiator_id: str, state):
        # find the absolute best offer for me. This will most likely has an
        # unrealistic price
        offer = self.best_offer(negotiator_id)

        # if there are no best offers, just return None to end the negotiation
        if not offer:
            return None

        nmi = self.get_nmi(negotiator_id)

        unit_price_issue = nmi.issues[UNIT_PRICE]
        quantity_issue = nmi.issues[QUANTITY]

        offer = list(offer)

        offer[QUANTITY] = min(
            offer[QUANTITY],
            max(
                int(
                    self._quantity_slack[self._cur_rank[negotiator_id]] * self._needed()
                ),
                1,
            ),
        )

        if len(self._prev_opp_quantity[negotiator_id]):
            # oppo_demand_upper_bound = np.median(self._prev_opp_quantity[negotiator_id][-3:])
            # oppo_demand_lower_bound = np.min(self._prev_opp_quantity[negotiator_id])
            # offer[QUANTITY] = min(max(oppo_demand_lower_bound, offer[QUANTITY]), oppo_demand_upper_bound)
            offer[QUANTITY] = max(
                1,
                min(
                    np.min(
                        self._prev_opp_quantity[negotiator_id][
                            -self._num_of_consider_day_quantity :
                        ]
                    ),
                    offer[QUANTITY],
                ),
            )

        u1 = self.ufun.from_offers(
            tuple(tuple(_) for _ in self.cur_offer_list.values()),
            tuple([self.awi.is_first_level] * len(self.cur_offer_list)),
        )

        u2 = self.ufun.from_offers(
            tuple(
                tuple(_) for _ in list(self.cur_offer_list.values()) + [tuple(offer)]
            ),
            tuple([self.awi.is_first_level] * (len(self.cur_offer_list) + 1)),
        )
        if u1 > u2:
            return None

        # over-write the unit price in the best offer with a good-enough price
        offer[UNIT_PRICE] = min(
            max(
                self._find_good_price(self.get_nmi(negotiator_id), state, offer),
                unit_price_issue.min_value,
            ),
            unit_price_issue.max_value,
        )

        # if(np.random.uniform() < 0.5 and len(self._prev_opp_price[negotiator_id]) >= 3):
        #     offer[UNIT_PRICE] = int(self._prev_opp_price[negotiator_id][-3]/(self._prev_opp_price[negotiator_id][-2] + 1e-6) * self._prev_self_price[-1])
        # self._prev_self_price.append(offer[UNIT_PRICE])
        # return tuple(offer)

        raw_price = offer[UNIT_PRICE]

        mn, mx = self._price_range(nmi)

        test_price_interval = (
            np.linspace(raw_price, nmi.issues[UNIT_PRICE].max_value, 50)
            if self.awi.is_first_level
            else np.linspace(raw_price, nmi.issues[UNIT_PRICE].min_value, 50)
        )

        candidate_price = None

        for p in test_price_interval:
            offer[QUANTITY] = min(
                offer[QUANTITY],
                max(
                    int(
                        self._quantity_slack[self._cur_rank[negotiator_id]]
                        * self._needed()
                    ),
                    1,
                ),
            )
            cur_offer = deepcopy(offer)
            cur_offer[UNIT_PRICE] = max(
                min(int(p), nmi.issues[UNIT_PRICE].max_value),
                nmi.issues[UNIT_PRICE].min_value,
            )

            u1 = self.ufun.from_offers(
                tuple(tuple(_) for _ in self.cur_offer_list.values()),
                tuple([self.awi.is_first_level] * len(self.cur_offer_list)),
            )

            u2 = self.ufun.from_offers(
                tuple(
                    tuple(_)
                    for _ in list(self.cur_offer_list.values()) + [tuple(cur_offer)]
                ),
                tuple([self.awi.is_first_level] * (len(self.cur_offer_list) + 1)),
            )

            if u1 < u2 and type(candidate_price) == type(None):
                candidate_price = cur_offer[UNIT_PRICE]

            if 1.1 * u1 < u2:
                self._prev_self_price.append(offer[UNIT_PRICE])
                return tuple(cur_offer)
        # return None

        # self._prev_self_price.append(offer[UNIT_PRICE])
        if type(candidate_price) == type(None):
            return None
        else:
            offer[QUANTITY] = min(
                offer[QUANTITY],
                max(
                    int(
                        self._quantity_slack[self._cur_rank[negotiator_id]]
                        * self._needed()
                    ),
                    1,
                ),
            )
            cur_offer = deepcopy(offer)
            cur_offer[UNIT_PRICE] = candidate_price
            return tuple(cur_offer)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        # find the quantity I still need and end negotiation if I need nothing more
        # my_needs = self._needed()
        # if my_needs < 0:
        #     return ResponseType.END_NEGOTIATION

        # reject any offers with quantities above my needs
        offer = list(offer)

        nmi = self.get_nmi(negotiator_id)

        up = offer[UNIT_PRICE]
        # qt = offer[QUANTITY]
        is_selling = self.awi.is_first_level
        if is_selling:
            self._best_selling = max(up, self._best_selling)
            partner = nmi.annotation["buyer"]
            self._best_opp_selling[partner] = max(up, self._best_opp_selling[partner])
        else:
            self._best_buying = min(up, self._best_buying)
            partner = nmi.annotation["seller"]
            self._best_opp_buying[partner] = min(up, self._best_opp_buying[partner])

        self._prev_opp_quantity[partner].append(offer[QUANTITY])
        self._prev_opp_price[partner].append(up)

        if (
            offer[QUANTITY] < nmi.issues[QUANTITY].min_value
            or offer[QUANTITY] > nmi.issues[QUANTITY].max_value
        ):
            return ResponseType.REJECT_OFFER

        if (
            offer[UNIT_PRICE] < nmi.issues[UNIT_PRICE].min_value
            or offer[UNIT_PRICE] > nmi.issues[UNIT_PRICE].max_value
        ):
            return ResponseType.REJECT_OFFER

        u1 = self.ufun.from_offers(
            tuple(tuple(_) for _ in self.cur_offer_list.values()),
            tuple([is_selling] * len(self.cur_offer_list)),
        )

        offers = [offer] + list(self.cur_offer_list.values())
        u2 = self.ufun.from_offers(
            tuple(tuple(_) for _ in offers),
            tuple([is_selling] * (len(self.cur_offer_list) + 1)),
        )

        if u1 > u2:
            return ResponseType.REJECT_OFFER

        # if(state.step >= nmi.n_steps - 2):
        #     return ResponseType.ACCEPT_OFFER

        if self._needed() <= 0:
            return ResponseType.ACCEPT_OFFER

        if self._is_good_price(nmi, state, offer):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def best_offer(self, negotiator_id):
        my_needs = int(self._needed())
        if my_needs <= 0 and not self.awi.is_first_level:
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

    # def best_offer(self, negotiator_id):
    #
    #     my_needs = int(self._needed())
    #     nmi = self.get_nmi(negotiator_id)
    #     if not nmi:
    #         return None
    #     quantity_issue = nmi.issues[QUANTITY]
    #     unit_price_issue = nmi.issues[UNIT_PRICE]
    #     offer = [-1] * 3
    #     offer[QUANTITY] = max(
    #         min(my_needs, quantity_issue.max_value), quantity_issue.min_value
    #     )
    #     offer[TIME] = self.awi.current_step
    #     if self._is_selling(nmi):
    #         offer[UNIT_PRICE] = unit_price_issue.max_value
    #     else:
    #         offer[UNIT_PRICE] = unit_price_issue.min_value
    #     return tuple(offer)

    def _needed(self):

        # summary = self.awi.exogenous_contract_summary
        # demand = min(summary[0][0], summary[-1][0]) / (self.awi.n_competitors + 1)
        demand = (
            self.awi.current_exogenous_input_quantity
            if self.awi.is_first_level
            else self.awi.current_exogenous_output_quantity
        )
        return demand - self._secured

    def _is_selling(self, nmi):
        if not nmi:
            return None
        return nmi.annotation["product"] == self.awi.my_output_product

    def _is_good_price(self, nmi, state, offer):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps, nmi)
        # a good price is one better than the threshold
        if self._is_selling(nmi):
            return (offer[UNIT_PRICE] - mn) >= th * (mx - mn)
        else:
            return (mx - offer[UNIT_PRICE]) >= th * (mx - mn)

    def _find_good_price(self, nmi, state, offer):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps, nmi)
        # offer a price that is around th of your best possible price
        if self._is_selling(nmi):
            return mn + th * (mx - mn)
        else:
            return mx - th * (mx - mn)

    def _price_range(self, nmi):
        """Limits the price by the best price received"""
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            # mx = max(
            #     mn * (1 + self._range_slack),
            #     min(
            #         [mx] +
            #         [
            #             p * (1 + slack)
            #             for p, slack in (
            #                 (self._best_selling, self._step_price_slack2),
            #                 (self._best_acc_selling, self._acc_price_slack2),
            #                 (self._best_opp_selling[partner], self._opp_price_slack2),
            #                 (self._best_opp_acc_selling[partner], self._opp_acc_price_slack2),
            #                 (self._cur_best_price, self._step_agg_price_slack2),
            #                 (self.awi.trading_prices[self.awi.my_output_product], self._trading_price_slack2),
            #             )
            #         ]
            #     ),
            # )
            mn = min(
                mx * (1 - self._range_slack),
                max(
                    [mn]
                    + [
                        p * (1 + slack)
                        for p, slack in (
                            (self._best_selling, self._step_price_slack),
                            (self._best_acc_selling, self._acc_price_slack),
                            (self._best_opp_selling[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_selling[partner],
                                self._opp_acc_price_slack,
                            ),
                            (self._cur_best_price, self._step_agg_price_slack),
                            (
                                self.awi.trading_prices[self.awi.my_output_product],
                                self._trading_price_slack,
                            ),
                        )
                    ]
                ),
            )

        else:
            partner = nmi.annotation["seller"]
            # mn = min(
            #     mx * (1 - self._range_slack),
            #     max(
            #         [mn] +
            #         [
            #             p * (1 - 1.2 * slack)
            #             for p, slack in (
            #                 (self._best_buying, self._step_price_slack2),
            #                 (self._best_acc_buying, self._acc_price_slack2),
            #                 (self._best_opp_buying[partner], self._opp_price_slack2),
            #                 (self._best_opp_acc_buying[partner], self._opp_acc_price_slack2),
            #                 (self._cur_best_price, self._step_agg_price_slack2),
            #                 (self.awi.trading_prices[self.awi.my_input_product]/2, self._trading_price_slack2),
            #             )
            #         ]
            #     ),
            # )
            mx = max(
                mn * (1 + self._range_slack),
                min(
                    [mx]
                    + [
                        p * (1 + slack)
                        for p, slack in (
                            (self._best_buying, self._step_price_slack2),
                            (self._best_acc_buying, self._acc_price_slack2),
                            (self._best_opp_buying[partner], self._opp_price_slack2),
                            (
                                self._best_opp_acc_buying[partner],
                                self._opp_acc_price_slack2,
                            ),
                            (self._cur_best_price, self._step_agg_price_slack2),
                            (
                                2 * self.awi.catalog_prices[self.awi.my_input_product],
                                self._trading_price_slack2,
                            ),
                        )
                    ]
                ),
            )
        return mn, mx

    def _th(self, step, n_steps, nmi):
        """calculates a descending threshold (0 <= th <= 1)"""
        partner = (
            nmi.annotation["buyer"]
            if self._is_selling(nmi)
            else nmi.annotation["seller"]
        )
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e[partner]

    # =====================
    # Negotiation Callbacks
    # =====================

    # =====================
    # Time-Driven Callbacks
    # =====================
    # def init(self):
    #     super().init()
    #     self._e = defaultdict(lambda: 0.2)
    #
    # def step(self):
    #     super().init()
    #     self._e = defaultdict(lambda: 0.2)


class StagHunterV5(OneShotAgent):
    def __init__(
        self,
        *args,
        acc_price_slack=0.25,  # accumlated agreed best price among all oppo
        acc_price_slack2=0.4,
        step_price_slack=0.2,  # current encounted best prices among all oppo
        step_price_slack2=0.4,
        opp_acc_price_slack=0.15,  # per-oppo accumlated agreed best price
        opp_acc_price_slack2=0.5,
        opp_price_slack=0.1,  # per-oppo best prices encounted
        opp_price_slack2=0.4,
        step_agg_price_slack=0.2,  # current encounted agreed best prices
        step_agg_price_slack2=0.3,
        quantity_slack=2 / 3,
        range_slack=0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # self._e = concession_exponent
        self._acc_price_slack = acc_price_slack
        self._acc_price_slack2 = acc_price_slack2
        self._step_price_slack = step_price_slack
        self._step_price_slack2 = step_price_slack2
        self._opp_acc_price_slack = opp_acc_price_slack
        self._opp_acc_price_slack2 = opp_acc_price_slack2
        self._opp_price_slack = opp_price_slack
        self._opp_price_slack2 = opp_price_slack2
        self._step_agg_price_slack = step_agg_price_slack
        self._step_agg_price_slack2 = step_agg_price_slack2
        self._range_slack = range_slack
        self._quantity_slack = defaultdict(lambda: quantity_slack)

    def init(self):
        """Initialize the quantities and best prices received so far"""

        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_acc_selling = defaultdict(lambda: 0.0)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))

        self._best_opp_acc_prices = (
            defaultdict(lambda: [0.0])
            if self.awi.is_first_level
            else defaultdict(lambda: [float("inf")])
        )
        self._best_acc_prices = [0.0] if self.awi.is_first_level else [float("inf")]

        self._prev_opp_quantity = defaultdict(lambda: [])
        self._prev_opp_price = defaultdict(lambda: [])
        self._prev_self_price = []
        self._secured = 0
        self.oppo_list = (
            self.awi.my_consumers if self.awi.is_first_level else self.awi.my_suppliers
        )
        self._e = defaultdict(lambda: 1)

        # self._e = defaultdict(lambda: 0.2 + random.random() * 0.8)

        self._acc_rev = {k: [] for k in self.oppo_list}

        # self._agreement_hist = {k: [] for k in self.oppo_list}
        # self._range_slack = random.random() * 0.2 + 0.05
        # self._step_price_slack = random.random() * 0.2 + 0.05
        # self._opp_price_slack = random.random() * 0.2 + 0.1

        self.cur_offer_list = {}
        # self._oppo_neg_history = {k: [] for k in self.oppo_list}
        # self._max_util_dist = {k: np.zeros(100) for k in self.oppo_list}

        self._lr = defaultdict(lambda: 0.2)
        self._is_prev_agg = defaultdict(lambda: False)
        self._prev_e = defaultdict(lambda: [1])
        self._prev_mag_profit = defaultdict(lambda: [0])

        self._rank = defaultdict(lambda: 1)

        # self._quantity_slack = [1/2 for _ in range(len(self.oppo_list))]
        # self._quantity_slack[0] = 1
        # self._quantity_slack[1] = 2/3
        self._quantity_slack = [2 / 3 for _ in range(len(self.oppo_list))]

    def before_step(self):
        self._best_selling, self._best_buying = 0.0, float("inf")

        self._best_opp_selling = defaultdict(lambda: 0.0)
        self._best_opp_buying = defaultdict(lambda: float("inf"))

        self._cur_best_price = 0 if self.awi.is_first_level else float("inf")

        self._cur_rank = deepcopy(self._rank)

        # self._quantity_slack = 1/2

        # self._stop_steps = {k: 0 for k in self.oppo_list}
        # for oppo_id in self.oppo_list:
        #     total_steps =  self.get_nmi(oppo_id).n_steps
        #     if(len(self._max_util_dist[oppo_id]) == 100):
        #         self._max_util_dist[oppo_id] = np.zeros(total_steps)
        #     #self._stop_steps[oppo_id] = np.random.choice(len(self._max_util_dist[oppo_id]), p=self._max_util_dist[oppo_id]/np.sum(self._max_util_dist[oppo_id]))
        #     #self._stop_steps[oppo_id] = np.argmax(self._max_util_dist[oppo_id]/np.sum(self._max_util_dist[oppo_id]))
        #
        #     self._stop_steps[oppo_id] = np.max(np.flatnonzero(self._max_util_dist[oppo_id] == self._max_util_dist[oppo_id].max()))
        # self._stop_steps[oppo_id] = np.argsort(self._max_util_dist[oppo_id])[-2:]

    def step(self, lr=0.1):
        """Initialize the quantities and best prices received for next step"""

        self._prev_opp_quantity = defaultdict(lambda: [])
        self._prev_opp_price = defaultdict(lambda: [])
        self._prev_self_price = []
        self._secured = 0

        u_total = self.ufun.from_offers(
            tuple(tuple(_) for _ in self.cur_offer_list.values()),
            tuple([self.awi.is_first_level] * (len(self.cur_offer_list))),
        )

        for oppo_id in self.oppo_list:
            if oppo_id in self.cur_offer_list:
                cur_offer = []
                for id_p in self.cur_offer_list:
                    if id_p != oppo_id:
                        cur_offer.append(tuple(self.cur_offer_list[id_p]))

                u_p = self.ufun.from_offers(
                    tuple(cur_offer),
                    tuple([self.awi.is_first_level] * (len(cur_offer))),
                )

                # if(u_total-u_p >= self._prev_best_mag_profit[oppo_id]):
                self._prev_mag_profit[oppo_id].append(u_total - u_p)
                self._prev_e[oppo_id].append(self._e[oppo_id])

                # old_inc = 0 if not len(self._acc_rev[oppo_id]) else np. median(self._acc_rev[oppo_id][-3:])
                #
                # inc = u_total-u_p
                # if not self._is_prev_agg[oppo_id]:
                #     self._lr[oppo_id] /= 2
                #     self._lr[oppo_id] = max(0.05, self._lr[oppo_id])

                self._e[oppo_id] = max(0.1, self._e[oppo_id] * (1 - self._lr[oppo_id]))
                #
                self._is_prev_agg[oppo_id] = True

            else:
                # self._acc_rev[oppo_id].append(0)
                old_e = self._e[oppo_id]

                if self._is_prev_agg[oppo_id]:
                    idx = np.argmax(self._prev_mag_profit[oppo_id][-3:]) - len(
                        self._prev_mag_profit[oppo_id][-3:]
                    )
                    self._e[oppo_id] = self._prev_e[oppo_id][idx]
                self._prev_e[oppo_id].append(old_e)
                self._prev_mag_profit[oppo_id].append(0)

                self._e[oppo_id] = min(self._e[oppo_id] * (1 + self._lr[oppo_id]), 10)
                self._is_prev_agg[oppo_id] = False

        if self.awi.current_step > 0:

            revs = (
                [
                    (oppo_id, np.mean(self._best_opp_acc_prices[oppo_id][-3:]))
                    for oppo_id in self.oppo_list
                ]
                if self.awi.is_first_level
                else [
                    (oppo_id, -np.mean(self._best_opp_acc_prices[oppo_id][-3:]))
                    for oppo_id in self.oppo_list
                ]
            )
            revs = sorted(revs, key=lambda x: x[1], reverse=True)

            self._rank = {revs[i][0]: i for i in range(len(revs))}

        # for oppo_id in self.oppo_list:
        #     other_offers = []
        #     for oppo_id_ in self.cur_offer_list:
        #         if(oppo_id != oppo_id_):
        #             other_offers.append(self.cur_offer_list[oppo_id_])
        #
        #     u_total = self.ufun.from_offers(other_offers, [self.awi.is_first_level] * (len(other_offers)))
        #
        #     marginal_utilities = []
        #
        #
        #     for (step, offer, _) in self._oppo_neg_history[oppo_id]:
        #         u_p = self.ufun.from_offers(other_offers + [offer], [self.awi.is_first_level] * (len(other_offers) + 1))
        #         if(u_p-u_total >=0):
        #             marginal_utilities.append((step, u_p-u_total))
        #
        #
        #
        #     #sort_hist = sorted(self._oppo_neg_history[oppo_id], key=lambda x: x[2], reverse=True)
        #     sort_hist = sorted(marginal_utilities, key=lambda x: (-x[1], -x[0]))
        #
        #     #self._max_util_dist[oppo_id] *= 0.9
        #
        #
        #     numerator = 0
        #     for i in range(len(sort_hist)):
        #         numerator += max(0, np.log(len(sort_hist)/2 + 1) - np.log(i + 1))
        #
        #     for i in range(len(sort_hist)):
        #         self._max_util_dist[oppo_id][sort_hist[i][0]] += (max(0, np.log(len(sort_hist)/2 + 1) - np.log(i + 1)))/numerator

        self.cur_offer_list = {}
        self._oppo_neg_history = {k: [] for k in self.oppo_list}
        self._best_acc_prices.append(self._cur_best_price)

        for oppo_id in self.oppo_list:

            if self.awi.is_first_level:
                self._best_opp_acc_selling[oppo_id] = max(
                    self._best_opp_acc_prices[oppo_id][-3:]
                )
            else:
                self._best_opp_acc_buying[oppo_id] = min(
                    self._best_opp_acc_prices[oppo_id][-3:]
                )

        self._best_acc_selling = max(self._best_acc_prices[-3:])
        self._best_acc_buying = min(self._best_acc_prices[-3:])

    def on_negotiation_success(self, contract, mechanism):
        """Record sales/supplies secured"""
        # super().on_negotiation_success(contract, mechanism)

        # update my current best price to use for limiting concession in other
        # negotiations
        self._secured += contract.agreement["quantity"]

        offer = [None for _ in range(3)]
        offer[QUANTITY] = contract.agreement["quantity"]
        offer[UNIT_PRICE] = contract.agreement["unit_price"]
        offer[TIME] = contract.agreement["time"]

        up = contract.agreement["unit_price"]

        if contract.annotation["product"] == self.awi.my_output_product:
            partner = contract.annotation["buyer"]
            self._cur_best_price = max(self._cur_best_price, up)
            # self._best_acc_selling = max(self._best_acc_selling, self._cur_best_price)

        else:
            partner = contract.annotation["seller"]
            self._cur_best_price = min(self._cur_best_price, up)
            # self._best_acc_buying = min(self._best_acc_buying, self._cur_best_price)

        self._best_opp_acc_prices[partner].append(up)
        self.cur_offer_list[partner] = offer

        for oppo_id in self.oppo_list:
            if self._cur_rank[oppo_id] > self._cur_rank[partner]:
                self._cur_rank[oppo_id] = max(0, self._cur_rank[oppo_id] - 1)

        # self._quantity_slack = (1 + 2 * self._quantity_slack)/3

        # self._agreement_hist[partner].append((offer, mechanism.state.step))

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        if self.awi.is_first_level:
            partner = annotation["buyer"]
            self._best_opp_acc_prices[partner].append(0)
        else:
            partner = annotation["seller"]
            self._best_opp_acc_prices[partner].append(float("inf"))

        # self._quantity_slack = (1 + 2 * self._quantity_slack)/3
        for oppo_id in self.oppo_list:
            if self._cur_rank[oppo_id] > self._cur_rank[partner]:
                self._cur_rank[oppo_id] = max(0, self._cur_rank[oppo_id] - 1)

    def _needed(self):

        # summary = self.awi.exogenous_contract_summary
        # demand = min(summary[0][0], summary[-1][0]) / (self.awi.n_competitors + 1)
        demand = (
            self.awi.current_exogenous_input_quantity
            if self.awi.current_exogenous_input_quantity
            else self.awi.current_exogenous_output_quantity
        )
        return demand - self._secured

    def _is_selling(self, nmi):
        if not nmi:
            return None
        return nmi.annotation["product"] == self.awi.my_output_product

    def _is_good_price(self, nmi, state, offer):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps, nmi)
        # a good price is one better than the threshold
        if self._is_selling(nmi):
            return (offer[UNIT_PRICE] - mn) >= th * (mx - mn)
        else:
            return (mx - offer[UNIT_PRICE]) >= th * (mx - mn)

    def _find_good_price(self, nmi, state, offer):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps, nmi)
        # offer a price that is around th of your best possible price
        if self._is_selling(nmi):
            return mn + th * (mx - mn)
        else:
            return mx - th * (mx - mn)

    def _price_range(self, nmi):
        """Limits the price by the best price received"""
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            # mx = max(
            #     mn * (1 + self._range_slack),
            #     min(
            #         [mx] +
            #         [
            #             p * (1 + slack)
            #             for p, slack in (
            #                 (self._best_selling, self._step_price_slack2),
            #                 (self._best_acc_selling, self._acc_price_slack2),
            #                 (self._best_opp_selling[partner], self._opp_price_slack2),
            #                 (self._best_opp_acc_selling[partner], self._opp_acc_price_slack2),
            #                 (self._cur_best_price, self._step_agg_price_slack2),
            #             )
            #         ]
            #     ),
            # )
            mn = min(
                mx * (1 - self._range_slack),
                max(
                    [mn]
                    + [
                        p * (1 - slack)
                        for p, slack in (
                            (self._best_selling, self._step_price_slack),
                            # (self._best_acc_selling, self._acc_price_slack),
                            (self._best_opp_selling[partner], self._opp_price_slack),
                            # (self._best_opp_acc_selling[partner], self._opp_acc_price_slack),
                            # (self._cur_best_price, self._step_agg_price_slack),
                        )
                    ]
                ),
            )

        else:
            partner = nmi.annotation["seller"]
            # mn = min(
            #     mx * (1 - self._range_slack),
            #     max(
            #         [mn] +
            #         [
            #             p * (1 - slack)
            #             for p, slack in (
            #                 (self._best_buying, self._step_price_slack2),
            #                 (self._best_acc_buying, self._acc_price_slack2),
            #                 (self._best_opp_buying[partner], self._opp_price_slack2),
            #                 (self._best_opp_acc_buying[partner], self._opp_acc_price_slack2),
            #                 (self._cur_best_price, self._step_agg_price_slack2),
            #             )
            #         ]
            #     ),
            # )
            mx = max(
                mn * (1 + self._range_slack),
                min(
                    [mx]
                    + [
                        p * (1 + slack)
                        for p, slack in (
                            (self._best_buying, self._step_price_slack),
                            # (self._best_acc_buying, self._acc_price_slack),
                            (self._best_opp_buying[partner], self._opp_price_slack),
                            # (self._best_opp_acc_buying[partner],self._opp_acc_price_slack),
                            # (self._cur_best_price, self._step_agg_price_slack),
                        )
                    ]
                ),
            )
        return mn, mx

    def _th(self, step, n_steps, nmi):
        """calculates a descending threshold (0 <= th <= 1)"""
        partner = (
            nmi.annotation["buyer"]
            if self._is_selling(nmi)
            else nmi.annotation["seller"]
        )
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e[partner]

    def propose(self, negotiator_id: str, state):
        # find the absolute best offer for me. This will most likely has an
        # unrealistic price
        # if there are no best offers, just return None to end the negotiation
        # if not offer:
        #     return None

        nmi = self.get_nmi(negotiator_id)
        unit_price_issue = nmi.issues[UNIT_PRICE]
        quantity_issue = nmi.issues[QUANTITY]

        offer = [-1] * 3
        if self._is_selling(nmi):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        offer[TIME] = self.awi.current_step

        if self._needed() > 0:
            mx = min(
                max(
                    int(
                        self._quantity_slack[self._cur_rank[negotiator_id]]
                        * self._needed()
                    ),
                    quantity_issue.min_value,
                ),
                quantity_issue.max_value,
            )

            if len(self._prev_opp_quantity[negotiator_id]):
                mx2 = np.min(self._prev_opp_quantity[negotiator_id][-3:])
                # if(mx2 < mx):
                #     offer[QUANTITY] = random.randint(mx2, mx+1)
                # else:
                #     offer[QUANTITY] = mx
                offer[QUANTITY] = min(mx, mx2)
            else:
                offer[QUANTITY] = mx
        else:
            # return None
            offer[QUANTITY] = quantity_issue.min_value
            if len(self._prev_opp_quantity[negotiator_id]):
                offer[QUANTITY] = min(self._prev_opp_quantity[negotiator_id][-3:])
            #     mx2 = np.min(self._prev_opp_quantity[negotiator_id][-3:])
            #     offer[QUANTITY] = random.randint(mx2, mx+1)
            # else:
            #     offer[QUANTITY] = random.randint(1, mx+1)

        u1 = self.ufun.from_offers(
            tuple(tuple(_) for _ in self.cur_offer_list.values()),
            tuple([self.awi.is_first_level] * len(self.cur_offer_list)),
        )
        #
        u2 = self.ufun.from_offers(
            tuple(
                tuple(_) for _ in list(self.cur_offer_list.values()) + [tuple(offer)]
            ),
            tuple([self.awi.is_first_level] * (len(self.cur_offer_list) + 1)),
        )
        #
        #
        if u1 > u2:
            return None

        # over-write the unit price in the best offer with a good-enough price
        offer[UNIT_PRICE] = self._find_good_price(
            self.get_nmi(negotiator_id), state, offer
        )

        raw_price = offer[UNIT_PRICE]

        # mn, mx = self._price_range(nmi)

        test_price_interval = (
            np.linspace(raw_price, unit_price_issue.max_value, 25)
            if self.awi.is_first_level
            else np.linspace(raw_price, unit_price_issue.min_value, 25)
        )

        for p in test_price_interval:
            cur_offer = deepcopy(offer)
            cur_offer[UNIT_PRICE] = int(p)

            u2 = self.ufun.from_offers(
                tuple(
                    tuple(_) for _ in list(self.cur_offer_list.values()) + [cur_offer]
                ),
                tuple([self.awi.is_first_level] * (len(self.cur_offer_list) + 1)),
            )

            if u1 < u2:
                return tuple(cur_offer)

        return tuple(offer)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        offer = list(offer)

        nmi = self.get_nmi(negotiator_id)

        up = offer[UNIT_PRICE]
        # qt = offer[QUANTITY]
        is_selling = self.awi.is_first_level
        if is_selling:
            self._best_selling = max(up, self._best_selling)
            partner = nmi.annotation["buyer"]
            self._best_opp_selling[partner] = max(up, self._best_opp_selling[partner])
        else:
            self._best_buying = min(up, self._best_buying)
            partner = nmi.annotation["seller"]
            self._best_opp_buying[partner] = min(up, self._best_opp_buying[partner])
        self._prev_opp_quantity[partner].append(offer[QUANTITY])
        self._prev_opp_price[partner].append(up)

        u1 = self.ufun.from_offers(
            tuple(tuple(_) for _ in self.cur_offer_list.values()),
            tuple([is_selling] * len(self.cur_offer_list)),
        )

        u2 = self.ufun.from_offers(
            tuple(tuple(_) for _ in [offer] + list(self.cur_offer_list.values())),
            tuple([is_selling] * (len(self.cur_offer_list) + 1)),
        )

        # self._oppo_neg_history[negotiator_id].append((state.step, offer, u2-u1,))

        # if u1 > u2 or (self._needed() > 0 and offer[QUANTITY] > self._quantity_slack * self._needed()):
        if u1 > u2:
            return ResponseType.REJECT_OFFER

        if state.step >= nmi.n_steps - 2:
            return ResponseType.ACCEPT_OFFER

        # if self.awi.current_step < 5:
        #     if self._is_good_price(nmi, state, offer):
        #         return ResponseType.ACCEPT_OFFER
        #
        #     return ResponseType.REJECT_OFFER

        # if(state.step < 3 or self._is_good_price(nmi, state, offer)):
        #     return ResponseType.ACCEPT_OFFER
        # if(len(self._prev_opp_quantity[negotiator_id]) >= 3):
        #     if(np.max(self._prev_opp_quantity[negotiator_id][-3:]) -  np.min(self._prev_opp_quantity[negotiator_id][-3:]) == 0):
        #         return ResponseType.REJECT_OFFER

        if self._needed() <= 0:
            return ResponseType.ACCEPT_OFFER

        # if(offer[QUANTITY] < self._quantity_slack[self._cur_rank[negotiator_id]] * self._needed()):
        #     return ResponseType.ACCEPT_OFFER

        if self._is_good_price(nmi, state, offer):
            return ResponseType.ACCEPT_OFFER

        # if(state.step >= self._stop_steps[negotiator_id] or self._is_good_price(nmi, state, offer)):
        #     return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


class StagHunterV6(StagHunterV5):
    def _is_good_price(self, nmi, state, offer):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps, nmi)
        # a good price is one better than the threshold
        is_selling = self.awi.is_first_level

        u2 = self.ufun.from_offers(
            tuple(tuple(_) for _ in [offer] + list(self.cur_offer_list.values())),
            tuple([is_selling] * (len(self.cur_offer_list) + 1)),
        )

        mn_offer = deepcopy(offer)
        mn_offer[UNIT_PRICE] = mn

        u_mn = self.ufun.from_offers(
            tuple(tuple(_) for _ in [mn_offer] + list(self.cur_offer_list.values())),
            tuple([is_selling] * (len(self.cur_offer_list) + 1)),
        )

        mx_offer = deepcopy(offer)
        mx_offer[UNIT_PRICE] = mx

        u_mx = self.ufun.from_offers(
            tuple(tuple(_) for _ in [mx_offer] + list(self.cur_offer_list.values())),
            tuple([is_selling] * (len(self.cur_offer_list) + 1)),
        )

        if self._is_selling(nmi):
            return (u2 - u_mn) >= th * (u_mx - u_mn)
        else:
            return (u2 - u_mx) >= th * (u_mn - u_mx)

    def _find_good_price(self, nmi, state, offer):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps, nmi)
        # offer a price that is around th of your best possible price
        is_selling = self.awi.is_first_level

        mn_offer = deepcopy(offer)
        mn_offer[UNIT_PRICE] = mn

        u_mn = self.ufun.from_offers(
            tuple(tuple(_) for _ in [mn_offer] + list(self.cur_offer_list.values())),
            tuple([is_selling] * (len(self.cur_offer_list) + 1)),
        )

        mx_offer = deepcopy(offer)
        mx_offer[UNIT_PRICE] = mx

        u_mx = self.ufun.from_offers(
            tuple(tuple(_) for _ in [mx_offer] + list(self.cur_offer_list.values())),
            tuple([is_selling] * (len(self.cur_offer_list) + 1)),
        )

        test_price_interval = (
            np.linspace(mn, mx, 50)
            if self.awi.is_first_level
            else np.linspace(mx, mn, 50)
        )

        for p in test_price_interval:

            offer_p = deepcopy(offer)
            offer_p[UNIT_PRICE] = p

            u2 = self.ufun.from_offers(
                tuple(tuple(_) for _ in [offer_p] + list(self.cur_offer_list.values())),
                tuple([is_selling] * (len(self.cur_offer_list) + 1)),
            )

            if self._is_selling(nmi):
                if (u2 - u_mn) >= th * (u_mx - u_mn):
                    return p
            else:
                if (u2 - u_mx) >= th * (u_mn - u_mx):
                    return p
        return int(p)


class StagHunterV7(StagHunter):
    def propose(self, negotiator_id: str, state):
        # find the absolute best offer for me. This will most likely has an
        # unrealistic price
        # if there are no best offers, just return None to end the negotiation
        # if not offer:
        #     return None

        nmi = self.get_nmi(negotiator_id)
        unit_price_issue = nmi.issues[UNIT_PRICE]
        quantity_issue = nmi.issues[QUANTITY]

        offer = [-1] * 3
        if self._is_selling(nmi):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        offer[TIME] = self.awi.current_step

        if self._needed() > 0:
            # mx = min(max(int(self._quantity_slack[self._cur_rank[negotiator_id]] * self._needed()), quantity_issue.min_value), quantity_issue.max_value)
            mx = min(
                max(int(self._needed()), quantity_issue.min_value),
                quantity_issue.max_value,
            )

            if len(self._prev_opp_quantity[negotiator_id]):
                mx2 = np.min(self._prev_opp_quantity[negotiator_id][-3:])
                # if(mx2 < mx):
                #     offer[QUANTITY] = random.randint(mx2, mx+1)
                # else:
                #     offer[QUANTITY] = mx
                offer[QUANTITY] = min(mx, mx2)
            else:
                offer[QUANTITY] = mx
        else:
            # return None
            offer[QUANTITY] = quantity_issue.min_value
            if len(self._prev_opp_quantity[negotiator_id]):
                offer[QUANTITY] = min(self._prev_opp_quantity[negotiator_id][-3:])

        u1 = self.ufun.from_offers(
            tuple(tuple(_) for _ in self.cur_offer_list.values()),
            tuple([self.awi.is_first_level] * len(self.cur_offer_list)),
        )

        u2 = self.ufun.from_offers(
            tuple(
                tuple(_) for _ in list(self.cur_offer_list.values()) + [tuple(offer)]
            ),
            tuple([self.awi.is_first_level] * (len(self.cur_offer_list) + 1)),
        )

        # if(u1 > u2):
        #     return None

        #
        # u2 = self.ufun.from_offers(list(self.cur_offer_list.values()) + [offer], [self.awi.is_first_level] * (len(self.cur_offer_list) + 1))
        #
        #
        # if(u1 > u2):
        #     return offer

        # over-write the unit price in the best offer with a good-enough price
        offer[UNIT_PRICE] = int(
            self._find_good_price(self.get_nmi(negotiator_id), state, offer)
        )

        if len(self._prev_opp_price[negotiator_id]) <= 3:
            self._prev_self_price.append(offer[UNIT_PRICE])
            return offer

        offer[UNIT_PRICE] = int(
            self._prev_opp_price[negotiator_id][-3]
            / max(1, self._prev_opp_price[negotiator_id][-2])
            * self._prev_self_price[-1]
        )

        self._prev_self_price.append(offer[UNIT_PRICE])
        return offer


def run(
    competition="oneshot",
    reveal_names=True,
    n_steps=50,
    n_configs=5,
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
    competitors = [
        StagHunterV5,
        StagHunterV6,
        StagHunter,
        GreedyOneShotAgent,
        GreedySyncAgent,
        StagHunterV7,
        RandomOneShotAgent,
    ]
    runner = anac2021_oneshot
    results = runner(
        competitors=competitors,
        verbose=True,
        n_steps=n_steps,
        n_configs=n_configs,
        compact=True,
        log_screen_level=logging.DEBUG,
    )
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(
        "."
    ).str[-1]
    # display results
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    # print(f"Finished in {humanize_time(time.perf_counter() - start)}")
    # StagHunters = [StagHunter0, StagHunter1, StagHunter2, StagHunter3, StagHunter4, StagHunter5, StagHunter6, StagHunter7, StagHunter8, StagHunter9, StagHunter10]
    #
    #
    # BR_str = []
    # for principle_SH in StagHunters:
    #     br_scores = -99999
    #     br_sh = None
    #     for dev_SH in StagHunters:
    #         competitors = [dev_SH]
    #         for _ in range(7):
    #             competitors.append(principle_SH)
    #         results = anac2021_oneshot(
    #             competitors=competitors,
    #             verbose=True,
    #             n_steps=n_steps,
    #             n_configs=n_configs,
    #             compact=True
    #         )
    #         results.total_scores.agent_type = results.total_scores.agent_type.str.split(
    #             "."
    #         ).str[-1]
    #         for i in range(len(results.total_scores.agent_type)):
    #             if(results.total_scores.agent_type[i] == dev_SH.__name__):
    #                 if(br_scores < results.total_scores.score[i]):
    #                     br_scores = results.total_scores.score[i]
    #                     br_sh = dev_SH.__name__
    #
    #
    #     BR_str.append(br_sh)
    # print(BR_str)

    # competitors = [StagHunter, StagHunterSync, RandomOneShotAgent, SyncRandomOneShotAgent, GreedyOneShotAgent, GreedySyncAgent, GreedySingleAgreementAgent]

    # just make names shorter
    # print(type(results))

    # display results
    # for i in range(len(results.total_scores.agent_type)):
    #     print(type(results.total_scores.agent_type[i]))
    # print(results.total_scores.agent_type)
    # print(results.total_scores.score)
    # print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    # print(f"Finished in {humanize_time(time.perf_counter() - start)}")


# if __name__ == "__main__":
#     import sys
#
#     run(sys.argv[1] if len(sys.argv) > 1 else "oneshot")
if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "oneshot")
