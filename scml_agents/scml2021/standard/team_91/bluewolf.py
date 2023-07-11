"""
**Submitted to ANAC 2021 SCML**
*Authors* type-your-team-member-names-with-their-emails here


This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2021 SCML.

This module implements a factory manager for the SCM 2021 league of ANAC 2021
competition. This version will use subcomponents. Please refer to the
[game description](http://www.yasserm.com/scml/scml2021.pdf) for all the
callbacks and subcomponents available.

Your agent can sense and act in the world by calling methods in the AWI it has.
For all properties/methods available only to SCM agents check:
  http://www.yasserm.com/scml/scml2020docs/api/scml.scml2020.AWI.html

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

import math
import time
from collections import defaultdict
from copy import deepcopy

# required for running the test tournament
from typing import Any, Dict, List, Optional

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
from negmas.sao import SAONMI, SAONegotiator, SAOState

# required for development
from scml.scml2020 import (
    DemandDrivenProductionStrategy,
    FixedTradePredictionStrategy,
    MeanERPStrategy,
    MovingRangeNegotiationManager,
    PredictionBasedTradingStrategy,
    SCML2020Agent,
    SupplyDrivenProductionStrategy,
    TradeDrivenProductionStrategy,
    TradingStrategy,
)
from scml.scml2020.agents import BuyCheapSellExpensiveAgent, DecentralizingAgent
from scml.scml2020.common import ANY_LINE, QUANTITY, TIME, UNIT_PRICE, is_system_agent
from scml.utils import anac2021_collusion, anac2021_oneshot, anac2021_std
from tabulate import tabulate

__all__ = [
    "BlueWolf",
]


class ObedientNegotiator(SAONegotiator):
    """
    A negotiator that controls a single negotiation with a single partner.

    Args:

        selling: Whether this negotiator is engaged on selling or buying
        requested: Whether this negotiator is created to manage a negotiation
                   requested by its owner (as opposed to one that is created
                   to respond to one created by the partner).

    Remarks:

        - This negotiator does nothing. It just passes calls to its owner.

    """

    def __init__(self, *args, selling, requested, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_selling = selling
        self.is_requested = requested

    # =====================
    # Negotiation Callbacks
    # =====================

    def propose(self, state: MechanismState) -> Optional[Outcome]:
        """Simply calls the corresponding method on the owner"""
        return self.owner.propose(state, self.nmi, self.is_selling, self.is_requested)

    def respond(self, state: SAOState) -> ResponseType:
        """Simply calls the corresponding method on the owner"""
        return self.owner.respond(state, self.nmi, self.is_selling, self.is_requested)


class BlueWolf(
    SupplyDrivenProductionStrategy,
    PredictionBasedTradingStrategy,
    SCML2020Agent,
):
    """
    This is the only class you *need* to implement. You can create the agent
    by combining the following strategies:

    1. A trading strategy that decides the quantities to sell and buy
    2. A negotiation manager that decides which negotiations to engage in and
       uses some controller(s) to control the behavior of all negotiators
    3. A production strategy that decides what to produce
    """

    def init(self):
        super().init()

        self._prev_oppo_agg_prices = defaultdict(lambda: [])

        self._supplier_agg_price = defaultdict(lambda: [float("inf")])

        self._consumer_agg_price = defaultdict(lambda: [0])

        self._prev_oppo_encounter_prices = defaultdict(lambda: [])

        self._prev_oppo_encounter_quantities = defaultdict(lambda: [])

        self._range_slack = 0.1

        self._step_price_slack = 0.25

        self._opp_price_slack = 0.2

        self._acc_oppo_slack = 0.3

        self._acc_slack = 0.2

        self._e = defaultdict(lambda: 1)

    def before_step(self):
        super().before_step()
        self._prev_oppo_encounter_prices = defaultdict(lambda: [])
        self._prev_oppo_encounter_quantities = defaultdict(lambda: [])
        self._cur_best_selling, self._cur_best_buying = 0, float("inf")
        self._cur_oppo_best_selling, self._cur_oppo_best_buying = (
            defaultdict(lambda: 0),
            defaultdict(lambda: float("inf")),
        )

        self._cur_inputs_needed = deepcopy(self.inputs_needed)
        self._cur_outputs_needed = deepcopy(self.outputs_needed)

        # for oppo_id in self.awi.my_consumers:
        #     toss = np.random.uniform()
        #     if(toss < 0.5):
        #         self._e[oppo_id] = np.random.uniform(0.1, 1)
        #     else:
        #         self._e[oppo_id] = np.random.uniform(1, 10)
        # for oppo_id in self.awi.my_suppliers:
        #     toss = np.random.uniform()
        #     if(toss < 0.5):
        #         self._e[oppo_id] = np.random.uniform(0.1, 1)
        #     else:
        #         self._e[oppo_id] = np.random.uniform(1, 10)

    def step(self):
        super().step()
        if self.awi.current_step == self.awi.n_steps - 1:
            return
        if self.awi.is_first_level:
            needs = np.max(
                self.outputs_needed[
                    self.awi.current_step + 1 : self.awi.current_step + 4
                ]
                - self.outputs_secured[
                    self.awi.current_step + 1 : self.awi.current_step + 4
                ]
            )
            price = self.awi.trading_prices[self.awi.my_output_product]
            for oppo_id in self.awi.my_consumers:
                self.awi.request_negotiation(
                    True,
                    self.awi.my_output_product,
                    (1, needs),
                    (price / 2, price * 2),
                    time=(self.awi.current_step + 1, self.awi.current_step + 4),
                    negotiator=ObedientNegotiator(
                        selling=True,
                        requested=True,
                        name=f"{self.id}>{oppo_id}",
                    ),
                    partner=oppo_id,
                )
        if self.awi.is_last_level:
            needs = np.max(
                self.inputs_needed[
                    self.awi.current_step + 1 : self.awi.current_step + 4
                ]
                - self.inputs_secured[
                    self.awi.current_step + 1 : self.awi.current_step + 4
                ]
            )
            price = self.awi.trading_prices[self.awi.my_input_product]
            for oppo_id in self.awi.my_suppliers:
                self.awi.request_negotiation(
                    False,
                    self.awi.my_input_product,
                    (1, needs),
                    (price / 2, price * 2),
                    time=(self.awi.current_step + 1, self.awi.current_step + 4),
                    negotiator=ObedientNegotiator(
                        selling=False,
                        requested=True,
                        name=f"{self.id}>{oppo_id}",
                    ),
                    partner=oppo_id,
                )

    def respond_to_negotiation_request(self, initiator, issues, annotation, mechanism):
        if self.awi.is_middle_level:
            return None
        if (
            issues[TIME].max_value < self.awi.current_step + 1
            or issues[TIME].min_value > self.awi.current_step + 4
        ):
            return None
        else:
            return ObedientNegotiator(
                selling=annotation["seller"] == self.id,
                requested=False,
                name=f"{initiator}>{self.id}",
            )

    def propose(
        self, state: SAOState, nmi: SAONMI, is_selling: bool, is_requested: bool
    ):
        partner_id = (
            nmi.annotation["buyer"]
            if nmi.annotation["seller"] == self.id
            else nmi.annotation["seller"]
        )
        is_selling = nmi.annotation["seller"] == self.id
        offer = [-1] * 3
        offer[TIME] = min(
            np.random.randint(
                low=self.awi.current_step + 1, high=self.awi.current_step + 4
            ),
            self.awi.n_steps - 1,
        )
        offer[TIME] = min(
            max(offer[TIME], nmi.issues[TIME].min_value), nmi.issues[TIME].max_value
        )

        offer[UNIT_PRICE] = self._find_good_price(nmi, state, offer)
        if is_selling:
            offer[QUANTITY] = int(2 / 3 * self._cur_outputs_needed[offer[TIME]])
        else:
            offer[QUANTITY] = int(2 / 3 * self._cur_inputs_needed[offer[TIME]])

        if len(self._prev_oppo_encounter_quantities[partner_id]):
            offer[QUANTITY] = min(
                offer[QUANTITY],
                np.min(self._prev_oppo_encounter_quantities[partner_id][-3:]),
            )

        offer[QUANTITY] = max(
            min(offer[QUANTITY], nmi.issues[QUANTITY].max_value),
            nmi.issues[QUANTITY].min_value,
        )

        return tuple(offer)

    def respond(self, state, nmi, is_selling, is_requested):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        offer = list(offer)

        partner_id = (
            nmi.annotation["buyer"]
            if nmi.annotation["seller"] == self.id
            else nmi.annotation["seller"]
        )
        is_selling = nmi.annotation["seller"] == self.id
        self._prev_oppo_encounter_prices[partner_id].append(offer[UNIT_PRICE])
        self._prev_oppo_encounter_quantities[partner_id].append(offer[QUANTITY])

        if is_selling:
            self._cur_best_selling = max(self._cur_best_selling, offer[UNIT_PRICE])
            self._cur_oppo_best_selling[partner_id] = max(
                self._cur_oppo_best_selling[partner_id], offer[UNIT_PRICE]
            )
        else:
            self._cur_best_buying = min(self._cur_best_buying, offer[UNIT_PRICE])
            self._cur_oppo_best_buying[partner_id] = min(
                self._cur_oppo_best_buying[partner_id], offer[UNIT_PRICE]
            )

        if offer[TIME] > self.awi.current_step + 3:
            return ResponseType.REJECT_OFFER

        if self._is_good_price(nmi, state, offer):
            return ResponseType.ACCEPT_OFFER

        else:
            return ResponseType.REJECT_OFFER

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
            return math.floor(mn + th * (mx - mn))
        else:
            return math.ceil(mx - th * (mx - mn))

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
                            (self._cur_best_selling, self._step_price_slack),
                            # (self._best_acc_selling, self._acc_price_slack),
                            (
                                self._cur_oppo_best_selling[partner],
                                self._opp_price_slack,
                            ),
                            (
                                np.max(self._consumer_agg_price[partner][-3:]),
                                self._acc_oppo_slack,
                            ),
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
                            (self._cur_best_buying, self._step_price_slack),
                            # (self._best_acc_buying, self._acc_price_slack),
                            (
                                self._cur_oppo_best_buying[partner],
                                self._opp_price_slack,
                            ),
                            (
                                np.min(self._supplier_agg_price[partner][-3:]),
                                self._acc_oppo_slack,
                            ),
                            # (self._best_opp_acc_buying[partner],self._opp_acc_price_slack),
                            # (self._cur_best_price, self._step_agg_price_slack),
                        )
                    ]
                ),
            )

        return math.floor(mn), math.ceil(mx)

    def _th(self, step, n_steps, nmi):
        """calculates a descending threshold (0 <= th <= 1)"""
        partner = (
            nmi.annotation["buyer"]
            if self._is_selling(nmi)
            else nmi.annotation["seller"]
        )
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e[partner]

    def on_negotiation_success(self, contract, mechanism):
        """Record sales/supplies secured"""
        # super().on_negotiation_success(contract, mechanism)

        # update my current best price to use for limiting concession in other
        # negotiations
        # self._secured += contract.agreement["quantity"]

        offer = [None for _ in range(3)]
        offer[QUANTITY] = contract.agreement["quantity"]
        offer[UNIT_PRICE] = contract.agreement["unit_price"]
        offer[TIME] = contract.agreement["time"]

        up = contract.agreement["unit_price"]
        if contract.annotation["product"] == self.awi.my_output_product:
            partner = contract.annotation["buyer"]
            # self._cur_best_price = max(self._cur_best_price, up)
            # self._best_acc_selling = max(self._best_acc_selling, self._cur_best_price)
            self._cur_outputs_needed[offer[TIME]] -= offer[QUANTITY]
            self._consumer_agg_price[partner].append(up)

        else:
            partner = contract.annotation["seller"]
            # self._cur_best_price = min(self._cur_best_price, up)
            # self._best_acc_buying = min(self._best_acc_buying, self._cur_best_price)
            self._cur_inputs_needed[offer[TIME]] -= offer[QUANTITY]

            self._supplier_agg_price[partner].append(up)

        # self.cur_offer_list[partner] = offer

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        # self._quantity_slack = (1 + 2 * self._quantity_slack)/3
        if self.awi.is_first_level:
            partner = annotation["buyer"]
            self._consumer_agg_price[partner].append(0)
        else:
            partner = annotation["seller"]
            self._supplier_agg_price[partner].append(float("inf"))

        # for oppo_id in self.oppo_list:
        #     if(self._cur_rank[oppo_id] > self._cur_rank[partner]):
        #         self._cur_rank[oppo_id] = max(0, self._cur_rank[oppo_id]-1)
        #
        # self._quantity_slack = [(quantity_slack * 2 + 1)/3 for quantity_slack in self._quantity_slack]


def run(
    competition="std",
    reveal_names=True,
    n_steps=10,
    n_configs=2,
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competition: The competition type to run (possibilities are std,
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
        BlueWolf,
        DecentralizingAgent,
        BuyCheapSellExpensiveAgent,
    ]
    start = time.perf_counter()
    if competition == "std":
        results = anac2021_std(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
        )
    elif competition == "collusion":
        results = anac2021_collusion(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
        )
    elif competition == "oneshot":
        # Standard agents can run in the OneShot environment but cannot win
        # the OneShot track!!
        from scml.oneshot.agents import GreedyOneShotAgent, RandomOneShotAgent

        competitors = [
            BlueWolf,
            RandomOneShotAgent,
            GreedyOneShotAgent,
        ]
        results = anac2021_oneshot(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
        )
    else:
        raise ValueError(f"Unknown competition type {competition}")
    # just make agent types shorter in the results
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(
        "."
    ).str[-1]
    # show results
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    # will run a short tournament against two built-in agents. Default is "std"
    # You can change this from the command line by running something like:
    # >> python3 myagent.py collusion
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "std")
