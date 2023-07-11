"""
**Submitted to ANAC 2021 SCML**
*Authors* type-your-team-member-names-with-their-emails here
This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2021 SCML.
This module implements a factory manager for the SCM 2021 league of ANAC 2021
competition (one-shot track). This version will not use subcomponents.
Please refer to the
[game description](http://www.yasserm.com/scml/scml2021oneshot.pdf) for all the
callbacks and subcomponents available.
Your agent can learn about the state of the world and itself by accessing
properties in the AWI it has. For example:
- The number of simulation steps (days): self.awi.n_steps
- The current step (day): self.awi.current_steps
- The factory state: self.awi.state
You can access the full list of these capabilities on the documentation.
"""

# required for running the test tournament
import collections
import logging
import time

# required for typing
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from negmas import AgentMechanismInterface, MechanismState, ResponseType
from negmas.helpers import humanize_time
from scml import QUANTITY, TIME, UNIT_PRICE, RandomOneShotAgent
from scml.oneshot import OneShotAgent
from scml.utils import anac2021_collusion, anac2021_oneshot, anac2021_std
from sklearn.linear_model import LinearRegression
from tabulate import tabulate

from .worker_agents import AdaptiveAgent, BetterAgent, LearningAgent

__all__ = [
    "Zilberan",
]


class Zilberan(OneShotAgent):
    def __init__(
        self,
        *args,
        opp_acc_price_slack=0.2,
        range_slack=0.03,
        concession_exponent=0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._opp_acc_price_slack = opp_acc_price_slack
        self._range_slack = range_slack
        self._e = concession_exponent

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

    def propose(self, negotiator_id: str, state) -> "Outcome":
        offer = self.best_offer(negotiator_id)
        if not offer:
            return None
        offer = list(offer)
        offer[UNIT_PRICE] = self._find_good_price(self.get_nmi(negotiator_id), state)
        return tuple(offer)

    def _is_good_price(self, nmi, state, price):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps)
        # a good price is one better than the threshold
        # We compromise more with time
        if self._is_selling(nmi):
            # We sell
            # Checking the distance from min
            if self.awi.current_step > 5:
                predicted_trading_price = self.model.predict(
                    np.array(self.awi.current_step).reshape(-1, 1)
                )
                good_price = min(mx, predicted_trading_price + mx / 5)
                # self.awi.logerror(f"cost: {self.awi.profile.cost} good price: {good_price}, predicted price: {predicted_trading_price}, prev: {self.awi.trading_prices[self.output_product]}")
                return (price - mn) >= th * (mx - (mx - good_price) / mx - mn)
            else:
                return (price - mn) >= th * (mx - mn)
        else:
            # We buy
            # Checking the distance from max
            if self.awi.current_step > 5:
                predicted_trading_price = self.model.predict(
                    np.array(self.awi.current_step).reshape(-1, 1)
                )
                # self.awi.logerror(f"predicted price: {predicted_trading_price}, prev: {self.awi.trading_prices[self.output_product]}")
                good_price = max(
                    mn, (predicted_trading_price - self.awi.profile.cost) + mn / 5
                )
                return (mx - price) >= th * (mx - (mx - good_price) / mx - mn)
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

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e

    def before_step(self):
        self.secured = 0
        self._best_selling, self._best_buying = 0.0, float("inf")

    def init(self):
        """Initialize the quantities and best prices received so far"""
        super().init()
        self.number_of_rounds = 0
        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._best_opp_acc_selling = defaultdict(float)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))
        self.partners_last_cash = defaultdict(float)
        self.we_terminated = False
        self.number_of_rounds = defaultdict(int)
        self.partners_respond_history = defaultdict(int)

        self.decrease_factor = 0.8
        self.increase_factor = 1.2

        self.output_product = self.awi.my_output_product
        self.output_product_trading_prices = []
        self.model = None

    def step(self):
        """Initialize the quantities and best prices received for next step"""
        super().step()
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self.we_terminated = False
        self.output_product_trading_prices.append(
            (self.awi.current_step, self.awi.trading_prices[self.output_product])
        )
        # self.awi.logerror(f"X,Y: {self.output_product_trading_prices}")
        if self.awi.current_step > 2:
            x = np.array([x for (x, y) in self.output_product_trading_prices]).reshape(
                (-1, 1)
            )
            y = np.array([y for (x, y) in self.output_product_trading_prices])
            # self.awi.logerror(f"X:{x}, Y:{y}")
            self.model = LinearRegression().fit(x, y)

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        # TODO: Perform a discount for getting next contracts.
        self._range_slack *= 1.1
        self._opp_acc_price_slack *= 1.1

        partner = [_ for _ in partners if _ != self.id][0]
        self.number_of_rounds[partner] += 1
        if not self.we_terminated:
            self.partners_respond_history[partner] -= 1
            # Partner rejected offer.
        self.we_terminated = False
        self.adjust_slack(partner)

    def on_negotiation_success(self, contract, mechanism):
        """Record sales/supplies secured"""
        partner = [_ for _ in contract.partners if _ != self.id][0]
        self.number_of_rounds[partner] += 1
        self.partners_respond_history[partner] += 1

        self.secured += contract.agreement["quantity"]
        self._opp_acc_price_slack *= 0.9

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

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        # find the quantity I still need and end negotiation if I need nothing more
        nmi = self.get_nmi(negotiator_id)
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            response = ResponseType.END_NEGOTIATION

        else:
            response = (
                ResponseType.ACCEPT_OFFER
                if offer[QUANTITY] <= my_needs
                else ResponseType.REJECT_OFFER
            )

        if response == ResponseType.ACCEPT_OFFER:
            response = (
                response
                if self._is_good_price(nmi, state, offer[UNIT_PRICE])
                else ResponseType.REJECT_OFFER
            )
            if self._is_selling(nmi):
                # The best price that we selled the product.
                self._best_selling = max(offer[UNIT_PRICE], self._best_selling)
            else:
                # The best price that we bought the product.
                self._best_buying = min(offer[UNIT_PRICE], self._best_buying)

        # update my current best price to use for limiting concession in other
        # negotiations
        up = offer[UNIT_PRICE]

        # TODO: Perform a de-discount for getting better contracts.
        # TODO: If the opponent balance is negative - decrease the slack.

        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            self._best_opp_selling[partner] = max(up, self._best_selling)

            if self.awi.profile.cost * 1.15 > up:
                self.we_terminated = True
                return ResponseType.REJECT_OFFER

            self.adjust_slack(partner)

        else:
            partner = nmi.annotation["seller"]

            self._best_opp_buying[partner] = min(up, self._best_buying)
            self.adjust_slack(partner)

        if response != ResponseType.ACCEPT_OFFER:
            self.we_terminated = True

        return response

    def adjust_slack(self, partner):
        if self._awi.reports_of_agent(partner) is not None:
            current_cash = self._awi.reports_of_agent(partner)[0].cash

            # if current_cash - self.partners_last_cash[partner] > 0.1 * current_cash:
            #     self.decrease_factor *= 0.8

            if current_cash != self.partners_last_cash[partner]:
                self.partners_last_cash[partner] = current_cash

        # If the partner is accepting more than he reject, be more strict to earn more.
        if self.partners_respond_history[partner] > 0:
            if self.partners_respond_history[partner] != 0:
                self.decrease_factor *= (
                    1
                    - (
                        self.number_of_rounds[partner]
                        - self.partners_respond_history[partner]
                    )
                    / self.number_of_rounds[partner]
                )
            self._opp_acc_price_slack *= self.decrease_factor
            # self.increase_factor = 1.2

        else:
            if self.partners_respond_history[partner] != 0:
                #  self.partners_respond_history[partner] is negative
                self.increase_factor *= (
                    1
                    + -1
                    * self.partners_respond_history[partner]
                    / self.number_of_rounds[partner]
                )
            self._opp_acc_price_slack *= self.increase_factor
            # self.decrease_factor = 0.8

    def _price_range(self, nmi):
        """Limits the price by the best price received"""
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            mn = max(
                [mn]
                + [
                    p * (1 - slack)
                    for p, slack in (  # ((1,2),(3,4),(5,6)) => (min, 12, 34, 56)
                        (
                            self._best_opp_acc_selling[partner],
                            self._opp_acc_price_slack,
                        ),
                    )
                ]
            )
        else:
            partner = nmi.annotation["seller"]
            mx = min(
                [mx]
                + [
                    p * (1 + slack)
                    for p, slack in (
                        (
                            self._best_opp_acc_buying[partner],
                            self._opp_acc_price_slack,
                        ),
                    )
                ]
            )

        return mn, mx


def run(
    competition="oneshot",
    reveal_names=True,
    n_steps=10,
    n_configs=1,  # =10  # =2
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
        competitors = [MyAgent, BetterAgent, AdaptiveAgent, LearningAgent]

    else:
        from scml.scml2020.agents import BuyCheapSellExpensiveAgent, DecentralizingAgent

        competitors = [
            MyAgent,
            DecentralizingAgent,
            BuyCheapSellExpensiveAgent,
        ]

    start = time.perf_counter()
    if competition == "std":
        runner = anac2021_std
    elif competition == "collusion":
        runner = anac2021_collusion
    else:
        runner = anac2021_oneshot
    results = runner(
        competitors=competitors,
        verbose=True,
        n_steps=n_steps,
        n_configs=n_configs,
        #   log_screen_level=logging.ERROR,
        #   log_to_screen=True,
    )

    # just make names shorter
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(
        "."
    ).str[-1]
    # display results
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    import sys

#    run(sys.argv[1] if len(sys.argv) > 1 else "oneshot")
