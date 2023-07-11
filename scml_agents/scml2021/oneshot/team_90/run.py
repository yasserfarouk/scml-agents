"""
**Submitted to ANAC 2021 SCML**
David Krongauz - kingkrong@gmail.com Hadar Ben-Efraim - hadarib@gmail.com
CS Dept., Bar-Ilan Uni., Israel


This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2021 SCML.

"""

import itertools
import os
import pickle

# required for running the test tournament
import time

# required for typing
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from negmas import (
    AgentMechanismInterface,
    Breach,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    ResponseType,
    SAOResponse,
)
from negmas.helpers import humanize_time
from scml.oneshot.agents import (
    GreedyOneShotAgent,
    GreedySingleAgreementAgent,
    GreedySyncAgent,
    OneshotDoNothingAgent,
    RandomOneShotAgent,
    SingleAgreementAspirationAgent,
    SingleAgreementRandomAgent,
    SyncRandomOneShotAgent,
)
from scml.scml2020 import SCML2020Agent

# required for development
from scml.scml2020.agents import (
    BuyCheapSellExpensiveAgent,
    DecentralizingAgent,
    DoNothingAgent,
    MarketAwareBuyCheapSellExpensiveAgent,
)
from scml.scml2020.common import QUANTITY, UNIT_PRICE
from scml.utils import anac2020_collusion, anac2020_std, anac2021_oneshot
from scml.scml2020.world import Failure
from tabulate import tabulate

__all__ = [
    "PDPSyncAgent",
]


class PDPSyncAgent(GreedySyncAgent):
    """Predictive Dynmical programming agent basd on GreedySyncAgent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def counter_all(self, offers, states):
        """Respond to a set of offers given the negotiation state of each."""

        if self.ufun.max_utility < 0:
            return dict(zip(offers.keys(), itertools.repeat(None)))

        good_prices = {
            k: self._find_good_price(self.get_nmi(k), s) for k, s in states.items()
        }

        responses = {
            k: SAOResponse(ResponseType.REJECT_OFFER, None) for k in offers.keys()
        }
        my_input_needs, my_output_needs = self._needs()
        input_offers = {
            k: v for k, v in offers.items() if not self._is_selling(self.get_nmi(k))
        }
        output_offers = {
            k: v for k, v in offers.items() if self._is_selling(self.get_nmi(k))
        }

        n_input = len(input_offers)
        n_output = len(output_offers)
        t_input = [[-1 for i in range(my_input_needs + 1)] for j in range(n_input + 1)]
        t_output = [
            [-1 for i in range(my_output_needs + 1)] for j in range(n_output + 1)
        ]

        def knapsack_agent(t, offers, my_needs, n, isSelling):
            """Knapscak algotrithm func. with memoization"""

            sorted_offers = sorted(
                offers.values(),
                key=lambda x: -x[UNIT_PRICE] if isSelling else x[UNIT_PRICE],
            )
            # base conditions
            if n == 0 or my_needs == 0:
                return 0
            if t[n][my_needs] != -1:
                return t[n][my_needs]

            # choice diagram code
            if sorted_offers[n - 1][QUANTITY] <= my_needs:
                t[n][my_needs] = max(
                    sorted_offers[n - 1][QUANTITY]
                    + knapsack_agent(
                        t,
                        offers,
                        my_needs - sorted_offers[n - 1][QUANTITY],
                        n - 1,
                        isSelling,
                    ),
                    knapsack_agent(t, offers, my_needs, n - 1, isSelling),
                )
                return t[n][my_needs]
            elif sorted_offers[n - 1][QUANTITY] > my_needs:
                t[n][my_needs] = knapsack_agent(t, offers, my_needs, n - 1, isSelling)
                return t[n][my_needs]

        def calc_items_idxs(t, sorted_offers, my_needs, n):
            """Recoveres offers indexes from memmoization matrix"""

            # stores the result of Knapsack
            res = t[n][my_needs]
            # print("This is res: %d" % res)
            items_idxs = []

            w = my_needs
            for i in range(n, 0, -1):
                if res <= 0:
                    break
                # either the result comes from the
                # top (K[i-1][w]) or from (val[i-1]
                # + K[i-1] [w-wt[i-1]]) as in Knapsack
                # table. If it comes from the latter
                # one/ it means the item is included.
                if res == t[i - 1][w]:
                    continue
                else:
                    # This item is included.
                    # print("Item wight: %d" % sorted_offers[i - 1][QUANTITY])
                    items_idxs.append(i - 1)

                    # Since this weight is included
                    # its value is deducted
                    res = (
                        res
                        - sorted_offers[i - 1][UNIT_PRICE]
                        * sorted_offers[i - 1][QUANTITY]
                    )
                    w = w - sorted_offers[i - 1][QUANTITY]
            # print(items_idxs)
            return items_idxs

        def change_threshold(is_selling):
            if is_selling:
                loaded_model = pickle.load(
                    open(os.path.dirname(__file__) + "/output_model.sav", "rb")
                )
            else:
                loaded_model = pickle.load(
                    open(os.path.dirname(__file__) + "/input_model.sav", "rb")
                )

            try:
                data = [
                    {
                        "day": self.awi.current_step,
                        "price": self.awi.current_exogenous_output_price,
                        "prev_price": self.awi.prev_exogenous_output_price,
                    }
                ]
            except:
                self.awi.prev_exogenous_output_price = 0
                data = [
                    {
                        "day": self.awi.current_step,
                        "price": self.awi.current_exogenous_output_price,
                        "prev_price": self.awi.prev_exogenous_output_price,
                    }
                ]

            X_test = pd.DataFrame(data)
            result = loaded_model.predict(X_test)

            result = np.around(result)
            result = result.astype(int)

            if result[0][0] == 0:
                if is_selling:
                    self._threshold = 0.2
                else:
                    self._threshold = 0.4
            else:
                if is_selling:
                    self._threshold = 0.4
                else:
                    self._threshold = 0.2

        def transform_idxs2keys(idxs, sorted_offers, offers, is_selling):
            """Given the indexes formalizes the responses according to the SCML environment"""
            secured, outputs, chosen = 0, [], dict()
            change_threshold(is_selling)

            for i, k in enumerate(offers.keys()):
                if i not in idxs:
                    continue
                offer = sorted_offers[i]
                secured += offer[QUANTITY]
                chosen[k] = offer
                outputs.append(is_selling)
            if (
                self.ufun.from_offers(tuple(chosen.values()), tuple(outputs))
                >= self._th(self.awi.current_step, self.awi.n_steps)
                * self.ufun.max_utility
            ):
                for k in chosen.keys():
                    responses[k] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
            return secured

        input_sorted_offers = sorted(offers.values(), key=lambda x: x[UNIT_PRICE])
        output_sorted_offers = sorted(offers.values(), key=lambda x: -x[UNIT_PRICE])
        check_input = knapsack_agent(
            t_input, input_offers, my_input_needs, n_input, False
        )
        check_output = knapsack_agent(
            t_output, output_offers, my_output_needs, n_output, True
        )
        input_idxs = calc_items_idxs(
            t_input, input_sorted_offers, my_input_needs, n_input
        )
        output_idxs = calc_items_idxs(
            t_output, output_sorted_offers, my_output_needs, n_output
        )

        secured = transform_idxs2keys(
            input_idxs, input_sorted_offers, input_offers, False
        )
        secured += transform_idxs2keys(
            output_idxs, output_sorted_offers, output_offers, True
        )

        if (
            self.awi.prev_exogenous_output_price
            != self.awi.current_exogenous_output_price
        ):
            self.awi.prev_exogenous_output_price = (
                self.awi.current_exogenous_output_price
            )

        for k, v in responses.items():
            if v.response != ResponseType.REJECT_OFFER:
                continue
            responses[k] = SAOResponse(
                ResponseType.REJECT_OFFER,
                (
                    max(1, my_input_needs + my_output_needs - secured),
                    self.awi.current_step,
                    good_prices[k],
                ),
            )
        return responses


# def run(competition='oneshot',
#         reveal_names=True,
#         n_steps=20,
#         n_configs=2,
#         max_n_worlds_per_config=None,
#         n_runs_per_world=1
#         ):
#     """
#     **Not needed for submission.** You can use this function to test your agent.
#
#     Args:
#         competition: The competition type to run (possibilities are std,
#                      collusion).
#         n_steps:     The number of simulation steps.
#         n_configs:   Number of different world configurations to try.
#                      Different world configurations will correspond to
#                      different number of factories, profiles
#                      , production graphs etc
#         n_runs_per_world: How many times will each world simulation be run.
#
#     Returns:
#         None
#
#     Remarks:
#
#         - This function will take several minutes to run.
#         - To speed it up, use a smaller `n_step` value
#
#     """
#     competitors = [GreedySyncAgent, RandomOneShotAgent, DPSyncAgent, SyncRandomOneShotAgent, OneshotDoNothingAgent]
#     start = time.perf_counter()
#     if competition == 'std':
#         results = anac2020_std(
#             competitors=competitors, verbose=True, n_steps=n_steps,
#             n_configs=n_configs, n_runs_per_world=n_runs_per_world
#         )
#     elif competition == 'collusion':
#         results = anac2020_collusion(
#             competitors=competitors, verbose=True, n_steps=n_steps,
#             n_configs=n_configs, n_runs_per_world=n_runs_per_world
#         )
#     elif competition == 'oneshot':
#         results = anac2021_oneshot(
#             competitors=competitors, verbose=True, n_steps=n_steps,
#             n_configs=n_configs, n_runs_per_world=n_runs_per_world
#         )
#     else:
#         raise ValueError(f'Unknown competition type {competition}')
#     print(tabulate(results.total_scores, headers='keys', tablefmt='psql'))
#     print(f'Finished in {humanize_time(time.perf_counter() - start)}')


# if __name__ == '__main__':
#     run()
