import logging
import pickle

# required for running tournaments and printing
import time
from functools import partial

# required for typing
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from negmas import (
    AgentMechanismInterface,
    Breach,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    ResponseType,
)
from negmas.helpers import humanize_time
from negmas.outcomes import Issue
from negmas.preferences import UtilityFunction
from negmas.sao import AspirationNegotiator
from scml.oneshot import OneShotAgent
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


class GreedyTestAgent(GreedyOneShotAgent):
    def init(self):
        super().init()
        self.oppo_list = (
            self.awi.my_consumers if self.awi.is_first_level else self.awi.my_suppliers
        )
        self.cur_offer_list = {}
        self._oppo_neg_history = {k: [] for k in self.oppo_list}
        self._max_util_dist = {k: np.zeros(100) for k in self.oppo_list}

    def step(self):
        super().step()

        # u_total = self.ufun.from_offers(list(self.cur_offer_list.values()), [self.awi.is_first_level] * (len(self.cur_offer_list)))

        for oppo_id in self.oppo_list:
            other_offers = []
            for oppo_id_ in self.cur_offer_list:
                if oppo_id != oppo_id_:
                    other_offers.append(self.cur_offer_list[oppo_id_])

            u_total = self.ufun.from_offers(
                tuple(other_offers),
                tuple([self.awi.is_first_level] * (len(other_offers))),
            )

            marginal_utilities = []

            for step, offer, _ in self._oppo_neg_history[oppo_id]:
                u_p = self.ufun.from_offers(
                    tuple(other_offers + [offer]),
                    tuple([self.awi.is_first_level] * (len(other_offers) + 1)),
                )
                marginal_utilities.append((step, u_p - u_total))

            # sort_hist = sorted(self._oppo_neg_history[oppo_id], key=lambda x: x[2], reverse=True)
            sort_hist = sorted(marginal_utilities, key=lambda x: x[1], reverse=True)

            if len(sort_hist):
                self._max_util_dist[oppo_id][sort_hist[0][0]] += 1.0

            if len(sort_hist) > 1:
                self._max_util_dist[oppo_id][sort_hist[1][0]] += 0.5

        self.cur_offer_list = {}
        self._oppo_neg_history = {k: [] for k in self.oppo_list}

        if self.awi.current_step == self.awi.n_steps - 1:
            for oppo_id in self.oppo_list:
                self._max_util_dist[oppo_id] = self._max_util_dist[oppo_id] / np.sum(
                    self._max_util_dist[oppo_id]
                )

            with open(f"./dis{time.time()}.pkl", "wb") as f:
                pickle.dump(self._max_util_dist, f)

    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)
        offer = [None for _ in range(3)]
        offer[QUANTITY] = contract.agreement["quantity"]
        offer[UNIT_PRICE] = contract.agreement["unit_price"]
        offer[TIME] = contract.agreement["time"]

        up = contract.agreement["unit_price"]
        if contract.annotation["product"] == self.awi.my_output_product:
            partner = contract.annotation["buyer"]
        else:
            partner = contract.annotation["seller"]

        self.cur_offer_list[partner] = offer

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        is_selling = self.awi.is_first_level
        u1 = self.ufun.from_offers(
            tuple(self.cur_offer_list.values()),
            tuple([is_selling] * len(self.cur_offer_list)),
        )

        offers = [offer] + list(self.cur_offer_list.values())
        u2 = self.ufun.from_offers(
            tuple(offers), tuple([is_selling] * (len(self.cur_offer_list) + 1))
        )

        self._oppo_neg_history[negotiator_id].append(
            (
                state.step,
                offer,
                u2 - u1,
            )
        )
        return super().respond(negotiator_id, state)


def run(
    competition="oneshot",
    reveal_names=True,
    n_steps=50,
    n_configs=5,
):
    competitors = [
        GreedyTestAgent,
        GreedyOneShotAgent,
        GreedyOneShotAgent,
        GreedySyncAgent,
        GreedySingleAgreementAgent,
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


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "oneshot")
