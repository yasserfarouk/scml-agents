#!/usr/bin/env python
"""

Agents in one shot games do not consider about long-term production and planning,
but focus on the negotiation between agents.
The game simulates a supply chain which consists of two layers and two kinds of products.
Each agent in the game is assigned a factory then manage the buying and selling of products.
As the programmer of agents, we need to think about the strategies and code.
The agents make the most profits and take least penalties for shortfall and disposal will win the game.


"""

# required for running tournaments and printing
import math
import random

# required for running tournaments and printing
import time
from collections import defaultdict
from pprint import pprint

# required for typing
# required for typing
from typing import Any, Dict, List, Optional

# from Learningagent import LearningAgent
import numpy as np
from matplotlib import pyplot as plt
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

# required for development
# from pandas._libs.internals import defaultdict
from scml.oneshot import *
from scml.oneshot import OneShotAgent
from scml.oneshot.agents import RandomOneShotAgent, SyncRandomOneShotAgent
from scml.scml2020 import is_system_agent
from scml.utils import anac2021_collusion, anac2021_oneshot, anac2021_std
from tabulate import tabulate

__all__ = [
    "Agent97",
]


class Agent97(OneShotAgent):
    def init(self):
        """Called once after the agent-world interface is initialized
        initiate records when every world begins
        """
        self._sold = 0
        self._bought = 0
        self._selling_log = defaultdict(float)
        self._buying_log = defaultdict(float)
        self.secured = 0
        self.failure_list = defaultdict(int)
        self.succ_list = defaultdict(int)
        self.reputation = defaultdict(float)
        self.best_offered_sell = 0
        self.best_offered_buy = 0

    def step(self):
        self._sold = 0
        self._bought = 0
        self.secured = 0
        self.succ_list = defaultdict(int)
        self._selling_log = defaultdict(float)
        self._buying_log = defaultdict(float)
        self.failure_list = defaultdict(int)
        self.reputation = defaultdict(float)
        """initiate the secured every step"""

    # =====================
    # Negotiation Callbacks
    # =====================

    def _is_selling(self, nmi):
        return nmi.annotation["product"] == self.awi.my_output_product

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def propose(self, negotiator_id, state):
        """Called when the agent is asking to propose in one negotiation"""
        my_needs = self._needed(negotiator_id)
        nmi = self.get_nmi(negotiator_id)
        rep = 0.1
        if self.reputation:
            rep = self.sigmoid(self.reputation[min(self.reputation)])
        if my_needs <= 0:
            return None

        if not nmi:
            return None
        quantity_issue = nmi.issues[QUANTITY]
        # unit_price_issue = nmi.issues[UNIT_PRICE]
        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value), quantity_issue.min_value
        )
        offer[TIME] = self.awi.current_step
        if self._is_selling(nmi):
            # offer[UNIT_PRICE] = self._find_good_price(
            # self.get_nmi(negotiator_id), state
            # )
            maxx, minn = self._find_good_price(self.get_nmi(negotiator_id), state)
            """
            if maxx!=0 and minn!=0:
                offer[UNIT_PRICE]= (maxx+minn)/2
            else:
                offer[UNIT_PRICE]= maxx
            """
            if self.reputation[nmi.annotation["buyer"]]:
                # threshold=0.7*abs(self.sigmoid(self.reputation[nmi.annotation["buyer"]]))
                threshold = 0.7 * abs(
                    self.sigmoid(self.reputation[nmi.annotation["buyer"]])
                )
            else:
                threshold = 0.2
            if abs(threshold) < 0.2:
                if threshold < 0:
                    threshold = -0.2
                else:
                    threshold = 0.2
            th = (maxx - minn) * threshold
            offer[UNIT_PRICE] = maxx - th
            # offer[UNIT_PRICE] = maxx - th

            """if self.reputation[nmi.annotation["buyer"]]<-6:
                offer[UNIT_PRICE]=minn*(1-rep)
                self.reputation[nmi.annotation["buyer"]] += 4
            else:
                offer[UNIT_PRICE] = maxx - th"""
            # offer[UNIT_PRICE] = maxx*(1+rep)

        else:
            # offer[UNIT_PRICE] = self._find_good_price(
            # self.get_nmi(negotiator_id), state
            # )
            maxx, minn = self._find_good_price(self.get_nmi(negotiator_id), state)
            """
            if maxx!=0 and minn!=0:
                offer[UNIT_PRICE]=(maxx+minn)/2
            else:
                offer[UNIT_PRICE] = minn
            """
            if self.reputation[nmi.annotation["seller"]]:
                threshold = 0.7 * abs(
                    self.sigmoid(self.reputation[nmi.annotation["seller"]])
                )
            else:
                threshold = 0.2
            if threshold < 0.2:
                if threshold < 0:
                    threshold = -0.2
                else:
                    threshold = 0.2
            th = threshold * (maxx - minn)
            offer[UNIT_PRICE] = minn + th
            # offer[UNIT_PRICE] = th + minn
            """if self.reputation[nmi.annotation["seller"]]<-6:
                offer[UNIT_PRICE]=maxx*(1+rep)
                self.reputation[nmi.annotation["seller"]]+=4
            else:
                offer[UNIT_PRICE] = th + minn"""
            # offer[UNIT_PRICE]=minn*(1-rep)

        # print(nmi.issues)
        return tuple(offer)

    def _find_good_price(self, nmi, state):
        """find the best proposing prices by considering time,quatities and history prices"""
        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            if self._selling_log[partner] or self.best_offered_sell:
                minmin = [
                    self.best_offered_sell,
                    self._selling_log[partner],
                    nmi.issues[UNIT_PRICE].max_value,
                    self._sold,
                    self._selling_log[nmi.annotation["buyer"]],
                    self.best_offered_sell,
                ]
                minin = []
                for item in minmin:
                    if item != 0:
                        minin.append(item)
                # return nmi.issues[UNIT_PRICE].min_value
                # return self._selling_log[partner]*self.get_concession(nmi,state)
                # print("selling sigmoid:" + str(1 + self.sigmoid(self.reputation[nmi.annotation["buyer"]])))
                return (
                    max(
                        self.best_offered_sell,
                        self._selling_log[partner],
                        nmi.issues[UNIT_PRICE].max_value,
                        self._sold,
                        self._selling_log[nmi.annotation["buyer"]],
                    ),
                    min(minin),
                )
            else:
                return (
                    nmi.issues[UNIT_PRICE].max_value * self.get_concession(nmi, state),
                    nmi.issues[UNIT_PRICE].min_value * self.get_concession(nmi, state),
                )
                # return nmi.issues[UNIT_PRICE].max_value*(((nmi.n_steps-state.step+1)/nmi.n_steps)**0.14)

        else:
            partner = nmi.annotation["seller"]
            if self._buying_log[partner] or self.best_offered_buy or self._bought:
                minmin = [
                    self.best_offered_buy,
                    self._buying_log[partner],
                    nmi.issues[UNIT_PRICE].min_value,
                    self._bought,
                    self._buying_log[nmi.annotation["seller"]],
                ]
                minin = []
                for item in minmin:
                    if item != 0:
                        minin.append(item)
                # return self._buying_log[partner] *self.get_concession(nmi,state)
                # print("buying sigmoid:"+str(1 - self.sigmoid(self.reputation[nmi.annotation["seller"]])))

                return (
                    max(
                        self.best_offered_buy,
                        self._buying_log[partner],
                        nmi.issues[UNIT_PRICE].min_value,
                        self._bought,
                        self._buying_log[nmi.annotation["seller"]],
                        self.best_offered_buy,
                    ),
                    min(minin),
                )
            else:
                return (
                    nmi.issues[UNIT_PRICE].max_value * self.get_concession(nmi, state),
                    nmi.issues[UNIT_PRICE].min_value * self.get_concession(nmi, state),
                )
                # return nmi.issues[UNIT_PRICE].min_value*((nmi.n_steps/(nmi.n_steps-state.step+1))**0.14)

    def find_respond_price(self, negotiator_id, offer, nmi, state):
        """find the best reponding price by considering both quatities and history prices"""
        my_need = self._needed(negotiator_id)
        if my_need <= 0:
            return (
                nmi.issues[UNIT_PRICE].max_value * (nmi.n_steps / state.step)
                if self._is_selling(nmi)
                else nmi.issues[UNIT_PRICE].min_value * (state.step / nmi.n_steps)
            )
        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            if self._selling_log[partner]:
                # return nmi.issues[UNIT_PRICE].min_value
                # print("I found a optimal selling price")
                # return (self._selling_log[partner]*my_need/offer[QUANTITY] if offer[QUANTITY]>my_need
                # else self._selling_log[partner]*offer[QUANTITY]/my_need)
                return self._selling_log[partner]

            else:
                return nmi.issues[UNIT_PRICE].max_value

        else:
            partner = nmi.annotation["seller"]
            if self._buying_log[partner]:
                # print("I found a optimal buying price")

                # return (self._buying_log[partner] * my_need / offer[QUANTITY] if my_need>offer[QUANTITY]
                # else self._buying_log[partner] * offer[QUANTITY] / my_need)
                return self._buying_log[partner]
            else:
                return nmi.issues[UNIT_PRICE].min_value

    def get_image(self, negotiator_id, state, offer, nmi):
        maxx, minn = self._find_good_price(nmi, state)
        expected_quantity = self._needed()
        if self._is_selling(nmi):
            if maxx != 0 and minn != 0:
                expected_price = (maxx + minn) / 2
            else:
                expected_price = maxx
            if self.reputation[nmi.annotation["buyer"]]:
                # threshold=0.7*abs(self.sigmoid(self.reputation[nmi.annotation["buyer"]]))
                threshold = 0.7 * abs(
                    self.sigmoid(self.reputation[nmi.annotation["buyer"]])
                )
            else:
                threshold = 0.2
            if abs(threshold) < 0.2:
                if threshold < 0:
                    threshold = -0.2
                else:
                    threshold = 0.2
            th = (maxx - minn) * threshold
            expected_price = maxx - th
            image = (
                nmi.annotation["buyer"],
                offer[UNIT_PRICE] - expected_price,
                expected_quantity - offer[QUANTITY],
            )
            image = list(image)

        else:
            if maxx != 0 and minn != 0:
                expected_price = (maxx + minn) / 2
            else:
                expected_price = minn
            if self.reputation[nmi.annotation["seller"]]:
                threshold = 0.7 * abs(
                    self.sigmoid(self.reputation[nmi.annotation["seller"]])
                )
            else:
                threshold = 0.2
            if threshold < 0.2:
                if threshold < 0:
                    threshold = -0.2
                else:
                    threshold = 0.2
            th = threshold * (maxx - minn)
            expected_price = minn + th
            image = (
                nmi.annotation["seller"],
                expected_price - offer[UNIT_PRICE],
                expected_quantity - offer[QUANTITY],
            )
            image = list(image)
            # print(maxx, minn)

        # print(image)
        return image

    def sigmoid(self, x):
        return 1 / (1 + math.exp(0 - x)) - 0.5

    def get_concession(self, nmi, state):
        if self._is_selling(nmi):
            return ((nmi.n_steps - state.step + 1) / nmi.n_steps) ** 0.04
            # return  1
        else:
            return (nmi.n_steps / (nmi.n_steps - state.step + 1)) ** 0.04
            # return 1

    def get_reputation(self, image, state, nmi):
        if self.reputation[image[0]]:
            self.reputation[image[0]] += math.sin(math.pi * image[1] / 100) * (
                (nmi.n_steps - state.step + 1) / nmi.n_steps
            )
            # self.reputation[image[0]] +=  image[1] / 5
        else:
            self.reputation[image[0]] = 0
            self.reputation[image[0]] += math.sin(math.pi * image[1] / 100) * (
                (nmi.n_steps - state.step + 1) / nmi.n_steps
            )
            # self.reputation[image[0]] += image[1] / 5
        # print(image[1])
        # print(self.reputation)
        # print(image)
        # print(self.awi.profile)
        # print(self.profile.shortfall_penalty_mean)
        # print(self.failure_list)
        return self.reputation[image[0]]

    def respond(self, negotiator_id, state):
        """Called when the agent is asked to respond to an offer
        relate time with the prices
        """
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        nmi = self.get_nmi(negotiator_id)
        image = self.get_image(negotiator_id, state, offer, nmi)
        reputation = self.get_reputation(image, state, nmi)
        # print(self._selling_log)
        # print(self._buying_log)
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            if self._is_selling(nmi):
                if offer[UNIT_PRICE] >= nmi.issues[UNIT_PRICE].max_value * (
                    nmi.n_steps / state.step
                ):
                    return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER
            else:
                if offer[UNIT_PRICE] <= nmi.issues[UNIT_PRICE].min_value * (
                    state.step / nmi.n_steps
                ):
                    return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER

        if self._is_selling(nmi):
            self.best_offered_sell = max(self.best_offered_sell, offer[UNIT_PRICE])

            # if self._selling_log[nmi.annotation["buyer"]]:
            # self._selling_log[nmi.annotation["buyer"]]=min(self._selling_log[nmi.annotation["buyer"]],offer[UNIT_PRICE])
            # else:
            # self._selling_log[nmi.annotation["buyer"]] =offer[UNIT_PRICE]
            """if self._sold!=0:
                if offer[UNIT_PRICE]-self._sold >= min(self._selling_log[nmi.annotation["buyer"]]*self.get_concession(nmi,state),
                                            (1+self.sigmoid(self.reputation[nmi.annotation["buyer"]]))*self.best_offered_sell,
                                            nmi.issues[UNIT_PRICE].max_value*self.get_concession(nmi,state),
                                            self._sold,
                                            self._selling_log[nmi.annotation["buyer"]]* (1+self.sigmoid(self.reputation[nmi.annotation["buyer"]])),
                                            self.get_concession(nmi,state)*self.best_offered_sell)- \
                                            max(self._selling_log[nmi.annotation["buyer"]] * self.get_concession(nmi, state),
                                                (1 + self.sigmoid(self.reputation[nmi.annotation["buyer"]])) * self.best_offered_sell,
                                                nmi.issues[UNIT_PRICE].max_value * self.get_concession(nmi, state),
                                                self._sold,
                                                self._selling_log[nmi.annotation["buyer"]] * (
                                                            1 + self.sigmoid(self.reputation[nmi.annotation["buyer"]])),
                                                self.get_concession(nmi, state) * self.best_offered_sell):
                #print("I accept")
                    return ResponseType.ACCEPT_OFFER
                else:
                    #print("I reject")
                    return ResponseType.REJECT_OFFER
            else:"""
            minmin = [
                self.best_offered_sell,
                self._selling_log[nmi.annotation["buyer"]],
                nmi.issues[UNIT_PRICE].max_value,
                self._sold,
                self._selling_log[nmi.annotation["buyer"]],
                self.best_offered_sell,
            ]
            minin = []
            for item in minmin:
                if item != 0:
                    minin.append(item)
            maxx = max(
                self._selling_log[nmi.annotation["buyer"]],
                nmi.issues[UNIT_PRICE].max_value,
                self._sold,
                self._selling_log[nmi.annotation["buyer"]],
            )
            ave = (maxx + min(minin)) / 2
            if self.reputation[nmi.annotation["buyer"]]:
                threshold = 0.7 * abs(
                    self.sigmoid(self.reputation[nmi.annotation["buyer"]])
                )
            else:
                threshold = 0.2
            if threshold < 0.2:
                threshold = 0.2
            th = (maxx - min(minin)) * threshold
            if maxx - offer[UNIT_PRICE] <= th:
                return ResponseType.ACCEPT_OFFER
            else:
                # print("I reject")
                return ResponseType.REJECT_OFFER
        else:
            if self.best_offered_buy != 0:
                self.best_offered_buy = min(self.best_offered_buy, offer[UNIT_PRICE])
            else:
                self.best_offered_buy = offer[UNIT_PRICE]

            # if self._selling_log[nmi.annotation["seller"]]:
            # self._selling_log[nmi.annotation["seller"]]=max(self._selling_log[nmi.annotation["seller"]],offer[UNIT_PRICE])
            # else:
            # self._selling_log[nmi.annotation["seller"]] =offer[UNIT_PRICE]
            """if self._bought!=0:
                if self._bought-offer[UNIT_PRICE] >= min(self._buying_log[nmi.annotation["seller"]]*self.get_concession(nmi,state),
                                            (1-self.sigmoid(self.reputation[nmi.annotation["seller"]]))*self.best_offered_buy
                                            ,nmi.issues[UNIT_PRICE].min_value*self.get_concession(nmi,state),
                                            self._bought,
                                            self._buying_log[nmi.annotation["seller"]]*(1-self.sigmoid(self.reputation[nmi.annotation["seller"]])),
                                            self.get_concession(nmi,state)*self.best_offered_buy)- \
                                                    max(self._buying_log[nmi.annotation["seller"]] * self.get_concession(nmi, state),
                                                        (1 - self.sigmoid(self.reputation[nmi.annotation["seller"]])) * self.best_offered_buy
                                                        , nmi.issues[UNIT_PRICE].min_value * self.get_concession(nmi, state),
                                                        self._bought,
                                                        self._buying_log[nmi.annotation["seller"]] * (
                                                                    1 - self.sigmoid(self.reputation[nmi.annotation["seller"]])),
                                                        self.get_concession(nmi, state) * self.best_offered_buy):
                    #print("I accept")
                    return ResponseType.ACCEPT_OFFER
                else:
                    # print("I reject")
                    return ResponseType.REJECT_OFFER

            else:"""
            minmin = [
                self._buying_log[nmi.annotation["seller"]],
                nmi.issues[UNIT_PRICE].min_value,
                self._bought,
                self._buying_log[nmi.annotation["seller"]],
            ]
            maxx = max(
                self._buying_log[nmi.annotation["seller"]],
                nmi.issues[UNIT_PRICE].min_value,
                self._bought,
                self._buying_log[nmi.annotation["seller"]],
                self.best_offered_buy,
                self.best_offered_buy,
            )
            minin = []
            for item in minmin:
                if item != 0:
                    minin.append(item)
            if self.reputation[nmi.annotation["seller"]]:
                threshold = 0.7 * abs(
                    self.sigmoid(self.reputation[nmi.annotation["seller"]])
                )
            else:
                threshold = 0.2
            if threshold < 0.2:
                threshold = 0.2
            th = threshold * (maxx - min(minin))
            if offer[UNIT_PRICE] - min(minin) <= th:
                return ResponseType.ACCEPT_OFFER
            else:
                # print("I reject")

                return ResponseType.REJECT_OFFER

    # =====================
    # Time-Driven Callbacks
    # =====================

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ):
        # print("failed")
        if self.reputation[partners[1]]:
            self.reputation[partners[1]] -= self.failure_list[partners[1]]
        if self.reputation[partners[0]]:
            self.reputation[partners[0]] -= self.failure_list[partners[0]]
        if self.failure_list[partners[1]]:
            self.failure_list[partners[1]] += 1
        else:
            self.failure_list[partners[1]] = 1
        if self.failure_list[partners[0]]:
            self.failure_list[partners[0]] += 1
        else:
            self.failure_list[partners[0]] = 1
        """Called when a negotiation the agent is a party of ends without
        agreement"""

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ):
        """record the history success prices"""
        # print("succeed")
        self.secured += contract.agreement["quantity"]
        if self._is_selling(mechanism):
            if self._sold == 0:
                self._sold = contract.agreement["unit_price"]
            else:
                self._sold = (self._sold + contract.agreement["unit_price"]) / 2
            partner = contract.annotation["buyer"]
            self._selling_log[partner] = (
                max(contract.agreement["unit_price"], self._selling_log[partner])
                if self._selling_log[partner] != 0
                else contract.agreement["unit_price"]
            )
            # self._selling_log[partner]=max(self.neg_log[partner],contract.agreement["unit_price"],self._selling_log[partner]) \
            # if self._selling_log[partner] else contract.agreement["unit_price"]
            if self.reputation[contract.annotation["buyer"]]:
                self.reputation[contract.annotation["buyer"]] += self.succ_list[
                    contract.annotation["buyer"]
                ]
            if self.succ_list[contract.annotation["buyer"]]:
                self.succ_list[contract.annotation["buyer"]] += 1
            else:
                self.succ_list[contract.annotation["buyer"]] = 1
        else:
            if self._bought == 0:
                self._bought = contract.agreement["unit_price"]
            else:
                self._bought = (self._bought + contract.agreement["unit_price"]) / 2
            partner = contract.annotation["seller"]
            self._buying_log[partner] = (
                min(contract.agreement["unit_price"], self._buying_log[partner])
                if self._buying_log[partner] != 0
                else contract.agreement["unit_price"]
            )
            # self._selling_log[partner] = min(self.neg_log[partner],contract.agreement["unit_price"] , self._buying_log[partner])  \
            # if self._buying_log[partner] else contract.agreement["unit_price"]
            if self.reputation[contract.annotation["seller"]]:
                self.reputation[contract.annotation["seller"]] += self.succ_list[
                    contract.annotation["seller"]
                ]
            if self.succ_list[contract.annotation["seller"]]:
                self.succ_list[contract.annotation["seller"]] += 1
            else:
                self.succ_list[contract.annotation["seller"]] = 1

        """Called when a negotiation the agent is a party of ends with
        agreement"""


def run(
    competition="oneshot",
    reveal_names=True,
    n_steps=10,
    n_configs=2,
    n_runs_per_world=1,
):
    if competition == "oneshot":
        competitors = [
            MyAgent,
            LearningAgent,
            SyncRandomOneShotAgent,
            RandomOneShotAgent,
            GreedyOneShotAgent,
        ]
        # competitors = [LearningAgent, MyAgent, RandomOneShotAgent, SyncRandomOneShotAgent, RandomOneShotAgent,GreedyOneShotAgent]
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

    run(sys.argv[1] if len(sys.argv) > 1 else "oneshot")
