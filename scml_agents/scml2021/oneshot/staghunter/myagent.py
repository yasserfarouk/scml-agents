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

import pickle
import random

# required for running tournaments and printing
import time
import warnings
from copy import deepcopy
from functools import partial

# required for typing
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from negmas import (
    AgentMechanismInterface,
    Contract,
    Issue,
    MechanismState,
    ResponseType,
)
from negmas.helpers import humanize_time
from negmas.outcomes import Issue
from negmas.preferences import UtilityFunction
from negmas.sao import AspirationNegotiator, ToughNegotiator

# required for development
from scml.oneshot import OneShotAgent
from scml.oneshot.agent import *
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE
from scml.utils import anac2021_collusion, anac2021_oneshot, anac2021_std
from tabulate import tabulate

# warnings.filterwarnings('error')

SOFTMAX = 0

__all__ = [
    # "StagHunter",
    "StagHunterTough",
]


class EstimatedUtility(UtilityFunction):
    def __init__(self, id, myowner, awi, beta, memory_size, **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self.awi = awi
        self.beta = beta
        self.myowner = myowner
        if memory_size < 0:
            memory_size = self.awi.n_steps
        self.memory_size = memory_size
        self.dataset = [None for _ in range(self.awi.n_steps + 1)]
        self.historical_offers = [[] for _ in range(self.awi.n_steps + 1)]
        self.historical_offers_hash = [set() for _ in range(self.awi.n_steps + 1)]
        self.historical_parameter = [None for _ in range(self.awi.n_steps + 1)]
        self.feature_dim = None
        self.updated = False
        self.buffer = {}

    def update(self, dataset_t, step):
        self.check_updated()
        self.dataset[step] = dataset_t
        self.updated = False
        if type(self.feature_dim) == type(None):
            self.feature_dim = len(self.myowner.get_features(self.id))
            self.weights = np.random.uniform(low=0, high=1, size=self.feature_dim)
            self.weights_bound = np.ones(self.feature_dim)

    def step(self, lr=1e-6):
        self.check_updated()
        if type(self.feature_dim) == type(None):
            self.feature_dim = len(self.myowner.get_features(self.id))
            self.weights = np.random.uniform(low=0, high=1, size=self.feature_dim)
            self.weights_bound = np.ones(self.feature_dim)

        train_sampled_time = list(range(self.awi.current_step, 0, -1))

        test_sampled_time = list(range(self.awi.current_step, 0, -1))

        cand_weights = None
        min_test_loss = float("inf")

        grad_info = []
        loss_info = []

        for _ in range(200):
            cur_grads = self.loss_grad(train_sampled_time)

            grad_info.append(cur_grads)
            loss_info.append(self.loss_function(train_sampled_time))

            self.weights -= lr * cur_grads
            self.weights = np.minimum(self.weights, self.beta)
            self.weights = np.maximum(self.weights, 0)

            test_loss = self.loss_function(test_sampled_time)
            if min_test_loss > test_loss:
                min_test_loss = test_loss
                cand_weights = np.copy(self.weights)

        self.weights = np.copy(cand_weights)

        # with open("./{}info.pkl".format(time.time()), "wb") as f:
        #     pickle.dump((loss_info, grad_info), f)

    def encode(self, t, j, i):
        k = str(t) + "/" + str(j) + "/" + str(i)
        return k

    def add_offer(self, offer, step):
        if hash(offer) not in self.historical_offers_hash[step]:
            self.historical_offers[step].append(offer)
            self.historical_offers_hash[step].add(hash(offer))

    def check_updated(self):
        if type(self.historical_parameter[self.awi.current_step]) == type(None):
            cur_market_parameters = {}
            cur_market_parameters[
                "current_exogenous_input_quantity"
            ] = self.awi.current_exogenous_input_quantity
            cur_market_parameters[
                "current_exogenous_input_price"
            ] = self.awi.current_exogenous_input_price
            cur_market_parameters[
                "current_exogenous_output_quantity"
            ] = self.awi.current_exogenous_output_quantity
            cur_market_parameters[
                "current_exogenous_output_price"
            ] = self.awi.current_exogenous_output_price
            cur_market_parameters["current_step"] = self.awi.current_step
            cur_market_parameters["current_balance"] = self.awi.current_balance
            cur_market_parameters["cost"] = self.awi.profile.cost
            cur_market_parameters["n_lines"] = self.awi.profile.n_lines
            cur_market_parameters[
                "current_shortfall_penalty"
            ] = self.awi.current_shortfall_penalty
            cur_market_parameters[
                "current_disposal_cost"
            ] = self.awi.current_disposal_cost
            self.historical_parameter[self.awi.current_step] = cur_market_parameters
            if self.awi.is_first_level:
                self.issues = self.awi.current_output_issues
            else:
                self.issues = self.awi.current_input_issues

    def loss_function(self, sampled_time):
        loss = 0
        self_output = True if self.awi.is_first_level else False
        for t in sampled_time:
            features, ground_truth_offer, ground_truth_output = (
                self.dataset[t][0],
                self.dataset[t][1],
                self.dataset[t][2],
            )
            loss_t = 0
            for j in range(len(self.historical_offers[t])):
                cur_offer = self.historical_offers[t][j]
                offers = [cur_offer] + ground_truth_offer
                outputs = [self_output] + ground_truth_output

                code = hash(str(cur_offer) + "*" + str(t))

                if code not in self.buffer:
                    self.buffer[code] = self.from_offers(
                        offers, outputs, self.historical_parameter[t]
                    )
                ground_truth = self.buffer[code]

                while True:
                    res = 0
                    counter = 0
                    trace = []
                    for i in range(self.awi.current_step, 0, -1):
                        if i == t:
                            continue
                        cur_features = self.dataset[i][0]
                        other_offers = self.dataset[i][1]
                        other_is_output = self.dataset[i][2]
                        offers = [cur_offer] + other_offers
                        outputs = [self_output] + other_is_output
                        cof = np.exp(
                            -np.sum(self.weights * np.square(cur_features - features))
                        )
                        self.weights_bound = np.maximum(
                            self.weights_bound, np.square(cur_features - features)
                        )
                        code = hash(str(cur_offer) + "*" + str(i))
                        if code not in self.buffer:
                            self.buffer[code] = self.from_offers(
                                offers, outputs, self.historical_parameter[t]
                            )
                        cur_payoff = self.buffer[code]
                        trace.append(
                            (
                                self.weights,
                                cur_features,
                                features,
                                np.square(cur_features - features),
                            )
                        )

                        res += cof * cur_payoff
                        counter += cof
                    if counter <= 1e-2:
                        self.weights = 1 / self.weights_bound
                    else:
                        break

                # with warnings.catch_warnings():
                #     try:
                #         loss_t += (res/counter - ground_truth) ** 2
                #     except:
                #         with open("{}loss.pkl".format(time.time()), "wb") as f:
                #             pickle.dump((trace, self.dataset), f)
                loss_t += (res / counter - ground_truth) ** 2
            if len(self.historical_offers[t]):
                loss += loss_t / len(self.historical_offers[t])
        return loss / len(sampled_time)

    def loss_grad(self, sampled_time):
        grad_weights = np.zeros(self.feature_dim)
        self_output = True if self.awi.is_first_level else False
        for t in sampled_time:
            features, ground_truth_offer, ground_truth_output = (
                self.dataset[t][0],
                self.dataset[t][1],
                self.dataset[t][2],
            )
            grad_weights_t = np.zeros(self.feature_dim)
            for j in range(len(self.historical_offers[t])):
                cur_offer = self.historical_offers[t][j]
                offers = [cur_offer] + ground_truth_offer
                outputs = [self_output] + ground_truth_output
                code = hash(str(cur_offer) + "*" + str(t))
                if code not in self.buffer:
                    self.buffer[code] = self.from_offers(
                        offers, outputs, self.historical_parameter[t]
                    )
                ground_truth = self.buffer[code]

                while True:
                    term1 = 0
                    term2 = 0
                    term3 = np.zeros(self.feature_dim)
                    term4 = np.zeros(self.feature_dim)

                    trace = []

                    for i in range(self.awi.current_step, 0, -1):
                        if i == t:
                            continue
                        cur_features = self.dataset[i][0]
                        other_offers = self.dataset[i][1]
                        other_is_output = self.dataset[i][2]
                        offers = [cur_offer] + other_offers
                        outputs = [self_output] + other_is_output
                        cof = np.exp(
                            -np.sum(self.weights * np.square(cur_features - features))
                        )
                        self.weights_bound = np.maximum(
                            self.weights_bound, np.square(cur_features - features)
                        )
                        code = hash(str(cur_offer) + "*" + str(i))
                        if code not in self.buffer:
                            self.buffer[code] = self.from_offers(
                                offers, outputs, self.historical_parameter[t]
                            )
                        cur_payoff = self.buffer[code]
                        trace.append(
                            (
                                self.weights,
                                cur_features,
                                features,
                                np.square(cur_features - features),
                            )
                        )
                        term1 += cof
                        term2 += cof * cur_payoff
                        term3 += cof * np.square(cur_features - features)
                        term4 += cof * cur_payoff * np.square(cur_features - features)
                    if term1 <= 1e-2:
                        self.weights = 1 / self.weights_bound
                    else:
                        break

                # with warnings.catch_warnings():
                #     try:
                #         cur_grad = 2 * (term2/term1 - ground_truth) * (term3 * term2 - term4 * term1)/(term1 ** 2)
                #     except:
                #         with open("{}grad.pkl".format(time.time()), "wb") as f:
                #             pickle.dump((term1, self.weights, trace, self.dataset), f)
                cur_grad = (
                    2
                    * (term2 / term1 - ground_truth)
                    * (term3 * term2 - term4 * term1)
                    / (term1**2)
                )
                grad_weights_t += cur_grad
            if len(self.historical_offers[t]):
                grad_weights += grad_weights_t / len(self.historical_offers[t])
        return grad_weights / len(sampled_time)

    def eval(self, offer):
        # self.historical_offers[self.awi.current_step].append(offer)

        self.check_updated()
        features = self.myowner.get_features(self.id)

        is_output = True if self.awi.is_first_level else False
        if self.awi.current_step == 0:
            return self.from_offers(
                [offer], [is_output], self.historical_parameter[self.awi.current_step]
            )

        while True:
            res = 0
            counter = 0
            for i in range(
                self.awi.current_step - 1, max(self.awi.current_step - 5, -1), -1
            ):
                # print(self.dataset)
                cur_features = self.dataset[i][0]
                other_offers = self.dataset[i][1]
                other_is_output = self.dataset[i][2]
                offers = [offer] + other_offers
                outputs = [is_output] + other_is_output
                cof = 1
                if SOFTMAX:
                    cof = np.exp(
                        -np.sum(self.weights * np.square(cur_features - features))
                    )

                self.weights_bound = np.maximum(
                    self.weights_bound, np.square(cur_features - features)
                )

                res += cof * self.from_offers(
                    offers, outputs, self.historical_parameter[self.awi.current_step]
                )
                counter += cof
            if counter <= 1e-2:
                self.weights = 1 / self.weights_bound
            else:
                break
        return res / counter

    def from_offers(self, offers, outputs, param):
        def order(x):
            offer, is_output, is_exogenous = x
            # if is_exogenous and self.force_exogenous:
            #     return float("-inf")
            return -offer[UNIT_PRICE] if is_output else offer[UNIT_PRICE]

        current_exogenous_input_quantity = param["current_exogenous_input_quantity"]
        current_exogenous_input_price = param["current_exogenous_input_price"]
        current_exogenous_output_quantity = param["current_exogenous_output_quantity"]
        current_exogenous_output_price = param["current_exogenous_output_price"]
        current_step = param["current_step"]
        current_balance = param["current_balance"]
        cost = param["cost"]
        n_lines = param["n_lines"]
        current_shortfall_penalty = param["current_shortfall_penalty"]
        current_disposal_cost = param["current_disposal_cost"]
        # copy inputs because we are going to modify them.
        original_offers = deepcopy(list(offers))
        offers, outputs = deepcopy(list(offers)), deepcopy(list(outputs))
        # indicate that all inputs are not exogenous and that we are adding two
        # exogenous contracts after them.
        exogenous = [False] * len(offers) + [True, True]
        # add exogenous contracts as offers one for input and another for output
        offers += [
            (
                current_exogenous_input_quantity,
                current_step,
                current_exogenous_input_price / current_exogenous_input_quantity
                if current_exogenous_input_quantity
                else 0,
            ),
            (
                current_exogenous_output_quantity,
                0,
                current_exogenous_output_price / current_exogenous_output_quantity
                if current_exogenous_output_quantity
                else 0,
            ),
        ]
        outputs += [False, True]
        # initialize some variables
        qin, qout, pin, pout = 0, 0, 0, 0
        qin_bar, going_bankrupt = 0, current_balance < 0
        pout_bar = 0
        # we are going to collect output contracts in output_offers
        output_offers = []
        # sort contracts in the optimal order of execution: from cheapest when
        # buying and from the most expensive when selling. See `order` above.
        # try:
        sorted_offers = list(sorted(zip(offers, outputs, exogenous), key=order))
        # except:
        #     with open("./{}offers{}.pkl".format(self.awi.current_step, time.time()), "wb") as f:
        #         pickle.dump((original_offers, self.awi.current_step, self.dataset, self.historical_offers), f)

        # we calculate the total quantity we are are required to pay for `qin` and
        # the associated amount of money we are going to pay `pin`. Moreover,
        # we calculate the total quantity we can actually buy given our limited
        # money balance (`qin_bar`).
        for offer, is_output, is_exogenous in sorted_offers:
            offer = self.outcome_as_tuple(offer)
            if is_output:
                output_offers.append((offer, is_exogenous))
                continue
            topay_this_time = offer[UNIT_PRICE] * offer[QUANTITY]
            if not going_bankrupt and (
                pin + topay_this_time + offer[QUANTITY] * cost > current_balance
            ):
                unit_total_cost = offer[UNIT_PRICE] + cost
                can_buy = int((current_balance - pin) // unit_total_cost)
                qin_bar = qin + can_buy
                going_bankrupt = True
            pin += topay_this_time
            qin += offer[QUANTITY]

        if not going_bankrupt:
            qin_bar = qin

        # calculate the maximum amount we can produce given our limited production
        # capacity and the input we CAN BUY
        n_lines = n_lines
        producible = min(qin_bar, n_lines)

        # No need to this test now because we test for the ability to produce with
        # the ability to buy items. The factory buys cheaper items and produces them
        # before attempting more expensive ones. This may or may not be optimal but
        # who cars. It is consistent that it is all that matters.
        # # if we do not have enough money to pay for production in full, we limit
        # # the producible quantity to what we can actually produce
        # if (
        #     self.production_cost
        #     and producible * self.production_cost > self.current_balance
        # ):
        #     producible = int(self.current_balance // self.production_cost)

        # find the total sale quantity (qout) and money (pout). Moreover find
        # the actual amount of money we will receive
        done_selling = False
        for offer, is_exogenous in output_offers:
            if not done_selling:
                if qout + offer[QUANTITY] >= producible:
                    assert producible >= qout, f"producible {producible}, qout {qout}"
                    can_sell = producible - qout
                    done_selling = True
                else:
                    can_sell = offer[QUANTITY]
                pout_bar += can_sell * offer[UNIT_PRICE]
            pout += offer[UNIT_PRICE] * offer[QUANTITY]
            qout += offer[QUANTITY]

        # should never produce more than we signed to sell
        producible = min(producible, qout)

        # we cannot produce more than our capacity or inputs and we should not
        # produce more than our required outputs
        producible = min(qin, n_lines, producible)

        # the scale with which to multiply disposal_cost and shortfall_penalty
        # if no scale is given then the unit price will be used.
        output_penalty = pout / qout if qout else 0
        output_penalty *= current_shortfall_penalty * max(0, qout - producible)

        input_penalty = pin / qin if qin else 0
        input_penalty *= current_disposal_cost * max(0, qin - producible)

        # call a helper method giving it the total quantity and money in and out.
        u = self.from_aggregates(
            qin,
            qout,
            producible,
            pin,
            pout_bar,
            input_penalty,
            output_penalty,
            cost,
            n_lines,
        )

        return u

    def from_aggregates(
        self,
        qin: int,
        qout_signed: int,
        qout_sold: int,
        pin: int,
        pout: int,
        input_penalty,
        output_penalty,
        cost,
        n_lines,
    ) -> float:
        """
        Calculates the utility from aggregates of input/output quantity/prices

        Args:
            qin: Input quantity (total including all exogenous contracts).
            qout_signed: Output quantity (total including all exogenous contracts)
                         that the agent agreed to sell.
            qout_sold: Output quantity (total including all exogenous contracts)
                       that the agent will actually sell.
            pin: Input total price (i.e. unit price * qin).
            pout: Output total price (i.e. unit price * qin).
            input_penalty: total disposal cost
            output_penalty: total shortfall penalty

        Remarks:
            - Most likely, you do not need to directly call this method. Consider
              `from_offers` and `from_contracts` that take current balance and
              exogenous contract information (passed during ufun construction)
              into account.
            - The method respects production capacity (n. lines). The
              agent cannot produce more than the number of lines it has.
            - This method does not take exogenous contracts or current balance
              into account.
            - The method assumes that the agent CAN pay for all input
              and production.

        """
        assert qout_sold <= qout_signed, f"sold: {qout_sold}, signed: {qout_signed}"

        # production capacity
        lines = n_lines

        # we cannot produce more than our capacity or inputs and we should not
        # produce more than our required outputs
        produced = min(qin, lines, qout_sold)

        # self explanatory. right?  few notes:
        # 1. You pay disposal costs for anything that you buy and do not produce
        #    and sell. Because we know that you sell no more than what you produce
        #    we can multiply the disposal cost with the difference between input
        #    quantity and the amount produced
        # 2. You pay shortfall penalty for anything that you should have sold but
        #    did not. The only reason you cannot sell something is if you cannot
        #    produce it. That is why the shortfall penalty is multiplied by the
        #    difference between what you should have sold and the produced amount.
        u = pout - pin - cost * produced - input_penalty - output_penalty

        return u

    def outcome_as_tuple(self, offer):
        if isinstance(offer, dict):
            outcome = [None] * 3
            outcome[QUANTITY] = offer["quantity"]
            outcome[TIME] = offer["time"]
            outcome[UNIT_PRICE] = offer["unit_price"]
            return tuple(outcome)
        return tuple(offer)

    def xml(self, issues: List[Issue]) -> str:
        return ""


class StagHunter(OneShotAgent):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details

    """

    # =====================
    # Negotiation Callbacks
    # =====================

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self, beta=5, memory_size=50):
        """Called once after the agent-world interface is initialized"""

        if not self.awi:
            raise ValueError(f"I do not know my AWI!!!")
        # GreedySyncAgent.init()

        if self.awi.is_first_level:
            self.oppo_id_list = self.awi.my_consumers
        else:
            self.oppo_id_list = self.awi.my_suppliers

        self.dataset = {
            id_: [[[], [], []] for _ in range(self.awi.n_steps + 1)]
            for id_ in self.oppo_id_list
        }

        self._ufuns__ = {
            id_: EstimatedUtility(
                id=id_,
                myowner=self,
                awi=self.awi,
                beta=beta,
                memory_size=memory_size,
                issues=self.awi.current_output_issues
                if self.awi.is_first_level
                else self.awi.current_input_issues,
            )
            for id_ in self.oppo_id_list
        }

        self.__negotiatiors = {
            id_: AspirationNegotiator(ufun=self._ufuns__[id_])
            for id_ in self.oppo_id_list
        }

        super().init()
        # print(self.ufun)

    def propose(self, negotiator_id, state):
        # with open("propose{}.pkl".format(self.awi.current_step), "wb") as f:
        #     pickle.dump((offer, self.awi.current_step), f)
        res = None
        if negotiator_id in self.oppo_id_list:
            res = self.__negotiatiors[negotiator_id].propose(state=state)
            if type(res) != type(None):
                self._ufuns__[negotiator_id].add_offer(
                    offer=res, step=self.awi.current_step
                )
        return res

    def response(self, negotiator_id, state, offer):
        # with open("response{}.pkl".format(self.awi.current_step), "wb") as f:
        #     pickle.dump((offer, self.awi.current_step), f)
        res = ResponseType.END_NEGOTIATION
        if negotiator_id in self.oppo_id_list:
            res = self.__negotiatiors[negotiator_id].response(state=state, offer=offer)
            if type(offer) != type(None):
                self._ufuns__[negotiator_id].add_offer(
                    offer=offer, step=self.awi.current_step
                )
        return res

    # def generate_ufuns(self):
    #
    #     if(self.awi.current_step < 3):
    #         if(self.awi.current_step == 2):
    #             for id_ in self.oppo_id_list:
    #                 self._ufuns__[id_].set_feature_dim(len(self.get_features(id_)))
    #
    #         return dict([(id_, self.ufun) for id_ in self.oppo_id_list])
    #     return self._ufuns__

    def step(self):
        for id_ in self.oppo_id_list:
            self._ufuns__[id_].update(
                self.dataset[id_][self.awi.current_step], self.awi.current_step
            )

            if self.awi.current_step > 0 and self.awi.current_step % 5 == 0 and SOFTMAX:
                self._ufuns__[id_].step()

        self.__negotiatiors = {
            id_: AspirationNegotiator(ufun=self._ufuns__[id_])
            for id_ in self.oppo_id_list
        }
        super().step()

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without
        agreement"""

        # print("failure !!!!!!")
        # with open("./failure.pkl", "wb") as f:
        #     pickle.dump([], f)
        partner_id = (
            annotation["buyer"]
            if annotation["seller"] == self._owner.id
            else annotation["seller"]
        )

        for id_ in self.oppo_id_list:
            if id_ != partner_id:
                if not len(self.dataset[id_][self.awi.current_step][0]):
                    self.dataset[id_][self.awi.current_step][0] = self.get_features(id_)

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        """Called when a negotiation the agent is a party of ends with
        agreement"""

        # print("success !!!!!!")
        # with open("./success.pkl", "wb") as f:
        #     pickle.dump([], f)

        a = contract.annotation
        partner_id = a["buyer"] if a["seller"] == self._owner.id else a["seller"]

        output_product = self.awi.my_output_product
        product = contract.annotation["product"]
        is_output = product == output_product
        offer = contract.agreement
        if isinstance(offer, dict):
            outcome = [None] * 3
            outcome[QUANTITY] = offer["quantity"]
            outcome[TIME] = offer["time"]
            outcome[UNIT_PRICE] = offer["unit_price"]
            offer = tuple(outcome)
        else:
            offer = tuple(offer)

        for id_ in self.oppo_id_list:
            if id_ != partner_id:
                if not len(self.dataset[id_][self.awi.current_step][0]):
                    self.dataset[id_][self.awi.current_step][0] = self.get_features(id_)
                self.dataset[id_][self.awi.current_step][1].append(offer)
                self.dataset[id_][self.awi.current_step][2].append(is_output)

        super().on_negotiation_success(contract, mechanism)

    def get_features(self, negotiator_id):
        oppo_features = []

        for id_ in self.oppo_id_list:
            try:
                reports = self.awi.reports_at_step(self.awi.current_step)
                oppo_report = reports[id_]
                oppo_features += [
                    self.awi.current_step - oppo_report.step,
                    oppo_report.breach_prob,
                    oppo_report.breach_level,
                ]
            # print(type(oppo_report))
            # if(type(oppo_report) == dict):
            #     print(oppo_report.keys())
            except:
                oppo_features += [0, 0, 0]

        # market_features = list(self.awi.trading_prices) + [product_info[0] for product_info in self.awi.exogenous_contract_summary] + [product_info[1] for product_info in self.awi.exogenous_contract_summary]
        # market_features = [product_info[0] for product_info in self.awi.exogenous_contract_summary] + [product_info[1]/(product_info[0]) for product_info in self.awi.exogenous_contract_summary]
        market_features = [
            self.awi.exogenous_contract_summary[0][0],
            self.awi.exogenous_contract_summary[0][1]
            / (self.awi.exogenous_contract_summary[0][0] * self.awi.trading_prices[0]),
            self.awi.exogenous_contract_summary[-1][0],
            self.awi.exogenous_contract_summary[-1][1]
            / (
                self.awi.exogenous_contract_summary[-1][0] * self.awi.trading_prices[-1]
            ),
        ]

        # my_features = [self.awi.current_disposal_cost, self.awi.current_shortfall_penalty, self.awi.current_balance]

        my_features = [
            self.awi.current_disposal_cost
            / self.awi.trading_prices[self.awi.my_input_product],
            self.awi.current_shortfall_penalty
            / self.awi.trading_prices[self.awi.my_output_product],
        ]

        if self.awi.is_first_level:
            my_features += [
                self.awi.current_exogenous_input_quantity,
                self.awi.current_exogenous_input_price
                / self.awi.trading_prices[self.awi.my_input_product],
            ]
        else:
            my_features += [
                self.awi.current_exogenous_output_quantity,
                self.awi.current_exogenous_output_price
                / self.awi.trading_prices[self.awi.my_output_product],
            ]

        features = oppo_features + market_features + my_features
        # print(features)
        return np.array(features)


class StagHunterTough(StagHunter):
    def init(self, beta=5, memory_size=50):
        super().init(beta, memory_size)
        self.__negotiatiors = {
            id_: ToughNegotiator(ufun=self._ufuns__[id_]) for id_ in self.oppo_id_list
        }


# class StagHunterTough2(StagHungerAsp):
#     def init(self, discounter_factor=0.9, memory_size=5):
#         super().init(discounter_factor, memory_size)
#         self.oppo_negotiators = dict([(id, ToughNegotiator(ufun=self.ufuncs[id])) for id in self.oppo_id_list])
#
# class StagHunterOB(StagHungerAsp):
#     def init(self, discounter_factor=0.99, memory_size=10):
#         super().init(discounter_factor, memory_size)
#         self.oppo_negotiators = dict([(id, TopFractionNegotiator(ufun=self.ufuncs[id])) for id in self.oppo_id_list])
#
#
# class StagHunterNTFT(StagHungerAsp):
#     def init(self, discounter_factor=0.99, memory_size=10):
#         super().init(discounter_factor, memory_size)
#         self.oppo_negotiators = dict([(id, NaiveTitForTatNegotiator(ufun=self.ufuncs[id])) for id in self.oppo_id_list])
#
#
# class StagHunterNTFT2(StagHungerAsp):
#     def init(self, discounter_factor=0.9, memory_size=5):
#         super().init(discounter_factor, memory_size)
#         self.oppo_negotiators = dict([(id, NaiveTitForTatNegotiator(ufun=self.ufuncs[id])) for id in self.oppo_id_list])
#
# class StagHunterAsp2(StagHungerAsp):
#     def init(self, discounter_factor=0.9, memory_size=5):
#         super().init(discounter_factor, memory_size)
#
# class StagHunterSTFT(StagHungerAsp):
#     def init(self, discounter_factor=0.99, memory_size=10):
#         super().init(discounter_factor, memory_size)
#         self.oppo_negotiators = dict([(id, SimpleTitForTatNegotiator(ufun=self.ufuncs[id])) for id in self.oppo_id_list])
#
#
# class StagHunterNice(StagHungerAsp):
#     def init(self, discounter_factor=0.99, memory_size=10):
#         super().init(discounter_factor, memory_size)
#         self.oppo_negotiators = dict([(id, NiceNegotiator(ufun=self.ufuncs[id])) for id in self.oppo_id_list])


def run(
    competition="oneshot",
    reveal_names=True,
    n_steps=100,
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
    from scml.oneshot.agents import (
        GreedyOneShotAgent,
        GreedySingleAgreementAgent,
        GreedySyncAgent,
        RandomOneShotAgent,
        SyncRandomOneShotAgent,
    )

    if competition == "oneshot":
        # competitors = [StagHungerAsp, StagHunterOB, StagHunterNice, StagHunterNTFT, StagHunterSTFT, StagHunterTough, RandomOneShotAgent, SyncRandomOneShotAgent, GreedyOneShotAgent, GreedySyncAgent, GreedySingleAgreementAgent]

        competitors = [
            StagHunter,
            StagHunterTough,
            RandomOneShotAgent,
            SyncRandomOneShotAgent,
            GreedyOneShotAgent,
            GreedySyncAgent,
            GreedySingleAgreementAgent,
        ]
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
        compact=True,
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
