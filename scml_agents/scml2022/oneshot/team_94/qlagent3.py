#!/usr/bin/env python
import argparse
import ast
import csv
import itertools
import math
import os
import pickle
import random

# other libraries
import sys

# required for running tournaments and printing
import time
from collections import defaultdict

# required for typing
from typing import Any, Dict, List, Optional

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

# required for development
from numpy.lib.arraysetops import isin
from scml.oneshot import OneShotAgent
from scml.oneshot.agents import RandomOneShotAgent, SyncRandomOneShotAgent
from scml.oneshot.world import SCML2020OneShotWorld, is_system_agent
from scml.utils import anac2021_collusion, anac2021_oneshot, anac2021_std
from tabulate import tabulate

__all__ = ["AdaptiveQlAgent"]


# Flags
DEBUG = False  # True <-> minor and many debug prints ("type #1")
DEBUG2 = False  # True <-> major debug prints ("type #2")
ENABLE_GRAPH = False  # True <-> enabling graphs in end of run (reward to time - learning curve and estimated balance to time)
TO_SUBMISSION = True  # True <-> for submission to SCML (tournament not running)
SAVE_LEARNING_DB = True  # True <-> save the learning DB (Q tables) to files

# consts to control price range min and max for agent. 1 - use the sim range and no "stretching" or "squizing" the range
G_C_MX_V = 1  # .2
G_C_MN_V = 1  # 0.8
G_C_MX4MN_V = 1
G_C_MN4MX_V = 1

# enabling inspection graphs
if ENABLE_GRAPH:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")
    # matplotlib.use('Agg')
    from scipy.signal import butter, filtfilt


# Debug print type #1
def DEBUG_PRINT(msg):
    if DEBUG:
        pass  # print(msg)


# Debug print type #2
def DEBUG2_PRINT(msg):
    if DEBUG2:
        pass  # print(msg)


# Offer field enum
class OFFER_FIELD_IDX:
    QUANTITY = 0
    TIME = 1
    UNIT_PRICE = 2


# State type enum
# Some are used directly, some are representing a range of values (expanded within the agent's code)
class STATE_TYPE:
    NO_NEGO = -1
    INIT = 0
    # OPP_OVERRIDE = 1
    END = 2
    ACCEPT = 3
    MY_COUNTER = 4
    OPP_COUNTER = 5


""""
The QlAgent class
-----------------
The QlAgent class implements our agent for the SCML competition.
The QlAgent implements an SCML negotation agent by implementing Q-Learning algorithm
with adaptation to the Negotation problem and SCML framework.
Note : there are some commented-out code parts. This is because we played a lot and tried
a lot of things until the last moment. We left it here for future optional development, as it
may have some value.
"""


class AdaptiveQlAgent(OneShotAgent):
    """
    Contructor
    """

    def __init__(
        self,
        load_q=True,  # load Q table from external file (pickle)
        load_q_type="exact",  # loading mode - "exact" for loading directly from the file without adaptation
        save_q=SAVE_LEARNING_DB,  # save Q table to external file (pickle)
        save_q_type="exact",  # saving mode - "exact" for saving directly from the file without adaptation
        alpha=0.1,  # 0.1,#0.1,            # Q learning alpha parameter (learning rate)
        alpha_decay_rate=0,  # 0.0001,      # Q learning learning rate decay parameter
        prune_quan=1.2,  # used to reduce the amount of quantities the agent can offer
        gamma=0.95,  # Q learning gamma parameter
        price_res=5,  # price resolution (amount of prices that can be offered / used in state)
        quantity_res=5,  # quantity resolution (amount of quantities that can be offered / used in state)
        needs_res=5,  # needs resolution (amount of possible values to represent needs status - how much do I left to sell/buy)
        epsilon_start=0.1,  # 0.1,#0.1,#0.2       # epsilon-greedy start value
        epsilon_end=0.005,  # 0.0001,#0.0001   # epsilon-greedy end value
        epsilon_decay=10,  # 10,           # epsilon-greedy decay value
        opportunistic_level=0.001,  # opportunistic level : probability that try and generate an opportunistic offer for us
        smart_init=True,  # should always be True. Makes the Q table init with good values, if no external Q file exists
        complex_state=True,  # should always be True. Makes the state space rich and represent the actual env state more accurately (the states are explained in the report)
        concession_exponent=0.3,  # None,  # LearningAgent "imitation" parameters. DISABLED - CURRENTLY NOT FULLY SUPPORTED - DO NOT CHANGE VALUE
        acc_price_slack=float(
            "inf"
        ),  # LearningAgent "imitation" parameters. DISABLED - CURRENTLY NOT FULLY SUPPORTED - DO NOT CHANGE VALUE
        step_price_slack=0.0,  # LearningAgent "imitation" parameters. DISABLED - CURRENTLY NOT FULLY SUPPORTED - DO NOT CHANGE VALUE
        opp_price_slack=0.0,  # LearningAgent "imitation" parameters. DISABLED - CURRENTLY NOT FULLY SUPPORTED - DO NOT CHANGE VALUE
        opp_acc_price_slack=0.2,  # LearningAgent "imitation" parameters. DISABLED - CURRENTLY NOT FULLY SUPPORTED - DO NOT CHANGE VALUE
        range_slack=0.03,  # LearningAgent "imitation" parameters. DISABLED - CURRENTLY NOT FULLY SUPPORTED - DO NOT CHANGE VALUE
        epsilon2_start=0,  # 0.5,          # epsilon2-greedy start value - for amount of LearningAgent "imitation" parameters. DISABLED - CURRENTLY NOT FULLY SUPPORTED - DO NOT CHANGE VALUE
        epsilon2_end=0,  # ,0.01,#0.1      # epsilon2-greedy end value - for amount of LearningAgent "imitation" parameters. DISABLED - CURRENTLY NOT FULLY SUPPORTED - DO NOT CHANGE VALUE
        epsilon2_decay=200,  # epsilon2-greedy decay value - for amount of LearningAgent "imitation" parameters. DISABLED - CURRENTLY NOT FULLY SUPPORTED - DO NOT CHANGE VALUE
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # attributes init
        self.secured = 0
        self.started = dict()
        self.opposite_price_gap = dict()
        self.success = 0
        self.failure = 0
        self.profit = 0
        self.state_per_opp = dict()
        self.last_a_per_opp = dict()
        self.q_table_per_opp = dict()
        self.alpha = alpha
        self.alpha_decay_rate = alpha_decay_rate
        self.gamma = gamma
        self.price_res = price_res
        self.quantity_res = quantity_res
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.action_vec = dict()
        self.r_after_episode = dict()
        self.q_after_episode = dict()
        self.q_steps = dict()
        self.profit_per_opp = dict()
        self.smart_init = smart_init
        self.complex_state = complex_state
        self.episodes = 0
        self.num_of_negotiations_per_step = dict()
        self.num_of_negotiations_per_step_all = 0
        self.prune_quan = prune_quan
        self.needs_res = needs_res
        self.load_q = load_q
        self.load_q_type = load_q_type
        self.save_q = save_q
        self.save_q_type = save_q_type
        self.is_seller = True
        self.epsilon2_start = epsilon2_start
        self.epsilon2_end = epsilon2_end
        self.epsilon2_decay = epsilon2_decay
        self.fail_counters = dict()
        self.opportunistic_level = opportunistic_level
        self.online_balance = None

        self.counters = dict()
        for i in range(12):
            self.counters[i + 1] = 0

        self.balance_over_time = []

        # loading external Q tables (per "seller" / "buyer")
        # "seller" = L0 position for agent
        # "buyer" = L1 position for agent
        # both may require different learning, because goals are opposite from negotiation price perspective.
        # that's why we have learning database for each type.
        if self.load_q:
            try:
                # with open('q_seller.pickle', 'rb') as handle:
                #     self.learned_q_seller_t = pickle.load(handle)
                with open("q_seller.txt") as handle_s:
                    self.learned_q_seller_t = ast.literal_eval(handle_s.read())
                DEBUG_PRINT("UPLOADED FROM q_seller.pickle")
            except FileNotFoundError:
                self.load_q = False

            try:
                # with open('q_buyer.pickle', 'rb') as handle:
                #     self.learned_q_buyer_t = pickle.load(handle)
                with open("q_buyer.txt") as handle_b:
                    self.learned_q_buyer_t = ast.literal_eval(handle_b.read())
                DEBUG_PRINT("UPLOADED FROM q_buyer.pickle")
            except FileNotFoundError:
                self.load_q = False

        if concession_exponent is None:
            concession_exponent = 0.2 + random.random() * 0.8
        if step_price_slack is None:
            step_price_slack = random.random() * 0.1 + 0.05
        if opp_price_slack is None:
            opp_price_slack = random.random() * 0.1 + 0.05
        if opp_acc_price_slack is None:
            opp_acc_price_slack = random.random() * 0.1 + 0.05
        if range_slack is None:
            range_slack = random.random() * 0.2 + 0.05

        self._e = concession_exponent
        self._acc_price_slack = acc_price_slack
        self._step_price_slack = step_price_slack
        self._opp_price_slack = opp_price_slack
        self._opp_acc_price_slack = opp_acc_price_slack
        self._range_slack = range_slack

        self.num_of_nego_total = 0

    """
    A method for initializing verious parameters that cannot be set on constructor
    (e.g. need some underlying framework to be up or just more convenient here)
    """

    def init(self):
        """Initialize the quantities and best prices received so far and more variables"""
        self.ne_range = np.concatenate(
            (np.array([-1]), np.linspace(0, self.awi.n_lines + 1, self.needs_res))
        )
        self.adviced_list = defaultdict(int)
        self.failure_list = defaultdict(int)
        self.failure_threshold = defaultdict(int)
        self._best_selling, self._best_buying = 0.0, float("inf")
        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._best_opp_acc_selling = defaultdict(float)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))
        self._sales = self._supplies = 0

        self.online_balance = self.awi.current_balance

    """
    A method called in each step of the simulation.
    Using this for:
    - updating / zeroing some state and tracking objects between steps
    - saving final Q table to file
    - displaying graph in end of sim
    - used for debug if enabled
    """

    def step(self):
        """Initialize the quantities and best prices received for next step"""
        self._best_selling, self._best_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._sales = self._supplies = 0

        # print(self.state_per_opp)
        # if self.awi.current_step == 0:
        #     print("-----------------------------------------------")
        DEBUG_PRINT("# STEP : " + str(self.awi.current_step))
        DEBUG_PRINT("# Episodes  : " + str(self.episodes))
        DEBUG_PRINT("# Needed : " + str(self._needed()))

        # updating state and debug objects
        self.online_balance = self.awi.current_balance
        self.balance_over_time.append(self.online_balance)
        self.failure_threshold = defaultdict(int)
        self.adviced_list = defaultdict(int)
        if self.awi.current_step % 100 == 0:
            DEBUG2_PRINT("# STEP : " + str(self.awi.current_step))
            DEBUG2_PRINT(self.counters)
            DEBUG2_PRINT(self.fail_counters)

        #    print(self.q_table_per_opp[partner])
        # print("s: " + str(self.success) + ", f: " + str(self.failure))
        # #print(self.current_agreed_price)
        # print("profit: " + str(self.profit))
        # print(self.q_table_per_opp)

        # init the "secured" value between steps (as inventory is not kept)
        self.secured = 0

        # various actions for inspection graphs purposes
        for partner in self.r_after_episode.keys():
            self.r_after_episode[partner].append(0)

        for partner in self.q_steps.keys():
            self.q_steps[partner] = 0

        if self.is_seller:
            postfix = "_seller"
        else:
            postfix = "_buyer"

        if self.awi.current_step == 0:
            DEBUG2_PRINT(postfix)
            for partner in self.started.keys():
                DEBUG2_PRINT(partner)

        # saving Q tables if enabled
        if (self.awi.current_step > 0) and (
            (self.awi.current_step % 5000 == 0)
            or (self.awi.current_step == self.awi.n_steps - 1)
        ):
            if self.save_q:
                with open("q" + postfix + ".txt", "w") as handle:  # .pickle
                    if self.save_q_type == "exact":
                        lea_k = None
                        k = None
                        for k in self.q_table_per_opp.keys():
                            if "Lea" in k:
                                # print("Saving Lea !")
                                lea_k = k
                                break
                        if lea_k is None:
                            my_q = list(self.q_table_per_opp.values())[0]
                        else:
                            my_q = self.q_table_per_opp[k]
                        # if self.load_q:
                        #     # if we came from a loaded table, we add entries from there we don't have in our new
                        #     # learned table, in order to gather as much Q value as possible for different scenarios
                        #     if self.is_seller:
                        #         for s in self.learned_q_seller_t:
                        #             if s not in my_q.keys():
                        #                 my_q[s] = self.learned_q_seller_t[s]
                        #             else:
                        #                 for a in self.learned_q_seller_t[s]:
                        #                     if a not in my_q[s].keys():
                        #                         my_q[s][a] = self.learned_q_seller_t[s][a]
                        #     else:
                        #         for s in self.learned_q_buyer_t:
                        #             if s not in my_q.keys():
                        #                 my_q[s] = self.learned_q_buyer_t[s]
                        #             else:
                        #                 for a in self.learned_q_buyer_t[s]:
                        #                     if a not in my_q[s].keys():
                        #                         my_q[s][a] = self.learned_q_buyer_t[s][a]

                        DEBUG2_PRINT("saving " + postfix)
                        handle.write(str(my_q))

                        DEBUG2_PRINT("saved " + postfix)
                        # pickle.dump(my_q, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        handle.write(
                            str(self.map_q(list(self.q_table_per_opp.values())[0]))
                        )
                        # pickle.dump(self.map_q(list(self.q_table_per_opp.values())[0]), handle, protocol=pickle.HIGHEST_PROTOCOL)

        # drawing graphs if enabled
        if (self.awi.current_step > 0) and (
            (self.awi.current_step % 40000 == 0)
            or (self.awi.current_step == self.awi.n_steps - 1)
        ):
            if ENABLE_GRAPH:
                # rewards in ref to time
                for partner in self.r_after_episode.keys():
                    # plt.plot(self.r_after_episode[partner])
                    b, a = butter(2, 0.01, btype="low", analog=False)
                    y = filtfilt(b, a, self.r_after_episode[partner])
                    plt.plot(y, label=partner + postfix)
                    plt.legend(loc="best")
                    # plt.plot(np.convolve(self.r_after_episode[partner], np.ones(int(self.awi.n_steps/4))/int(self.awi.n_steps/4), mode='valid'))
                # plt.show()

                # plt.plot(self.r_after_episode[partner])
                # for partner in self.q_after_episode.keys():
                #     plt.plot(self.q_after_episode[partner])
                # plt.show()
                # for partner in self.profit_per_opp.keys():
                #     #plt.plot(self.profit_per_opp[partner])
                #     plt.plot(np.convolve(self.profit_per_opp[partner], np.ones(int(self.awi.n_steps/4))/int(self.awi.n_steps/4), mode='valid'))
                plt.show()

                # balance in ref to time
                plt.plot(self.balance_over_time)
                plt.show()

    """
    Called when negotiation succeeded
    :param contract - the contract that was accepted
    :param mechanism - an object from which we can know various negotiation info
    """

    def on_negotiation_success(self, contract, mechanism):
        self.num_of_nego_total += 1
        DEBUG_PRINT("^^^^^ start of on_negotiation_success " + str(id(self)))
        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        self.success += 1
        self.episodes += 1
        # self.profit += contract.agreement["unit_price"]*contract.agreement["quantity"]
        self.secured += contract.agreement["quantity"]

        # verifying we're not in an invalid framework call where there's 0 quantity.
        # if we detect, we return as this call has no meaning (and avoid division by 0 later)
        if self.is_seller:
            partner = contract.annotation["buyer"]
            if self.awi.current_exogenous_input_quantity <= 0:
                self.state_per_opp[partner] = STATE_TYPE.ACCEPT
                self.last_a_per_opp[partner] = None
                return
        else:
            partner = contract.annotation["seller"]
            if self.awi.current_exogenous_output_quantity <= 0:
                self.state_per_opp[partner] = STATE_TYPE.ACCEPT
                self.last_a_per_opp[partner] = None
                return
        if self.failure_list[partner]:
            self.failure_list[partner] = 0
        # print("SUCCESS to QL")
        # print("QL price : " + str(contract.agreement["unit_price"]))
        # print("QL quantity : " + str(contract.agreement["quantity"]))

        # Getting partner ID and calculating profit (or good estimation to it) we got from this negotation
        if self._is_selling(mechanism):
            partner = contract.annotation["buyer"]
            cur_profit = (
                contract.agreement["unit_price"]
                - float(self.awi.current_exogenous_input_price)
                / (self.awi.current_exogenous_input_quantity + 0.05)
                - self.awi.profile.cost
            ) * contract.agreement[
                "quantity"
            ]  # -self.awi.current_exogenous_input_price#*self.awi.current_exogenous_input_quantity
            self.online_balance = (
                self.online_balance
                - (
                    float(self.awi.current_exogenous_input_price)
                    / (self.awi.current_exogenous_input_quantity + 0.05)
                    + self.awi.profile.cost
                )
                * contract.agreement["quantity"]
            )
            fctr = contract.agreement["unit_price"]  # **2)
        else:
            partner = contract.annotation["seller"]
            cur_profit = (
                float(self.awi.current_exogenous_output_price)
                / (self.awi.current_exogenous_output_quantity + 0.05)
                - contract.agreement["unit_price"]
                - self.awi.profile.cost
            ) * contract.agreement[
                "quantity"
            ]  # *self.awi.current_exogenous_output_quantity
            self.online_balance = (
                self.online_balance
                - (contract.agreement["unit_price"] + self.awi.profile.cost)
                * contract.agreement["quantity"]
            )
            fctr = -(contract.agreement["unit_price"])  # **2)

        # cur_profit = cur_profit*(1.01**(self.num_of_nego_total))#*(1+0.5*self.num_of_nego_total) #cur_profit*(1+1.1**(self.num_of_nego_total-100))

        DEBUG_PRINT("on_negotiation_success, " + partner)

        ami = self.get_ami(partner)
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]
        # print("QL p range : " + str(unit_price_issue))
        # print("QL q range : " + str(quantity_issue))
        DEBUG_PRINT("state: " + str(self.state_per_opp[partner]))
        DEBUG_PRINT("action: " + str(self.last_a_per_opp[partner]))
        # for s in self.q_table_per_opp[partner].keys():
        #     for a in self.q_table_per_opp[partner][s].keys():
        #         if self.q_table_per_opp[partner][s][a] != 0:
        #             print(s, a, self.q_table_per_opp[partner][s][a])

        # for debug
        if partner not in self.profit_per_opp:
            self.profit_per_opp[partner] = [cur_profit]
        else:
            self.profit_per_opp[partner].append(cur_profit)

        # DEBUG_PRINT("on_negotiation_success " + partner)
        DEBUG_PRINT("--------------------------------")
        DEBUG_PRINT("SUCCESS <=>" + partner)

        DEBUG_PRINT("price: " + str(contract.agreement["unit_price"]))
        # if self._is_selling(mechanism):
        #     DEBUG_PRINT("exo price (sell): " + str(float(self.awi.current_exogenous_input_price)/self.awi.current_exogenous_input_quantity))
        # else:
        #     DEBUG_PRINT("exo price (buy): " + str(float(self.awi.current_exogenous_output_price)/self.awi.current_exogenous_output_quantity))
        DEBUG_PRINT("cost :" + str(self.awi.profile.cost))

        # getting current needs
        my_needs = self._needed()
        # print("QL needs: " + str(my_needs))
        self.q_steps[partner] += 1
        if partner not in self.num_of_negotiations_per_step:
            self.num_of_negotiations_per_step[partner] = 0

        # updating num of negotations, for scaling purposes
        self.num_of_negotiations_per_step[partner] += 1
        self.num_of_negotiations_per_step_all += 1
        # Setting the correct reward per case, and updateing Q-table
        if my_needs == 0:
            # No more needs - we sold/bought everything
            DEBUG_PRINT("success : no needs")

            # setting reward
            if cur_profit < 0:
                r = cur_profit  # + fctr#10*cur_profit# + fctr
                # r = r/(unit_price_issue.max_value*quantity_issue.max_value)
                # if self.counters[2] != 0:
                #     r = r*(self.counters[1]/self.counters[2])
                self.counters[1] += 1
            else:
                r = cur_profit  # + fctr#100*cur_profit# + fctr
                # r = r/(unit_price_issue.max_value*quantity_issue.max_value)
                # r = r+0.5
                self.counters[2] += 1

            # r += 0.2*unit_price_issue.max_value*quantity_issue.max_value

            # updating Q table

            self._q_learning_update_q(
                self.state_per_opp[partner],
                self.last_a_per_opp[partner],
                self.q_table_per_opp[partner],
                r,
                STATE_TYPE.ACCEPT,
                unit_price_issue,
                quantity_issue,
                partner,
            )
        elif my_needs < 0:
            # No more needs - we sold/bought more than needed
            DEBUG_PRINT("success : - needs")

            # setting reward. taking in account excessive needs
            if cur_profit < 0:
                r = (
                    cur_profit
                    + my_needs
                    * self._too_much_penalty(mechanism)
                    / self.num_of_negotiations_per_step_all
                )  # self.num_of_negotiations_per_step[partner]#*self.awi.current_step
                # print(my_needs, r)
                # r = r/(unit_price_issue.max_value*quantity_issue.max_value)
                # if self.counters[4] != 0:
                #     r = r*(self.counters[3]/self.counters[4])
                # r += 0.2*unit_price_issue.max_value*quantity_issue.max_value
                self.counters[3] += 1
            else:
                r = (
                    cur_profit
                    + my_needs
                    * self._too_much_penalty(mechanism)
                    / self.num_of_negotiations_per_step_all
                )  # self.num_of_negotiations_per_step[partner]#*self.awi.current_step
                # print(my_needs, r)
                # r = r/(unit_price_issue.max_value*quantity_issue.max_value)
                # r = r+0.5
                # r += 0.2*unit_price_issue.max_value*quantity_issue.max_value
                self.counters[4] += 1
            # updating Q tables
            self._q_learning_update_q(
                self.state_per_opp[partner],
                self.last_a_per_opp[partner],
                self.q_table_per_opp[partner],
                r,
                STATE_TYPE.ACCEPT,
                unit_price_issue,
                quantity_issue,
                partner,
            )
        else:  # my_needs > 0
            # Still there are needs
            DEBUG_PRINT("success : + needs")

            # setting reward. taking in account unused needs
            if cur_profit < 0:
                r = (
                    cur_profit
                    - my_needs
                    * self._too_less_penalty(mechanism)
                    / self.num_of_negotiations_per_step_all
                )  # self.num_of_negotiations_per_step[partner]#*self.awi.current_step
                # r = r/(unit_price_issue.max_value*quantity_issue.max_value)
                # if self.counters[6] != 0:
                #     r = r*(self.counters[5]/self.counters[6])
                self.counters[5] += 1
            else:
                r = (
                    cur_profit
                    - my_needs
                    * self._too_less_penalty(mechanism)
                    / self.num_of_negotiations_per_step_all
                )  # self.num_of_negotiations_per_step[partner]#*self.awi.current_step
                # r = r/(unit_price_issue.max_value*quantity_issue.max_value)
                # r = r+0.5
                self.counters[6] += 1
            # r += 0.2*unit_price_issue.max_value*quantity_issue.max_value

            # updating Q tables
            self._q_learning_update_q(
                self.state_per_opp[partner],
                self.last_a_per_opp[partner],
                self.q_table_per_opp[partner],
                r,
                STATE_TYPE.ACCEPT,
                unit_price_issue,
                quantity_issue,
                partner,
            )

        # print("r good : ", r)
        DEBUG_PRINT("state: " + str(self.state_per_opp[partner]))
        DEBUG_PRINT("action: " + str(self.last_a_per_opp[partner]))
        DEBUG_PRINT("reward: " + str(r))

        # for debug
        if partner not in self.r_after_episode:
            self.r_after_episode[partner] = [0]

        if partner not in self.q_after_episode:
            self.q_after_episode[partner] = [0]

        self.r_after_episode[partner][-1] = (
            self.r_after_episode[partner][-1] + r
        )  # /self.q_steps[partner]

        # setting current state correctlys
        self.state_per_opp[partner] = STATE_TYPE.ACCEPT
        self.last_a_per_opp[partner] = None

        # update my current best price to use for limiting concession in other
        # negotiations
        up = contract.agreement["unit_price"]
        if contract.annotation["product"] == self.awi.my_output_product:
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

        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        DEBUG_PRINT("^^^^^ end of on_negotiation_success " + str(id(self)))

    """
    Called when negotiation fails
    :param partners - unused
    :param annotation - an object from which we can know various info like partner ID
    :param mechanism - an object from which we can know various negotiation info
    :param state - an framework object of state (unrelated to MDP state)
    """

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        DEBUG_PRINT("^^^^^ start of on_negotiation_failure " + str(id(self)))
        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))

        # Getting partner ID and calculating profit (or good estimation to it) we got from this negotation
        if self._is_selling(mechanism):
            is_seller = True
            partner = annotation["buyer"]
        else:
            is_seller = False
            partner = annotation["seller"]

        DEBUG_PRINT("on_negotiation_failure, " + partner)

        ami = self.get_ami(partner)
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]
        if self.failure_list[partner]:
            self.failure_list[partner] += 1
        else:
            self.failure_list[partner] = 1
        # Various filters to avoid errorneus cases
        if partner not in self.state_per_opp:
            return None

        if self.state_per_opp[partner] == STATE_TYPE.NO_NEGO:
            return

        if self.last_a_per_opp is None:
            return

        if partner not in self.last_a_per_opp:
            return

        self.num_of_nego_total += 1

        self.episodes += 1
        self.failure += 1

        # DEBUG_PRINT("on_negotiation_failure " + partner)
        DEBUG_PRINT("--------------------------------")
        DEBUG_PRINT("FAILURE <=>" + partner)

        my_needs = self._needed()
        self.q_steps[partner] += 1
        # if partner not in self.num_of_negotiations_per_step:
        #     self.num_of_negotiations_per_step[partner] = 0
        # self.num_of_negotiations_per_step[partner] += 1

        # Setting the correct reward per case, and updateing Q-table
        if my_needs == 0:
            # No more needs - we sold/bought everything

            # setting reward. this is good - we didn't buy/sell when we didn't need to
            r = 0.5 * unit_price_issue.max_value * quantity_issue.max_value  # 0.7

            self.counters[7] += 1
            # update Q-table
            self._q_learning_update_q(
                self.state_per_opp[partner],
                self.last_a_per_opp[partner],
                self.q_table_per_opp[partner],
                r,
                STATE_TYPE.END,
                unit_price_issue,
                quantity_issue,
                partner,
            )
        elif my_needs < 0:
            # No more needs - we sold/bought more than needed
            DEBUG_PRINT("need -")

            # setting reward
            if isinstance(self.last_a_per_opp[partner], str):
                if self.last_a_per_opp[partner] == "end":
                    # this is good - we decided not to buy/sell when we didn't need to
                    r = (
                        0.5 * unit_price_issue.max_value * quantity_issue.max_value
                    )  # 0.7 #1*100
                    self.counters[8] += 1
                else:
                    # shouldn't happen - this means "acc" but we weren't supposed to get here
                    r = 0  # 0.5*unit_price_issue.max_value*quantity_issue.max_value
                    self.counters[9] += 1
                    pass  # print("not supposed to happen")
            else:
                # we didn't buy/sell eventually - good, but didn't decide on ourselves - bad. So in general, neutral.
                r = (
                    -5
                )  # -0.5*unit_price_issue.max_value*quantity_issue.max_value #-1*100

            # r = 1*10#
            # r -= my_needs*self._too_much_penalty(mechanism)/self.num_of_negotiations_per_step_all#self.num_of_negotiations_per_step[partner]#*self.awi.current_step

            # update Q-table
            self._q_learning_update_q(
                self.state_per_opp[partner],
                self.last_a_per_opp[partner],
                self.q_table_per_opp[partner],
                r,
                STATE_TYPE.END,
                unit_price_issue,
                quantity_issue,
                partner,
            )
        else:  # my_needs > 0
            # There are still needs to sell/buy
            DEBUG_PRINT("needs +")

            # setting reward. we had needs to trade - so need to evaluate the utility of the case
            # carefully: taking in account how far we were from what we wanted. If for example
            # opponent wanted to give us something a lot worse than we wanted, this may actually
            # be positive that it didn't work out - but to some limit (depending how much we had to trade.
            # if we had a lot - this makes it bad not to trade at all, even if price is not so good for us)
            if isinstance(self.state_per_opp[partner], tuple):
                if len(self.state_per_opp[partner]) == 5:  # 4:
                    self.counters[10] += 1
                    if self.last_a_per_opp[partner] == "end":
                        if partner in self.fail_counters:
                            self.fail_counters[partner] += 1
                        else:
                            self.fail_counters[partner] = 1
                    if is_seller:
                        r = (
                            self.state_per_opp[partner][0]
                            - self.state_per_opp[partner][2]
                        )
                        # -1000*my_needs*self._too_less_penalty(mechanism)/self.num_of_negotiations_per_step[partner]#*self.awi.current_step
                    else:
                        r = (
                            self.state_per_opp[partner][2]
                            - self.state_per_opp[partner][0]
                        )

                elif len(self.state_per_opp[partner]) == 3:  # 2:
                    self.counters[11] += 1
                    if is_seller:
                        r = unit_price_issue.max_value - self.state_per_opp[partner][0]
                    else:
                        r = self.state_per_opp[partner][0] - unit_price_issue.min_value
                else:
                    pass  # print("not supposed to happen")
                # r = r/self.awi.current_step#*10
                DEBUG_PRINT("case 1")
                r = r / (unit_price_issue.max_value)

                # taking left needs into account + balancing reward according to emprical testing
                if r > 0:
                    r = (
                        0.1
                        * unit_price_issue.max_value
                        * quantity_issue.max_value
                        * r
                        * (
                            1
                            - (2.0 * my_needs)
                            / (quantity_issue.min_value + quantity_issue.max_value)
                        )
                    )
                elif r < 0:
                    r = 0.5 * r * unit_price_issue.max_value * quantity_issue.max_value
                else:
                    # if we're price neutral, and needs > 0, it's not good for us - we needed to trade.
                    # Thus negative reward
                    r = -0.5 * unit_price_issue.max_value * quantity_issue.max_value

                # print(r)
                # print(self.state_per_opp[partner])
                # print(self.last_a_per_opp[partner])
                # print("Last A:", self.last_a_per_opp[partner])

                # if r < 0:
                #     r = 0.1*self.counters[8]*r
                # else:
                # if self.counters[2] == 0 and self.counters[4] == 0 and self.counters[6] == 0:
                #     if r > 0:
                #         r = r*(1-0.05*self.counters[8])
                #     else:
                #         r = r*(1+0.05*self.counters[8])
                # else:
                #     if r > 0:
                #         r = r*(1-0.1*self.counters[8]/(self.counters[2] + self.counters[4] + self.counters[6]))
                #     else:
                #         r = r*(1+0.1*self.counters[8]/(self.counters[2] + self.counters[4] + self.counters[6]))

                # else:
                # r -= my_needs*self._too_less_penalty(mechanism)/self.num_of_negotiations_per_step[partner]
            else:
                self.counters[12] += 1
                DEBUG_PRINT("case 2")
                # setting reward - not intermediate state, not much interaction - so no smart things to say.
                # thus giving 0.
                r = 0
                # r = -my_needs*self._too_less_penalty(mechanism)/self.num_of_negotiations_per_step[partner]

            # r -= 0.5*unit_price_issue.max_value*quantity_issue.max_value

            # print("r bad : ", r)

            DEBUG_PRINT("state: " + str(self.state_per_opp[partner]))
            DEBUG_PRINT("action: " + str(self.last_a_per_opp[partner]))
            DEBUG_PRINT("reward: " + str(r))
            # for k in self.q_table_per_opp[partner].keys():
            #     print(self.q_table_per_opp[partner][k])

            # updating Q-table
            self._q_learning_update_q(
                self.state_per_opp[partner],
                self.last_a_per_opp[partner],
                self.q_table_per_opp[partner],
                r,
                STATE_TYPE.END,
                unit_price_issue,
                quantity_issue,
                partner,
            )

        # Setting the correct state
        self.state_per_opp[partner] = STATE_TYPE.END
        # print("reward: ", r)

        # for debug
        if partner not in self.profit_per_opp:
            self.profit_per_opp[partner] = [0]
        else:
            self.profit_per_opp[partner].append(0)

        if partner not in self.r_after_episode:
            self.r_after_episode[partner] = [0]

        self.r_after_episode[partner][-1] = (
            self.r_after_episode[partner][-1] + r
        )  # /self.q_steps[partner]

        # zering action
        self.last_a_per_opp[partner] = None

        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        DEBUG_PRINT("^^^^^ end of on_negotiation_failure " + str(id(self)))

    """
    Called in order so our agent will propose to an opposing one
    :param: negotiator_id - the ID of the opposing negotiator
    :param: state - unused (not related to Q-learning)
    :return: the offer to be proposed
    """

    def propose(self, negotiator_id: str, state) -> "Outcome":
        # print("# STEP : " + str(self.awi.current_step))
        DEBUG_PRINT("^^^^^ start of propose " + negotiator_id + " " + str(id(self)))
        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        # print("propose " + negotiator_id)

        # Getting our current needs
        my_needs = self._needed(negotiator_id)
        # DEBUG_PRINT("needs : " + str(my_needs))
        ami = self.get_ami(negotiator_id)
        if not ami:
            pass  # print("No AMI !")
            return None

        # Getting opposing partner ID
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            is_seller = True
        else:
            partner = ami.annotation["seller"]
            is_seller = False

        self.is_seller = is_seller
        # if partner in self.finished:
        #     return None
        DEBUG_PRINT("propose " + partner)
        DEBUG_PRINT("------------------------------")
        # self._init_opposites_if_needed(ami)
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]

        # initializing offer structure
        offer = [-1] * 3

        # checking if we already encountered this partner
        if partner not in self.started.keys():
            # first proposal with this negotiator. this means first proposal in the simulation
            # against it.
            DEBUG_PRINT("INIT, " + partner)
            self.started[partner] = True
            self.q_table_per_opp[partner] = dict()
            # print(self.q_table_per_opp)
            price_gap = unit_price_issue.max_value - unit_price_issue.min_value
            quantity_gap = quantity_issue.max_value - quantity_issue.min_value

            # initializing an action vector with all action and a Q table against this partner.
            self.action_vec[partner] = dict()  # []
            # self.q_table_per_opp[partner], self.action_vec[partner] = self._q_learning_q_init(self.q_table_per_opp[partner], unit_price_issue, quantity_issue, self.action_vec[partner], is_seller, ami)
            self._q_learning_q_init(
                self.q_table_per_opp[partner],
                unit_price_issue,
                quantity_issue,
                self.action_vec[partner],
                is_seller,
                ami,
            )
            # print('iv ', self.action_vec[partner])

            # print("a", partner, len(self.q_table_per_opp[partner]))

            # if no needs, this negotiation is skipped
            if my_needs <= 0:
                self.state_per_opp[partner] = STATE_TYPE.NO_NEGO
                DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
                DEBUG_PRINT(
                    "state_per_opp: " + str(self.state_per_opp) + " " + str(id(self))
                )
                DEBUG_PRINT(
                    "^^^^^ end of propose 1" + negotiator_id + " " + str(id(self))
                )
                return None

            if self.is_seller:
                if self.awi.current_exogenous_input_quantity <= 0:
                    self.state_per_opp[partner] = STATE_TYPE.NO_NEGO
                    DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
                    DEBUG_PRINT(
                        "state_per_opp: "
                        + str(self.state_per_opp)
                        + " "
                        + str(id(self))
                    )
                    DEBUG_PRINT(
                        "^^^^^ end of propose 1" + negotiator_id + " " + str(id(self))
                    )
                    return None
            else:
                if self.awi.current_exogenous_output_quantity <= 0:
                    self.state_per_opp[partner] = STATE_TYPE.NO_NEGO
                    DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
                    DEBUG_PRINT(
                        "state_per_opp: "
                        + str(self.state_per_opp)
                        + " "
                        + str(id(self))
                    )
                    DEBUG_PRINT(
                        "^^^^^ end of propose 1" + negotiator_id + " " + str(id(self))
                    )
                    return None

            # initializing the state
            if not self.complex_state:
                self.state_per_opp[partner] = STATE_TYPE.INIT
            else:
                self.state_per_opp[partner] = str(
                    self._find_nearest(self.ne_range, my_needs)
                )  # str(int(np.ceil(my_needs)))

            self.q_steps[partner] = 1

        else:
            # print("b", partner, len(self.q_table_per_opp[partner]))
            # not first proposal with this negotiator
            if partner not in self.state_per_opp:
                # this is not supposed to happen - there should be coordination between the different
                # data structs in use against a partner
                print(
                    "issue, "
                    + str(self.state_per_opp.keys())
                    + ", "
                    + str(self.started)
                )
                return None
            if (
                self.state_per_opp[partner] == STATE_TYPE.END
                or self.state_per_opp[partner] == STATE_TYPE.ACCEPT
                or self.state_per_opp[partner] == STATE_TYPE.NO_NEGO
            ):
                # This means that a negotiation against this partner just finished, so starting from beginning
                # DEBUG_PRINT("INIT")

                # if no needs, this negotiation is skipped
                if my_needs <= 0:
                    self.state_per_opp[partner] = STATE_TYPE.NO_NEGO
                    DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
                    DEBUG_PRINT(
                        "state_per_opp: "
                        + str(self.state_per_opp)
                        + " "
                        + str(id(self))
                    )
                    DEBUG_PRINT(
                        "^^^^^ end of propose 2" + negotiator_id + " " + str(id(self))
                    )
                    return None

                # last state was terminal, thus we now start new q learning step
                if not self.complex_state:
                    self.state_per_opp[partner] = STATE_TYPE.INIT
                else:
                    self.state_per_opp[partner] = str(
                        self._find_nearest(self.ne_range, my_needs)
                    )  # str(int(np.ceil(my_needs)))
                # price_gap = unit_price_issue.max_value-unit_price_issue.min_value
                # quantity_gap = quantity_issue.max_value-quantity_issue.min_value
                # self._q_learning_q_init(self.q_table_per_opp[partner], price_gap, quantity_gap)

                self.q_steps[partner] = 1

            # if we get to the "elif", this means this is during negotiation.
            elif self.state_per_opp[partner] == STATE_TYPE.OPP_COUNTER or isinstance(
                self.state_per_opp[partner], tuple
            ):
                # We are in the middle of negotiation (got here to offer something in return to partner's offer - we give a counter offer)

                # DEBUG_PRINT("OPP_COUNTER")
                # print("propose: " + str(self.state_per_opp[partner]))
                # after the respond
                # update q
                # self._q_learning_update_q(STATE_TYPE.OPP_COUNTER, self.last_a_per_opp[partner], self.q_table_per_opp[partner], 0, self.state_per_opp[partner])
                # self.state_per_opp[partner] = STATE_TYPE.MY_COUNTER
                self.q_steps[partner] += 1
            else:
                # This case is in order to overcome issues in the infra, like propose after propose for same partner.
                # We just start from scratch.

                # print(self.state_per_opp[partner])
                # print("error - invalid state")
                # sys.exit(1)
                # DEBUG_PRINT("INIT2")
                self.started[partner] = True
                # self.q_table_per_opp[partner] = dict()
                # print(self.q_table_per_opp)
                price_gap = unit_price_issue.max_value - unit_price_issue.min_value
                quantity_gap = quantity_issue.max_value - quantity_issue.min_value
                # self.action_vec[partner] = []
                # self._q_learning_q_init(self.q_table_per_opp[partner], unit_price_issue, quantity_issue, self.action_vec[partner])

                if not self.complex_state:
                    self.state_per_opp[partner] = STATE_TYPE.INIT
                else:
                    self.state_per_opp[partner] = str(
                        self._find_nearest(self.ne_range, my_needs)
                    )  # str(int(np.ceil(my_needs)))
                self.q_steps[partner] = 1

        # print("here6")
        # print(self.q_table_per_opp)
        # print(self.state_per_opp[partner])

        # setting action correctly (might been decided before)
        if not self.complex_state:
            if self.state_per_opp[partner] == STATE_TYPE.INIT:
                a = self._q_learning_select_action(
                    self.q_table_per_opp[partner],
                    self.state_per_opp[partner],
                    self.action_vec[partner],
                    caller="propose",
                    negotiator_id=negotiator_id,
                    state_sys=state,
                    offer=None,
                    unit_price_issue=unit_price_issue,
                    quantity_issue=quantity_issue,
                    partner=partner,
                )
                self.last_a_per_opp[partner] = a
            else:
                # coming from OPP_COUNTER state, I already selected my action
                a = self.last_a_per_opp[partner]
                if a == "acc" or a == "end":
                    pass  # print("should not happen")
                    return None
        else:
            if isinstance(self.state_per_opp[partner], str):
                a = self._q_learning_select_action(
                    self.q_table_per_opp[partner],
                    self.state_per_opp[partner],
                    self.action_vec[partner],
                    caller="propose",
                    negotiator_id=negotiator_id,
                    state_sys=state,
                    offer=None,
                    unit_price_issue=unit_price_issue,
                    quantity_issue=quantity_issue,
                    partner=partner,
                )
                self.last_a_per_opp[partner] = a
            else:
                # coming from OPP_COUNTER state, I already selected my action
                a = self.last_a_per_opp[partner]
                if a == "acc" or a == "end":
                    pass  # print("should not happen")
                    return None

        # print("here7")
        DEBUG_PRINT("-------------------------")
        DEBUG_PRINT("PROPOSE -->" + partner)
        DEBUG_PRINT(ami.annotation["buyer"] + " " + ami.annotation["seller"])
        DEBUG_PRINT("executed action : " + str(a))
        DEBUG_PRINT("state : " + str(self.state_per_opp[partner]))

        # Building the offer
        offer[OFFER_FIELD_IDX.UNIT_PRICE] = a[0]
        offer[OFFER_FIELD_IDX.QUANTITY] = np.ceil(a[1])
        offer[OFFER_FIELD_IDX.TIME] = self.awi.current_step

        # print("-----------------------------------------------")
        # print("# STEP : " + str(self.awi.current_step))
        # print("# Needed to produce : " + str(self._needed()))
        # print("propose to " + str(negotiator_id) + ":" + str(self.best_offer(negotiator_id)))
        # print(partner)
        # print(self.q_table_per_opp[partner])
        # print("proposing !")

        # opportunistic behavior: in some small probabilty, propose something which is very good for us
        # (used to exploit opponent's weakness). This overrides the learning process and employed heuristicall
        # since in the real tournament we won't have a lot of time to just "learn"
        p_op = np.random.random()

        if p_op < self.opportunistic_level * 50:  # or self.awi.current_step == 0:
            if self.is_seller:
                if p_op < self.opportunistic_level * 5:
                    # in some very small probability, make this offer even "unrealistic" but
                    # highly profitable if it happens
                    offer[OFFER_FIELD_IDX.UNIT_PRICE] = 10 * unit_price_issue.max_value
                else:
                    offer[OFFER_FIELD_IDX.UNIT_PRICE] = unit_price_issue.max_value
            else:
                offer[OFFER_FIELD_IDX.UNIT_PRICE] = unit_price_issue.min_value
            if self._needed() > self.awi.n_lines / 2:
                offer[OFFER_FIELD_IDX.QUANTITY] = self.awi.n_lines / 2
            else:
                offer[OFFER_FIELD_IDX.QUANTITY] = self._needed()
            self.last_a_per_opp[partner] = (
                offer[OFFER_FIELD_IDX.UNIT_PRICE],
                offer[OFFER_FIELD_IDX.QUANTITY],
            )

        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        DEBUG_PRINT("^^^^^ end of propose 3" + negotiator_id + " " + str(id(self)))
        return tuple(offer)

    """
    Called in order so our agent will respond to an opposing one
    :param: negotiator_id - the ID of the opposing negotiator
    :param: state - unused (not related to Q-learning)
    :param: offer - the offer given by the opposing partner
    :return: the response for the opponent partner (accept/reject(and offer something else in next propose())/end)
    """

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        # DEBUG_PRINT("respond " + negotiator_id)
        DEBUG_PRINT("^^^^^ start of respond " + negotiator_id + " " + str(id(self)))
        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        # print("-----------------------------------------------")
        # print("# STEP : " + str(self.awi.current_step))
        # print("# Needed to produce : " + str(self._needed()))

        # Getting current left needs
        my_needs = self._needed(negotiator_id)

        ami = self.get_ami(negotiator_id)
        if not ami:
            DEBUG_PRINT("No AMI !")
            return None

        # Getting the opposite partner ID
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            self.is_seller = True
        else:
            partner = ami.annotation["seller"]
            self.is_seller = False

        # Filtering for debug / catch issues / avoid infra bugs
        if partner in self.last_a_per_opp:
            if (
                self.last_a_per_opp[partner] == "acc"
                or self.last_a_per_opp[partner] == "end"
            ):
                # print("should not happen - we should have junped to success or failure methods")
                return None

        # Calculation potential profit - current this value won't be used
        if self._is_selling(ami):
            # if self.awi.current_exogenous_input_price <= 0:
            #     #print(self.awi.current_exogenous_input_price)
            #     potential_profit = -1
            # else:
            #     potential_profit = (offer[OFFER_FIELD_IDX.UNIT_PRICE]-float(self.awi.current_exogenous_input_price)/self.awi.current_exogenous_input_quantity-self.awi.profile.cost)*offer[OFFER_FIELD_IDX.QUANTITY]
            is_seller = True
        else:
            # if self.awi.current_exogenous_output_quantity <= 0:
            #     #print(self.awi.current_exogenous_output_quantity)
            #     potential_profit = -1
            # else:
            #     potential_profit = (float(self.awi.current_exogenous_output_price)/self.awi.current_exogenous_output_quantity-offer[OFFER_FIELD_IDX.UNIT_PRICE]-self.awi.profile.cost)*offer[OFFER_FIELD_IDX.QUANTITY]
            is_seller = False

        if self.is_seller:
            if self.awi.current_exogenous_input_quantity <= 0:
                self.state_per_opp[partner] = STATE_TYPE.NO_NEGO
                return None
        else:
            if self.awi.current_exogenous_output_quantity <= 0:
                self.state_per_opp[partner] = STATE_TYPE.NO_NEGO
                return None

        # Check if this the first encounter with this partner
        if partner not in self.state_per_opp:
            # it is the first encounter. Init everything - state, action vec, Q table
            # return None
            self.q_steps[partner] = 0
            self.started[partner] = True
            if not self.complex_state:
                self.state_per_opp[partner] = STATE_TYPE.INIT
            else:
                self.state_per_opp[partner] = str(
                    self._find_nearest(self.ne_range, my_needs)
                )  # str(int(np.ceil(my_needs)))
            self.q_table_per_opp[partner] = dict()
            unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
            quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]
            price_gap = unit_price_issue.max_value - unit_price_issue.min_value
            quantity_gap = quantity_issue.max_value - quantity_issue.min_value
            self.action_vec[partner] = dict()  # []
            # self.q_table_per_opp[partner], self.action_vec[partner] = self._q_learning_q_init(self.q_table_per_opp[partner], unit_price_issue, quantity_issue, self.action_vec[partner], is_seller, ami)
            self._q_learning_q_init(
                self.q_table_per_opp[partner],
                unit_price_issue,
                quantity_issue,
                self.action_vec[partner],
                is_seller,
                ami,
            )
            # print('av ', self.action_vec[partner])

        # print("respond: " + str(self.state_per_opp[partner]))

        # Filtering unneeded calls to respond() in case nothing to do
        if partner in self.state_per_opp:
            if self.state_per_opp[partner] == STATE_TYPE.NO_NEGO:
                return ResponseType.REJECT_OFFER

        # print("responding !")
        DEBUG_PRINT("-------------------------")
        DEBUG_PRINT("RESPOND <--" + partner)
        if partner in self.last_a_per_opp:
            DEBUG_PRINT("my last offer : " + str(self.last_a_per_opp[partner]))
        DEBUG_PRINT(
            "new offer : "
            + str([offer[OFFER_FIELD_IDX.UNIT_PRICE], offer[OFFER_FIELD_IDX.QUANTITY]])
        )
        DEBUG_PRINT("last state : " + str(self.state_per_opp[partner]))

        if not self.complex_state:
            new_state = STATE_TYPE.OPP_COUNTER
            unit_price_issue = None
            quantity_issue = None
        else:
            # Calculating new state (mapping from env status to best matching state in Q table)
            # Setting the new state if needed
            unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
            quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]
            price_gap = unit_price_issue.max_value - unit_price_issue.min_value
            quantity_gap = quantity_issue.max_value - quantity_issue.min_value
            # print(self.last_a_per_opp)
            if (
                self.last_a_per_opp == {}
                or self.last_a_per_opp is None
                or partner not in self.last_a_per_opp
                or self.last_a_per_opp[partner] is None
            ):
                # This case is when we init here, first encounter or starting new nogotiation.

                new_state = self._find_state_mapping(
                    0,
                    0,
                    offer[OFFER_FIELD_IDX.UNIT_PRICE],
                    offer[OFFER_FIELD_IDX.QUANTITY],
                    unit_price_issue,
                    quantity_issue,
                    my_needs,
                )[2:]
                if not self.complex_state:
                    self.state_per_opp[partner] = STATE_TYPE.INIT
                else:
                    self.state_per_opp[partner] = str(
                        self._find_nearest(self.ne_range, my_needs)
                    )  # str(int(np.ceil(my_needs)))
            else:
                # We're in the middle of existing negotiation
                new_state = self._find_state_mapping(
                    self.last_a_per_opp[partner][0],
                    self.last_a_per_opp[partner][1],
                    offer[OFFER_FIELD_IDX.UNIT_PRICE],
                    offer[OFFER_FIELD_IDX.QUANTITY],
                    unit_price_issue,
                    quantity_issue,
                    my_needs,
                )
                # print(self.last_a_per_opp[partner])
                # update q
                # print("update 1")
                # Since we're in the middle of a negotiation process (Q learning episode) - update Q-table
                self._q_learning_update_q(
                    self.state_per_opp[partner],
                    self.last_a_per_opp[partner],
                    self.q_table_per_opp[partner],
                    0,
                    new_state,
                    unit_price_issue,
                    quantity_issue,
                    partner,
                )

        self.q_steps[partner] += 1

        # progress state
        self.state_per_opp[partner] = new_state

        # choose new action
        a = self._q_learning_select_action(
            self.q_table_per_opp[partner],
            new_state,
            self.action_vec[partner],
            caller="respond",
            negotiator_id=negotiator_id,
            state_sys=state,
            offer=offer,
            unit_price_issue=unit_price_issue,
            quantity_issue=quantity_issue,
            partner=partner,
        )

        # DEBUG_PRINT("new state : " + str(self.state_per_opp[partner]))
        # DEBUG_PRINT("next action : " + str(a))
        # in case no more needs - end
        # if my_needs <= 0:
        #     a = "end"

        up = offer[OFFER_FIELD_IDX.UNIT_PRICE]
        if self._is_selling(ami):
            self._best_selling = max(up, self._best_selling)
            partner = ami.annotation["buyer"]
            self._best_opp_selling[partner] = self._best_selling
        else:
            self._best_buying = min(up, self._best_buying)
            partner = ami.annotation["seller"]
            self._best_opp_buying[partner] = self._best_buying

        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        DEBUG_PRINT("^^^^^ end of respond " + negotiator_id + " " + str(id(self)))

        # remember action for next method calls to update the Q-learning DBs and execute needed actions
        # if needed there. Return the correct negotiation responce as needed by the framework, according to
        # the action needed to be executed.
        self.last_a_per_opp[partner] = a
        if isinstance(a, str):
            # state will be updated in on_failure and in on_success
            if a == "end":
                return ResponseType.END_NEGOTIATION
            elif a == "acc":
                return ResponseType.ACCEPT_OFFER

        else:
            # this means we're going to propose()
            # print("rejecting !")
            return ResponseType.REJECT_OFFER

    """
    Q-Learning update - a method to update the Q table according to algorithm. Internally maps the current MDP state,
    given in terms of current run properties (prices, etc.) to index space to access Q table.
    :param: state - QL. state
    :param: action - QL. action
    :param: q - Q table
    :param: reward - the reward obtained
    :param: new_state - the new state
    :param: unit_price_issue - the unit price issue
    :param: quantity_issue - the quantity issue
    :param: partner - the partner (opponent) the learning process refers to
    """

    def _q_learning_update_q(
        self,
        state,
        action,
        q,
        reward,
        new_state,
        unit_price_issue=None,
        quantity_issue=None,
        partner=None,
    ):
        # print(id(self))
        # print(state)
        # print(action)
        # print(new_state)
        # print(reward)
        # print(len(q.keys()))
        # print("--------")
        # q[state][action] = q[state][action] + self.alpha*(reward + self.gamma*max([q[new_state][a] for a in q[new_state].keys()]) - q[state][action])
        # print(self.action_vec[partner])
        # print(action)

        # print("state: ", state)
        # mapping current MDP state to generic "index space" states
        state_m = self._state_mapper(state, unit_price_issue, quantity_issue)
        # print("state_m: ", state_m)
        # print(action)
        action_m = None
        for a in self.action_vec[partner].keys():
            if action == self.action_vec[partner][a]:
                action_m = a
                break

        # mapping current MDP new state to generic "index space" states
        new_state_m = self._state_mapper(new_state, unit_price_issue, quantity_issue)

        # if state_m in q:
        #     if action_m in q[state_m]:
        #         if new_state_m in q:
        #             if q[new_state_m]:
        # print(state_m)
        # print(q[state_m].keys())
        # print(action_m)

        # learning rate impacted by decay
        alpha_d = self.alpha * (
            1.0 / (1 + self.alpha_decay_rate * self.awi.current_step)
        )

        # avoiding trying to update table on non-existant action (which can happen due to the
        # Q-learning constraining in action selection, as well as opportunistic mechanism)
        if action_m is None:
            return
        # updating Q table
        q[state_m][action_m] = (1 - alpha_d) * q[state_m][action_m] + alpha_d * (
            reward + self.gamma * max(q[new_state_m][a] for a in q[new_state_m].keys())
        )

        # q[state_m][action_m] = (1-self.alpha)*q[state_m][action_m] + self.alpha*(reward + self.gamma*max([q[new_state_m][a] for a in q[new_state_m].keys()]))
        #             else:
        #                 q[new_state_m][action_m] = 0
        #                 q[state_m][action_m] = self.alpha*(reward + self.gamma*max([q[new_state_m][a] for a in q[new_state_m].keys()]))
        #         else:
        #             q[new_state_m] = dict()
        #             q[new_state_m][action_m] = 0
        #             q[state_m][action_m] = self.alpha*(reward + self.gamma*max([q[new_state_m][a] for a in q[new_state_m].keys()]))
        #     else:
        #         q[state_m] = dict()
        #         q[state_m][action_m] = 0
        #         if new_state_m in q:
        #             if q[new_state_m]:
        #                 q[state_m][action_m] = q[state_m][action_m] + self.alpha*(reward + self.gamma*max([q[new_state_m][a] for a in q[new_state_m].keys()]) - q[state_m][action_m])
        #             else:
        #                 q[new_state_m][action_m] = 0
        #                 q[state_m][action_m] = self.alpha*(reward + self.gamma*max([q[new_state_m][a] for a in q[new_state_m].keys()]))
        #         else:
        #             q[new_state_m] = dict()
        #             q[new_state_m][action_m] = 0
        #             q[state_m][action_m] = self.alpha*(reward + self.gamma*max([q[new_state_m][a] for a in q[new_state_m].keys()]))
        # else:
        #     q[state_m] = dict()
        #     q[state_m][action_m] = 0
        #     if action_m in q[state_m]:
        #         if new_state_m in q:
        #             if q[new_state_m]:
        #                 q[state_m][action_m] = q[state_m][action_m] + self.alpha*(reward + self.gamma*max([q[new_state_m][a] for a in q[new_state_m].keys()]) - q[state_m][action_m])
        #             else:
        #                 q[new_state_m][action_m] = 0
        #                 q[state_m][action_m] = self.alpha*(reward + self.gamma*max([q[new_state_m][a] for a in q[new_state_m].keys()]))
        #         else:
        #             q[new_state_m] = dict()
        #             q[new_state_m][action_m] = 0
        #             q[state_m][action_m] = self.alpha*(reward + self.gamma*max([q[new_state_m][a] for a in q[new_state_m].keys()]))
        #     else:
        #         q[state_m] = dict()
        #         q[state_m][action_m] = 0
        #         if new_state_m in q:
        #             if q[new_state_m]:
        #                 q[state_m][action_m] = q[state_m][action_m] + self.alpha*(reward + self.gamma*max([q[new_state_m][a] for a in q[new_state_m].keys()]) - q[state_m][action_m])
        #             else:
        #                 q[new_state_m][action_m] = 0
        #                 q[state_m][action_m] = self.alpha*(reward + self.gamma*max([q[new_state_m][a] for a in q[new_state_m].keys()]))
        #         else:
        #             q[new_state_m] = dict()
        #             q[new_state_m][action_m] = 0
        #             q[state_m][action_m] = self.alpha*(reward + self.gamma*max([q[new_state_m][a] for a in q[new_state_m].keys()]))

    def sigmoid(self, x):
        return math.exp(x / 30)

    """
    Q-Learning action selection - a method to select the next action according to algorithm
    :param: state - QL. state
    :param: q - Q table
    :param: action_vec - the vector with all possible actions
    :param: caller - who called - propose / respond
    :param: negotiator_id - refers to partner
    :param: state_sys - framework state (unrelated to Q-learning) - not really used now.
    :param: offer - offer that was received - not really used now.
    :param: unit_price_issue - the unit price issue
    :param: quantity_issue - the quantity issue
    :return: the selected action
    """

    def _q_learning_select_action(
        self,
        q,
        state,
        action_vec,
        caller=None,
        negotiator_id=None,
        state_sys=None,
        offer=None,
        unit_price_issue=None,
        quantity_issue=None,
        partner=None,
    ):
        # DEBUG_PRINT("_q_learning_select_actionq.keys(state)

        # epsilon-greedy action selection with decay - for imitation learning. not really used now.
        eps2_threshold = self.epsilon2_end + (
            self.epsilon2_start - self.epsilon2_end
        ) * math.exp(-1.0 * (self.awi.current_step + 1) / self.epsilon2_decay)

        # self.awi.current_step
        # num_of_nego_total
        # random sampling
        p2 = np.random.random()

        if self.failure_threshold[partner]:
            if self.failure_list[partner] >= self.failure_threshold[partner]:
                if self.adviced_list[partner]:
                    self.adviced_list[partner] += 1
                else:
                    self.adviced_list[partner] = 1
                # self.failure_threshold[partner] *= self.sigmoid(self.adviced_list[partner])
                # action selection with decay - for imitation learning (imitating LearningAgent). not really used now.
                # SO THIS CODE IS NOT REALLY RUNNING. ONLY THE "ELSE"
                # print(self.failure_threshold[partner])
                if caller == "propose":
                    return self.propose_greedy(negotiator_id, state_sys)
                if caller == "respond":
                    return self.respond_greedy(negotiator_id, state_sys, offer)
        else:
            self.failure_threshold[partner] = 40
            if self.adviced_list[partner]:
                self.adviced_list[partner] += 1
            else:
                self.adviced_list[partner] = 1
            # self.failure_threshold[partner] *= self.sigmoid(self.adviced_list[partner])
            if self.failure_list[partner] >= self.failure_threshold[partner]:
                # action selection with decay - for imitation learning (imitating LearningAgent). not really used now.
                # SO THIS CODE IS NOT REALLY RUNNING. ONLY THE "ELSE"
                # print(self.failure_threshold[partner])
                if caller == "propose":
                    return self.propose_greedy(negotiator_id, state_sys)
                if caller == "respond":
                    return self.respond_greedy(negotiator_id, state_sys, offer)

        if p2 < eps2_threshold:
            # action selection with decay - for imitation learning (imitating LearningAgent). not really used now.
            # SO THIS CODE IS NOT REALLY RUNNING. ONLY THE "ELSE"
            if caller == "propose":
                return self.propose_greedy(negotiator_id, state_sys)
            if caller == "respond":
                return self.respond_greedy(negotiator_id, state_sys, offer)
        else:
            # THIS IS ALWAYS RUNNING NOW (as imitation learning is not used)
            # the Q-learning original epsilon-greedy action selection with decay
            eps_threshold = self.epsilon_end + (
                self.epsilon_start - self.epsilon_end
            ) * math.exp(-1.0 * (self.awi.current_step + 1) / self.epsilon_decay)
            # self.awi.current_step

            # random sampling
            p = np.random.random()

            # act
            if p < eps_threshold:
                # select randomly
                if not self.complex_state:
                    if state != STATE_TYPE.INIT:
                        # print(np.array(action_vec, dtype=object))
                        a = np.random.choice(
                            np.array(list(action_vec.values()), dtype=object)
                        )
                        # print(a)
                        return a
                    else:
                        # print(np.array(action_vec, dtype=object))
                        # without "acc" and "end" since not available in INIT
                        a = np.random.choice(
                            np.array(list(action_vec.values()), dtype=object)[:-2]
                        )
                        # print(a)
                        return a
                else:
                    if not isinstance(state, str):
                        # print(np.array(action_vec, dtype=object))
                        a = np.random.choice(
                            np.array(list(action_vec.values()), dtype=object)
                        )
                        # print(a)
                        return a
                    else:
                        # print(np.array(action_vec, dtype=object))
                        # without "acc" and "end" since not available in INIT
                        a = np.random.choice(
                            np.array(list(action_vec.values()), dtype=object)[:-2]
                        )
                        # print(a)
                        return a
            else:
                # map current MDP state to "index space" state
                state_m = self._state_mapper(state, unit_price_issue, quantity_issue)
                # print("q")

                # select greedily
                max_q = max(q[state_m][action] for action in q[state_m].keys())
                values = []
                for action in q[state_m].keys():
                    if max_q != q[state_m][action]:
                        values.append(q[state_m][action])
                second_q = max(values)
                if self.failure_threshold[partner]:
                    if (
                        self.failure_list[partner]
                        >= self.failure_threshold[partner] - 2
                    ):
                        if self.adviced_list[partner]:
                            self.adviced_list[partner] += 1
                        else:
                            self.adviced_list[partner] = 1
                        max_q = second_q
                else:
                    self.failure_threshold[partner] = 8
                    if self.adviced_list[partner]:
                        self.adviced_list[partner] += 1
                    else:
                        self.adviced_list[partner] = 1
                    if (
                        self.failure_list[partner]
                        >= self.failure_threshold[partner] - 2
                    ):
                        # action selection with decay - for imitation learning (imitating LearningAgent). not really used now.
                        # SO THIS CODE IS NOT REALLY RUNNING. ONLY THE "ELSE"
                        # print(self.failure_threshold[partner])
                        max_q = second_q
                # second_q=max(list(q[state_m].values()).pop(max_q))
                constrained_ql = True
                # if caller == "respond":
                #     p_aux = np.random.random()
                #     if isinstance(state_m, tuple):
                #         if len(state_m) == 5:
                #             if self.is_seller:
                #                 cur_profit = (state_m[3]-float(self.awi.current_exogenous_input_price)/self.awi.current_exogenous_input_quantity-self.awi.profile.cost)*state_m[4]
                #             else:
                #                 cur_profit = (float(self.awi.current_exogenous_output_price)/self.awi.current_exogenous_output_quantity-state_m[3]-self.awi.profile.cost)*state_m[4]#*self.awi.current_exogenous_output_quantity

                #             if cur_profit > 0:
                #                 #if p_aux > (0.75*float(self.counters[10]+2))/(self.counters[10]+1):
                #                 if p_aux > 1.0/float(self.counters[10]+1):
                #                     constrained_ql = False

                if constrained_ql:
                    # print([q[state][action] for action in q[state].keys()])
                    # print("max_q: ", max_q)
                    # print(q)
                    # print("state_m: ", state_m)
                    # print(q[state_m].keys())

                    # searching the action index
                    for a in q[state_m].keys():
                        if q[state_m][a] == max_q:
                            # print(a)
                            # print("a:", a)
                            selected_a = a
                            break

                    balance_est = self.online_balance  # self.awi.current_balance

                    ######################################################
                    #               DIRECT RISK MANAGEMENT
                    ######################################################
                    # heuristics in order to constrain the q learning to make it avoid pitfalls
                    if selected_a == "end":
                        return "end"
                        # consider_dev = False
                        # if len(state) == 5:
                        #     pr_i = 2
                        #     q_i = 3
                        #     consider_dev = True
                        # elif len(state) == 3:
                        #     pr_i = 0
                        #     q_i = 1
                        #     consider_dev = True

                        # if consider_dev:
                        #     if (state[q_i]-self._needed() > -2) and (self._needed() > 0):
                        #         if self.is_seller:
                        #             cur_profit = (state[pr_i]-float(self.awi.current_exogenous_input_price)/self.awi.current_exogenous_input_quantity-self.awi.profile.cost)*state[q_i]
                        #             if cur_profit > 0 and state[pr_i] > 0: #unit_price_issue.min_value:
                        #                 if balance_est >= (float(self.awi.current_exogenous_input_price)/self.awi.current_exogenous_input_quantity+self.awi.profile.cost)*state[q_i]:
                        #                     return "acc"
                        #         else:
                        #             cur_profit = (float(self.awi.current_exogenous_output_price)/self.awi.current_exogenous_output_quantity-state[pr_i]-self.awi.profile.cost)*state[q_i]#*self.awi.current_exogenous_output_quantity
                        #             if cur_profit > 0 and state[pr_i] <= unit_price_issue.max_value and state[pr_i] <= float(balance_est)/4:
                        #                 if balance_est >= (state[pr_i]+self.awi.profile.cost)*state[q_i]:
                        #                     return "acc"

                    # inverting acceptance to trace to ending (non-acceptance) in case it is putting us in risk
                    elif selected_a == "acc":
                        consider_dev = False
                        if len(state) == 5:
                            pr_i = 2
                            q_i = 3
                            consider_dev = True
                        elif len(state) == 3:
                            pr_i = 0
                            q_i = 1
                            consider_dev = True

                        if consider_dev:
                            if self.is_seller:
                                if self.awi.current_exogenous_input_quantity <= 0:
                                    return "end"

                                if (
                                    balance_est
                                    < (
                                        float(self.awi.current_exogenous_input_price)
                                        / (
                                            self.awi.current_exogenous_input_quantity
                                            + 0.05
                                        )
                                        + self.awi.profile.cost
                                    )
                                    * state[q_i]
                                ):
                                    # print("1")
                                    return "end"
                                if state[pr_i] <= 0:
                                    # print("2")
                                    return "end"
                                cur_profit = (
                                    state[pr_i]
                                    - float(self.awi.current_exogenous_input_price)
                                    / (self.awi.current_exogenous_input_quantity + 0.05)
                                    - self.awi.profile.cost
                                ) * state[q_i]
                            else:
                                if self.awi.current_exogenous_output_quantity <= 0:
                                    return "end"

                                if (
                                    balance_est
                                    < (state[pr_i] + self.awi.profile.cost) * state[q_i]
                                ):
                                    # print("1")
                                    return "end"
                                if state[pr_i] > unit_price_issue.max_value:
                                    # print("2")
                                    return "end"
                                if state[pr_i] > float(balance_est) / 4:
                                    # print("3")
                                    return "end"

                                cur_profit = (
                                    float(self.awi.current_exogenous_output_price)
                                    / (
                                        self.awi.current_exogenous_output_quantity
                                        + 0.05
                                    )
                                    - state[pr_i]
                                    - self.awi.profile.cost
                                ) * state[
                                    q_i
                                ]  # *self.awi.current_exogenous_output_quantity

                            if (
                                (cur_profit <= 0)
                                or (state[q_i] - self._needed() < -2)
                                or (self._needed() <= 0)
                            ):
                                # print("4")
                                return "end"

                    # inverting various proposal we consider to other ones in order to avoid pitfalls / create
                    # advantages directly
                    else:
                        if self._needed() <= 0:
                            return "end"

                        for a in action_vec.values():
                            if a == action_vec[selected_a]:
                                # print(a)
                                selected_mapped_a = a
                                break

                        pro_price = selected_mapped_a[0]
                        pro_quan = selected_mapped_a[1]
                        # fixing price for minimal profit
                        price_modified = False
                        if self.is_seller:
                            if self.awi.current_exogenous_input_quantity <= 0:
                                return "end"
                            if pro_price < unit_price_issue.min_value:
                                pro_price = unit_price_issue.min_value
                                price_modified = True
                            if (
                                pro_price
                                - float(self.awi.current_exogenous_input_price)
                                / (self.awi.current_exogenous_input_quantity + 0.05)
                                - self.awi.profile.cost
                                <= 0
                            ):
                                # print("5")
                                pro_price = (
                                    float(self.awi.current_exogenous_input_price)
                                    / (self.awi.current_exogenous_input_quantity + 0.05)
                                    + self.awi.profile.cost
                                    + 1
                                )
                                price_modified = True
                        else:
                            if self.awi.current_exogenous_output_quantity <= 0:
                                return "end"
                            if pro_price > unit_price_issue.max_value:
                                pro_price = unit_price_issue.max_value
                                price_modified = True
                            if (
                                float(self.awi.current_exogenous_output_price)
                                / (self.awi.current_exogenous_output_quantity + 0.05)
                                - pro_price
                                - self.awi.profile.cost
                                <= 0
                            ):
                                # print("6")
                                pro_price = max(
                                    float(self.awi.current_exogenous_output_price)
                                    / (
                                        self.awi.current_exogenous_output_quantity
                                        + 0.05
                                    )
                                    - self.awi.profile.cost
                                    - 1,
                                    unit_price_issue.min_value,
                                )
                                price_modified = True

                        # limiting quantity to avoid negative balance
                        if self.is_seller:
                            price = (
                                float(self.awi.current_exogenous_input_price)
                                / (self.awi.current_exogenous_input_quantity + 0.05)
                            ) + self.awi.profile.cost
                            if balance_est < price * pro_quan:
                                # print("7")
                                return (pro_price, float(balance_est) / price)
                                # pro_quan = float(balance_est)/price
                            elif price_modified:
                                return (pro_price, pro_quan)
                                # pro_quan = pro_quan
                        else:
                            if (
                                balance_est
                                < (pro_price + self.awi.profile.cost) * pro_quan
                            ):
                                # print("8")
                                return (pro_price, float(balance_est) / pro_price)
                                # pro_quan = float(balance_est)/pro_price
                            elif price_modified:
                                return (pro_price, pro_quan)
                                # pro_quan = pro_quan

                        # best effort greediness: try to present an offer which is very good for us
                        # in some small probability - maybe it will catch and help us to gain more reward
                        # unrelated to the "smart" strategy
                        p3 = np.random.random()
                        # DEBUG2_PRINT("opportunistic !")

                        # small-probability opportunistic behavior here as well
                        if self.is_seller:
                            if p3 < self.opportunistic_level:
                                return (4 * unit_price_issue.max_value, pro_quan)
                        else:
                            if p3 < self.opportunistic_level / 10:
                                return (unit_price_issue.min_value, pro_quan)

                    # if we got here (should be most cases), we act as a regular Q-learning process
                    for a in action_vec.values():
                        if a == action_vec[selected_a]:
                            # print(a)
                            return a
                else:
                    # not happening
                    return "acc"

    """
    Q-Learning Q table init - a method to initialize the Q table and action_vec
    We initialize to generic keys - indexes - so we can be independent of specific simulation' details
    (since state and actions are prices nd quanities and these may change, we encode them in the range as indexes.
    there is a rough assumption here that the relations between values count - where they are in the range -
    so we can expolit that to enable learning from simulation to simulation)
    :param: q_t - Q table
    :param: unit_price_issue - includes the min and max prices
    :param: quantity_issue - includes the min and max quantities
    :param: action_vec - an empty vector to be initialized with all possible actions
    :param: is_seller - True if I'm "seller" (L0)
    :param: ami - the framework's AMI
    """

    def _q_learning_q_init(
        self,
        q_t,
        unit_price_issue,
        quantity_issue,
        action_vec,
        is_seller=True,
        ami=None,
    ):
        # DEBUG_PRINT("_q_learning_q_init")

        # print(quantity_issue)
        # print(unit_price_issue)

        # if self.q_table_per_opp:
        #     if self.q_table_per_opp[list(self.q_table_per_opp.keys())[0]]:
        #         #print(len(self.q_table_per_opp[list(self.q_table_per_opp.keys())[0]]))
        #         q_t = self.q_table_per_opp[list(self.q_table_per_opp.keys())[0]]
        #         action_vec = self.action_vec[list(self.q_table_per_opp.keys())[0]]
        #         return [q_t, action_vec]

        # limiting possible actions for the agent: this gives better performance
        min_q = (
            quantity_issue.min_value
            + self.prune_quan
            * (quantity_issue.max_value - quantity_issue.min_value)
            / 2
        )
        max_q = quantity_issue.max_value

        # Calculating ranges for all states and actions
        p_range = np.linspace(
            unit_price_issue.min_value, unit_price_issue.max_value, self.price_res
        )
        my_q_range = np.linspace(
            min_q, max_q, self.quantity_res
        )  # [0:int(np.ceil(self.quantity_res/5))]
        # print(my_q_range)
        q_range = np.linspace(
            quantity_issue.min_value, quantity_issue.max_value, self.quantity_res
        )
        c_mx_v = G_C_MX_V  # 1.1#1.5
        c_mn_v = G_C_MN_V  # 0.9#0.5
        if self.is_seller:
            my_p_range = np.linspace(
                G_C_MX4MN_V * unit_price_issue.min_value,
                c_mx_v * unit_price_issue.max_value,
                self.price_res,
            )
        else:
            my_p_range = np.linspace(
                c_mn_v * unit_price_issue.min_value,
                G_C_MN4MX_V * unit_price_issue.max_value,
                self.price_res,
            )

        # print(list(p_range))
        # print(list(my_q_range))
        # print(list(my_p_range))
        # print(list(q_range))

        # Going over all state type and building a basic table
        for s in [
            STATE_TYPE.INIT,
            STATE_TYPE.OPP_COUNTER,
            STATE_TYPE.END,
            STATE_TYPE.ACCEPT,
        ]:  # , STATE_TYPE.MY_COUNTER
            if not self.complex_state:
                q_t[s] = dict()
            else:
                if s == STATE_TYPE.INIT:
                    # for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                    for i_n, needs in enumerate(self.ne_range):
                        # q_t[str(needs)] = dict()
                        q_t[str(i_n)] = dict()
                elif s == STATE_TYPE.OPP_COUNTER:
                    for i_ps_so, p_so in enumerate(p_range):
                        for i_q_so, q_so in enumerate(q_range):
                            for i_p_s, p_s in enumerate(my_p_range):
                                for i_q_s, q_s in enumerate(my_q_range):
                                    # for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                    for i_n, needs in enumerate(self.ne_range):
                                        # q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)] = dict()
                                        q_t[
                                            (i_p_s, i_q_s, i_ps_so, i_q_so, i_n)
                                        ] = dict()
                            # for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                            for i_n, needs in enumerate(self.ne_range):
                                q_t[(i_ps_so, i_q_so, i_n)] = dict()
                else:
                    q_t[s] = dict()

            # print(unit_price_issue.min_value, unit_price_issue.max_value, self.price_res)
            # print(quantity_issue.min_value, quantity_issue.max_value, self.quantity_res)

            # initializing all values in the table
            for i_p, p in enumerate(my_p_range):
                # print(q_t)
                for i_q, q in enumerate(my_q_range):
                    # print(q_t)
                    # print((p,q))
                    if self.smart_init:
                        if is_seller:
                            if s == STATE_TYPE.INIT:
                                # for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                for i_n, needs in enumerate(self.ne_range):
                                    # q_t[str(needs)][(p,np.ceil(q))] = self._initial_q_seller(p,np.ceil(q),needs, ami, str(needs))#10000
                                    q_t[str(i_n)][(i_p, i_q)] = self._initial_q_seller(
                                        p,
                                        np.ceil(q),
                                        needs,
                                        ami,
                                        str(needs),
                                        None,
                                        (i_p, i_q),
                                    )  # 10000
                            elif s == STATE_TYPE.OPP_COUNTER:
                                for i_ps_so, p_so in enumerate(p_range):
                                    for i_q_so, q_so in enumerate(q_range):
                                        for i_p_s, p_s in enumerate(my_p_range):
                                            for i_q_s, q_s in enumerate(my_q_range):
                                                # for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                                for i_n, needs in enumerate(
                                                    self.ne_range
                                                ):
                                                    # q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)][(p,np.ceil(q))] = self._initial_q_seller(p,np.ceil(q),needs, ami, (p_s,np.ceil(q_s),p_so,q_so,needs))#10000
                                                    q_t[
                                                        (
                                                            i_p_s,
                                                            i_q_s,
                                                            i_ps_so,
                                                            i_q_so,
                                                            i_n,
                                                        )
                                                    ][
                                                        (i_p, i_q)
                                                    ] = self._initial_q_seller(
                                                        p,
                                                        np.ceil(q),
                                                        needs,
                                                        ami,
                                                        (
                                                            p_s,
                                                            np.ceil(q_s),
                                                            p_so,
                                                            q_so,
                                                            needs,
                                                        ),
                                                        None,
                                                        (i_p, i_q),
                                                    )  # 10000

                                        # for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                        for i_n, needs in enumerate(self.ne_range):
                                            # q_t[(p_so,q_so,needs)][(p,np.ceil(q))] = self._initial_q_seller(p,np.ceil(q),needs, ami, (p_so,q_so,needs))#0
                                            q_t[(i_ps_so, i_q_so, i_n)][
                                                (i_p, i_q)
                                            ] = self._initial_q_seller(
                                                p,
                                                np.ceil(q),
                                                needs,
                                                ami,
                                                (p_so, q_so, needs),
                                                None,
                                                (i_p, i_q),
                                            )  # 0
                            else:
                                # q_t[s][(p,np.ceil(q))] = 0#self._initial_q_seller(p,np.ceil(q),needs,ami)
                                q_t[s][
                                    (i_p, i_q)
                                ] = 0  # self._initial_q_seller(p,np.ceil(q),needs,ami)
                        else:
                            if s == STATE_TYPE.INIT:
                                # for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                for i_n, needs in enumerate(self.ne_range):
                                    # q_t[str(needs)][(p,np.ceil(q))] = self._initial_q_buyer(p,np.ceil(q),needs, ami, str(needs))#10000
                                    q_t[str(i_n)][(i_p, i_q)] = self._initial_q_buyer(
                                        p,
                                        np.ceil(q),
                                        needs,
                                        ami,
                                        str(needs),
                                        None,
                                        (i_p, i_q),
                                    )  # 10000
                            elif s == STATE_TYPE.OPP_COUNTER:
                                for i_ps_so, p_so in enumerate(p_range):
                                    for i_q_so, q_so in enumerate(q_range):
                                        for i_p_s, p_s in enumerate(my_p_range):
                                            for i_q_s, q_s in enumerate(my_q_range):
                                                # for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                                for i_n, needs in enumerate(
                                                    self.ne_range
                                                ):
                                                    # q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)][(p,np.ceil(q))] = self._initial_q_buyer(p,np.ceil(q),needs, ami, (p_s,np.ceil(q_s),p_so,q_so,needs))#10000
                                                    q_t[
                                                        (
                                                            i_p_s,
                                                            i_q_s,
                                                            i_ps_so,
                                                            i_q_so,
                                                            i_n,
                                                        )
                                                    ][
                                                        (i_p, i_q)
                                                    ] = self._initial_q_buyer(
                                                        p,
                                                        np.ceil(q),
                                                        needs,
                                                        ami,
                                                        (
                                                            p_s,
                                                            np.ceil(q_s),
                                                            p_so,
                                                            q_so,
                                                            needs,
                                                        ),
                                                        None,
                                                        (i_p, i_q),
                                                    )  # 10000
                                        # for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                        for i_n, needs in enumerate(self.ne_range):
                                            # q_t[(p_so,q_so,needs)][(p,np.ceil(q))] = self._initial_q_buyer(p,np.ceil(q),needs, ami, (p_so,q_so,needs))#0
                                            q_t[(i_ps_so, i_q_so, i_n)][
                                                (i_p, i_q)
                                            ] = self._initial_q_buyer(
                                                p,
                                                np.ceil(q),
                                                needs,
                                                ami,
                                                (p_so, q_so, needs),
                                                None,
                                                (i_p, i_q),
                                            )  # 0
                            else:
                                # q_t[s][(p,np.ceil(q))] = 0#self._initial_q_buyer(p, np.ceil(q), )
                                q_t[s][(i_p, i_q)] = 0
                    else:
                        if s == STATE_TYPE.INIT:
                            # for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                            for i_n, needs in enumerate(self.ne_range):
                                # q_t[str(i_n)][(p,np.ceil(q))] = 0#10000
                                q_t[str(i_n)][(i_p, i_q)] = 0
                        elif s == STATE_TYPE.OPP_COUNTER:
                            for i_ps_so, p_so in enumerate(p_range):
                                for i_q_so, q_so in enumerate(q_range):
                                    for i_p_s, p_s in enumerate(my_p_range):
                                        for i_q_s, q_s in enumerate(my_q_range):
                                            # for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                            for i_n, needs in enumerate(self.ne_range):
                                                # q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)][(p,np.ceil(q))] = 0
                                                q_t[
                                                    (i_p_s, i_q_s, i_ps_so, i_q_so, i_n)
                                                ][(i_p, i_q)] = 0
                                    # for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                    for i_n, needs in enumerate(self.ne_range):
                                        # q_t[(p_so,q_so,needs)][(p,np.ceil(q))] = 0
                                        q_t[(i_ps_so, i_q_so, i_n)][(i_p, i_q)] = 0
                        else:
                            # q_t[s][(p,np.ceil(q))] = 0
                            q_t[s][(i_p, i_q)] = 0

            # if s == STATE_TYPE.OPP_COUNTER:
            # if s != STATE_TYPE.INIT:
            if s == STATE_TYPE.OPP_COUNTER:
                if not self.complex_state:
                    q_t[s]["end"] = 0
                    q_t[s]["acc"] = 0
                else:
                    for i_ps_so, p_so in enumerate(p_range):
                        for i_q_so, q_so in enumerate(q_range):
                            for i_p_s, p_s in enumerate(my_p_range):
                                for i_q_s, q_s in enumerate(my_q_range):
                                    # for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                    for i_n, needs in enumerate(self.ne_range):
                                        if is_seller:
                                            # q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)]["end"] = self._initial_q_seller(p_so,q_so,needs,ami, (p_s,np.ceil(q_s),p_so,q_so,needs), "end")
                                            q_t[(i_p_s, i_q_s, i_ps_so, i_q_so, i_n)][
                                                "end"
                                            ] = self._initial_q_seller(
                                                p_so,
                                                q_so,
                                                needs,
                                                ami,
                                                (p_s, np.ceil(q_s), p_so, q_so, needs),
                                                "end",
                                            )
                                            # q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)]["acc"] = self._initial_q_seller(p_so,q_so,needs,ami, (p_s,np.ceil(q_s),p_so,q_so,needs), "acc")#p_so*q_so-self.awi.current_exogenous_input_price
                                            q_t[(i_p_s, i_q_s, i_ps_so, i_q_so, i_n)][
                                                "acc"
                                            ] = self._initial_q_seller(
                                                p_so,
                                                q_so,
                                                needs,
                                                ami,
                                                (p_s, np.ceil(q_s), p_so, q_so, needs),
                                                "acc",
                                            )  # p_so*q_so-self.awi.current_exogenous_input_price
                                        else:
                                            # q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)]["end"] = self._initial_q_buyer(p_so,q_so,needs,ami, (p_s,np.ceil(q_s),p_so,q_so,needs), "end")
                                            q_t[(i_p_s, i_q_s, i_ps_so, i_q_so, i_n)][
                                                "end"
                                            ] = self._initial_q_buyer(
                                                p_so,
                                                q_so,
                                                needs,
                                                ami,
                                                (p_s, np.ceil(q_s), p_so, q_so, needs),
                                                "end",
                                            )
                                            # q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)]["acc"] = self._initial_q_buyer(p_so,q_so,needs,ami, (p_s,np.ceil(q_s),p_so,q_so,needs), "acc") #self.awi.current_exogenous_input_price-(p_so-p_s)*(q_so-q_s)
                                            q_t[(i_p_s, i_q_s, i_ps_so, i_q_so, i_n)][
                                                "acc"
                                            ] = self._initial_q_buyer(
                                                p_so,
                                                q_so,
                                                needs,
                                                ami,
                                                (p_s, np.ceil(q_s), p_so, q_so, needs),
                                                "acc",
                                            )  # self.awi.current_exogenous_input_price-(p_so-p_s)*(q_so-q_s)
                            # for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                            for i_n, needs in enumerate(self.ne_range):
                                if is_seller:
                                    # q_t[(p_so,q_so,needs)]["end"] = self._initial_q_seller(p_so,q_so,needs,ami, (p_so,q_so,needs), "end")
                                    q_t[(i_ps_so, i_q_so, i_n)][
                                        "end"
                                    ] = self._initial_q_seller(
                                        p_so,
                                        q_so,
                                        needs,
                                        ami,
                                        (p_so, q_so, needs),
                                        "end",
                                    )
                                    # q_t[(p_so,q_so,needs)]["acc"] = self._initial_q_seller(p_so,q_so,needs,ami, (p_so,q_so,needs), "acc")
                                    q_t[(i_ps_so, i_q_so, i_n)][
                                        "acc"
                                    ] = self._initial_q_seller(
                                        p_so,
                                        q_so,
                                        needs,
                                        ami,
                                        (p_so, q_so, needs),
                                        "acc",
                                    )
                                else:
                                    # q_t[(p_so,q_so,needs)]["end"] = self._initial_q_buyer(p_so,q_so,needs,ami, (p_so,q_so,needs), "end")
                                    q_t[(i_ps_so, i_q_so, i_n)][
                                        "end"
                                    ] = self._initial_q_buyer(
                                        p_so,
                                        q_so,
                                        needs,
                                        ami,
                                        (p_so, q_so, needs),
                                        "end",
                                    )
                                    # q_t[(p_so,q_so,needs)]["acc"] = self._initial_q_buyer(p_so,q_so,needs,ami, (p_so,q_so,needs), "acc")
                                    q_t[(i_ps_so, i_q_so, i_n)][
                                        "acc"
                                    ] = self._initial_q_buyer(
                                        p_so,
                                        q_so,
                                        needs,
                                        ami,
                                        (p_so, q_so, needs),
                                        "acc",
                                    )
        # initializing the action vector
        # action_vec = dict()
        for i_p_s, p_s in enumerate(my_p_range):
            for i_q_s, q_s in enumerate(my_q_range):
                # action_vec.append((p,np.ceil(q)))
                action_vec[(i_p_s, i_q_s)] = (p_s, q_s)  # (p,np.ceil(q))

        # action_vec.append("end")
        # action_vec.append("acc")
        action_vec["end"] = "end"
        action_vec["acc"] = "acc"

        # print(q_t)

        # for k in q_t.keys():
        #     for a in q_t[k]:
        #         print(q_t[k][a], a)
        # exit()

        # return [q_t, action_vec]
        # print("init ", len(q_t.keys()))

    """
    Map env's statuses to best matching state
    :param: p - my last offer's unit price
    :param: q - my last offer's quantity
    :param: po - opponent offer's unit price
    :param: qo - opponent offer's quantity
    :param: unit_price_issue - includes the min and max prices
    :param: quantity_issue - includes the min and max quantities
    :param: needs - current needs
    :return: the matching state
    """

    def _find_state_mapping(
        self, p, q, po, qo, unit_price_issue, quantity_issue, needs
    ):
        # limiting possible actions for the agent: this gives better performance
        min_q = (
            quantity_issue.min_value
            + self.prune_quan
            * (quantity_issue.max_value - quantity_issue.min_value)
            / 2
        )
        max_q = quantity_issue.max_value

        # Calculating ranges for all states and actions
        p_range = np.linspace(
            unit_price_issue.min_value, unit_price_issue.max_value, self.price_res
        )
        my_q_range = np.linspace(
            min_q, max_q, self.quantity_res
        )  # [0:int(np.ceil(self.quantity_res/5))]
        # print(my_q_range)
        q_range = np.linspace(
            quantity_issue.min_value, quantity_issue.max_value, self.quantity_res
        )
        c_mx_v = G_C_MX_V  # 1.1#1.5
        c_mn_v = G_C_MN_V  # 0.9#0.5
        if self.is_seller:
            my_p_range = np.linspace(
                G_C_MX4MN_V * unit_price_issue.min_value,
                c_mx_v * unit_price_issue.max_value,
                self.price_res,
            )
        else:
            my_p_range = np.linspace(
                c_mn_v * unit_price_issue.min_value,
                G_C_MN4MX_V * unit_price_issue.max_value,
                self.price_res,
            )

        if needs >= 0:
            return (
                self._find_nearest(my_p_range, p),
                self._find_nearest(my_q_range, q),
                self._find_nearest(p_range, po),
                self._find_nearest(q_range, qo),
                self._find_nearest(self.ne_range, needs),
            )
        else:
            return (
                self._find_nearest(my_p_range, p),
                self._find_nearest(my_q_range, q),
                self._find_nearest(p_range, po),
                self._find_nearest(q_range, qo),
                -1,
            )

    """
    Find nearest array item to value:
    :param: array - the array
    :param: value - the value we search closest item to it in the array
    :return: the found item closest to value
    """

    def _find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    """
    Needed quanity still need to sell/buy (from exogenous contracts)
    :param: negotiator_id - unused
    :return: the needs
    """

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    """
    Check if we are a "seller" (L0)
    :param: ami - the AMI object of the framework
    :return: True iff "seller", else "buyer"
    """

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    """
    Penalty calculating for buying too much
    :param: ami - the AMI object of the framework
    :return: the penalty per unit
    """

    # storage : buying to much == disposal, delivery : selling too much == shortfall
    def _too_much_penalty(self, ami):
        if self._is_selling(ami):
            return (
                self.awi.profile.shortfall_penalty_mean
            )  # , self.awi.profile.shortfall_penalty_dev
        else:
            return (
                self.awi.profile.disposal_cost_mean
            )  # , self.awi.profile.disposal_cost_dev

    """
    Penalty calculating for buying too less
    :param: ami - the AMI object of the framework
    :return: the penalty per unit
    """

    def _too_less_penalty(self, ami):
        if self._is_selling(ami):
            return (
                self.awi.profile.disposal_cost_mean
            )  # , self.awi.profile.disposal_cost_dev
        else:
            return (
                self.awi.profile.shortfall_penalty_mean
            )  # , self.awi.profile.shortfall_penalty_dev

    """
    Initialize Q values for "seller" (L0)
    :param: p - action's price
    :param: q - action's quantity
    :param: needs - the needs
    :param: ami - AMI object
    :param: state - the state
    :param: terminal_action - in case not known, this is the action and not (p,q)
    :return: value to initialize in Q table for the state,action pair
    """

    def _initial_q_seller(
        self, p, q, needs, ami, state, terminal_action=None, a_i=None
    ):
        if self.load_q:
            if self.load_q_type == "exact":
                if state in self.learned_q_seller_t:
                    if terminal_action == "acc":
                        if "acc" in self.learned_q_seller_t[state]:
                            return self.learned_q_seller_t[state]["acc"]
                    elif terminal_action == "end":
                        if "end" in self.learned_q_seller_t[state]:
                            return self.learned_q_seller_t[state]["end"]
                    elif isinstance(a_i, tuple):
                        return self.learned_q_seller_t[state][a_i]
                    else:
                        if (p, q) in self.learned_q_seller_t[state]:
                            return self.learned_q_seller_t[state][(p, q)]
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        mxp = unit_price_issue.max_value
        mnp = unit_price_issue.min_value
        # if q <= needs:
        # return np.random.random()-0.5
        if needs < 0:
            if terminal_action == "end":
                return unit_price_issue.max_value  # 1
            else:
                return -unit_price_issue.max_value  # -1#*(q/needs)#p*q
        else:
            if terminal_action == "end":
                return -0.01 * unit_price_issue.max_value  # *(1-0.001*abs(needs-q))
            elif terminal_action == "acc":
                return (
                    1.001 * p
                )  # *(1+0.001*abs(needs-q))#1.01*(p/mxp)#1.001*1.001*(p-mnp)/(mxp-mnp) # + 0.01#1.01*p/unit_price_issue.max_value
            else:
                return p  # *(1+0.001*abs(needs-q))

    """
    Initialize Q values for "buyer" (L1)
    :param: p - action's price
    :param: q - action's quantity
    :param: needs - the needs
    :param: ami - AMI object
    :param: state - the state
    :param: terminal_action - in case not known, this is the action and not (p,q)
    :return: value to initialize in Q table for the state,action pair
    """

    def _initial_q_buyer(self, p, q, needs, ami, state, terminal_action=None, a_i=None):
        if self.load_q:
            if self.load_q_type == "exact":
                if state in self.learned_q_buyer_t:
                    if terminal_action == "acc":
                        if "acc" in self.learned_q_buyer_t[state]:
                            return self.learned_q_buyer_t[state]["acc"]
                    elif terminal_action == "end":
                        if "end" in self.learned_q_buyer_t[state]:
                            return self.learned_q_buyer_t[state]["end"]
                    elif isinstance(a_i, tuple):
                        return self.learned_q_buyer_t[state][a_i]
                    else:
                        if (p, q) in self.learned_q_buyer_t[state]:
                            return self.learned_q_buyer_t[state][(p, q)]

        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        mxp = unit_price_issue.max_value
        mnp = unit_price_issue.min_value
        # return np.random.random()-0.5
        # if q <= needs:

        if needs < 0:
            if terminal_action == "end":
                return unit_price_issue.max_value
            else:
                return (
                    -unit_price_issue.max_value
                )  # *(needs/q)#q*(unit_price_issue.max_value-p)
        else:
            if terminal_action == "end":
                return -0.01 * unit_price_issue.max_value  # *(1+0.001*abs(needs-q))
            elif terminal_action == "acc":
                # return 1.01*(unit_price_issue.max_value-p)/unit_price_issue.max_value
                return 1.001 * (
                    mxp - p
                )  # *(1+0.001*abs(needs-q))#1.01*(mxp-p)/mxp#1.001*1.001*(mxp-p-mnp)/(mxp-mnp) #+ 0.01
            else:
                # return (unit_price_issue.max_value-p)/unit_price_issue.max_value#q*(unit_price_issue.max_value-p)#-p#q*(unit_price_issue.max_value-p)-(needs-q)*self._too_much_penalty(ami)
                return mxp - p  # *(1+0.001*abs(needs-q))

    """
    STATUS : currently unused. here for future usage if needed. Currently "state_mapper" used instead
    Mapping vector of values linearly to fixed targets
    :param: v : (v0, v1, ...) - the values to map
    :param: mx_mn : [[max0, min0], [max1, min1], ...] - the values' ranges
    :param: sizes : [size0, size1, ...] - the number of possible values in each range
    :return: mapped value
    """

    def mapl_v(self, v, mx_mn, sizes):
        v_mapped = []
        for i, val in enumerate(v):
            v_mapped.append(
                sizes[i] * (float(val - mx_mn[i][0]) / (mx_mn[i][1] - mx_mn[i][0]))
            )
        return tuple(v_mapped)

    """
    STATUS : currently unused. here for future usage if needed. Currently "state_mapper" used instead
    Mapping scalar linearly to fixed target
    :param: z - the scalar to map
    :param: mx - the max of the z's range
    :param: mn - the min of the z's range
    :param: size - the number of possible values z's range
    :return: mapped value
    """

    def mapl_s(self, z, mx, mn, size):
        # print("mapl_s")
        # print(z)
        # print(mx)
        # print(mn)
        # print(size)
        return size * (float(z - mn) / (mx - mn))

    """
    STATUS : currently unused. here for future usage if needed. Currently "state_mapper" used instead
    Mapping Q table's keys linearly target Q with fixed global keys
    :param: q_t - the Q table
    :return: the mapped Q table
    """

    def map_q(self, q_t):
        new_q = dict()
        tuple_5_s = []
        tuple_3_s = []
        for s in q_t.keys():
            if isinstance(s, tuple):
                if len(s) == 5:
                    tuple_5_s.append(s)
                elif len(s) == 3:
                    tuple_3_s.append(s)
                else:
                    pass  # print("fatal1")

        pv_s5 = [p for p, _, _, _, _ in tuple_5_s]
        qv_s5 = [q for _, q, _, _, _ in tuple_5_s]
        pv_o5 = [p for _, _, p, _, _ in tuple_5_s]
        qv_o5 = [q for _, _, _, q, _ in tuple_5_s]
        len_p_s5 = len(np.unique(pv_s5))
        len_q_s5 = len(np.unique(qv_s5))
        len_p_o5 = len(np.unique(pv_o5))
        len_q_o5 = len(np.unique(qv_o5))
        max_p_s5 = max(pv_s5)
        min_p_s5 = min(pv_s5)
        max_p_o5 = max(pv_o5)
        min_p_o5 = min(pv_o5)
        max_q_s5 = max(qv_s5)
        min_q_s5 = min(qv_s5)
        max_q_o5 = max(qv_o5)
        min_q_o5 = min(qv_o5)

        pv_o3 = [p for p, _, _ in tuple_3_s]
        qv_o3 = [q for _, q, _ in tuple_3_s]
        len_p_o3 = len(np.unique(pv_o3))
        len_q_o3 = len(np.unique(qv_o3))
        max_p_o3 = max(pv_o3)
        min_p_o3 = min(pv_o3)
        max_q_o3 = max(qv_o3)
        min_q_o3 = min(qv_o3)

        for s in q_t.keys():
            if isinstance(s, str):
                # INIT
                pv_a = [p for p, _ in q_t[s].keys()]
                qv_a = [q for _, q in q_t[s].keys()]
                len_p_a = len(np.unique(pv_a))
                len_q_a = len(np.unique(qv_a))
                max_p_a = max(pv_a)
                min_p_a = min(pv_a)
                max_q_a = max(qv_a)
                min_q_a = min(qv_a)

                mapped_s = self.mapl_s(int(s), -1, self.awi.n_lines, self.needs_res)
                new_q[mapped_s] = dict()
                # print(list(q[s].keys()))
                for p, q in list(q_t[s].keys()):
                    # print(p)
                    # print(q)
                    mapped_p = self.mapl_s(p, max_p_a, min_p_a, len_p_a)
                    mapped_q = self.mapl_s(q, max_q_a, min_q_a, len_q_a)
                    # print(mapped_s, (mapped_p, mapped_q))
                    # print(new_q)
                    # print(q_t[s])
                    # print(q_t[s][(p, q)])
                    new_q[mapped_s][(mapped_p, mapped_q)] = q_t[s][(p, q)]

            elif isinstance(s, tuple):
                # TODO : need to work on all the 5-tuples under q, not q[s]. need to fix !
                actions = list(q_t[s].keys())
                if len(s) == 5:
                    # OPP
                    pv_a = [p for p, _ in actions[:-2]]
                    qv_a = [q for _, q in actions[:-2]]
                    len_p_a = len(np.unique(pv_a))
                    len_q_a = len(np.unique(qv_a))
                    max_p_a = max(pv_a)
                    min_p_a = min(pv_a)
                    max_q_a = max(qv_a)
                    min_q_a = min(qv_a)

                    p_s, q_s, p_o, q_o, n = s
                    mapped_p_s = self.mapl_s(p_s, max_p_s5, min_p_s5, len_p_s5)
                    mapped_q_s = self.mapl_s(q_s, max_q_s5, min_q_s5, len_q_s5)
                    mapped_p_o = self.mapl_s(p_o, max_p_o5, min_p_o5, len_p_o5)
                    mapped_q_o = self.mapl_s(q_o, max_q_o5, min_q_o5, len_q_o5)

                    mapped_s = self.mapl_s(n, -1, self.awi.n_lines, self.needs_res)
                    new_q[
                        (mapped_p_s, mapped_q_s, mapped_p_o, mapped_q_o, mapped_s)
                    ] = dict()
                    for p, q in actions[:-2]:
                        mapped_p = self.mapl_s(p, max_p_a, min_p_a, len_p_a)
                        mapped_q = self.mapl_s(q, max_q_a, min_q_a, len_q_a)
                        new_q[
                            (mapped_p_s, mapped_q_s, mapped_p_o, mapped_q_o, mapped_s)
                        ][(mapped_p, mapped_q)] = q_t[s][(p, q)]
                    new_q[(mapped_p_s, mapped_q_s, mapped_p_o, mapped_q_o, mapped_s)][
                        "acc"
                    ] = q_t[s]["acc"]
                    new_q[(mapped_p_s, mapped_q_s, mapped_p_o, mapped_q_o, mapped_s)][
                        "end"
                    ] = q_t[s]["end"]

                elif len(s) == 3:
                    # OPP
                    pv_a = [p for p, _ in actions[:-2]]
                    qv_a = [q for _, q in actions[:-2]]
                    len_p_a = len(np.unique(pv_a))
                    len_q_a = len(np.unique(qv_a))
                    max_p_a = max(pv_a)
                    min_p_a = min(pv_a)
                    max_q_a = max(qv_a)
                    min_q_a = min(qv_a)

                    p_o, q_o, n = s
                    mapped_p_o = self.mapl_s(p_o, max_p_o3, min_p_o3, len_p_o3)
                    mapped_q_o = self.mapl_s(q_o, max_q_o3, min_q_o3, len_q_o3)

                    mapped_s = self.mapl_s(n, -1, self.awi.n_lines, self.needs_res)
                    new_q[(mapped_p_o, mapped_q_o, mapped_s)] = dict()
                    for p, q in actions[:-2]:
                        mapped_p = self.mapl_s(p, max_p_a, min_p_a, len_p_a)
                        mapped_q = self.mapl_s(q, max_q_a, min_q_a, len_q_a)
                        new_q[(mapped_p_o, mapped_q_o, mapped_s)][
                            (mapped_p, mapped_q)
                        ] = q_t[s][(p, q)]
                    new_q[(mapped_p_o, mapped_q_o, mapped_s)]["acc"] = q_t[s]["acc"]
                    new_q[(mapped_p_o, mapped_q_o, mapped_s)]["end"] = q_t[s]["end"]

                else:
                    pass  # print("fatal2")

            elif s == STATE_TYPE.END or s == STATE_TYPE.ACCEPT:
                pv_a = [p for p, _ in actions[:-2]]
                qv_a = [q for _, q in actions[:-2]]
                len_p_a = len(np.unique(pv_a))
                len_q_a = len(np.unique(qv_a))
                max_p_a = max(pv_a)
                min_p_a = min(pv_a)
                max_q_a = max(qv_a)
                min_q_a = min(qv_a)

                new_q[s] = dict()
                for p, q in actions[:-2]:
                    mapped_p = self.mapl_s(p, max_p_a, min_p_a, len_p_a)
                    mapped_q = self.mapl_s(q, max_q_a, min_q_a, len_q_a)
                    new_q[s][(mapped_p, mapped_q)] = q_t[s][(p, q)]
                # new_q[s]["acc"] = q_t[s]["acc"]
                # new_q[s]["end"] = q_t[s]["end"]

            else:
                pass  # print("fatal3")
                # END , ACCEPT

        return new_q

    """
    Mapping state from currently sim to Q table generic DB keys (indexes)
    :param: state - this simulation's state
    :param: unit_price_issue - the unit price issue
    :param: quantity_issue - the quantity issue
    :return: the mapped Q table
    """

    def _state_mapper(self, state, unit_price_issue, quantity_issue):
        # DEBUG_PRINT("_q_learning_q_init")

        # print(quantity_issue)
        # print(unit_price_issue)

        # if self.q_table_per_opp:
        #     if self.q_table_per_opp[list(self.q_table_per_opp.keys())[0]]:
        #         #print(len(self.q_table_per_opp[list(self.q_table_per_opp.keys())[0]]))
        #         q_t = self.q_table_per_opp[list(self.q_table_per_opp.keys())[0]]
        #         action_vec = self.action_vec[list(self.q_table_per_opp.keys())[0]]
        #         return [q_t, action_vec]

        # limiting possible actions for the agent: this gives better performance
        min_q = (
            quantity_issue.min_value
            + self.prune_quan
            * (quantity_issue.max_value - quantity_issue.min_value)
            / 2
        )
        max_q = quantity_issue.max_value

        # Calculating ranges for all states and actions
        p_range = np.linspace(
            unit_price_issue.min_value, unit_price_issue.max_value, self.price_res
        )
        my_q_range = np.linspace(
            min_q, max_q, self.quantity_res
        )  # [0:int(np.ceil(self.quantity_res/5))]
        # print(my_q_range)
        q_range = np.linspace(
            quantity_issue.min_value, quantity_issue.max_value, self.quantity_res
        )
        c_mx_v = G_C_MX_V  # 1.1#1.5
        c_mn_v = G_C_MN_V  # 0.9#0.5
        if self.is_seller:
            my_p_range = np.linspace(
                G_C_MX4MN_V * unit_price_issue.min_value,
                c_mx_v * unit_price_issue.max_value,
                self.price_res,
            )
        else:
            my_p_range = np.linspace(
                c_mn_v * unit_price_issue.min_value,
                G_C_MN4MX_V * unit_price_issue.max_value,
                self.price_res,
            )

        if isinstance(state, int):
            return state
        elif isinstance(state, str):
            return str(np.where(self.ne_range == float(state))[0][0])
        elif isinstance(state, tuple):
            if len(state) == 5:
                return (
                    np.where(my_p_range == state[0])[0][0],
                    np.where(my_q_range == state[1])[0][0],
                    np.where(p_range == state[2])[0][0],
                    np.where(q_range == state[3])[0][0],
                    np.where(self.ne_range == state[4])[0][0],
                )

            elif len(state) == 3:
                return (
                    np.where(p_range == state[0])[0][0],
                    np.where(q_range == state[1])[0][0],
                    np.where(self.ne_range == state[2])[0][0],
                )
            else:
                pass  # print("fatal - no supposed to happen")

    """###########################################################

    From here, these are the tutorial "LearningAgent"'s methods (we asked if possible to use them and
    were aswered it is allowed). We tried at some point to
    use them to perform a kind of "imitation learning". This option is now disabled since it didn't
    help so much in the online tournaments (we tried to initially imitate the LearningAgent's
    policy and with time give more room to Q-Learning to happen)

    """  ###########################################################

    def best_offer(self, negotiator_id):
        my_needs = int(self._needed(negotiator_id))
        if my_needs <= 0:
            return None
        ami = self.get_ami(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        offer = [-1] * 3
        # mx = max(min(my_needs, quantity_issue.max_value), quantity_issue.min_value)
        # offer[OFFER_FIELD_IDX.QUANTITY] = random.randint(
        #     max(1, int(0.5 + mx * self.awi.current_step / self.awi.n_steps)), mx
        # )
        offer[OFFER_FIELD_IDX.QUANTITY] = max(
            min(my_needs, quantity_issue.max_value), quantity_issue.min_value
        )
        offer[OFFER_FIELD_IDX.TIME] = self.awi.current_step
        if self._is_selling(ami):
            offer[OFFER_FIELD_IDX.UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[OFFER_FIELD_IDX.UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

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
            return int(mn + th * (mx - mn))
        else:
            return int(mx - th * (mx - mn))

    def _price_range(self, ami):
        """Limits the price by the best price received"""
        mn = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE].min_value
        mx = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE].max_value
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
        return int(mn), int(mx)

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e

    def propose_greedy(self, negotiator_id: str, state):
        # find the absolute best offer for me. This will most likely has an
        # unrealistic price
        offer = self.best_offer(negotiator_id)

        # if there are no best offers, just return None to end the negotiation
        if not offer:
            return None

        # over-write the unit price in the best offer with a good-enough price
        offer = list(offer)
        offer[OFFER_FIELD_IDX.UNIT_PRICE] = self._find_good_price(
            self.get_ami(negotiator_id), state
        )
        return (offer[OFFER_FIELD_IDX.UNIT_PRICE], offer[OFFER_FIELD_IDX.QUANTITY])

    def respond_greedy(self, negotiator_id, state, offer):
        # find the quantity I still need and end negotiation if I need nothing more
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return "end"

        # reject any offers with quantities above my needs
        response = (
            "acc"
            if offer[OFFER_FIELD_IDX.QUANTITY] <= my_needs
            else self.propose_greedy(negotiator_id, state)
        )
        if response != "acc":
            return response

        # reject offers with prices that are deemed NOT good-enough
        ami = self.get_ami(negotiator_id)
        response = (
            response
            if self._is_good_price(ami, state, offer[OFFER_FIELD_IDX.UNIT_PRICE])
            else self.propose_greedy(negotiator_id, state)
        )

        # update my current best price to use for limiting concession in other
        # negotiations
        up = offer[OFFER_FIELD_IDX.UNIT_PRICE]
        if self._is_selling(ami):
            self._best_selling = max(up, self._best_selling)
            partner = ami.annotation["buyer"]
            self._best_opp_selling[partner] = self._best_selling
        else:
            self._best_buying = min(up, self._best_buying)
            partner = ami.annotation["seller"]
            self._best_opp_buying[partner] = self._best_buying
        return response


# from myagent import *

# This is in order to run tournaments
'''
from Learningagent import LearningAgent
if not TO_SUBMISSION:
    #from agents_pool import *  # SimpleAgent, BetterAgent, LearningAgent, AdaptiveAgent

    def run(
        competition="oneshot",
        reveal_names=True,
        n_steps=125,  # 150,
        n_configs=1,  # ,
        # controlled_env=True
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
            competitors = [
                # HunterAgent,
                QlAgent,
                # QlAgent,
                LearningAgent,
                #GreedyOneShotAgent,
                # RandomOneShotAgent,
                # SyncRandomOneShotAgent,
                # SimpleAgent,
                # BetterAgent,
                # AdaptiveAgent,
                # LearningAgent
            ]
        else:
            from scml.scml2020.agents import (
                BuyCheapSellExpensiveAgent,
                DecentralizingAgent,
            )

            competitors = [
                # HunterAgent,
                DecentralizingAgent,
                BuyCheapSellExpensiveAgent,
            ]

        start = time.perf_counter()

        # if controlled_env:
        #     if competition == "oneshot":
        #         world = SCML2020OneShotWorld(
        #             **SCML2020OneShotWorld.generate(
        #                 agent_types = competitors,
        #                 n_agents_per_process=1,
        #                 n_processes=2,
        #                 n_steps=n_steps,
        #                 construct_graphs=True,
        #                 compact=True
        #             )
        #         )

        #         world.draw(what=["contracts-concluded"])
        #         plt.show()
        #         world.run()
        #         plt.show()
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
            # parallelism="serial",
            compact=True,
            # random_agent_types=False,
            # max_worlds_per_config=1,
            # n_runs_per_world=1,
            # min_factories_per_level=1
            disable_agent_printing=False,
        )
        # just make names shorter
        results.total_scores.agent_type = results.total_scores.agent_type.str.split(
            "."
        ).str[-1]
        # display results
        pass # print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
        pass # print(f"Finished in {humanize_time(time.perf_counter() - start)}")
        # scores_df = results.total_scores
        # max_score = scores_df['score'].max()
        # final_score = scores_df[scores_df['agent_type']=='HunterAgent']['score'].values[0]
        # place = scores_df[scores_df['agent_type']=='HunterAgent']['score'].index[0]
        # values = [args.start_price_type,args.price_delta_down,args.price_delta_up,args.profit_epsilon,
        #         args.acceptance_price_th,args.acceptance_quantity_th,args.cutoff_rate,args.cutoff_precentile,
        #         args.cutoff_stop_amount,final_score-max_score]#place,final_score]
        # with open(r'parametrs_scores.csv','a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(values)
        # print(f"Finished in {humanize_time(time.perf_counter() - start)}")

    # if __name__ == "__main__":
    import sys

    #run("oneshot")
    #run(sys.argv[1] if len(sys.argv) > 1 else "oneshot")
'''
