"""
An implementation of SCML Oneshot game's agent
Author: Junjie Jordi Chen
supervised by Dr. Dave de Jonge
Last modified: March-04-2023
negmas version: 0.9.8
scml version: 0.5.6
"""
import time

import matplotlib.pyplot as plt
import pandas as pd
from negmas import ResponseType
from negmas.common import AgentMechanismInterface, MechanismState
from negmas.situated import Contract
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agent import OneShotAgent

from scml_agents.scml2021.oneshot.team_73.oneshot_agents import (
    AdaptiveAgent,
    BetterAgent,
)

__all__ = ["TwoOneFive"]


class TwoOneFive(OneShotAgent):
    def init(self) -> None:
        """Called once after the agent-world interface is initialized"""
        # secured quantity value of each step
        self.quantity_secured = 0
        self.my_unitprice = -1
        self.partners = []
        self.actual_price = []
        self.cur_neg_partners = []

        # record the partner/competitor ratio, if there is many partners but less competitors we can have a greedier strategy when negotiating and vise versa
        # the more the value, the greeder the negotiation strategy

        self.consumers_1 = self.awi.all_consumers[1].copy()
        self.consumers_0 = self.awi.all_consumers[0].copy()

        self.p_c_ratio = (
            len(self.consumers_0) / len(self.consumers_1)
            if self.awi.level == 1
            else len(self.consumers_1) / len(self.consumers_0)
        )

        if self.awi.level == 0:
            self.cur_neg_partners = self.awi.all_consumers[1].copy()
        else:
            self.cur_neg_partners = self.awi.all_consumers[0].copy()

        # for collecting negotiation result information
        # key: partner; value: (success_or_not: bool,
        # negotiation_steps: int,
        # accepted/end_neg by: int (0: accepted/end_neg by me, 1: accepted/end_neg by partner, 2: time_over)
        self.neg_results = {}
        # record past exo_quantity and negotiation-reached quantity history
        self.exo_quantity_history = [-1 for _ in range(self.awi.n_steps)]
        self.neg_quantity_history = [-1 for _ in range(self.awi.n_steps)]

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""
        pass

    def step(self) -> None:
        """Called at at the END of every production step (day)"""
        # sleep 0.1ms to wait on_negotiation_success/failure() functions to finish
        time.sleep(0.0001)

        # record exo_quantity and negotiation-reached quantity
        self.exo_quantity_history[self.awi.current_step] = (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
        )
        self.neg_quantity_history[self.awi.current_step] = self.quantity_secured
        # reset quantity_secured value for next step

        self.quantity_secured = 0

    def propose(self, negotiator_id: str, state: MechanismState):
        """Called when the agent is asking to propose in one negotiation"""
        # collect info
        if len(self.cur_neg_partners) == 0:
            return None
        if len(self.awi.all_consumers[0]) == 0:
            return None
        if len(self.awi.all_consumers[1]) == 0:
            return None

        nmi = self.get_nmi(negotiator_id)
        partner = (
            nmi.annotation["buyer"] if self.awi.level == 0 else nmi.annotation["seller"]
        )
        offer = [-1] * 3
        offer[TIME] = self.awi.current_step
        isSelling = self._is_selling(nmi)
        time_th = self.p_c_ratio / 2
        if isSelling:
            if self.awi.relative_time <= time_th:
                offer[UNIT_PRICE] = nmi.issues[UNIT_PRICE].max_value
            else:
                offer[UNIT_PRICE] = nmi.issues[UNIT_PRICE].min_value
        else:
            if self.awi.relative_time <= time_th:
                offer[UNIT_PRICE] = nmi.issues[UNIT_PRICE].min_value
            else:
                offer[UNIT_PRICE] = nmi.issues[UNIT_PRICE].max_value

        self.my_unitprice = offer[UNIT_PRICE]
        offer[QUANTITY] = self.calculate_quant()
        if offer[QUANTITY] == 0:
            return None
        return tuple(offer)

    def calculate_quant(self):
        """Finds a good-enough quantity"""
        mx_q = self._required_quantity() / len(self.cur_neg_partners)
        average_quant = (mx_q + self._required_quantity()) / 2
        return average_quant

    def respond(self, negotiator_id: str, state, source) -> ResponseType:
        """Called when the agent is asked to respond to an offer"""
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        # collect info
        nmi = self.get_nmi(negotiator_id)
        offer_quantity = offer[QUANTITY]
        offer_unit_price = offer[UNIT_PRICE]

        # record required_quantity
        # when respond with ACCEPT re-check the required_quantity
        required_quantity = self._required_quantity()
        if required_quantity <= 0:
            return ResponseType.REJECT_OFFER

        if self.awi.level == 0:
            if (
                offer_quantity <= required_quantity
                and offer_unit_price >= self.my_unitprice
            ):
                # quantity and price meet requirement
                return ResponseType.ACCEPT_OFFER
            elif offer_unit_price < self.my_unitprice:
                # too cheap
                return ResponseType.REJECT_OFFER
            else:
                return ResponseType.REJECT_OFFER

        else:  # self.awi.level == 1
            if (
                offer_quantity <= required_quantity
                and offer_unit_price <= self.my_unitprice
            ):
                # quantity and price meet requirement
                return ResponseType.ACCEPT_OFFER
            elif offer_unit_price > self.my_unitprice:
                # too expensive
                return ResponseType.REJECT_OFFER
            else:
                return ResponseType.REJECT_OFFER
        raise Exception("ERROR NO RESPONSE FROM THE NEGOTIATION")

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        if self.awi.all_consumers[1] == 0:
            return None
        if self.awi.all_consumers[0] == 0:
            return None
        """Called when a negotiation the agent is a party of ends with agreement"""
        # update required_quantity
        quantity = contract.agreement["quantity"]
        self.quantity_secured += quantity

        # record data
        partner = (
            mechanism.annotation["buyer"]
            if self.awi.level == 0
            else mechanism.annotation["seller"]
        )
        no_of_step = len(mechanism["_mechanism"]._history)
        if (
            partner
            == mechanism["_mechanism"]._history[no_of_step - 1].current_proposer_agent
        ):
            accepted_by = 1
        else:
            accepted_by = 0

        self.neg_results[partner] = (True, no_of_step, accepted_by)

        if partner in self.cur_neg_partners:
            if len(self.cur_neg_partners) > 1:
                self.cur_neg_partners.remove(partner)

        if self.awi.level == 0:
            if partner in self.consumers_1:
                if len(self.consumers_1) > 1:
                    self.consumers_1.remove(partner)
                    self.p_c_ratio = len(self.consumers_1) / len(self.consumers_0)
        else:
            if partner in self.consumers_0:
                if len(self.consumers_0) > 1:
                    self.consumers_0.remove(partner)
                    self.p_c_ratio = len(self.consumers_0) / len(self.consumers_1)

    def on_negotiation_failure(
        self,
        partners,
        annotation,
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        if self.awi.all_consumers[0] == 0:
            return None
        if self.awi.all_consumers[1] == 0:
            return None
        """Called when a negotiation the agent is a party of ends without agreement"""

        # record data
        partner = (
            mechanism.annotation["buyer"]
            if self.awi.level == 0
            else mechanism.annotation["seller"]
        )
        no_of_step = len(mechanism["_mechanism"]._history)

        if (
            partner
            == mechanism["_mechanism"]._history[no_of_step - 1].current_proposer_agent
        ):
            rejected_by = 1
        else:
            rejected_by = 0

        self.neg_results[partner] = (False, no_of_step, rejected_by)
        if partner in self.cur_neg_partners:
            if len(self.cur_neg_partners) > 1:
                self.cur_neg_partners.remove(partner)

        if self.awi.level == 0:
            if partner in self.consumers_1:
                if len(self.consumers_1) > 1 and len(self.consumers_0) > 1:
                    self.consumers_1.remove(partner)
                    self.p_c_ratio = len(self.consumers_1) / len(self.consumers_0)
        else:
            if partner in self.consumers_0:
                if len(self.consumers_0) > 1 and len(self.consumers_1) > 1:
                    self.consumers_0.remove(partner)
                    self.p_c_ratio = len(self.consumers_0) / len(self.consumers_1)

    def _required_quantity(self) -> int:
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.quantity_secured
        )

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product


import random
from collections import defaultdict
from pprint import pprint

from negmas import ResponseType
from scml.oneshot import *
from scml.scml2020 import is_system_agent


def try_agent(agent_type, n_processes=2, draw=True):
    """Runs an agent in a world simulation against a randomly behaving agent"""
    return try_agents([BetterAgent, AdaptiveAgent, agent_type], n_processes, draw=draw)


# EVEAgent,Agent112,
def try_agents(agent_types, n_processes=2, n_trials=1, draw=True, agent_params=None):
    """
    Runs a simulation with the given agent_types, and n_processes n_trial times.
    Optionally also draws a graph showing what happened
    """
    type_scores = defaultdict(float)
    counts = defaultdict(int)
    agent_scores = dict()
    for _ in range(n_trials):
        p = (
            n_processes
            if isinstance(n_processes, int)
            else random.randint(*n_processes)
        )
        world = SCML2023OneShotWorld(
            **SCML2023OneShotWorld.generate(
                agent_types,
                agent_params=agent_params,
                n_steps=5,
                n_processes=p,
                random_agent_types=True,
            ),
            construct_graphs=True,
        )
        world.run()

        all_scores = world.scores()
        for aid, agent in world.agents.items():
            if is_system_agent(aid):
                continue
            key = aid if n_trials == 1 else f"{aid}@{world.id[:4]}"
            agent_scores[key] = (
                agent.type_name.split(":")[-1].split(".")[-1],
                all_scores[aid],
                "(bankrupt)" if world.is_bankrupt[aid] else "",
            )
        for aid, agent in world.agents.items():
            if is_system_agent(aid):
                continue
            type_ = agent.type_name.split(":")[-1].split(".")[-1]
            type_scores[type_] += all_scores[aid]
            counts[type_] += 1
    type_scores = {k: v / counts[k] if counts[k] else v for k, v in type_scores.items()}
    if draw:
        world.draw(
            what=["contracts-concluded"],
            steps=(0, world.n_steps - 1),
            together=True,
            ncols=1,
            figsize=(20, 20),
        )
        plt.show()

    return world, agent_scores, type_scores


def analyze_contracts(world):
    """
    Analyzes the contracts signed in the given world
    """

    data = pd.DataFrame.from_records(world.saved_contracts)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    return data.groupby(["seller_name", "buyer_name"])[
        ["quantity", "unit_price"]
    ].mean()


def print_agent_scores(agent_scores):
    """
    Prints scores of individiual agent instances
    """
    for aid, (type_, score, bankrupt) in agent_scores.items():
        pass  # print(f"Agent {aid} of type {type_} has a final score of {score} {bankrupt}")


def print_type_scores(type_scores):
    """Prints scores of agent types"""
    pprint(sorted(tuple(type_scores.items()), key=lambda x: -x[1]))


if __name__ == "__main__":
    world, ascores, tscores = try_agent(TwoOneFive)
    pass  # print(analyze_contracts(world))
    contracts = world.contracts_df
    signed = contracts.loc[contracts.signed_at >= 0, :]
    fields = [
        "seller_name",
        "buyer_name",
        "quantity",
        "unit_price",
        "signed_at",
        "executed",
        "breached",
        "nullified",
        "erred",
    ]
    signed[fields].sort_values(["quantity", "unit_price"], ascending=False).head(10)
    middle_cols = signed[fields].iloc[:, 0:5]
    pass  # print(middle_cols)
    print_type_scores(tscores)
