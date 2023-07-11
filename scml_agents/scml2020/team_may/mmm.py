from typing import Any, Dict, List, Optional

import numpy as np
from negmas import (
    AgentMechanismInterface,
    Breach,
    Contract,
    Issue,
    LinearUtilityFunction,
    MechanismState,
    Negotiator,
    ResponseType,
)
from negmas.sao import RandomNegotiator, SAONegotiator

# required for development
from scml import SupplyDrivenProductionStrategy

# from negmas.helpers import humanize_time
from scml.scml2020 import Failure, SCML2020Agent
from scml.scml2020.agents import DecentralizingAgent
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE
from scml.scml2020.components import IndependentNegotiationsManager
from scml.utils import anac2020_collusion, anac2020_std

# from tabulate import tabulate

__all__ = ["MMM"]


class YesMan(RandomNegotiator):
    def respond_(self, state):
        return self.respond(state)

    def respond(self, state):
        return ResponseType.ACCEPT_OFFER


class StdAgent(
    IndependentNegotiationsManager, SupplyDrivenProductionStrategy, SCML2020Agent
):
    def init(self):
        self.n_shipments = 0
        self.n_arrivals = 0
        self.ip = self.awi.catalog_prices[self.awi.my_input_product] - 2
        self.op = self.awi.catalog_prices[self.awi.my_output_product] + 2
        self.Pbuy = 2
        self.Psell = 1
        self.buyschedule = [0 for _ in range(self.awi.n_steps)]
        self.sellschedule = [0 for _ in range(self.awi.n_steps)]
        self.inventory = [0 for _ in range(self.awi.n_steps)]
        super().init()

    def step(self):
        for i in range(1, self.awi.n_steps):
            self.inventory[i] = max(
                self.inventory[i - 1] + self.buyschedule[i - 1] - self.awi.n_lines, 0
            )
        self.update_price()
        super().step()

    def sign_all_contracts(self, contracts):
        signed = [None] * len(contracts)
        bfast = self.awi.n_steps
        buy, sell = [], []
        qb, qs = 0, 0
        for c in contracts:
            if c.annotation["buyer"] == self.id:
                buy.append(c)
                if c.agreement["time"] < bfast:
                    bfast = c.agreement["time"]
            else:
                sell.append(c)
        for d in range(5):
            for b in buy:
                q, p, t = (
                    b.agreement["quantity"],
                    b.agreement["unit_price"],
                    b.agreement["time"],
                )
                if t > self.awi.n_steps * 0.9:
                    continue
                arrivals = sum(self.buyschedule[t + 1 :])
                productivity = self.awi.n_lines * (self.awi.n_steps - t - 1)
                if self.inventory[t] + arrivals + q <= productivity:
                    if (
                        t == bfast
                        and self.inventory[t] + self.buyschedule[t] < self.awi.n_lines
                    ) or p <= self.ip + d:
                        signed[contracts.index(b)] = self.id
                        qb += q
                        if d > 0 and qb > self.Pbuy * self.awi.n_lines:
                            break
            if qb > self.Pbuy * self.awi.n_lines:
                break
        for d in range(5):
            for s in sell:
                inv = sum(self.buyschedule[: s.agreement["time"] - 1])
                shipments = sum(self.sellschedule[: s.agreement["time"]])
                if inv >= shipments + s.agreement["quantity"]:
                    if s.agreement["unit_price"] >= self.op - d:
                        signed[contracts.index(s)] = self.id
                        qs += s.agreement["quantity"]
                        if d > 0 and qs > self.Psell * self.awi.n_lines:
                            break
            if qs > self.Psell * self.awi.n_lines:
                break
        return signed

    def update_price(self):
        if self.n_arrivals > self.Pbuy * self.awi.n_lines:
            self.ip -= 1
        else:
            self.ip = min(
                self.ip + 1, self.awi.catalog_prices[self.awi.my_input_product] - 3
            )
        if self.n_shipments > self.Psell * self.awi.n_lines:
            self.op += 1
        elif self.awi.current_step < self.awi.n_steps * 0.7:
            self.op = max(
                self.op - 1, self.awi.catalog_prices[self.awi.my_output_product] + 3
            )
        else:
            self.op -= 1
        self.n_arrivals, self.n_shipments = 0, 0
        time = self.awi.current_step / self.awi.n_steps
        self.Pbuy = 2 - time**0.9
        self.Psell = 1 + time**2

    def acceptable_unit_price(self, step, sell):
        if sell:
            return self.op - 3
        return self.ip + 3

    def target_quantity(self, step, sell):
        if sell:
            return self.awi.n_lines // 2
        return self.awi.n_lines

    def create_ufun(self, is_seller, issues, outcomes):
        if is_seller:
            return LinearUtilityFunction((1, 5, 5), issues=issues, outcomes=outcomes)
        return LinearUtilityFunction((1, -5, -5), issues=issues, outcomes=outcomes)

    def on_contracts_finalized(self, signed, cancelled, rejectors):
        for c in signed:
            if (
                self.awi.current_step <= c.agreement["time"]
                and c.agreement["time"] < self.awi.n_steps
            ):
                if c.annotation["seller"] == self.id:
                    self.sellschedule[c.agreement["time"]] += c.agreement["quantity"]
                    self.n_shipments += c.agreement["quantity"]
                else:
                    self.buyschedule[c.agreement["time"]] += c.agreement["quantity"]
                    self.n_arrivals += c.agreement["quantity"]

    def on_agent_bankrupt(self, agent, contracts, quantities, compensation_money):
        for c in contracts:
            if c.annotation["buyer"] != self.id:
                self.sellschedule[c.agreement["time"]] -= c.agreement["quantity"]
            else:
                self.buyschedule[c.agreement["time"]] -= c.agreement["quantity"]


class MMM(StdAgent):
    def init(self):
        self.code = 93010
        self.score = 0.25
        self.consumer = []
        self.supplier = []
        self.gifter = False
        self.gifted = False
        self.gifterscost = 999999
        self.giftersID = ""
        self.victimscost = 0
        self.victimsID = ""
        self.mustsign = {}
        self.initbal = self.awi.state.balance
        # print(self.awi.all_suppliers)
        super().init()

    def step(self):
        # 下層全員にcodeを送る（⇒上層の把握）
        if self.awi.current_step == 0:
            super()._start_negotiations(
                product=self.awi.my_input_product,
                sell=False,
                step=1,
                qvalues=[self.code, self.code],
                uvalues=[
                    1,
                    int(np.max(self.awi.profile.costs[:, self.awi.my_input_product])),
                ],
                tvalues=[1, 1],
                partners=self.awi.my_suppliers,
            )
        # 上層の仲間にcodeを送る（⇒下層の把握）
        if self.awi.current_step == 1:
            super()._start_negotiations(
                product=self.awi.my_output_product,
                sell=True,
                step=1,
                qvalues=[self.code, self.code + len(self.consumer)],
                uvalues=[1, 1],
                tvalues=[1, 1],
                partners=self.awi.my_consumers,
            )

        if self.gifted and self.awi.state.balance + self.awi.catalog_prices[
            self.awi.my_output_product
        ] * self.awi.state.inventory[self.awi.my_output_product] > self.initbal * (
            1 + self.score
        ):
            super()._start_negotiations(
                product=self.awi.my_input_product,
                sell=False,
                step=1,
                qvalues=[self.code, self.code],
                uvalues=[self.code, self.code],
                tvalues=[1, 1],
                partners=self.awi.my_suppliers,
            )

        if self.gifter and len(self.consumer) > 1:
            qua = self.awi.state.inventory[self.awi.my_output_product] - 1
            qua = qua // (len(self.consumer) - 1)
            if qua > 0:
                for partner in list(set(self.consumer) - set(self.victimsID)):
                    self.awi.request_negotiation(
                        is_buy=False,
                        product=self.awi.my_output_product,
                        quantity=[qua, qua],
                        unit_price=[1, 1],
                        time=[self.awi.current_step + 1, self.awi.current_step + 1],
                        partner=partner,
                        negotiator=YesMan(),
                    )
                    self.mustsign[partner] = 1

        if self.consumer != [] and (
            self.awi.current_step >= self.awi.n_steps - self.awi.n_processes * 2
            or self.awi.state.balance < self.initbal // 2
        ):
            val = int(self.initbal * (1 + self.score) - self.awi.state.balance)
            if val > self.awi.catalog_prices[self.awi.my_output_product]:
                self.awi.request_negotiation(
                    is_buy=False,
                    product=self.awi.my_output_product,
                    quantity=[1, 1],
                    unit_price=[val, val],
                    time=[self.awi.current_step + 1, self.awi.current_step + 1],
                    partner=self.consumer[0],
                    negotiator=YesMan(),
                )
                self.mustsign[self.consumer[0]] = val
        super().step()

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:

        qmax, qmin = issues[QUANTITY].max_value, issues[QUANTITY].min_value
        tmax, tmin = issues[TIME].max_value, issues[TIME].min_value
        umax, umin = issues[UNIT_PRICE].max_value, issues[UNIT_PRICE].min_value
        is_seller = annotation["seller"] == self.id

        partner = annotation["buyer"] if is_seller else annotation["seller"]

        if (
            qmin == qmax
            and tmax == tmin
            and umax == umin
            and (partner in self.consumer or partner in self.supplier)
        ):
            self.mustsign[partner] = umin
            if self.consumer == []:
                self.gifted = True
            return YesMan()
        if qmin == self.code:
            if umin == self.code:
                self.gifter = False
            elif not is_seller and annotation["seller"] not in self.supplier:
                self.supplier.append(partner)
                if umax > self.victimscost:
                    self.victimscost = umax
                    self.victimsID = partner
            elif is_seller and annotation["buyer"] not in self.consumer:
                self.consumer.append(partner)
                if qmax - self.code == 0 and len(self.consumer) > 1:
                    self.gifter = True
        else:
            return super().respond_to_negotiation_request(
                initiator, issues, annotation, mechanism
            )

    def sign_all_contracts(self, contracts):
        signed = [None] * len(contracts)
        if self.gifter:
            for i, c in enumerate(contracts):
                if c.annotation["buyer"] == self.id:
                    signed[i] = super().sign_all_contracts([c])[0]
        # elif self.gifted:
        #     for i, c in enumerate(contracts):
        #         if c.annotation["seller"]==self.id:
        #             signed[i]=super().sign_all_contracts([c])[0]
        else:
            signed = super().sign_all_contracts(contracts)
        for c in contracts:
            partner = (
                c.annotation["buyer"]
                if c.annotation["buyer"] != self.id
                else c.annotation["seller"]
            )
            if partner in self.mustsign:
                if self.mustsign[partner] == c.agreement["unit_price"]:
                    del self.mustsign[partner]
                    signed[contracts.index(c)] = self.id
        return signed

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: int,
        compensation_money: int,
    ) -> None:
        if agent in self.consumer:
            self.consumer.remove(agent)
            if self.victimsID == agent:
                self.victimsID = self.consumer[0]
        super().on_agent_bankrupt(agent, contracts, quantities, compensation_money)

    # def __del__(self):
    #     print(self.awi.my_input_product,(self.awi.state.balance-self.initbal)/self.initbal,self.awi.state.inventory)
    #     # print(self.awi.state.inventory[self.awi.my_output_product], self.reserved)
    #     # print((self.awi.state.balance-self.initbal)/self.initbal)
    #     pass

    # # ================================
    # # Negotiation Control and Feedback
    # # ================================
    #
    #     """Called whenever an agent requests a negotiation with you.
    #     Return either a negotiator to accept or None (default) to reject it"""
    #
    # def on_negotiation_failure(self,
    #                            partners: List[str],
    #                            annotation: Dict[str, Any],
    #                            mechanism: AgentMechanismInterface,
    #                            state: MechanismState
    #                           ) -> None:
    #     """Called when a negotiation the agent is a party of ends without
    #     agreement"""
    #
    # def on_negotiation_success(self,
    #                            contract: Contract,
    #                            mechanism: AgentMechanismInterface) -> None:
    #     """Called when a negotiation the agent is a party of ends with
    #     agreement"""
    #
    # # =============================
    # # Contract Control and Feedback
    # # =============================
    #

    #
    #
    # def on_contract_executed(self, contract: Contract) -> None:
    #     """Called when a contract executes successfully and fully"""
    #
    # def on_contract_breached(self,
    #                          contract: Contract,
    #                          breaches: List[Breach],
    #                          resolution: Optional[Contract]
    #                         ) -> None:
    #     """Called when a breach occur. In 2020, there will be no resolution
    #     (i.e. resoluion is None)"""
    #
    # # ====================
    # # Production Callbacks
    # # ====================
    #
    # def confirm_production(self, commands: np.ndarray, balance: int,
    #                        inventory: np.ndarray) -> np.ndarray:
    #     """
    #     Called just before production starts at every step allowing the
    #     agent to change what is to be produced in its factory on that step.
    #     """
    #     return commands
    #
    # def on_failures(self, failures: List[Failure]) -> None:
    #     """Called when production fails. If you are careful in
    #     what you order in `confirm_production`, you should never see that."""
    #
    #
    # # ==========================
    # # Callback about Bankruptcy
    # # ==========================
    #
    #
