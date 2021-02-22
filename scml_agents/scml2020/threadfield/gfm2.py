"""
**Submitted to ANAC 2020 SCML**
*Authors* type-your-team-member-names-with-their-emails here


This code is free to use or update given that proper attribution is given to 
the authors and the ANAC 2020 SCML. 

This module implements a factory manager for the SCM 2020 league of ANAC 2019 
competition. This version will not use subcomponents. Please refer to the 
[game description](http://www.yasserm.com/scml/scml2020.pdf) for all the 
callbacks and subcomponents available.

Your agent can learn about the state of the world and itself by accessing 
properties in the AWI it has. For example:

- The number of simulation steps (days): self.awi.n_steps  
- The current step (day): self.awi.current_steps
- The factory state: self.awi.state
- Availability for producton: self.awi.available_for_production


Your agent can act in the world by calling methods in the AWI it has. 
For example:

- *self.awi.request_negotiation(...)*  # requests a negotiation with one partner
- *self.awi.request_negotiations(...)* # requests a set of negotiations

 
You can access the full list of these capabilities on the documentation.

- For properties/methods available only to SCM agents, check the list 
  [here](https://scml.readthedocs.io/en/latest/api/scml.scml2020.AWI.html)

"""

# required for running the test tournament
import time

# required for typing
from typing import Any, Dict, List, Optional, Type, Tuple

import numpy as np
import copy
from itertools import chain, combinations
from random import randint, sample
import math

from negmas import (
    AgentMechanismInterface,
    Breach,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    UtilityFunction,
    SAONegotiator,
    AspirationNegotiator,
    UtilityValue,
    Outcome,
)
from negmas.helpers import humanize_time
from scml.scml2020 import SCML2020Agent

# required for development
from scml.scml2020.agents import BuyCheapSellExpensiveAgent, DecentralizingAgent
from scml.scml2020.utils import anac2020_collusion, anac2020_std
from scml.scml2020 import Failure, AWI
from tabulate import tabulate

__all__ = ["GreedyFactoryManager2"]


class GFM2(SCML2020Agent):
    """
    This is the only class you *need* to implement. The current skeleton has a 
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by 
    calling methods in the agent-world-interface instantiated as `self.awi` 
    in your agent. See the documentation for more details

    """

    # ~~~~~ REMOVE BEFORE FLIGHT ~~~~~
    # @property
    # def awi(self) -> AWI:
    #     """Gets the Agent-world interface."""
    #     return self._awi
    #
    # @awi.setter
    # def awi(self, awi: AWI):
    #     """Sets the Agent-world interface. Should only be called by the world."""
    #     self._awi = awi
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        """Called once after the agent-world interface is initialized"""

        print_log("my_input_product = " + str(self.awi.my_input_product))
        print_log("balance = " + str(self.awi.state.balance))
        print_log("inventory = " + str(self.awi.state.inventory))

        print_log(
            "catalog_in = " + str(self.awi.catalog_prices[self.awi.my_input_product])
        )
        print_log(
            "catalog_out = " + str(self.awi.catalog_prices[self.awi.my_output_product])
        )
        print_log(
            "production cost = "
            + str(self.awi.profile.costs[0, self.awi.my_input_product])
        )

        print_log(self.awi.state.commands)

        # ラインと在庫の余剰予測リスト
        self.line_cap = [self.awi.n_lines for _ in range(self.awi.n_steps)]
        self.stock_cap = [0 for _ in range(self.awi.n_steps)]

        print_log(self.line_cap)
        print_log(self.stock_cap)

    def step(self):
        """Called at every production step by the world"""

        # 在庫予測を実在庫に合わせる
        for i in range(self.awi.current_step):
            self.stock_cap[i] = 0
        self.stock_cap[self.awi.current_step] = self.awi.state.inventory[
            self.awi.my_output_product
        ]

        print_log("balance = " + str(self.awi.state.balance))
        print_log("inventory = " + str(self.awi.state.inventory))
        print_log(self.line_cap)
        print_log(self.stock_cap)

        # 在庫がある限り生産する (SupplyDrivenProductionStrategyより)
        commands = -1 * np.ones(self.awi.n_lines, dtype=int)
        inputs = min(self.awi.state.inventory[self.awi.my_input_product], len(commands))
        commands[:inputs] = self.awi.my_input_product
        commands[inputs:] = -1
        self.awi.set_commands(commands)

        # Request all partners to negotiate
        # input products
        if self.awi.current_step + 1 <= self.awi.n_steps - (
            len(self.awi.state.inventory) - self.awi.my_input_product
        ):
            qvalues = (1, int(self.awi.n_lines * 0.5))
            tvalues = (
                self.awi.current_step + 1,
                min(
                    self.awi.current_step + 6,
                    self.awi.n_steps
                    - (len(self.awi.state.inventory) - self.awi.my_input_product),
                ),
            )
            uvalues = (1, self.awi.catalog_prices[self.awi.my_input_product])
            print_log(f"tvalues={tvalues}")
            issues = [
                Issue(qvalues, name="quantity"),
                Issue(tvalues, name="time"),
                Issue(uvalues, name="uvalues"),
            ]
            for _ in range(5):
                for partner in self.awi.my_suppliers:
                    self.awi.request_negotiation(
                        is_buy=True,
                        product=self.awi.my_input_product,
                        quantity=qvalues,
                        unit_price=uvalues,
                        time=tvalues,
                        partner=partner,
                        negotiator=self.negotiator(False, issues=issues),
                    )

        # output products
        if self.awi.current_step + 3 <= self.awi.n_steps - 1:
            qvalues = (1, int(self.awi.n_lines * 0.5))
            tvalues = (self.awi.current_step + 3, self.awi.n_steps - 1)
            uvalues = (
                max(
                    int(self.awi.catalog_prices[self.awi.my_output_product] * 1.7),
                    self.awi.catalog_prices[self.awi.my_input_product]
                    + self.awi.profile.costs[0, self.awi.my_input_product] * 3,
                ),
                self.awi.catalog_prices[self.awi.my_output_product] * 4,
            )
            issues = [
                Issue(qvalues, name="quantity"),
                Issue(tvalues, name="time"),
                Issue(uvalues, name="uvalues"),
            ]
            for _ in range(5):
                for partner in self.awi.my_consumers:
                    self.awi.request_negotiation(
                        is_buy=False,
                        product=self.awi.my_output_product,
                        quantity=qvalues,
                        unit_price=uvalues,
                        time=tvalues,
                        partner=partner,
                        negotiator=self.negotiator(True, issues=issues),
                    )

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        """Called whenever an agent requests a negotiation with you.
        Return either a negotiator to accept or None (default) to reject it"""

        # todo: 応じられるものだけ応じるようにしたいけど今は全部応じる
        # return self.negotiator(annotation["seller"] == self.id, issues=issues)
        return None

    def create_ufun(self, is_seller: bool, issues=None) -> UtilityFunction:
        """Creates a utility function"""
        return MySpecialUtilityFunction(
            is_seller=is_seller, issues=issues, awi=self.awi
        )

    # IndependentNegotiationsManagerより
    def negotiator(self, is_seller: bool, issues=None) -> Optional[SAONegotiator]:
        """Creates a negotiator"""

        if issues is None or not Issue.enumerate(issues, astype=tuple):
            return None

        return AspirationNegotiator(
            ufun=self.create_ufun(is_seller=is_seller, issues=issues),
            assume_normalized=True,
        )

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without
        agreement"""

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        """Called when a negotiation the agent is a party of ends with
        agreement"""

    # =============================
    # Contract Control and Feedback
    # =============================

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Called to ask you to sign all contracts that were concluded in
        one step (day)"""

        # Choose the good set of contracts
        buy_conts = [
            contract
            for contract in contracts
            if contract.annotation["seller"] != self.id
        ]
        sell_conts = [
            contract
            for contract in contracts
            if contract.annotation["seller"] == self.id
        ]

        print_log(
            "Got "
            + str(len(buy_conts))
            + " buy contracts and "
            + str(len(sell_conts))
            + " sell contracts on day "
            + str(self.awi.current_step)
        )
        print_log("=== BUY ===")
        for c in buy_conts:
            print_log(
                f"quantity = {c.agreement['quantity']}, time = {c.agreement['time']}, price = {c.agreement['unit_price']}"
            )
        print_log("=== SELL ===")
        for c in sell_conts:
            print_log(
                f"quantity = {c.agreement['quantity']}, time = {c.agreement['time']}, price = {c.agreement['unit_price']}"
            )

        # # TODO: ココの選び方は要検討
        # # Make sets of contracts to evaluate
        # if len(buy_conts) + len(sell_conts) <= 10:
        #     cont_sets = [(b, s) for b in power_set(buy_conts) for s in power_set(sell_conts)]
        # else:
        #     cont_sets = [((), ())]
        #     if len(buy_conts) < 5:
        #         max_b = len(buy_conts)
        #         max_s = 10 - len(buy_conts)
        #     elif len(sell_conts) < 5:
        #         max_b = 10 - len(sell_conts)
        #         max_s = len(sell_conts)
        #     else:
        #         max_b = 5
        #         max_s = 5
        #
        #     for _ in range(1000):
        #         n_b = randint(0, max_b)
        #         n_s = randint(0, max_s)
        #         cont_sets.append((sample(buy_conts, n_b), sample(sell_conts, n_s)))
        #
        # good_cont_set = max(cont_sets, key=self.eval_set)
        # good_cont_set = [c.id for c in good_cont_set[0]] + [c.id for c in good_cont_set[1]]

        good_cont_set = []
        est_line_cap = self.line_cap
        est_stock_cap = self.stock_cap

        # for contract in buy_conts:
        for contract in sorted(
            buy_conts,
            key=lambda x: (
                x.agreement["time"],
                x.agreement["unit_price"],
                -1 * x.agreement["quantity"],
            ),
        ):
            new_line_cap, new_stock_cap = self.schedule_process(
                est_line_cap,
                est_stock_cap,
                contract.agreement["time"],
                contract.agreement["quantity"],
            )
            if new_line_cap is not None:
                good_cont_set.append(contract.id)
                est_line_cap = new_line_cap
                est_stock_cap = new_stock_cap

        # for contract in sell_conts:
        for contract in sorted(
            sell_conts,
            key=lambda x: (
                x.agreement["time"],
                -1 * x.agreement["unit_price"],
                -1 * x.agreement["quantity"],
            ),
        ):
            new_stock_cap, num_of_shortage = self.pick_stock(
                est_stock_cap,
                contract.agreement["time"],
                contract.agreement["quantity"],
            )
            if num_of_shortage is None:
                good_cont_set.append(contract.id)
                est_stock_cap = new_stock_cap

        print_log("GCS = " + str(good_cont_set))
        return [
            "Hello world!!" if contract.id in good_cont_set else None
            for contract in contracts
        ]

    def eval_set(self, cont_set) -> float:
        est_line_cap, est_stock_cap, profit = self.simulate_set(*cont_set)

        # 生産ラインのキャパオーバー
        if est_line_cap is None:
            return -1000000.0

        if min(est_stock_cap) < 0:
            return sum([s for s in est_stock_cap if s < 0])

        # 各要素に対する評価値 [0,1]
        # 生産ライン
        l_nml = (self.awi.n_lines * self.awi.n_steps - sum(est_line_cap)) / (
            self.awi.n_lines * self.awi.n_steps
        )
        # 製品余剰
        s_nml = gauss(
            sum(est_stock_cap), mu=2 * self.awi.n_lines, sigma=self.awi.n_lines
        )
        # 利益
        p_nml = 1 / (
            1
            + math.exp(
                -(profit - 2 * self.awi.profile.costs[0, self.awi.my_input_product])
                / self.awi.catalog_prices[self.awi.my_input_product]
            )
        )

        w_p = 0.2 + 0.8 * self.awi.current_step / self.awi.n_steps
        # w_p = 0.3

        # return w_p * p_nml + (1.0 - w_p) * s_nml
        return w_p * p_nml + (1.0 - w_p) * l_nml

    def simulate_set(
        self, buy_conts: List[Contract], sell_conts: List[Contract]
    ) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[int]]:
        """ returns (estimated_line_cap, profit, estimated_stock_cap) """

        profit = 0
        est_line_cap = self.line_cap
        est_stock_cap = self.stock_cap

        for contract in buy_conts:
            profit -= (
                contract.agreement["unit_price"]
                + self.awi.profile.costs[0, self.awi.my_input_product]
            ) * contract.agreement["quantity"]
            est_line_cap, est_stock_cap = self.schedule_process(
                est_line_cap,
                est_stock_cap,
                contract.agreement["time"],
                contract.agreement["quantity"],
            )
            if est_line_cap is None:
                return None, None, None

        for contract in sell_conts:
            profit += contract.agreement["unit_price"] * contract.agreement["quantity"]
            est_stock_cap, num_of_shortage = self.pick_stock(
                est_stock_cap,
                contract.agreement["time"],
                contract.agreement["quantity"],
            )
            if num_of_shortage:
                est_stock_cap[contract.agreement["time"]] -= num_of_shortage

        return est_line_cap, est_stock_cap, profit

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        """Called to inform you about the final status of all contracts in
        a step (day)"""

        # Update the estimation of line capacity and stock
        buy_conts = [
            contract for contract in signed if contract.annotation["seller"] != self.id
        ]
        sell_conts = [
            contract for contract in signed if contract.annotation["seller"] == self.id
        ]

        print_log(
            "Finalized "
            + str(len(buy_conts))
            + " buy contracts and "
            + str(len(sell_conts))
            + " sell contracts on day "
            + str(self.awi.current_step)
        )
        print_log("=== BUY ===")
        for c in buy_conts:
            print_log(
                f"quantity = {c.agreement['quantity']}, time = {c.agreement['time']}, price = {c.agreement['unit_price']}"
            )
        print_log("=== SELL ===")
        for c in sell_conts:
            print_log(
                f"quantity = {c.agreement['quantity']}, time = {c.agreement['time']}, price = {c.agreement['unit_price']}"
            )

        for contract in buy_conts:
            new_line_cap, new_stock_cap = self.schedule_process(
                self.line_cap,
                self.stock_cap,
                contract.agreement["time"],
                contract.agreement["quantity"],
            )
            if new_line_cap is None:
                print_log(
                    "######################### Never Printed #########################"
                )
                new_line_cap = [-1 for _ in range(self.awi.n_steps)]
                new_stock_cap = [-1 for _ in range(self.awi.n_steps)]
            self.line_cap = new_line_cap
            self.stock_cap = new_stock_cap

        for contract in sell_conts:
            new_stock_cap, num_of_shortage = self.pick_stock(
                self.stock_cap,
                contract.agreement["time"],
                contract.agreement["quantity"],
            )
            if num_of_shortage:
                new_stock_cap[contract.agreement["time"]] -= num_of_shortage
            self.stock_cap = new_stock_cap

    # I think NegMas has a clever scheduler, but it's mendokusai to learn how to use...
    def schedule_process(
        self, line_cap: List[int], stock_cap: List[int], input_date: int, input_num: int
    ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        line_cap = copy.copy(line_cap)
        stock_cap = copy.copy(stock_cap)

        if input_date >= self.awi.n_steps - 1:
            return None, None

        if line_cap[input_date] >= input_num:
            line_cap[input_date] -= input_num
            stock_cap[input_date + 1] += input_num
            return line_cap, self.cover_shortage(stock_cap)

        input_num -= line_cap[input_date]
        stock_cap[input_date + 1] += line_cap[input_date]
        line_cap[input_date] = 0
        input_date += 1

        return self.schedule_process(
            line_cap, self.cover_shortage(stock_cap), input_date, input_num
        )

    def pick_stock(
        self, stock_cap: List[int], output_date: int, output_num: int
    ) -> Tuple[
        List[int], Optional[int]
    ]:  # returns (estimated_stock_cap, num_of_shortage)
        stock_cap = copy.copy(stock_cap)

        if output_date < 0:
            return stock_cap, output_num

        if stock_cap[output_date] >= output_num:
            stock_cap[output_date] -= output_num
            return stock_cap, None

        if stock_cap[output_date] > 0:
            output_num -= stock_cap[output_date]
            stock_cap[output_date] = 0
        output_date -= 1
        return self.pick_stock(stock_cap, output_date, output_num)

    def cover_shortage(
        self, stock_cap: List[int]
    ) -> List[int]:  # returns estimated_stock_cap
        stock_cap = copy.copy(stock_cap)

        for i in range(len(stock_cap)):
            if stock_cap[i] < 0:
                for j in range(i - 1, -1, -1):
                    if stock_cap[j] > 0:
                        if stock_cap[i] + stock_cap[j] >= 0:
                            stock_cap[j] += stock_cap[i]
                            stock_cap[i] = 0
                            break
                        stock_cap[i] += stock_cap[j]
                        stock_cap[j] = 0

        return stock_cap

    def on_contract_executed(self, contract: Contract) -> None:
        """Called when a contract executes successfully and fully"""

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        """Called when a breach occur. In 2020, there will be no resolution
        (i.e. resoluion is None)"""
        print_log("!!!!!BREACHED!!!!!")
        print_log(contract)
        print_log(breaches)
        print_log("!!!!!!!!!!!!!!!!!!")

    # ====================
    # Production Callbacks
    # ====================

    def confirm_production(
        self, commands: np.ndarray, balance: int, inventory: np.ndarray
    ) -> np.ndarray:
        """
        Called just before production starts at every step allowing the
        agent to change what is to be produced in its factory on that step.
        """
        return commands

    def on_failures(self, failures: List[Failure]) -> None:
        """Called when production fails. If you are careful in
        what you order in `confirm_production`, you should never see that."""

    # ==========================
    # Callback about Bankruptcy
    # ==========================

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: int,
        compensation_money: int,
    ) -> None:
        """Called whenever any agent goes bankrupt. It informs you about changes
        in future contracts you have with you (if any)."""


class MySpecialUtilityFunction(UtilityFunction):
    def __init__(
        self,
        is_seller: bool,
        name: Optional[str] = None,
        reserved_value: UtilityValue = float("-inf"),
        outcome_type: Optional[Type] = None,
        issue_names: Optional[List[str]] = None,
        issues: List["Issue"] = None,
        ami: AgentMechanismInterface = None,
        awi: AWI = None,
    ) -> None:
        self.is_seller = is_seller
        self.awi = awi

        super().__init__(
            name=name,
            reserved_value=reserved_value,
            outcome_type=outcome_type,
            issue_names=issue_names,
            issues=issues,
            ami=ami,
        )

    def eval(self, offer: "Outcome") -> UtilityValue:
        price, quantity, date = offer

        if self.is_seller:
            price_nml = (
                price
                - self.awi.catalog_prices[self.awi.my_output_product]
                + self.awi.profile.costs[0, self.awi.my_input_product]
            ) / (4 * self.awi.profile.costs[0, self.awi.my_input_product])
            quantity_nml = (
                1.0
                if quantity < self.awi.n_lines
                else 2.0 - quantity / self.awi.n_lines
            )
            date_nml = (
                1.0
                - (date - self.awi.current_step)
                / (self.awi.n_steps - self.awi.current_step)
                if date > self.awi.current_step
                else -1.0
            )

            util = 0.5 * price_nml + 0.3 * quantity_nml + 0.2 * date_nml

        else:
            price_nml = (
                self.awi.catalog_prices[self.awi.my_output_product]
                - price
                + self.awi.profile.costs[0, self.awi.my_input_product]
            ) / (4 * self.awi.profile.costs[0, self.awi.my_input_product])
            quantity_nml = (
                1.0
                if quantity < self.awi.n_lines
                else 2.0 - quantity / self.awi.n_lines
            )
            date_nml = (
                (date - self.awi.current_step)
                / (self.awi.n_steps - self.awi.current_step)
                if date > self.awi.current_step
                else -1.0
            )

            util = 0.5 * price_nml + 0.3 * quantity_nml + 0.2 * date_nml

        # print("\nis_seller = " + str(self.is_seller))
        # print("offer = " + str(offer))
        # print("util = " + str(util))

        return util

    def xml(self, issues: List[Issue]) -> str:
        return "Sorry, xml() Not supported"


class GreedyFactoryManager2(GFM2):
    pass


def power_set(iterable):
    """power_set([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def gauss(x, a=1, mu=0, sigma=1):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


debug = True
debug = False


def print_log(s):
    if debug:
        print(s)


def run(
    competition="std",
    reveal_names=True,
    n_steps=20,
    n_configs=2,
    max_n_worlds_per_config=None,
    n_runs_per_world=1,
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
        n_runs_per_world: How many times will each world simulation be run.

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value

    """
    competitors = [MyAgent, DecentralizingAgent, BuyCheapSellExpensiveAgent]
    start = time.perf_counter()
    if competition == "std":
        results = anac2020_std(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
            n_runs_per_world=n_runs_per_world,
            parallelism="serial",
        )
    elif competition == "collusion":
        results = anac2020_collusion(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
            n_runs_per_world=n_runs_per_world,
        )
    else:
        raise ValueError(f"Unknown competition type {competition}")
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    run()
