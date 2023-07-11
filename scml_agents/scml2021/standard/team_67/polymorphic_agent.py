"""
**Submitted to ANAC 2021 SCML**
*Authors* type-your-team-member-names-with-their-emails here
"""

import math

# required for running the test tournament
import time

# Libraries
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from negmas import (
    AgentMechanismInterface,
    AspirationNegotiator,
    Contract,
    Issue,
    LinearUtilityFunction,
    MappingUtilityFunction,
    Negotiator,
    SAOMetaNegotiatorController,
    SAONegotiator,
    UtilityFunction,
    make_issue,
)
from negmas.helpers import humanize_time, instantiate
from negmas.outcomes.issue_ops import enumerate_issues

# required for development
from scml.scml2020 import (
    DemandDrivenProductionStrategy,
    IndependentNegotiationsManager,
    MarketAwareBuyCheapSellExpensiveAgent,
    MarketAwarePredictionBasedTradingStrategy,
    MovingRangeNegotiationManager,
    PredictionBasedTradingStrategy,
    ProductionStrategy,
    ReactiveTradingStrategy,
    SCML2020Agent,
    StepNegotiationManager,
    SupplyDrivenProductionStrategy,
    TradeDrivenProductionStrategy,
    TradePredictionStrategy,
    TradingStrategy,
)
from scml.scml2020.agents import (
    BuyCheapSellExpensiveAgent,
    DecentralizingAgent,
    MarketAwareDecentralizingAgent,
)
from scml.scml2020.common import (
    ANY_LINE,
    NO_COMMAND,
    QUANTITY,
    TIME,
    UNIT_PRICE,
    is_system_agent,
)
from scml.utils import anac2021_collusion, anac2021_oneshot, anac2021_std
from tabulate import tabulate

import scml_agents

__all__ = ["PolymorphicAgent"]


class PrintLogger:
    def __init__(self, is_active=True):
        self.is_active = is_active
        # if self.is_active:
        #     self.log_file = open("log_out.txt", "w")
        #     self.log_file.truncate(0)

    def log(self, data):
        ...
        # if self.is_active:
        #     self.log_file.write(str(data) + "\n")


logger = PrintLogger(is_active=True)


class PolymorphicProductionStrategy(ProductionStrategy):
    def step(self):
        super().step()

        if self.awi.current_step >= self.awi.n_steps * 0.8:
            self.supply_driven_step(self)

    def on_contracts_finalized(
        self: "SCML2020Agent",
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        if self.awi.current_step < self.awi.n_steps * 0.2:
            # logger.log("trade_driven " + "step: " + str(self.awi.current_step))
            self.trade_driven_on_contracts_finalized(self, signed, cancelled, rejectors)
        elif self.awi.current_step < self.awi.n_steps * 0.8:
            # logger.log("demand_driven " + "step: " + str(self.awi.current_step))
            self.demand_driven_on_contracts_finalized(
                self, signed, cancelled, rejectors
            )
        else:
            # logger.log("supply_driven " + "step: " + str(self.awi.current_step))
            self.supply_driven_on_contracts_finalized(
                self, signed, cancelled, rejectors
            )

    @staticmethod
    def supply_driven_step(self):
        super().step()
        commands = NO_COMMAND * np.ones(self.awi.n_lines, dtype=int)
        inputs = min(self.awi.state.inventory[self.awi.my_input_product], len(commands))
        commands[:inputs] = self.awi.my_input_product
        commands[inputs:] = NO_COMMAND
        self.awi.set_commands(commands)

    @staticmethod
    def supply_driven_on_contracts_finalized(
        self: "SCML2020Agent",
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        latest = self.awi.n_steps - 2
        earliest_production = self.awi.current_step
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            if is_seller:
                continue
            step = contract.agreement["time"]
            # find the earliest time I can do anything about this contract
            if step > latest + 1 or step < earliest_production:
                continue
            # if I am a seller, I will schedule production
            input_product = contract.annotation["product"]
            steps, _ = self.awi.schedule_production(
                process=input_product,
                repeats=contract.agreement["quantity"],
                step=(step, latest),
                line=-1,
                partial_ok=True,
            )
            self.schedule_range[contract.id] = (
                min(steps) if len(steps) > 0 else -1,
                max(steps) if len(steps) > 0 else -1,
                is_seller,
            )

    @staticmethod
    def trade_driven_on_contracts_finalized(
        self: "SCML2020Agent",
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        for contract in signed:
            if contract.annotation["caller"] == self.id:
                continue
            is_seller = contract.annotation["seller"] == self.id
            step = contract.agreement["time"]
            # find the earliest time I can do anything about this contract
            earliest_production = self.awi.current_step
            if step > self.awi.n_steps - 1 or step < earliest_production:
                continue
            # if I am a seller, I will schedule production
            output_product = contract.annotation["product"]
            input_product = output_product - 1
            if is_seller:
                steps, _ = self.awi.schedule_production(
                    process=input_product,
                    repeats=contract.agreement["quantity"],
                    step=(earliest_production, step - 2),
                    line=-1,
                    partial_ok=True,
                )
            else:
                steps, _ = self.awi.schedule_production(
                    process=input_product,
                    repeats=contract.agreement["quantity"],
                    step=(step, self.awi.n_steps - 2),
                    line=-1,
                    partial_ok=True,
                )
            self.schedule_range[contract.id] = (
                min(steps) if len(steps) > 0 else -1,
                max(steps) if len(steps) > 0 else -1,
                is_seller,
            )

    @staticmethod
    def demand_driven_on_contracts_finalized(
        self: "SCML2020Agent",
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            if not is_seller:
                continue
            step = contract.agreement["time"]
            # find the earliest time I can do anything about this contract
            earliest_production = self.awi.current_step
            if step > self.awi.n_steps - 1 or step < earliest_production:
                continue
            # if I am a seller, I will schedule production
            output_product = contract.annotation["product"]
            input_product = output_product - 1
            steps, _ = self.awi.schedule_production(
                process=input_product,
                repeats=contract.agreement["quantity"],
                step=(earliest_production, step - 1),
                line=-1,
                partial_ok=True,
            )
            self.schedule_range[contract.id] = (
                min(steps) if len(steps) > 0 else -1,
                max(steps) if len(steps) > 0 else -1,
                is_seller,
            )


class MyTradingStrategy(PredictionBasedTradingStrategy):
    def init(self):
        super().init()
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])

        self.input_cost = (
            self.awi.catalog_prices[self.awi.my_input_product]
        ) * np.ones(self.awi.n_steps, dtype=int)
        self.output_price = (
            self.awi.catalog_prices[self.awi.my_output_product] + production_cost
        ) * np.ones(self.awi.n_steps, dtype=int)

        self.inputs_needed = (self.awi.n_steps * self.awi.n_lines) * np.ones(
            self.awi.n_steps, dtype=int
        )

        self.outputs_needed = np.zeros(self.awi.n_steps, dtype=int)

    def step(self):
        super().step()

        trading_prices = self.awi.trading_prices
        if trading_prices is None:
            self.prices = [
                self.awi.catalog_prices[self.awi.my_input_product],
                self.awi.catalog_prices[self.awi.my_output_product],
            ]
        else:
            self.prices = [
                self.awi.trading_prices[self.awi.my_input_product],
                self.awi.trading_prices[self.awi.my_output_product],
            ]

        self.input_cost[self.awi.current_step :] = self.prices[0]
        self.output_price[self.awi.current_step :] = self.prices[1]

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)

        for contract in signed:
            seller = contract.annotation["seller"]
            quantity = contract.agreement["quantity"]
            ctime = contract.agreement["time"]

            is_seller = seller == self.id

            if is_seller:
                self.outputs_secured[ctime] += quantity
                self.outputs_needed[ctime:] -= quantity

            else:
                self.inputs_secured[ctime] += quantity
                self.inputs_needed[ctime:] -= quantity
                self.outputs_needed[ctime + 1 :] += quantity


class MyNegotiationManager(IndependentNegotiationsManager):
    def _start_negotiations(
        self,
        product: int,
        sell: bool,
        step: int,
        qvalues: Tuple[int, int],
        uvalues: Tuple[int, int],
        tvalues: Tuple[int, int],
        partners: List[str],
    ) -> None:
        issues = [
            make_issue((int(qvalues[0]), int(max(qvalues))), name="quantity"),
            make_issue((int(tvalues[0]), int(max(tvalues))), name="time"),
            make_issue((int(uvalues[0]), int(max(uvalues))), name="unit_price"),
        ]

        for partner in partners:
            self.awi.request_negotiation(
                is_buy=not sell,
                product=product,
                quantity=qvalues,
                unit_price=uvalues,
                time=tvalues,
                partner=partner,
                negotiator=self.negotiator(sell, partner=partner, issues=issues),
            )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        is_seller = annotation["seller"] == self.id
        if is_seller:
            if self.consumers_financial_trust[initiator] <= self.minimum_viable_trust:
                return None
        else:
            if self.suppliers_financial_trust[initiator] <= self.minimum_viable_trust:
                return None

        return self.negotiator(
            annotation["seller"] == self.id, issues=issues, partner=initiator
        )

    def negotiator(
        self, is_seller: bool, issues=None, outcomes=None, partner=None
    ) -> SAONegotiator:
        """Creates a negotiator"""
        params = self.negotiator_params
        params["ufun"] = self.create_ufun(
            is_seller=is_seller, outcomes=outcomes, issues=issues, partner=partner
        )
        return instantiate(self.negotiator_type, id=partner, **params)

    def create_ufun(
        self, is_seller: bool, issues=None, outcomes=None, partner=None
    ) -> UtilityFunction:
        if is_seller:
            trust = self.consumers_financial_trust[partner]
            return LinearUtilityFunction(
                (1.0 * trust, 1.0 * trust, 10.0 * trust),
                issues=issues,
                outcomes=outcomes,
            )
        trust = self.suppliers_financial_trust[partner]
        return LinearUtilityFunction(
            (1.0 * trust, -1.0 * trust, -10.0 * trust), issues=issues, outcomes=outcomes
        )


class PolymorphicAgent(
    MyTradingStrategy,
    MyNegotiationManager,
    PolymorphicProductionStrategy,
    SCML2020Agent,
):
    def init(self):
        super().init()
        logger.log("Agent Init ----------------------------------")
        logger.log("Input Product: " + str(self.awi.my_input_product))
        logger.log("Output Product: " + str(self.awi.my_output_product))
        logger.log("Catalog prices: " + str(self.awi.catalog_prices))
        logger.log("---------------------------------------------")

        trading_prices = self.awi.trading_prices
        if trading_prices is None:
            self.prices = [
                self.awi.catalog_prices[self.awi.my_input_product],
                self.awi.catalog_prices[self.awi.my_output_product],
            ]
        else:
            self.prices = [
                self.awi.trading_prices[self.awi.my_input_product],
                self.awi.trading_prices[self.awi.my_output_product],
            ]

        # Trust
        self.price_penalty = 1.4
        self.price_discount = 0.7
        self.minimum_viable_trust = 0.15
        self.my_suppliers = self.awi.my_suppliers
        self.my_consumers = self.awi.my_consumers
        self.suppliers_financial_trust = dict()
        self.consumers_financial_trust = dict()

        for supplier in self.my_suppliers:
            self.suppliers_financial_trust[supplier] = 1
        for consumer in self.my_consumers:
            self.consumers_financial_trust[consumer] = 1

    def step(self):
        super().step()
        if self.awi.my_input_product > 0:
            self.update_supplier_trust()
        if self.awi.my_output_product < self.awi.n_products:
            self.update_consumer_trust()
        logger.log("Supplier Trust: " + str(self.suppliers_financial_trust))
        logger.log("Consumer Trust: " + str(self.consumers_financial_trust))

    def update_supplier_trust(self):
        if self.awi.current_step >= 1:
            for supplier in self.my_suppliers:
                report = self.awi.reports_of_agent(supplier)
                if report is not None:
                    keys = list(report.keys())
                    trust = 0
                    for key in keys:
                        breach_prob = report[key].breach_prob
                        breach_level = report[key].breach_level
                        trust += 1 / 1.001 - (breach_level * breach_prob)
                    trust /= len(keys)
                    self.suppliers_financial_trust[supplier] = trust

    def update_consumer_trust(self):
        if self.awi.current_step >= 1:
            for consumer in self.my_consumers:
                report = self.awi.reports_of_agent(consumer)
                if report is not None:
                    keys = list(report.keys())
                    trust = 0
                    for key in keys:
                        breach_prob = report[key].breach_prob
                        breach_level = report[key].breach_level
                        trust += 1 / 1.001 - (breach_level * breach_prob)
                    trust /= len(keys)
                    self.consumers_financial_trust[consumer] = trust

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        if sell:
            return self.output_price[step]
        else:
            return self.input_cost[step]

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            if step == self.awi.n_steps:
                return self.outputs_needed[step - 1]
            return self.outputs_needed[step]
        else:
            if step == self.awi.n_steps:
                return 0
            return self.inputs_needed[step]


def run(
    competition="std",
    reveal_names=True,
    n_steps=100,
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

    competitors_2020 = list(scml_agents.get_agents(2020, track="std"))[0:7]
    competitors_2020.append(PolymorphicAgent)

    competitors = [
        PolymorphicAgent,
        MarketAwareDecentralizingAgent,
        DecentralizingAgent,
        BuyCheapSellExpensiveAgent,
        MarketAwareBuyCheapSellExpensiveAgent,
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
