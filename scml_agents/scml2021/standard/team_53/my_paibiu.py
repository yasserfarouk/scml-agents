from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from negmas import Contract, LinearUtilityFunction, SAOMetaNegotiatorController
from scml.scml2020 import QUANTITY, TIME, UNIT_PRICE, SCML2020Agent, SCML2021World
from scml.scml2020.agents import (
    DecentralizingAgent,
    MarketAwareDecentralizingAgent,
    RandomAgent,
)

# from steady_mgr import SteadyMgr
from scml.scml2020.common import is_system_agent
from scml.scml2020.components.prediction import MarketAwareTradePredictionStrategy
from scml.scml2020.components.production import (
    DemandDrivenProductionStrategy,
    ProductionStrategy,
)
from scml.scml2020.components.trading import PredictionBasedTradingStrategy

__all__ = ["MyPaibiuAgent"]


class MyTradingStrategy(PredictionBasedTradingStrategy):
    def init(self):
        super().init()
        # # Maximum number of products that can be produced
        # self.inputs_needed = (self.awi.n_steps * self.awi.n_lines) * np.ones(
        #     self.awi.n_steps, dtype=int
        # )
        # # If the last production is assigned, make it three quarters (because the number of products that can be sold is limited)
        # if self.awi.my_output_product == (self.awi.n_products - 1):
        #     self.inputs_needed = self.inputs_needed * 3 // 4
        #
        # # Zero at first (avoid breach of contract)
        # self.outputs_needed = np.zeros(self.awi.n_steps, dtype=int)

    # def step(self):
    #     super().step()
    #     step = self.awi.current_step
    #     # The maximum number reduced every step
    #     self.inputs_needed[step:] -= self.awi.n_lines

    # def on_contracts_finalized(
    #         self,
    #         signed: List[Contract],
    #         cancelled: List[Contract],
    #         rejectors: List[List[str]],
    # ) -> None:
    #     super().on_contracts_finalized(signed, cancelled, rejectors)
    #
    #     sold, bought = 0, 0  # count the number of sold/bought contracts
    #     q_current_step = 0
    #     for contract in signed:
    #         is_seller = contract.annotation["seller"] == self.id
    #         q, u, t = (
    #             contract.agreement["quantity"],
    #             contract.agreement["unit_price"],
    #             contract.agreement["time"],
    #         )
    #         if is_seller:
    #             self.outputs_secured[t] += q  # Add the number of sells at t
    #             self.outputs_needed[
    #             t:
    #             ] -= q  # Subtract the number of products that need to sell after t
    #             sold += 1
    #
    #         else:
    #             self.inputs_secured[t] += q  # Add the number of buys at t
    #             # self.inputs_needed[
    #             # t:
    #             # ] -= q  # Subtract the number of units that need to buy after t
    #             self.outputs_needed[
    #             t + 1:
    #             ] += q  # Add 'outputs_needed' as many as buys
    #             bought += 1
    #
    #     step = self.awi.current_step

    # Update sell and buy prices
    # Update the selling price except of then the last process is assigned (because the selling price of the last process is fixed)
    # if self.awi.my_output_product != (self.awi.n_products - 1):
    #     if (
    #             sold > 0
    #     ):  # If products are sold, the price is increased by 10% (upper limit is twice of the catalog price)
    #         self.output_price[step:] = min(
    #             self.output_price[step] + self.output_price[step] // 10,
    #             self.awi.catalog_prices[self.awi.my_output_product] * 2,
    #         )
    #     else:  # If the product doesn't sell, the price is reduced by 10% (lower limit is the initial selling price)
    #         self.output_price[step:] = max(
    #             self.output_price[step] - self.output_price[step] // 10,
    #             self.output_price[0],
    #         )
    #
    # # Update the buying price except of when the first process is assigned (because the buying price of the first process is fixed)
    # if self.awi.my_input_product != 0:
    #     if (
    #             bought > 0
    #     ):  # If I could buy the units, reduce the buying price by 10% (lower limit is half of the catalogue price)
    #         self.input_cost[step:] = max(
    #             self.input_cost[step] - self.input_cost[step] // 10,
    #             self.awi.catalog_prices[self.awi.my_input_product] // 2,
    #         )
    #     else:  # If I could not buy, increase the buying price by 10% (upper limit is the initial buying price)
    #         self.input_cost[step:] = min(
    #             self.input_cost[step] + self.input_cost[step] // 10,
    #             self.input_cost[0],
    #         )

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        # sort contracts by time and then put system contracts first within each time-step
        signatures = [None] * len(contracts)
        contracts = sorted(
            zip(contracts, range(len(contracts))),
            key=lambda x: (
                x[0].agreement["unit_price"],
                x[0].agreement["time"],
                0
                if is_system_agent(x[0].annotation["seller"])
                or is_system_agent(x[0].annotation["buyer"])
                else 1,
                x[0].agreement["unit_price"],
            ),
        )

        sold, bought = 0, 0  # count the number of sold/bought products during the loop
        s = self.awi.current_step

        for contract, indx in contracts:
            is_seller = contract.annotation["seller"] == self.id
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            if t < s and len(contract.issues) == 3:
                continue

            if is_seller:
                # Sign the first contract when the final process is assigned
                if s == 0 and self.awi.my_output_product == (self.awi.n_products - 1):
                    signatures[indx] = self.id
                # I don't sign contracts for less than the selling price
                if u < self.output_price[t]:
                    continue

                est = 0  # Estimated number of products
                # Calculate the maximum production possible before delivery date
                for i in range(1, t - s + 1):
                    est += min(self.inputs_secured[t - i], i * self.awi.n_lines)
                est = min(est, (t - s) * self.awi.n_lines)

                available = (
                    est
                    + self.internal_state["_output_inventory"]
                    - (self.outputs_secured[s:]).sum()
                )  # Add stock and sub contracted
                # Only sign contracts that ensure production is on time.
                if available - sold > q:
                    signatures[indx] = self.id
                    sold += q

            else:
                # I don't make contracts to buy at the end of the game.
                if t > self.awi.n_steps * 3 // 4:
                    continue

                # I don't sign contracts over the buying price
                if u > self.input_cost[t]:
                    continue

                needed = self.inputs_needed[
                    self.awi.n_steps - 1
                ]  # Maximum number of products that can be produced
                if needed - bought > q:
                    signatures[indx] = self.id
                    bought += q

        return signatures

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: List[int],
        compensation_money: int,
    ) -> None:
        super().on_agent_bankrupt(agent, contracts, quantities, compensation_money)
        for contract, new_quantity in zip(contracts, quantities):
            q = contract.agreement["quantity"]
            if new_quantity == q:
                continue
            t = contract.agreement["time"]
            missing = q - new_quantity
            if t < self.awi.current_step:
                continue
            if contract.annotation["seller"] == self.id:
                self.outputs_secured[t] -= missing
                if t > 0:
                    self.outputs_needed[t - 1 :] += missing
            else:
                self.inputs_secured[t] -= missing
                self.inputs_needed[t:] += missing


class MyPaibiuAgent(
    MarketAwareTradePredictionStrategy,
    MyTradingStrategy,
    DemandDrivenProductionStrategy,
    SCML2020Agent,
):
    def __init__(
        self,
        *args,
        price_weight=0.7,
        time_horizon=0.1,
        utility_threshold=0.9,
        time_threshold=0.9,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.index: List[int] = None
        self.time_horizon = time_horizon
        self._price_weight = price_weight
        self._time_threshold = time_threshold
        self._utility_threshold = utility_threshold
        self._current_end = -1
        self._current_start = -1

    def step(self):
        super().step()
        # find the range of steps about which we plan to negotiate
        step = self.awi.current_step
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if step == 0:
            self.output_price = (
                max(
                    self.awi.catalog_prices[self.awi.my_input_product]
                    + production_cost,
                    self.awi.catalog_prices[self.awi.my_output_product] + 1,
                )
            ) * np.ones(self.awi.n_steps, dtype=int)
            self.input_cost = (
                self.awi.catalog_prices[self.awi.my_output_product] - production_cost
            ) * np.ones(self.awi.n_steps, dtype=int)
        else:
            self.output_price[step:] = max(
                max(
                    self.awi.catalog_prices[self.awi.my_input_product]
                    + production_cost,
                    self.awi.catalog_prices[self.awi.my_output_product] + 1,
                ),
                self.awi.trading_prices[self.awi.my_output_product],
            )
            self.input_cost[step:] = min(
                self.awi.catalog_prices[self.awi.my_output_product] - production_cost,
                self.awi.trading_prices[self.awi.my_output_product] - production_cost,
            )
        self._current_start = step + 1
        self._current_end = min(
            self.awi.n_steps - 1,
            self._current_start + max(1, int(self.time_horizon * self.awi.n_steps)),
        )
        if self._current_start >= self._current_end:
            return

        for seller, needed, secured, product in [
            (False, self.inputs_needed, self.inputs_secured, self.awi.my_input_product),
            (
                True,
                self.outputs_needed,
                self.outputs_secured,
                self.awi.my_output_product,
            ),
        ]:
            # find the maximum amount needed at any time-step in the given range
            needs = np.max(
                needed[self._current_start : self._current_end]
                - secured[self._current_start : self._current_end]
            )
            if needs < 1:
                continue
            # set a range of prices
            if seller:
                # for selling set a price that is at least the catalog price
                min_price = self.output_price[step]
                price_range = (min_price, 2 * min_price)
                controller = SAOMetaNegotiatorController(
                    ufun=LinearUtilityFunction(
                        (
                            (1 - self._price_weight),
                            0.0,
                            self._price_weight,
                        ),
                    )
                )
            else:
                # for buying sell a price that is at most the catalog price
                max_price = self.input_cost[step]
                price_range = (max_price // 2, max_price)
                controller = SAOMetaNegotiatorController(
                    ufun=LinearUtilityFunction(
                        (
                            (1 - self._price_weight),
                            0.0,
                            -self._price_weight,
                        ),
                    )
                )

            self.awi.request_negotiations(
                not seller,
                product,
                (1, needs),
                price_range,
                time=(self._current_start, self._current_end),
                controller=controller,
            )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List["Issue"],
        annotation: Dict[str, Any],
        mechanism: "AgentMechanismInterface",
    ) -> Optional["Negotiator"]:
        # refuse to negotiate if the time-range does not intersect
        # the current range
        if not (
            issues[TIME].min_value < self._current_end
            or issues[TIME].max_value > self._current_start
        ):
            return None
        if self.id == annotation["seller"]:
            controller = SAOMetaNegotiatorController(
                ufun=LinearUtilityFunction(
                    (
                        (1 - self._price_weight),
                        0.0,
                        self._price_weight,
                    ),
                    issues=issues,
                )
            )
        else:
            if self.awi.current_step == 0:
                try:
                    needs = np.max(
                        self.inputs_needed[0 : self.awi.n_steps - 1]
                        - self.inputs_secured[0 : self.awi.n_steps - 1],
                        initial=0,
                    )
                except:
                    return None
            else:
                try:
                    needs = np.max(
                        self.inputs_needed[self._current_start : self._current_end]
                        - self.inputs_secured[self._current_start : self._current_end],
                        initial=0,
                    )
                except:
                    return None
            if needs < 1:
                return None
            controller = SAOMetaNegotiatorController(
                ufun=LinearUtilityFunction(
                    (
                        (1 - self._price_weight),
                        0.0,
                        -self._price_weight,
                    ),
                    issues=issues,
                )
            )
        return controller.create_negotiator()

    """My agent"""

    # def target_quantity(self, step: int, sell: bool) -> int:
    #     """A fixed target quantity of half my production capacity"""
    #     return self.awi.n_lines // 2
    #
    # def acceptable_unit_price(self, step: int, sell: bool) -> int:
    #     """The catalog price seems OK"""
    #     return self.awi.catalog_prices[self.awi.my_output_product] if sell else self.awi.catalog_prices[
    #         self.awi.my_input_product]
    #
    # def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
    #     """A utility function that penalizes high cost and late delivery for buying and and awards them for selling"""
    #     if is_seller:
    #         return LinearUtilityFunction((0, 0.25, 1))
    #     return LinearUtilityFunction((0, -0.5, -0.8))


def show_agent_scores(world):
    scores = defaultdict(list)
    for aid, score in world.scores().items():
        scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    plt.bar(list(scores.keys()), list(scores.values()), width=0.2)
    plt.show()
