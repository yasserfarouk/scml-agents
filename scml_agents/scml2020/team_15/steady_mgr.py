"""
**Submitted to ANAC 2020 SCML**
*Authors* Masahito Okuno:okuno.masahito@otsukalab.nitech.ac.jp;

"""
# Libraries
from typing import List, Optional
import numpy as np
from negmas import Contract
from scml.scml2020 import (
    SCML2020Agent,
    TradingStrategy,
    StepNegotiationManager,
    SupplyDrivenProductionStrategy,
)
from scml.scml2020.common import is_system_agent

__all__ = ["SteadyMgr"]


class MyTradingStrategy(TradingStrategy):
    def init(self):
        super().init()
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])

        # Calculate the initial price (catalog price ± production cost)
        self.input_cost = (
            self.awi.catalog_prices[self.awi.my_output_product] - production_cost
        ) * np.ones(self.awi.n_steps, dtype=int)
        self.output_price = (
            self.awi.catalog_prices[self.awi.my_input_product] + production_cost
        ) * np.ones(self.awi.n_steps, dtype=int)

        # Maximum number of products that can be produced
        self.inputs_needed = (self.awi.n_steps * self.awi.n_lines) * np.ones(
            self.awi.n_steps, dtype=int
        )
        # If the last production is assigned, make it three quarters (because the number of products that can be sold is limited)
        if self.awi.my_output_product == (self.awi.n_products - 1):
            self.inputs_needed = self.inputs_needed * 3 // 4

        # Zero at first (avoid breach of contract)
        self.outputs_needed = np.zeros(self.awi.n_steps, dtype=int)

    def step(self):
        super().step()
        step = self.awi.current_step
        # The maximum number reduced every step
        self.inputs_needed[step:] -= self.awi.n_lines

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)

        sold, bought = 0, 0  # count the number of sold/bought contracts
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            if is_seller:
                self.outputs_secured[t] += q  # Add the number of sells at t
                self.outputs_needed[
                    t:
                ] -= q  # Subtract the number of products that need to sell after t
                sold += 1

            else:
                self.inputs_secured[t] += q  # Add the number of buys at t
                self.inputs_needed[
                    t:
                ] -= q  # Subtract the number of units that need to buy after t
                self.outputs_needed[
                    t + 1 :
                ] += q  # Add 'outputs_needed' as many as buys
                bought += 1

        step = self.awi.current_step

        # Update sell and buy prices
        # Update the selling price except of then the last process is assigned (because the selling price of the last process is fixed)
        if self.awi.my_output_product != (self.awi.n_products - 1):
            if (
                sold > 0
            ):  # If products are sold, the price is increased by 10% (upper limit is twice of the catalog price)
                self.output_price[step:] = min(
                    self.output_price[step] + self.output_price[step] // 10,
                    self.awi.catalog_prices[self.awi.my_output_product] * 2,
                )
            else:  # If the product doesn't sell, the price is reduced by 10% (lower limit is the initial selling price)
                self.output_price[step:] = max(
                    self.output_price[step] - self.output_price[step] // 10,
                    self.output_price[0],
                )

        # Update the buying price except of when the first process is assigned (because the buying price of the first process is fixed)
        if self.awi.my_input_product != 0:
            if (
                bought > 0
            ):  # If I could buy the units, reduce the buying price by 10% (lower limit is half of the catalogue price)
                self.input_cost[step:] = max(
                    self.input_cost[step] - self.input_cost[step] // 10,
                    self.awi.catalog_prices[self.awi.my_input_product] // 2,
                )
            else:  # If I could not buy, increase the buying price by 10% (upper limit is the initial buying price)
                self.input_cost[step:] = min(
                    self.input_cost[step] + self.input_cost[step] // 10,
                    self.input_cost[0],
                )

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


class MyNegotiationManager(StepNegotiationManager):
    def _urange(self, step, is_seller, time_range):
        price = self.acceptable_unit_price(step, is_seller)
        if is_seller:
            return price, price * 2

        return 1, price


class SteadyMgr(
    MyTradingStrategy,
    MyNegotiationManager,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        if sell:
            return self.output_price[step]
        else:
            return self.input_cost[step]

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            # In the final step, return the maximum number
            if step >= self.awi.n_steps:
                return self.outputs_needed[step - 1]
            # Up to twice the number of production lines
            return min(self.outputs_needed[step], self.awi.n_lines * 2)
        else:
            # I'm not buying in the final step
            if step >= self.awi.n_steps:
                return 0
            # Up to twice the number of production lines
            return min(self.inputs_needed[step], self.awi.n_lines * 2)


# if __name__ == '__main__':
#    run()
