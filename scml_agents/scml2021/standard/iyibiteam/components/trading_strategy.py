from typing import List, Optional

import numpy as np
from negmas import Contract
from numpy.core.fromnumeric import take
from scml.scml2020 import PredictionBasedTradingStrategy
from scml.scml2020.common import ANY_LINE, is_system_agent


class MyTradingStrategy(PredictionBasedTradingStrategy):
    """
    Placeholder
    """

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
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

        latest_production = self.awi.n_steps - 2
        earliest_production = self.awi.current_step
        step = self.awi.current_step

        catalog_buy = self.awi.catalog_prices[self.awi.my_input_product]
        catalog_sell = self.awi.catalog_prices[self.awi.my_output_product]

        sold_count, bought_count = 0, 0
        for contract, idx in contracts:
            is_seller = contract.annotation["seller"] == self.id
            t, u, q = (
                contract.agreement["time"],
                contract.agreement["unit_price"],
                contract.agreement["quantity"],
            )

            if t > latest_production + 1 or t < earliest_production:
                continue
            if u < self.output_price[t]:
                continue

            # check that the contract is executable in principle
            if t < step and len(contract.issues) == 3:
                continue

            progress = self.awi.current_step / self.awi.n_steps
            if is_seller and u < max(progress, 0.8) * catalog_sell:
                continue
            if not is_seller and u > min(progress * 2, 1.2) * catalog_buy:
                continue

            trange = (step, t) if is_seller else (t + 1, self.awi.n_steps - 1)

            steps, _ = self.awi.available_for_production(
                q, trange, ANY_LINE, override=False, method="all"
            )

            if len(steps) - (sold_count if is_seller else bought_count) < q:
                continue

            secured = self.outputs_secured if is_seller else self.inputs_secured
            needed = self.outputs_needed if is_seller else self.inputs_needed

            if (
                secured[trange[0] : trange[1] + 1].sum()
                + q
                + (sold_count if is_seller else bought_count)
                <= needed[trange[0] : trange[1] + 1].sum()
            ):
                signatures[idx] = self.id

                sold_count = sold_count + q if is_seller else sold_count
                bought_count = bought_count + q if not is_seller else bought_count

        return signatures

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        # call the production strategy
        step = self.awi.current_step

        sell_surplus = 0
        buy_surplus = 0
        sell_count = 0
        buy_count = 0
        for contract in signed:
            seller = contract.annotation["seller"] == self.id
            t, q = contract.agreement["time"], contract.agreement["quantity"]
            if contract.annotation["caller"] == self.id:
                continue
            if contract.annotation["product"] != self.awi.my_output_product:
                continue

            if seller:
                self.outputs_secured[t] += q
                self.outputs_needed[t:] -= q
                sell_surplus += (
                    self.output_price[step]
                    - self.awi.catalog_prices[self.awi.my_output_product]
                )
                sell_count += 1

            else:
                self.inputs_secured[t] += q
                self.inputs_secured[t:] -= q
                self.outputs_needed[t:] += q
                buy_surplus += (
                    self.awi.catalog_prices[self.awi.my_input_product]
                    - self.input_cost[step]
                )
                buy_count += 1

        if sell_surplus and sell_count:
            self.output_price[step:] = self.output_price[step] + (
                sell_surplus / sell_count
            )
        else:
            self.output_price[step:] = (
                self.awi.catalog_prices[self.awi.my_output_product] * 0.95
            )

        if buy_surplus and buy_count:
            self.input_cost[step:] = self.input_cost[step] - (buy_surplus / buy_count)
        else:
            self.input_cost[step:] = self.input_cost[step:] * 1.05
