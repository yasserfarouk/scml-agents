"""
SCM League 2021
Augur agent
"""

from typing import List, Optional

import numpy as np
from negmas import Contract
from scml.scml2020 import (
    SCML2020Agent,
    StepNegotiationManager,
    SupplyDrivenProductionStrategy,
    TradingStrategy,
)
from scml.scml2020.common import is_system_agent

__all__ = ["AugurAgent"]


class MyProductionStrategy(SupplyDrivenProductionStrategy):
    def on_contracts_finalized(
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


class MyTradingStrategy(TradingStrategy):
    def init(self):
        super().init()
        self.pro_cost = np.max(
            self.awi.profile.costs[:, self.awi.my_input_product]
        )  # calculate production cost
        self.my_base_input_cost = (
            self.awi.catalog_prices[self.awi.my_output_product] - self.pro_cost
        )
        self.my_base_output_cost = (
            self.awi.catalog_prices[self.awi.my_input_product] + self.pro_cost
        )


class AugurAgent(MyProductionStrategy, SCML2020Agent):
    pass
