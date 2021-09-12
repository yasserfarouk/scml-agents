from typing import Dict, List, Tuple

import numpy as np
from negmas import Contract
from scml.scml2020 import DemandDrivenProductionStrategy
from scml.scml2020.common import NO_COMMAND


class MyProductionStrategy(DemandDrivenProductionStrategy):
    """
    Placeholder
    """

    def should_i_produce(self):
        product_in_price = self.awi.catalog_prices[self.awi.my_input_product]
        product_out_price = self.awi.catalog_prices[self.awi.my_output_product]
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if product_out_price > production_cost + product_in_price:
            return True
        else:
            return False

    def on_contracts_finalized(
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

            if self.should_i_produce():
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
