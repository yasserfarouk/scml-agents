from typing import List

import numpy as np
from negmas import Contract
from scml import NO_COMMAND
from scml.scml2020.components.production import ProductionStrategy


class YIYProductionStrategy(ProductionStrategy):
    def is_supply(self):
        product_in_price = self.awi.catalog_prices[self.awi.my_input_product]
        product_out_price = self.awi.catalog_prices[self.awi.my_output_product]
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        return (product_out_price / 2) > (product_in_price / 2) + production_cost

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ):
        super().on_contracts_finalized(signed, cancelled, rejectors)
        for signed_contract in signed:
            is_seller = signed_contract.annotation["seller"] == self.id
            current_step_prod = self.awi.current_step
            latest = self.awi.n_steps - 2
            step = signed_contract.agreement["time"]

            if self.is_supply():
                if is_seller or step > latest + 1 or step < current_step_prod:
                    continue

                # I will schedule production
                input_product = signed_contract.annotation["product"]
                steps, _ = self.awi.schedule_production(
                    process=input_product,
                    repeats=signed_contract.agreement["quantity"],
                    step=(step, latest),
                    line=-1,
                    partial_ok=True,
                )

            else:
                if not is_seller or step > latest or step < current_step_prod:
                    continue

                # if I am a seller, I will schedule production
                output_product = signed_contract.annotation["product"]
                input_product = output_product - 1
                steps, _ = self.awi.schedule_production(
                    process=input_product,
                    repeats=signed_contract.agreement["quantity"],
                    step=(current_step_prod, step - 1),
                    line=-1,
                    partial_ok=True,
                )

            self.set_schedule_range(is_seller, signed_contract, steps)

    def set_schedule_range(self, is_seller, signed_contract, steps):
        self.schedule_range[signed_contract.id] = (
            min(steps) if len(steps) > 0 else -1,
            max(steps) if len(steps) > 0 else -1,
            is_seller,
        )

    def step(self):
        super().step()
        commands = NO_COMMAND * np.ones(self.awi.n_lines, dtype=int)
        inputs = min(self.awi.state.inventory[self.awi.my_input_product], len(commands))
        commands[:inputs] = self.awi.my_input_product
        commands[inputs:] = NO_COMMAND
        self.awi.set_commands(commands)
