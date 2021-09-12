import os
import sys

sys.path.append(os.path.dirname(__file__))

from typing import List

import numpy as np
from negmas import Contract
from scml.scml2020.common import NO_COMMAND
from scml.scml2020.components.production import ProductionStrategy


class MySupplyDrivenProductionStrategy(ProductionStrategy):
    def step(self):
        super().step()
        commands = NO_COMMAND * np.ones(self.awi.n_lines, dtype=int)
        inputs = min(self.awi.state.inventory[self.awi.my_input_product], len(commands))
        commands[:inputs] = self.awi.my_input_product
        commands[inputs:] = NO_COMMAND
        self.awi.set_commands(commands)

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
            latest = self.awi.n_steps - 2
            step = contract.agreement["time"]
            # find the earliest time I can do anything about this contract
            earliest_production = self.awi.current_step
            if step > self.awi.n_steps - 1 or step < earliest_production:
                continue
            # if I am a seller, I will schedule production
            output_product = contract.annotation["product"]
            input_product = output_product - 1
            quantity = contract.agreement["quantity"]

            p_supply = 0.8  # secure p_supply in advance

            steps1, _ = self.awi.schedule_production(
                process=input_product,
                repeats=int(p_supply * quantity),
                step=(earliest_production, step - 1),
                line=-1,
                partial_ok=True,
            )

            # TODO: maybe don't produce this part?
            steps2, _ = self.awi.schedule_production(
                process=input_product,
                repeats=quantity - int(p_supply * quantity),
                step=(step, latest),
                line=-1,
                partial_ok=True,
            )

            steps = list(steps1) + list(steps2)

            self.schedule_range[contract.id] = (
                min(steps) if len(steps) > 0 else -1,
                max(steps) if len(steps) > 0 else -1,
                is_seller,
            )
