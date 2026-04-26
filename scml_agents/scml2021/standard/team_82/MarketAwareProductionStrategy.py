from typing import List

import numpy as np
from negmas import Contract
from scml import NO_COMMAND, ProductionStrategy


class MarketAwareProductionStrategy(ProductionStrategy):
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
        input_price = self.awi.trading_prices[self.awi.my_input_product]
        output_price = self.awi.catalog_prices[self.awi.my_output_product]
        cost = np.amin(self.awi.profile.costs)

        if output_price / 2 > input_price / 2 + cost:
            # Adopt a supply based production strategy if my output product has more value than input + cost
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
        else:
            # Adopt a demand based production strategy if my output product has less value than input + cost
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
