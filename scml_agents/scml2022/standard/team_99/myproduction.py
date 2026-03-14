from typing import List

import numpy as np
from negmas import (
    Contract,
)
from scml.scml2020 import (
    ProductionStrategy,
    SCML2020Agent,
)
from scml.scml2020.common import NO_COMMAND

# required for development
# required for running the test tournament


class MyProductionStrategy(ProductionStrategy):
    def step(self):
        super().step()
        # インベントリ内の原材料を各工場に割り振る
        # 最大で工場の生産ライン数(awi.n_lines)まで割り振れる
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
        latest = self.awi.n_steps - 2
        earliest_production = self.awi.current_step
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            # if is_sellerになってた。
            # 自分が売り手の時にスケジュールを更新するのだからif not is_sellerが正しいのでは？
            if not is_seller:
                continue
            step = contract.agreement["time"]
            # find the earliest time I can do anything about this contract
            if step > latest + 1 or step < earliest_production:
                continue
            # if I am a seller, I will schedule production
            # 契約によって必要となるinput_productのID
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
