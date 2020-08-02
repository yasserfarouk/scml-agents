# required for typing
import numpy as np
from negmas import Contract
from typing import List, Dict, Tuple

from scml.scml2020.common import NO_COMMAND

# my need
from scml.scml2020.components.production import *
from scml.scml2020 import *
from negmas import *
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
import seaborn as sns


class MyProductor(ProductionStrategy):
    """
    ProductionStrategy
    *SupplyDrivenProductionStrategy  # 買う契約を締結したら，買ったもの全てを作る
    DemandDrivenProductionStrategy  # 売る契約を締結したら，それまでに作る
    TradeDrivenProductionStrategy  # 買う契約と売る契約それぞれに対して，生産ラインを確保しているが，そんなことする必要ある？
    """

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

        current_step = self.awi.current_step
        defcit = (
            (self.output_price[current_step] - self.input_cost[current_step]) / 2
            < self.awi.profile.costs[0, self.awi.my_input_product]
        )  # 在庫の生産が，そのステップにおいて赤字になるか
        passive = self.awi.n_steps * 0.8  # どのへんから生産を慎重になるか
        latest = self.awi.n_steps - 2  # 最悪どこまで遅い生産を許容するか
        earliest_production = self.awi.current_step
        for contract in signed:
            step = contract.agreement["time"]
            # if step < passive:
            is_seller = contract.annotation["seller"] == self.id
            if is_seller:
                continue
            # find the earliest time I can do anything about this contract
            if step > latest + 1 or step < earliest_production:
                continue
            # if I am a buyer, I will schedule production
            input_product = contract.annotation["product"]
            steps, _ = self.awi.schedule_production(
                process=input_product,
                repeats=contract.agreement["quantity"],
                step=(step, latest),
                line=-1,
                method="earliest",
                partial_ok=True,
            )
            self.schedule_range[contract.id] = (
                min(steps) if len(steps) > 0 else -1,
                max(steps) if len(steps) > 0 else -1,
                is_seller,
            )

            # elif not defcit: # 最終盤に余りそうな在庫は，生産を調整して余分な生産を控える(DemandDrivenProductionStrategy)
            #     is_seller = contract.annotation["seller"] == self.id
            #     if not is_seller:
            #         continue
            #     # step = contract.agreement["time"]
            #     # find the earliest time I can do anything about this contract
            #     earliest_production = self.awi.current_step
            #     if step > self.awi.n_steps - 1 or step < earliest_production:
            #         continue
            #     # if I am a seller, I will schedule production
            #     output_product = contract.annotation["product"]
            #     input_product = output_product - 1
            #     steps, _ = self.awi.schedule_production(
            #         process=input_product,
            #         repeats=contract.agreement["quantity"],
            #         step=(earliest_production, step - 1),
            #         line=-1,
            #         partial_ok=True,
            #     )
            #     self.schedule_range[contract.id] = (
            #         min(steps) if len(steps) > 0 else -1,
            #         max(steps) if len(steps) > 0 else -1,
            #         is_seller,
            #     )
