import os
import sys

sys.path.append(os.path.dirname(__file__))

from typing import Tuple

import numpy as np
from negotiation import MyNegotiationManager
from production import MySupplyDrivenProductionStrategy
from scml.scml2020 import DecentralizingAgent, SCML2020Agent
from scml.scml2020.components.negotiation import IndependentNegotiationsManager
from scml.scml2020.components.production import SupplyDrivenProductionStrategy
from scml.scml2020.components.trading import PredictionBasedTradingStrategy
from trade import MyPredictionBasedTradingStrategy

__all__ = ["UnicornAgent"]


class SaveHistoryWrapper(SCML2020Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs).__init__(*args, **kwargs)
        self.history = []  # None

    def init(self):
        super().init()
        state = self.internal_state
        balance = state["_balance"]
        inventory_input = state[
            "_input_inventory"
        ]  # includes both production lines and the inventory
        inventory_output = state["_output_inventory"]
        productivity = state["execution_fraction"]
        t = self.awi.current_step
        needed_input1 = self.inputs_needed[t - 1] if t - 1 >= 0 else 0
        needed_input2 = self.inputs_needed[t]
        needed_input3 = (
            self.inputs_needed[t + 1] if t + 1 < len(self.inputs_secured) else 0
        )
        needed_output1 = self.outputs_needed[t - 1] if t - 1 >= 0 else 0
        needed_output2 = self.outputs_needed[t]
        needed_output3 = (
            self.outputs_needed[t + 1] if t + 1 < len(self.outputs_secured) else 0
        )

        self.history = [
            [
                balance,
                inventory_input,
                inventory_output,
                productivity,
                needed_input1,
                needed_input2,
                needed_input3,
                needed_output1,
                needed_output2,
                needed_output3,
            ]
        ]

    def step(self):
        super().step()
        state = self.internal_state
        balance = state["_balance"]
        inventory_input = state[
            "_input_inventory"
        ]  # includes both production lines and the inventory
        inventory_output = state["_output_inventory"]
        productivity = state["execution_fraction"]
        t = self.awi.current_step
        needed_input1 = self.inputs_needed[t - 1] if t - 1 >= 0 else 0
        needed_input2 = self.inputs_needed[t]
        needed_input3 = (
            self.inputs_needed[t + 1] if t + 1 < len(self.inputs_needed) else 0
        )
        needed_output1 = self.outputs_needed[t - 1] if t - 1 >= 0 else 0
        needed_output2 = self.outputs_needed[t]
        needed_output3 = (
            self.outputs_needed[t + 1] if t + 1 < len(self.outputs_needed) else 0
        )

        self.history.append(
            [
                balance,
                inventory_input,
                inventory_output,
                productivity,
                needed_input1,
                needed_input2,
                needed_input3,
                needed_output1,
                needed_output2,
                needed_output3,
            ]
        )


class CountStepsWrapper(SCML2020Agent):
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        self.i = 0

    def step(self):
        super().step()
        print(self.i)
        self.i += 1


class _NegotiationCallbacks:
    def average_production_cost(self):
        return (
            sum(self.awi.profile.costs[:, self.awi.my_input_product]) / self.awi.n_lines
        )

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return production_cost + self.input_cost[step]
        return self.output_price[step] - production_cost

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[step] - secured[step]

    def target_quantities(self, steps: Tuple[int, int], sell: bool) -> np.ndarray:
        """Implemented for speed but not really required"""

        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[steps[0] : steps[1]] - secured[steps[0] : steps[1]]


class ComponentAgent(
    IndependentNegotiationsManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SaveHistoryWrapper,
    SCML2020Agent,
):
    pass


class NegotiatorAgent(MyNegotiationManager, SaveHistoryWrapper, DecentralizingAgent):
    pass


class MyLearnNegotiationAgent(
    _NegotiationCallbacks,
    MySupplyDrivenProductionStrategy,
    MyNegotiationManager,
    MyPredictionBasedTradingStrategy,
    SaveHistoryWrapper,
    DecentralizingAgent,
):
    def step(self):
        print(self.awi.current_step)
        super().step()


class MyLearnUtilityAgent(
    _NegotiationCallbacks,
    MySupplyDrivenProductionStrategy,
    MyNegotiationManager,
    MyPredictionBasedTradingStrategy,
    SaveHistoryWrapper,
    DecentralizingAgent,
):
    def step(self):
        print(self.awi.current_step)
        super().step()


class UnicornAgent(
    _NegotiationCallbacks,
    MySupplyDrivenProductionStrategy,
    MyNegotiationManager,
    MyPredictionBasedTradingStrategy,
    SaveHistoryWrapper,
    DecentralizingAgent,
):
    pass
