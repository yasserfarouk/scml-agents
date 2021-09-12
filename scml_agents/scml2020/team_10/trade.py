import os
import sys

sys.path.append(os.path.dirname(__file__))

from typing import Iterable, List

import numpy as np
import torch
from hyperparameters import *
from negmas import Contract
from scml.scml2020.components.prediction import TradePredictionStrategy
from scml.scml2020.components.trading import PredictionBasedTradingStrategy
from trade_model import load_trade_model


class MyTradePredictionStrategy(TradePredictionStrategy):
    """
    Predicts a fixed amount of trade both for the input and output products.

    Hooks Into:
        - `internal_state`
        - `on_contracts_finalized`

    Remarks:
        - `Attributes` section describes the attributes that can be used to construct the component (passed to its
          `__init__` method).
        - `Provides` section describes the attributes (methods, properties, data-members) made available by this
          component directly. Note that everything provided by the bases of this components are also available to the
          agent (Check the `Bases` section above for all the bases of this component).
        - `Requires` section describes any requirements from the agent using this component. It defines a set of methods
          or properties/data-members that must exist in the agent that uses this component. These requirement are
          usually implemented as abstract methods in the component
        - `Abstract` section describes abstract methods that MUST be implemented by any descendant of this component.
        - `Hooks Into` section describes the methods this component overrides calling `super` () which allows other
          components to hook into the same method (by overriding it). Usually callbacks starting with `on_` are
          hooked into this way.
        - `Overrides` section describes the methods this component overrides without calling `super` effectively
          disallowing any other components after it in the MRO to call this method. Usually methods that do some
          action (i.e. not starting with `on_`) are overridden this way.

    """

    def trade_prediction_init(self):
        inp = self.awi.my_input_product

        def adjust(x, demand):
            """Adjust the predicted demand/supply filling it with a default value or repeating as needed"""
            if x is None:
                x = max(1, self.awi.n_lines // 2)
            elif isinstance(x, Iterable):
                return np.array(x)
            predicted = int(x) * np.ones(self.awi.n_steps, dtype=int)
            if demand:
                predicted[: inp + 1] = 0
            else:
                predicted[inp - self.awi.n_processes :] = 0
            return predicted

        # adjust predicted demand and supply
        self.expected_outputs = adjust(self.expected_outputs, True)
        self.expected_inputs = adjust(self.expected_inputs, False)

        # now we deal with the prices
        production_cost = self.average_production_cost()
        tradeoff = self.output_price - self.input_cost - production_cost - 1
        self.output_price = self.output_price - 0.5 * tradeoff
        self.input_cost = self.input_cost + 0.5 * tradeoff

    def trade_prediction_step(self):
        # predict next output
        history = torch.from_numpy(np.array(self.history)).float()
        # expected_input = round(self.input_model(history)[-1].item())
        # expected_output = round(self.output_model(history)[-1].item())
        expected_input = list(
            map(lambda x: round(x.item()), self.input_model(history)[-1])
        )
        expected_output = list(
            map(lambda x: round(x.item()), self.output_model(history)[-1])
        )

        t = self.awi.current_step
        # If I expect to sell x outputs at step t + 2, I should buy  x inputs at t + 1
        if t + len(expected_input) < len(self.outputs_secured):
            self.expected_inputs[t + 1 : t + 1 + len(expected_input)] = expected_input
        else:
            self.expected_inputs[t + 1 :] = expected_input[
                : len(self.expected_inputs[t + 1 :])
            ]
        # If I expect to buy x inputs at step t + 1, I should sell x inputs at t + 2
        if t + len(expected_output) < len(self.expected_outputs):
            self.expected_outputs[t : t + len(expected_output)] = expected_output
        else:
            self.expected_inputs[t + 1 :] = expected_input[
                : len(self.expected_inputs[t + 1 :])
            ]

    @property
    def internal_state(self):
        state = super().internal_state
        state.update(
            {
                "expected_inputs": self.expected_inputs,
                "expected_outputs": self.expected_outputs,
                "input_cost": self.input_cost,
                "output_price": self.output_price,
            }
        )
        return state

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        if not self._add_trade:
            return
        # TODO: assumes all signed contracts are fully executed, can be improved
        for contract in signed:
            t, q = contract.agreement["time"], contract.agreement["quantity"]
            if contract.annotation["seller"] == self.id:
                self.expected_outputs[t] += q
            else:
                self.expected_inputs[t] += q


class MyPredictionBasedTradingStrategy(
    MyTradePredictionStrategy, PredictionBasedTradingStrategy
):
    def __init__(
        self,
        input_model_path=TRADE_INPUT_PATH,
        output_model_path=TRADE_OUTPUT_PATH,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.input_model = load_trade_model(path=input_model_path)
        self.output_model = load_trade_model(path=output_model_path)
        self.input_model.eval()
        self.output_model.eval()

    def init(self):
        super().init()

    def step(self):
        super().step()
        # If I expect to sell x outputs at step t, I should buy  x inputs at t-1
        self.inputs_needed[:-1] = self.expected_outputs[1:]
        # If I expect to buy x inputs at step t, I should sell x inputs at t+1
        self.outputs_needed[1:] = self.expected_inputs[:-1]
