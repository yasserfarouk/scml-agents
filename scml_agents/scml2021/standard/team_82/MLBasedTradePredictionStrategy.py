from typing import Iterable

import numpy as np
from scml import TradePredictionStrategy


class MLBasedTradePredictionStrategy(TradePredictionStrategy):
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

    def trade_prediction_step(self):
        input = [
            self.awi.my_input_product,
            self.awi.my_output_product,
            self.awi.relative_time,
            self.awi.n_competitors,
            self.awi.n_processes,
            self.awi.n_products,
            self.awi.current_balance,
            self.awi.current_inventory[self.awi.my_input_product],
            self.awi.current_inventory[self.awi.my_output_product],
        ]

        input = np.array(input)
        prediction = self.prediction_model.predict([input])
        a, b = self.awi.current_step, self.awi.current_step + 1

        # If I expect to sell x outputs at step t, I should buy  x inputs at t-1
        self.expected_inputs[a:b] = prediction[0][0]
        # If I expect to buy x inputs at step t, I should sell x inputs at t+1
        self.expected_outputs[a:b] = prediction[0][1]
